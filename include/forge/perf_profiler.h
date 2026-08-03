#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

// Phase 2: 事件 ID 化。用 16bit 整数替代字符串做热路径 key,
// 消除每帧字符串拷贝 / 哈希 / 堆分配。
using PerfEventId = uint16_t;
inline constexpr PerfEventId kInvalidPerfEventId = 0;
inline constexpr PerfEventId kMaxPerfEvents = 512;

struct PerfAccumulator {
    uint64_t count = 0;
    double total_ms = 0.0;
    double min_ms = 1e18;
    double max_ms = 0.0;
    double last_ms = 0.0;
};

struct PerfRecord {
    std::string name;
    double total_ms = 0.0;
    int64_t count = 0;
    double min_ms = 1e18;
    double max_ms = 0.0;
    double last_ms = 0.0;
};

// Phase 4: 结构化上下文。forward 入口设置，layers 读取，不再拼字符串。
struct PerfContext {
    int64_t request_id = -1;   // -1 表示未设置
    const char* phase = "";     // "prefill" / "decode" / "sampler" / "layer"
    int layer = -1;             // 层索引，-1 表示未设置
    const char* device = "";    // "cpu" / "cuda"
    int tokens = 0;             // 当前 token 数
};

// Phase 4: Trace 事件。固定大小，无堆分配。
struct PerfTraceEvent {
    PerfEventId id;
    int64_t timestamp_us;   // 开始时间（微秒）
    int32_t duration_us;    // 持续时长（微秒），-1 表示未完成
    PerfContext ctx;        // 结构化上下文快照
};

// Phase 4: Thread-local 上下文设置/读取。
// forward 入口设置，layers 读取，不再把 layer/device 拼进字符串。
struct PerfThreadContext {
    PerfContext ctx;
    bool set = false;
};

inline PerfThreadContext& thread_local_perf_context() {
    thread_local PerfThreadContext ctx;
    return ctx;
}

inline void set_perf_context(const PerfContext& ctx) {
    thread_local_perf_context().ctx = ctx;
    thread_local_perf_context().set = true;
}

inline const PerfContext& get_perf_context() {
    return thread_local_perf_context().ctx;
}

inline void clear_perf_context() {
    thread_local_perf_context().set = false;
    thread_local_perf_context().ctx = PerfContext{};
}

#ifdef USE_CUDA
// Phase 3: CUDA event 池。每线程复用 CUDA event pair，避免 scope 级 create/destroy。
struct CudaEventSlot {
    cudaEvent_t start{};
    cudaEvent_t end{};
    bool in_use = false;

    CudaEventSlot() {
        cudaEventCreate(&start);
        cudaEventCreate(&end);
    }
    ~CudaEventSlot() {
        if (start)
            cudaEventDestroy(start);
        if (end)
            cudaEventDestroy(end);
    }
};

// 每线程 event 池：复用空闲 slot，不足时懒创建。
struct PerfThreadCudaState {
    std::vector<std::unique_ptr<CudaEventSlot>> pool;
    // pending 列表：PerfScopeTimer 析构时把 (id, slot_idx) 推入，flush 时处理。
    struct PendingEvent {
        PerfEventId id;
        size_t slot_idx;
    };
    std::vector<PendingEvent> pending;
};
#endif

// Phase 4: Trace ring buffer。固定大小，原子写指针，满则丢弃并置 overflow。
// 定义在 PerfProfiler 之前，to_json() 需要访问 PerfProfiler::name_for_id()。
class PerfTrace {
public:
    static constexpr size_t kCapacity = 65536;

    PerfTrace() : buffer_(std::make_unique<PerfTraceEvent[]>(kCapacity)) {}

    // 开启/关闭 trace。关闭时不记录，但不清空已有数据。
    void set_enabled(bool v) { enabled_.store(v, std::memory_order_relaxed); }
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

    // 记录一个事件。返回 false 表示 buffer 已满（overflow）。
    bool record(PerfEventId id, int64_t duration_us, const PerfContext& ctx) {
        if (!enabled())
            return true;
        size_t idx = write_pos_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= kCapacity) {
            overflow_.store(true, std::memory_order_relaxed);
            return false;
        }
        buffer_[idx].id = id;
        buffer_[idx].timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count();
        buffer_[idx].duration_us = static_cast<int32_t>(duration_us);
        buffer_[idx].ctx = ctx;
        return true;
    }

    // 导出为 JSON 数组（Chrome Trace 格式）。
    // 返回 JSON 字符串。
    std::string to_json(const std::unordered_map<PerfEventId, std::string>& id_to_name) const {
        std::string json = "[";
        bool first = true;
        size_t end = std::min(static_cast<size_t>(write_pos_.load(std::memory_order_relaxed)), kCapacity);
        for (size_t i = 0; i < end; ++i) {
            const auto& ev = buffer_[i];
            if (!first)
                json += ",";
            first = false;
            auto it = id_to_name.find(ev.id);
            const char* event_name = it != id_to_name.end() ? it->second.c_str() : "unknown";
            json += "{\"name\":\"";
            json += event_name;
            json += "\",\"ph\":\"X\",\"ts\":";
            json += std::to_string(ev.timestamp_us);
            json += ",\"dur\":";
            json += std::to_string(ev.duration_us);
            json += ",\"pid\":";
            json += std::to_string(ev.ctx.request_id >= 0 ? ev.ctx.request_id : 0);
            json += ",\"tid\":";
            json += std::to_string(ev.ctx.layer >= 0 ? ev.ctx.layer : 0);
            if (ev.ctx.phase[0])
                json += std::string(",\"args\":{\"phase\":\"") + ev.ctx.phase + "\"}";
            json += "}";
        }
        json += "]";
        return json;
    }

    // 重置 trace buffer，不清空 names_。
    void reset() {
        write_pos_.store(0, std::memory_order_relaxed);
        overflow_.store(false, std::memory_order_relaxed);
    }

    bool overflow() const { return overflow_.load(std::memory_order_relaxed); }
    size_t size() const {
        return std::min(static_cast<size_t>(write_pos_.load(std::memory_order_relaxed)), kCapacity);
    }

private:
    std::unique_ptr<PerfTraceEvent[]> buffer_;
    std::atomic<size_t> write_pos_{0};
    std::atomic<bool> overflow_{false};
    std::atomic<bool> enabled_{false};
};

class PerfProfiler {
public:
    static PerfProfiler& instance();

    void enable() { enabled_.store(true, std::memory_order_relaxed); }
    void disable() { enabled_.store(false, std::memory_order_relaxed); }
    bool enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }

    // Trace 开启/关闭
    void set_trace_enabled(bool v) { trace_.set_enabled(v); }
    bool trace_enabled() const { return trace_.enabled(); }
    PerfTrace& trace() { return trace_; }
    const PerfTrace& trace() const { return trace_; }

    // 导出 trace 为 JSON（Chrome Trace 格式）
    std::string trace_to_json() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::unordered_map<PerfEventId, std::string> id_to_name;
        id_to_name.reserve(id_to_name_.size());
        for (size_t i = 0; i < id_to_name_.size(); ++i) {
            if (id_to_name_[i])
                id_to_name[i] = *id_to_name_[i];
        }
        return trace_.to_json(id_to_name);
    }

    bool trace_overflow() const { return trace_.overflow(); }
    size_t trace_size() const { return trace_.size(); }

#ifdef USE_CUDA
    void set_use_cuda_events(bool use_cuda) {
        use_cuda_events_.store(use_cuda, std::memory_order_relaxed);
    }
    bool use_cuda_events() const {
        return use_cuda_events_.load(std::memory_order_relaxed);
    }
#else
    void set_use_cuda_events(bool) {}
    bool use_cuda_events() const {
        return false;
    }
#endif

    // 全局名字 -> id 注册表。线程安全，同一 name 多次调用返回同一 id。
    // 只写一次：注册后不再修改 names_。

    // thread-local name -> id 缓存，供 PERF_SCOPE_FMT 等动态层名使用。
    // 避免每次调用都走全局互斥。
    PerfEventId intern_name_fast(const char* name) {
        thread_local std::unordered_map<std::string, PerfEventId> tl_cache;
        auto it = tl_cache.find(name);
        if (it != tl_cache.end())
            return it->second;
        PerfEventId id = intern_name(name);
        tl_cache[name] = id;
        return id;
    }

    // ID 化记录：热路径无字符串拷贝 / 哈希。
    void record(PerfEventId id, double ms) {
        if (!enabled() || id == kInvalidPerfEventId)
            return;
        // Lazily register buffer on first use per thread
        thread_local bool buf_registered = false;
        if (!buf_registered) {
            register_thread_local(&thread_local_buffer());
            buf_registered = true;
        }
        auto& buf = thread_local_buffer();
        auto& acc = buf[id % kMaxPerfEvents];
        acc.count++;
        acc.total_ms += ms;
        acc.min_ms = std::min(acc.min_ms, ms);
        acc.max_ms = std::max(acc.max_ms, ms);
        acc.last_ms = ms;
    }

#ifdef USE_CUDA
    // Phase 3: 从 thread-local 池获取空闲 slot，不足时懒创建。
    size_t acquire_slot() {
        auto& cuda_state = thread_local_cuda_state();
        for (size_t i = 0; i < cuda_state.pool.size(); ++i) {
            if (!cuda_state.pool[i]->in_use) {
                cuda_state.pool[i]->in_use = true;
                return i;
            }
        }
        auto slot = std::make_unique<CudaEventSlot>();
        slot->in_use = true;
        size_t idx = cuda_state.pool.size();
        cuda_state.pool.push_back(std::move(slot));
        return idx;
    }

    // Phase 3: deferred 事件存入 thread-local pending（无全局锁）。
    void record_deferred(PerfEventId id, size_t slot_idx) {
        if (!enabled() || id == kInvalidPerfEventId)
            return;
        auto& cuda_state = thread_local_cuda_state();
        cuda_state.pending.push_back({id, slot_idx});
    }

    // Phase 3: 遍历全部线程的 pending，同步最后一个 end 事件后批量取耗时，
    // 并入累加器，再归还 slot。
    void flush_deferred() {
        std::lock_guard<std::mutex> lock(mutex_);
        struct Snapshot {
            PerfEventId id;
            CudaEventSlot* slot;
        };
        std::vector<Snapshot> all_pending;
        for (auto* p : all_thread_cuda_locals()) {
            auto& pending = p->pending;
            all_pending.reserve(all_pending.size() + pending.size());
            for (auto& ev : pending) {
                if (ev.slot_idx < p->pool.size()) {
                    all_pending.push_back({ev.id, p->pool[ev.slot_idx].get()});
                }
            }
            pending.clear();
        }
        if (all_pending.empty())
            return;
        cudaEventSynchronize(all_pending.back().slot->end);
        for (auto& ev : all_pending) {
            float ms = 0;
            cudaEventElapsedTime(&ms, ev.slot->start, ev.slot->end);
            // Lazily register buffer on first use per thread
            thread_local bool buf_registered = false;
            if (!buf_registered) {
                register_thread_local(&thread_local_buffer());
                buf_registered = true;
            }
            auto& buf = thread_local_buffer();
            auto& acc = buf[ev.id % kMaxPerfEvents];
            acc.count++;
            acc.total_ms += static_cast<double>(ms);
            acc.min_ms = std::min(acc.min_ms, static_cast<double>(ms));
            acc.max_ms = std::max(acc.max_ms, static_cast<double>(ms));
            acc.last_ms = static_cast<double>(ms);
            ev.slot->in_use = false;
        }
    }
#endif

    // 合并全部线程 buffer 到全局 records_（字符串格式，供 Python/CLI 使用）。
    // reset() 不清 names_，因此静态驻留的 id 仍然有效。
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.clear();
        for (auto* ptr : all_thread_locals()) {
            ptr->fill(PerfAccumulator{});
        }
#ifdef USE_CUDA
        for (auto* ptr : all_thread_cuda_locals()) {
            ptr->pending.clear();
            for (auto& slot : ptr->pool) {
                slot->in_use = false;
            }
        }
#endif
        trace_.reset();
    }

    std::vector<std::pair<std::string, PerfRecord>> summary() const {
        std::lock_guard<std::mutex> lock(mutex_);
        merge_thread_locals();
        std::vector<std::pair<std::string, PerfRecord>> result;
        for (const auto& [k, v] : records_) {
            result.push_back({k, v});
        }
        return result;
    }

    void print_summary() const {
#ifdef USE_CUDA
        const_cast<PerfProfiler*>(this)->flush_deferred();
#endif
        auto data = summary();
        if (data.empty())
            return;

        printf("\n========== Performance Profile ==========\n");
        printf("%-45s %8s %10s %10s %10s %10s\n", "Operation", "Count", "Total(ms)", "Avg(ms)",
               "Min(ms)", "Max(ms)");
        printf("%-45s %8s %10s %10s %10s %10s\n", "---------------------------------------------",
               "--------", "----------", "----------", "----------", "----------");

        std::sort(data.begin(), data.end(), [](const auto& a, const auto& b) {
            return a.second.total_ms > b.second.total_ms;
        });

        double grand_total = 0;
        for (const auto& [name, rec] : data) {
            double avg = rec.count > 0 ? rec.total_ms / rec.count : 0;
            printf("%-45s %8lld %10.2f %10.3f %10.3f %10.3f\n", name.c_str(), (long long)rec.count,
                   rec.total_ms, avg, rec.min_ms, rec.max_ms);
            grand_total += rec.total_ms;
        }

        printf("%-45s %8s %10.2f\n", "TOTAL", "", grand_total);
        printf("==========================================\n\n");
    }

private:
    PerfProfiler() = default;

    // ---- 全局名字注册表 (name -> id) ----
    std::mutex name_mutex_;
    std::unordered_map<std::string, PerfEventId> names_;

    // ---- Thread-local 固定数组累加器 ----
    using ThreadLocalBuffer = std::array<PerfAccumulator, kMaxPerfEvents>;

    static ThreadLocalBuffer& thread_local_buffer() {
        thread_local ThreadLocalBuffer buf;
        return buf;
    }

    struct ThreadLocalWrapper {
        ThreadLocalBuffer* ptr;
        bool registered = false;
        ThreadLocalWrapper() {
            ptr = &thread_local_buffer();
            register_thread_local(ptr);
            registered = true;
        }
    };

    static ThreadLocalBuffer* thread_local_buffer_ptr() {
        thread_local ThreadLocalWrapper wrapper;
        return wrapper.ptr;
    }

    // Track all thread-local buffers for cleanup and merging
    static std::vector<ThreadLocalBuffer*>& all_thread_locals() {
        static std::vector<ThreadLocalBuffer*> instances;
        return instances;
    }

    static std::mutex& tl_registry_mutex() {
        static std::mutex m;
        return m;
    }

    static void register_thread_local(ThreadLocalBuffer* ptr) {
        std::lock_guard<std::mutex> lock(tl_registry_mutex());
        all_thread_locals().push_back(ptr);
    }

#ifdef USE_CUDA
    // Track all thread-local CUDA states for flush
    static std::vector<PerfThreadCudaState*>& all_thread_cuda_locals() {
        static std::vector<PerfThreadCudaState*> instances;
        return instances;
    }

    static void register_thread_cuda_local(PerfThreadCudaState* ptr) {
        std::lock_guard<std::mutex> lock(tl_registry_mutex());
        all_thread_cuda_locals().push_back(ptr);
    }

#endif

    // 合并线程 buffer -> 全局 records_（只合并有数据的 id）
    void merge_thread_locals() const {
        for (auto* buf : all_thread_locals()) {
            for (PerfEventId id = 1; id < kMaxPerfEvents; ++id) {
                const auto& acc = (*buf)[id];
                if (acc.count == 0)
                    continue;
                const std::string* name = name_for_id(id);
                if (!name)
                    continue;
                auto& rec = records_[*name];
                rec.name = *name;
                rec.total_ms += acc.total_ms;
                rec.count += acc.count;
                rec.min_ms = std::min(rec.min_ms, acc.min_ms);
                rec.max_ms = std::max(rec.max_ms, acc.max_ms);
                rec.last_ms = acc.last_ms;
            }
        }
    }

    // 根据 id 反查名字（只读，不持有锁；names_ 注册后只读）。
    const std::string* name_for_id(PerfEventId id) const {
        if (id == kInvalidPerfEventId || id >= names_.size() + 1)
            return nullptr;
        if (id >= id_to_name_.size())
            return nullptr;
        const std::string* p = id_to_name_[id].get();
        return p;
    }

    // id -> name 反向映射。注册名字时同步填充。
    mutable std::vector<std::unique_ptr<std::string>> id_to_name_;

    // 线程安全注册名字并填充 id_to_name_。
    PerfEventId intern_name_with_reverse(const char* name) {
        std::lock_guard<std::mutex> lock(name_mutex_);
        auto it = names_.find(name);
        if (it != names_.end())
            return it->second;
        PerfEventId id = static_cast<PerfEventId>(names_.size() + 1);
        names_[name] = id;
        if (id >= static_cast<PerfEventId>(id_to_name_.size())) {
            id_to_name_.resize(id + 1);
        }
        id_to_name_[id] = std::make_unique<std::string>(name);
        return id;
    }

    // 全局 records_，供 summary/print_summary 使用。
    mutable std::mutex mutex_;
    mutable std::unordered_map<std::string, PerfRecord> records_;

#ifdef USE_CUDA
    std::vector<std::pair<PerfEventId, size_t>> deferred_events_;
#endif

    std::atomic<bool> enabled_{false};
#ifdef USE_CUDA
    // Default false: must be explicitly enabled when running on CUDA device.
    // CPU-only inference must NOT use CUDA events (they record nothing on CPU ops).
    std::atomic<bool> use_cuda_events_{false};
#else
    std::atomic<bool> use_cuda_events_{false};
#endif

    // Phase 4: Trace ring buffer
    PerfTrace trace_;

public:
    // 注册名字并返回 id；线程安全，同一 name 多次调用返回同一 id。
    PerfEventId intern_name(const char* name) {
        return intern_name_with_reverse(name);
    }

#ifdef USE_CUDA
    // 供 PerfScopeTimer 访问当前线程的 CUDA event 池。
    static PerfThreadCudaState& thread_local_cuda_state() {
        thread_local PerfThreadCudaState state;
        static thread_local bool registered = false;
        if (!registered) {
            register_thread_cuda_local(&state);
            registered = true;
        }
        return state;
    }
#endif
};

// ---- RAII timer ----
// Phase 3: 增加 stream 参数，显式记录到指定 CUDA stream。
// Phase 4: 析构时同时写入 trace ring buffer（如果开启）。
#ifdef USE_CUDA
class PerfScopeTimer {
public:
    explicit PerfScopeTimer(PerfEventId id, cudaStream_t stream = 0) : id_(id), stream_(stream) {
        if (!PerfProfiler::instance().enabled() || id_ == kInvalidPerfEventId)
            return;
        if (PerfProfiler::instance().use_cuda_events()) {
            use_cuda_ = true;
            slot_idx_ = PerfProfiler::instance().acquire_slot();
        } else {
            use_cuda_ = false;
            start_ = std::chrono::steady_clock::now();
        }
        trace_start_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
    }

    ~PerfScopeTimer() {
        if (!PerfProfiler::instance().enabled() || id_ == kInvalidPerfEventId)
            return;
        int64_t duration_us = 0;
        if (use_cuda_) {
            auto& slot = PerfProfiler::instance().thread_local_cuda_state().pool[slot_idx_];
            cudaEventRecord(slot->end, stream_);
            PerfProfiler::instance().record_deferred(id_, slot_idx_);
            // Trace 使用时间戳估算，不等待 GPU 完成
            auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
            duration_us = static_cast<int32_t>(now - trace_start_us_);
        } else {
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start_).count();
            PerfProfiler::instance().record(id_, ms);
            auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
            duration_us = static_cast<int32_t>(now - trace_start_us_);
        }
        // Phase 4: 写入 trace ring buffer
        if (PerfProfiler::instance().trace_enabled()) {
            PerfContext ctx = get_perf_context();
            PerfProfiler::instance().trace().record(id_, duration_us, ctx);
        }
    }

    PerfScopeTimer(const PerfScopeTimer&) = delete;
    PerfScopeTimer& operator=(const PerfScopeTimer&) = delete;

private:
    PerfEventId id_;
    cudaStream_t stream_;
    bool use_cuda_ = false;
    size_t slot_idx_ = 0;
    std::chrono::steady_clock::time_point start_;
    int64_t trace_start_us_ = 0;
};
#else
// 非 CUDA 构建：去掉 cudaStream_t 依赖。
class PerfScopeTimer {
public:
    explicit PerfScopeTimer(PerfEventId id) : id_(id) {
        if (!PerfProfiler::instance().enabled() || id_ == kInvalidPerfEventId)
            return;
        start_ = std::chrono::steady_clock::now();
        trace_start_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
    }

    ~PerfScopeTimer() {
        if (!PerfProfiler::instance().enabled() || id_ == kInvalidPerfEventId)
            return;
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        PerfProfiler::instance().record(id_, ms);
        // Phase 4: 写入 trace ring buffer
        if (PerfProfiler::instance().trace_enabled()) {
            auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
            int64_t duration_us = static_cast<int32_t>(now - trace_start_us_);
            PerfContext ctx = get_perf_context();
            PerfProfiler::instance().trace().record(id_, duration_us, ctx);
        }
    }

    PerfScopeTimer(const PerfScopeTimer&) = delete;
    PerfScopeTimer& operator=(const PerfScopeTimer&) = delete;

private:
    PerfEventId id_;
    std::chrono::steady_clock::time_point start_;
    int64_t trace_start_us_ = 0;
};
#endif

// ---- Macros ----
// 字面量 name 按调用点 static PerfEventId 驻留一次，热路径只传 id。
// PERF_SCOPE_FMT 对动态层名使用 intern_name_fast() 做 thread-local 缓存。

#if defined(FORGE_PROFILING) && FORGE_PROFILING == 1
#    define PERF_SCOPE(name)                                                  \
        static ::forge::PerfEventId _eid_##__LINE__ =                         \
            ::forge::PerfProfiler::instance().intern_name(name);              \
        ::forge::PerfScopeTimer _perf_timer_##__LINE__(_eid_##__LINE__)

#    define PERF_RECORD(name, ms)                                             \
        do {                                                                  \
            static ::forge::PerfEventId _eid_##__LINE__ =                     \
                ::forge::PerfProfiler::instance().intern_name(name);          \
            ::forge::PerfProfiler::instance().record(_eid_##__LINE__, ms);    \
        } while (0)

#    define PERF_SCOPE_FMT(fmt, ...)                                          \
            char _perf_name_buf_##__LINE__[256];                              \
            std::snprintf(_perf_name_buf_##__LINE__, sizeof(_perf_name_buf_##__LINE__), \
                          fmt, __VA_ARGS__);                                  \
            static ::forge::PerfEventId _eid_##__LINE__ =                     \
                ::forge::PerfProfiler::instance().intern_name_fast(           \
                    _perf_name_buf_##__LINE__);                               \
            ::forge::PerfScopeTimer _perf_timer_##__LINE__(_eid_##__LINE__)

#    define SET_PERF_CONTEXT(...) ::forge::set_perf_context({__VA_ARGS__})
#    define CLEAR_PERF_CONTEXT() ::forge::clear_perf_context()
#else
#    define PERF_SCOPE(name) ((void)0)
#    define PERF_RECORD(name, ms) ((void)0)
#    define PERF_SCOPE_FMT(fmt, ...) ((void)0)
#    define SET_PERF_CONTEXT(...) ((void)0)
#    define CLEAR_PERF_CONTEXT() ((void)0)
#endif

}  // namespace forge
