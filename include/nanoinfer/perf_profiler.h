#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nanoinfer {

struct PerfRecord {
    std::string name;
    double total_ms = 0.0;
    int64_t count = 0;
    double min_ms = 1e18;
    double max_ms = 0.0;
    double last_ms = 0.0;
};

class PerfProfiler {
public:
    static PerfProfiler& instance();

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool enabled() const { return enabled_; }

#ifdef USE_CUDA
    void set_use_cuda_events(bool use_cuda) { use_cuda_events_ = use_cuda; }
    bool use_cuda_events() const { return use_cuda_events_; }
#else
    void set_use_cuda_events(bool) {}
    bool use_cuda_events() const { return false; }
#endif

    void record(const std::string& name, double ms) {
        if (!enabled_) return;
        // Use thread-local buffer to avoid mutex contention in parallel regions.
        // Each thread accumulates into its own map; merge on summary/reset.
        auto& local = thread_local_records();
        auto& rec = local[name];
        rec.name = name;
        rec.total_ms += ms;
        rec.count++;
        rec.min_ms = std::min(rec.min_ms, ms);
        rec.max_ms = std::max(rec.max_ms, ms);
        rec.last_ms = ms;
    }

#ifdef USE_CUDA
    // Store deferred CUDA event pairs for later resolution
    void record_deferred(const std::string& name, cudaEvent_t start, cudaEvent_t end) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        deferred_events_.push_back({name, start, end});
    }

    // Resolve all deferred CUDA events: synchronize once, then compute all times
    void flush_deferred() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (deferred_events_.empty()) return;
        // Single synchronization point for all events
        cudaEventSynchronize(deferred_events_.back().end);
        for (auto& ev : deferred_events_) {
            float ms = 0;
            cudaEventElapsedTime(&ms, ev.start, ev.end);
            auto& rec = records_[ev.name];
            rec.name = ev.name;
            rec.total_ms += static_cast<double>(ms);
            rec.count++;
            rec.min_ms = std::min(rec.min_ms, static_cast<double>(ms));
            rec.max_ms = std::max(rec.max_ms, static_cast<double>(ms));
            rec.last_ms = static_cast<double>(ms);
            cudaEventDestroy(ev.start);
            cudaEventDestroy(ev.end);
        }
        deferred_events_.clear();
    }
#endif

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.clear();
        // Clear all thread-local buffers
        for (auto* ptr : all_thread_locals()) {
            ptr->clear();
        }
#ifdef USE_CUDA
        // Clean up any unresolved events
        for (auto& ev : deferred_events_) {
            cudaEventDestroy(ev.start);
            cudaEventDestroy(ev.end);
        }
        deferred_events_.clear();
#endif
    }

    std::vector<std::pair<std::string, PerfRecord>> summary() const {
        std::lock_guard<std::mutex> lock(mutex_);
        // Merge thread-local records into global map
        const_cast<PerfProfiler*>(this)->merge_thread_locals();
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
        if (data.empty()) return;

        printf("\n========== Performance Profile ==========\n");
        printf("%-45s %8s %10s %10s %10s %10s\n",
               "Operation", "Count", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)");
        printf("%-45s %8s %10s %10s %10s %10s\n",
               "---------------------------------------------", "--------",
               "----------", "----------", "----------", "----------");

        // Sort by total_ms descending
        std::sort(data.begin(), data.end(),
                  [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });

        double grand_total = 0;
        for (const auto& [name, rec] : data) {
            double avg = rec.count > 0 ? rec.total_ms / rec.count : 0;
            printf("%-45s %8lld %10.2f %10.3f %10.3f %10.3f\n",
                   name.c_str(), (long long)rec.count,
                   rec.total_ms, avg, rec.min_ms, rec.max_ms);
            grand_total += rec.total_ms;
        }

        printf("%-45s %8s %10.2f\n", "TOTAL", "", grand_total);
        printf("==========================================\n\n");
    }

private:
    PerfProfiler() = default;
    bool enabled_ = false;
#ifdef USE_CUDA
    bool use_cuda_events_ = true;  // default to CUDA events when built with CUDA
#else
    bool use_cuda_events_ = false;
#endif
    mutable std::mutex mutex_;
    std::unordered_map<std::string, PerfRecord> records_;
#ifdef USE_CUDA
    struct DeferredEvent {
        std::string name;
        cudaEvent_t start;
        cudaEvent_t end;
    };
    std::vector<DeferredEvent> deferred_events_;
#endif

    // Thread-local storage for lock-free recording in parallel regions
    using RecordMap = std::unordered_map<std::string, PerfRecord>;

    static RecordMap& thread_local_records() {
        thread_local RecordMap local_map;
        return local_map;
    }

    // Track all thread-local maps for cleanup and merging
    static std::vector<RecordMap*>& all_thread_locals() {
        static std::vector<RecordMap*> instances;
        return instances;
    }

    static std::mutex& tl_registry_mutex() {
        static std::mutex m;
        return m;
    }

    // Register a thread-local map on first use (for cleanup)
    static void register_thread_local(RecordMap* ptr) {
        std::lock_guard<std::mutex> lock(tl_registry_mutex());
        all_thread_locals().push_back(ptr);
    }

    void merge_thread_locals() {
        // Access thread-local directly for the calling thread
        auto& local = thread_local_records();
        if (!local.empty()) {
            for (auto& [name, rec] : local) {
                auto& global = records_[name];
                global.name = name;
                global.total_ms += rec.total_ms;
                global.count += rec.count;
                global.min_ms = std::min(global.min_ms, rec.min_ms);
                global.max_ms = std::max(global.max_ms, rec.max_ms);
            }
            local.clear();
        }
    }
};

// RAII scoped timer with runtime CUDA event support
// Uses std::chrono for CPU inference, CUDA events for GPU inference
class PerfScopeTimer {
public:
    explicit PerfScopeTimer(const std::string& name) : name_(name) {
#ifdef USE_CUDA
        if (PerfProfiler::instance().use_cuda_events()) {
            use_cuda_ = true;
            cudaEventCreate(&start_event_);
            cudaEventCreate(&end_event_);
            cudaEventRecord(start_event_);
        } else
#endif
        {
            use_cuda_ = false;
            start_ = std::chrono::high_resolution_clock::now();
        }
    }

    ~PerfScopeTimer() {
#ifdef USE_CUDA
        if (use_cuda_) {
            cudaEventRecord(end_event_);
            PerfProfiler::instance().record_deferred(name_, start_event_, end_event_);
        } else
#endif
        {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start_).count();
            PerfProfiler::instance().record(name_, ms);
        }
    }

    // Non-copyable
    PerfScopeTimer(const PerfScopeTimer&) = delete;
    PerfScopeTimer& operator=(const PerfScopeTimer&) = delete;

private:
    std::string name_;
    bool use_cuda_;
#ifdef USE_CUDA
    cudaEvent_t start_event_;
    cudaEvent_t end_event_;
#endif
    std::chrono::high_resolution_clock::time_point start_;
};

// Convenience macros
#define PERF_SCOPE(name) ::nanoinfer::PerfScopeTimer _perf_timer_##__LINE__(name)
#define PERF_RECORD(name, ms) ::nanoinfer::PerfProfiler::instance().record(name, ms)

} // namespace nanoinfer
