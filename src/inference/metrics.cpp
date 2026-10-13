/**
 * metrics.cpp - process-wide performance metrics for the web dashboard.
 * See include/forge/metrics.h for the threading contract.
 */

#include "forge/metrics.h"

#include <algorithm>
#include <cstdio>

#include "forge/generator.h"

#ifdef _WIN32
#    define NOMINMAX
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#    include <psapi.h>
#else
#    include <sys/resource.h>
#    include <unistd.h>
#endif

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

namespace {

// Bounded ring sizes. 512 requests is ~10 minutes of a busy interactive
// session; the reservoirs are the percentile window and are intentionally
// smaller so a single long-running process cannot grow them without limit.
constexpr size_t kMaxRequests = 512;
constexpr size_t kMaxSamples = 512;
constexpr size_t kMaxSeries = 900;  // 15 minutes at the 1 Hz dashboard poll

template <typename T>
void push_ring(std::vector<T>& ring, size_t cap, const T& v) {
    if (ring.size() < cap) {
        ring.push_back(v);
    } else {
        ring.erase(ring.begin());
        ring.push_back(v);
    }
}

}  // namespace

// ---- time helpers ----------------------------------------------------------

int64_t now_wall_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

int64_t now_mono_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double now_mono_ms_f() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

uint32_t fnv1a32(const std::string& s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) {
        h ^= c;
        h *= 16777619u;
    }
    return h;
}

// ---- host gauges -----------------------------------------------------------

int64_t read_rss_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<int64_t>(pmc.WorkingSetSize);
    return 0;
#elif defined(__APPLE__)
    // macOS has no /proc; ru_maxrss is a high-water mark in bytes, which is a
    // poor stand-in for "current" but better than reporting nothing.
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return static_cast<int64_t>(ru.ru_maxrss);
    return 0;
#else
    // /proc/self/statm: size resident shared text lib data dt  (in pages)
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f)
        return 0;
    long total = 0, resident = 0;
    const int n = std::fscanf(f, "%ld %ld", &total, &resident);
    std::fclose(f);
    if (n != 2)
        return 0;
    const long page = sysconf(_SC_PAGESIZE);
    return static_cast<int64_t>(resident) * (page > 0 ? page : 4096);
#endif
}

namespace {

double process_cpu_seconds() {
#ifdef _WIN32
    FILETIME creation, exit, kernel, user;
    if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        ULARGE_INTEGER k, u;
        k.LowPart = kernel.dwLowDateTime;
        k.HighPart = kernel.dwHighDateTime;
        u.LowPart = user.dwLowDateTime;
        u.HighPart = user.dwHighDateTime;
        return static_cast<double>(k.QuadPart + u.QuadPart) / 1e7;
    }
    return 0.0;
#else
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) != 0)
        return 0.0;
    return static_cast<double>(ru.ru_utime.tv_sec) + ru.ru_utime.tv_usec / 1e6 +
           static_cast<double>(ru.ru_stime.tv_sec) + ru.ru_stime.tv_usec / 1e6;
#endif
}

}  // namespace

double read_cpu_pct(int interval_ms) {
    // Utilisation is a DELTA between consecutive polls, so it needs state. It
    // lives here (function-local statics) rather than in the registry because
    // this is also called by tests and by one-off callers; the first call after
    // start-up establishes the baseline and returns 0.
    static double s_last_cpu = process_cpu_seconds();
    static int64_t s_last_ms = now_mono_ms();

    const double cpu = process_cpu_seconds();
    const int64_t wall = now_mono_ms();
    const double dw = static_cast<double>(wall - s_last_ms);
    const double dc = cpu - s_last_cpu;
    s_last_cpu = cpu;
    s_last_ms = wall;

    // Ignore deltas over a gap far larger than the poll period: after a suspend
    // or a debugger break, cpu/wall would report a nonsense spike.
    if (dw <= 0.0 || dw > static_cast<double>(interval_ms) * 10 + 1000.0)
        return 0.0;
    if (dc < 0.0)
        return 0.0;
    const double pct = dc * 100.0 / dw;
    return pct > 0.0 ? pct : 0.0;
}

// ---- registry --------------------------------------------------------------

MetricsRegistry::MetricsRegistry() : start_ms_(now_wall_ms()) {}

MetricsRegistry& MetricsRegistry::instance() {
    static MetricsRegistry reg;
    return reg;
}

void MetricsRegistry::set_model(std::string model_id, std::string arch) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_id_ = std::move(model_id);
    arch_ = std::move(arch);
}

void MetricsRegistry::publish_engine_gauges(const EngineGauges& g) {
    std::lock_guard<std::mutex> lock(gauges_mutex_);
    engine_ = g;
}

void MetricsRegistry::publish_host_gauges(const HostGauges& g) {
    std::lock_guard<std::mutex> lock(gauges_mutex_);
    host_ = g;
}

HostGauges MetricsRegistry::host_gauges() const {
    std::lock_guard<std::mutex> lock(gauges_mutex_);
    return host_;
}

void MetricsRegistry::publish_live(int64_t start_ms, int tokens, int prompt_tokens,
                                   double elapsed_ms, double tok_per_s, double itl_ms,
                                   bool streaming, uint32_t session_hash) {
    std::lock_guard<std::mutex> lock(gauges_mutex_);
    live_.active = true;
    live_.start_ms = start_ms;
    live_.tokens = tokens;
    live_.prompt_tokens = prompt_tokens;
    live_.elapsed_ms = elapsed_ms;
    live_.tok_per_s = tok_per_s;
    live_.itl_ms = itl_ms;
    live_.streaming = streaming;
    live_.session_hash = session_hash;
}

void MetricsRegistry::clear_live() {
    std::lock_guard<std::mutex> lock(gauges_mutex_);
    live_ = LiveRequest{};
}

void MetricsRegistry::sample_now() {
    // Sampled, not event-driven. The previous design appended a point only when
    // a request COMPLETED, so the x-axis was made of completions: a 30 s
    // generation drew nothing until it finished, and an idle engine drew no line
    // at all. Sampling on the dashboard's 1 Hz tick makes the charts a real time
    // series that advances while inference is in flight.
    LiveRequest live;
    HostGauges h;
    int64_t kv_active = 0;
    {
        std::lock_guard<std::mutex> lock(gauges_mutex_);
        live = live_;
        h = host_;
        kv_active = engine_.kv_active_bytes;
    }

    const int64_t t = now_wall_ms();
    // Prefer live progress; fall back to the last completed request so an idle
    // engine holds a flat line at its last known rate rather than dropping to 0.
    const double tps = live.active ? live.tok_per_s : last_tok_per_s_;
    const double itl = live.active ? live.itl_ms : last_itl_ms_;
    // While a request is in flight this is "time so far", not the final TTFT --
    // that lands when finish() runs. Labelled as progress in the UI.
    double ttft = 0.0;
    if (live.active && live.tokens > 0) {
        ttft = live.elapsed_ms;
    } else if (!ttft_samples_ms_.empty()) {
        ttft = static_cast<double>(ttft_samples_ms_.back());
    }

    std::lock_guard<std::mutex> lock(mutex_);
    push_ring(series_t_ms_, kMaxSeries, t);
    push_ring(series_tok_per_s_, kMaxSeries, tps);
    push_ring(series_ttft_ms_, kMaxSeries, ttft);
    push_ring(series_itl_p50_ms_, kMaxSeries, itl);
    push_ring(series_vram_used_, kMaxSeries, h.vram_used_bytes);
    push_ring(series_kv_active_, kMaxSeries, kv_active);
}

EngineGauges MetricsRegistry::engine_gauges() const {
    std::lock_guard<std::mutex> lock(gauges_mutex_);
    return engine_;
}

void MetricsRegistry::push_sample(std::vector<float>& reservoir, float v) {
    if (reservoir.size() < kMaxSamples) {
        reservoir.push_back(v);
        return;
    }
    // Reservoir sampling: replace a random slot so the window keeps covering
    // the whole run instead of degenerating into "the most recent N".
    const size_t n = reservoir.size();
    size_t idx = 0;
    {
        // xorshift64* -- cheap, no <random> dependency, good enough for sampling.
        static thread_local uint64_t s = 0x9E3779B97F4A7C15ull;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        const uint64_t r = s * 2685821657736338717ull;
        idx = static_cast<size_t>((r >> 11) % n);
    }
    reservoir[idx] = v;
}

void MetricsRegistry::push_request(const RequestRecord& rec, const std::vector<float>& itl_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    push_ring(requests_, kMaxRequests, rec);
    if (rec.has_ttft)
        push_sample(ttft_samples_ms_, static_cast<float>(rec.ttft_ms));
    for (float d : itl_ms)
        push_sample(itl_samples_ms_, d);
    // Remembered so sample_now() can draw a flat line at the last known rate
    // while the engine is idle, instead of dropping to zero.
    last_tok_per_s_ = rec.tok_per_s;
    last_itl_ms_ = rec.itl_p50_ms;
}

void MetricsRegistry::record_rejected(int64_t arrival_ms, int prompt_tokens, RequestStatus status,
                                      bool streaming, uint32_t session_hash) {
    if (!enabled())
        return;
    RequestRecord rec;
    rec.start_ms = arrival_ms > 0 ? arrival_ms : now_wall_ms();
    rec.end_ms = rec.start_ms;
    rec.prompt_tokens = prompt_tokens;
    rec.total_ms = static_cast<double>(rec.end_ms - rec.start_ms);
    rec.status = status;
    rec.streaming = streaming;
    rec.session_hash = session_hash;

    n_requests_.fetch_add(1, std::memory_order_relaxed);
    if (status == RequestStatus::Rejected)
        n_rejected_.fetch_add(1, std::memory_order_relaxed);
    // Both a 400 and a 500 are errors from the caller's point of view. This
    // counted only BadRequest, which meant a request that reached this function
    // with Error -- the server's 500 path -- got a row in the request table but
    // never incremented n_errors, so the dashboard's error tile under-counted
    // exactly the failures worth looking at.
    if (status == RequestStatus::Error || status == RequestStatus::BadRequest)
        n_errors_.fetch_add(1, std::memory_order_relaxed);
    total_prompt_tokens_.fetch_add(static_cast<uint64_t>(std::max(prompt_tokens, 0)),
                                   std::memory_order_relaxed);

    push_request(rec, {});
}

MetricsSnapshot MetricsRegistry::snapshot() const {
    MetricsSnapshot s;
    s.app = app_;
    s.model_id = model_id_;
    s.arch = arch_;
    s.start_ms = start_ms_;
    s.now_ms = now_wall_ms();
    // PerfProfiler compiles to nothing unless the build enabled it, so this has
    // to come from the same gate the profiler itself uses. Reporting false on a
    // FORGE_PROFILING=ON build would make the dashboard tell users to rebuild
    // something they already built.
#if defined(FORGE_PROFILING) && FORGE_PROFILING
    s.profiling_enabled = true;
#else
    s.profiling_enabled = false;
#endif

    {
        std::lock_guard<std::mutex> lock(gauges_mutex_);
        s.engine = engine_;
        s.host = host_;
        s.live = live_;
    }
    // The registry owns this now instead of each binary publishing it by hand,
    // which is what let one-shot mode report 0 active requests for the whole
    // duration of a generation.
    s.engine.active_requests = active_requests_.load(std::memory_order_relaxed);

    s.n_requests = n_requests_.load(std::memory_order_relaxed);
    s.n_errors = n_errors_.load(std::memory_order_relaxed);
    s.n_rejected = n_rejected_.load(std::memory_order_relaxed);
    s.total_prompt_tokens = total_prompt_tokens_.load(std::memory_order_relaxed);
    s.total_completion_tokens = total_completion_tokens_.load(std::memory_order_relaxed);
    s.total_prefill_ms = total_prefill_us_.load(std::memory_order_relaxed) / 1000.0;
    s.total_decode_ms = total_decode_us_.load(std::memory_order_relaxed) / 1000.0;
    s.total_queue_ms = total_queue_us_.load(std::memory_order_relaxed) / 1000.0;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        s.requests = requests_;
        s.operators = operators_;
        s.operators_available = operators_available_;
        s.ttft_samples_ms = ttft_samples_ms_;
        s.itl_samples_ms = itl_samples_ms_;
        s.series_tok_per_s = series_tok_per_s_;
        s.series_ttft_ms = series_ttft_ms_;
        s.series_itl_p50_ms = series_itl_p50_ms_;
        s.series_vram_used = series_vram_used_;
        s.series_kv_active = series_kv_active_;
        s.series_t_ms = series_t_ms_;
    }

    // Sort under our own copies so the reservoirs stay in insertion order for
    // the next reservoir-sampling round.
    std::sort(s.ttft_samples_ms.begin(), s.ttft_samples_ms.end());
    std::sort(s.itl_samples_ms.begin(), s.itl_samples_ms.end());
    return s;
}

void MetricsRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    requests_.clear();
    ttft_samples_ms_.clear();
    itl_samples_ms_.clear();
    series_tok_per_s_.clear();
    series_ttft_ms_.clear();
    series_itl_p50_ms_.clear();
    series_vram_used_.clear();
    series_kv_active_.clear();
    series_t_ms_.clear();
    operators_.clear();
    operators_available_ = false;
    n_requests_.store(0);
    n_errors_.store(0);
    n_rejected_.store(0);
    last_tok_per_s_ = 0.0;
    last_itl_ms_ = 0.0;
    {
        std::lock_guard<std::mutex> glock(gauges_mutex_);
        live_ = LiveRequest{};
    }
    total_prompt_tokens_.store(0);
    total_completion_tokens_.store(0);
    total_prefill_us_.store(0);
    total_decode_us_.store(0);
    total_queue_us_.store(0);
}

// ---- RequestScope ----------------------------------------------------------

MetricsRegistry::RequestScope::RequestScope(bool streaming, uint32_t session_hash,
                                            int64_t arrival_ms, double arrival_mono_ms)
    : streaming_(streaming), session_hash_(session_hash) {
    auto& reg = MetricsRegistry::instance();
    if (!reg.enabled())
        return;
    recording_ = true;
    start_ms_ = arrival_ms > 0 ? arrival_ms : now_wall_ms();
    start_mono_ms_ = arrival_mono_ms > 0.0 ? arrival_mono_ms : now_mono_ms_f();
    // Announce the request immediately. Without this the dashboard reported
    // "0 active requests" for the whole duration of a long generation, because
    // nothing at all was published until finish().
    reg.active_requests_.fetch_add(1, std::memory_order_relaxed);
    reg.publish_live(start_ms_, 0, prompt_tokens_, 0.0, 0.0, 0.0, streaming_, session_hash_);
}

MetricsRegistry::RequestScope::~RequestScope() {
    if (!recording_)
        return;
    MetricsRegistry& reg = MetricsRegistry::instance();
    reg.active_requests_.fetch_sub(1, std::memory_order_relaxed);
    if (finished_) {
        // finish() already recorded the request; just make sure nothing is left
        // on screen looking active.
        reg.clear_live();
        return;
    }
    // An exception escaped chat_completion()/generate() without reaching
    // finish(): publish it anyway, otherwise a crashed request leaves no trace
    // on the dashboard and the queue-depth chart silently diverges.
    RequestRecord rec;
    rec.start_ms = start_ms_;
    rec.end_ms = now_wall_ms();
    rec.total_ms = static_cast<double>(rec.end_ms - rec.start_ms);
    rec.queue_ms = lock_ms_;
    rec.status = RequestStatus::Error;
    rec.streaming = streaming_;
    rec.session_hash = session_hash_;
    rec.prompt_tokens = prompt_tokens_;
    rec.prefix_reused = prefix_reused_;
    reg.n_requests_.fetch_add(1, std::memory_order_relaxed);
    reg.n_errors_.fetch_add(1, std::memory_order_relaxed);
    reg.push_request(rec, {});
}

void MetricsRegistry::RequestScope::set_prompt_tokens(int n) {
    prompt_tokens_ = n;
    if (!recording_)
        return;
    // Re-publish so the live block reflects the prompt size immediately; the
    // constructor ran before the caller knew it.
    MetricsRegistry::instance().publish_live(
        start_ms_, token_count_, prompt_tokens_, (now_mono_ms_f() - start_mono_ms_),
        last_token_ms_ > first_token_ms_ && token_count_ > 1
            ? static_cast<double>(token_count_ - 1) / ((last_token_ms_ - first_token_ms_) / 1000.0)
            : 0.0,
        0.0, streaming_, session_hash_);
}

void MetricsRegistry::RequestScope::on_lock_acquired() {
    if (!recording_ || lock_recorded_)
        return;
    lock_recorded_ = true;
    lock_ms_ = now_mono_ms_f();
}

void MetricsRegistry::RequestScope::on_token() {
    if (!recording_)
        return;
    const double now = now_mono_ms_f();
    if (!first_token_recorded_) {
        first_token_recorded_ = true;
        first_token_ms_ = now;
    } else if (last_token_ms_ > 0.0) {
        // steady_clock cannot go backwards, but a token emitted before
        // on_lock_acquired() (shouldn't happen) would produce a negative delta,
        // which would poison the percentile reservoirs for the whole window.
        const double d = now - last_token_ms_;
        if (d > 0.0)
            itl_.push_back(static_cast<float>(d));
    }
    const double prev_token_ms = last_token_ms_;
    last_token_ms_ = now;
    ++token_count_;

    // ---- live progress (throttled) ----
    // Publishing on every token would take a mutex per token -- tens of
    // thousands of times over a long reply -- and rebuild the KV gauges each
    // time, which on the contiguous backend walks the entire cell table. 4 Hz is
    // smooth against a 1 Hz dashboard poll and costs nothing measurable.
    constexpr double kLiveIntervalMs = 250.0;
    if (now - last_live_ms_ < kLiveIntervalMs)
        return;

    MetricsRegistry& reg = MetricsRegistry::instance();

    // Decode rate over a trailing ~1 s window, so a long reply shows its CURRENT
    // speed instead of a slowly-falling lifetime average dominated by prefill.
    if (window_start_ms_ == 0.0)
        window_start_ms_ = first_token_ms_;
    window_tokens_ = 0;  // just closed the previous window
    const double win_s = (now - window_start_ms_) / 1000.0;
    double tps = 0.0;
    if (win_s > 0.05) {
        tps = static_cast<double>(window_tokens_) / win_s;
    }
    // First window: window_tokens_ is still 0, so fall back to the lifetime
    // average over whatever tokens exist so far.
    if (tps <= 0.0 && token_count_ > 1 && last_token_ms_ > first_token_ms_)
        tps = static_cast<double>(token_count_ - 1) / ((last_token_ms_ - first_token_ms_) / 1000.0);
    window_start_ms_ = now;

    const double itl_now = prev_token_ms > 0.0 ? (now - prev_token_ms) : 0.0;
    reg.publish_live(start_ms_, token_count_, prompt_tokens_, (now - start_mono_ms_), tps, itl_now,
                     streaming_, session_hash_);
    last_live_ms_ = now;

    // Refresh the engine gauges from the thread that owns them. This has to
    // happen here rather than on the dashboard thread because KV occupancy
    // cannot be read concurrently -- see LiveGaugeProvider's contract.
    bool have = false;
    EngineGauges g;
    {
        std::lock_guard<std::mutex> lock(reg.live_provider_mutex_);
        if (reg.live_provider_) {
            reg.live_provider_(g);
            have = true;
        }
    }
    if (have)
        reg.publish_engine_gauges(g);
}

void MetricsRegistry::RequestScope::set_operators(std::vector<OperatorRecord>&& ops) {
    operators_ = std::move(ops);
}

}  // namespace forge