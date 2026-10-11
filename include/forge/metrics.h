/**
 * metrics.h - process-wide performance metrics for the web dashboard.
 *
 * Why this exists
 * ---------------
 * The repo had no metrics subsystem at all: `forge-server --perf` printed one
 * stdout line per request and `forge-cli` printed one stats line per turn, so
 * there was nothing to chart and no TTFT / inter-token latency anywhere. This
 * header is the single place that accumulates them, so `forge-cli --dashboard`
 * and `forge-server --dashboard` show the same numbers instead of two
 * independently drifting implementations.
 *
 * Deliberate constraints
 * ----------------------
 * * No httplib, no nlohmann/json. This header lives in `forge_infer`, which the
 *   Python module also links; dragging an HTTP stack or a JSON library in there
 *   would bloat `forge.so`. JSON serialization lives in the dashboard module,
 *   which consumes the POD `MetricsSnapshot` returned by snapshot().
 * * `Generator::generate()` is NOT instrumented. Both binaries already own a
 *   per-token callback (`forge-cli`'s stream_cb, the server's Generator
 *   TokenCallback), so `RequestScope::on_token()` is called from there and the
 *   decode hot path stays untouched.
 *
 * Threading contract (the important part)
 * ----------------------------------------
 * Two kinds of numbers, two different owners:
 *
 *   EngineGauges -- KV occupancy, prefix-cache hit rate, session and queue
 *                   counts. Owned by the INFERENCE THREAD and published only
 *                   from inside its own lock. They must never be polled from
 *                   another thread: `KVCache::active_bytes()` (kv_cache.cpp)
 *                   and `KVCache::num_free_slots()` walk the per-layer cell
 *                   tables with no lock at all, so a concurrent read is a data
 *                   race, not just a stale value. (The PagedKVStorage
 *                   equivalents do lock, so the two backends differ here.)
 *
 *   HostGauges -- VRAM, host RSS, CPU%. Read from any thread, because
 *                   `cudaMemGetInfo` and /proc are independent of engine state.
 *
 * Counters are atomics; the recent-request ring and the percentile reservoirs
 * sit behind one mutex that is taken once per request and once per dashboard
 * poll (1 Hz), never on the per-token path.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace forge {

class GenerationResult;

// ---- Per-request record ----------------------------------------------------

enum class RequestStatus : uint32_t {
    Ok = 0,
    Error = 1,       // 500: inference threw
    Aborted = 2,     // client disconnected mid-stream
    Rejected = 3,    // 429: admission control refused it
    BadRequest = 4,  // 400: prompt exceeds capacity / unparseable body
};

struct RequestRecord {
    int64_t start_ms = 0;  // wall clock when the request arrived
    int64_t end_ms = 0;
    // Wall-clock milliseconds. queue_ms is arrival -> inference lock acquired,
    // i.e. how long the request waited behind other queued work. ttft_ms is
    // arrival -> first token handed to the client, so it INCLUDES queue_ms --
    // that is what a user-perceived "time to first token" means, and it is not
    // the same number as GenerationResult::prefill_ms (which starts after the
    // lock and covers the prompt forward plus the first sample).
    double queue_ms = 0.0;
    double ttft_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    // Prompt tokens served from a cached prefix instead of being re-prefilled.
    int prefix_reused = 0;
    double tok_per_s = 0.0;
    double itl_p50_ms = 0.0;  // inter-token latency, median over this request
    double itl_p90_ms = 0.0;
    // Speculative acceptance rate in [0,1]; negative means "not applicable".
    double spec_accept = -1.0;
    RequestStatus status = RequestStatus::Ok;
    bool streaming = false;
    bool has_ttft = false;
    // FNV-1a of the session id. The dashboard groups by session for the
    // conversation view, and a hash is enough for that -- the raw id would leak
    // client-supplied strings into a page served on localhost.
    uint32_t session_hash = 0;
};

// ---- Gauges ----------------------------------------------------------------

// Published by the inference thread while it holds its own lock. Never read
// straight out of the engine from another thread; see the header comment.
struct EngineGauges {
    int64_t kv_active_bytes = 0;
    int64_t kv_total_bytes = 0;
    // -1 when the backend does not expose pages (contiguous storage).
    int kv_free_pages = -1;
    int kv_pool_tokens = 0;
    int kv_page_size = 0;
    int sessions = 0;
    int active_requests = 0;
    int64_t prefix_cache_hits = 0;
    int64_t prefix_cache_misses = 0;
    // Hits served by restoring pages from the host blob pool (paged mode only).
    int64_t prefix_cache_host_hits = 0;
    bool tiered_kv = false;
    bool paged_kv = false;
};

// Polled from any thread.
struct HostGauges {
    int64_t vram_total_bytes = 0;
    int64_t vram_used_bytes = 0;
    int64_t rss_bytes = 0;
    double cpu_pct = 0.0;
};

// ---- Per-operator snapshot -------------------------------------------------

// PerfProfiler::summary() is NOT idempotent: it merges every thread-local
// accumulator into the global records_ on each call, so calling it twice
// double-counts (see perf_profiler.h and the note in server_state.cpp).
// Whoever needs the breakdown must call it exactly once per request window and
// hand the result here; the dashboard only ever reads this copy.
struct OperatorRecord {
    std::string name;
    double total_ms = 0.0;
    int64_t count = 0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double avg_ms = 0.0;
};

// ---- Live request progress -------------------------------------------------

// What the request currently in flight is doing. This is the difference between
// a dashboard that animates during generation and one that looks frozen: a
// 30-second decode used to publish nothing at all until it finished, so every
// chart sat flat and "active requests" read 0 the whole time.
struct LiveRequest {
    bool active = false;
    int tokens = 0;         // tokens emitted so far by the in-flight request
    int prompt_tokens = 0;  // estimated from the message count until finish()
    double elapsed_ms = 0;  // wall time since it started
    // Decode rate over the last ~1 s window rather than the whole request, so a
    // long generation shows its current speed instead of a slowly-falling
    // average that is dominated by prefill.
    double tok_per_s = 0;
    // Inter-token latency over the same recent window.
    double itl_ms = 0;
    int64_t start_ms = 0;
    bool streaming = false;
    uint32_t session_hash = 0;
};

// ---- Snapshot --------------------------------------------------------------

struct MetricsSnapshot {
    // Most recent requests, newest last.
    std::vector<RequestRecord> requests;
    // Reservoirs backing the percentile fields (sorted by finalize()).
    std::vector<float> ttft_samples_ms;
    std::vector<float> itl_samples_ms;
    // Instantaneous time series the dashboard appends to on each poll.
    std::vector<double> series_tok_per_s;
    std::vector<double> series_ttft_ms;
    std::vector<double> series_itl_p50_ms;
    std::vector<int64_t> series_vram_used;
    std::vector<int64_t> series_kv_active;
    std::vector<int64_t> series_t_ms;

    EngineGauges engine;
    HostGauges host;
    LiveRequest live;

    uint64_t n_requests = 0;
    uint64_t n_errors = 0;
    uint64_t n_rejected = 0;
    uint64_t total_prompt_tokens = 0;
    uint64_t total_completion_tokens = 0;
    double total_prefill_ms = 0.0;
    double total_decode_ms = 0.0;
    uint64_t total_queue_ms = 0.0;

    // Operator breakdown of the most recent request window (not accumulated).
    std::vector<OperatorRecord> operators;
    bool operators_available = false;

    std::string model_id;
    std::string arch;
    std::string app;  // "forge-cli" | "forge-server"
    int64_t start_ms = 0;
    int64_t now_ms = 0;
    bool profiling_enabled = false;
};

// ---- Registry --------------------------------------------------------------

class MetricsRegistry {
public:
    static MetricsRegistry& instance();

    // Off by default and zero-cost when off: RequestScope::on_token() and the
    // per-request publishes all short-circuit on this one relaxed atomic load.
    void set_enabled(bool on) { enabled_.store(on, std::memory_order_relaxed); }
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

    void set_app(std::string app) { app_ = std::move(app); }
    void set_model(std::string model_id, std::string arch);

    MetricsSnapshot snapshot() const;

    void publish_engine_gauges(const EngineGauges& g);
    void publish_host_gauges(const HostGauges& g);
    HostGauges host_gauges() const;
    EngineGauges engine_gauges() const;

    /// Called by the inference thread (from inside RequestScope::on_token) a few
    /// times per second while a request is in flight, so the KV gauges on screen
    /// track the decode instead of updating only at the boundaries.
    ///
    /// This exists because KV occupancy cannot be read from another thread --
    /// see the threading contract at the top of this file -- so refreshing it
    /// mid-request has to happen on the thread that owns the engine. Both
    /// binaries install one of these when they start a dashboard; the callback
    /// must be cheap (it runs once per throttle window, not per token) and must
    /// not take any lock the inference path already holds.
    using LiveGaugeProvider = std::function<void(EngineGauges&)>;
    void set_live_gauge_provider(LiveGaugeProvider fn) {
        std::lock_guard<std::mutex> lock(live_provider_mutex_);
        live_provider_ = std::move(fn);
    }

    /// Appends one point to every time series from the current state. Called by
    /// the dashboard's ~1 Hz sampler so the charts advance in wall-clock time,
    /// including while a request is in flight. Deliberately NOT called by
    /// RequestScope::finish(): an event-per-request series has an x-axis made of
    /// completions, so a slow request leaves a misleading gap and an idle server
    /// produces no line at all.
    void sample_now();

    /// Progress of the request currently in flight. Called from the inference
    /// thread (RequestScope); the dashboard only reads the result.
    void publish_live(int64_t start_ms, int tokens, int prompt_tokens, double elapsed_ms,
                      double tok_per_s, double itl_ms, bool streaming, uint32_t session_hash);
    void clear_live();

    // Records a request that never finished normally: refused by admission
    // control (429), rejected before inference (400), or abandoned after a
    // thrown exception (500). Bypasses RequestScope so the counters and the
    // recent-requests table stay truthful about traffic the engine never
    // completed.
    //
    // If a live RequestScope exists for the same request, call abandon() on it
    // first, or its destructor will record that request a second time as an
    // Error.
    void record_rejected(int64_t arrival_ms, int prompt_tokens, RequestStatus status,
                         bool streaming, uint32_t session_hash);

    void reset();

    // ---- RAII per-request handle -----------------------------------------
    //
    // Usage from a binary that owns a per-token callback:
    //
    //   MetricsRegistry::RequestScope scope(creq.stream, session_hash(creq.session_id));
    //   ...take the inference lock...
    //   scope.on_lock_acquired();
    //   scope.on_token();            // once per emitted token
    //   scope.finish(result, RequestStatus::Ok);
    //
    // The destructor publishes an Aborted record if finish() never ran, so a
    // thrown exception cannot silently vanish from the dashboard.
    class RequestScope {
    public:
        // `arrival_ms` / `arrival_mono_ms` let a caller that already stamped the
        // request (an HTTP handler, before JSON parsing and admission control)
        // backdate the scope. Pass 0 / 0.0 to mean "now".
        //
        // `streaming` and `session_hash` are known only after the body is
        // parsed, which is exactly why the arrival stamp is separate: TTFT has
        // to include the parsing and queueing the client also waited through.
        RequestScope(bool streaming, uint32_t session_hash = 0, int64_t arrival_ms = 0,
                     double arrival_mono_ms = 0.0);
        ~RequestScope();
        RequestScope(const RequestScope&) = delete;
        RequestScope& operator=(const RequestScope&) = delete;

        void on_lock_acquired();
        void on_token();
        void set_prefix_reused(int n) { prefix_reused_ = n; }
        /// Called once per request, right after construction. Also refreshes the
        /// live block so the prompt size is on screen before the first token
        /// arrives -- the constructor cannot know it yet.
        void set_prompt_tokens(int n);
        void finish(const GenerationResult& result, RequestStatus status);
        /// Give up on this request WITHOUT recording anything.
        ///
        /// Needed by callers that bail out after constructing the scope but
        /// before inference -- admission control refusing a request (429), an
        /// unparseable body (400). Those paths record via
        /// MetricsRegistry::record_rejected(), so without this the destructor
        /// would publish a SECOND record for the same HTTP request, as an
        /// Error. A 429 showing up in both the rejection and the error counter
        /// is how this was found.
        void abandon() { finished_ = true; }
        // Publishes an operator breakdown captured from PerfProfiler exactly
        // once for this request window. See OperatorRecord's note.
        void set_operators(std::vector<OperatorRecord>&& ops);

        bool recording() const { return recording_; }

    private:
        bool recording_ = false;
        bool finished_ = false;
        bool lock_recorded_ = false;
        bool first_token_recorded_ = false;
        int64_t start_ms_ = 0;      // wall clock arrival
        double start_mono_ms_ = 0;  // monotonic arrival; all deltas use this
        double lock_ms_ = 0;
        double first_token_ms_ = 0;
        double last_token_ms_ = 0;
        // Live-progress bookkeeping. last_live_ms_ throttles the publish;
        // window_start_ms_ / window_tokens_ drive the trailing-window rate.
        double last_live_ms_ = 0.0;
        double window_start_ms_ = 0.0;
        int window_tokens_ = 0;
        int token_count_ = 0;
        int prompt_tokens_ = 0;
        int prefix_reused_ = 0;
        bool streaming_ = false;
        uint32_t session_hash_ = 0;
        std::vector<float> itl_;  // stack-local: no shared state on the hot path
        std::vector<OperatorRecord> operators_;
    };

private:
    MetricsRegistry();

    // Appends the record, the percentile reservoirs and one point per time
    // series. Takes the registry mutex once per completed request.
    void push_request(const RequestRecord& rec, const std::vector<float>& itl_ms);
    // Caller must hold mutex_.
    void push_sample(std::vector<float>& reservoir, float v);

    std::atomic<bool> enabled_{false};
    mutable std::mutex mutex_;

    std::vector<RequestRecord> requests_;
    std::vector<float> ttft_samples_ms_;
    std::vector<float> itl_samples_ms_;
    std::vector<double> series_tok_per_s_;
    std::vector<double> series_ttft_ms_;
    std::vector<double> series_itl_p50_ms_;
    std::vector<int64_t> series_vram_used_;
    std::vector<int64_t> series_kv_active_;
    std::vector<int64_t> series_t_ms_;
    std::vector<OperatorRecord> operators_;
    bool operators_available_ = false;

    std::atomic<uint64_t> n_requests_{0};
    std::atomic<uint64_t> n_errors_{0};
    std::atomic<uint64_t> n_rejected_{0};
    std::atomic<uint64_t> total_prompt_tokens_{0};
    std::atomic<uint64_t> total_completion_tokens_{0};
    std::atomic<uint64_t> total_prefill_us_{0};
    std::atomic<uint64_t> total_decode_us_{0};
    std::atomic<uint64_t> total_queue_us_{0};

    // Plain structs under their own mutex rather than std::atomic<>: both are
    // far wider than a lock-free atomic can hold, so std::atomic would emit
    // libatomic calls and force a -latomic link dependency on every consumer.
    // They are written at most a few times per second (once per request, once
    // per gauge tick), so a mutex costs nothing and keeps the hot request path
    // free of any lock at all.
    mutable std::mutex gauges_mutex_;
    EngineGauges engine_;
    HostGauges host_;
    // Progress of the request currently in flight. Guarded by gauges_mutex_ so a
    // dashboard poll and an on_token() update cannot tear.
    LiveRequest live_;

    // Number of RequestScopes alive right now. Atomic rather than derived from
    // the gauges: the CLI used to publish this by hand from four call sites and
    // one-shot mode simply forgot, which is how "0 active requests" showed up
    // during a 30-second generation.
    std::atomic<int> active_requests_{0};

    std::mutex live_provider_mutex_;
    LiveGaugeProvider live_provider_;

    // Last completed request's rate / ITL, so sample_now() can draw a flat line
    // instead of a cliff to zero while the engine is idle. Guarded by mutex_.
    double last_tok_per_s_ = 0.0;
    double last_itl_ms_ = 0.0;
    std::string model_id_;
    std::string arch_;
    std::string app_;
    int64_t start_ms_ = 0;
};

// ---- helpers ---------------------------------------------------------------

int64_t now_wall_ms();
int64_t now_mono_ms();
double now_mono_ms_f();
uint32_t fnv1a32(const std::string& s);

// Percentile of an already-sorted vector (nearest-rank). Returns 0 for empty.
double percentile_sorted(const std::vector<float>& sorted, double p);
double percentile_sorted(const std::vector<double>& sorted, double p);

// Current process RSS in bytes; 0 when unavailable.
int64_t read_rss_bytes();
// Process CPU utilisation over the last `interval_ms` wall milliseconds.
double read_cpu_pct(int interval_ms);

}  // namespace forge