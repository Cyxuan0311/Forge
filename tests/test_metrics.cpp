/**
 * Metrics registry unit tests.
 *
 * Host-only and model-free. Everything asserted here is about the accounting
 * itself -- ring bounds, percentile math, TTFT/queue semantics, the disabled
 * path being genuinely free -- because those are the parts that silently
 * produce *wrong* numbers rather than obviously broken ones. A dashboard that
 * renders plausible-but-wrong tok/s is worse than no dashboard.
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "forge/generator.h"
#include "forge/metrics.h"

using namespace forge;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        ++g_checks;                                                   \
        if (!(cond)) {                                                \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); \
            ++g_fail;                                                 \
        } else {                                                      \
            std::printf("ok   %s\n", msg);                            \
        }                                                             \
    } while (0)

// ---------------------------------------------------------------------------
// Percentiles
// ---------------------------------------------------------------------------
static void test_percentile() {
    std::printf("\n--- percentile ---\n");

    std::vector<float> empty;
    CHECK(percentile_sorted(empty, 0.5) == 0.0, "empty vector -> 0.0");

    std::vector<float> one{42.0f};
    CHECK(percentile_sorted(one, 0.0) == 42.0f && percentile_sorted(one, 0.99) == 42.0f,
          "single sample -> that sample at every p");

    // Linear interpolation between neighbours, so p50 of {1,2,3,4} is 2.5.
    std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(std::fabs(percentile_sorted(v, 0.5) - 2.5f) < 1e-6, "p50 of 1..4 == 2.5");
    CHECK(std::fabs(percentile_sorted(v, 0.0) - 1.0f) < 1e-6, "p0 == min");
    CHECK(std::fabs(percentile_sorted(v, 1.0) - 4.0f) < 1e-6, "p100 == max");

    // Out-of-range percentiles must clamp rather than index off the end.
    CHECK(percentile_sorted(v, -1.0) == 1.0f, "p<0 clamps to min");
    CHECK(percentile_sorted(v, 2.0) == 4.0f, "p>1 clamps to max");
}

// ---------------------------------------------------------------------------
// Disabled registry is inert
// ---------------------------------------------------------------------------
static void test_disabled_is_free() {
    std::printf("\n--- disabled registry ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(false);
    reg.reset();

    {
        MetricsRegistry::RequestScope scope(true);
        CHECK(!scope.recording(), "scope is inert while disabled");
        scope.on_lock_acquired();
        for (int i = 0; i < 1000; ++i)
            scope.on_token();
        GenerationResult r;
        r.num_prompt_tokens = 10;
        r.num_generated_tokens = 20;
        r.prefill_ms = 5.0;
        r.decode_ms = 100.0;
        scope.finish(r, RequestStatus::Ok);
    }

    const MetricsSnapshot s = reg.snapshot();
    CHECK(s.n_requests == 0, "a disabled registry records no requests");
    CHECK(s.requests.empty(), "a disabled registry keeps no rows");
    CHECK(s.ttft_samples_ms.empty() && s.itl_samples_ms.empty(),
          "a disabled registry fills no percentile reservoirs");
    CHECK(s.series_tok_per_s.empty(), "a disabled registry appends no series points");
}

// ---------------------------------------------------------------------------
// TTFT / queue / ITL semantics
// ---------------------------------------------------------------------------
static void test_request_scope() {
    std::printf("\n--- request scope ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();
    reg.set_model("unit-test-model", "test-arch");
    reg.set_app("forge-test");

    GenerationResult r;
    r.num_prompt_tokens = 57;
    r.num_generated_tokens = 11;  // -> tokens_per_sec() uses generated-1
    r.prefill_ms = 41.0;
    r.decode_ms = 200.0;

    {
        MetricsRegistry::RequestScope scope(true, /*session_hash=*/0xABCD);
        CHECK(scope.recording(), "scope records while enabled");

        // Simulate 12 ms of queueing, then 4 tokens spaced 20 ms apart.
        std::this_thread::sleep_for(std::chrono::milliseconds(12));
        scope.on_lock_acquired();
        for (int i = 0; i < 4; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            scope.on_token();
        }
        scope.set_prefix_reused(9);
        scope.finish(r, RequestStatus::Ok);
    }

    const MetricsSnapshot s = reg.snapshot();
    CHECK(s.n_requests == 1, "one request recorded");
    CHECK(s.requests.size() == 1, "one row in the request table");

    const RequestRecord& rec = s.requests[0];
    CHECK(rec.has_ttft, "TTFT is present when a token was emitted");
    // TTFT must INCLUDE the queue wait, not start at the lock: that is the
    // whole point of stamping arrival before admission control.
    CHECK(rec.queue_ms >= 10.0, "queue_ms captures the pre-lock wait");
    CHECK(rec.ttft_ms > rec.queue_ms, "ttft_ms is measured from arrival, not from the lock");
    CHECK(rec.ttft_ms < 400.0, "ttft_ms is in a sane range for 12ms + 4x20ms");
    CHECK(rec.prompt_tokens == 57, "prompt token count comes from GenerationResult");
    CHECK(rec.completion_tokens == 11, "completion token count comes from GenerationResult");
    CHECK(rec.prefix_reused == 9, "prefix reuse is recorded");
    CHECK(rec.session_hash == 0xABCD, "session hash is carried through");
    CHECK(rec.streaming, "streaming flag is recorded");
    CHECK(rec.status == RequestStatus::Ok, "status is recorded");
    // tokens_per_sec() deliberately uses (generated - 1) because the first
    // token comes out of prefill.
    CHECK(std::fabs(rec.tok_per_s - 10.0 / 0.2) < 1e-6, "tok/s uses (generated-1) / decode_ms");

    // 3 inter-token deltas from 4 tokens; the first token only sets TTFT.
    CHECK(s.itl_samples_ms.size() == 3, "3 ITL samples from 4 tokens (no delta before the first)");
    CHECK(!s.ttft_samples_ms.empty(), "TTFT reservoir got a sample");
    CHECK(!rec.itl_p50_ms || rec.itl_p50_ms > 0.0, "per-request ITL p50 computed");

    // The series is NO LONGER event-per-request: it is sampled on the dashboard's
    // 1 Hz tick by sample_now(). Appending on completion meant the x-axis was made
    // of completions, so a long generation drew nothing until it finished and an
    // idle engine drew no line at all -- which is the "frozen page during
    // inference" symptom this replaced.
    CHECK(s.series_tok_per_s.empty(), "finishing a request does NOT append a series point");
    CHECK(s.series_t_ms.empty(), "series stays empty until sample_now() runs");
    // finish() must also leave nothing looking active.
    CHECK(!s.live.active, "live block is cleared on finish");
    CHECK(s.engine.active_requests == 0, "active_requests returns to 0 after finish");
}

// ---------------------------------------------------------------------------
// TTFT includes a caller-supplied arrival stamp
// ---------------------------------------------------------------------------
static void test_backdated_arrival() {
    std::printf("\n--- backdated arrival ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();

    // This is what the HTTP handlers do: stamp arrival, then parse/admit, then
    // build the scope. The queue wait must be counted even though the scope was
    // constructed after it happened.
    const int64_t arrival_wall = now_wall_ms();
    const double arrival_mono = now_mono_ms_f();
    std::this_thread::sleep_for(std::chrono::milliseconds(25));

    {
        MetricsRegistry::RequestScope scope(false, 0, arrival_wall, arrival_mono);
        scope.on_lock_acquired();
        scope.on_token();
        GenerationResult r;
        r.decode_ms = 1.0;
        scope.finish(r, RequestStatus::Ok);
    }

    const MetricsSnapshot s = reg.snapshot();
    CHECK(s.requests.size() == 1, "backdated request recorded");
    CHECK(s.requests[0].queue_ms >= 20.0,
          "queue_ms counts the wait that happened before the scope existed");
    CHECK(s.requests[0].ttft_ms >= 20.0, "ttft_ms includes the backdated wait");
}

// ---------------------------------------------------------------------------
// RAII: an exception that never reaches finish() still shows up
// ---------------------------------------------------------------------------
static void test_scope_destructor() {
    std::printf("\n--- scope destructor on exception ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();

    try {
        MetricsRegistry::RequestScope scope(true);
        scope.on_lock_acquired();
        scope.on_token();
        throw std::runtime_error("simulated inference failure");
    } catch (const std::exception&) {}

    const MetricsSnapshot s = reg.snapshot();
    CHECK(s.n_requests == 1, "an un-finished scope still records a request");
    CHECK(s.n_errors == 1, "an un-finished scope counts as an error");
    CHECK(s.requests.size() == 1 && s.requests[0].status == RequestStatus::Error,
          "the record is marked Error");
}

// ---------------------------------------------------------------------------
// finish() is idempotent
// ---------------------------------------------------------------------------
static void test_double_finish() {
    std::printf("\n--- double finish ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();

    {
        MetricsRegistry::RequestScope scope(true);
        scope.on_token();
        GenerationResult r;
        r.decode_ms = 10.0;
        scope.finish(r, RequestStatus::Ok);
        // A handler that both finishes explicitly and lets the destructor run
        // must not double-count: the dashboard's request count is the number
        // people sanity-check against their access log.
        scope.finish(r, RequestStatus::Ok);
    }

    const MetricsSnapshot s = reg.snapshot();
    CHECK(s.n_requests == 1, "a second finish() is a no-op");
    CHECK(s.requests.size() == 1, "a second finish() adds no second row");
}

// ---------------------------------------------------------------------------
// Rejections (429 / 400) never reached inference
// ---------------------------------------------------------------------------
static void test_record_rejected() {
    std::printf("\n--- rejected requests ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();

    // Insertion order is preserved in the table (oldest first), so index 0 is
    // the 429 and index 1 the 400.
    const int64_t t0 = now_wall_ms();
    reg.record_rejected(t0, 11, RequestStatus::Rejected, false, 0);
    reg.record_rejected(t0, 22, RequestStatus::BadRequest, true, 7);

    const MetricsSnapshot s = reg.snapshot();
    CHECK(s.n_requests == 2, "rejections count toward the request total");
    CHECK(s.n_rejected == 1, "429 increments the rejection counter");
    CHECK(s.n_errors == 1, "400 counts as an error");
    CHECK(s.requests.size() == 2, "rejections appear in the request table");
    CHECK(s.requests[0].status == RequestStatus::Rejected, "429 status is preserved");
    CHECK(s.requests[1].status == RequestStatus::BadRequest, "400 status is preserved");
    CHECK(!s.requests[0].has_ttft && !s.requests[1].has_ttft,
          "a request that never ran has no TTFT");
    CHECK(s.requests[0].prompt_tokens == 11 && s.requests[1].prompt_tokens == 22,
          "prompt token estimate is preserved");
    CHECK(s.requests[0].streaming == false && s.requests[1].streaming == true,
          "streaming flag is preserved");
    CHECK(s.requests[1].session_hash == 7, "session hash is preserved");
    CHECK(s.total_prompt_tokens == 33, "rejected prompt tokens still count toward the total");
}

// ---------------------------------------------------------------------------
// abandon(): a refusal before inference must be recorded exactly once
// ---------------------------------------------------------------------------
// Regression test. The HTTP handlers construct a RequestScope before admission
// control, so a 429 takes a path that records via MetricsRegistry::record_rejected()
// and never calls finish(). Without an explicit abandon(), ~RequestScope also
// published an Error row for the same request -- which showed up as a 429 landing
// in BOTH the rejection counter and the error counter.
static void test_abandon() {
    std::printf("\n--- abandon (early-exit) ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();

    // Exactly what a 429 handler does: construct the scope, bail out, record.
    {
        MetricsRegistry::RequestScope scope(false, 0);
        scope.abandon();
        reg.record_rejected(now_wall_ms(), 0, RequestStatus::Rejected, false, 0);
    }
    // And the 400-from-a-thrown-exception variant.
    {
        MetricsRegistry::RequestScope scope(true, 3);
        scope.abandon();
        reg.record_rejected(now_wall_ms(), 0, RequestStatus::Error, true, 3);
    }
    // A healthy request for contrast: finished, so the destructor stays quiet.
    {
        MetricsRegistry::RequestScope scope(true, 0);
        scope.on_lock_acquired();
        scope.on_token();
        GenerationResult r;
        r.decode_ms = 1.0;
        scope.finish(r, RequestStatus::Ok);
    }

    const MetricsSnapshot s = reg.snapshot();
    CHECK(s.n_requests == 3, "three requests total (2 refusals + 1 ok)");
    CHECK(s.requests.size() == 3, "each HTTP request produced exactly ONE row");
    CHECK(s.n_rejected == 1, "one 429");
    CHECK(s.n_errors == 1, "one error -- the abandoned scope did NOT add a second");
    int rejected_rows = 0, error_rows = 0, ok_rows = 0;
    for (const auto& r : s.requests) {
        if (r.status == RequestStatus::Rejected)
            ++rejected_rows;
        if (r.status == RequestStatus::Error)
            ++error_rows;
        if (r.status == RequestStatus::Ok)
            ++ok_rows;
    }
    CHECK(rejected_rows == 1 && error_rows == 1 && ok_rows == 3 - 2,
          "row statuses are 1 rejected / 1 error / 1 ok");
}

// ---------------------------------------------------------------------------
// Live progress -- the fix for a dashboard that looked frozen during inference
// ---------------------------------------------------------------------------
// Before this existed, nothing at all was published while a request was in
// flight: the request counter stayed 0, the time series had no points, and the KV
// gauge held its previous value. A 30 s decode therefore showed a completely
// static page until the very last moment. These assertions are the regression net.
static void test_live_progress() {
    std::printf("\n--- live progress ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();

    {
        MetricsRegistry::RequestScope scope(true, /*session_hash=*/42);
        scope.set_prompt_tokens(57);

        // A scope is announced immediately, before any token exists.
        MetricsSnapshot s = reg.snapshot();
        CHECK(s.live.active, "an in-flight request is visible before its first token");
        CHECK(s.engine.active_requests == 1, "active_requests is 1 while in flight");
        CHECK(s.live.session_hash == 42, "live carries the session hash");
        CHECK(s.live.prompt_tokens == 57, "live carries the prompt estimate");

        // Emit tokens slowly enough that the 250 ms throttle opens repeatedly.
        for (int i = 0; i < 6; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
            scope.on_token();
        }
        s = reg.snapshot();
        // The counter is throttled (see kLiveIntervalMs), so it can lag the true
        // token count by up to one window -- and since the dashboard only polls at
        // 1 Hz, publishing faster could not make it look any smoother anyway.
        // What matters is that it ADVANCES and stays within one window of truth.
        CHECK(s.live.tokens >= 1 && s.live.tokens <= 6,
              "live token count advances and stays within one throttle window");
        CHECK(s.live.elapsed_ms > 0.0, "live elapsed time advances");
        CHECK(s.live.tok_per_s > 0.0, "live decode rate is non-zero mid-request");
        CHECK(s.live.itl_ms > 0.0, "live inter-token latency is reported");
        // Still active, and the lifetime counters must NOT have moved yet.
        CHECK(s.live.active, "still marked active mid-request");
        CHECK(s.n_requests == 0, "an unfinished request is not yet counted as complete");

        GenerationResult r;
        r.num_prompt_tokens = 57;
        r.num_generated_tokens = 6;
        r.prefill_ms = 20.0;
        r.decode_ms = 400.0;
        scope.finish(r, RequestStatus::Ok);
    }

    MetricsSnapshot s = reg.snapshot();
    CHECK(!s.live.active, "live block cleared once the request finishes");
    CHECK(s.live.tokens == 0, "live token count reset on completion");
    CHECK(s.engine.active_requests == 0, "active_requests back to 0");
    CHECK(s.n_requests == 1, "the request is now counted once");
    CHECK(s.requests.size() == 1, "one row in the table");
}

static void test_live_concurrency() {
    std::printf("\n--- live concurrency ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();

    {
        MetricsRegistry::RequestScope a(true, 1);
        CHECK(reg.snapshot().engine.active_requests == 1, "one scope -> 1 active");
        {
            MetricsRegistry::RequestScope b(false, 2);
            MetricsSnapshot s = reg.snapshot();
            CHECK(s.engine.active_requests == 2, "two overlapping scopes -> 2 active");
            // The registry owns this count; the gauges' own copy is overwritten.
            CHECK(s.live.session_hash == 2, "the newest scope owns the live block");
        }
        CHECK(reg.snapshot().engine.active_requests == 1,
              "inner scope ending drops the count back to 1");
    }
    CHECK(reg.snapshot().engine.active_requests == 0, "count returns to 0");

    // sample_now() must not crash or block when nothing is in flight.
    reg.sample_now();
    CHECK(reg.snapshot().series_t_ms.size() == 1, "sampling while idle still advances the series");
}

// ---------------------------------------------------------------------------
// Ring bounds
// ---------------------------------------------------------------------------
static void test_ring_bounds() {
    std::printf("\n--- ring bounds ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();

    // 700 rejections: the table is capped at 512, the series at 900 (so the
    // series is not the binding limit here). What matters is that the row
    // count stops growing instead of tracking the request count.
    const int64_t t0 = now_wall_ms();
    for (int i = 0; i < 700; ++i) {
        reg.record_rejected(t0 + i, i, RequestStatus::Rejected, false, 0);
    }

    const MetricsSnapshot s = reg.snapshot();
    CHECK(s.requests.size() == 512, "request table is capped at 512 rows");
    CHECK(s.n_requests == 700, "the lifetime counter is NOT capped");
    // Newest last, and the tail must be the most recent ids we fed in.
    CHECK(s.requests.back().start_ms == t0 + 699, "the newest request is last");
    CHECK(s.requests.front().start_ms == t0 + (700 - 512), "the oldest retained row is correct");
    // Rejections never reached inference, so they contribute no samples.
    CHECK(s.ttft_samples_ms.empty(), "a rejected request feeds no TTFT reservoir");

    // The series ring is now driven by sample_now(). Drive it well past the cap.
    for (int i = 0; i < 1200; ++i)
        reg.sample_now();
    const MetricsSnapshot sampled = reg.snapshot();
    CHECK(sampled.series_t_ms.size() == 900, "series ring is capped at 900 points");
    CHECK(sampled.series_tok_per_s.size() == 900, "all series share the same length");
    CHECK(sampled.series_t_ms.back() >= sampled.series_t_ms.front(),
          "series timestamps are monotonic");
    CHECK(sampled.series_vram_used.back() >= 0, "host gauges land in the series");

    reg.reset();
    const MetricsSnapshot after = reg.snapshot();
    CHECK(after.n_requests == 0 && after.requests.empty(), "reset() clears everything");
    CHECK(after.operators_available == false, "reset() clears the operator snapshot");
}

// ---------------------------------------------------------------------------
// Gauges
// ---------------------------------------------------------------------------
static void test_gauges() {
    std::printf("\n--- gauges ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();

    EngineGauges g;
    g.kv_active_bytes = 1234;
    g.kv_total_bytes = 5678;
    g.kv_free_pages = -1;  // contiguous backend has no page pool
    g.sessions = 2;
    g.active_requests = 1;
    g.prefix_cache_hits = 7;
    g.prefix_cache_misses = 3;
    reg.publish_engine_gauges(g);

    HostGauges h;
    h.vram_total_bytes = 8ull * 1024 * 1024 * 1024;
    h.vram_used_bytes = 2ull * 1024 * 1024 * 1024;
    h.rss_bytes = read_rss_bytes();
    reg.publish_host_gauges(h);

    const MetricsSnapshot s = reg.snapshot();
    CHECK(s.engine.kv_active_bytes == 1234, "engine gauge published");
    CHECK(s.engine.kv_free_pages == -1, "contiguous kv_free_pages sentinel survives");
    CHECK(s.engine.sessions == 2, "session count published");
    CHECK(s.host.vram_used_bytes == 2ull * 1024 * 1024 * 1024, "host gauge published");
    CHECK(s.host.rss_bytes > 0, "RSS is readable on this platform");
    CHECK(reg.engine_gauges().sessions == 2, "engine_gauges() reads the published snapshot");
    CHECK(reg.host_gauges().rss_bytes == s.host.rss_bytes, "host_gauges() round-trips");
}

// ---------------------------------------------------------------------------
// Time helpers
// ---------------------------------------------------------------------------
static void test_time_helpers() {
    std::printf("\n--- helpers ---\n");

    CHECK(now_wall_ms() > 1'600'000'000'000LL, "wall clock is a plausible epoch-ms value");

    const double a = now_mono_ms_f();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const double b = now_mono_ms_f();
    CHECK(b - a >= 25.0, "monotonic clock advances");

    // FNV-1a: stable, and distinct inputs give distinct hashes (the dashboard
    // groups by session using this and nothing else).
    CHECK(fnv1a32("") == 2166136261u, "fnv1a32 of empty string is the offset basis");
    CHECK(fnv1a32("abc") == fnv1a32("abc"), "fnv1a32 is deterministic");
    CHECK(fnv1a32("session-1") != fnv1a32("session-2"), "different ids hash differently");
    CHECK(fnv1a32("session-1") != 0u, "a real id does not hash to 0");

    // CPU utilisation needs two samples; the first establishes the baseline.
    const double c0 = read_cpu_pct(1000);
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    volatile double sink = 0.0;
    for (int i = 0; i < 40'000'000; ++i)
        sink += static_cast<double>(i) * 1e-9;
    const double c1 = read_cpu_pct(1000);
    CHECK(c0 >= 0.0, "first CPU sample is non-negative");
    CHECK(c1 >= 0.0, "CPU sample after work is non-negative");
}

// ---------------------------------------------------------------------------
// Concurrent publisher vs poller (the dashboard thread really is concurrent)
// ---------------------------------------------------------------------------
static void test_concurrent_access() {
    std::printf("\n--- concurrent access ---\n");

    auto& reg = MetricsRegistry::instance();
    reg.set_enabled(true);
    reg.reset();

    std::atomic<bool> stop{false};
    std::atomic<int> polls{0};

    // A poller thread hammering snapshot() while the "inference" thread records.
    // Under TSan/ASan this is the test that would catch a missing lock; without
    // them it at least exercises the interleaving and the mutex ordering.
    std::thread poller([&] {
        // do/while, not while: the writer below can finish all 300 iterations
        // before this thread is ever scheduled, which would turn "the poller ran"
        // into a scheduling accident rather than a property of the code.
        do {
            const MetricsSnapshot s = reg.snapshot();
            (void)s.requests.size();
            (void)s.series_tok_per_s.size();
            polls.fetch_add(1, std::memory_order_relaxed);
        } while (!stop.load(std::memory_order_relaxed));
    });

    GenerationResult r;
    r.num_prompt_tokens = 5;
    r.num_generated_tokens = 5;
    r.decode_ms = 1.0;
    for (int i = 0; i < 300; ++i) {
        MetricsRegistry::RequestScope scope(i % 2 == 0);
        scope.on_lock_acquired();
        for (int t = 0; t < 3; ++t)
            scope.on_token();
        scope.finish(r, RequestStatus::Ok);

        EngineGauges g;
        g.kv_active_bytes = i;
        g.active_requests = i % 4;
        reg.publish_engine_gauges(g);

        HostGauges h;
        h.vram_used_bytes = i * 1024;
        reg.publish_host_gauges(h);
    }

    stop.store(true, std::memory_order_relaxed);
    poller.join();

    const MetricsSnapshot s = reg.snapshot();
    CHECK(s.n_requests == 300, "all 300 concurrent requests were counted");
    CHECK(polls.load() >= 1, "the poller thread ran at least once");
    CHECK(s.requests.size() == 300, "all 300 rows retained (under the 512 cap)");
    reg.reset();
}

int main() {
    std::printf("=== forge-metrics-test ===\n");
    test_percentile();
    test_disabled_is_free();
    test_request_scope();
    test_backdated_arrival();
    test_scope_destructor();
    test_double_finish();
    test_record_rejected();
    test_abandon();
    test_live_progress();
    test_live_concurrency();
    test_ring_bounds();
    test_gauges();
    test_time_helpers();
    test_concurrent_access();
    std::printf("\n=== %d/%d checks passed ===\n", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}