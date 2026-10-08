/**
 * dashboard_server.cpp - loopback HTTP listener + JSON for the web dashboard.
 *
 * Routes:
 *   GET /                 -> 302 /dashboard/
 *   GET /dashboard/       -> embedded index.html
 *   GET /dashboard/<path> -> embedded asset (exact match against the table)
 *   GET /api/metrics      -> JSON snapshot of MetricsRegistry
 *   GET /api/health       -> {"status":"ok"}
 *
 * Polling, not SSE. cpp-httplib 0.51 has no set_sse_handler, and its default
 * worker pool is 8 threads: one long-lived streaming response per browser tab
 * would starve the API server's own workers, which is a much worse failure than
 * a one-second refresh interval.
 */

#include <httplib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>

#include "forge/backend.h"
#include "forge/dashboard.h"
#include "dashboard/launcher.h"

#ifdef _WIN32
#    include <io.h>
#    include <process.h>
#else
#    include <unistd.h>
#endif

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

namespace {

using json = nlohmann::json;

// Sample interval for the host gauges. 1 Hz matches the frontend poll and is
// slow enough that cudaMemGetInfo does not show up in a profile.
constexpr int kGaugeIntervalMs = 1000;

// Read every CUDA device's memory and sum them: a model sharded across GPUs
// (-ngl N:M:K) is only meaningful in aggregate. cudaMemGetInfo is independent of
// engine state, so calling it from this thread while inference runs on another
// is safe -- unlike the KV counters, which are published by the inference
// thread precisely because they are NOT safe to read here.
void sample_vram(int64_t& total, int64_t& used) {
    total = 0;
    used = 0;
#ifdef USE_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        return;
    for (int d = 0; d < count; ++d) {
        size_t free_b = 0, total_b = 0;
        if (cudaSetDevice(d) != cudaSuccess)
            continue;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
            continue;
        total += static_cast<int64_t>(total_b);
        used += static_cast<int64_t>(total_b - free_b);
    }
#endif
}

const char* status_name(RequestStatus s) {
    switch (s) {
    case RequestStatus::Ok:
        return "ok";
    case RequestStatus::Error:
        return "error";
    case RequestStatus::Aborted:
        return "aborted";
    case RequestStatus::Rejected:
        return "rejected";
    case RequestStatus::BadRequest:
        return "bad_request";
    }
    return "unknown";
}

json pct(const std::vector<float>& sorted, double p) {
    if (sorted.empty())
        return json{{"value", 0.0}, {"count", 0}};
    return json{{"value", percentile_sorted(sorted, p)}, {"count", sorted.size()}};
}

// Scoped so an operator list with 200 entries does not bloat every 1 Hz poll.
constexpr size_t kMaxOperatorsInJson = 24;

json operator_to_json(const OperatorRecord& op) {
    return json{{"name", op.name},     {"total_ms", op.total_ms}, {"count", op.count},
                {"min_ms", op.min_ms}, {"max_ms", op.max_ms},     {"avg_ms", op.avg_ms}};
}

}  // namespace

std::string DashboardServer::snapshot_to_json(const MetricsSnapshot& s) {
    json j;
    j["meta"] = json{{"app", s.app},
                     {"model", s.model_id},
                     {"arch", s.arch},
                     {"start_ms", s.start_ms},
                     {"now_ms", s.now_ms},
                     {"uptime_s", static_cast<double>(s.now_ms - s.start_ms) / 1000.0},
                     {"profiling", s.profiling_enabled}};

    j["engine"] = json{{"kv_active_bytes", s.engine.kv_active_bytes},
                       {"kv_total_bytes", s.engine.kv_total_bytes},
                       {"kv_free_pages", s.engine.kv_free_pages},
                       {"kv_pool_tokens", s.engine.kv_pool_tokens},
                       {"kv_page_size", s.engine.kv_page_size},
                       {"sessions", s.engine.sessions},
                       {"active_requests", s.engine.active_requests},
                       {"prefix_cache_hits", s.engine.prefix_cache_hits},
                       {"prefix_cache_misses", s.engine.prefix_cache_misses},
                       {"tiered_kv", s.engine.tiered_kv},
                       {"paged_kv", s.engine.paged_kv}};

    // Live progress for the request currently in flight. This block is what makes
    // the dashboard animate during a generation; without it every chart sat flat
    // until the request completed.
    j["live"] = json{{"active", s.live.active},
                     {"tokens", s.live.tokens},
                     {"prompt_tokens", s.live.prompt_tokens},
                     {"elapsed_ms", s.live.elapsed_ms},
                     {"tok_per_s", s.live.tok_per_s},
                     {"itl_ms", s.live.itl_ms},
                     {"start_ms", s.live.start_ms},
                     {"streaming", s.live.streaming},
                     {"session", s.live.session_hash}};

    j["host"] = json{{"vram_total_bytes", s.host.vram_total_bytes},
                     {"vram_used_bytes", s.host.vram_used_bytes},
                     {"rss_bytes", s.host.rss_bytes},
                     {"cpu_pct", s.host.cpu_pct}};

    const uint64_t prefix_total =
        static_cast<uint64_t>(s.engine.prefix_cache_hits + s.engine.prefix_cache_misses);
    const double prefix_rate = prefix_total > 0 ? static_cast<double>(s.engine.prefix_cache_hits) /
                                                      static_cast<double>(prefix_total)
                                                : 0.0;
    // Lifetime decode throughput, straight off the accumulated decode time. The
    // per-request tok/s in `requests` is the one to read for a single request;
    // this is the honest "how fast has this process ever decoded" number.
    const double lifetime_tps =
        s.total_decode_ms > 0.0
            ? static_cast<double>(s.total_completion_tokens) / (s.total_decode_ms / 1000.0)
            : 0.0;

    j["totals"] = json{{"requests", s.n_requests},
                       {"errors", s.n_errors},
                       {"rejected", s.n_rejected},
                       {"prompt_tokens", s.total_prompt_tokens},
                       {"completion_tokens", s.total_completion_tokens},
                       {"prefill_ms", s.total_prefill_ms},
                       {"decode_ms", s.total_decode_ms},
                       {"queue_ms", static_cast<double>(s.total_queue_ms)},
                       {"lifetime_tok_per_s", lifetime_tps},
                       {"prefix_hit_rate", prefix_rate}};

    j["latency"] =
        json{{"ttft_p50", pct(s.ttft_samples_ms, 0.50)}, {"ttft_p90", pct(s.ttft_samples_ms, 0.90)},
             {"ttft_p99", pct(s.ttft_samples_ms, 0.99)}, {"itl_p50", pct(s.itl_samples_ms, 0.50)},
             {"itl_p90", pct(s.itl_samples_ms, 0.90)},   {"itl_p99", pct(s.itl_samples_ms, 0.99)}};

    j["series"] = json{{"t", s.series_t_ms},
                       {"tok_per_s", s.series_tok_per_s},
                       {"ttft_ms", s.series_ttft_ms},
                       {"itl_p50_ms", s.series_itl_p50_ms},
                       {"vram_used", s.series_vram_used},
                       {"kv_active", s.series_kv_active}};

    json reqs = json::array();
    for (auto it = s.requests.rbegin(); it != s.requests.rend(); ++it) {
        const RequestRecord& r = *it;
        reqs.push_back(json{{"start_ms", r.start_ms},
                            {"end_ms", r.end_ms},
                            {"queue_ms", r.queue_ms},
                            {"ttft_ms", r.has_ttft ? json(r.ttft_ms) : json(nullptr)},
                            {"prefill_ms", r.prefill_ms},
                            {"decode_ms", r.decode_ms},
                            {"total_ms", r.total_ms},
                            {"prompt_tokens", r.prompt_tokens},
                            {"completion_tokens", r.completion_tokens},
                            {"prefix_reused", r.prefix_reused},
                            {"tok_per_s", r.tok_per_s},
                            {"itl_p50_ms", r.itl_p50_ms},
                            {"itl_p90_ms", r.itl_p90_ms},
                            {"spec_accept", r.spec_accept},
                            {"status", status_name(r.status)},
                            {"streaming", r.streaming},
                            {"session", r.session_hash}});
    }
    j["requests"] = std::move(reqs);

    json ops = json::array();
    if (s.operators_available) {
        // Operators arrive already sorted by cost (the caller sorts before
        // handing them over) -- keep the top N.
        for (size_t i = 0; i < s.operators.size() && i < kMaxOperatorsInJson; ++i)
            ops.push_back(operator_to_json(s.operators[i]));
    }
    j["operators"] = std::move(ops);
    j["operators_available"] = s.operators_available;

    return j.dump();
}

struct DashboardServer::Impl {
    httplib::Server svr;
    forge::ProcessLauncher launcher;  // start/stop/status a child forge-server
};

DashboardServer::~DashboardServer() {
    stop();
}

std::string DashboardServer::url() const {
    if (port_ <= 0)
        return std::string();
    std::ostringstream oss;
    oss << "http://" << (host_.empty() ? "127.0.0.1" : host_) << ":" << port_ << "/dashboard/";
    return oss.str();
}

namespace {

// Serving the "not built with npm" page instead of 404: the dashboard HTML is
// missing but /api/metrics still works, and a bare 404 would look like the
// server is broken rather than the frontend simply was not compiled in.
constexpr const char* kNoAssetsBody =
    R"(<!doctype html><meta charset="utf-8"><title>Forge dashboard</title>
<style>
/* Dark is the default, so it goes first as the plain declarations; the light
   overrides come after in a media query. Order matters here: an equal-specificity
   :root rule placed after the media block would win over it and leave
   color-scheme dark on a light machine, which is the thing being fixed.
   Mirrors the two theme blocks in web/src/style.css. */
:root{color-scheme:dark}
body{background:#0f1115;color:#e6e6e6;font:14px/1.6 ui-monospace,monospace;margin:0;padding:48px}
code{background:#1b1f27;padding:2px 6px;border-radius:4px}
a{color:#4da3ff}
@media(prefers-color-scheme:light){
  :root{color-scheme:light}
  body{background:#f4f6f9;color:#1b1f24}
  code{background:#eef1f5}
  a{color:#1668c4}
}
</style>
<h1>Dashboard assets not embedded</h1>
<p>This binary was built without the web frontend. The JSON API still works at
<a href="/api/metrics"><code>/api/metrics</code></a>.</p>
<p>Rebuild with Node.js 18+ installed:</p>
<pre><code>cmake -B build -DFORGE_BUILD_DASHBOARD=ON
cmake --build build -j</code></pre>)";

// Cache policy for the embedded frontend.
//
// The asset filenames are content-hashed by Vite (index-XXXXXXXX.js), which is
// what makes the immutable policy below safe AND what makes the no-store policy
// on the shell necessary:
//
//   * Hashed assets may be cached forever -- a changed byte of content changes
//     the name, so a cached copy can never be stale.
//   * index.html MUST NOT be cached. It is the only thing that maps hashed names
//     back to files, and it is served without a validator, so a browser falls
//     back to HEURISTIC caching -- typically keeping it for a fraction of its
//     Last-Modified age, with no expiry at all here since there is no
//     Last-Modified. A stale shell then requests chunk names that this binary no
//     longer contains, they 404, and the dashboard renders blank (or the user
//     sees a console full of module-resolution errors).
//
// That is not hypothetical: it is exactly the "the web page does not show up"
// report this replaced. Set the headers explicitly rather than relying on
// whatever httplib or the browser defaults to.
constexpr const char* kShellCacheControl = "no-store, no-cache, must-revalidate";
constexpr const char* kAssetCacheControl = "public, max-age=31536000, immutable";

void send_asset(httplib::Response& res, const std::string& path) {
    if (path == "/dashboard" || path == "/dashboard/") {
        res.set_redirect("/dashboard/index.html", 302);
        return;
    }
    static const std::string kPrefix = "/dashboard/";
    if (path.compare(0, kPrefix.size(), kPrefix) != 0) {
        res.status = 404;
        res.set_content("not found", "text/plain");
        return;
    }
    // Match the remainder against the table with exact string equality, so ".."
    // or an absolute path can never escape the table.
    const std::string rel = path.substr(kPrefix.size());
    const DashboardAsset* assets = dashboard_assets();
    const size_t n = dashboard_asset_count();
    for (size_t i = 0; i < n; ++i) {
        if (rel == assets[i].path) {
            const bool is_shell = rel == "index.html";
            res.set_header("Cache-Control", is_shell ? kShellCacheControl : kAssetCacheControl);
            if (is_shell)
                res.set_header("Pragma", "no-cache");  // HTTP/1.0 holdouts
            res.set_content(assets[i].body, assets[i].size, assets[i].mime);
            return;
        }
    }
    if (n == 0) {
        // The "not embedded" page is generated per request and depends on how
        // the binary was built, so it must not be cached either.
        res.set_header("Cache-Control", kShellCacheControl);
        res.set_content(kNoAssetsBody, "text/html; charset=utf-8");
        return;
    }
    res.status = 404;
    res.set_content("asset not found", "text/plain");
}

}  // namespace

bool DashboardServer::start(const DashboardArgs& args, std::string& error) {
    if (running_.load(std::memory_order_acquire))
        return true;
    args_ = args;
    host_ = args.host.empty() ? "127.0.0.1" : args.host;
    stopping_.store(false, std::memory_order_release);

    impl_ = std::make_shared<Impl>();

    impl_->svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    impl_->svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_redirect("/dashboard/index.html", 302);
    });
    impl_->svr.Get("/dashboard", [](const httplib::Request& req, httplib::Response& res) {
        send_asset(res, req.path);
    });
    impl_->svr.Get("/dashboard/", [](const httplib::Request& req, httplib::Response& res) {
        send_asset(res, req.path);
    });
    // Explicit handler rather than set_mount_point: the assets are in memory,
    // and mount_point would also swallow every unmatched /dashboard/* path.
    impl_->svr.Get(R"(/dashboard/.*)", [](const httplib::Request& req, httplib::Response& res) {
        send_asset(res, req.path);
    });
    impl_->svr.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    impl_->svr.Get("/api/metrics", [](const httplib::Request&, httplib::Response& res) {
        // Refresh the host gauges on read as well as on the sampler tick, so a
        // freshly opened page is never blank for up to a second.
        HostGauges h = MetricsRegistry::instance().host_gauges();
        sample_vram(h.vram_total_bytes, h.vram_used_bytes);
        h.rss_bytes = read_rss_bytes();
        h.cpu_pct = read_cpu_pct(kGaugeIntervalMs);
        MetricsRegistry::instance().publish_host_gauges(h);
        res.set_content(DashboardServer::snapshot_to_json(MetricsRegistry::instance().snapshot()),
                        "application/json");
    });

    // ---- Chat launcher: start / stop / status a child forge-server ----------
    // The dashboard lets the browser launch a model-serving child (with the same
    // CLI the server binary takes) so the Chat page can talk to a real server
    // without the operator dropping to a terminal. The child runs on its own API
    // port; this dashboard stays on its own loopback port.
    impl_->svr.Post("/api/launch", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"invalid JSON body"})", "application/json");
            return;
        }
        forge::LaunchConfig cfg;
        cfg.model = body.value("model", std::string());
        cfg.ngl = body.value("ngl", -1);
        cfg.ctx_size = body.value("ctx", 0);
        cfg.threads = body.value("threads", 0);
        cfg.host = body.value("host", std::string("127.0.0.1"));
        cfg.port = body.value("port", 8080);
        cfg.api_key = body.value("apiKey", std::string());
        cfg.mmproj = body.value("mmproj", std::string());
        if (body.contains("extra") && body["extra"].is_array()) {
            for (const auto& e : body["extra"]) {
                if (e.is_string())
                    cfg.extra.push_back(e.get<std::string>());
            }
        }
        std::string err;
        const bool ok = impl_->launcher.start(cfg, err);
        res.set_content(json{{"ok", ok}, {"error", err}, {"port", cfg.port}}.dump(),
                        "application/json");
    });
    impl_->svr.Post("/api/launch/stop", [&](const httplib::Request&, httplib::Response& res) {
        impl_->launcher.stop();
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });
    impl_->svr.Get("/api/launch", [&](const httplib::Request&, httplib::Response& res) {
        const forge::LaunchStatus s = impl_->launcher.status();
        res.set_content(json{{"running", s.running},
                            {"pid", s.pid},
                            {"host", s.host},
                            {"port", s.port},
                            {"model", s.model},
                            {"started_at", s.started_at},
                            {"logs", s.logs},
                            {"error", s.error}}
                              .dump(),
                        "application/json");
    });

    // Bind before announcing, so the banner and the browser both get the port
    // that actually worked. A requested port is a preference: 9090 is commonly
    // taken, and failing the whole run over it is worse than landing on 9091.
    int bound = 0;
    bool ok = false;
    if (args_.port > 0) {
        for (int attempt = 0; attempt < 10 && !ok; ++attempt) {
            const int want = args_.port + attempt;
            if (want > 65535)
                break;
            if (impl_->svr.bind_to_port(host_, want)) {
                bound = want;
                ok = true;
                break;
            }
        }
        if (!ok) {
            error = "could not bind " + host_ + ": any port in [" + std::to_string(args_.port) +
                    ", " + std::to_string(args_.port + 9) + "]";
            return false;
        }
    } else {
        bound = impl_->svr.bind_to_any_port(host_);
        ok = bound > 0;
        if (!ok) {
            error = "could not bind an ephemeral port on " + host_;
            return false;
        }
    }
    port_ = bound;

    thread_ = std::thread([this] {
        impl_->svr.listen_after_bind();
        running_.store(false, std::memory_order_release);
    });
    running_.store(true, std::memory_order_release);

    // Sample once up front rather than waiting for the first 1 s tick. A request
    // that completes inside the first second would otherwise append a series
    // point with vram_used = 0, drawing a visible dip to zero at the left edge of
    // every memory chart.
    {
        HostGauges h;
        sample_vram(h.vram_total_bytes, h.vram_used_bytes);
        h.rss_bytes = read_rss_bytes();
        // The first read_cpu_pct call only establishes its delta baseline;
        // calling it here anchors that baseline at start-up.
        h.cpu_pct = read_cpu_pct(kGaugeIntervalMs);
        MetricsRegistry::instance().publish_host_gauges(h);
    }

    // The gauge sampler is separate from the HTTP worker threads so a slow
    // client cannot delay the numbers, and vice versa.
    std::thread([this] { sampler_loop(); }).detach();

    if (args_.open_browser) {
        const std::string u = url();
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", u.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif __APPLE__
        if (std::system(("open '" + u + "' >/dev/null 2>&1 &").c_str()) != 0) { /* non-fatal */
        }
#else
        if (std::system(("xdg-open '" + u + "' >/dev/null 2>&1 &").c_str()) != 0) { /* non-fatal */
        }
#endif
    }
    return true;
}

void DashboardServer::sampler_loop() {
    while (!stopping_.load(std::memory_order_acquire)) {
        HostGauges h = MetricsRegistry::instance().host_gauges();
        sample_vram(h.vram_total_bytes, h.vram_used_bytes);
        h.rss_bytes = read_rss_bytes();
        h.cpu_pct = read_cpu_pct(kGaugeIntervalMs);
        MetricsRegistry::instance().publish_host_gauges(h);
        // Advance the charts on every tick. This is what makes the dashboard
        // animate DURING a generation rather than jumping once at the end --
        // the series used to only gain a point when a request completed, so a
        // long decode left every chart flat for its whole duration.
        MetricsRegistry::instance().sample_now();
        for (int slept = 0; slept < kGaugeIntervalMs && !stopping_.load(std::memory_order_acquire);
             slept += 50) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

void DashboardServer::stop() {
    if (!stopping_.exchange(true)) {
        if (impl_ && running_.load(std::memory_order_acquire))
            impl_->svr.stop();
    }
    if (thread_.joinable())
        thread_.join();
    running_.store(false, std::memory_order_release);
    // Drop the listener only after the accept thread has been joined; the
    // running_ flag above is set by that thread as it unwinds.
    impl_.reset();
}

#ifndef FORGE_DASHBOARD_EMBEDDED
// Built without the npm frontend (FORGE_BUILD_DASHBOARD=OFF, or ON on a machine
// without Node.js). dashboard_server.cpp answers /api/metrics normally and
// /dashboard/ renders the "assets not embedded" page from the empty table.
size_t dashboard_asset_count() {
    return 0;
}
const DashboardAsset* dashboard_assets() {
    return nullptr;
}
// The chat UI follows the dashboard: no npm build means no embedded assets,
// and /chat/ answers 404 with a hint (see serve_chat_asset in server_routes.cpp).
size_t chat_asset_count() {
    return 0;
}
const DashboardAsset* chat_assets() {
    return nullptr;
}
#endif

}  // namespace forge