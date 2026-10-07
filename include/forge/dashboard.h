/**
 * dashboard.h - the local web dashboard shared by forge-cli and forge-server.
 *
 * A second, small httplib listener on its own loopback port. Deliberately NOT
 * mounted on the API port: forge-server's pre-routing handler enforces
 * "Authorization: Bearer <key>" for every request, and a browser cannot send
 * that header. Serving the dashboard from the same listener would mean carving
 * an exemption out of the auth path and exposing process metrics to anyone who
 * can reach the API.
 *
 * The frontend is built by npm (web/, Vue 3 + Vite + ECharts) and embedded into
 * the binary by cmake/EmbedDashboardAssets.cmake, because the release tarball
 * ships a bare `forge-cli` with no asset directory next to it. When the build
 * was configured without Node.js the server still starts and still answers
 * /api/metrics; only the HTML is missing and /dashboard/ says so.
 */

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "forge/cli_dashboard_args.h"
#include "forge/metrics.h"

namespace forge {

class DashboardServer {
public:
    DashboardServer() = default;
    ~DashboardServer();

    DashboardServer(const DashboardServer&) = delete;
    DashboardServer& operator=(const DashboardServer&) = delete;

    // Binds, starts the listener thread and starts the host-gauge sampler.
    // Returns false and fills `error` when nothing could be bound.
    bool start(const DashboardArgs& args, std::string& error);

    // Idempotent. Safe to call from a signal handler path indirectly (the
    // callers guard with their own exchange flag, matching the API server).
    void stop();

    // The port actually bound. Differs from the requested one when the request
    // was a preference and the preferred port was busy.
    int port() const { return port_; }
    bool running() const { return running_.load(std::memory_order_acquire); }

    // "http://127.0.0.1:9090/dashboard/" for the startup banner / browser.
    std::string url() const;

    // Serializes a MetricsSnapshot to the JSON the frontend consumes. Exposed
    // for tests; the registry itself stays free of any JSON dependency.
    static std::string snapshot_to_json(const MetricsSnapshot& s);

private:
    void sampler_loop();

    DashboardArgs args_;
    std::string host_;
    int port_ = 0;
    // Opaque so dashboard.h does not drag httplib.h into every includer.
    // shared_ptr, not unique_ptr: a unique_ptr<Impl> with an incomplete Impl
    // forces sizeof(Impl) into every translation unit that merely declares a
    // DashboardServer (for the unwinding path), which defeats the point of the
    // forward declaration. The shared_ptr's deleter is created inside
    // dashboard_server.cpp, where Impl is complete.
    struct Impl;
    std::shared_ptr<Impl> impl_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopping_{false};
};

// Assets embedded at build time. Declared here and defined either by the
// generated source (FORGE_DASHBOARD_EMBEDDED=1) or by the stub below.
struct DashboardAsset {
    const char* path;
    const char* mime;
    const char* body;
    size_t size;
};

// Empty table when the build had no Node.js.
size_t dashboard_asset_count();
const DashboardAsset* dashboard_assets();

}  // namespace forge