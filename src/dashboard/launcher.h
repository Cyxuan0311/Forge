#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace forge {

// Configuration for launching a child forge-server from the dashboard. The
// launcher turns the UI's form fields into the exact CLI the server binary
// accepts (-m, -ngl, -c, -t, --port, --host, --api-key, --mmproj, plus any
// raw extra tokens the user types).
struct LaunchConfig {
    std::string model;               // -m (required)
    int ngl = -1;                    // -ngl   (-1 => omit, i.e. auto)
    int ctx_size = 0;                // -c     (0  => omit, native ctx)
    int threads = 0;                 // -t     (0  => omit, auto)
    std::string host = "127.0.0.1";  // --host
    int port = 8080;                 // --port
    std::string api_key;             // --api-key (optional)
    std::string mmproj;              // --mmproj (optional, vision)
    std::vector<std::string> extra;  // raw tokens appended verbatim
};

// A point-in-time snapshot of the launcher, including a tail of the child's
// stdout/stderr so the UI can show server startup progress.
struct LaunchStatus {
    bool running = false;
    int pid = 0;
    std::string host;
    int port = 0;
    std::string model;
    std::string started_at;  // local time of launch (empty if never started)
    std::string log_path;    // where child output is captured
    std::string logs;        // recent tail of child output
    std::string error;       // last spawn/stop error (if any)
};

// Owns a single child forge-server process launched from the dashboard.
//
// Cross-platform: POSIX fork/exec + SIGTERM/SIGKILL, Windows CreateProcess +
// TerminateProcess. The child's stdout/stderr are redirected to a log file the
// dashboard reads for the "logs" tail -- simpler and more robust than a
// streaming pipe, and it survives the browser polling the status endpoint.
class ProcessLauncher {
public:
    ProcessLauncher();
    ~ProcessLauncher();

    ProcessLauncher(const ProcessLauncher&) = delete;
    ProcessLauncher& operator=(const ProcessLauncher&) = delete;

    // Launch a child forge-server. Returns false (filling `error`) if already
    // running, the executable cannot be resolved, or spawn fails.
    bool start(const LaunchConfig& cfg, std::string& error);

    // Stop the child (graceful SIGTERM/SIGKILL, or TerminateProcess). No-op if
    // not running.
    void stop();

    // Snapshot of current state with a tail of recent child output.
    LaunchStatus status() const;

    // Resolve the absolute path of the executable that is currently running.
    static std::string current_executable_path();

private:
    void reap_loop();
    void read_logs(std::string& out) const;

    mutable std::mutex mu_;
    bool running_ = false;
    int pid_ = 0;
    std::string log_path_;
    LaunchConfig cfg_;
    std::string started_at_;
    std::string spawn_error_;

#ifdef _WIN32
    void* proc_handle_ = nullptr;  // HANDLE (avoid pulling <windows.h> into the header)
#else
    int sigkill_grace_ms_ = 2500;
#endif

    std::thread monitor_;
    std::atomic<bool> stop_monitor_{false};
};

}  // namespace forge
