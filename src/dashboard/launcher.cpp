/**
 * launcher.cpp - spawn / stop / observe a child forge-server from the dashboard.
 *
 * The dashboard already ships inside forge-server, so the child is launched from
 * the *current* executable (resolved via /proc/self/exe, _NSGetExecutablePath,
 * or GetModuleFileName). Launching a child rather than re-using the parent is
 * deliberate: each chat session wants its own model + port + KV cache, and a
 * fresh process is the cleanest isolation boundary -- stop it and everything it
 * allocated is released.
 *
 * Child output goes to a log file (temp dir) that status() tails, so the UI can
 * show "Loading model..." / "listening at ..." without a streaming pipe.
 */

#include "launcher.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <signal.h>
#    include <sys/wait.h>
#    include <unistd.h>
#endif

#ifdef __APPLE__
#    include <mach-o/dyld.h>
#endif

#if __cplusplus >= 201703L
#    include <filesystem>
#endif

#include <thread>

namespace forge {

// ---------------------------------------------------------------------------
// executable path
// ---------------------------------------------------------------------------

std::string ProcessLauncher::current_executable_path() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH] = {0};
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0) return {};
    // Convert wide to UTF-8 narrow.
    int need = WideCharToMultiByte(CP_UTF8, 0, buf, -1, nullptr, 0, nullptr, nullptr);
    if (need <= 0) return {};
    std::string out(static_cast<size_t>(need) - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, buf, -1, &out[0], need, nullptr, nullptr);
    return out;
#elif defined(__APPLE__)
    char buf[4096] = {0};
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) != 0) return {};  // buffer too small; ignore
    // _NSGetExecutablePath may be relative; resolve.
    if (char* resolved = realpath(buf, nullptr)) {
        std::string s(resolved);
        free(resolved);
        return s;
    }
    return std::string(buf);
#else  // Linux / generic POSIX
    char buf[4096] = {0};
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) return std::string(buf, static_cast<size_t>(n));
    return {};
#endif
}

// ---------------------------------------------------------------------------
// log file
// ---------------------------------------------------------------------------

static std::string make_log_path() {
#if __cplusplus >= 201703L
    std::error_code ec;
    const auto tmp = std::filesystem::temp_directory_path(ec);
    const std::string dir = ec ? "/tmp" : tmp.string();
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    return dir + "/forge-launch-" + std::to_string(ms) + ".log";
#else
    return "/tmp/forge-launch.log";
#endif
}

// ---------------------------------------------------------------------------
// ctor / dtor
// ---------------------------------------------------------------------------

ProcessLauncher::ProcessLauncher() = default;

ProcessLauncher::~ProcessLauncher() {
    stop_monitor_.store(true);
    if (monitor_.joinable())
        monitor_.join();
    // Leave the child running across dashboard restarts? No -- the dashboard owns
    // the chat server lifecycle, so tear it down on destruction.
    stop();
}

// ---------------------------------------------------------------------------
// spawn
// ---------------------------------------------------------------------------

bool ProcessLauncher::start(const LaunchConfig& cfg, std::string& error) {
    // Join any previous monitor thread *before* taking the lock: a prior child
    // may have exited and its monitor is about to set running_=false, and joining
    // while holding mu_ would deadlock against that final lock acquisition.
    if (monitor_.joinable())
        monitor_.join();

    std::lock_guard<std::mutex> lock(mu_);
    if (running_) {
        error = "a server is already running (stop it first)";
        return false;
    }
    if (cfg.model.empty()) {
        error = "model path (-m) is required";
        return false;
    }

    const std::string exe = current_executable_path();
    if (exe.empty()) {
        error = "could not resolve the running executable path";
        return false;
    }

    // Build the argv. Only emit flags whose value is "set" so defaults stay the
    // server's own defaults (e.g. -ngl auto, -c native).
    std::vector<std::string> args;
    args.push_back(exe);
    args.push_back("-m");
    args.push_back(cfg.model);
    if (cfg.ngl >= 0) {
        args.push_back("-ngl");
        args.push_back(std::to_string(cfg.ngl));
    }
    if (cfg.ctx_size > 0) {
        args.push_back("-c");
        args.push_back(std::to_string(cfg.ctx_size));
    }
    if (cfg.threads > 0) {
        args.push_back("-t");
        args.push_back(std::to_string(cfg.threads));
    }
    args.push_back("--host");
    args.push_back(cfg.host);
    args.push_back("--port");
    args.push_back(std::to_string(cfg.port));
    if (!cfg.api_key.empty()) {
        args.push_back("--api-key");
        args.push_back(cfg.api_key);
    }
    if (!cfg.mmproj.empty()) {
        args.push_back("--mmproj");
        args.push_back(cfg.mmproj);
    }
    for (const auto& e : cfg.extra) {
        if (!e.empty())
            args.push_back(e);
    }

    log_path_ = make_log_path();
    cfg_ = cfg;
    spawn_error_.clear();
    started_at_.clear();

    // Timestamp for the UI.
    {
        const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char ts[64] = {0};
        std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        started_at_ = ts;
    }

#ifdef _WIN32
    // Open the log file for the child to inherit.
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE logh = CreateFileA(log_path_.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, &sa,
                              CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (logh == INVALID_HANDLE_VALUE) {
        error = "could not open launcher log file";
        return false;
    }

    // Build a command line (CreateProcess wants a single string).
    std::ostringstream cmd;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i) cmd << ' ';
        // Quote tokens that contain spaces.
        if (args[i].find(' ') != std::string::npos)
            cmd << '"' << args[i] << '"';
        else
            cmd << args[i];
    }

    STARTUPINFOA si{sizeof(si)};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = logh;
    si.hStdError = logh;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(exe.c_str(), const_cast<char*>(cmd.str().c_str()), nullptr, nullptr, TRUE, 0,
                       nullptr, nullptr, &si, &pi)) {
        CloseHandle(logh);
        error = "CreateProcess failed (code " + std::to_string(GetLastError()) + ")";
        return false;
    }
    CloseHandle(logh);  // child has its own handle
    CloseHandle(pi.hThread);
    proc_handle_ = pi.hProcess;
    pid_ = static_cast<int>(pi.dwProcessId);
#else
    // POSIX: fork + exec, redirect stdout/stderr to the log file.
    const int logfd = ::open(log_path_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (logfd < 0) {
        error = "could not open launcher log file";
        return false;
    }

    // argv as char* array (null terminated).
    std::vector<char*> argv(args.size() + 1, nullptr);
    for (size_t i = 0; i < args.size(); ++i)
        argv[i] = const_cast<char*>(args[i].c_str());

    const pid_t pid = fork();
    if (pid < 0) {
        ::close(logfd);
        error = "fork failed";
        return false;
    }
    if (pid == 0) {
        // Child: detach from the controlling terminal's stdio and point
        // stdout/stderr at the log file.
        dup2(logfd, 1);
        dup2(logfd, 2);
        if (logfd > 2) ::close(logfd);
        // Best-effort: don't let the child inherit the dashboard's listen fds
        // beyond what it needs. setsid to detach from the terminal.
        setsid();
        execv(exe.c_str(), argv.data());
        // If execv returns, it failed.
        _exit(127);
    }
    ::close(logfd);
    pid_ = static_cast<int>(pid);
#endif

    running_ = true;
    stop_monitor_.store(false);
    monitor_ = std::thread([this] { reap_loop(); });
    return true;
}

// ---------------------------------------------------------------------------
// monitor: wait for the child to exit, then mark not-running
// ---------------------------------------------------------------------------

void ProcessLauncher::reap_loop() {
#ifdef _WIN32
    if (proc_handle_) {
        WaitForSingleObject(proc_handle_, INFINITE);
        CloseHandle(proc_handle_);
        proc_handle_ = nullptr;
    }
#else
    int status = 0;
    int ret = 0;
    do {
        ret = waitpid(pid_, &status, 0);
    } while (ret < 0 && errno == EINTR && !stop_monitor_.load());
#endif
    std::lock_guard<std::mutex> lock(mu_);
    running_ = false;
    pid_ = 0;
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------

void ProcessLauncher::stop() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!running_)
        return;
#ifdef _WIN32
    if (proc_handle_) {
        TerminateProcess(proc_handle_, 0);
        CloseHandle(proc_handle_);
        proc_handle_ = nullptr;
    }
#else
    if (pid_ > 0) {
        ::kill(pid_, SIGTERM);
        // Give it a moment to exit gracefully; then force-kill.
        for (int i = 0; i < 25 && waitpid(pid_, nullptr, WNOHANG) == 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (waitpid(pid_, nullptr, WNOHANG) == 0)
            ::kill(pid_, SIGKILL);
    }
#endif
    running_ = false;
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

void ProcessLauncher::read_logs(std::string& out) const {
    std::ifstream f(log_path_, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        return;
    const std::streamoff size = f.tellg();
    if (size <= 0)
        return;
    // Tail: last 16 KiB.
    const std::streamoff want = 16384;
    const std::streamoff start = size > want ? size - want : 0;
    f.seekg(start, std::ios::beg);
    std::string buf(static_cast<size_t>(size - start), '\0');
    f.read(&buf[0], static_cast<std::streamsize>(buf.size()));
    out = std::move(buf);
}

LaunchStatus ProcessLauncher::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    LaunchStatus s;
    s.running = running_;
    s.pid = pid_;
    s.port = cfg_.port;
    s.host = cfg_.host;
    s.model = cfg_.model;
    s.started_at = started_at_;
    s.log_path = log_path_;
    s.error = spawn_error_;
    if (running_)
        read_logs(s.logs);
    return s;
}

}  // namespace forge
