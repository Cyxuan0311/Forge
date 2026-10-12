/**
 * Agent module — run_python tool.
 *
 * Executes model-authored Python in a forked child with a wall-clock timeout, a
 * private temp directory as cwd, and rlimits on address space / CPU / file size
 * / process count. stdout and stderr are captured (stdout first, then stderr)
 * and returned as the tool result.
 *
 * SECURITY: this runs arbitrary code the model wrote. The isolation here is
 * deliberately described as *best effort* -- rlimits and a timeout bound runaway
 * resource use, they do NOT sandbox. A hostile or confused model can still read
 * the filesystem and open sockets as the server user. That is why the tool is
 * opt-in (--enable-code-exec / FORGE_ALLOW_CODE_EXEC=1) and reports itself as
 * unavailable otherwise, so a deployment that never enables it is unaffected.
 * Real containment needs a container/gVisor-style boundary, which is out of
 * scope for an in-process tool.
 */

#include "forge/agent/tool_registry.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace forge {

using json = nlohmann::json;

namespace {

// Executed as: python3 _runner.py snippet.py
//
// If the snippet's last top-level statement is a bare expression, wrap it in a
// print() so `f(x)` behaves like it does in a REPL. Nothing else is altered.
constexpr const char* kRunnerSource = R"PYRUNNER(
import ast
import sys
import traceback

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    source = fh.read()

try:
    tree = ast.parse(source, filename=path)
except SyntaxError:
    traceback.print_exc()
    sys.exit(1)

if tree.body and isinstance(tree.body[-1], ast.Expr):
    # Only echo it when nothing was printed on that line already; a bare
    # string literal is almost always intentional dead code, not a value.
    last = tree.body[-1]
    if not (isinstance(last.value, ast.Constant) and isinstance(last.value.value, str)):
        tree.body[-1] = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[last.value],
                keywords=[],
            )
        )

ast.fix_missing_locations(tree)
glb = {"__name__": "__main__", "__file__": path, "__builtins__": __builtins__}
try:
    exec(compile(tree, path, "exec"), glb, glb)
except SystemExit:
    raise
except BaseException:
    traceback.print_exc()
    sys.exit(1)
)PYRUNNER";

// Deliberately generous: a data-analysis snippet should finish well inside
// this, and the UI can tighten it per call via ToolContext::timeout_sec.
constexpr int kDefaultTimeoutSec = 15;

// 512 MiB address space is enough for numpy-free work and stops a fork bomb or
// a huge allocation from taking the box down.
constexpr rlim_t kAddressSpaceBytes = 512ull * 1024 * 1024;
constexpr rlim_t kFileSizeBytes = 64ull * 1024 * 1024;
constexpr rlim_t kMaxProcesses = 64;
constexpr rlim_t kMaxFileDescriptors = 256;

#if !defined(_WIN32)

/// Best-effort cleanup of a partially-written temp dir on a failed run.
void remove_tree(const std::filesystem::path& p) {
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
}

void set_limit(int resource, rlim_t value) {
    struct rlimit rl;
    rl.rlim_cur = value;
    // Hard limit left at infinity: lowering it would be irreversible for the
    // child and buys nothing over the soft limit for a kill-on-timeout.
    rl.rlim_max = RLIM_INFINITY;
    setrlimit(resource, &rl);  // best effort: a denied limit is not fatal
}

struct ExecOutcome {
    int exit_code = -1;
    bool timed_out = false;
    bool spawn_failed = false;
    std::string output;  // stdout, then stderr
};

/// Run argv with `code` on stdin-ish (written to a temp file), capturing both
/// streams. `workdir` becomes the child's cwd.
ExecOutcome run_python_child(const std::string& code, const std::filesystem::path& workdir,
                             int timeout_sec) {
    ExecOutcome out;

    // The snippet goes in a file we exec with `python3 <file>`: writing to a
    // temp file (rather than piping stdin) keeps the child's stdin a real file,
    // so interactive reads get EOF immediately instead of hanging until the
    // timeout.
    const std::filesystem::path script = workdir / "snippet.py";
    {
        std::ofstream f(script, std::ios::binary);
        if (!f) {
            out.spawn_failed = true;
            out.output = "failed to write the snippet to " + script.string();
            return out;
        }
        f << code;
    }

    // A code interpreter echoes a trailing bare expression the way a REPL does.
    // Models constantly end a snippet with `fibonacci(15)` instead of
    // `print(fibonacci(15))`; run as a plain script that prints nothing and the
    // model is left reasoning about an empty result. The runner below rewrites
    // only that last top-level expression into a print() call -- everything else
    // is compiled and run verbatim.
    const std::filesystem::path runner = workdir / "_runner.py";
    {
        std::ofstream f(runner, std::ios::binary);
        if (!f) {
            out.spawn_failed = true;
            out.output = "failed to write the runner to " + runner.string();
            return out;
        }
        f << kRunnerSource;
    }

    int out_pipe[2];
    int err_pipe[2];
    if (pipe(out_pipe) != 0) {
        out.spawn_failed = true;
        out.output = std::string("pipe() failed: ") + std::strerror(errno);
        return out;
    }
    if (pipe(err_pipe) != 0) {
        close(out_pipe[0]);
        close(out_pipe[1]);
        out.spawn_failed = true;
        out.output = std::string("pipe() failed: ") + std::strerror(errno);
        return out;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        out.spawn_failed = true;
        out.output = std::string("fork() failed: ") + std::strerror(errno);
        return out;
    }

    if (pid == 0) {
        // ---- child ----
        // Detach from the parent's process group so a Ctrl-C on the server does
        // not race our timeout kill, and so a runaway cannot signal the server.
        setpgid(0, 0);

        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);

        // stdin from /dev/null: a script that reads input gets EOF, not a hang.
        const int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            if (devnull > STDERR_FILENO)
                close(devnull);
        }

        if (chdir(workdir.c_str()) != 0)
            _exit(126);

        set_limit(RLIMIT_AS, kAddressSpaceBytes);
        set_limit(RLIMIT_CPU, static_cast<rlim_t>(timeout_sec));
        set_limit(RLIMIT_FSIZE, kFileSizeBytes);
        set_limit(RLIMIT_NPROC, kMaxProcesses);
        set_limit(RLIMIT_NOFILE, kMaxFileDescriptors);
        // Core dumps from a killed snippet are noise on disk.
        set_limit(RLIMIT_CORE, 0);

        // Environment: keep it minimal and non-interactive. A model-authored
        // snippet should not depend on the server's environment, and this keeps
        // PYTHONPATH-style surprises out of the child.
        setenv("PATH", "/usr/local/bin:/usr/bin:/bin", 1);
        setenv("HOME", workdir.c_str(), 1);
        setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
        setenv("PYTHONUNBUFFERED", "1", 1);
        unsetenv("PYTHONPATH");
        unsetenv("LD_PRELOAD");
        unsetenv("LD_LIBRARY_PATH");

        const char* argv[] = {"python3", runner.c_str(), script.c_str(), nullptr};
        execvp("python3", const_cast<char* const*>(argv));
        _exit(127);  // exec failed
    }

    // ---- parent ----
    close(out_pipe[1]);
    close(err_pipe[1]);

    // Non-blocking reads so the timeout is enforced by us rather than by a
    // blocking read waiting on a child that ignores SIGTERM.
    for (int fd : {out_pipe[0], err_pipe[0]})
        fcntl(fd, F_SETFL, O_NONBLOCK);

    std::string stdout_text, stderr_text;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    bool both_closed = false;

    while (!both_closed) {
        if (std::chrono::steady_clock::now() > deadline) {
            out.timed_out = true;
            // Kill the whole child group: python can be mid-fork.
            kill(-pid, SIGKILL);
            kill(pid, SIGKILL);
            break;
        }

        struct pollfd fds[2];
        fds[0] = {out_pipe[0], POLLIN, 0};
        fds[1] = {err_pipe[0], POLLIN, 0};
        const int n = poll(fds, 2, 200);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            break;
        }

        for (int i = 0; i < 2; ++i) {
            if (!(fds[i].revents & (POLLIN | POLLHUP | POLLERR)))
                continue;
            char buf[8192];
            const ssize_t got = read(fds[i].fd, buf, sizeof(buf));
            if (got > 0) {
                (i == 0 ? stdout_text : stderr_text).append(buf, static_cast<size_t>(got));
            } else if (got == 0) {
                close(fds[i].fd);
                fds[i].fd = -1;
            } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                close(fds[i].fd);
                fds[i].fd = -1;
            }
        }
        both_closed = fds[0].fd < 0 && fds[1].fd < 0;
    }

    // Drain whatever is still buffered so a truncated read does not lose tail output.
    for (int i = 0; i < 2; ++i) {
        const int fd = (i == 0 ? out_pipe[0] : err_pipe[0]);
        if (fd < 0)
            continue;
        char buf[8192];
        ssize_t got;
        while ((got = read(fd, buf, sizeof(buf))) > 0)
            (i == 0 ? stdout_text : stderr_text).append(buf, static_cast<size_t>(got));
        close(fd);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    out.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (WIFSIGNALED(status))
        out.exit_code = 128 + WTERMSIG(status);

    out.output = stdout_text;
    if (!stderr_text.empty()) {
        if (!out.output.empty() && out.output.back() != '\n')
            out.output += "\n";
        out.output += stderr_text;
    }
    return out;
}

#endif  // !_WIN32

std::string python_interpreter() {
    const char* env = std::getenv("FORGE_PYTHON");
    return (env && *env) ? std::string(env) : std::string("python3");
}

}  // namespace

namespace {
// -1 = not overridden (fall back to the env), 0 = off, 1 = on.
int g_code_exec_override = -1;
}  // namespace

void set_code_exec_enabled(bool enabled) { g_code_exec_override = enabled ? 1 : 0; }

bool code_exec_enabled() {
    static const bool from_env = [] {
        const char* e = std::getenv("FORGE_ALLOW_CODE_EXEC");
        return e != nullptr && *e && std::strcmp(e, "0") != 0;
    }();
    if (g_code_exec_override >= 0)
        return g_code_exec_override == 1;
    return from_env;
}

void register_python_tool() {
    auto def = std::make_unique<ToolDefinition>();
    def->name = "run_python";
    def->description =
        "Execute a short Python 3 snippet and return whatever it prints. Use this for "
        "calculations, data wrangling, and quick simulations. Print the values you want "
        "to see; a trailing bare expression is echoed automatically, but anything not "
        "printed produces no output. Runs with a time limit.";
    def->parameters_json = R"({
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python 3 source to execute."},
            "timeout_sec": {"type": "integer", "description": "Optional wall-clock limit in seconds (1-60)."}
        },
        "required": ["code"]
    })";

    def->available = code_exec_enabled();
    if (!def->available) {
        def->unavailable_reason =
            "code execution is disabled; start forge-server with --enable-code-exec or "
            "FORGE_ALLOW_CODE_EXEC=1 to allow it";
    }

    const std::string interp = python_interpreter();
    def->ctx.timeout_sec = kDefaultTimeoutSec;

    def->exec = [interp](const json& args, const ToolContext& ctx) -> ToolResult {
        ToolResult r;
        const std::string code = args.value("code", "");
        if (code.find_first_not_of(" \t\r\n") == std::string::npos) {
            r.ok = false;
            r.error = "empty code";
            r.content = "Error: 'code' was empty.";
            return r;
        }

#if !defined(_WIN32)
        int timeout = ctx.timeout_sec > 0 ? ctx.timeout_sec : kDefaultTimeoutSec;
        if (args.contains("timeout_sec") && args["timeout_sec"].is_number_integer()) {
            timeout = args["timeout_sec"].get<int>();
        }
        timeout = std::max(1, std::min(60, timeout));

        // mkdtemp: a private 0700 dir under TMPDIR, removed on every path out.
        const char* tmpdir = std::getenv("TMPDIR");
        std::string tmpl = std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") + "/forge_py_XXXXXX";
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        if (!mkdtemp(buf.data())) {
            r.ok = false;
            r.error = std::string("mkdtemp failed: ") + std::strerror(errno);
            r.content = "Error: could not create a working directory.";
            return r;
        }
        const std::filesystem::path workdir(buf.data());

        const ExecOutcome res = run_python_child(code, workdir, timeout);
        remove_tree(workdir);

        if (res.spawn_failed) {
            r.ok = false;
            r.error = res.output;
            r.content = "Error: could not start python: " + res.output;
            return r;
        }
        if (res.timed_out) {
            r.ok = false;
            r.error = "timeout";
            r.content = "Error: the snippet exceeded the " + std::to_string(timeout) +
                        "s limit and was killed. Partial output:\n" + res.output;
            return r;
        }
        if (res.output.empty() && res.exit_code != 0) {
            r.ok = false;
            r.error = "exit " + std::to_string(res.exit_code);
            r.content = "Error: the snippet exited with code " + std::to_string(res.exit_code) +
                        " and produced no output.";
            return r;
        }

        r.ok = (res.exit_code == 0);
        if (!r.ok)
            r.error = "exit " + std::to_string(res.exit_code);
        r.content = res.output;
        if (!r.ok) {
            r.content = "The snippet exited with code " + std::to_string(res.exit_code) +
                        ".\nOutput:\n" + res.output;
        } else {
            r.content = "Output:\n" + res.output;
        }
        return r;
#else
        (void)interp;
        r.ok = false;
        r.error = "unsupported platform";
        r.content = "Error: code execution is only supported on POSIX platforms.";
        return r;
#endif
    };

    tool_registry_detail::add_tool(std::move(def));
}

}  // namespace forge