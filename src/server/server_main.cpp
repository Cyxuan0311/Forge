/**
 * forge-server - OpenAI-compatible local inference API server (Phase 1)
 *
 * Usage:
 *   forge-server -m model.gguf                       # http://127.0.0.1:8080/v1
 *   forge-server -m model.gguf --port 8080 -ngl 99
 *
 * OpenCode (opencode.json):
 *   { "provider": { "forge": {
 *       "npm": "@ai-sdk/openai-compatible",
 *       "options": { "baseURL": "http://localhost:8080/v1", "apiKey": "none" },
 *       "models": { "<model-file-stem>": {} } } } }
 */

#include <httplib.h>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

#include "forge/dashboard.h"
#include "forge/logger.h"
#include "server.h"

namespace fsrv = forge;

#ifdef USE_CUDA
#    include <cuda_runtime.h>

#    include "forge/cuda_kernels.h"
#endif

namespace {

fsrv::ServerOptions parse_args(int argc, char** argv) {
    fsrv::ServerOptions opt;

    auto value_of = [&](int& i, const char* name) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << "Error: " << name << " requires a value\n";
            std::exit(1);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Shared verbatim with forge-cli: -m, -ngl, -c, -t, the KV cache dtype
        // and the separate -ctk/-ctv types, --paged-kv/--cuda-graph/--kv-offload,
        // --load-mode/--offload-embedding/--prefetch and the whole --kvmem block.
        // Lives in forge_infer so the two binaries cannot drift again.
        {
            std::string err;
            const forge::ParseResult r =
                forge::parse_common_arg(arg, argc, argv, i, opt.common, &err);
            if (r == forge::ParseResult::Ok)
                continue;
            if (r == forge::ParseResult::Error) {
                std::cerr << err << "\n";
                std::exit(1);
            }
        }

        // Speculative decoding, shared with forge-cli.
        {
            std::string err;
            const forge::SpecParseResult r =
                forge::parse_spec_arg(arg, argc, argv, i, opt.spec, &err);
            if (r == forge::SpecParseResult::Ok)
                continue;
            if (r == forge::SpecParseResult::Error) {
                std::cerr << err << "\n";
                std::exit(1);
            }
        }

        // Web dashboard, shared verbatim with forge-cli (see
        // forge/cli_dashboard_args.h for why these live in forge_infer).
        {
            std::string err;
            const forge::DashboardParseResult r =
                forge::parse_dashboard_arg(arg, argc, argv, i, opt.dashboard, &err);
            if (r == forge::DashboardParseResult::Ok)
                continue;
            if (r == forge::DashboardParseResult::Error) {
                std::cerr << err << "\n";
                std::exit(1);
            }
        }

        if (arg == "--chat") {
            opt.chat_ui = true;
        } else if (arg == "--host") {
            opt.host = value_of(i, "--host");
        } else if (arg == "--port") {
            opt.port = std::stoi(value_of(i, "--port"));
        } else if (arg == "--n-keep") {
            opt.n_keep = std::stoi(value_of(i, "--n-keep"));
        } else if (arg == "--log-prefix") {
            opt.log_prompt_prefix = true;
        } else if (arg == "--dequant-cache-mb") {
            opt.dequant_cache_mb = std::stol(value_of(i, "--dequant-cache-mb"));
        } else if (arg == "--api-key") {
            opt.api_key = value_of(i, "--api-key");
        } else if (arg == "--mmproj") {
            opt.mmproj_path = value_of(i, "--mmproj");
        } else if (arg == "--max-inflight") {
            opt.max_inflight = std::stoi(value_of(i, "--max-inflight"));
        } else if (arg == "--enable-code-exec") {
            opt.enable_code_exec = true;
        } else if (arg == "--perf") {
            opt.perf = true;
        } else if (arg == "--perf-detail") {
            opt.perf = true;
            opt.perf_detail = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "forge-server - OpenAI-compatible local inference API server\n\n"
                << "Usage: forge-server -m <model.gguf> [options]\n\n"
                << "  -m, --model PATH        GGUF model path (required)\n"
                << "      --host HOST         Bind address (default: 127.0.0.1)\n"
                << "      --port PORT         Listen port (default: 8080)\n"
                << "  -ngl,--n-gpu-layers N   Layers to offload to GPU (default: auto; also\n"
                << "                          accepts auto / all / N:M[:K] for multi-GPU)\n"
                << "  -c, --ctx-size N        Context window / KV cache size in tokens\n"
                << "                          (default: 0 = the model's native trained\n"
                << "                          context, reduced only when VRAM cannot hold it)\n"
                << "  -t, --threads N         CPU threads (default: auto)\n"
                << "      --kv-cache-dtype T  KV dtype for BOTH K and V (default: f16)\n"
                << "                          allowed: " << forge::kv_cache_dtype_allowed_values()
                << "\n"
                << "      -ctk, --cache-type-k T  KV dtype for K only\n"
                << "      -ctv, --cache-type-v T  KV dtype for V only\n"
                << "                          At equal bit width K is the more sensitive of the\n"
                << "                          two (it feeds softmax(Q.K), where error is\n"
                << "                          amplified, while V is a convex combination that\n"
                << "                          smooths it) -- prefer K at higher precision, e.g.\n"
                << "                          -ctk q8_0 -ctv q4_0.\n"
                << "      --cuda-graph        Enable CUDA Graph for decode\n"
                << "      --paged-kv          Use paged KV cache storage (default: off)\n"
                << "      --kvmem             Enable the tiered KV working set (implies\n"
                << "                          --paged-kv)\n"
                << "      --kvmem-budget N    GPU resident budget in tokens (default: 0 = the\n"
                << "                          paged pool size)\n"
                << "      --kvmem-sink N      Prefix tokens pinned resident (default: 256)\n"
                << "      --kvmem-gen-reserve N  Window slots held back for this turn's output\n"
                << "      --kvmem-retrieval M recent (default) | embed (reserved)\n"
                << "      --kv-write-policy P  Multi-level KV prefix policy (paged KV only):\n"
                << "                          write_back (default) | write_through | off.\n"
                << "                          Flags override the FORGE_KV_* env vars\n"
                << "      --kv-host-pool-bytes N  Host blob pool budget in bytes\n"
                << "                          (default: 1073741824, 0 = grow-only)\n"
                << "      --kv-async-backup B  Background device->host backup: 1 (default) | 0\n"
                << "      --kv-prefetch B      Restore prefetch of host-resident prefixes:\n"
                << "                          0 (default) | 1\n"
                << "      --kv-l3-backend B    L3 disk backend: file | off (default: off)\n"
                << "      --kv-l3-dir DIR      L3 segment directory\n"
                << "                          (default: <system tmp>/forge_kv_l3)\n"
                << "      --kv-l3-bytes N      L3 capacity cap in bytes (default: 1073741824)\n"
                << "      --kv-l3-segment-bytes N  L3 segment file size in bytes\n"
                << "                          (default: 67108864)\n"
                << "      --n-keep N         Leading tokens never evicted on context shift\n"
                << "      --log-prefix       Diagnostic: per-depth prompt-prefix hashes\n"
                << "      --dequant-cache-mb MB  Per-weight FP16 dequant cache budget in MB\n"
                << "      --load-mode MODE    mmap | mlock | mmap_mlock (default: mmap)\n"
                << "      --api-key KEY       Require 'Authorization: Bearer KEY' (default: off)\n"
                << "      --mmproj PATH       Load a vision encoder (mmproj GGUF) so the model can\n"
                << "                          accept image inputs (OpenAI vision messages)\n"
                << "      --max-inflight N    Max queued requests (default: 4, excess get 429)\n"
                << "\n"
                << "Speculative decoding (shared with forge-cli):\n"
                << "      --spec             Enable speculative decoding (n-gram draft)\n"
                << "      --spec-model PATH  DSpark/DFlash draft model GGUF\n"
                << "      --spec-draft-arch NAME  dflash | dspark\n"
                << "      --spec-draft N     Max drafted tokens per round (default: 5)\n"
                << "      --spec-draft-n N   Speculative tokens proposed per round (default: 5)\n"
                << "      --spec-draft-ngl N GPU layers for the draft model\n"
                << "      --spec-draft-target-layers LIST  Target layer indices, e.g. \"2,5,8\"\n"
                << "      --spec-draft-mask-id N   MASK token id (default: from GGUF)\n"
                << "      --spec-dspark-topk N     DSpark markov top-k (0 = full vocab)\n"
                << "      --spec-dspark-conf F     DSpark confidence truncation (0 = off)\n"
                << "      --spec-min-acc F   Disable drafting while acceptance EWMA < F\n"
                << "      --adaptive-draft        Size each round's draft to the recent accepted "
                   "length (off)\n"
                << "      --spec-n-min N, --spec-p-min F, --spec-ngram-n N,\n"
                << "      --spec-ngram-min N, --spec-ngram-mod[-n|-min|-max],\n"
                << "      --no-ngram, --spec-mtp, --spec-stats   See forge-cli --help\n"
                << "\n"
                << "      --perf              Log a per-request perf summary after each\n"
                << "                          completion (prefill/decode ms, tok/s)\n"
                << "      --perf-detail       --perf + top operators from PerfProfiler\n"
                << "\n"
                << "Web dashboard (shared with forge-cli):\n"
                << "      --dashboard         Serve a live performance dashboard on its OWN\n"
                << "                          loopback port (never on --port, so --api-key\n"
                << "                          is not bypassed). Opens a browser unless\n"
                << "                          --no-open / stdout is not a TTY\n"
                << "      --dashboard-port N  Preferred port (default: 9090; if busy the next\n"
                << "                          free port is used; 0 = let the OS choose)\n"
                << "      --dashboard-host H  Dashboard bind address (default: 127.0.0.1)\n"
                << "      --dashboard-ops     Also record the per-operator breakdown panel.\n"
                << "                          Opt-in: arming PerfProfiler slows decode, and only\n"
                << "                          works on a -DFORGE_PROFILING=ON build\n"
                << "      --no-open           Do not launch a browser\n"
                << "\n"
                << "Chat UI:\n"
                << "      --chat              Open the embedded chat UI in a browser after\n"
                << "                          startup (served at http://HOST:PORT/chat/; static\n"
                << "                          assets are key-exempt, /v1 still requires the\n"
                << "                          API key). Needs a Node.js build\n"
                << "\n"
                << "Tools:\n"
                << "      --enable-code-exec  Allow POST /v1/tools/exec to run the run_python\n"
                << "                          tool, which executes model-authored Python as\n"
                << "                          this user. Off by default: the rlimits and timeout\n"
                << "                          bound runaway resource use but are NOT a sandbox.\n"
                << "                          (env: FORGE_ALLOW_CODE_EXEC=1)\n"
                << "  -h, --help              Show this help\n";
            std::exit(0);
        } else {
            std::cerr << "Error: unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }

    if (opt.common.model_path.empty()) {
        std::cerr << "Error: -m/--model is required\n";
        std::exit(1);
    }
    // Cross-flag checks live in forge::validate_common_args() now.
    {
        std::string err;
        if (!forge::validate_common_args(opt.common, &err)) {
            std::cerr << err << "\n";
            std::exit(1);
        }
    }
    // Reserve is sized for a request's default max_tokens.
    opt.common.generation_budget = 512;
    return opt;
}

httplib::Server* g_svr = nullptr;
forge::DashboardServer* g_dash = nullptr;
std::atomic<bool> g_stopping{false};

void handle_signal(int) {
    // Idempotent: stop() must run at most once. A second SIGINT (e.g. a double
    // Ctrl-C) before the listen loop unwinds would otherwise re-enter stop()
    // while is_running_ is still true but svr_sock_ is already INVALID_SOCKET,
    // tripping httplib's assertion in Server::stop() and aborting the process.
    if (g_svr && !g_stopping.exchange(true)) {
        g_svr->stop();
    }
    // The dashboard owns a second listener with the same stop() contract, so it
    // needs its own guard rather than sharing the exchange above -- by the time a
    // second signal arrives the API listener is already unwinding.
    if (g_dash && g_dash->running()) {
        g_dash->stop();
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Log level: the default (WARN) hides the KV-cache sizing messages that
    // explain which context length was chosen and why. FORGE_LOG_LEVEL raises
    // it (error|warn|info|debug|trace) so the decision is observable.
    if (const char* ll = std::getenv("FORGE_LOG_LEVEL")) {
        const std::string s = ll;
        forge::LogLevel lvl = forge::LogLevel::WARN;
        if (s == "error")
            lvl = forge::LogLevel::LOG_ERROR;
        else if (s == "warn")
            lvl = forge::LogLevel::WARN;
        else if (s == "info")
            lvl = forge::LogLevel::INFO;
        else if (s == "debug")
            lvl = forge::LogLevel::DEBUG;
        else if (s == "trace")
            lvl = forge::LogLevel::TRACE;
        forge::Logger::instance().set_level(lvl);
    }

    fsrv::ServerOptions opt = parse_args(argc, argv);

#ifdef USE_CUDA
    // Apply the dequant-cache budget before any weight is loaded/used, so the
    // very first launch_cublas_gemm_quant_fp16 call honors --dequant-cache-mb
    // (or falls back to FORGE_DEQUANT_CACHE_MB env / 1024 MB default).
    if (opt.dequant_cache_mb >= 0) {
        forge::cuda::set_dequant_cache_budget_mb_explicit(opt.dequant_cache_mb);
    }
#endif

    fsrv::ServerState state;
    std::string error;
    std::cout << "Loading model: " << opt.common.model_path << "\n";
    if (!state.load(opt, error)) {
        std::cerr << "Error: " << error << "\n";
        return 1;
    }

    httplib::Server svr;

    // CORS + OPTIONS preflight + optional Bearer api-key auth.
    fsrv::setup_middleware(svr, opt);

    fsrv::register_routes(svr, state);

    // Graceful shutdown: SIGINT/SIGTERM stop the accept loop so the process
    // unwinds normally and CUDA resources are released.
    g_svr = &svr;
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::cout << "\nforge-server listening at http://" << opt.host << ":" << opt.port
              << "  (model: " << state.model_id() << ")\n"
              << "OpenAI-compatible base URL: http://" << opt.host << ":" << opt.port << "/v1\n"
              << "Endpoints: GET /health | GET /v1/models | POST /v1/chat/completions\n\n";

    // ---- Embedded chat UI (--chat) ----
    // Served on the MAIN port under /chat/ (static assets are exempt from the
    // Bearer gate; every /v1 call still authenticates). The browser is opened
    // just before listen(), same as the dashboard: start-up latency covers the
    // tiny window before the accept loop is live. --no-open (shared with the
    // dashboard flags) suppresses it.
    if (opt.chat_ui) {
        const std::string ui_host = opt.host == "0.0.0.0" ? "127.0.0.1" : opt.host;
        const std::string chat_url =
            "http://" + ui_host + ":" + std::to_string(opt.port) + "/chat/";
        std::cout << "Chat UI:   " << chat_url << "\n";
        if (opt.dashboard.open_browser) {
#ifdef _WIN32
            ShellExecuteA(nullptr, "open", chat_url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif __APPLE__
            if (std::system(("open '" + chat_url + "' >/dev/null 2>&1 &").c_str()) != 0) {
                /* non-fatal */
            }
#else
            if (std::system(("xdg-open '" + chat_url + "' >/dev/null 2>&1 &").c_str()) != 0) {
                /* non-fatal */
            }
#endif
        }
        std::cout << "\n";
    }

    // ---- Live web dashboard (--dashboard) ----
    // Started AFTER the API listener has printed its banner but BEFORE
    // listen(), so the URL is visible even if the API port turns out to be busy
    // and the process exits. Own port, own thread: the API listener's auth
    // middleware must not have to be bypassed for a browser to see charts.
    if (opt.dashboard.enabled) {
        forge::MetricsRegistry::instance().set_app("forge-server");
        forge::MetricsRegistry::instance().set_model(state.model_id(), state.arch());
        forge::MetricsRegistry::instance().set_enabled(true);
        // Safe before listen(): no request can be in flight yet, so this is the
        // one publish_engine_gauges() call made outside the inference lock.
        state.publish_engine_gauges();
        // Mid-generation KV refresh, driven from the inference thread by
        // MetricsRegistry. See ServerState::fill_engine_gauges.
        forge::MetricsRegistry::instance().set_live_gauge_provider(
            [&state](forge::EngineGauges& g) { state.fill_engine_gauges(g); });

        forge::DashboardServer dash;
        std::string dash_err;
        if (dash.start(opt.dashboard, dash_err)) {
            g_dash = &dash;
            std::cout << "Dashboard: " << dash.url() << "\n\n";
        } else {
            // A dashboard that cannot bind must not take the server down with
            // it: the API is the thing the user actually asked for.
            std::cerr << "Warning: dashboard not started (" << dash_err << ")\n\n";
        }

        if (!svr.listen(opt.host, opt.port)) {
            std::cerr << "Error: failed to listen on " << opt.host << ":" << opt.port << "\n";
            dash.stop();
            return 1;
        }
        dash.stop();
        return 0;
    }

    if (!svr.listen(opt.host, opt.port)) {
        std::cerr << "Error: failed to listen on " << opt.host << ":" << opt.port << "\n";
        return 1;
    }
    return 0;
}
