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

#include <atomic>
#include <csignal>
#include <iostream>
#include <string>

#include <httplib.h>

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
            if (r == forge::ParseResult::Ok) continue;
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
            if (r == forge::SpecParseResult::Ok) continue;
            if (r == forge::SpecParseResult::Error) {
                std::cerr << err << "\n";
                std::exit(1);
            }
        }

        if (arg == "--host") {
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
        } else if (arg == "--max-inflight") {
            opt.max_inflight = std::stoi(value_of(i, "--max-inflight"));
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
                << "                          allowed: " << forge::kv_cache_dtype_allowed_values() << "\n"
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
                << "      --n-keep N         Leading tokens never evicted on context shift\n"
                << "      --log-prefix       Diagnostic: per-depth prompt-prefix hashes\n"
                << "      --dequant-cache-mb MB  Per-weight FP16 dequant cache budget in MB\n"
                << "      --load-mode MODE    mmap | mlock | mmap_mlock (default: mmap)\n"
                << "      --api-key KEY       Require 'Authorization: Bearer KEY' (default: off)\n"
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
                << "      --adaptive-draft        Size each round's draft to the recent accepted length (off)\n"
                << "      --spec-n-min N, --spec-p-min F, --spec-ngram-n N,\n"
                << "      --spec-ngram-min N, --spec-ngram-mod[-n|-min|-max],\n"
                << "      --no-ngram, --spec-mtp, --spec-stats   See forge-cli --help\n"
                << "\n"
                << "      --perf              Log a per-request perf summary after each\n"
                << "                          completion (prefill/decode ms, tok/s)\n"
                << "      --perf-detail       --perf + top operators from PerfProfiler\n"
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
std::atomic<bool> g_stopping{false};

void handle_signal(int) {
    // Idempotent: stop() must run at most once. A second SIGINT (e.g. a double
    // Ctrl-C) before the listen loop unwinds would otherwise re-enter stop()
    // while is_running_ is still true but svr_sock_ is already INVALID_SOCKET,
    // tripping httplib's assertion in Server::stop() and aborting the process.
    if (g_svr && !g_stopping.exchange(true)) {
        g_svr->stop();
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

    if (!svr.listen(opt.host, opt.port)) {
        std::cerr << "Error: failed to listen on " << opt.host << ":" << opt.port << "\n";
        return 1;
    }
    return 0;
}
