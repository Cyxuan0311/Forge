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

#include <csignal>
#include <iostream>
#include <string>

#include <httplib.h>

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
        if (arg == "-m" || arg == "--model") {
            opt.model_path = value_of(i, "--model");
        } else if (arg == "--host") {
            opt.host = value_of(i, "--host");
        } else if (arg == "--port") {
            opt.port = std::stoi(value_of(i, "--port"));
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            opt.n_gpu_layers = std::stoi(value_of(i, "--n-gpu-layers"));
        } else if (arg == "-c" || arg == "--ctx-size") {
            opt.ctx_size = std::stoi(value_of(i, "--ctx-size"));
        } else if (arg == "-t" || arg == "--threads") {
            opt.threads = std::stoi(value_of(i, "--threads"));
        } else if (arg == "--kv-cache-dtype") {
            opt.kv_cache_dtype = value_of(i, "--kv-cache-dtype");
        } else if (arg == "--cuda-graph") {
            opt.cuda_graph = true;
        } else if (arg == "--paged-kv") {
            opt.paged_kv = true;
        } else if (arg == "--dequant-cache-mb") {
            opt.dequant_cache_mb = std::stol(value_of(i, "--dequant-cache-mb"));
        } else if (arg == "--load-mode") {
            opt.load_mode = value_of(i, "--load-mode");
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
                << "  -ngl, --n-gpu-layers N  Layers to offload to GPU (-1 = max)\n"
                << "  -c,  --ctx-size N       Context window / KV cache size in tokens\n"
                << "                          (default: auto, capped at 4096; increase for\n"
                << "                          long agent sessions, e.g. opencode)\n"
                << "  -t, --threads N         CPU threads (default: auto)\n"
                << "      --kv-cache-dtype T  fp32 | f16 | q8_0 | q4_0 (default: fp32)\n"
                << "      --cuda-graph        Enable CUDA Graph for decode\n"
                << "      --paged-kv          Use paged KV cache storage (default: off)\n"
                << "      --dequant-cache-mb MB  Per-weight FP16 dequant cache budget in MB\n"
                << "                          (default 1024; 0 disables the cache; env\n"
                << "                          FORGE_DEQUANT_CACHE_MB also honored)\n"
                << "      --load-mode MODE    mmap | mlock | mmap_mlock (default: mmap)\n"
                << "      --api-key KEY       Require 'Authorization: Bearer KEY' (default: off)\n"
                << "      --max-inflight N    Max queued inference requests (default: 4,\n"
                << "                          excess requests get HTTP 429)\n"
                << "      --perf              Log per-request perf summary after every\n"
                << "                          completion (prefill/decode ms, tok/s)\n"
                << "      --perf-detail       --perf + top operators from PerfProfiler\n"
                << "                          (operator data needs FORGE_PROFILING=ON build)\n"
                << "  -h, --help              Show this help\n";
            std::exit(0);
        } else {
            std::cerr << "Error: unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }

    if (opt.model_path.empty()) {
        std::cerr << "Error: -m/--model is required\n";
        std::exit(1);
    }
    return opt;
}

httplib::Server* g_svr = nullptr;

void handle_signal(int) {
    if (g_svr) {
        g_svr->stop();
    }
}

}  // namespace

int main(int argc, char** argv) {
    fsrv::ServerOptions opt = parse_args(argc, argv);

#ifdef USE_CUDA
    // Apply the dequant-cache budget before any weight is loaded/used, so the
    // very first launch_cublas_gemm_quant_fp16 call honors --dequant-cache-mb
    // (or falls back to FORGE_DEQUANT_CACHE_MB env / 1024 MB default).
    if (opt.dequant_cache_mb >= 0) {
        forge::cuda::set_dequant_cache_budget_mb(opt.dequant_cache_mb);
    }
#endif

    fsrv::ServerState state;
    std::string error;
    std::cout << "Loading model: " << opt.model_path << "\n";
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
