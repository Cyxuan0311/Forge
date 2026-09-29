/**
 * Forge CLI - Argument parsing
 */

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cli_common.h"

#include "forge/cli_spec_args.h"

static void print_usage(const char* prog) {
    std::cout
        << "Forge CLI - Lightweight LLM inference tool v0.5.0\n"
           "\n"
           "Usage: "
        << prog
        << " [options]\n"
           "\n"
           "Basic:\n"
           "  -m,  --model PATH          Model file path (.gguf / .ninf)\n"
           "  -p,  --prompt TEXT         Prompt (for non-interactive mode)\n"
           "  -n,  --n-predict N         Number of tokens to predict (default: 256, -1=unlimited)\n"
           "       --mmproj PATH         Multimodal vision encoder path (mmproj-model.gguf)\n"
           "       --image PATH          Input image path (multimodal mode)\n"
           "\n"
           "Performance:\n"
           "  -ngl,--n-gpu-layers N      GPU layers (default: -1=auto=all, 0=CPU only, N=first N layers on GPU, \"auto\"/\"all\" supported)\n"
           "                           Multi-GPU: -ngl N:M (e.g., -ngl 16:16 = GPU0:16 layers + GPU1:16 layers)\n"
           "  -kvo,--kv-offload           Offload KV cache to GPU (default: on)\n"
           "  -nkvo,--no-kv-offload        Keep KV cache on CPU (for long context + small VRAM)\n"
           "  -t,  --threads N           CPU threads (default: auto)\n"
           "  -b,  --batch-size N        Prompt processing batch size (default: 512)\n"
           "  -c,  --ctx-size N          Context window / KV cache size in tokens (default: 0 = model\n"
           "                           native length, clamped to fit memory; use it to cap KV RAM)\n"
           "       --kv-cache-dtype TYPE KV cache dtype: fp32, f16, q8_0, q4_0 (default: fp32)\n"
           "       --cuda-graph          Enable CUDA Graph for decode (default: off)\n"
           "       --paged-kv            Use paged KV cache storage (default: off, contiguous)\n"
           "\n"
           "Tiered KV (KVMem, arXiv:2609.04852):\n"
           "       --kvmem               Enable the tiered KV working set: a bounded GPU\n"
           "                           resident window over host-archived history, so the\n"
           "                           logical context (-c) can exceed the GPU pool. Implies\n"
           "                           --paged-kv.\n"
           "       --kvmem-budget N      GPU resident budget in tokens (default: 0 = the\n"
           "                           paged pool size; the pool is capped to it)\n"
           "       --kvmem-sink N        Prefix tokens pinned resident forever (default: 256)\n"
           "       --kvmem-gen-reserve N Window slots held back for this turn's output\n"
           "                           (default: 0)\n"
           "       --kvmem-retrieval M   Recall policy for archived history: recent (default)\n"
           "                           | embed (reserved)\n"
           "\n"
           "Memory:\n"
            "       --load-mode MODE      Memory mapping mode: mmap (default), mlock, mmap_mlock\n"
            "       --prefetch            Warm model page cache at load (default: on)\n"
            "       --no-prefetch         Skip load-time page prefetch\n"
            "       --offload-embedding    Offload token_embedding to GPU (default: on)\n"
           "       --no-offload-embedding Keep token_embedding on CPU during partial offload\n"
           "\n"
            "Speculative Decoding:\n"
            "       --spec                 Enable speculative decoding (n-gram, off by default)\n"
            "       --spec-mtp             Use bundled qwen35 DeepSeek-MTP head\n"
            "       --spec-draft N         Max drafted tokens per round (default: 5)\n"
            "       --spec-n-min N         Min candidates per round or discard (default: 0)\n"
            "       --spec-ngram-n N       Longest suffix match length (default: 5)\n"
            "       --spec-ngram-min N     Shortest suffix match length (default: 2)\n"
            "       --spec-ngram-mod       Enable global hash-pool ngram-mod (llama.cpp style)\n"
            "       --spec-ngram-mod-n N   Hash key length for ngram-mod (default: 24)\n"
            "       --spec-ngram-mod-min N Min draft for ngram-mod to keep (default: 1)\n"
            "       --spec-ngram-mod-max N Max draft for ngram-mod (default: 64)\n"
            "       --no-ngram            Disable the plain n-gram fallback (use only with --spec-ngram-mod to isolate mod)\n"
            "       --spec-p-min FLOAT     Draft confidence early-stop threshold (default: 0.0)\n"
            "       --spec-model PATH      Standalone small draft model path (Phase 3)\n"
            "       --spec-draft-ngl N     GPU layers for the draft model (default: same as -ngl)\n"
            "       --spec-draft-arch NAME Draft model architecture: dflash | dspark\n"
            "       --spec-draft-target-layers LIST  Target layer indices feeding the draft encoder, e.g. \"2,5,8\"\n"
            "       --spec-draft-mask-id N MASK token id for the DFlash query block (default: from GGUF)\n"
            "       --spec-draft-n N       Speculative tokens proposed per round (default: 5)\n"
            "       --spec-dspark-topk N  DSpark: apply markov bias only to top-N base-logit candidates (0 = full vocab)\n"
            "       --spec-dspark-conf F   DSpark: drop drafted positions whose confidence < F (default: 0.5)\n"
            "       --spec-min-acc F       Disable drafting (fall back to plain decode) while acceptance EWMA < F\n"
            "       --no-adaptive-draft     Always draft the full --spec-draft budget instead of sizing it to the\n"
            "                             recent accepted length (adaptive sizing is on by default)\n"
            "                           (default: 0 = always speculate; useful when the drafter is larger\n"
            "                           than the target, e.g. MiniCPM5-2.6B-DSpark over a 2B target)\n"
            "       --perf                Print per-phase performance breakdown after generation\n"
            "                           (requires a build with -DFORGE_PROFILING=ON; host wall clock)\n"
            "       --perf-cuda-events    Same breakdown timed with CUDA events (device time, but\n"
            "                           costs several extra cudaEventRecord per scope and slows decode)\n"
            "       --sync-prof           Leaf-level device timing (FORGE_SYNC_PROF=1 internally):\n"
            "                           hard-syncs around every op, so it is diagnostic-only and slow.\n"
            "                           Use with --perf; the leaf sum vs forward total is the signal.\n"
            "       --spec-stats           Print speculation stats after generation\n"
            "       --expert-stats         Print MoE expert router-distribution after generation\n"
            "       --expert-paging        Enable per-expert weight movement (partial activation)\n"
           "       --expert-budget-mb MB  VRAM budget for resident experts, MB (0 = unbounded)\n"
           "\n"
           "Sampling:\n"
           "       --temp FLOAT          Sampling temperature (default: 0.7, 0=greedy)\n"
           "       --top-k N             Top-K sampling (default: 40, 0=disabled)\n"
           "       --top-p FLOAT         Top-P sampling (default: 0.9)\n"
           "       --repeat-penalty FLOAT Repeat penalty (default: 1.1)\n"
           "       --repeat-last-n N    Window for repeat penalty (default: 64, 0=full history)\n"
           "       --seed N              Random seed (default: 0)\n"
           "\n"
           "Mode:\n"
           "  -i,  --interactive         Interactive chat mode\n"
           "       --no-stream           Disable streaming output\n"
           "       --system-prompt TEXT  System prompt\n"
           "\n"
           "       --no-jinja             Force fallback to hardcoded chat templates\n"
           "\n"
           "Info & Debug:\n"
           "       --info                Show model info and exit\n"
           "       --bench               Run performance benchmark\n"
           "  -v,  --verbose             Verbose logging\n"
           "  -h,  --help                Show this help\n"
           "\n"
           "Interactive commands:\n"
           "  /quit, /exit     Exit\n"
           "  /clear           Clear conversation history\n"
           "  /system TEXT     Set system prompt\n"
           "  /image PATH      Load image (multimodal)\n"
           "  /save PATH       Save conversation to file\n"
           "  /help            Show help\n"
           "\n"
           "Examples:\n"
           "  # Interactive chat\n"
           "  "
        << prog
        << " -m model.gguf\n"
           "\n"
           "  # Text completion\n"
           "  "
        << prog
        << " -m model.gguf -p \"What is machine learning?\"\n"
           "\n"
           "  # CPU-only inference\n"
           "  "
        << prog
        << " -m model.gguf -ngl 0\n"
           "\n"
           "  # Multimodal inference\n"
           "  "
        << prog
        << " -m model.gguf --mmproj mmproj.gguf --image photo.jpg -p \"Describe this image\"\n"
           "\n"
           "  # Model info\n"
           "  "
        << prog << " -m model.gguf --info\n";
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];


        // Shared with forge-server: -m, -ngl, -c, -t, the KV cache dtype and the
        // separate -ctk/-ctv types, --paged-kv/--cuda-graph/--kv-offload,
        // --load-mode/--offload-embedding/--prefetch and the whole --kvmem block.
        // Lives in forge_infer so the two binaries cannot drift again. Handled
        // here, ahead of the CLI-only chain below.
        {
            std::string err;
            const forge::ParseResult r =
                forge::parse_common_arg(arg, argc, argv, i, args.common, &err);
            if (r == forge::ParseResult::Ok) continue;
            if (r == forge::ParseResult::Error) {
                std::cerr << err << "\n";
                std::exit(1);
            }
        }

        // Speculative decoding, shared with forge-server. The server had none of
        // these flags and never put a SpeculativeConfig into ContextParams, so
        // Generator never enabled speculation there and the server's own
        // spec_stats reporting was unreachable.
        {
            std::string err;
            const forge::SpecParseResult r =
                forge::parse_spec_arg(arg, argc, argv, i, args.spec, &err);
            if (r == forge::SpecParseResult::Ok) continue;
            if (r == forge::SpecParseResult::Error) {
                std::cerr << err << "\n";
                std::exit(1);
            }
        }

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "-m" || arg == "--model") {  // shared parser owns this
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.common.model_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.prompt = argv[i];
        } else if (arg == "-n" || arg == "--n-predict") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.n_predict = std::stoi(argv[i]);
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            std::string val = argv[i];
            if (val == "auto" || val == "-1") {
                args.common.n_gpu_layers = -1;  // auto: all layers on GPU (llama.cpp auto default)
            } else if (val == "all" || val == "-2") {
                args.common.n_gpu_layers = -2;  // all: explicit all-on-GPU
            } else {
                // Check for multi-GPU syntax: N:M or N:M:K...
                size_t colon_pos = val.find(':');
                if (colon_pos != std::string::npos) {
                    // Multi-GPU: parse "16:16" → [16, 16]
                    size_t pos = 0;
                    while (pos < val.size()) {
                        size_t next = val.find(':', pos);
                        std::string part = val.substr(pos, next - pos);
                        if (!part.empty()) {
                            args.common.gpu_layers_per_dev.push_back(std::stoi(part));
                        }
                        if (next == std::string::npos) break;
                        pos = next + 1;
                    }
                    // Sum all per-device layers as total gpu_layers
                    args.common.n_gpu_layers = 0;
                    for (int n : args.common.gpu_layers_per_dev) args.common.n_gpu_layers += n;
                } else {
                    args.common.n_gpu_layers = std::stoi(val);
                }
            }
        } else if (arg == "-kvo" || arg == "--kv-offload") {
            args.common.offload_kqv = true;
        } else if (arg == "-nkvo" || arg == "--no-kv-offload") {
            args.common.offload_kqv = false;
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.common.threads = std::stoi(argv[i]);
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.batch_size = std::stoi(argv[i]);
        } else if (arg == "-c" || arg == "--ctx-size" || arg == "--n-ctx") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.common.ctx_size = std::stoi(argv[i]);
            if (args.common.ctx_size < 0) {
                std::cerr << "Error: --ctx-size must be >= 0 (0 = auto = model native length)\n";
                std::exit(1);
            }
        } else if (arg == "--temp" || arg == "--temperature") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.top_p = std::stof(argv[i]);
        } else if (arg == "--repeat-penalty") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.repeat_penalty = std::stof(argv[i]);
        } else if (arg == "--repeat-last-n") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.repeat_last_n = std::stoi(argv[i]);
        } else if (arg == "--seed") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.seed = std::stoull(argv[i]);
        } else if (arg == "-i" || arg == "--interactive") {
            args.interactive = true;
        } else if (arg == "--no-stream") {
            args.no_stream = true;
        } else if (arg == "--mmproj") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.mmproj_path = argv[i];
        } else if (arg == "--image") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.image_path = argv[i];
        } else if (arg == "--system-prompt") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.system_prompt = argv[i];
        } else if (arg == "--cuda-graph") {
            args.common.cuda_graph = true;
        } else if (arg == "--paged-kv") {
            args.common.paged_kv = true;
        } else if (arg == "--kvmem") {
            args.common.kvmem = true;
            args.common.paged_kv = true;  // the tier needs the page pool it caps
        } else if (arg == "--kvmem-budget") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.common.kvmem_budget = std::stoi(argv[i]);
        } else if (arg == "--kvmem-sink") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.common.kvmem_sink = std::stoi(argv[i]);
        } else if (arg == "--kvmem-gen-reserve") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.common.kvmem_gen_reserve = std::stoi(argv[i]);
        } else if (arg == "--kvmem-retrieval") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.common.kvmem_retrieval = argv[i];
            if (args.common.kvmem_retrieval != "recent" && args.common.kvmem_retrieval != "embed") {
                std::cerr << "Error: Invalid --kvmem-retrieval: " << args.common.kvmem_retrieval
                          << ". Valid: recent, embed\n";
                std::exit(1);
            }
        } else if (arg == "--load-mode") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.common.load_mode = argv[i];
            if (args.common.load_mode != "mmap" && args.common.load_mode != "mlock" && args.common.load_mode != "mmap_mlock") {
                std::cerr << "Error: Invalid --load-mode: " << args.common.load_mode
                          << ". Valid: mmap, mlock, mmap_mlock\n";
                std::exit(1);
            }
        } else if (arg == "--offload-embedding") {
            args.common.offload_embedding = true;
        } else if (arg == "--no-offload-embedding") {
            args.common.offload_embedding = false;
        } else if (arg == "--no-jinja") {
            args.no_jinja = true;
        } else if (arg == "--prefetch") {
            args.common.prefetch = true;
        } else if (arg == "--no-prefetch") {
            args.common.prefetch = false;
        } else if (arg == "--info") {
            args.info_only = true;
        } else if (arg == "--bench") {
            args.bench = true;
        } else if (arg == "--spec") {
            args.spec.enabled = true;
            args.spec.use_ngram = true;
        } else if (arg == "--spec-mtp") {
            args.spec.enabled = true;
            args.spec.use_mtp = true;
        } else if (arg == "--spec-draft") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.use_ngram = true;
            args.spec.n_draft = std::stoi(argv[i]);
        } else if (arg == "--spec-n-min") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.use_ngram = true;
            args.spec.n_min = std::stoi(argv[i]);
        } else if (arg == "--spec-ngram-n") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.use_ngram = true;
            args.spec.ngram_n = std::stoi(argv[i]);
        } else if (arg == "--spec-ngram-min") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.use_ngram = true;
            args.spec.ngram_min = std::stoi(argv[i]);
        } else if (arg == "--spec-p-min") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.p_min = std::stof(argv[i]);
        } else if (arg == "--spec-model") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.draft_model_path = argv[i];
        } else if (arg == "--spec-draft-ngl") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.draft_gpu_layers = std::stoi(argv[i]);
        } else if (arg == "--spec-draft-arch") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            const std::string arch = argv[i];
            if (arch != "dflash" && arch != "dspark") {
                std::cerr << "Error: --spec-draft-arch must be 'dflash' or 'dspark'\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.draft_arch = arch;
        } else if (arg == "--spec-draft-target-layers") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.draft_target_layers.clear();
            const std::string list = argv[i];
            size_t start = 0;
            while (true) {
                const size_t comma = list.find(',', start);
                const std::string piece =
                    list.substr(start, comma == std::string::npos ? std::string::npos
                                                                  : comma - start);
                if (!piece.empty()) {
                    try {
                        args.spec.draft_target_layers.push_back(std::stoi(piece));
                    } catch (const std::exception&) {
                        std::cerr << "Error: --spec-draft-target-layers expects "
                                     "comma-separated layer indices\n";
                        std::exit(1);
                    }
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (args.spec.draft_target_layers.empty()) {
                std::cerr << "Error: --spec-draft-target-layers is empty\n";
                std::exit(1);
            }
        } else if (arg == "--spec-draft-mask-id") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.draft_mask_token_id = std::stoi(argv[i]);
        } else if (arg == "--spec-draft-n") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.draft_n_spec = std::stoi(argv[i]);
        } else if (arg == "--spec-dspark-topk") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.dspark_top_k = std::stoi(argv[i]);
        } else if (arg == "--spec-dspark-conf") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.dspark_conf_threshold = std::stof(argv[i]);
        } else if (arg == "--spec-min-acc") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.min_accept_rate = std::stof(argv[i]);
        } else if (arg == "--perf") {
            // Enable the per-phase PerfProfiler breakdown at generation end. Takes
            // effect only when forge was built with -DFORGE_PROFILING=ON; the
            // generator's decode-only dump (DECODE-ONLY PROFILE) reads this env.
            setenv("FORGE_PERF_DECODE", "1", 1);
        } else if (arg == "--perf-cuda-events") {
            // Same breakdown, but timed with CUDA events instead of host wall
            // clock. More faithful per-op DEVICE time (host wall mis-attributes
            // async queue drains), but each scope then costs two cudaEventRecord
            // calls, which dominates wall time on launch-bound decode shapes --
            // measured ~5x end-to-end slowdown on a 2B/42-layer target. Use only
            // when the device/host split itself is the question.
            setenv("FORGE_PERF_DECODE", "1", 1);
            setenv("FORGE_PERF_CUDA_EVENTS", "1", 1);
        } else if (arg == "--sync-prof") {
            // Leaf-level device timing. Arms the hard-sync timers that bracket
            // each op individually; only meaningful together with --perf, and
            // diagnostic-only (a full pipeline drain per scope).
            setenv("FORGE_PERF_DECODE", "1", 1);
            setenv("FORGE_SYNC_PROF", "1", 1);
        } else if (arg == "--spec-ngram-mod") {
            args.spec.enabled = true;
            args.spec.use_ngram_mod = true;
        } else if (arg == "--no-ngram") {
            args.spec.enabled = true;
            args.spec.no_ngram = true;
            args.spec.use_ngram = false;
        } else if (arg == "--no-adaptive-draft") {
            args.spec.enabled = true;
            args.spec.adaptive_draft = false;
        } else if (arg == "--spec-ngram-mod-n") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.use_ngram_mod = true;
            args.spec.ngram_mod_n = std::stoi(argv[i]);
        } else if (arg == "--spec-ngram-mod-min") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.use_ngram_mod = true;
            args.spec.ngram_mod_n_min = std::stoi(argv[i]);
        } else if (arg == "--spec-ngram-mod-max") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.spec.enabled = true;
            args.spec.use_ngram_mod = true;
            args.spec.ngram_mod_n_max = std::stoi(argv[i]);
        } else if (arg == "--spec-stats") {
            args.spec.print_stats = true;
        } else if (arg == "--expert-stats") {
            args.expert_stats = true;
        } else if (arg == "--expert-paging") {
            args.expert_paging = true;
            args.expert_stats = true;  // paging implies stats visibility
        } else if (arg == "--expert-budget-mb") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.expert_budget_mb = std::stoi(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = 2;
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }

    if (args.n_predict == -1) {
        args.n_predict = 256;
    }

    if (args.temperature == 0.0f) {
        args.no_sample = true;
    }

    // Cross-flag checks live in forge::validate_common_args() now (the KVMem
    // budget/sink rule forge-server carried a near-identical copy of, which is
    // how the two drifted).
    {
        std::string err;
        if (!forge::validate_common_args(args.common, &err)) {
            std::cerr << err << "\n";
            std::exit(1);
        }
    }
    // The KVMem reserve is sized for this run's generation budget.
    args.common.generation_budget = args.n_predict > 0 ? args.n_predict : 256;

    if (args.prompt.empty() && !args.info_only && !args.bench) {
        args.interactive = true;
    }

    return args;
}
