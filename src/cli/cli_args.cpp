/**
 * Forge CLI - Argument parsing
 */

#include <cstdlib>
#include <iostream>
#include <string>

#include "cli_common.h"

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
           "       --kv-cache-dtype TYPE KV cache dtype: fp32, q4_0 (default: fp32)\n"
           "       --cuda-graph          Enable CUDA Graph for decode (default: off)\n"
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
           "       --spec-draft N         Max drafted tokens per round (default: 5)\n"
           "       --spec-n-min N         Min candidates per round or discard (default: 0)\n"
           "       --spec-ngram-n N       Longest suffix match length (default: 5)\n"
           "       --spec-ngram-min N     Shortest suffix match length (default: 2)\n"
           "       --spec-p-min FLOAT     Draft confidence early-stop threshold (default: 0.0)\n"
           "       --spec-model PATH      Standalone small draft model path (Phase 3)\n"
           "       --spec-draft-ngl N     GPU layers for the draft model (default: same as -ngl)\n"
           "       --spec-stats           Print speculation stats after generation\n"
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

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.model_path = argv[i];
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
                args.n_gpu_layers = -1;  // auto: all layers on GPU (llama.cpp auto default)
            } else if (val == "all" || val == "-2") {
                args.n_gpu_layers = -2;  // all: explicit all-on-GPU
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
                            args.gpu_layers_per_dev.push_back(std::stoi(part));
                        }
                        if (next == std::string::npos) break;
                        pos = next + 1;
                    }
                    // Sum all per-device layers as total gpu_layers
                    args.n_gpu_layers = 0;
                    for (int n : args.gpu_layers_per_dev) args.n_gpu_layers += n;
                } else {
                    args.n_gpu_layers = std::stoi(val);
                }
            }
        } else if (arg == "-kvo" || arg == "--kv-offload") {
            args.offload_kqv = true;
        } else if (arg == "-nkvo" || arg == "--no-kv-offload") {
            args.offload_kqv = false;
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.threads = std::stoi(argv[i]);
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.batch_size = std::stoi(argv[i]);
        } else if (arg == "--kv-cache-dtype") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.kv_cache_dtype = argv[i];
            if (args.kv_cache_dtype != "fp32" && args.kv_cache_dtype != "q4_0") {
                std::cerr << "Error: Unsupported KV cache dtype: " << args.kv_cache_dtype << "\n";
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
            args.cuda_graph = true;
        } else if (arg == "--load-mode") {
            if (++i >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(1);
            }
            args.load_mode = argv[i];
            if (args.load_mode != "mmap" && args.load_mode != "mlock" && args.load_mode != "mmap_mlock") {
                std::cerr << "Error: Invalid --load-mode: " << args.load_mode
                          << ". Valid: mmap, mlock, mmap_mlock\n";
                std::exit(1);
            }
        } else if (arg == "--offload-embedding") {
            args.offload_embedding = true;
        } else if (arg == "--no-offload-embedding") {
            args.offload_embedding = false;
        } else if (arg == "--no-jinja") {
            args.no_jinja = true;
        } else if (arg == "--prefetch") {
            args.prefetch = true;
        } else if (arg == "--no-prefetch") {
            args.prefetch = false;
        } else if (arg == "--info") {
            args.info_only = true;
        } else if (arg == "--bench") {
            args.bench = true;
        } else if (arg == "--spec") {
            args.spec.enabled = true;
            args.spec.use_ngram = true;
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
        } else if (arg == "--spec-stats") {
            args.spec.print_stats = true;
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

    if (args.prompt.empty() && !args.info_only && !args.bench) {
        args.interactive = true;
    }

    return args;
}
