/**
 * Forge CLI - Command-line inference tool
 *
 * Inspired by llama.cpp's llama-cli design, providing fine-grained control
 * over model inference. Supports interactive chat, text completion,
 * multimodal inference, and more.
 *
 * Usage:
 *   forge-cli -m model.gguf                    # Interactive chat
 *   forge-cli -m model.gguf -p "Hello"         # Text completion
 *   forge-cli -m model.gguf --mmproj mm.gguf   # Multimodal
 *   forge-cli -m model.gguf --info             # Model info
 */

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

#include "cli_common.h"

// Forge headers
#include "forge/arch_registry.h"
#include "forge/context.h"
#include "forge/engine.h"
#include "forge/inference/forward_request.h"
#include "forge/kv_cache.h"
#include "forge/kv_memory.h"
#include "forge/kv_storage.h"
#include "forge/logger.h"

#include "forge/model.h"
#include "forge/perf_profiler.h"
#include "forge/tensor.h"
#include "forge/threads.h"
#include "forge/tokenizer.h"
#include "forge/types.h"
#include "forge/vision_encoder.h"
#include "forge/vision_registry.h"

// Engine headers
#include "forge/engines/deepseek_engine.h"
#include "forge/engines/llama_engine.h"
#include "forge/engines/qwen35_engine.h"
#include "forge/engines/transformer_engine.h"

// Model loader headers
#include "forge/gguf_model.h"
#include "forge/model_loader.h"
#include "forge/ninf_model.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

using namespace forge;

// ============================================================================
// Global state
// ============================================================================

volatile bool g_interrupted = false;

static void signal_handler(int sig) {
    (void)sig;
    g_interrupted = true;
}

// ============================================================================
// Engine & loader registration
// ============================================================================

static void ensure_engines_registered() {
    // Force arch_registrations.cpp to be linked (contains all static auto-registrations).
    (void)forge::_arch_registrations_linked;
    // Trigger static initializers by accessing the registry.
    (void)EngineRegistry::instance().registered_archs();
}

static void ensure_loaders_registered() {
    static bool registered = false;
    if (registered)
        return;
    registered = true;

    auto& reg = ModelLoaderRegistry::instance();
    reg.register_loader(
        "ninf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<NinfModel>(); });
    reg.register_loader(
        "gguf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<GgufModel>(); });
}

// ============================================================================
// Utility functions
// ============================================================================

std::string format_bytes(size_t bytes) {
    if (bytes < 1024)
        return std::to_string(bytes) + " B";
    if (bytes < 1024 * 1024)
        return std::to_string(bytes / 1024) + " KB";
    if (bytes < 1024ULL * 1024 * 1024)
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    return std::to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
}

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

void print_logo() {
    std::cout << "\n";
    std::cout << "  ______                   \n";
    std::cout << " |  ___|                  \n";
    std::cout << " | |_ ___  _ __ __ _  ___ \n";
    std::cout << " |  _/ _ \\| '__/ _` |/ _ \\ \n";
    std::cout << " | || (_) | | | (_| |  __/ \n";
    std::cout << " \\_| \\___/|_|  \\__, |\\___| \n";
    std::cout << "                __/ |     \n";
    std::cout << "               |___/      \n";
}

// ============================================================================
// Model info display
// ============================================================================

void print_model_info(const Model& model, const Tokenizer& tokenizer) {
    const auto& cfg = model.config();

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║               Forge - Model Info                ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "  File path:       " << model.path() << "\n";
    std::cout << "  File format:     " << model.format() << "\n";
    std::cout << "  Architecture:    " << cfg.arch_type << "\n";
    std::cout << "\n";

    std::cout << "  ── Model Structure ──────────────────────────────\n";
    std::cout << "  Vocab size:      " << cfg.vocab_size << "\n";
    std::cout << "  Hidden dim:      " << cfg.hidden_dim << "\n";
    std::cout << "  Intermediate:    " << cfg.intermediate_dim << "\n";
    std::cout << "  Layers:          " << cfg.num_layers << "\n";
    std::cout << "  Attention heads: " << cfg.num_heads << "\n";
    std::cout << "  KV heads:        " << cfg.num_kv_heads << (cfg.use_gqa ? " (GQA)" : "") << "\n";
    std::cout << "  Head dim:        " << cfg.head_dim << "\n";
    std::cout << "  Max seq length:  " << cfg.max_seq_len << "\n";
    std::cout << "\n";

    std::cout << "  ── RoPE ─────────────────────────────────────────\n";
    std::cout << "  RoPE theta:      " << cfg.rope_theta << "\n";
    std::cout << "  RoPE type:       ";
    switch (cfg.rope_type) {
    case RopeType::None:
        std::cout << "None";
        break;
    case RopeType::Standard:
        std::cout << "Standard";
        break;
    case RopeType::LinearScaling:
        std::cout << "Linear scaling (factor=" << cfg.rope_scaling_factor << ")";
        break;
    case RopeType::NTK_Scaled:
        std::cout << "NTK scaled";
        break;
    }
    std::cout << "\n";
    if (cfg.use_mrope) {
        std::cout << "  MRoPE:           Enabled (dim_count=" << cfg.rope_dimension_count << ")\n";
    }
    std::cout << "\n";

    std::cout << "  ── Normalization & Activation ───────────────────\n";
    std::cout << "  Norm:            "
              << (cfg.norm_type == NormType::RMSNorm ? "RMSNorm" : "LayerNorm")
              << " (eps=" << cfg.rms_norm_eps << ")\n";
    std::cout << "  FFN activation:  ";
    switch (cfg.ffn_activation) {
    case ActivationType::SiLU_GELU:
        std::cout << "SiLU+GELU (SwiGLU)";
        break;
    case ActivationType::GELU:
        std::cout << "GELU";
        break;
    case ActivationType::ReLU:
        std::cout << "ReLU";
        break;
    }
    std::cout << "\n";
    std::cout << "  Tie embeddings:  " << (cfg.tie_embeddings ? "Yes" : "No") << "\n";
    std::cout << "\n";

    if (cfg.use_ssm) {
        std::cout << "  ── SSM/Mamba Hybrid ────────────────────────────\n";
        std::cout << "  SSM groups:      " << cfg.ssm_group_count << "\n";
        std::cout << "  SSM time rank:   " << cfg.ssm_time_step_rank << "\n";
        std::cout << "  SSM inner size:  " << cfg.ssm_inner_size << "\n";
        std::cout << "  SSM state size:  " << cfg.ssm_state_size << "\n";
        std::cout << "  SSM conv kernel: " << cfg.ssm_conv_kernel << "\n";
        std::cout << "  Full attn intvl: " << cfg.full_attention_interval << "\n";
        std::cout << "\n";
    }

    if (cfg.n_routed_experts > 0) {
        std::cout << "  ── MoE Experts ──────────────────────────────────\n";
        std::cout << "  Routed experts:  " << cfg.n_routed_experts << "\n";
        std::cout << "  Shared experts:  " << cfg.n_shared_experts << "\n";
        std::cout << "  Experts/token:   " << cfg.num_expert_per_tok << "\n";
        std::cout << "\n";
    }

    if (cfg.use_mla) {
        std::cout << "  ── MLA (Multi-head Latent Attention) ────────────\n";
        std::cout << "  KV LoRA rank:    " << cfg.kv_lora_rank << "\n";
        std::cout << "  Q LoRA rank:     " << cfg.q_lora_rank << "\n";
        std::cout << "\n";
    }

    std::cout << "  ── Tokenizer ────────────────────────────────────\n";
    std::cout << "  Vocab size:      " << tokenizer.vocab_size() << "\n";
    std::cout << "  Model type:      "
              << (tokenizer.model_type() == TokenizerModelType::SPM ? "SPM" : "BPE") << "\n";
    std::cout << "  BOS ID:          " << tokenizer.bos_token_id() << "\n";
    std::cout << "  EOS ID:          " << tokenizer.eos_token_id() << "\n";
    std::cout << "  PAD ID:          " << tokenizer.pad_token_id() << "\n";
    std::cout << "\n";

    std::cout << "  ── Weights ──────────────────────────────────────\n";
    std::cout << "  Num weights:     " << model.weights().size() << "\n";
    std::cout << "  Total size:      " << format_bytes(model.weights().total_bytes()) << "\n";
    std::cout << "\n";
}

// ============================================================================
// Performance benchmark
// ============================================================================

void run_benchmark(InferenceContext& ctx, const Tokenizer& tokenizer, int n_gpu_layers) {
    const auto& cfg = ctx.model().config();

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║                Forge - Benchmark                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "  Model: " << cfg.arch_type << " (" << cfg.num_layers << " layers, "
              << cfg.hidden_dim << "d)\n";
    std::cout << "  GPU layers: " << n_gpu_layers << "\n";
    std::cout << "\n";

    std::cout << "  Warming up...\n";
    ctx.warmup();
    std::cout << "  Warmup done\n\n";

    std::vector<int> prompt_sizes = {32, 128, 512};
    for (int ps : prompt_sizes) {
        std::vector<int32_t> dummy_tokens(ps, 0);
        ctx.reset_kv_cache();

        auto input_ids =
            std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{ps}, DeviceType::CPU);
        std::memcpy(input_ids->data(), dummy_tokens.data(), ps * sizeof(int32_t));
        if (ctx.device() == DeviceType::CUDA) {
            input_ids->to_device(DeviceType::CUDA);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 3; ++i) {
            ctx.reset_kv_cache();
            ctx.engine()->forward_request(ForwardRequest::from_ids(input_ids, 0));
        }
#ifdef USE_CUDA
        if (ctx.device() == DeviceType::CUDA)
            cudaDeviceSynchronize();
#endif
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 3.0;
        double tok_per_s = ps / (ms / 1000.0);
        printf("  Prefill %4d tokens: %7.2f ms  (%8.1f tok/s)\n", ps, ms, tok_per_s);
    }

    ctx.reset_kv_cache();
    {
        auto input_ids =
            std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1}, DeviceType::CPU);
        *static_cast<int32_t*>(input_ids->data()) = 0;
        if (ctx.device() == DeviceType::CUDA) {
            input_ids->to_device(DeviceType::CUDA);
        }
        ctx.engine()->forward_request(ForwardRequest::from_ids(input_ids, 0));
    }

    int num_decode_steps = 128;
    forge::PerfProfiler::instance().enable();
    forge::PerfProfiler::instance().reset();
    // FORGE_PERF_DECODE=1: switch the profiler to CUDA-event timing so the
    // dump below reflects per-op device-stream time (host wall mis-attributes
    // async queue drains to whichever op happens to block the pipeline).
    const bool perf_decode = std::getenv("FORGE_PERF_DECODE") != nullptr;
    if (perf_decode)
        forge::PerfProfiler::instance().set_use_cuda_events(true);
    // FORGE_BENCH_PIPELINE=1: feed every step from a pre-staged device tensor
    // (same token id, no writes) so no per-step sync H2D drains the pipeline.
    // Wall/step then approaches pure device time and CUDA-event scopes lose
    // the idle-gap inflation, isolating real kernel execution from queue
    // drain artifacts.
    const bool pipeline = std::getenv("FORGE_BENCH_PIPELINE") != nullptr;
    std::shared_ptr<Tensor> pipe_ids;
    if (pipeline && ctx.device() == DeviceType::CUDA) {
        pipe_ids =
            std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1}, DeviceType::CUDA);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_decode_steps; ++i) {
        if (pipe_ids) {
            ctx.engine()->forward_request(ForwardRequest::from_ids(pipe_ids, i + 1));
            continue;
        }
        auto input_ids =
            std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1}, DeviceType::CPU);
        *static_cast<int32_t*>(input_ids->data()) = 0;
        if (ctx.device() == DeviceType::CUDA) {
            input_ids->to_device(DeviceType::CUDA);
        }
        ctx.engine()->forward_request(ForwardRequest::from_ids(input_ids, i + 1));
    }
#ifdef USE_CUDA
    if (ctx.device() == DeviceType::CUDA)
        cudaDeviceSynchronize();
#endif
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double tok_per_s = num_decode_steps / (ms / 1000.0);
    printf("  Decode  %4d tokens: %7.2f ms  (%8.1f tok/s)\n", num_decode_steps, ms, tok_per_s);

    // Skip print_summary under perf_decode: the dump below supersedes it, and
    // print_summary's merge is not idempotent (calling both would double-count
    // every record).
    if (!perf_decode)
        forge::PerfProfiler::instance().print_summary();

    if (perf_decode) {
        auto& prof = forge::PerfProfiler::instance();
#ifdef USE_CUDA
        prof.flush_deferred();
#endif
        forge::dump_phase_profile("DECODE-ONLY", ms, num_decode_steps, prof.summary());
    }

    std::cout << "\n  Benchmark complete\n";
}

// ============================================================================
// Multimodal image loading
// ============================================================================

static std::vector<uint8_t> load_image_ppm(const std::string& path, int& width, int& height) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open image file: " << path << "\n";
        return {};
    }

    std::string magic;
    file >> magic;

    if (magic != "P6" && magic != "P3") {
        std::cerr << "Error: Only PPM format (P3/P6) supported. Convert with: convert input.jpg "
                     "output.ppm\n";
        return {};
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        break;
    }

    std::istringstream iss(line);
    iss >> width >> height;
    if (width <= 0 || height <= 0) {
        file >> width >> height;
    }

    int maxval;
    file >> maxval;
    file.get();

    std::vector<uint8_t> pixels(width * height * 3);

    if (magic == "P6") {
        file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
    } else {
        for (size_t i = 0; i < pixels.size(); ++i) {
            int val;
            file >> val;
            pixels[i] = static_cast<uint8_t>(val);
        }
    }

    return pixels;
}

std::vector<float> encode_image(VisionEncoder& vision, const std::string& image_path,
                                int& num_tokens) {
    int width, height;
    auto pixels = load_image_ppm(image_path, width, height);
    if (pixels.empty()) {
        return {};
    }

    std::vector<float> rgb(width * height * 3);
    for (size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = static_cast<float>(pixels[i]);
    }

    auto embeddings = vision.encode(rgb.data(), width, height, 3);
    num_tokens = static_cast<int>(embeddings.size()) / vision.config().projection_dim;
    return embeddings;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (args.common.model_path.empty()) {
        std::cerr << "Error: Please specify a model path (-m, --model)\n";
        std::cerr << "Use --help for usage information\n";
        return 1;
    }

    Logger::instance().set_level(static_cast<LogLevel>(args.verbose));

    ensure_engines_registered();
    ensure_loaders_registered();

    // ---- Load tokenizer ----
    auto t0 = std::chrono::high_resolution_clock::now();
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(args.common.model_path)) {
        std::cerr << "Error: Failed to load tokenizer from: " << args.common.model_path << "\n";
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double tok_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Tokenizer loaded: vocab_size=" << tokenizer.vocab_size()
              << ", type=" << (tokenizer.model_type() == TokenizerModelType::SPM ? "SPM" : "BPE")
              << ", bos=" << tokenizer.bos_token_id() << ", eos=" << tokenizer.eos_token_id()
              << " [" << tok_ms << " ms]\n";

    // ---- Load model ----
    auto t2 = std::chrono::high_resolution_clock::now();
    Model model;

    // Apply memory tuning options before loading
    if (!args.common.load_mode.empty()) {
        model.set_load_mode(args.common.load_mode);
    }
    model.set_offload_embedding(args.common.offload_embedding);
    model.set_prefetch(args.common.prefetch);

    // --info only prints metadata, so it must never offload weights. With the
    // default -ngl -1 (all layers on GPU) a model larger than VRAM died here
    // ("CUDA malloc failed" -> uncaught -> core dump) before printing anything.
    const int load_gpu_layers = args.info_only ? 0 : args.common.n_gpu_layers;
    bool load_ok = model.load(args.common.model_path, load_gpu_layers);
    if (!load_ok) {
        std::cerr << "Error: Failed to load model: " << args.common.model_path << "\n";
        return 1;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double model_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    const auto& cfg = model.config();
    std::cout << "Model loaded: arch=" << cfg.arch_type << ", layers=" << cfg.num_layers
              << ", hidden=" << cfg.hidden_dim << ", heads=" << cfg.num_heads << " [" << model_ms
              << " ms]\n";


    // ---- Model info only ----
    if (args.info_only) {
        print_model_info(model, tokenizer);
        return 0;
    }

    // ---- Load vision encoder (multimodal) ----
    std::unique_ptr<VisionEncoder> vision;
    if (!args.mmproj_path.empty()) {
        // Force-link vision registry (auto-registrations for siglip, etc.)
        (void)forge::_vision_registrations_linked;
        (void)VisionEncoderRegistry::instance().registered_encoders();

        if (!model.load_vision_weights(args.mmproj_path, DeviceType::CPU)) {
            std::cerr << "Warning: Failed to load vision encoder: " << args.mmproj_path << "\n";
        } else {
            VisionConfig vis_cfg;
            std::string vis_arch = "siglip";
            auto loader = ModelLoaderRegistry::instance().create_loader(args.mmproj_path);
            if (loader && loader->load(args.mmproj_path)) {
                vis_arch = detect_vision_arch(*loader);
                vis_cfg = VisionConfigParserRegistry::instance().parse(*loader, vis_arch);
                loader->close();
            }
            vis_cfg.projection_dim = cfg.hidden_dim;

            vision = VisionEncoderRegistry::instance().create(vis_arch);
            if (!vision) {
                std::cerr << "Warning: No vision encoder registered for arch '" << vis_arch << "'\n";
            } else if (!vision->init(model.weights(), vis_cfg)) {
                std::cerr << "Warning: Vision encoder initialization failed\n";
                vision.reset();
            } else {
                std::cout << "Vision encoder loaded: arch=" << vis_arch
                          << ", image_size=" << vis_cfg.image_size
                          << ", patches=" << vis_cfg.patch_size
                          << ", blocks=" << vis_cfg.block_count << "\n";
            }
        }
    }

    // ---- Create inference context ----
    auto t4 = std::chrono::high_resolution_clock::now();

    InferenceContext ctx(model);
    // CPU threads are tuned later (after the engine reports its real backend),
    // because some engines (bailingmoe3) cannot offload to GPU and must still
    // get properly auto-detected threads even when -ngl -1 was requested.
    if (args.common.paged_kv) {
        ctx.params_mut().kv_storage_mode = KVStorageMode::Paged;
        std::cout << "Paged KV cache storage enabled\n";
    }
    auto engine = EngineRegistry::instance().create(cfg.arch_type, model, ctx);
    if (!engine) {
        std::cerr << "Error: No matching engine for architecture: " << cfg.arch_type << "\n";
        std::cerr << "Available engines: ";
        for (const auto& a : EngineRegistry::instance().registered_archs()) {
            std::cerr << a << " ";
        }
        std::cerr << "\n";
        return 1;
    }

    auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine.get());

    // One shared application of every common flag (model, -ngl, -c, -t, the KV
    // dtype pair, --cuda-graph, offload switches). forge-server calls the same
    // function, so the two can no longer disagree about a default or drop a flag.
    {
        std::string err;
        if (!forge::apply_common_args(args.common, nullptr, ctx, tfm_eng, &err)) {
            std::cerr << err << "\n";
            return 1;
        }
    }
    if (tfm_eng) {
        if (args.common.cuda_graph && tfm_eng->uses_gpu()) {
            tfm_eng->set_use_graph(true);
            tfm_eng->set_cuda_graph_enabled(true);
        }
        // -b/--batch-size was parsed but never read, so it did nothing. Wire it
        // to the forward-batch budget it always claimed to control.
        if (args.batch_size > 0) ctx.params_mut().n_batch = args.batch_size;
        if (args.common.threads > 0) {
            std::cout << "CPU threads set to: " << args.common.threads << "\n";
        } else if (args.common.n_gpu_layers < cfg.num_layers) {
            std::cout << "CPU threads auto-detected (hybrid offload): "
                      << ctx.params().n_threads << "\n";
        }
        if (args.expert_stats || args.expert_paging) {
            tfm_eng->set_expert_paging(args.expert_paging);
            if (args.expert_paging) {
                std::cerr
                    << "Warning: --expert-paging is EXPERIMENTAL and currently only pays off when\n"
                       "         the MoE expert weights do NOT fit in VRAM. If the model already\n"
                       "         fits, paging adds host<->device transfer cost and is SLOWER, and\n"
                       "         VRAM does not shrink because freed expert blocks stay in the CUDA\n"
                       "         pool. Measured (RTX 4050 6GB, Phi-mini-MoE 2.75GB, -ngl 99):\n"
                       "           baseline            20.5 tok/s, 4245 MiB\n"
                       "           paging budget 1GB   10.5 tok/s, 5816 MiB\n"
                       "         Output stays correct in both cases. Use only for MoE models that\n"
                       "         exceed VRAM, and prefer --expert-stats alone for router analysis.\n";
            }
            if (args.expert_budget_mb > 0) {
                tfm_eng->set_expert_budget_bytes(
                    static_cast<int64_t>(args.expert_budget_mb) * 1024 * 1024);
            }
        }
    }

    // ---- KVMem tiered KV: cap the GPU page pool BEFORE init_kv_cache ----
    // The working-set manager itself is pure policy, but the pool cap is what
    // turns the paged arena into a bounded working set: PagedKVStorage::init()
    // reads it while sizing the page pools, so it has to be set before
    // ctx.init_kv_cache() runs. The manager is attached afterwards, once the
    // pool geometry is known.
    bool kvmem_active = false;
    if (args.common.kvmem) {
        auto* kv_mem = tfm_eng ? tfm_eng->kv_memory() : nullptr;
        if (!tfm_eng || !kv_mem) {
            std::cerr << "Warning: --kvmem needs a transformer engine with KV memory; "
                         "tiered KV disabled\n";
        } else if (!kv_mem->is_paged()) {
            std::cerr << "Warning: --kvmem needs paged KV storage; tiered KV disabled\n";
        } else if (cfg.n_swa > 0) {
            // PagedKVStorage::kvmem_supported() rejects mixed policies: pages can
            // only be archived uniformly when every layer is Full-attention.
            std::cerr << "Warning: --kvmem needs all-Full attention layers; model has "
                      << cfg.n_swa << " sliding-window layer(s); tiered KV disabled\n";
        } else {
            // 0 = keep the device-sized pool; the budget then falls back to it.
            if (args.common.kvmem_budget > 0)
                kv_mem->storage().set_pool_token_cap(args.common.kvmem_budget);
            kvmem_active = true;
        }
    }

    ctx.set_engine(std::move(engine));

    ctx.params_mut().offload_kqv = args.common.offload_kqv;
    ctx.params_mut().offload_embedding = model.offload_embedding();
    // Publish the speculative config BEFORE init_kv_cache(). The KV cache is sized
    // here, and TransformerEngine::init_kv_cache reserves headroom for the draft
    // model when it sees one configured -- otherwise the cache swallows the VRAM
    // the drafter needs and the matmul planner silently drops to MMQ. The config
    // used to be applied later, in generate_streaming, which is too late.
    if (args.spec.enabled)
        ctx.params_mut().speculative_config = args.spec;
    ctx.init_kv_cache();
    if (tfm_eng) {
        // In paged mode the contiguous KVCache is intentionally left
        // uninitialized (transformer_engine.cpp skips kv_cache_.init_quantized
        // when paged), so reading kv_cache() would report fp32/0B/0. Report the
        // active backend instead: the PagedKVStorage when paged, else the
        // contiguous KVCache.
        auto kv_dtype_to_name = [](KVCacheDType t) -> const char* {
            switch (t) {
            case KVCacheDType::F16:      return "f16";
            case KVCacheDType::Q8_0:     return "q8_0";
            case KVCacheDType::Q4_0:     return "q4_0";
            case KVCacheDType::FP8_E4M3: return "fp8_e4m3";
            case KVCacheDType::FP8_E5M2: return "fp8_e5m2";
            default:                     return "fp32";
            }
        };
        auto* kv_mem = tfm_eng->kv_memory();
        // Report K and V separately. With -ctk/-ctv they can differ, and a line
        // that only named one of them made an asymmetric config indistinguishable
        // from a symmetric one. llama.cpp does the same (llama-kv-cache.cpp:301).
        const auto type_str = [&](const KVCacheTypeConfig& c) {
            const char* k = kv_cache_dtype_name(c.type_k);
            const char* v = kv_cache_dtype_name(c.type_v);
            return c.type_k == c.type_v ? std::string(k)
                                       : std::string(k) + "/" + v + " (K/V)";
        };
        if (kv_mem && kv_mem->storage().is_paged()) {
            const auto& st = static_cast<const PagedKVStorage&>(kv_mem->storage());
            KVCacheTypeConfig c{st.kv_type_k(), st.kv_type_v()};
            std::cout << "KV Cache (paged): dtype=" << type_str(c)
                      << ", size=" << format_bytes(st.nbytes())
                      << ", max_seq=" << st.max_seq_len()
                      << " (pool_tokens=" << st.max_pool_tokens() << ")\n";
        } else {
            const KVCache* cache = tfm_eng->kv_cache();
            std::cout << "KV Cache: dtype=" << type_str(cache->kv_config())
                      << ", size=" << format_bytes(cache->nbytes())
                      << ", max_seq=" << cache->max_seq_len() << "\n";
        }
    }

    // ---- Attach the KVMem working-set manager ----
    // Built after init_kv_cache so it sees the pool geometry the storage
    // actually settled on (page_size + pool_tokens) — kvmem_supported() also
    // needs the page pools to exist, so this is the earliest point the gate
    // can run. From here the generator reads window/rope metadata off it and
    // the logical context (-c) may exceed the GPU pool.
    if (kvmem_active && tfm_eng) {
        auto* kv_mem = tfm_eng->kv_memory();
        if (!kv_mem->storage().kvmem_supported()) {
            std::cerr << "Warning: --kvmem unsupported by this KV storage "
                         "(mixed layer policies); tiered KV disabled\n";
        } else {
            const int page_size = kv_mem->storage().page_size();
            const int pool_tokens = kv_mem->max_pool_tokens();
            auto ws = std::make_unique<KVWorkingSetManager>(args.common.kvem_config(), page_size,
                                                           pool_tokens);
            std::cout << "KVMem tiered KV enabled: budget=" << ws->budget()
                      << ", sink=" << ws->sink_tokens()
                      << ", gen_reserve=" << ws->gen_reserve()
                      << ", retrieval=" << args.common.kvmem_retrieval << "\n"
                      << "  logical ctx=" << kv_mem->max_seq_len()
                      << " tokens, gpu pool=" << pool_tokens
                      << " tokens, page=" << page_size << "\n";
            kv_mem->set_working_set(std::move(ws));
        }
    }

    if (model.device() == DeviceType::CUDA) {
        std::cout << "CUDA warmup...\n";
        try {
            ctx.warmup();
            std::cout << "CUDA warmup done\n";
        } catch (const std::runtime_error& e) {
            std::cout << "Warmup skipped (" << e.what() << ")\n";
        }
    }

    auto t5 = std::chrono::high_resolution_clock::now();
    double ctx_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double total_ms = std::chrono::duration<double, std::milli>(t5 - t0).count();
    std::cout << "Startup total: " << total_ms << " ms"
              << " (tokenizer=" << tok_ms << "ms"
              << ", model=" << model_ms << "ms"
              << ", ctx=" << ctx_ms << "ms)\n";

    // ---- Benchmark mode ----
    if (args.bench) {
        run_benchmark(ctx, tokenizer, args.common.n_gpu_layers);
        return 0;
    }

    // ---- Text completion mode ----
    if (!args.interactive) {
        // Respect the model's tokenizer metadata (tokenizer.ggml.add_bos_token /
        // add_eos_token) instead of unconditionally prepending BOS.
        auto prompt_ids = tokenizer.encode(args.prompt, tokenizer.add_bos_token(),
                                           tokenizer.add_eos_token());
        std::cout << "\n";

        GenerationStats stats;
        if (args.no_stream) {
            stats = generate_batch(ctx, tokenizer, prompt_ids, args.n_predict, args.temperature,
                                   args.top_k, args.top_p, args.repeat_penalty, args.repeat_last_n,
                                   !args.no_sample, args.seed, tokenizer.eos_token_id(), args.spec);
        } else {
            stats = generate_streaming(ctx, tokenizer, prompt_ids, args.n_predict, args.temperature,
                                       args.top_k, args.top_p, args.repeat_penalty, args.repeat_last_n,
                                       !args.no_sample, args.seed, tokenizer.eos_token_id(), args.spec);
        }

        std::cout << "\n\n";
        forge_print_stats_line(stats);

        // Print speculative decoding stats if requested. Not gated on
        // n_spec_steps: a run where every round fell back is precisely the case
        // worth seeing, and it used to print nothing at all.
        if (args.spec.print_stats && args.spec.enabled &&
            (stats.spec.n_spec_steps > 0 || stats.spec.n_fallback_steps > 0)) {
            std::cout << "\n  --- Speculative Decoding Stats ---\n";
            std::cout << "  Spec steps:      " << stats.spec.n_spec_steps << "\n";
            std::cout << "  Fallback steps:  " << stats.spec.n_fallback_steps << "\n";
            // Attribute the fallbacks: the total alone cannot distinguish "the
            // drafter is wrong" from "the drafter was never asked for anything".
            if (stats.spec.n_fallback_steps > 0) {
                std::cout << "    no-draft:      " << stats.spec.n_fb_no_draft << "\n";
                std::cout << "    min-acc floor: " << stats.spec.n_fb_min_acc << "\n";
                std::cout << "    verify failed: " << stats.spec.n_fb_verify_fail << "\n";
                std::cout << "    empty result:  " << stats.spec.n_fb_empty << "\n";
                std::cout << "    guard/other:   " << stats.spec.n_fb_guard << "\n";
            }
            std::cout << "  Draft tokens:    " << stats.spec.n_draft_tokens << "\n";
            std::cout << "  Accepted tokens: " << stats.spec.n_accepted_tokens << "\n";
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "  Acceptance rate: " << (stats.spec.acceptance_rate() * 100.0) << "%\n";
            std::cout << "  Tokens/step:     " << stats.spec.tokens_per_step() << "\n";
            // Acceptance by proposed position. A steep falloff is normal and
            // expected; a low value at position 0 means the drafter's first guess
            // is wrong, which is a conditioning problem rather than a depth one.
            {
                static const char* kPosNames[] = {"", "0", "1", "2", "3", "4", "5"};
                bool any = false;
                std::cout << "  Acceptance by position:";
                for (int p = 0; p < 6; ++p) {
                    const double a = stats.spec.acceptance_at(p);
                    if (a < 0.0) continue;
                    any = true;
                    std::cout << " " << kPosNames[p + 1] << "=" << (a * 100.0) << "%";
                }
                if (!any) std::cout << " (none)";
                std::cout << "\n";
            }
            if (stats.spec.n_rank_samples > 0) {
                std::cout << "  Draft rank in target dist (at 1st rejection):";
                static const char* kRankNames[] = {"#1", "#2", "#3", "#4", "#5", "5+"};
                for (int b = 0; b < 6; ++b) {
                    if (stats.spec.n_rank_hist[b] == 0) continue;
                    std::cout << " " << kRankNames[b] << "x" << stats.spec.n_rank_hist[b];
                }
                std::cout << std::fixed << std::setprecision(3)
                          << "  (mean target prob on draft "
                          << (stats.spec.rank_prob_sum / stats.spec.n_rank_samples) << ")\n";
            }
            // Proposed length per round: a single dominant bucket means the
            // draft width is effectively constant.
            {
                static const char* kLenNames[] = {"0", "1", "2", "3", "4-5", "6-7", "8+"};
                std::cout << "  Rounds by draft length:";
                for (int b = 0; b < 7; ++b) {
                    if (stats.spec.n_rounds_by_len[b] == 0) continue;
                    std::cout << " " << kLenNames[b] << "x" << stats.spec.n_rounds_by_len[b];
                }
                std::cout << "\n";
            }
        }

        // Print MoE expert router-distribution stats if requested
        if (args.expert_stats && tfm_eng) {
            std::cout << "\n  --- MoE Expert Router Stats ---\n";
            std::cout << tfm_eng->expert_stats_report();
        }

        return 0;
    }

    // ---- Interactive chat mode ----
    // Reuse the fully initialized ctx (engine + KV cache sized with the full
    // VRAM budget) instead of letting interactive_chat build a second context,
    // which would get its KV max_seq clamped to a tiny value (256) and crash
    // with "KVCache::update: cache overflow" once generation exceeds it.
    interactive_chat(ctx, model, tokenizer, vision.get(), args);

    return 0;
}
