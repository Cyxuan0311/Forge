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

#include "cli_common.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <csignal>
#include <sstream>
#include <cstdio>

// Forge headers
#include "forge/model.h"
#include "forge/tokenizer.h"
#include "forge/context.h"
#include "forge/engine.h"
#include "forge/logger.h"
#include "forge/types.h"
#include "forge/vision_encoder.h"
#include "forge/kv_cache.h"
#include "forge/tensor.h"

// Engine headers
#include "forge/engines/transformer_engine.h"
#include "forge/engines/llama_engine.h"
#include "forge/engines/deepseek_engine.h"
#include "forge/engines/qwen35_engine.h"

// Model loader headers
#include "forge/model_loader.h"
#include "forge/ninf_model.h"
#include "forge/gguf_model.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
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
    // Static registration via EngineAutoRegister may not work in all link scenarios.
    // Explicitly register engines as a reliable fallback.
    static bool registered = false;
    if (registered) return;
    registered = true;

    auto& reg = EngineRegistry::instance();
    (void)reg.registered_archs();

    if (!reg.has("llama")) {
        reg.register_engine("llama", [](Model& model, InferenceContext& ctx) {
            return std::make_unique<LlamaEngine>(model, ctx);
        });
    }
    if (!reg.has("mistral")) {
        reg.register_engine("mistral", [](Model& model, InferenceContext& ctx) {
            return std::make_unique<LlamaEngine>(model, ctx);
        });
    }
    if (!reg.has("qwen")) {
        reg.register_engine("qwen", [](Model& model, InferenceContext& ctx) {
            return std::make_unique<LlamaEngine>(model, ctx);
        });
    }
    if (!reg.has("qwen2")) {
        reg.register_engine("qwen2", [](Model& model, InferenceContext& ctx) {
            return std::make_unique<LlamaEngine>(model, ctx);
        });
    }
    if (!reg.has("yi")) {
        reg.register_engine("yi", [](Model& model, InferenceContext& ctx) {
            return std::make_unique<LlamaEngine>(model, ctx);
        });
    }
    if (!reg.has("deepseek")) {
        reg.register_engine("deepseek", [](Model& model, InferenceContext& ctx) {
            return std::make_unique<LlamaEngine>(model, ctx);
        });
    }
    if (!reg.has("deepseek_v2")) {
        reg.register_engine("deepseek_v2", [](Model& model, InferenceContext& ctx) {
            return std::make_unique<DeepSeekEngine>(model, ctx);
        });
    }
    if (!reg.has("deepseek_v3")) {
        reg.register_engine("deepseek_v3", [](Model& model, InferenceContext& ctx) {
            return std::make_unique<DeepSeekEngine>(model, ctx);
        });
    }
    if (!reg.has("qwen35")) {
        reg.register_engine("qwen35", [](Model& model, InferenceContext& ctx) {
            return std::make_unique<Qwen35Engine>(model, ctx);
        });
    }
}

static void ensure_loaders_registered() {
    static bool registered = false;
    if (registered) return;
    registered = true;

    auto& reg = ModelLoaderRegistry::instance();
    reg.register_loader("ninf", []() -> std::unique_ptr<ModelLoader> {
        return std::make_unique<NinfModel>();
    });
    reg.register_loader("gguf", []() -> std::unique_ptr<ModelLoader> {
        return std::make_unique<GgufModel>();
    });
}

// ============================================================================
// Utility functions
// ============================================================================

std::string format_bytes(size_t bytes) {
    if (bytes < 1024) return std::to_string(bytes) + " B";
    if (bytes < 1024 * 1024) return std::to_string(bytes / 1024) + " KB";
    if (bytes < 1024ULL * 1024 * 1024) return std::to_string(bytes / (1024 * 1024)) + " MB";
    return std::to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
}

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

void print_logo() {
    std::cout << "\n";
    std::cout << "                             _       ____         \n";
    std::cout << "    ____  ____ _____  ____  (_)___  / __/__  _____ \n";
    std::cout << "   / __ \\/ __ `/ __ \\/ __ \\/ / __ \\/ /_/ _ \\/ ___/ \n";
    std::cout << "  / / / / /_/ / / / / /_/ / / / / / __/  __/ /     \n";
    std::cout << " /_/ /_/\\__,_/_/ /_/\\____/_/_/ /_/_/  \\___/_/      \n";
    std::cout << "                                                    \n";
}

// ============================================================================
// Model info display
// ============================================================================

void print_model_info(const Model& model, const Tokenizer& tokenizer) {
    const auto& cfg = model.config();

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║            Forge - Model Info               ║\n";
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
    std::cout << "  KV heads:        " << cfg.num_kv_heads
              << (cfg.use_gqa ? " (GQA)" : "") << "\n";
    std::cout << "  Head dim:        " << cfg.head_dim << "\n";
    std::cout << "  Max seq length:  " << cfg.max_seq_len << "\n";
    std::cout << "\n";

    std::cout << "  ── RoPE ─────────────────────────────────────────\n";
    std::cout << "  RoPE theta:      " << cfg.rope_theta << "\n";
    std::cout << "  RoPE type:       ";
    switch (cfg.rope_type) {
        case RopeType::None: std::cout << "None"; break;
        case RopeType::Standard: std::cout << "Standard"; break;
        case RopeType::LinearScaling: std::cout << "Linear scaling (factor=" << cfg.rope_scaling_factor << ")"; break;
        case RopeType::NTK_Scaled: std::cout << "NTK scaled"; break;
    }
    std::cout << "\n";
    if (cfg.use_mrope) {
        std::cout << "  MRoPE:           Enabled (dim_count=" << cfg.rope_dimension_count << ")\n";
    }
    std::cout << "\n";

    std::cout << "  ── Normalization & Activation ───────────────────\n";
    std::cout << "  Norm:            " << (cfg.norm_type == NormType::RMSNorm ? "RMSNorm" : "LayerNorm")
              << " (eps=" << cfg.rms_norm_eps << ")\n";
    std::cout << "  FFN activation:  ";
    switch (cfg.ffn_activation) {
        case ActivationType::SiLU_GELU: std::cout << "SiLU+GELU (SwiGLU)"; break;
        case ActivationType::GELU: std::cout << "GELU"; break;
        case ActivationType::ReLU: std::cout << "ReLU"; break;
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
    std::cout << "  Model type:      " << (tokenizer.model_type() == TokenizerModelType::SPM ? "SPM" : "BPE") << "\n";
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
    std::cout << "║            Forge - Benchmark                ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "  Model: " << cfg.arch_type
              << " (" << cfg.num_layers << " layers, " << cfg.hidden_dim << "d)\n";
    std::cout << "  GPU layers: " << n_gpu_layers << "\n";
    std::cout << "\n";

    std::cout << "  Warming up...\n";
    ctx.warmup();
    std::cout << "  Warmup done\n\n";

    std::vector<int> prompt_sizes = {32, 128, 512};
    for (int ps : prompt_sizes) {
        std::vector<int32_t> dummy_tokens(ps, 0);
        ctx.reset_kv_cache();

        auto input_ids = std::make_shared<Tensor>(DataType::INT32,
            std::vector<int64_t>{ps}, DeviceType::CPU);
        std::memcpy(input_ids->data(), dummy_tokens.data(), ps * sizeof(int32_t));
        if (ctx.device() == DeviceType::CUDA) {
            input_ids->to_device(DeviceType::CUDA);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 3; ++i) {
            ctx.reset_kv_cache();
            ctx.engine()->forward(input_ids, 0);
        }
#ifdef USE_CUDA
        if (ctx.device() == DeviceType::CUDA) cudaDeviceSynchronize();
#endif
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 3.0;
        double tok_per_s = ps / (ms / 1000.0);
        printf("  Prefill %4d tokens: %7.2f ms  (%8.1f tok/s)\n", ps, ms, tok_per_s);
    }

    ctx.reset_kv_cache();
    {
        auto input_ids = std::make_shared<Tensor>(DataType::INT32,
            std::vector<int64_t>{1}, DeviceType::CPU);
        *static_cast<int32_t*>(input_ids->data()) = 0;
        if (ctx.device() == DeviceType::CUDA) {
            input_ids->to_device(DeviceType::CUDA);
        }
        ctx.engine()->forward(input_ids, 0);
    }

    int num_decode_steps = 128;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_decode_steps; ++i) {
        auto input_ids = std::make_shared<Tensor>(DataType::INT32,
            std::vector<int64_t>{1}, DeviceType::CPU);
        *static_cast<int32_t*>(input_ids->data()) = 0;
        if (ctx.device() == DeviceType::CUDA) {
            input_ids->to_device(DeviceType::CUDA);
        }
        ctx.engine()->forward(input_ids, i + 1);
    }
#ifdef USE_CUDA
    if (ctx.device() == DeviceType::CUDA) cudaDeviceSynchronize();
#endif
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double tok_per_s = num_decode_steps / (ms / 1000.0);
    printf("  Decode  %4d tokens: %7.2f ms  (%8.1f tok/s)\n", num_decode_steps, ms, tok_per_s);

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
        std::cerr << "Error: Only PPM format (P3/P6) supported. Convert with: convert input.jpg output.ppm\n";
        return {};
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
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

std::vector<float> encode_image(
    VisionEncoder& vision,
    const std::string& image_path,
    int& num_tokens)
{
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

    if (args.model_path.empty()) {
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
    if (!tokenizer.load_from_gguf(args.model_path)) {
        std::cerr << "Error: Failed to load tokenizer from: " << args.model_path << "\n";
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double tok_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Tokenizer loaded: vocab_size=" << tokenizer.vocab_size()
              << ", type=" << (tokenizer.model_type() == TokenizerModelType::SPM ? "SPM" : "BPE")
              << ", bos=" << tokenizer.bos_token_id()
              << ", eos=" << tokenizer.eos_token_id()
              << " [" << tok_ms << " ms]\n";

    // ---- Load model ----
    auto t2 = std::chrono::high_resolution_clock::now();
    Model model;
    DeviceType device = (args.n_gpu_layers == 0) ? DeviceType::CPU : DeviceType::CUDA;

    bool load_ok = model.load(args.model_path, device);
    if (!load_ok) {
        std::cerr << "Error: Failed to load model: " << args.model_path << "\n";
        return 1;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double model_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    const auto& cfg = model.config();
    std::cout << "Model loaded: arch=" << cfg.arch_type
              << ", layers=" << cfg.num_layers
              << ", hidden=" << cfg.hidden_dim
              << ", heads=" << cfg.num_heads
              << " [" << model_ms << " ms]\n";

    // ---- Model info only ----
    if (args.info_only) {
        print_model_info(model, tokenizer);
        return 0;
    }

    // ---- Load vision encoder (multimodal) ----
    std::unique_ptr<VisionEncoder> vision;
    if (!args.mmproj_path.empty()) {
        vision = std::make_unique<VisionEncoder>();
        if (!model.load_vision_weights(args.mmproj_path, DeviceType::CPU)) {
            std::cerr << "Warning: Failed to load vision encoder: " << args.mmproj_path << "\n";
            vision.reset();
        } else {
            VisionConfig vis_cfg;
            auto loader = ModelLoaderRegistry::instance().create_loader(args.mmproj_path);
            if (loader && loader->load(args.mmproj_path)) {
                vis_cfg.image_size = static_cast<int>(loader->get_metadata_int("clip.vision.image_size", 448));
                vis_cfg.patch_size = static_cast<int>(loader->get_metadata_int("clip.vision.patch_size", 14));
                vis_cfg.embedding_length = static_cast<int>(loader->get_metadata_int("v.embedding_length", 1152));
                vis_cfg.feed_forward_length = static_cast<int>(loader->get_metadata_int("v.feed_forward_length", 4304));
                vis_cfg.block_count = static_cast<int>(loader->get_metadata_int("v.block_count", 27));
                vis_cfg.head_count = static_cast<int>(loader->get_metadata_int("v.attention.head_count", 16));
                vis_cfg.projection_dim = cfg.hidden_dim;
                loader->close();
            }
            if (!vision->init(model.weights(), vis_cfg)) {
                std::cerr << "Warning: Vision encoder initialization failed\n";
                vision.reset();
            } else {
                std::cout << "Vision encoder loaded: image_size=" << vis_cfg.image_size
                          << ", patches=" << vis_cfg.patch_size
                          << ", blocks=" << vis_cfg.block_count << "\n";
            }
        }
    }

    // ---- Create inference context ----
    auto t4 = std::chrono::high_resolution_clock::now();

    InferenceContext ctx(model);
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
    if (tfm_eng) {
        KVCacheDType kv_dtype = KVCacheDType::FP32;
        if (args.kv_cache_dtype == "q4_0") kv_dtype = KVCacheDType::Q4_0;
        tfm_eng->set_kv_cache_dtype(kv_dtype);
        tfm_eng->set_gpu_layers(args.n_gpu_layers);
    }
    ctx.set_engine(std::move(engine));

    ctx.init_kv_cache();
    if (tfm_eng) {
        const auto& cache = tfm_eng->kv_cache();
        std::cout << "KV Cache: dtype=" << (cache.kv_dtype() == KVCacheDType::Q4_0 ? "q4_0" : "fp32")
                  << ", size=" << format_bytes(cache.nbytes())
                  << ", max_seq=" << cache.max_seq_len() << "\n";
    }

    if (device == DeviceType::CUDA) {
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
              << " (tokenizer=" << tok_ms << "ms, model=" << model_ms << "ms, ctx=" << ctx_ms << "ms)\n";

    // ---- Benchmark mode ----
    if (args.bench) {
        run_benchmark(ctx, tokenizer, args.n_gpu_layers);
        return 0;
    }

    // ---- Text completion mode ----
    if (!args.interactive) {
        auto prompt_ids = tokenizer.encode(args.prompt, true, false);
        std::cout << "\n";

        GenerationStats stats;
        if (args.no_stream) {
            stats = generate_batch(ctx, tokenizer, prompt_ids,
                args.n_predict, args.temperature, args.top_k, args.top_p,
                args.repeat_penalty, !args.no_sample, args.seed,
                tokenizer.eos_token_id());
        } else {
            stats = generate_streaming(ctx, tokenizer, prompt_ids,
                args.n_predict, args.temperature, args.top_k, args.top_p,
                args.repeat_penalty, !args.no_sample, args.seed,
                tokenizer.eos_token_id());
        }

        std::cout << "\n\n";
        if (stats.num_generated_tokens > 0) {
            double prompt_tok_s = stats.num_prompt_tokens / (stats.prompt_eval_ms / 1000.0);
            double gen_tok_s = stats.num_generated_tokens / (stats.elapsed_ms / 1000.0);
            printf("  [%d prompt, %.1f ms, %.1f tok/s | %d generated, %.1f ms, %.1f tok/s]\n",
                   stats.num_prompt_tokens, stats.prompt_eval_ms, prompt_tok_s,
                   stats.num_generated_tokens, stats.elapsed_ms, gen_tok_s);
        }

        return 0;
    }

    // ---- Interactive chat mode ----
    interactive_chat(model, tokenizer, vision.get(), args);

    return 0;
}
