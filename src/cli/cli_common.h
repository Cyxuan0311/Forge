/**
 * Forge CLI - Shared definitions
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "forge/chat_template.h"
#include "forge/kv_working_set.h"
#include "forge/prefix_cache.h"
#include "forge/speculative.h"

// Forward declarations
namespace forge {
class Tokenizer;
class InferenceContext;
class Model;
class VisionEncoder;
}  // namespace forge

// ============================================================================
// CLI arguments
// ============================================================================

struct CliArgs {
    // Basic
    std::string model_path;
    std::string prompt;
    std::string mmproj_path;
    std::string image_path;

    // Performance tuning
    int n_gpu_layers = -1;
    std::vector<int> gpu_layers_per_dev;   // -ngl N:M => [N, M] (Phase 5 multi-GPU)
    bool offload_kqv = true;  // true = KV on GPU when layers on GPU
    int threads = -1;
    int batch_size = 512;
    int n_ctx = 0;  // 0 = auto (model native length, clamped to fit memory); >0 = explicit KV cap
    int n_predict = -1;
    std::string kv_cache_dtype = "f16";  // fp32 | f16 | q8_0 | q4_0 | fp8_e4m3 | fp8_e5m2
    bool cuda_graph = false;  // Phase 10: CUDA Graph for decode
    bool paged_kv = false;    // Use paged KV cache storage (default: contiguous)

    // Memory / load tuning
    std::string load_mode;          // "mmap" (default), "mlock", "mmap_mlock"
    bool offload_embedding = true;  // true = token_embedding follows first layer device
    bool prefetch = true;           // warm page cache at load (--no-prefetch disables)

    // Sampling control
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;
    uint64_t seed = 0;
    bool no_sample = false;

    // Mode control
    bool interactive = false;
    bool no_stream = false;
    bool info_only = false;
    bool bench = false;

    // Chat template
    bool no_jinja = false;  // Force fallback to hardcoded templates

    // System prompt
    std::string system_prompt;

    // Verbosity
    int verbose = 1;

    // Speculative decoding
    forge::SpeculativeConfig spec;

    // MoE expert router statistics / partial-activation paging
    bool expert_stats = false;   // print per-expert activation distribution after generation
    bool expert_paging = false;  // enable per-expert weight movement (ExpertPageCache)
    int expert_budget_mb = 0;    // VRAM budget for resident experts, MB (0 = unbounded)

    // KVMem tiered KV working set (arXiv:2609.04852). --kvmem turns paged mode on
    // and attaches a tiered working set so the logical context (-c) can exceed the
    // GPU pool; the remaining knobs mirror kvmem-llama.cpp's flags. Requires
    // --paged-kv (or sets it itself).
    bool kvmem = false;         // enable tiered KV working set
    int kvmem_budget = 0;       // GPU resident budget (tokens); 0 = the paged pool cap
    int kvmem_sink = 256;       // tokens pinned resident forever at the prefix
    int kvmem_gen_reserve = 0;  // window slots reserved for this turn's output
    std::string kvmem_retrieval = "recent";  // foundational recall: recent | embed

    forge::KVemConfig kvem_config() const {
        forge::KVemConfig c;
        c.enabled = kvmem;
        c.budget_tokens = kvmem_budget;
        c.sink_tokens = kvmem_sink;
        // The window must not fill before the reply is done: a full window makes
        // the first decode step write past the pool ("page pool exhausted").
        // With no explicit --kvmem-gen-reserve, reserve this run's generation
        // budget (the -n default is 256 when -n is not given).
        c.gen_reserve = kvmem_gen_reserve > 0 ? kvmem_gen_reserve
                                             : (n_predict > 0 ? n_predict : 256);
        c.retrieval = kvmem_retrieval;
        return c;
    }
};

CliArgs parse_args(int argc, char** argv);

// ============================================================================
// Chat template
// ============================================================================

enum class ChatTemplateType {
    ChatML,    // Qwen, Yi: <|im_start|>...<|im_end|>
    DeepSeek,  // DeepSeek: <｜User｜><｜Assistant｜>
    Llama,     // Llama: [INST]...[/INST]
    Plain,     // No template, plain text
};

struct ChatMessage {
    std::string role;  // "system", "user", "assistant"
    std::string content;
};

ChatTemplateType detect_template_type(const forge::Tokenizer& tokenizer);

std::vector<int32_t> apply_chat_template(const forge::Tokenizer& tokenizer,
                                         const std::vector<ChatMessage>& messages,
                                         ChatTemplateType tmpl_type,
                                         bool add_generation_prompt = true);

// ============================================================================
// Generation stats
// ============================================================================

struct GenerationStats {
    int num_prompt_tokens = 0;
    int num_generated_tokens = 0;
    double elapsed_ms = 0;
    double prompt_eval_ms = 0;
    forge::SpeculativeStats spec;  // speculative decoding stats (zeroed when not enabled)
};

// ============================================================================
// Terminal color helpers
//   - 仅在输出为 TTY 且未设置 FORGE_NO_COLOR / NO_COLOR 时启用
//   - 输入 prompt 标记用蓝色, 其余字段用多种颜色区分
// ============================================================================

namespace cli_color {
inline bool enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* fc = std::getenv("FORGE_NO_COLOR");
        const char* nc = std::getenv("NO_COLOR");
        bool on = !(fc && fc[0] && std::strcmp(fc, "0") != 0);
        if (on && nc && nc[0])
            on = false;
        if (on) {
#ifdef _WIN32
            on = false;
#else
            on = ::isatty(fileno(stdout)) != 0;
#endif
        }
        cached = on ? 1 : 0;
    }
    return cached == 1;
}
inline std::string wrap(const char* code, const std::string& s) {
    if (!enabled())
        return s;
    return std::string("\x1b[") + code + "m" + s + "\x1b[0m";
}
inline std::string blue(const std::string& s)    { return wrap("1;34", s); }
inline std::string cyan(const std::string& s)    { return wrap("36", s); }
inline std::string green(const std::string& s)   { return wrap("32", s); }
inline std::string yellow(const std::string& s)  { return wrap("33", s); }
inline std::string magenta(const std::string& s) { return wrap("35", s); }
inline std::string red(const std::string& s)     { return wrap("1;31", s); }
inline std::string dim(const std::string& s)     { return wrap("2", s); }
inline std::string bold(const std::string& s)    { return wrap("1", s); }
}  // namespace cli_color

// 安全地格式化 tok/s: 分母为 0 时返回 "—" (避免 inf tok/s)
inline std::string forge_fmt_tok_s(double tokens, double ms) {
    char buf[32];
    if (ms <= 0.0) {
        std::snprintf(buf, sizeof(buf), "—");
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f", tokens / (ms / 1000.0));
    }
    return std::string(buf);
}

// 带颜色的生成统计行: 不同字段用不同颜色
inline void forge_print_stats_line(const GenerationStats& stats) {
    if (stats.num_generated_tokens <= 0)
        return;
    using namespace cli_color;
    std::string s;
    s += dim("  [");
    s += bold(cyan(std::to_string(stats.num_prompt_tokens)));
    s += cyan(" prompt");
    s += dim(", ");
    s += dim(std::to_string((long long)stats.prompt_eval_ms)) + " " + dim("ms");
    s += dim(", ");
    s += yellow(forge_fmt_tok_s(stats.num_prompt_tokens, stats.prompt_eval_ms)) + " " + dim("tok/s");
    s += dim(" | ");
    s += bold(green(std::to_string(stats.num_generated_tokens)));
    s += green(" generated");
    s += dim(", ");
    s += dim(std::to_string((long long)stats.elapsed_ms)) + " " + dim("ms");
    s += dim(", ");
    s += yellow(forge_fmt_tok_s(stats.num_generated_tokens, stats.elapsed_ms)) + " " + dim("tok/s");
    s += dim("]");
    s += "\n";
    std::cout << s;
}

// ============================================================================
// Generation functions
// ============================================================================

GenerationStats generate_streaming(forge::InferenceContext& ctx, const forge::Tokenizer& tokenizer,
                                   const std::vector<int32_t>& prompt_tokens, int max_new_tokens,
                                   float temperature, int top_k, float top_p, float repeat_penalty,
                                   int repeat_last_n, bool do_sample, uint64_t seed, int eos_token_id,
                                   const forge::SpeculativeConfig& spec_cfg = forge::SpeculativeConfig{},
                                   forge::PrefixCache* prefix_cache = nullptr);

GenerationStats generate_batch(forge::InferenceContext& ctx, const forge::Tokenizer& tokenizer,
                               const std::vector<int32_t>& prompt_tokens, int max_new_tokens,
                               float temperature, int top_k, float top_p, float repeat_penalty,
                               int repeat_last_n, bool do_sample, uint64_t seed, int eos_token_id,
                               const forge::SpeculativeConfig& spec_cfg = forge::SpeculativeConfig{},
                               forge::PrefixCache* prefix_cache = nullptr);

// ============================================================================
// Interactive chat
// ============================================================================

void interactive_chat(forge::InferenceContext& ctx, forge::Model& model,
                      forge::Tokenizer& tokenizer, forge::VisionEncoder* vision,
                      const CliArgs& args);

// ============================================================================
// Utility
// ============================================================================

std::string format_bytes(size_t bytes);
std::string trim(const std::string& s);
void print_logo();
void print_model_info(const forge::Model& model, const forge::Tokenizer& tokenizer);
void run_benchmark(forge::InferenceContext& ctx, const forge::Tokenizer& tokenizer,
                   int n_gpu_layers);
std::vector<float> encode_image(forge::VisionEncoder& vision, const std::string& image_path,
                                int& num_tokens);

// Global interrupt flag
extern volatile bool g_interrupted;
