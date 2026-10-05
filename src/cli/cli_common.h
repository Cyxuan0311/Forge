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
#include "forge/cli_common_args.h"
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
    // Flags shared verbatim with forge-server: model, -ngl, -c, -t, the KV cache
    // dtype and the separate -ctk/-ctv types, paged/cuda-graph/kv-offload,
    // load-mode/embedding-offload/prefetch, and the whole --kvmem block. Parsed by
    // forge::parse_common_arg() and applied by forge::apply_common_args(), so
    // the two binaries cannot drift again (they had: different --kv-cache-dtype
    // defaults, and the server accepted a typo'd dtype as f16 without a word).
    forge::CommonArgs common;

    // ---- CLI-only ----
    std::string prompt;
    std::string mmproj_path;
    std::string image_path;
    int n_predict = -1;
    // Max tokens per forward_batch() call. This was parsed and then read nowhere,
    // so -b silently did nothing; apply_common_args now wires it to n_batch.
    int batch_size = 512;

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
    double elapsed_ms = 0;       // wall clock: prefill + decode + sampling overhead
    double prompt_eval_ms = 0;   // prompt forward + first-token sample
    double decode_ms = 0;        // decode loop only (excludes prefill)
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
//   分拆 prefill / decode 两阶段, 避免把 prefill 时间算进 "generated tok/s"
//   分母 (旧行为会把 decode 速度低估). 同时把投机解码接受率内联展示.
inline void forge_print_stats_line(const GenerationStats& stats) {
    if (stats.num_generated_tokens <= 0)
        return;
    using namespace cli_color;

    // 防御: decode_ms 来自 generator, 但若未填充则用墙钟减去 prefill 兜底.
    double decode_ms = stats.decode_ms;
    if (decode_ms <= 0.0 && stats.elapsed_ms > stats.prompt_eval_ms)
        decode_ms = stats.elapsed_ms - stats.prompt_eval_ms;

    std::string s;
    s += dim("  [");

    // ---- Prefill ----
    s += bold(cyan(std::to_string(stats.num_prompt_tokens)));
    s += cyan(" prompt");
    s += dim(", prefill ");
    s += dim(std::to_string((long long)stats.prompt_eval_ms)) + dim(" ms");
    s += dim(" (");
    s += yellow(forge_fmt_tok_s(stats.num_prompt_tokens, stats.prompt_eval_ms));
    s += dim(" t/s)");

    s += dim(" | ");

    // ---- Decode ----
    s += bold(green(std::to_string(stats.num_generated_tokens)));
    s += green(" gen");
    s += dim(", decode ");
    s += dim(std::to_string((long long)decode_ms)) + dim(" ms");
    s += dim(" (");
    s += yellow(forge_fmt_tok_s(stats.num_generated_tokens, decode_ms));
    s += dim(" t/s");
    if (stats.num_generated_tokens > 0 && decode_ms > 0.0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f", decode_ms / stats.num_generated_tokens);
        s += dim(", ") + dim(buf) + dim(" ms/tok");
    }
    s += dim(")");

    // ---- Speculative inline summary ----
    if (stats.spec.n_spec_steps > 0 || stats.spec.n_fallback_steps > 0) {
        s += dim(" | spec ");
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f%%", stats.spec.acceptance_rate() * 100.0);
        s += magenta(buf);
    }

    // ---- Total wall ----
    s += dim(" | total ");
    s += dim(std::to_string((long long)stats.elapsed_ms)) + dim(" ms");
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
                                   forge::PrefixCache* prefix_cache = nullptr,
                                   std::string* out_text = nullptr);

GenerationStats generate_batch(forge::InferenceContext& ctx, const forge::Tokenizer& tokenizer,
                               const std::vector<int32_t>& prompt_tokens, int max_new_tokens,
                               float temperature, int top_k, float top_p, float repeat_penalty,
                               int repeat_last_n, bool do_sample, uint64_t seed, int eos_token_id,
                               const forge::SpeculativeConfig& spec_cfg = forge::SpeculativeConfig{},
                               forge::PrefixCache* prefix_cache = nullptr,
                               std::string* out_text = nullptr);

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
