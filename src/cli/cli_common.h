/**
 * Forge CLI - Shared definitions
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "forge/chat_template.h"
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
    int n_predict = -1;
    std::string kv_cache_dtype = "fp32";
    bool cuda_graph = false;  // Phase 10: CUDA Graph for decode

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
// Generation functions
// ============================================================================

GenerationStats generate_streaming(forge::InferenceContext& ctx, const forge::Tokenizer& tokenizer,
                                   const std::vector<int32_t>& prompt_tokens, int max_new_tokens,
                                   float temperature, int top_k, float top_p, float repeat_penalty,
                                   int repeat_last_n, bool do_sample, uint64_t seed, int eos_token_id,
                                   const forge::SpeculativeConfig& spec_cfg = forge::SpeculativeConfig{});

GenerationStats generate_batch(forge::InferenceContext& ctx, const forge::Tokenizer& tokenizer,
                               const std::vector<int32_t>& prompt_tokens, int max_new_tokens,
                               float temperature, int top_k, float top_p, float repeat_penalty,
                               int repeat_last_n, bool do_sample, uint64_t seed, int eos_token_id,
                               const forge::SpeculativeConfig& spec_cfg = forge::SpeculativeConfig{});

// ============================================================================
// Interactive chat
// ============================================================================

void interactive_chat(forge::Model& model, forge::Tokenizer& tokenizer,
                      forge::VisionEncoder* vision, const CliArgs& args);

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
