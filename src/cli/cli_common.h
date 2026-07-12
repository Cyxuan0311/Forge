/**
 * NanoInfer CLI - Shared definitions
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Forward declarations
namespace nanoinfer {
class Tokenizer;
class InferenceContext;
class Model;
class VisionEncoder;
}

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
    int threads = -1;
    int batch_size = 512;
    int n_predict = -1;
    std::string kv_cache_dtype = "fp32";

    // Sampling control
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    float repeat_penalty = 1.1f;
    uint64_t seed = 0;
    bool no_sample = false;

    // Mode control
    bool interactive = false;
    bool no_stream = false;
    bool info_only = false;
    bool bench = false;

    // System prompt
    std::string system_prompt;

    // Verbosity
    int verbose = 1;
};

CliArgs parse_args(int argc, char** argv);

// ============================================================================
// Chat template
// ============================================================================

enum class ChatTemplateType {
    ChatML,       // Qwen, Yi: <|im_start|>...<|im_end|>
    DeepSeek,     // DeepSeek: <｜User｜><｜Assistant｜>
    Llama,        // Llama: [INST]...[/INST]
    Plain,        // No template, plain text
};

struct ChatMessage {
    std::string role;     // "system", "user", "assistant"
    std::string content;
};

ChatTemplateType detect_template_type(const nanoinfer::Tokenizer& tokenizer);

std::vector<int32_t> apply_chat_template(
    const nanoinfer::Tokenizer& tokenizer,
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
};

// ============================================================================
// Generation functions
// ============================================================================

GenerationStats generate_streaming(
    nanoinfer::InferenceContext& ctx,
    const nanoinfer::Tokenizer& tokenizer,
    const std::vector<int32_t>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    float repeat_penalty,
    bool do_sample,
    uint64_t seed,
    int eos_token_id);

GenerationStats generate_batch(
    nanoinfer::InferenceContext& ctx,
    const nanoinfer::Tokenizer& tokenizer,
    const std::vector<int32_t>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    float repeat_penalty,
    bool do_sample,
    uint64_t seed,
    int eos_token_id);

// ============================================================================
// Interactive chat
// ============================================================================

void interactive_chat(
    nanoinfer::Model& model,
    nanoinfer::Tokenizer& tokenizer,
    nanoinfer::VisionEncoder* vision,
    const CliArgs& args);

// ============================================================================
// Utility
// ============================================================================

std::string format_bytes(size_t bytes);
std::string trim(const std::string& s);
void print_logo();
void print_model_info(const nanoinfer::Model& model, const nanoinfer::Tokenizer& tokenizer);
void run_benchmark(nanoinfer::InferenceContext& ctx, const nanoinfer::Tokenizer& tokenizer, int n_gpu_layers);
std::vector<float> encode_image(nanoinfer::VisionEncoder& vision, const std::string& image_path, int& num_tokens);

// Global interrupt flag
extern volatile bool g_interrupted;
