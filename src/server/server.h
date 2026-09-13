/**
 * forge-server - OpenAI-compatible local inference HTTP server (Phase 1)
 *
 * Built from the same libraries as forge-cli. Endpoints:
 *   GET  /health                 -> {"status":"ok"}
 *   GET  /v1/models              -> OpenAI models list
 *   POST /v1/chat/completions    -> OpenAI chat completions (stream + non-stream)
 *
 * Concurrency model (mirrors llama.cpp server): httplib's worker threads
 * handle HTTP; all inference is serialized through one mutex since the model,
 * engine and KV cache are single instances.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "forge/chat_template.h"
#include "forge/generator.h"
#include "forge/model.h"
#include "forge/tokenizer.h"

namespace httplib {
class Server;
}  // namespace httplib

namespace forge {

// ---- CLI options ----

struct ServerOptions {
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    int n_gpu_layers = -1;
    int ctx_size = 0;  // 0 = auto (KV cap 4096 for long-ctx models); >0 = user context size
    int threads = -1;  // -1 = auto (probe fused GEMV on CPU, full GPU offload untouched)
    // f16 matches llama.cpp's default; fp32 wastes 2x VRAM for no quality gain.
    // fp8_* are the most VRAM-efficient (1 byte/element, no block overhead).
    std::string kv_cache_dtype = "f16";  // fp32 | f16 | q8_0 | q4_0 | fp8_e4m3 | fp8_e5m2
    bool cuda_graph = false;
    bool paged_kv = false;  // Use paged KV cache storage (default: contiguous)
    // Budget (MB) for the per-weight FP16 dequant cache used by the TensorCore
    // matmul path. -1 = default (1024 MB); 0 = cache disabled (callers fall
    // back to transient scratch); >0 = explicit budget.
    long dequant_cache_mb = -1;
    std::string load_mode;   // "" = default (mmap)
    std::string api_key;     // "" = no auth; otherwise "Authorization: Bearer <key>" required
    int max_inflight = 4;    // max queued inference requests (excess -> 429)
    // Per-request performance logging on stdout after every chat completion:
    //   --perf        one summary line (tokens, prefill/decode ms, tok/s)
    //   --perf-detail summary + top operator list from PerfProfiler (needs a
    //                 FORGE_PROFILING=ON build for operator-level data; the
    //                 summary line works in any build)
    bool perf = false;
    bool perf_detail = false;
};

// ---- Tool calling (OpenAI function-calling contract) ----

struct ToolCall {
    std::string id;          // e.g. "call-<hex>-<n>"
    std::string type = "function";
    std::string name;        // function name
    std::string arguments;   // JSON string (arguments object serialized)
};

// A tool the client exposes; parsed from the OpenAI `tools` array.
struct ToolSpec {
    std::string name;
    std::string description;
    std::string parameters_json;  // JSON schema, serialized
};

// ---- Parsed OpenAI chat-completion request ----

struct ChatCompletionRequest {
    std::vector<ChatTemplateMessage> messages;
    // Defaults are anti-degeneration engineering choices (repeat penalty etc.),
    // not OpenAI's bare defaults - small models (e.g. 1.1B) loop indefinitely
    // without them. Explicit request fields still override these.
    float temperature = 0.7f;
    float top_p = 1.0f;
    int top_k = 40;  // not an OpenAI field
    float repeat_penalty = 1.1f;  // not an OpenAI field
    int repeat_last_n = 64;       // not an OpenAI field
    int max_tokens = -1;  // <=0 -> engine default
    uint64_t seed = 0;
    bool stream = false;
    std::vector<std::string> stop;  // OpenAI stop strings (up to 4, OpenAI spec)
    // Function calling: OpenAI `tools` (functions) and `tool_choice`.
    // tool_choice "none" disables calling; any other value ("auto" or a
    // forced function object) enables it. Best-effort: no grammar constraint.
    std::vector<ToolSpec> tools;
    std::string tool_choice = "auto";
};

struct ChatCompletionResult {
    std::string text;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::string finish_reason;  // normalized to "stop" | "length" | "tool_calls"
    std::vector<ToolCall> tool_calls;  // non-empty -> finish_reason "tool_calls"
};

// Owns tokenizer / model / inference context. One instance per process.
class ServerState {
public:
    /// Load tokenizer + model + engine + KV cache (blocking, called once at
    /// startup). Returns false and fills `error` on failure.
    bool load(const ServerOptions& opt, std::string& error);

    /// Run one chat completion. When `on_token` is provided it receives
    /// UTF-8-safe decoded deltas (streaming); returning false from the
    /// callback signals client disconnect (generation still completes in
    /// Phase 1 - abort propagation is deferred).
    ChatCompletionResult chat_completion(
        const ChatCompletionRequest& req,
        const std::function<bool(const std::string& delta)>& on_token);

    const std::string& model_id() const { return model_id_; }

    // ---- Admission control (queue-full -> 429) ----
    /// Returns false when max_inflight requests are already queued.
    bool try_acquire_slot();
    void release_slot();

private:
    ServerOptions opt_;
    std::unique_ptr<Model> model_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<InferenceContext> ctx_;
    std::unique_ptr<ChatTemplateEngine> chat_template_;
    std::mutex inference_mutex_;  // serializes all inference (single KV cache)
    std::string model_id_;
    std::atomic<int> active_requests_{0};

    // ---- Session prefix reuse ----
    // Token ids currently resident in the KV cache: the previous request's
    // prompt plus the tokens it generated. Chat front-ends (opencode, ...)
    // resend the whole conversation every turn, so the next prompt usually
    // starts with exactly these ids and only the tail has to be prefilled.
    // Guarded by inference_mutex_.
    std::vector<int32_t> cached_kv_tokens_;
};

// CORS + OPTIONS preflight + optional Bearer api-key auth (server_routes.cpp).
void setup_middleware(httplib::Server& svr, const ServerOptions& opt);

// Route registration (server_routes.cpp).
void register_routes(httplib::Server& svr, ServerState& state);

}  // namespace forge
