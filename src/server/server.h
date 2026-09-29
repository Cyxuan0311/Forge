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

#include "forge/agent/agent_pipeline.h"
#include "forge/chat_template.h"
#include "forge/generator.h"
#include "forge/cli_common_args.h"
#include "forge/cli_spec_args.h"
#include "forge/kv_working_set.h"
#include "forge/model.h"
#include "forge/prefix_cache.h"
#include "forge/tokenizer.h"

namespace httplib {
class Server;
}  // namespace httplib

namespace forge {

// ---- CLI options ----

struct ServerOptions {
    // Flags shared verbatim with forge-cli: model, -ngl, -c, -t, the KV cache
    // dtype and the separate -ctk/-ctv types, --paged-kv/--cuda-graph/--kv-offload,
    // --load-mode/--offload-embedding/--prefetch, and the whole --kvmem block.
    // Parsed by forge::parse_common_arg() and applied by
    // forge::apply_common_args(), so the two binaries cannot drift again -- which
    // they had: this struct defaulted --kv-cache-dtype to q8_0 while the CLI
    // defaulted it to f16, -ngl here was a bare std::stoi (so "-ngl auto"
    // terminated the process), and a typo'd dtype silently became f16.
    forge::CommonArgs common;

    // Speculative decoding. Shared with forge-cli through
    // forge::parse_spec_arg() (see cli_spec_args.h) so the two agree on every
    // default and validation rule. The server previously had no speculation at
    // all: no --spec* flags, and no SpeculativeConfig ever reached
    // ContextParams, so Generator's `speculative_config.enabled` was never true
    // and the spec_stats branch in handle_chat was unreachable.
    forge::SpeculativeConfig spec;

    // ---- server-only ----
    std::string host = "127.0.0.1";
    int port = 8080;
    int n_keep = 0;  // llama.cpp-style: leading tokens never evicted on context shift
                      // (0 = keep only the system prompt; >0 = keep that many prefix tokens)
    bool log_prompt_prefix = false;  // Diagnostic: per-depth prompt-prefix hashes
    // Budget (MB) for the per-weight FP16 dequant cache used by the TensorCore
    // matmul path. -1 = default (1024 MB); 0 = cache disabled (callers fall
    // back to transient scratch); >0 = explicit budget.
    long dequant_cache_mb = -1;
    std::string api_key;     // "" = no auth; otherwise "Authorization: Bearer <key>" required
    int max_inflight = 4;    // max queued inference requests (excess -> 429)
    // Per-request performance logging on stdout after every chat completion:
    //   --perf        one summary line (tokens, prefill/decode ms, tok/s)
    //   --perf-detail summary + top operator list from PerfProfiler (needs a
    //                 FORGE_PROFILING=ON build for operator-level data; the
    //                 summary line works in any build)
    bool perf = false;
    bool perf_detail = false;
    // The reserve fallback is sized for a request's default max_tokens (512),
    // which is what this binary's callers use; forge-cli sizes it for -n. Both
    // go through CommonArgs::kvem_config(), which had been duplicated in each
    // binary and had drifted (256 in the CLI, 512 here).
    forge::KVemConfig kvem_config() const { return common.kvem_config(); }

};

// ---- Tool calling (OpenAI function-calling contract) ----
// ToolCall / ToolSpec now live in forge/agent/tool_call.h (included above);
// dialect parsing and reasoning separation live in the forge_agent module.

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
    std::string reasoning;      // <think> content separated from text (may be empty)
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::string finish_reason;  // normalized to "stop" | "length" | "tool_calls"
    std::vector<ToolCall> tool_calls;  // non-empty -> finish_reason "tool_calls"
};

// Thrown before prefill when the rendered prompt + generation budget exceeds
// the KV capacity (paged pool tokens or contiguous max_seq_len). Routes map it
// to HTTP 400 invalid_request_error — a client-fixable request, not a server
// fault (vLLM: `This model's maximum context length is N tokens...`).
struct ContextLengthExceededError : std::runtime_error {
    using std::runtime_error::runtime_error;
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
    /// Phase 1 - abort propagation is deferred). `on_reasoning`, when
    /// provided, receives the <think> deltas separated from the content.
    ChatCompletionResult chat_completion(
        const ChatCompletionRequest& req,
        const std::function<bool(const std::string& delta)>& on_token,
        const std::function<bool(const std::string& reasoning_delta)>& on_reasoning = {});

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
    AgentPipelineConfig agent_cfg_;  // resolved once in load() (FORGE_TOOL_DIALECT)
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

    // ---- Paged-mode radix prefix cache ----
    // Mirrors the RequestScheduler's page-level prefix sharing. Each request's
    // working set is seq_id 0; cached prefixes live in dedicated cache seq_ids
    // whose KV pages are reference-counted. Guarded by inference_mutex_.
    PrefixCache prefix_cache_;
};

// CORS + OPTIONS preflight + optional Bearer api-key auth (server_routes.cpp).
void setup_middleware(httplib::Server& svr, const ServerOptions& opt);

// Route registration (server_routes.cpp).
void register_routes(httplib::Server& svr, ServerState& state);

}  // namespace forge
