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
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    int n_gpu_layers = -1;
    int ctx_size = 0;  // 0 = auto (KV cap 4096 for long-ctx models); >0 = user context size
    int n_keep = 0;    // llama.cpp-style: leading tokens never evicted on context shift
                        // (0 = keep only the system prompt; >0 = keep that many prefix tokens)
    int threads = -1;  // -1 = auto (probe fused GEMV on CPU, full GPU offload untouched)
    // q8_0 is the default: ~2x less VRAM than f16 (4x vs fp32) for a ~0.1%
    // accuracy loss (cosine >= 0.999 vs healthy f16), which roughly doubles the
    // context length a fixed VRAM budget can hold. f16 / fp8_* remain available
    // for quality- or throughput-critical setups.
    std::string kv_cache_dtype = "q8_0";  // fp32 | f16 | q8_0 | q4_0 | fp8_e4m3 | fp8_e5m2
    bool cuda_graph = false;
    bool paged_kv = false;  // Use paged KV cache storage (default: contiguous)
    bool log_prompt_prefix = false;  // Diagnostic: per-depth prompt-prefix hashes
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
    // KVMem tiered KV working set (arXiv:2609.04852). --kvmem implies
    // --paged-kv: the paged pool is capped to the working set and the logical
    // context (-c) may exceed it, with cold history archived off-device and
    // recalled per turn by KVWorkingSetManager.
    bool kvmem = false;
    int kvmem_budget = 0;       // GPU resident budget (tokens); 0 = the paged pool cap
    int kvmem_sink = 256;       // prefix tokens pinned resident forever
    int kvmem_gen_reserve = 0;  // window slots held back for this turn's output
    std::string kvmem_retrieval = "recent";  // recent | embed (reserved)

    KVemConfig kvem_config() const {
        KVemConfig c;
        c.enabled = kvmem;
        c.budget_tokens = kvmem_budget;
        c.sink_tokens = kvmem_sink;
        // Same reasoning as the CLI: a window that fills before the reply is
        // done makes the first decode step write past the pool. With no
        // explicit --kvmem-gen-reserve, reserve the default generation budget
        // (a request's max_tokens defaults to 512).
        c.gen_reserve = kvmem_gen_reserve > 0 ? kvmem_gen_reserve : 512;
        c.retrieval = kvmem_retrieval;
        return c;
    }
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
