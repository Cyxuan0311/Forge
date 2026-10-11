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
#include <unordered_map>
#include <vector>

#include "forge/agent/agent_pipeline.h"
#include "forge/chat_template.h"
#include "forge/cli_common_args.h"
#include "forge/cli_dashboard_args.h"
#include "forge/cli_spec_args.h"
#include "forge/generator.h"
#include "forge/kv_working_set.h"
#include "forge/metrics.h"
#include "forge/model.h"
#include "forge/prefix_cache.h"
#include "forge/tokenizer.h"

namespace httplib {
class Server;
}  // namespace httplib

namespace forge {

class VisionEncoder;  // forward-declared; defined in forge/vision_encoder.h

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
    std::string api_key;   // "" = no auth; otherwise "Authorization: Bearer <key>" required
    int max_inflight = 4;  // max queued inference requests (excess -> 429)
    std::string mmproj_path;  // "" = no vision encoder; otherwise the mmproj GGUF
    // Per-request performance logging on stdout after every chat completion:
    //   --perf        one summary line (tokens, prefill/decode ms, tok/s)
    //   --perf-detail summary + top operator list from PerfProfiler (needs a
    //                 FORGE_PROFILING=ON build for operator-level data; the
    //                 summary line works in any build)
    bool perf = false;
    bool perf_detail = false;
    // Live web dashboard (--dashboard). Parsed by forge::parse_dashboard_arg()
    // and served by forge_dashboard on its OWN loopback port, never on --port:
    // the API listener's pre-routing handler demands a Bearer key for every
    // request and a browser cannot send one, so mounting the dashboard there
    // would mean punching a hole in authentication.
    forge::DashboardArgs dashboard;
    // --chat: open the embedded chat UI (web-chat/, Vue) in a browser. Unlike
    // the dashboard it is served on the MAIN port under /chat/ — static assets
    // are exempt from the Bearer gate in setup_middleware (see server_routes.cpp
    // for the rationale), so no second port is needed. A no-op when the binary
    // was built without Node.js: /chat/ then returns a 404 with rebuild hints.
    bool chat_ui = false;
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
    // Optional conversation key. Requests sharing a session reuse each other's KV
    // prefix and token mirror, so a multi-turn chat only prefills its tail.
    // Absent/empty means the single default session -- the historical behaviour.
    std::string session_id;
    // Incremental history. When non-empty these messages are APPENDED to the
    // session's stored history instead of replacing it, so a client stops
    // resending the whole conversation every turn. The rendered prompt is still
    // produced from the complete history, so prefix reuse is unaffected.
    //
    // Requires session_id: without one there is no history to append to. Sending
    // both messages and delta_messages is an error rather than a silent merge --
    // the intent would be ambiguous.
    std::vector<ChatTemplateMessage> delta_messages;
    // Discard the session's stored history and KV working set before handling this
    // request (the equivalent of a client's "/clear"). Lets a delta client start
    // over without the server accumulating an ever-growing conversation it can no
    // longer correct.
    bool reset_session = false;
    // Defaults are anti-degeneration engineering choices (repeat penalty etc.),
    // not OpenAI's bare defaults - small models (e.g. 1.1B) loop indefinitely
    // without them. Explicit request fields still override these.
    float temperature = 0.7f;
    float top_p = 1.0f;
    int top_k = 40;               // not an OpenAI field
    float repeat_penalty = 1.1f;  // not an OpenAI field
    int repeat_last_n = 64;       // not an OpenAI field
    int max_tokens = -1;          // <=0 -> engine default
    uint64_t seed = 0;
    bool stream = false;
    // OpenAI's stream_options.include_usage: when streaming, emit a final
    // usage-only chunk (choices: []) carrying token counts and timings.
    bool stream_include_usage = false;
    std::vector<std::string> stop;  // OpenAI stop strings (up to 4, OpenAI spec)
    // Function calling: OpenAI `tools` (functions) and `tool_choice`.
    // tool_choice "none" disables calling; any other value ("auto" or a
    // forced function object) enables it. Best-effort: no grammar constraint.
    std::vector<ToolSpec> tools;
    std::string tool_choice = "auto";
};

struct ChatCompletionResult {
    std::string text;
    std::string reasoning;  // <think> content separated from text (may be empty)
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::string finish_reason;         // normalized to "stop" | "length" | "tool_calls"
    std::vector<ToolCall> tool_calls;  // non-empty -> finish_reason "tool_calls"
    // Server-side timings, already computed for the --perf line. Reported in
    // `usage` (and in the streaming usage chunk) so a client can show real
    // numbers per turn instead of estimating them from its own clocks.
    double prefill_ms = 0.0;  // prompt forward + first-token sample
    double decode_ms = 0.0;   // decode loop, excludes prefill
    double prefill_ms_per_1k() const {
        return prompt_tokens > 0 ? prefill_ms * 1000.0 / prompt_tokens : 0.0;
    }
    double tokens_per_sec() const {
        return completion_tokens > 0 ? completion_tokens / (decode_ms / 1000.0) : 0.0;
    }
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

    /// Defined out-of-line so the vision encoder (forward-declared here) is a
    /// complete type when destroyed.
    ~ServerState();

    /// Run one chat completion. When `on_token` is provided it receives
    /// UTF-8-safe decoded deltas (streaming); returning false from the
    /// callback signals client disconnect (generation still completes in
    /// Phase 1 - abort propagation is deferred). `on_reasoning`, when
    /// provided, receives the <think> deltas separated from the content.
    ///
    /// `scope`, when non-null, is the per-request dashboard handle created by
    /// the route handler so that its arrival timestamp is the moment the HTTP
    /// request landed, not the moment this function was entered (admission
    /// control, JSON parsing and the queue wait all sit in between and the
    /// user perceives them). chat_completion fills it in and finishes it.
    ChatCompletionResult chat_completion(
        const ChatCompletionRequest& req,
        const std::function<bool(const std::string& delta)>& on_token,
        const std::function<bool(const std::string& reasoning_delta)>& on_reasoning = {},
        MetricsRegistry::RequestScope* scope = nullptr);

    const std::string& model_id() const { return model_id_; }
    /// Model architecture string (e.g. "qwen35"), for the dashboard header.
    std::string arch() const;

    // ---- Dashboard accessors (read from the dashboard thread) ----
    // Only lock-free atomics and immutable geometry are exposed here; KV
    // occupancy is published by publish_engine_gauges() from inside the
    // inference lock, because KVCache::active_bytes() and num_free_slots()
    // walk their cell tables without any synchronization.
    int active_requests() const { return active_requests_.load(std::memory_order_relaxed); }
    /// Snapshot of the engine-side gauges. Safe to call from any thread: it
    /// only reads the last atomically published values.
    EngineGauges engine_gauges() const;
    /// Recomputes and republishes the engine gauges. MUST be called from the
    /// thread holding inference_mutex_ (i.e. from inside chat_completion).
    void publish_engine_gauges();
    /// Fills `g` without publishing. Safe only from the inference thread; this
    /// is what the dashboard's live hook calls mid-generation.
    void fill_engine_gauges(EngineGauges& g);

    // ---- Admission control (queue-full -> 429) ----
    /// Returns false when max_inflight requests are already queued.
    bool try_acquire_slot();

    // ---- Per-session state ----
    // The KV cache is a single shared resource, so before sessions every request
    // addressed seq 0 and reused one token mirror: two interleaved conversations
    // overwrote each other's prefix and the next turn re-prefilled from scratch.
    // A session owns a seq_id plus its own mirror, so conversations stop stepping on
    // each other.
    //
    // session_id is optional in the request. Absent (or empty) means the default
    // session, which is exactly the old single-slot behaviour -- existing clients are
    // unaffected.
    struct SessionState {
        // Working-set seq. Allocated from a range above the radix cache's cache
        // seq_ids so the two allocators cannot collide.
        int seq_id = 0;
        // Tokens this session has resident: prompt + generated. The prefix-reuse
        // mirror for the contiguous backend.
        std::vector<int32_t> kv_tokens;
        // Conversation history as sent by the client, accumulated across turns so a
        // delta request can be rendered in full context. Empty for sessions that
        // only ever send the complete history.
        std::vector<ChatTemplateMessage> messages;
        // The reply we produced last turn. A delta client does not know it, so the
        // assistant turn is re-inserted from here to keep the conversation
        // alternating (and so tool_calls replies survive, which a client cannot
        // reproduce byte-for-byte).
        std::string last_reply;
        // Tokens fed into the working set's logical history (KVMem path), kept in
        // step with kv_tokens so a session can be restored or inspected.
        int64_t last_active_ms = 0;
        uint64_t use_count = 0;
    };

    // Maximum retained sessions. Beyond this the least-recently-used one has its seq
    // released, so a long tail of distinct ids cannot pin the whole page pool.
    static constexpr int kMaxSessions = 32;

    /// Resolve (creating on first use) the state for `session_id`, evicting the
    /// least-recently-used session if the table is full. Empty `session_id` maps to
    /// the default session, preserving the pre-session single-slot behaviour.
    SessionState& acquire_session(const std::string& session_id);

    /// Drop a session and release its KV working set. Called on eviction and on
    /// request failure.
    void drop_session(const std::string& session_id, bool release_kv);

    /// Number of live sessions (diagnostics/tests).
    int session_count() const { return static_cast<int>(sessions_.size()); }

    /// True when this server was started with a vision encoder (--mmproj) that a
    /// multimodal model can consume. Requests carrying images are rejected with a
    /// clear error when this is false.
    bool has_vision() const { return vision_ != nullptr; }

    /// True when the KVMem tiered working set is active. It owns seq 0's slot layout
    /// with process-wide state (logical_hist_, archived_chunks_), so it serves one
    /// conversation at a time regardless of max_inflight -- see try_acquire_slot().
    bool tiered_kv_active() const;

    /// Default session id used when a request does not carry one.
    static const char* default_session_id() { return ""; }
    void release_slot();

private:
    ServerOptions opt_;
    std::unique_ptr<Model> model_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<InferenceContext> ctx_;
    std::unique_ptr<ChatTemplateEngine> chat_template_;
    VisionEncoder* vision_ = nullptr;  // loaded only when --mmproj is set (raw: avoids incomplete-type dtor)
    AgentPipelineConfig agent_cfg_;  // resolved once in load() (FORGE_TOOL_DIALECT)
    std::mutex inference_mutex_;     // serializes all inference (single KV cache)
    std::string model_id_;
    std::atomic<int> active_requests_{0};

    // Session token mirrors live in SessionState::kv_tokens (see acquire_session()).

    // Live sessions, keyed by the request's session_id ("" = default session).
    // Guarded by inference_mutex_.
    std::unordered_map<std::string, SessionState> sessions_;
    int next_session_seq_ = 0;

    // ---- Paged-mode radix prefix cache ----
    // Mirrors the RequestScheduler's page-level prefix sharing. Each request's
    // working set is seq_id 0; cached prefixes live in dedicated cache seq_ids
    // whose KV pages are reference-counted. Guarded by inference_mutex_.
    PrefixCache prefix_cache_;
    // False for hybrid architectures whose recurrent state lives outside the
    // paged KV storage (see InferenceEngine::prefix_cache_supported()); gates
    // the server's radix lookup/register paths below.
    bool prefix_cache_enabled_ = true;
};

// CORS + OPTIONS preflight + optional Bearer api-key auth (server_routes.cpp).
void setup_middleware(httplib::Server& svr, const ServerOptions& opt);

// Route registration (server_routes.cpp).
void register_routes(httplib::Server& svr, ServerState& state);

}  // namespace forge
