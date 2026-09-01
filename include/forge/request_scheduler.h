#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "context.h"
#include "model.h"
#include "prefix_cache.h"
#include "sampler.h"

namespace forge {

enum class RequestStatus {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
    Failed,
};

// Cached prompt entry for prefix caching via seq_cp zero-copy sharing.
struct CachedPrompt {
    std::vector<int32_t> tokens;  // the cached prompt tokens
    int seq_id = -1;              // seq_id owning the cached KV cells
    bool valid = false;
};

struct GenerateRequest {
    int request_id;
    std::vector<int32_t> prompt_tokens;
    int max_new_tokens = 256;
    int eos_token_id = -1;
    SamplerConfig sampler_config;

    RequestStatus status = RequestStatus::Waiting;
    std::vector<int32_t> output_tokens;
    int num_generated = 0;
    int current_pos = 0;
    std::string finish_reason;

    // Prefix cache fields
    int prefix_len = 0;       // number of tokens shared from cache (0 = no prefix hit)
    int prefix_seq_id = -1;   // seq_id of the cached prefix (-1 = no cache)
    bool from_cache = false;  // true if this request used a prefix cache hit

    // Roadmap 1.3 (chunked prefill): how many prompt tokens have already been
    // pushed through the model. Prefill advances in prefill_chunk_size slices so
    // a long prompt is interleaved with the decode steps of other requests
    // instead of monopolising one giant step.
    int prefill_done = 0;

    using Callback =
        std::function<void(int request_id, int32_t token_id, int step, RequestStatus status)>;
    Callback callback;
};

class RequestScheduler {
public:
    explicit RequestScheduler(Model& model, int block_size = 16, int max_num_seqs = 4);

    int submit(const std::vector<int32_t>& prompt_tokens, int max_new_tokens = 256,
               int eos_token_id = -1, const SamplerConfig& sampler_cfg = SamplerConfig{},
               GenerateRequest::Callback callback = nullptr);

    bool step();

    std::vector<GenerateRequest> get_finished();
    std::vector<GenerateRequest> get_all_requests() const;

    int num_active() const;
    int num_waiting() const;
    bool has_pending() const;

    void abort(int request_id);
    void reset();

    Model& model() { return model_; }
    InferenceContext& context() { return ctx_; }
    const InferenceContext& context() const { return ctx_; }

    // ---- Roadmap 1.3: chunked prefill / continuous batching ----
    //
    // A long prompt is split across several steps instead of being processed in
    // one step, so decode requests interleaved with it keep producing tokens at
    // a steady rate rather than waiting for the whole prompt.

    // Max prompt tokens one request may consume per step. 0 (default) follows
    // ContextParams::n_ubatch. Set to a negative value to disable chunking
    // (whole prompt per step — the pre-1.3 behaviour).
    void set_prefill_chunk_size(int n) { prefill_chunk_size_ = n; }
    int prefill_chunk_size() const { return prefill_chunk_size_; }

    // Per-step token accounting (last completed step).
    int last_step_prefill_tokens() const { return last_step_prefill_tokens_; }
    int last_step_decode_tokens() const { return last_step_decode_tokens_; }
    // Share of the last step's tokens that came from decode requests.
    double last_step_decode_ratio() const {
        const int total = last_step_prefill_tokens_ + last_step_decode_tokens_;
        return total > 0 ? static_cast<double>(last_step_decode_tokens_) / total : 0.0;
    }
    // Slowest step observed so far, in milliseconds. A long prompt that is
    // chunked keeps this flat; an unchunked one makes it spike.
    double max_step_latency_ms() const { return max_step_latency_ms_; }
    // Steps in which a decode token was produced while some other request was
    // still prefilling — the interleaving guarantee, counted directly.
    int interleaved_steps() const { return interleaved_steps_; }
    // Number of prefill chunks issued (grows >1 per request when chunking).
    int prefill_chunks_issued() const { return prefill_chunks_issued_; }

    // Prefix cache stats
    int prefix_cache_hits() const {
        return paged_mode_ ? prefix_cache_.hits() : prefix_cache_hits_;
    }
    int prefix_cache_misses() const {
        return paged_mode_ ? prefix_cache_.misses() : prefix_cache_misses_;
    }

    // ---- Roadmap 1.1: KV host offload (swap) ----
    //
    // When the paged KV pool is under pressure the scheduler evicts whole
    // sequences' pages to a pinned host pool (KVBlockSwapper) instead of
    // rejecting new work. Evicted sequences skip one step and are brought back
    // before their next forward pass. See PagedKVStorage::offload_seq.

    // Fraction of the total page capacity that must stay free before swap-out
    // triggers. 0 disables proactive swap-out (only exact exhaustion triggers);
    // the default 0.15 keeps headroom so evictions are rare and batched.
    void set_kv_swap_watermark(float w) { kv_swap_watermark_ = w; }
    float kv_swap_watermark() const { return kv_swap_watermark_; }

    // Swap activity (paged mode only).
    int swap_events() const { return swap_events_; }
    int num_offloaded_pages() const { return num_offloaded_pages_; }
    int num_brought_back_pages() const { return num_brought_back_pages_; }
    int num_free_pages() const;
    int num_total_pages() const;
    size_t host_pool_bytes() const;

private:
    void schedule();

    // Roadmap 1.1: evict low-value active sequences' pages to the host pool
    // when free pages fall below the watermark. Returns the rids whose pages
    // were evicted (they must skip this step's batch and be brought back later).
    std::vector<int> try_swap_out();
    void update_swap_stats();

    // Prefix cache helpers (contiguous mode: legacy prompt_cache_)
    static size_t hash_prompt(const std::vector<int32_t>& tokens);
    bool try_prefix_cache(GenerateRequest& req);
    void evict_prefix_cache(int seq_id);
    void preserve_prefix_cache(int seq_id, int prompt_len);

    // Prefix cache helpers (paged mode: page-level PrefixCache)
    bool try_prefix_cache_paged(GenerateRequest& req);
    void finish_request_paged(GenerateRequest& req);

    // Release a sequence's KV cache entries
    void release_seq_kv(int seq_id, int prompt_len = 0);

    Model& model_;
    InferenceContext ctx_;
    Sampler sampler_;

    std::queue<int> waiting_queue_;
    std::unordered_map<int, GenerateRequest> requests_;
    std::vector<int> active_ids_;
    int next_request_id_ = 0;
    int max_num_seqs_ = 4;

    // Prefix cache: keyed by hash of prompt tokens
    static constexpr int MIN_CACHE_PROMPT_LEN = 16;
    std::unordered_map<size_t, CachedPrompt> prompt_cache_;
    int prefix_cache_hits_ = 0;
    int prefix_cache_misses_ = 0;

    // Page-level prefix cache (paged mode only)
    PrefixCache prefix_cache_;
    bool paged_mode_ = false;

    // Roadmap 1.3: chunked prefill state and metrics.
    int prefill_chunk_size_ = 0;  // 0 => follow ContextParams::n_ubatch
    int last_step_prefill_tokens_ = 0;
    int last_step_decode_tokens_ = 0;
    double max_step_latency_ms_ = 0.0;
    int interleaved_steps_ = 0;
    int prefill_chunks_issued_ = 0;

    // Roadmap 1.1: host offload (swap) state.
    float kv_swap_watermark_ = 0.15f;
    int swap_events_ = 0;
    int num_offloaded_pages_ = 0;
    int num_brought_back_pages_ = 0;

    mutable std::mutex mutex_;
};

}  // namespace forge
