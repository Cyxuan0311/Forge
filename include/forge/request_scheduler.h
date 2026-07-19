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
    int prefix_len = 0;        // number of tokens shared from cache (0 = no prefix hit)
    int prefix_seq_id = -1;    // seq_id of the cached prefix (-1 = no cache)
    bool from_cache = false;   // true if this request used a prefix cache hit

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

    // Prefix cache stats
    int prefix_cache_hits() const { return prefix_cache_hits_; }
    int prefix_cache_misses() const { return prefix_cache_misses_; }

private:
    void schedule();

    // Prefix cache helpers
    static size_t hash_prompt(const std::vector<int32_t>& tokens);
    bool try_prefix_cache(GenerateRequest& req);
    void evict_prefix_cache(int seq_id);
    void preserve_prefix_cache(int seq_id, int prompt_len);

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

    mutable std::mutex mutex_;
};

}  // namespace forge
