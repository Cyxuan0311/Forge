#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <functional>
#include <string>
#include <unordered_map>
#include <mutex>
#include "model.h"
#include "context.h"
#include "sampler.h"
#include "paged_kv_cache.h"

namespace nanoinfer {

enum class RequestStatus {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
    Failed,
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

    using Callback = std::function<void(int request_id, int32_t token_id, int step, RequestStatus status)>;
    Callback callback;
};

class RequestScheduler {
public:
    explicit RequestScheduler(Model& model, int block_size = 16, int max_num_seqs = 4);

    int submit(const std::vector<int32_t>& prompt_tokens,
               int max_new_tokens = 256,
               int eos_token_id = -1,
               const SamplerConfig& sampler_cfg = SamplerConfig{},
               GenerateRequest::Callback callback = nullptr);

    bool step();

    std::vector<GenerateRequest> get_finished();
    std::vector<GenerateRequest> get_all_requests() const;

    int num_active() const;
    int num_waiting() const;
    bool has_pending() const;

    void abort(int request_id);
    void reset();

    PagedKVCache& paged_cache() { return paged_cache_; }
    const PagedKVCache& paged_cache() const { return paged_cache_; }

    Model& model() { return model_; }
    InferenceContext& context() { return ctx_; }

private:
    void schedule();
    bool prefill_request(GenerateRequest& req);
    bool decode_step(GenerateRequest& req);

    Model& model_;
    InferenceContext ctx_;
    PagedKVCache paged_cache_;
    Sampler sampler_;

    std::queue<int> waiting_queue_;
    std::unordered_map<int, GenerateRequest> requests_;
    std::vector<int> active_ids_;
    int next_request_id_ = 0;
    int max_num_seqs_ = 4;

    mutable std::mutex mutex_;
};

} // namespace nanoinfer
