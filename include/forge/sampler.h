#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "inference_batch.h"
#include "tensor.h"

namespace forge {

struct SamplerConfig {
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;  // Window size for repeat penalty (<=0 means use full history)
    bool do_sample = true;
    uint64_t seed = 0;
    float logit_softcapping = 0.0f;  // >0 means apply softcap: tanh(x/cap)*cap
};

// Result of sampling a single sequence within a batch
struct BatchSampleResult {
    int seq_index;     // index into InferenceBatch.items
    int seq_id;        // sequence ID
    int32_t token_id;  // sampled token
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& config = SamplerConfig{});
    ~Sampler();

    int sample(const TensorPtr& logits, int64_t pos);

    int sample_greedy(const TensorPtr& logits);

    int sample_temperature(const TensorPtr& logits, float temperature);

    // ---- Raw-pointer entries (speculative verification fast path) ----
    //
    // Operate directly on a host-resident logits row without wrapping it in a
    // Tensor. Dispatch mirrors sample(): greedy when do_sample=false or
    // temperature<=0, full temperature/top-k/top-p chain otherwise.
    //
    // Unlike sample(), they NEVER touch the token history -- the speculative
    // verifier appends each sampled token itself (every sampled token is a
    // confirmed output under resample-consistency).
    // Logits pointer must be CPU-resident and stay valid for the call.
    int sample_ptr(const float* logits, int vocab_size);
    int sample_greedy_ptr(const float* logits, int vocab_size);
    int sample_temperature_ptr(const float* logits, int vocab_size, float temperature);

    // Sample a token for each sequence in the batch.
    // logits_batch: [n_sequences, vocab_size] or [total_tokens, vocab_size]
    // For decode (1 token/seq), logits are [n_seq, vocab].
    // For prefill, each sequence's last-token logits are extracted.
    // Returns one token per sequence that requested logits.
    std::vector<BatchSampleResult> sample_batch(const TensorPtr& logits_batch,
                                                 const InferenceBatch& batch);

    void set_config(const SamplerConfig& config);
    const SamplerConfig& config() const;

    void add_token_to_history(int32_t token_id);
    void clear_history();
    const std::vector<int32_t>& token_history() const { return token_history_; }

private:
    void apply_repeat_penalty(float* logits, int n) const;
    // Shared CPU cores: copy `logits` into scratch, then run the configured
    // chain. Used by both the TensorPtr APIs and the raw-pointer entries.
    int greedy_sample_impl(const float* logits, int vocab_size);
    int temperature_sample_impl(const float* logits, int vocab_size, float temperature);
    void ensure_token_history_buffer(int n);
    // Reusable per-token scratch buffers. resize() is called with the current
    // vocab_size so begin()/end() always span exactly the live range; capacity
    // grows monotonically to absorb the per-token allocation churn.
    void ensure_scratch(int vocab_size);

    SamplerConfig config_;
    uint64_t rng_state_ = 12345;
    std::vector<int32_t> token_history_;

    std::vector<float> logits_scratch_;
    std::vector<float> probs_scratch_;
    std::vector<std::pair<float, int>> indexed_scratch_;

    void* cuda_argmax_buf_ = nullptr;
    int32_t* d_token_history_ = nullptr;
    int d_token_history_capacity_ = 0;

    uint64_t next_rng();
    float next_uniform();
};

}  // namespace forge
