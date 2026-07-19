#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "inference_batch.h"
#include "tensor.h"

namespace forge {

struct SamplerConfig {
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float repeat_penalty = 1.0f;
    bool do_sample = true;
    uint64_t seed = 0;
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
    void apply_repeat_penalty(std::vector<float>& logits) const;

    SamplerConfig config_;
    uint64_t rng_state_ = 12345;
    std::vector<int32_t> token_history_;

    void* cuda_argmax_buf_ = nullptr;

    uint64_t next_rng();
    float next_uniform();
};

}  // namespace forge
