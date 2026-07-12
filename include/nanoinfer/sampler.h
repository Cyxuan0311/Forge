#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include "tensor.h"

namespace nanoinfer {

struct SamplerConfig {
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float repeat_penalty = 1.0f;
    bool do_sample = true;
    uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& config = SamplerConfig{});
    ~Sampler();

    int sample(const TensorPtr& logits, int64_t pos);

    int sample_greedy(const TensorPtr& logits);

    int sample_temperature(const TensorPtr& logits, float temperature);

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

} // namespace nanoinfer
