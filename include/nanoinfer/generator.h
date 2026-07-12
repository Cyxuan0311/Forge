#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <functional>
#include "model.h"
#include "context.h"
#include "sampler.h"

namespace nanoinfer {

struct GenerationConfig {
    int max_new_tokens = 256;
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float repeat_penalty = 1.0f;
    bool do_sample = true;
    uint64_t seed = 0;
    int eos_token_id = -1;
};

struct GenerationResult {
    std::vector<int32_t> token_ids;
    std::string text;
    int num_prompt_tokens = 0;
    int num_generated_tokens = 0;
    bool finished = false;
    std::string finish_reason;
};

class Generator {
public:
    Generator(InferenceContext& ctx, const SamplerConfig& sampler_cfg = SamplerConfig{});
    ~Generator();

    GenerationResult generate(const std::vector<int32_t>& prompt_tokens,
                               const GenerationConfig& config);

    using TokenCallback = std::function<void(int32_t token_id, int step)>;

    GenerationResult generate(const std::vector<int32_t>& prompt_tokens,
                               const GenerationConfig& config,
                               const TokenCallback& callback);

private:
    InferenceContext& ctx_;
    Sampler sampler_;
    TensorPtr decode_input_gpu_;
    void* decode_token_buf_ = nullptr;
};

} // namespace nanoinfer
