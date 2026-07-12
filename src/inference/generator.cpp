#include "forge/generator.h"
#include "forge/engine.h"
#include "forge/engines/llama_engine.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"
#include <stdexcept>
#include <cstring>
#include <chrono>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace forge {

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

Generator::Generator(InferenceContext& ctx, const SamplerConfig& sampler_cfg)
    : ctx_(ctx), sampler_(sampler_cfg) {
#ifdef USE_CUDA
    if (ctx_.device() == DeviceType::CUDA) {
        decode_input_gpu_ = std::make_shared<Tensor>(DataType::INT32,
            std::vector<int64_t>{1}, DeviceType::CUDA);
        cudaMalloc(&decode_token_buf_, sizeof(int32_t));
    }
#endif
}

Generator::~Generator() {
#ifdef USE_CUDA
    if (decode_token_buf_) {
        cudaFree(decode_token_buf_);
        decode_token_buf_ = nullptr;
    }
#endif
}

GenerationResult Generator::generate(const std::vector<int32_t>& prompt_tokens,
                                      const GenerationConfig& config) {
    return generate(prompt_tokens, config, nullptr);
}

GenerationResult Generator::generate(const std::vector<int32_t>& prompt_tokens,
                                      const GenerationConfig& config,
                                      const TokenCallback& callback) {
    GenerationResult result;
    result.num_prompt_tokens = static_cast<int>(prompt_tokens.size());

    auto* engine = ctx_.engine();
    if (!engine) {
        throw std::runtime_error("Generator: no inference engine available");
    }

    const auto& cfg = ctx_.model().config();
    DeviceType dev = ctx_.device();

    SamplerConfig sampler_cfg;
    sampler_cfg.temperature = config.temperature;
    sampler_cfg.top_k = config.top_k;
    sampler_cfg.top_p = config.top_p;
    sampler_cfg.repeat_penalty = config.repeat_penalty;
    sampler_cfg.do_sample = config.do_sample;
    sampler_cfg.seed = config.seed;
    sampler_.set_config(sampler_cfg);
    sampler_.clear_history();

    ctx_.reset_kv_cache();

    int prompt_len = static_cast<int>(prompt_tokens.size());

    // ---- Prefill phase ----
    int token_id;
    {
        PERF_SCOPE("generator/prefill");
        auto input_ids = std::make_shared<Tensor>(DataType::INT32,
            std::vector<int64_t>{prompt_len}, DeviceType::CPU);
        std::memcpy(input_ids->data(), prompt_tokens.data(), prompt_len * sizeof(int32_t));

        if (dev == DeviceType::CUDA) {
            input_ids->to_device(DeviceType::CUDA);
        }

        auto logits = engine->forward(input_ids, 0);

        auto last_logits = std::make_shared<Tensor>(logits->slice(0, prompt_len - 1, prompt_len));

        {
            PERF_SCOPE("generator/prefill_sampler");
            token_id = sampler_.sample(last_logits, prompt_len - 1);
        }

        result.token_ids.push_back(token_id);

        if (callback) {
            callback(token_id, 0);
        }

        if (config.eos_token_id >= 0 && token_id == config.eos_token_id) {
            result.finished = true;
            result.finish_reason = "eos";
            result.num_generated_tokens = 1;
            return result;
        }
    }

    int64_t pos = prompt_len;

    // ---- Decode phase ----
    for (int step = 1; step < config.max_new_tokens; ++step) {
        int32_t last_token = result.token_ids.back();

        TensorPtr input_ids;
        {
            PERF_SCOPE("decode/prepare_input");
            if (dev == DeviceType::CUDA && decode_input_gpu_) {
#ifdef USE_CUDA
                cudaMemcpyAsync(decode_input_gpu_->data(), &last_token,
                               sizeof(int32_t), cudaMemcpyHostToDevice);
#endif
                input_ids = decode_input_gpu_;
            } else {
                input_ids = std::make_shared<Tensor>(DataType::INT32,
                    std::vector<int64_t>{1}, DeviceType::CPU);
                *static_cast<int32_t*>(input_ids->data()) = last_token;
            }
        }

        TensorPtr logits;
        {
            PERF_SCOPE("decode/forward");
            logits = engine->forward(input_ids, pos);
        }

        {
            PERF_SCOPE("decode/sampler");
            token_id = sampler_.sample(logits, pos);
        }

        result.token_ids.push_back(token_id);
        pos += 1;

        if (callback) {
            callback(token_id, step);
        }

        if (config.eos_token_id >= 0 && token_id == config.eos_token_id) {
            result.finished = true;
            result.finish_reason = "eos";
            break;
        }
    }

    if (!result.finished) {
        result.finished = true;
        result.finish_reason = "length";
    }

    result.num_generated_tokens = static_cast<int>(result.token_ids.size());

    return result;
}

} // namespace forge
