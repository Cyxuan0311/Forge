/**
 * NanoInfer CLI - Streaming and batch generation
 */

#include "cli_common.h"

#include <iostream>
#include <cstring>
#include <chrono>

#include "nanoinfer/tokenizer.h"
#include "nanoinfer/context.h"
#include "nanoinfer/generator.h"
#include "nanoinfer/sampler.h"
#include "nanoinfer/engine.h"
#include "nanoinfer/types.h"
#include "nanoinfer/tensor.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace nanoinfer;

GenerationStats generate_streaming(
    InferenceContext& ctx,
    const Tokenizer& tokenizer,
    const std::vector<int32_t>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    float repeat_penalty,
    bool do_sample,
    uint64_t seed,
    int eos_token_id)
{
    GenerationStats stats;
    stats.num_prompt_tokens = static_cast<int>(prompt_tokens.size());

    auto* engine = ctx.engine();
    if (!engine) {
        std::cerr << "Error: No inference engine\n";
        return stats;
    }

    DeviceType dev = ctx.device();

    SamplerConfig sampler_cfg;
    sampler_cfg.temperature = temperature;
    sampler_cfg.top_k = top_k;
    sampler_cfg.top_p = top_p;
    sampler_cfg.repeat_penalty = repeat_penalty;
    sampler_cfg.do_sample = do_sample;
    sampler_cfg.seed = seed;
    Sampler sampler(sampler_cfg);

    ctx.reset_kv_cache();

    int prompt_len = static_cast<int>(prompt_tokens.size());

    // ---- Prefill phase ----
    auto t_prefill_start = std::chrono::high_resolution_clock::now();

    auto input_ids = std::make_shared<Tensor>(DataType::INT32,
        std::vector<int64_t>{prompt_len}, DeviceType::CPU);
    std::memcpy(input_ids->data(), prompt_tokens.data(), prompt_len * sizeof(int32_t));
    if (dev == DeviceType::CUDA) {
        input_ids->to_device(DeviceType::CUDA);
    }

    auto logits = engine->forward(input_ids, 0);

    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    stats.prompt_eval_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    auto last_logits = std::make_shared<Tensor>(logits->slice(0, prompt_len - 1, prompt_len));
    int token_id = sampler.sample(last_logits, prompt_len - 1);

    std::vector<int32_t> token_buffer;
    token_buffer.push_back(token_id);
    stats.num_generated_tokens = 1;

    if (token_id == eos_token_id) {
        auto text = tokenizer.decode(token_buffer, true, false);
        std::cout << text;
        std::cout.flush();
        stats.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_prefill_start).count();
        return stats;
    }

    // ---- Decode phase ----
    auto t_decode_start = std::chrono::high_resolution_clock::now();

    TensorPtr decode_input_gpu;
    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        decode_input_gpu = std::make_shared<Tensor>(DataType::INT32,
            std::vector<int64_t>{1}, DeviceType::CUDA);
#endif
    }

    int64_t pos = prompt_len;

    for (int step = 1; step < max_new_tokens && !g_interrupted; ++step) {
        int32_t last_token = token_id;

        TensorPtr next_input;
        if (dev == DeviceType::CUDA && decode_input_gpu) {
#ifdef USE_CUDA
            cudaMemcpyAsync(decode_input_gpu->data(), &last_token,
                           sizeof(int32_t), cudaMemcpyHostToDevice);
#endif
            next_input = decode_input_gpu;
        } else {
            next_input = std::make_shared<Tensor>(DataType::INT32,
                std::vector<int64_t>{1}, DeviceType::CPU);
            *static_cast<int32_t*>(next_input->data()) = last_token;
        }

        logits = engine->forward(next_input, pos);
        token_id = sampler.sample(logits, pos);
        pos += 1;
        stats.num_generated_tokens++;

        token_buffer.push_back(token_id);

        if (token_buffer.size() >= 4 || token_id == eos_token_id) {
            auto text = tokenizer.decode(token_buffer, true, false);
            std::cout << text;
            std::cout.flush();
            token_buffer.clear();
        }

        if (token_id == eos_token_id) {
            break;
        }
    }

    if (!token_buffer.empty()) {
        auto text = tokenizer.decode(token_buffer, true, false);
        std::cout << text;
        std::cout.flush();
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

    return stats;
}

GenerationStats generate_batch(
    InferenceContext& ctx,
    const Tokenizer& tokenizer,
    const std::vector<int32_t>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    float repeat_penalty,
    bool do_sample,
    uint64_t seed,
    int eos_token_id)
{
    GenerationConfig gen_cfg;
    gen_cfg.max_new_tokens = max_new_tokens;
    gen_cfg.temperature = temperature;
    gen_cfg.top_k = top_k;
    gen_cfg.top_p = top_p;
    gen_cfg.repeat_penalty = repeat_penalty;
    gen_cfg.do_sample = do_sample;
    gen_cfg.seed = seed;
    gen_cfg.eos_token_id = eos_token_id;

    SamplerConfig sampler_cfg;
    sampler_cfg.temperature = temperature;
    sampler_cfg.top_k = top_k;
    sampler_cfg.top_p = top_p;
    sampler_cfg.repeat_penalty = repeat_penalty;
    sampler_cfg.do_sample = do_sample;
    sampler_cfg.seed = seed;

    Generator gen(ctx, sampler_cfg);

    auto t_start = std::chrono::high_resolution_clock::now();
    auto result = gen.generate(prompt_tokens, gen_cfg);
    auto t_end = std::chrono::high_resolution_clock::now();

    auto text = tokenizer.decode(result.token_ids, true, false);
    std::cout << text;
    std::cout.flush();

    GenerationStats stats;
    stats.num_prompt_tokens = result.num_prompt_tokens;
    stats.num_generated_tokens = result.num_generated_tokens;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return stats;
}
