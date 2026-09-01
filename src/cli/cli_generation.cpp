/**
 * Forge CLI - Streaming and batch generation
 */

#include <chrono>
#include <cstring>
#include <iostream>

#include "cli_common.h"
#include "forge/context.h"
#include "forge/engine.h"
#include "forge/generator.h"
#include "forge/inference/forward_request.h"
#include "forge/sampler.h"
#include "forge/tensor.h"
#include "forge/tokenizer.h"
#include "forge/types.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

using namespace forge;

GenerationStats generate_streaming(InferenceContext& ctx, const Tokenizer& tokenizer,
                                   const std::vector<int32_t>& prompt_tokens, int max_new_tokens,
                                   float temperature, int top_k, float top_p, float repeat_penalty,
                                   int repeat_last_n, bool do_sample, uint64_t seed,
                                   int eos_token_id, const SpeculativeConfig& spec_cfg) {
    // Apply speculative config to context before generation
    if (spec_cfg.enabled) {
        ctx.params_mut().speculative_config = spec_cfg;
    }

    GenerationConfig gen_cfg;
    gen_cfg.max_new_tokens = max_new_tokens;
    gen_cfg.temperature = temperature;
    gen_cfg.top_k = top_k;
    gen_cfg.top_p = top_p;
    gen_cfg.repeat_penalty = repeat_penalty;
    gen_cfg.repeat_last_n = repeat_last_n;
    gen_cfg.do_sample = do_sample;
    gen_cfg.seed = seed;
    gen_cfg.eos_token_id = eos_token_id;

    SamplerConfig sampler_cfg;
    sampler_cfg.temperature = temperature;
    sampler_cfg.top_k = top_k;
    sampler_cfg.top_p = top_p;
    sampler_cfg.repeat_penalty = repeat_penalty;
    sampler_cfg.repeat_last_n = repeat_last_n;
    sampler_cfg.do_sample = do_sample;
    sampler_cfg.seed = seed;
    sampler_cfg.logit_softcapping = ctx.model().config().f_final_logit_softcapping;

    Generator gen(ctx, sampler_cfg);

    // Streaming decode via the Generator. When speculative decoding is enabled the
    // Generator routes through SpeculativeExecutor, so draft/verify/accept (and the
    // resulting stats) are now exercised in streaming mode too. Flush decoded text in
    // small chunks to mimic the previous incremental-output behavior.
    std::vector<int32_t> token_buffer;
    auto stream_cb = [&](int32_t tok, int /*step*/) {
        token_buffer.push_back(tok);
        if (token_buffer.size() >= 4) {
            std::cout << tokenizer.decode(token_buffer, true, false);
            std::cout.flush();
            token_buffer.clear();
        }
    };

    auto t_start = std::chrono::high_resolution_clock::now();
    auto result = gen.generate(prompt_tokens, gen_cfg, stream_cb);
    auto t_end = std::chrono::high_resolution_clock::now();

    if (!token_buffer.empty()) {
        std::cout << tokenizer.decode(token_buffer, true, false);
        std::cout.flush();
    }

    GenerationStats stats;
    stats.num_prompt_tokens = result.num_prompt_tokens;
    stats.num_generated_tokens = result.num_generated_tokens;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (result.spec_stats) {
        stats.spec.n_spec_steps = result.spec_stats->n_spec_steps;
        stats.spec.n_fallback_steps = result.spec_stats->n_fallback_steps;
        stats.spec.n_draft_tokens = result.spec_stats->n_draft_tokens;
        stats.spec.n_accepted_tokens = result.spec_stats->n_accepted_tokens;
        stats.spec.n_output_tokens = result.spec_stats->n_output_tokens;
    }

    return stats;
}

GenerationStats generate_batch(InferenceContext& ctx, const Tokenizer& tokenizer,
                               const std::vector<int32_t>& prompt_tokens, int max_new_tokens,
                               float temperature, int top_k, float top_p, float repeat_penalty,
                               int repeat_last_n, bool do_sample, uint64_t seed, int eos_token_id,
                               const SpeculativeConfig& spec_cfg) {
    // Apply speculative config to context before generation
    if (spec_cfg.enabled) {
        ctx.params_mut().speculative_config = spec_cfg;
    }
    GenerationConfig gen_cfg;
    gen_cfg.max_new_tokens = max_new_tokens;
    gen_cfg.temperature = temperature;
    gen_cfg.top_k = top_k;
    gen_cfg.top_p = top_p;
    gen_cfg.repeat_penalty = repeat_penalty;
    gen_cfg.repeat_last_n = repeat_last_n;
    gen_cfg.do_sample = do_sample;
    gen_cfg.seed = seed;
    gen_cfg.eos_token_id = eos_token_id;

    SamplerConfig sampler_cfg;
    sampler_cfg.temperature = temperature;
    sampler_cfg.top_k = top_k;
    sampler_cfg.top_p = top_p;
    sampler_cfg.repeat_penalty = repeat_penalty;
    sampler_cfg.repeat_last_n = repeat_last_n;
    sampler_cfg.do_sample = do_sample;
    sampler_cfg.seed = seed;
    sampler_cfg.logit_softcapping = ctx.model().config().f_final_logit_softcapping;

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
    if (result.spec_stats) {
        stats.spec.n_spec_steps = result.spec_stats->n_spec_steps;
        stats.spec.n_fallback_steps = result.spec_stats->n_fallback_steps;
        stats.spec.n_draft_tokens = result.spec_stats->n_draft_tokens;
        stats.spec.n_accepted_tokens = result.spec_stats->n_accepted_tokens;
        stats.spec.n_output_tokens = result.spec_stats->n_output_tokens;
    }

    return stats;
}
