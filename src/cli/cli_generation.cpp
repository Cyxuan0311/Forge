/**
 * Forge CLI - Streaming and batch generation
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "cli_common.h"
#include "forge/context.h"
#include "forge/engine.h"
#include "forge/generator.h"
#include "forge/inference/forward_request.h"
#include "forge/kv_memory.h"
#include "forge/prefix_cache.h"
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
                                   int eos_token_id, const SpeculativeConfig& spec_cfg,
                                   forge::PrefixCache* prefix_cache,
                                   std::string* out_text) {
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

    // Paged-mode prefix reuse (radix PrefixCache). Mirrors forge-server: each
    // request's working set is seq_id 0; on a hit we share the cached prefix pages
    // into seq 0 via one seq_share and skip prefilling them. We never reset the whole
    // KV cache (that would wipe the radix cache's seq_ids); release(0) clears only the
    // previous request's working set. See PrefixCache / RequestScheduler for details.
    const bool paged = (ctx.params().kv_storage_mode == KVStorageMode::Paged);
    // KVMem: seq 0's resident window IS the conversation state, tracked across turns
    // by the working-set manager. Releasing it, or sharing its pages into radix cache
    // seq_ids, invalidates the window layout on_prompt() plans against -- the next
    // turn would then skip prefilling a prefix that no longer exists. Mirrors
    // forge-server's tiered branch (server_state.cpp).
    const bool tiered = paged && ctx.engine()->kv_memory() &&
                        ctx.engine()->kv_memory()->working_set() != nullptr;
    if (paged && prefix_cache && !tiered) {
        if (auto* km = ctx.engine()->kv_memory()) {
            km->storage().release(0);
            gen_cfg.prefix_len = prefix_cache->try_lookup(prompt_tokens, 0, km->storage());
        }
        gen_cfg.reset_kv_cache = false;
    }

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

    if (std::getenv("FORGE_GEN_DEBUG")) {
        std::fprintf(stderr, "[gen-debug] prompt_ids(%zu):", prompt_tokens.size());
        for (int32_t t : prompt_tokens) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n");
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    auto result = gen.generate(prompt_tokens, gen_cfg, stream_cb);
    auto t_end = std::chrono::high_resolution_clock::now();

    if (std::getenv("FORGE_GEN_DEBUG")) {
        std::fprintf(stderr, "[gen-debug] gen_ids(%zu):", result.token_ids.size());
        for (int32_t t : result.token_ids) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n[gen-debug] per-token decode:");
        for (int32_t t : result.token_ids) {
            std::fprintf(stderr, " [%d]=%s", t,
                         tokenizer.decode(std::vector<int32_t>{t}, true, false).c_str());
        }
        std::fprintf(stderr, "\n");
    }

    // Decode the reply once, from the token ids, rather than reassembling the
    // streamed chunks: streaming decodes in 4-token groups, so the chunk
    // boundaries can split a multi-byte character.
    if (out_text && !result.token_ids.empty())
        *out_text = tokenizer.decode(result.token_ids, true, false);

    if (paged && prefix_cache && !tiered) {
        if (auto* km = ctx.engine()->kv_memory()) {
            // Register prompt AND reply: decode wrote result.token_ids into seq 0
            // right after the prompt, and the next turn's prompt begins with
            // exactly that sequence. Registering only the prompt left the reply's
            // KV unreferenced, and the next turn's release(0) discarded it -- so
            // every turn re-prefilled the whole previous assistant message.
            std::vector<int32_t> resident(prompt_tokens);
            resident.insert(resident.end(), result.token_ids.begin(),
                            result.token_ids.end());
            prefix_cache->register_prefix(resident, 0, km->storage());
            prefix_cache->evict_lru(PrefixCache::DEFAULT_MAX_ENTRIES, km->storage());
        }
    }

    if (!token_buffer.empty()) {
        const std::string tail_text = tokenizer.decode(token_buffer, true, false);
        std::cout << tail_text;
        std::cout.flush();
        if (out_text)
            *out_text += tail_text;
    }

    GenerationStats stats;
    stats.num_prompt_tokens = result.num_prompt_tokens;
    stats.num_generated_tokens = result.num_generated_tokens;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (result.spec_stats) {
        // Copy the whole struct rather than a hand-picked field list, which is
        // how the fallback-reason and per-position counters silently read as
        // zero in --spec-stats output.
        stats.spec = *result.spec_stats;
    }

    return stats;
}

GenerationStats generate_batch(InferenceContext& ctx, const Tokenizer& tokenizer,
                               const std::vector<int32_t>& prompt_tokens, int max_new_tokens,
                               float temperature, int top_k, float top_p, float repeat_penalty,
                               int repeat_last_n, bool do_sample, uint64_t seed, int eos_token_id,
                               const SpeculativeConfig& spec_cfg, forge::PrefixCache* prefix_cache,
                               std::string* out_text) {
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

    // Paged-mode prefix reuse (radix PrefixCache): share cached prefix on hit,
    // register this request's prefix afterwards. Never reset the whole cache.
    const bool paged = (ctx.params().kv_storage_mode == KVStorageMode::Paged);
    // See the streaming path above: under KVMem the tier owns seq 0's window.
    const bool tiered = paged && ctx.engine()->kv_memory() &&
                        ctx.engine()->kv_memory()->working_set() != nullptr;
    if (paged && prefix_cache && !tiered) {
        if (auto* km = ctx.engine()->kv_memory()) {
            km->storage().release(0);
            gen_cfg.prefix_len = prefix_cache->try_lookup(prompt_tokens, 0, km->storage());
        }
        gen_cfg.reset_kv_cache = false;
    }

    Generator gen(ctx, sampler_cfg);

    if (std::getenv("FORGE_GEN_DEBUG")) {
        std::fprintf(stderr, "[gen-debug] prompt_ids(%zu):", prompt_tokens.size());
        for (int32_t t : prompt_tokens) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n");
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    auto result = gen.generate(prompt_tokens, gen_cfg);
    auto t_end = std::chrono::high_resolution_clock::now();

    if (std::getenv("FORGE_GEN_DEBUG")) {
        std::fprintf(stderr, "[gen-debug] gen_ids(%zu):", result.token_ids.size());
        for (int32_t t : result.token_ids) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n[gen-debug] per-token decode:");
        for (int32_t t : result.token_ids) {
            std::fprintf(stderr, " [%d]=%s", t,
                         tokenizer.decode(std::vector<int32_t>{t}, true, false).c_str());
        }
        std::fprintf(stderr, "\n");
    }

    if (paged && prefix_cache && !tiered) {
        if (auto* km = ctx.engine()->kv_memory()) {
            std::vector<int32_t> resident(prompt_tokens);
            resident.insert(resident.end(), result.token_ids.begin(),
                            result.token_ids.end());
            prefix_cache->register_prefix(resident, 0, km->storage());
            prefix_cache->evict_lru(PrefixCache::DEFAULT_MAX_ENTRIES, km->storage());
        }
    }

    auto text = tokenizer.decode(result.token_ids, true, false);
    if (out_text)
        *out_text = text;
    std::cout << text;
    std::cout.flush();

    GenerationStats stats;
    stats.num_prompt_tokens = result.num_prompt_tokens;
    stats.num_generated_tokens = result.num_generated_tokens;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (result.spec_stats) {
        // Copy the whole struct rather than a hand-picked field list, which is
        // how the fallback-reason and per-position counters silently read as
        // zero in --spec-stats output.
        stats.spec = *result.spec_stats;
    }

    return stats;
}
