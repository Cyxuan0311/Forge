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
#include "forge/metrics.h"
#include "forge/perf_profiler.h"
#include "forge/prefix_cache.h"
#include "forge/sampler.h"
#include "forge/tensor.h"
#include "forge/tokenizer.h"
#include "forge/types.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

using namespace forge;

// One-shot completion and --bench do not keep a conversation between turns, so
// the dashboard reports 0 sessions there and 1 in interactive mode (which sets
// it explicitly via publish_cli_engine_gauges's `sessions` argument).
static constexpr int kCliSessions = 0;

// Opens a per-request PerfProfiler window for the dashboard's operator panel.
//
// Only called when the user passed --dashboard-ops. Recording a scope per
// operator is not free, so arming it implicitly whenever the dashboard happened
// to be enabled would trade decode throughput for a chart nobody asked for.
//
// Takes the scope for the same reason end_operator_window() does: when the
// dashboard is off, scope.recording() is false and there is nothing to attach
// rows to, so the profiler must not be armed either. Checking it here rather
// than only at close time is what keeps --dashboard-ops from slowing decode
// down for a panel that will never be rendered.
static void begin_operator_window(const forge::MetricsRegistry::RequestScope& scope) {
    if (!scope.recording())
        return;
    PerfProfiler::instance().reset();
    PerfProfiler::instance().enable();
}

// Closes the window opened by begin_operator_window() and attaches the rows to
// `scope`. Must be called AT MOST ONCE per window: PerfProfiler::summary()
// merges every thread-local accumulator into the global records_ on each call,
// so a second call in the same window double-counts every operator (the same
// trap forge-server's --perf-detail path documents).
static void end_operator_window(forge::MetricsRegistry::RequestScope& scope) {
    if (!scope.recording())
        return;
#ifdef USE_CUDA
    // Merge CUDA-event timings recorded asynchronously during the request.
    PerfProfiler::instance().flush_deferred();
#endif
    auto rows = PerfProfiler::instance().summary();
    std::sort(rows.begin(), rows.end(),
              [](const std::pair<std::string, PerfRecord>& a,
                 const std::pair<std::string, PerfRecord>& b) {
                  return a.second.total_ms > b.second.total_ms;
              });
    std::vector<forge::OperatorRecord> ops;
    ops.reserve(rows.size());
    for (auto& [name, r] : rows) {
        forge::OperatorRecord o;
        o.name = name;
        o.total_ms = r.total_ms;
        o.count = r.count;
        o.min_ms = r.min_ms;
        o.max_ms = r.max_ms;
        o.avg_ms = r.count > 0 ? r.total_ms / static_cast<double>(r.count) : 0.0;
        ops.push_back(std::move(o));
    }
    scope.set_operators(std::move(ops));
}

void fill_cli_engine_gauges(InferenceContext& ctx, const PrefixCache* prefix_cache, int sessions,
                            forge::EngineGauges& g) {
    g.sessions = sessions;
    // The CLI keeps its PrefixCache as a function-local (interactive mode owns
    // one per process; the one-shot path builds one inline), so the hit rate has
    // to be handed in rather than read off the context.
    if (prefix_cache) {
        g.prefix_cache_hits = prefix_cache->hits();
        g.prefix_cache_misses = prefix_cache->misses();
        g.prefix_cache_host_hits = prefix_cache->host_hits();
    }
    if (auto* eng = ctx.engine()) {
        if (auto* mem = eng->kv_memory()) {
            g.paged_kv = mem->is_paged();
            g.tiered_kv = mem->tiered();
            auto& st = mem->storage();
            g.kv_total_bytes = static_cast<int64_t>(st.nbytes());
            g.kv_active_bytes = static_cast<int64_t>(st.active_bytes());
            g.kv_free_pages = mem->is_paged() ? st.num_free_pages() : -1;
            g.kv_pool_tokens = mem->is_paged() ? mem->max_pool_tokens() : 0;
            g.kv_page_size = st.page_size();
        }
    }
}

void publish_cli_engine_gauges(InferenceContext& ctx, const PrefixCache* prefix_cache,
                               int sessions) {
    auto& reg = forge::MetricsRegistry::instance();
    if (!reg.enabled())
        return;
    forge::EngineGauges g;
    fill_cli_engine_gauges(ctx, prefix_cache, sessions, g);
    reg.publish_engine_gauges(g);
}

GenerationStats generate_streaming(InferenceContext& ctx, const Tokenizer& tokenizer,
                                   const std::vector<int32_t>& prompt_tokens, int max_new_tokens,
                                   float temperature, int top_k, float top_p, float repeat_penalty,
                                   int repeat_last_n, bool do_sample, uint64_t seed,
                                   int eos_token_id, const SpeculativeConfig& spec_cfg,
                                   forge::PrefixCache* prefix_cache, std::string* out_text,
                                   bool capture_operators) {
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
    const bool tiered =
        paged && ctx.engine()->kv_memory() && ctx.engine()->kv_memory()->working_set() != nullptr;
    // Hybrid engines keep per-seq state outside the paged KV storage; radix
    // hits would resume the suffix with zeroed state (see prefix_cache_supported).
    const bool prefix_ok = prefix_cache && ctx.engine() && ctx.engine()->prefix_cache_supported();
    if (paged && prefix_ok && !tiered) {
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
    // Dashboard per-request handle. The CLI is single-threaded, so there is no
    // queue to wait behind and queue_ms will be ~0 -- the same code path is kept
    // anyway so the two binaries report identical fields.
    forge::MetricsRegistry::RequestScope scope(/*streaming=*/true);
    scope.set_prompt_tokens(static_cast<int>(prompt_tokens.size()));
    if (capture_operators)
        begin_operator_window(scope);

    std::vector<int32_t> token_buffer;
    auto stream_cb = [&](int32_t tok, int /*step*/) {
        // TTFT and inter-token latency come from here, not from
        // generator.cpp: this callback already fires once per emitted token, so
        // the decode loop needs no new instrumentation at all.
        if (scope.recording())
            scope.on_token();
        token_buffer.push_back(tok);
        if (token_buffer.size() >= 4) {
            std::cout << tokenizer.decode(token_buffer, true, false);
            std::cout.flush();
            token_buffer.clear();
        }
    };

    if (std::getenv("FORGE_GEN_DEBUG")) {
        std::fprintf(stderr, "[gen-debug] prompt_ids(%zu):", prompt_tokens.size());
        for (int32_t t : prompt_tokens)
            std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n");
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    auto result = gen.generate(prompt_tokens, gen_cfg, stream_cb);
    auto t_end = std::chrono::high_resolution_clock::now();

    if (std::getenv("FORGE_GEN_DEBUG")) {
        std::fprintf(stderr, "[gen-debug] gen_ids(%zu):", result.token_ids.size());
        for (int32_t t : result.token_ids)
            std::fprintf(stderr, " %d", t);
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

    if (paged && prefix_ok && !tiered) {
        if (auto* km = ctx.engine()->kv_memory()) {
            // Register prompt AND reply: decode wrote result.token_ids into seq 0
            // right after the prompt, and the next turn's prompt begins with
            // exactly that sequence. Registering only the prompt left the reply's
            // KV unreferenced, and the next turn's release(0) discarded it -- so
            // every turn re-prefilled the whole previous assistant message.
            std::vector<int32_t> resident(prompt_tokens);
            resident.insert(resident.end(), result.token_ids.begin(), result.token_ids.end());
            prefix_cache->register_prefix(resident, 0, km->storage());
            prefix_cache->evict_lru(PrefixCache::DEFAULT_MAX_ENTRIES, km->storage());
        }
    }
    if (scope.recording()) {
        scope.set_prefix_reused(gen_cfg.prefix_len);
        if (capture_operators)
            end_operator_window(scope);
        scope.finish(result, forge::RequestStatus::Ok);
    }
    // Published after the request, from the thread that owns the context: the
    // dashboard thread must not read KV occupancy itself (see
    // publish_cli_engine_gauges).
    publish_cli_engine_gauges(ctx, prefix_cache, kCliSessions);

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
    stats.prompt_eval_ms = result.prefill_ms;
    stats.decode_ms = result.decode_ms;
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
                               std::string* out_text, bool capture_operators) {
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
    const bool tiered =
        paged && ctx.engine()->kv_memory() && ctx.engine()->kv_memory()->working_set() != nullptr;
    const bool prefix_ok = prefix_cache && ctx.engine() && ctx.engine()->prefix_cache_supported();
    if (paged && prefix_ok && !tiered) {
        if (auto* km = ctx.engine()->kv_memory()) {
            km->storage().release(0);
            gen_cfg.prefix_len = prefix_cache->try_lookup(prompt_tokens, 0, km->storage());
        }
        gen_cfg.reset_kv_cache = false;
    }

    Generator gen(ctx, sampler_cfg);

    // --no-stream has no per-token callback, so ITL degrades to the aggregate
    // decode rate here; TTFT is still exact because the first token comes out of
    // prefill. The scope is created at the same point as in the streaming path so
    // the two produce the same field set.
    forge::MetricsRegistry::RequestScope scope(/*streaming=*/false);
    scope.set_prompt_tokens(static_cast<int>(prompt_tokens.size()));
    if (capture_operators)
        begin_operator_window(scope);

    if (std::getenv("FORGE_GEN_DEBUG")) {
        std::fprintf(stderr, "[gen-debug] prompt_ids(%zu):", prompt_tokens.size());
        for (int32_t t : prompt_tokens)
            std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n");
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    auto result = gen.generate(prompt_tokens, gen_cfg);
    auto t_end = std::chrono::high_resolution_clock::now();
    if (scope.recording()) {
        // No callback fired, so stamp the first token at the end of the call:
        // generation is synchronous, so "now" is the completion instant and the
        // resulting ttft_ms equals total_ms, which is the honest answer for a
        // request whose first byte is only observable after it finishes.
        scope.on_token();
        scope.set_prefix_reused(gen_cfg.prefix_len);
        if (capture_operators)
            end_operator_window(scope);
        scope.finish(result, forge::RequestStatus::Ok);
    }

    if (std::getenv("FORGE_GEN_DEBUG")) {
        std::fprintf(stderr, "[gen-debug] gen_ids(%zu):", result.token_ids.size());
        for (int32_t t : result.token_ids)
            std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n[gen-debug] per-token decode:");
        for (int32_t t : result.token_ids) {
            std::fprintf(stderr, " [%d]=%s", t,
                         tokenizer.decode(std::vector<int32_t>{t}, true, false).c_str());
        }
        std::fprintf(stderr, "\n");
    }

    if (paged && prefix_ok && !tiered) {
        if (auto* km = ctx.engine()->kv_memory()) {
            std::vector<int32_t> resident(prompt_tokens);
            resident.insert(resident.end(), result.token_ids.begin(), result.token_ids.end());
            prefix_cache->register_prefix(resident, 0, km->storage());
            prefix_cache->evict_lru(PrefixCache::DEFAULT_MAX_ENTRIES, km->storage());
        }
    }

    // Same rationale as the streaming path above: publish after the prefix
    // registration so the hit rate is current.
    publish_cli_engine_gauges(ctx, prefix_cache, kCliSessions);

    auto text = tokenizer.decode(result.token_ids, true, false);
    if (out_text)
        *out_text = text;
    std::cout << text;
    std::cout.flush();

    GenerationStats stats;
    stats.num_prompt_tokens = result.num_prompt_tokens;
    stats.num_generated_tokens = result.num_generated_tokens;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    stats.prompt_eval_ms = result.prefill_ms;
    stats.decode_ms = result.decode_ms;
    if (result.spec_stats) {
        // Copy the whole struct rather than a hand-picked field list, which is
        // how the fallback-reason and per-position counters silently read as
        // zero in --spec-stats output.
        stats.spec = *result.spec_stats;
    }

    return stats;
}
