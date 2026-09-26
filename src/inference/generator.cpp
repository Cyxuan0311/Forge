#include "forge/generator.h"

#include <algorithm>
#include <limits>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include <vector>

#include "forge/engine.h"
#include "forge/engines/llama_engine.h"
#include "forge/inference/forward_request.h"
#include "forge/inference_batch.h"
#include "forge/kv_working_set.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"
#include "forge/speculative_executor.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static bool aborted(const GenerationConfig& config) {
    return config.abort_flag && config.abort_flag->load(std::memory_order_relaxed);
}

// ---- Prefill-only profiler snapshot (FORGE_PERF_PREFILL=1) ----
// The PerfProfiler is reset at the start of every request by the server, so at
// the moment right after the prefill forward returns (and before any decode
// step) its thread-local buffers hold *only* prefill work. Flushing the
// deferred CUDA events and dumping the summary here yields a clean prefill-only
// breakdown, isolated from the decode phase that would otherwise dominate the
// per-request --perf-detail totals for long generations.
// dump_phase_profile lives in perf_profiler.cpp (shared with --bench's decode
// dump); see perf_profiler.h for the contract.
static void dump_prefill_profile(double prefill_ms, int n_tokens) {
    auto& prof = forge::PerfProfiler::instance();
#ifdef USE_CUDA
    prof.flush_deferred();
#endif
    dump_phase_profile("PREFILL-ONLY", prefill_ms, n_tokens, prof.summary());
}

Generator::Generator(InferenceContext& ctx, const SamplerConfig& sampler_cfg)
    : ctx_(ctx), sampler_(sampler_cfg) {
#ifdef USE_CUDA
    if (ctx_.device() == DeviceType::CUDA) {
        decode_input_gpu_ =
            std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1}, DeviceType::CUDA);
        cudaMalloc(&decode_token_buf_, sizeof(int32_t));
    }
#endif

    if (ctx_.params().speculative_config.enabled) {
        spec_executor_ = std::make_unique<SpeculativeExecutor>(
            ctx_, sampler_, ctx_.params().speculative_config);
    }
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

const SpeculativeStats* Generator::spec_stats() const {
    return spec_executor_ ? &spec_executor_->stats() : nullptr;
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
    sampler_cfg.repeat_last_n = config.repeat_last_n;
    sampler_cfg.do_sample = config.do_sample;
    sampler_cfg.seed = config.seed;
    sampler_cfg.logit_softcapping = cfg.f_final_logit_softcapping;
    sampler_.set_config(sampler_cfg);
    sampler_.clear_history();

    int prompt_len = static_cast<int>(prompt_tokens.size());

    // ---- KVMem tiered working set ----
    // When tiered KV is attached, the *logical* context (-c) is what the caller
    // reasons about, but the engine only ever sees the bounded window. The
    // working set decides, once per turn, which prefix of the logical history is
    // still resident (kept_window_len slots) and where the window starts in
    // logical space (rope_pos_offset). Everything downstream -- the prefill
    // cursor, the attention mask, KV writes -- then uses window slots, while
    // RoPE uses logical positions, so a 256K conversation can run in a 32K GPU
    // window (arXiv:2609.04852).
    //
    // With no working set attached this whole block is skipped and every value
    // below keeps its legacy meaning, so the untiered path is unchanged.
    KVemTurnPlan ws_plan;
    bool tiered = false;
    int64_t rope_pos_offset = 0;
    if (auto* kv_mem = engine->kv_memory()) {
        if (auto* ws = kv_mem->working_set()) {
            tiered = true;
            // Per-request generation reserve: the window reserves decode headroom
            // for exactly this request's max_new_tokens (clamped inside the plan).
            ws_plan = ws->on_prompt(prompt_tokens, config.max_new_tokens);
            if (ws_plan.window_overflow) {
                throw std::runtime_error(
                    "Generator: KVMem window overflow -- the prompt does not fit the "
                    "working-set budget; raise --kvmem-budget or lower --ctx-size");
            }
            // Only the non-resident tail needs a forward; the resident prefix is
            // already in window slots [0, kept_window_len).
            prompt_len = static_cast<int>(ws_plan.new_to_compute.size());
            rope_pos_offset = ws_plan.new_rope_off - ws_plan.kept_window_len;
        }
    }

    // ---- Session prefix reuse ----
    // The caller guarantees that KV positions [0, prefix_len) already hold
    // prompt_tokens[0, prefix_len), so only the tail needs a forward. Clamped
    // to leave at least one token: we need a logits row to sample from.
    // In tiered mode prefix_len is expressed in *window* slots instead: the
    // resident prefix is kept_window_len slots, and the tail to forward is
    // exactly the window slots [kept_window_len, new_window_len).
    int prefix_len = 0;
    if (tiered) {
        prefix_len = static_cast<int>(ws_plan.kept_window_len);
    } else if (config.prefix_len > 0 && prompt_len > 1) {
        prefix_len = std::min(config.prefix_len, prompt_len - 1);
    }
    // Tiered: the resident prefix is already in window slots [0, kept_window_len)
    // and must NOT be re-forwarded, so the tail to compute is the whole
    // new_to_compute vector (length = kept_window_len + new_window_len tail),
    // not prompt_tokens[prefix_len:] which would drop kept_window_len tokens of
    // the assistant's turn and desync the output.
    const int fwd_len = tiered ? static_cast<int>(ws_plan.new_to_compute.size())
                               : prompt_len - prefix_len;

    // The KV write cursor is monotonic, so a previous request can leave rows
    // beyond prefix_len. Prefill masks them out (causal: j > q_pos), but decode
    // attends to everything below total_len, so rewind to the reuse point.
    // In tiered mode the window slot count after the previous turn is already
    // tracked by the working set, so recomputing the window is the caller's
    // side of the contract: the stale slots beyond kept_window_len are masked
    // by the causal prefill, and the window length the next turn reports comes
    // from the plan, not from the KV cursor.
    if (prefix_len > 0) {
        if (!tiered) {
            if (auto* kv = engine->kv_cache()) {
                kv->rollback(prefix_len);
            }
        }
    } else if (!tiered && config.reset_kv_cache) {
        ctx_.reset_kv_cache();
    }

    // ---- Prefill phase ----
    int token_id;
    {
        const auto prefill_start = std::chrono::steady_clock::now();
        PERF_SCOPE("generator/prefill");
        SET_PERF_CONTEXT(0, "prefill", -1, dev == DeviceType::CUDA ? "cuda" : "cpu", fwd_len);

        // ---- Prefill-only profiler snapshot (FORGE_PERF_PREFILL=1) ----
        // Turn the PerfProfiler on (CUDA-event accurate) so we can dump a clean
        // prefill-only breakdown right after this forward, before decode starts.
        // The server resets the profiler per request, so at dump time the buffers
        // hold only prefill work.
        if (std::getenv("FORGE_PERF_PREFILL")) {
            forge::PerfProfiler::instance().enable();
            forge::PerfProfiler::instance().set_use_cuda_events(true);
        }
        auto dump_prefill_if_enabled = [&]() {
            if (std::getenv("FORGE_PERF_PREFILL")) {
                dump_prefill_profile(result.prefill_ms, fwd_len);
            }
        };
        auto input_ids = std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{fwd_len},
                                                  DeviceType::CPU);
        // Tiered: forward ws_plan.new_to_compute (the non-resident tail), not
        // prompt_tokens[prefix_len:].
        const int32_t* fwd_src = tiered ? ws_plan.new_to_compute.data()
                                        : prompt_tokens.data() + prefix_len;
        std::memcpy(input_ids->data(), fwd_src, fwd_len * sizeof(int32_t));

        if (dev == DeviceType::CUDA) {
            input_ids->to_device(DeviceType::CUDA);
        }

        // In tiered mode the mask/KV cursors use window slots (prefix_len) while
        // RoPE must use logical positions: the window's first slot carries
        // logical position rope_pos_offset, which keeps archived KV at its true
        // relative distance. Untiered, rope_pos_offset stays 0 and this is a
        // no-op.
        auto prefill_req = ForwardRequest::from_ids(input_ids, prefix_len);
        prefill_req.rope_pos_offset = rope_pos_offset;
        auto logits = engine->forward_request(prefill_req);

        // The engine may return sparse logits ([1, vocab]: prefill projects only
        // the final row) or the legacy full [fwd_len, vocab] buffer; the row we
        // sample from is the last one either way. Slicing with fwd_len-1 here
        // used to read past the end of the sparse buffer (garbage token / SEGV).
        const int64_t logits_rows = logits->shape()[0];
        auto last_logits =
            std::make_shared<Tensor>(logits->slice(0, logits_rows - 1, logits_rows));

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
            dump_prefill_if_enabled();
            return result;
        }
        // Check additional stop tokens
        for (int stop_id : config.stop_token_ids) {
            if (token_id == stop_id) {
                result.finished = true;
                result.finish_reason = "stop";
                result.num_generated_tokens = 1;
                result.prefill_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - prefill_start)
                                        .count();
                dump_prefill_if_enabled();
                return result;
            }
        }
        result.prefill_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - prefill_start)
                                .count();
        dump_prefill_if_enabled();
    }

    // Tiered: the decode cursor starts right after the prefilled tail, i.e. the
    // full new window length (kept_window_len + new_to_compute.size()).
    int64_t pos = tiered ? static_cast<int64_t>(ws_plan.new_window_len) : prompt_len;

    // Tiered: cap generation at the remaining window slots so the pool never
    // overflows mid-reply. Non-tiered: unbounded.
    const int64_t gen_cap_steps = tiered
        ? std::max<int64_t>(0, ws_plan.max_generate)
        : (std::numeric_limits<int64_t>::max)();
    const auto decode_start = std::chrono::steady_clock::now();

    // ---- Decode-only profiler snapshot (FORGE_PERF_DECODE=1) ----
    // Enables the profiler with CUDA-event timing for the decode loop and
    // resets the accumulator so the decode_done dump holds only decode work.
    // Host-wall profiles mis-attribute async queue drains; CUDA events give
    // per-op device-stream time.
    const bool perf_decode = std::getenv("FORGE_PERF_DECODE") != nullptr;
    if (perf_decode) {
        forge::PerfProfiler::instance().reset();
        forge::PerfProfiler::instance().enable();
        forge::PerfProfiler::instance().set_use_cuda_events(true);
    }

    // ---- Decode phase (speculative decoding via SpeculativeExecutor) ----
    if (spec_executor_) {
        spec_executor_->begin_generation(prompt_tokens);
    }

    for (int step = 1; step < config.max_new_tokens && step <= gen_cap_steps;) {
        if (aborted(config)) {
            goto decode_done;
        }
        // Speculation round: draft -> verify -> accept
        if (spec_executor_ && spec_executor_->valid()) {
            auto out = spec_executor_->step(
                {result.token_ids.back(), pos, config.max_new_tokens - step});
            if (out.speculated) {
                const int64_t step_start_pos = pos;
                size_t consumed = 0;
                bool stop_gen = false;

                for (int32_t tok : out.tokens) {
                    // Tiered: a speculative draft can exceed the remaining window
                    // slots; stop before writing past the pool.
                    if (tiered && step > gen_cap_steps) {
                        stop_gen = true;
                        break;
                    }
                    result.token_ids.push_back(tok);
                    pos += 1;
                    step += 1;
                    consumed += 1;
                    if (callback) callback(tok, step - 1);

                    if (aborted(config)) {
                        stop_gen = true;
                        break;
                    }
                    if (config.eos_token_id >= 0 && tok == config.eos_token_id) {
                        result.finished = true;
                        result.finish_reason = "eos";
                        stop_gen = true;
                        break;
                    }
                    bool hit_stop = false;
                    for (int stop_id : config.stop_token_ids) {
                        if (tok == stop_id) {
                            hit_stop = true;
                            break;
                        }
                    }
                    if (hit_stop) {
                        result.finished = true;
                        result.finish_reason = "stop";
                        stop_gen = true;
                        break;
                    }
                }

                // Early stop mid-output: truncate leftover speculated KV rows.
                if (consumed < out.tokens.size()) {
                    spec_executor_->rollback_kv(step_start_pos +
                                                static_cast<int64_t>(consumed));
                }
                if (stop_gen) goto decode_done;
                continue;
            }
        }

        // Plain decode fallback: one token per forward.
        {
            int32_t last_token = result.token_ids.back();

        TensorPtr input_ids;
        {
            PERF_SCOPE("decode/prepare_input");
            if (dev == DeviceType::CUDA && decode_input_gpu_) {
#ifdef USE_CUDA
                cudaMemcpyAsync(decode_input_gpu_->data(), &last_token, sizeof(int32_t),
                                cudaMemcpyHostToDevice);
#endif
                input_ids = decode_input_gpu_;
            } else {
                input_ids = std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1},
                                                     DeviceType::CPU);
                *static_cast<int32_t*>(input_ids->data()) = last_token;
            }
        }

        TensorPtr logits;
        {
            PERF_SCOPE("decode/forward");
            SET_PERF_CONTEXT(0, "decode", -1, dev == DeviceType::CUDA ? "cuda" : "cpu", 1);
            // Tiered mode: `pos` is the window cursor, so RoPE needs the
            // logical offset added back in (see the prefill forward above).
            auto decode_req = ForwardRequest::from_ids(input_ids, pos);
            decode_req.rope_pos_offset = rope_pos_offset;
            logits = engine->forward_request(decode_req);
        }

        {
            PERF_SCOPE("decode/sampler");
            token_id = sampler_.sample(logits, pos);
        }

        result.token_ids.push_back(token_id);
        pos += 1;
        step++;


        if (spec_executor_ && spec_executor_->valid()) {
            spec_executor_->notify_confirmed(token_id);
        }

        if (callback) {
            callback(token_id, step - 1);
        }

        if (aborted(config)) {
            break;
        }

        if (config.eos_token_id >= 0 && token_id == config.eos_token_id) {
            result.finished = true;
            result.finish_reason = "eos";
            break;
        }
        // Check additional stop tokens
        bool hit_stop = false;
        for (int stop_id : config.stop_token_ids) {
            if (token_id == stop_id) {
                hit_stop = true;
                break;
            }
        }
        if (hit_stop) {
            result.finished = true;
            result.finish_reason = "stop";
            break;
        }
        }  // end normal_decode block
    }

decode_done:
    if (!result.finished) {
        result.finished = true;
        result.finish_reason = aborted(config) ? "aborted" : "length";
    }

    result.num_generated_tokens = static_cast<int>(result.token_ids.size());
    result.decode_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - decode_start)
                          .count();
    result.spec_stats = spec_executor_ ? &spec_executor_->stats() : nullptr;

    if (perf_decode) {
        auto& prof = forge::PerfProfiler::instance();
#ifdef USE_CUDA
        prof.flush_deferred();
#endif
        dump_phase_profile("DECODE-ONLY", result.decode_ms,
                           result.num_generated_tokens, prof.summary());
    }
    return result;
}

}  // namespace forge
