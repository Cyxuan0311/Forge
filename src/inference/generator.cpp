#include "forge/generator.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "forge/engine.h"
#include "forge/engines/llama_engine.h"
#include "forge/inference/forward_request.h"
#include "forge/inference_batch.h"
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
static void dump_prefill_profile(double prefill_ms, int n_tokens) {
    auto& prof = forge::PerfProfiler::instance();
#ifdef USE_CUDA
    prof.flush_deferred();
#endif
    auto rows = prof.summary();
    if (rows.empty()) {
        fprintf(stderr,
                "[prefill-profile] no operator data. Build with -DFORGE_PROFILING=ON and "
                "run with --perf-detail (or just FORGE_PERF_PREFILL=1) to capture it.\n");
        return;
    }

    // Split layer-scoped records (forward/layer_*) from everything else so we can
    // print layers in index order and the rest sorted by cost.
    std::vector<std::pair<std::string, forge::PerfRecord>> layers, others;
    for (auto& r : rows) {
        if (r.first.rfind("forward/layer_", 0) == 0)
            layers.push_back(r);
        else
            others.push_back(r);
    }
    std::sort(layers.begin(), layers.end(), [](const auto& a, const auto& b) {
        int la = 0, lb = 0;
        sscanf(a.first.c_str(), "forward/layer_%d", &la);
        sscanf(b.first.c_str(), "forward/layer_%d", &lb);
        return la < lb;
    });
    std::sort(others.begin(), others.end(),
              [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });

    double layer_total = 0.0;
    for (auto& l : layers)
        layer_total += l.second.total_ms;

    fprintf(stderr, "\n===== PREFILL-ONLY PROFILE (snapshot before decode) =====\n");
    fprintf(stderr, "[prefill-profile] wall prefill = %.1f ms (%d tokens)\n", prefill_ms, n_tokens);
    fprintf(stderr, "[prefill-profile] layers total = %.1f ms across %zu layer(s)\n", layer_total,
            layers.size());
    fprintf(stderr, "[prefill-profile] per-layer (ms):\n");
    for (auto& l : layers)
        fprintf(stderr, "  %-22s %9.2f ms\n", l.first.c_str(), l.second.total_ms);
    fprintf(stderr, "[prefill-profile] top other operators (by total ms):\n");
    for (size_t i = 0; i < others.size() && i < 16; ++i) {
        const auto& [name, rec] = others[i];
        double avg = rec.count > 0 ? rec.total_ms / rec.count : 0.0;
        fprintf(stderr, "  %-38s n=%-6lld total=%9.2f ms avg=%8.3f ms\n", name.c_str(),
                (long long)rec.count, rec.total_ms, avg);
    }
    fprintf(stderr, "==========================================================\n\n");
    fflush(stderr);
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

    // ---- Session prefix reuse ----
    // The caller guarantees that KV positions [0, prefix_len) already hold
    // prompt_tokens[0, prefix_len), so only the tail needs a forward. Clamped
    // to leave at least one token: we need a logits row to sample from.
    int prefix_len = 0;
    if (config.prefix_len > 0 && prompt_len > 1) {
        prefix_len = std::min(config.prefix_len, prompt_len - 1);
    }
    const int fwd_len = prompt_len - prefix_len;

    // The KV write cursor is monotonic, so a previous request can leave rows
    // beyond prefix_len. Prefill masks them out (causal: j > q_pos), but decode
    // attends to everything below total_len, so rewind to the reuse point.
    if (prefix_len > 0) {
        if (auto* kv = engine->kv_cache()) {
            kv->rollback(prefix_len);
        }
    } else if (config.reset_kv_cache) {
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
        std::memcpy(input_ids->data(), prompt_tokens.data() + prefix_len,
                    fwd_len * sizeof(int32_t));

        if (dev == DeviceType::CUDA) {
            input_ids->to_device(DeviceType::CUDA);
        }

        auto logits = engine->forward_request(ForwardRequest::from_ids(input_ids, prefix_len));

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

    int64_t pos = prompt_len;
    const auto decode_start = std::chrono::steady_clock::now();

    // ---- Decode phase (speculative decoding via SpeculativeExecutor) ----
    if (spec_executor_) {
        spec_executor_->begin_generation(prompt_tokens);
    }

    for (int step = 1; step < config.max_new_tokens;) {
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
            logits = engine->forward_request(ForwardRequest::from_ids(input_ids, pos));
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
    return result;
}

}  // namespace forge
