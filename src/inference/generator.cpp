#include "forge/generator.h"

#include <chrono>
#include <cstring>
#include <stdexcept>

#include "forge/engine.h"
#include "forge/engines/llama_engine.h"
#include "forge/inference_batch.h"
#include "forge/kv_cache.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"
#include "forge/speculative.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
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
        auto input_ids = std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{prompt_len},
                                                  DeviceType::CPU);
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
        // Check additional stop tokens
        for (int stop_id : config.stop_token_ids) {
            if (token_id == stop_id) {
                result.finished = true;
                result.finish_reason = "stop";
                result.num_generated_tokens = 1;
                return result;
            }
        }
    }

    int64_t pos = prompt_len;

    // ---- Decode phase (支持 speculative decoding) ----
    const auto& spec_cfg = ctx_.params().speculative_config;
    std::unique_ptr<DraftModel> draft_model;
    if (spec_cfg.enabled && spec_cfg.use_ngram) {
        draft_model = std::make_unique<NgramDraftModel>(spec_cfg.ngram_n, spec_cfg.ngram_min);
        // 用 prompt 初始化 n-gram 索引
        draft_model->accept(prompt_tokens);
    }

    for (int step = 1; step < config.max_new_tokens;) {
        // Speculative decoding: draft → verify → accept
        if (spec_cfg.enabled && draft_model) {
            int n_draft = spec_cfg.n_draft;
            int remaining = config.max_new_tokens - step;
            if (n_draft > remaining) n_draft = remaining;

            // Step 1: 生成 draft tokens
            auto drafts = draft_model->draft(result.token_ids, n_draft);
            if (drafts.empty()) {
                // 无 draft 可用，退回普通 decode
                goto normal_decode;
            }

            {
                // Step 2: 构建 batch 验证 — 喂入 [last_token, d0, d1, ...]
                InferenceBatch batch;
                batch.all_logits = true;
                InferenceBatchItem item;
                item.seq_id = 0;
                item.logits = true;
                item.tokens = {result.token_ids.back()};
                item.tokens.insert(item.tokens.end(), drafts.begin(), drafts.end());
                item.start_pos = pos;
                item.positions.resize(item.tokens.size());
                for (size_t j = 0; j < item.tokens.size(); j++)
                    item.positions[j] = pos + static_cast<int64_t>(j);
                batch.items.push_back(std::move(item));

                TensorPtr logits_all;
                {
                    PERF_SCOPE("speculative/forward_batch");
                    logits_all = engine->forward_batch(batch);
                }

                if (!logits_all || logits_all->ndim() < 2) {
                    // forward_batch 失败，退回普通 decode
                    goto normal_decode;
                }

                int vocab_size = static_cast<int>(logits_all->shape()[1]);
                const float* logits_data = static_cast<const float*>(logits_all->data());

                // Step 3: 验证 draft tokens
                auto vr = verify_draft_tokens(logits_data, vocab_size, drafts,
                                              sampler_, spec_cfg);

                // Step 4: 回滚被拒绝位置的 KV cache
                int64_t accepted_pos = pos + vr.n_accepted + 1;  // +1 for base token
                KVCache* kv = engine->kv_cache();
                if (kv && vr.n_accepted < static_cast<int>(drafts.size())) {
                    kv->rollback(accepted_pos);
                }

                // Step 5: 接受已验证的 tokens
                for (int i = 0; i < vr.n_accepted; ++i) {
                    result.token_ids.push_back(vr.accepted_tokens[i]);
                    pos += 1;
                    step++;
                    if (callback) callback(vr.accepted_tokens[i], step - 1);
                    if (config.eos_token_id >= 0 && vr.accepted_tokens[i] == config.eos_token_id) {
                        result.finished = true;
                        result.finish_reason = "eos";
                        goto decode_done;
                    }
                }

                // 添加重采样或 bonus token
                int32_t next_token = -1;
                if (vr.resampled >= 0) {
                    next_token = vr.resampled;
                } else if (vr.bonus >= 0) {
                    next_token = vr.bonus;
                }

                if (next_token >= 0) {
                    result.token_ids.push_back(next_token);
                    pos += 1;
                    step++;
                    if (callback) callback(next_token, step - 1);
                    if (config.eos_token_id >= 0 && next_token == config.eos_token_id) {
                        result.finished = true;
                        result.finish_reason = "eos";
                        goto decode_done;
                    }
                    for (int stop_id : config.stop_token_ids) {
                        if (next_token == stop_id) {
                            result.finished = true;
                            result.finish_reason = "stop";
                            goto decode_done;
                        }
                    }
                }

                // 更新 draft model 的历史
                draft_model->accept(result.token_ids);
                continue;
            }
        }

    normal_decode:
        // 普通 decode: 逐 token 生成
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
            logits = engine->forward(input_ids, pos);
        }

        {
            PERF_SCOPE("decode/sampler");
            token_id = sampler_.sample(logits, pos);
        }

        result.token_ids.push_back(token_id);
        pos += 1;
        step++;

        if (callback) {
            callback(token_id, step - 1);
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
        result.finish_reason = "length";
    }

    result.num_generated_tokens = static_cast<int>(result.token_ids.size());

    return result;
}

}  // namespace forge
