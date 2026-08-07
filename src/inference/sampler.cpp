#include "forge/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"

#include "cpu/simd.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

Sampler::Sampler(const SamplerConfig& config) : config_(config) {
    rng_state_ = config.seed != 0 ? config.seed : 12345;
}

Sampler::~Sampler() {
#ifdef USE_CUDA
    if (cuda_argmax_buf_) {
        cudaFree(cuda_argmax_buf_);
        cuda_argmax_buf_ = nullptr;
    }
    if (d_token_history_) {
        cudaFree(d_token_history_);
        d_token_history_ = nullptr;
    }
#endif
}

uint64_t Sampler::next_rng() {
    rng_state_ ^= rng_state_ << 13;
    rng_state_ ^= rng_state_ >> 7;
    rng_state_ ^= rng_state_ << 17;
    return rng_state_;
}

float Sampler::next_uniform() {
    return static_cast<float>(next_rng() & 0x7FFFFFFF) / static_cast<float>(0x7FFFFFFF);
}

int Sampler::sample(const TensorPtr& logits, int64_t pos) {
    if (!config_.do_sample || config_.temperature <= 0.0f) {
        int token_id = sample_greedy(logits);
        add_token_to_history(static_cast<int32_t>(token_id));
        return token_id;
    }
    int token_id = sample_temperature(logits, config_.temperature);
    add_token_to_history(static_cast<int32_t>(token_id));
    return token_id;
}

void Sampler::ensure_token_history_buffer(int n) {
#ifdef USE_CUDA
    if (n > d_token_history_capacity_) {
        if (d_token_history_) cudaFree(d_token_history_);
        cudaMalloc(&d_token_history_, n * sizeof(int32_t));
        d_token_history_capacity_ = n;
    }
#else
    (void)n;
#endif
}

int Sampler::sample_greedy(const TensorPtr& logits) {
    int vocab_size = static_cast<int>(logits->numel());
    SET_PERF_CONTEXT(-1, "sampler", -1, logits->device() == DeviceType::CUDA ? "cuda" : "cpu", 1);

    if (logits->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        if (config_.repeat_penalty != 1.0f && !token_history_.empty()) {
            PERF_SCOPE("sampler/repeat_penalty_gpu");
            int n = static_cast<int>(token_history_.size());
            int last_n = config_.repeat_last_n;
            int start = (last_n > 0 && last_n < n) ? (n - last_n) : 0;
            int hist_count = n - start;
            ensure_token_history_buffer(hist_count);
            cudaMemcpy(d_token_history_, token_history_.data() + start,
                       hist_count * sizeof(int32_t), cudaMemcpyHostToDevice);
            cuda::launch_repeat_penalty(static_cast<float*>(logits->data()),
                                         d_token_history_,
                                         hist_count,
                                         config_.repeat_penalty, vocab_size);
        }
        {
            PERF_SCOPE("sampler/argmax_gpu");
            if (!cuda_argmax_buf_) {
                cudaMalloc(&cuda_argmax_buf_, sizeof(int32_t));
            }
            cuda::launch_argmax(static_cast<const float*>(logits->data()),
                                static_cast<int32_t*>(cuda_argmax_buf_), vocab_size);
        }
        int32_t result;
        {
            PERF_SCOPE("sampler/d2h_argmax");
            cudaMemcpy(&result, cuda_argmax_buf_, sizeof(int32_t), cudaMemcpyDeviceToHost);
        }
        return result;
#endif
    }

    std::vector<float> host_logits(vocab_size);
    if (logits->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        PERF_SCOPE("sampler/logits_d2h");
        cudaMemcpy(host_logits.data(), logits->data(), vocab_size * sizeof(float),
                   cudaMemcpyDeviceToHost);
#endif
    } else {
        PERF_SCOPE("sampler/logits_memcpy");
        std::memcpy(host_logits.data(), logits->data(), vocab_size * sizeof(float));
    }

    if (config_.repeat_penalty != 1.0f && !token_history_.empty()) {
        PERF_SCOPE("sampler/repeat_penalty");
        apply_repeat_penalty(host_logits);
    }

    {
        PERF_SCOPE("sampler/argmax_cpu");
        int best;
        bool has_softcap = (config_.logit_softcapping > 0.0f);

        if (has_softcap) {
            float cap = config_.logit_softcapping;
            best = forge::cpu::softcap_and_argmax_f32(host_logits.data(), vocab_size, cap);
        } else {
            best = forge::cpu::argmax_f32(host_logits.data(), vocab_size);
        }
        return best;
    }
}

int Sampler::sample_temperature(const TensorPtr& logits, float temperature) {
    int vocab_size = static_cast<int>(logits->numel());

    // GPU sampling path: Gumbel-max trick on device
    // Falls back to CPU when top_k > 0 or top_p < 1.0 (complex filtering)
    if (logits->device() == DeviceType::CUDA && config_.top_k <= 0 && config_.top_p >= 1.0f) {
#ifdef USE_CUDA
        {
            PERF_SCOPE("sampler/gumbel_gpu");
            if (config_.repeat_penalty != 1.0f && !token_history_.empty()) {
                int n = static_cast<int>(token_history_.size());
                int last_n = config_.repeat_last_n;
                int start = (last_n > 0 && last_n < n) ? (n - last_n) : 0;
                int hist_count = n - start;
                ensure_token_history_buffer(hist_count);
                cudaMemcpy(d_token_history_, token_history_.data() + start,
                           hist_count * sizeof(int32_t), cudaMemcpyHostToDevice);
                cuda::launch_repeat_penalty(static_cast<float*>(logits->data()),
                                            d_token_history_,
                                            hist_count,
                                            config_.repeat_penalty, vocab_size);
            }
            if (config_.logit_softcapping > 0.0f) {
                cuda::launch_logit_softcap(static_cast<float*>(logits->data()),
                                           config_.logit_softcapping, true,
                                           nullptr, 0, vocab_size);
            }
            if (!cuda_argmax_buf_) {
                cudaMalloc(&cuda_argmax_buf_, sizeof(int32_t));
            }
            cuda::launch_gumbel_sample(static_cast<const float*>(logits->data()),
                                        static_cast<int32_t*>(cuda_argmax_buf_),
                                        temperature, config_.seed, vocab_size);
        }
        int32_t result;
        cudaMemcpy(&result, cuda_argmax_buf_, sizeof(int32_t), cudaMemcpyDeviceToHost);
        return result;
#endif
    }

    // CPU fallback path
    std::vector<float> host_logits(vocab_size);
    if (logits->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        PERF_SCOPE("sampler/logits_d2h");
        cudaMemcpy(host_logits.data(), logits->data(), vocab_size * sizeof(float),
                   cudaMemcpyDeviceToHost);
#endif
    } else {
        PERF_SCOPE("sampler/logits_memcpy");
        std::memcpy(host_logits.data(), logits->data(), vocab_size * sizeof(float));
    }

    if (config_.repeat_penalty != 1.0f && !token_history_.empty()) {
        PERF_SCOPE("sampler/repeat_penalty");
        apply_repeat_penalty(host_logits);
    }
    {
        PERF_SCOPE("sampler/softmax_sample");

        float max_val;
        bool has_softcap = (config_.logit_softcapping > 0.0f);
        if (has_softcap) {
            float cap = config_.logit_softcapping;
            max_val = forge::cpu::softcap_and_max_f32(host_logits.data(), vocab_size, cap);
        } else {
            max_val = forge::cpu::max_f32(host_logits.data(), vocab_size);
        }

        std::vector<float> probs(vocab_size);
        float inv_temp = 1.0f / temperature;
        float sum = forge::cpu::exp_and_sum_f32(host_logits.data(), probs.data(), vocab_size, max_val, inv_temp);

        float inv_sum = 1.0f / sum;
        forge::cpu::scale_normalize_f32(probs.data(), vocab_size, inv_sum);

        if (config_.top_k > 0) {
            std::vector<std::pair<float, int>> indexed(vocab_size);
            for (int i = 0; i < vocab_size; ++i) {
                indexed[i] = {probs[i], i};
            }
            std::partial_sort(indexed.begin(), indexed.begin() + config_.top_k, indexed.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

            std::fill(probs.begin(), probs.end(), 0.0f);
            float top_sum = 0.0f;
            for (int i = 0; i < config_.top_k && i < vocab_size; ++i) {
                probs[indexed[i].second] = indexed[i].first;
                top_sum += indexed[i].first;
            }
            for (int i = 0; i < vocab_size; ++i) {
                probs[i] /= top_sum;
            }
        }

        if (config_.top_p < 1.0f) {
            std::vector<std::pair<float, int>> indexed(vocab_size);
            for (int i = 0; i < vocab_size; ++i) {
                indexed[i] = {probs[i], i};
            }
            std::sort(indexed.begin(), indexed.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

            float cumsum = 0.0f;
            int cutoff = vocab_size;
            for (int i = 0; i < vocab_size; ++i) {
                cumsum += indexed[i].first;
                if (cumsum > config_.top_p) {
                    cutoff = i + 1;
                    break;
                }
            }

            std::fill(probs.begin(), probs.end(), 0.0f);
            float top_p_sum = 0.0f;
            for (int i = 0; i < cutoff; ++i) {
                probs[indexed[i].second] = indexed[i].first;
                top_p_sum += indexed[i].first;
            }
            for (int i = 0; i < vocab_size; ++i) {
                probs[i] /= top_p_sum;
            }
        }

        float r = next_uniform();
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            cumsum += probs[i];
            if (cumsum >= r)
                return i;
        }

        return vocab_size - 1;
    }
}

void Sampler::set_config(const SamplerConfig& config) {
    config_ = config;
    if (config.seed != 0)
        rng_state_ = config.seed;
}

const SamplerConfig& Sampler::config() const {
    return config_;
}

void Sampler::apply_repeat_penalty(std::vector<float>& logits) const {
    float penalty = config_.repeat_penalty;
    int n = static_cast<int>(token_history_.size());
    int last_n = config_.repeat_last_n;
    // last_n <= 0: use full history (legacy behavior); otherwise only the last N tokens.
    int start = (last_n > 0 && last_n < n) ? (n - last_n) : 0;
    for (int i = start; i < n; ++i) {
        int32_t tid = token_history_[i];
        if (tid >= 0 && tid < static_cast<int>(logits.size())) {
            if (logits[tid] > 0.0f) {
                logits[tid] /= penalty;
            } else {
                logits[tid] *= penalty;
            }
        }
    }
}

void Sampler::add_token_to_history(int32_t token_id) {
    token_history_.push_back(token_id);
}

void Sampler::clear_history() {
    token_history_.clear();
}

std::vector<BatchSampleResult> Sampler::sample_batch(const TensorPtr& logits_batch,
                                                      const InferenceBatch& batch) {
    std::vector<BatchSampleResult> results;
    if (!logits_batch || batch.empty())
        return results;

    // Determine vocab_size from logits shape
    auto& shape = logits_batch->shape();
    int vocab_size = static_cast<int>(shape.back());
    int total_rows = static_cast<int>(logits_batch->numel()) / vocab_size;

    // For the default sequential fallback, forward_batch() returns only
    // the last sequence's logits [1, vocab_size]. In that case we need
    // to call forward() individually per sequence.
    //
    // For a true batch implementation that returns [total_tokens, vocab],
    // we extract each sequence's last-token logits.
    //
    // Heuristic: if total_rows == 1, it's the sequential fallback.
    // If total_rows == batch.n_tokens(), it's a true batch result.
    if (total_rows == 1) {
        // Sequential fallback: only the last sequence's logits are available.
        // Sample only the last item that requested logits.
        for (int i = static_cast<int>(batch.items.size()) - 1; i >= 0; --i) {
            if (batch.items[i].logits) {
                BatchSampleResult r;
                r.seq_index = i;
                r.seq_id = batch.items[i].seq_id;
                r.token_id = static_cast<int32_t>(sample(logits_batch, batch.items[i].start_pos));
                results.push_back(r);
                break;  // only last sequence's logits available
            }
        }
    } else {
        // True batch: extract each sequence's last-token logits.
        // Logits are [total_tokens, vocab_size], and we need the last token
        // of each sequence.
        auto offsets = batch.token_offsets();

        // Ensure logits are on CPU for extraction
        TensorPtr logits_cpu = logits_batch;
        if (logits_batch->device() == DeviceType::CUDA) {
            logits_cpu = std::make_shared<Tensor>(DataType::FP32, logits_batch->shape(),
                                                   DeviceType::CPU);
            logits_cpu->copy_from(*logits_batch);
        }
        const float* data = static_cast<const float*>(logits_cpu->data());

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.items[i].logits)
                continue;

            int seq_len = static_cast<int>(batch.items[i].tokens.size());
            int last_row = offsets[i] + seq_len - 1;

            // Extract last-token logits for this sequence
            auto seq_logits = std::make_shared<Tensor>(DataType::FP32,
                                                        std::vector<int64_t>{vocab_size},
                                                        DeviceType::CPU);
            std::memcpy(seq_logits->data(), data + last_row * vocab_size,
                        vocab_size * sizeof(float));

            BatchSampleResult r;
            r.seq_index = i;
            r.seq_id = batch.items[i].seq_id;
            r.token_id = static_cast<int32_t>(sample(seq_logits, batch.items[i].start_pos + seq_len - 1));
            results.push_back(r);
        }
    }

    return results;
}

}  // namespace forge
