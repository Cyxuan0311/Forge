#include "forge/sampler.h"
#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
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

int Sampler::sample_greedy(const TensorPtr& logits) {
    int vocab_size = static_cast<int>(logits->numel());

    if (logits->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        if (config_.repeat_penalty == 1.0f || token_history_.empty()) {
            if (!cuda_argmax_buf_) {
                cudaMalloc(&cuda_argmax_buf_, sizeof(int32_t));
            }
            {
                PERF_SCOPE("sampler/argmax_gpu");
                cuda::launch_argmax(
                    static_cast<const float*>(logits->data()),
                    static_cast<int32_t*>(cuda_argmax_buf_),
                    vocab_size);
            }
            int32_t result;
            {
                PERF_SCOPE("sampler/d2h_argmax");
                cudaMemcpy(&result, cuda_argmax_buf_, sizeof(int32_t), cudaMemcpyDeviceToHost);
            }
            token_history_.push_back(result);
            return result;
        }
#endif
    }

    std::vector<float> host_logits(vocab_size);
    if (logits->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        PERF_SCOPE("sampler/logits_d2h");
        cudaMemcpy(host_logits.data(), logits->data(), vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
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
        int best = 0;
        float best_val = host_logits[0];
#ifdef USE_AVX2
        {
            __m256 vmax = _mm256_set1_ps(-1e30f);
            __m256i vidx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            int i = 0;
            for (; i + 8 <= vocab_size; i += 8) {
                __m256 v = _mm256_loadu_ps(&host_logits[i]);
                __m256 cmp = _mm256_cmp_ps(v, vmax, _CMP_GT_OS);
                vmax = _mm256_blendv_ps(vmax, v, cmp);
                __m256i new_idx = _mm256_add_epi32(_mm256_set1_epi32(i), _mm256_setr_epi32(0,1,2,3,4,5,6,7));
                vidx = _mm256_blendv_epi8(vidx, new_idx, _mm256_castps_si256(cmp));
            }
            // Find max among the 8 remaining candidates
            float vals[8]; int idxs[8];
            _mm256_storeu_ps(vals, vmax);
            _mm256_storeu_si256((__m256i*)idxs, vidx);
            for (int j = 0; j < 8; ++j) {
                if (vals[j] > best_val) { best_val = vals[j]; best = idxs[j]; }
            }
            for (; i < vocab_size; ++i) {
                if (host_logits[i] > best_val) { best_val = host_logits[i]; best = i; }
            }
        }
#else
        for (int i = 1; i < vocab_size; ++i) {
            if (host_logits[i] > best_val) {
                best_val = host_logits[i];
                best = i;
            }
        }
#endif
        return best;
    }
}

int Sampler::sample_temperature(const TensorPtr& logits, float temperature) {
    int vocab_size = static_cast<int>(logits->numel());

    std::vector<float> host_logits(vocab_size);
    if (logits->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        PERF_SCOPE("sampler/logits_d2h");
        cudaMemcpy(host_logits.data(), logits->data(), vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
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

        // AVX2-accelerated max reduction
        float max_val;
#ifdef USE_AVX2
        {
            __m256 vmax = _mm256_set1_ps(-1e30f);
            int i = 0;
            for (; i + 8 <= vocab_size; i += 8) {
                __m256 v = _mm256_loadu_ps(&host_logits[i]);
                vmax = _mm256_max_ps(vmax, v);
            }
            // Horizontal max
            __m128 hi = _mm256_extractf128_ps(vmax, 1);
            __m128 lo = _mm256_castps256_ps128(vmax);
            __m128 m = _mm256_castps256_ps128(_mm256_max_ps(vmax, _mm256_permute2f128_ps(vmax, vmax, 0x01)));
            m = _mm_max_ps(m, _mm_movehl_ps(m, m));
            m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
            max_val = _mm_cvtss_f32(m);
            for (; i < vocab_size; ++i) {
                if (host_logits[i] > max_val) max_val = host_logits[i];
            }
        }
#else
        max_val = *std::max_element(host_logits.begin(), host_logits.end());
#endif

        // AVX2-accelerated exp + sum
        std::vector<float> probs(vocab_size);
        float sum = 0.0f;
        float inv_temp = 1.0f / temperature;
#ifdef USE_AVX2
        {
            __m256 vsum = _mm256_setzero_ps();
            __m256 vshift = _mm256_set1_ps(max_val);
            __m256 vscale = _mm256_set1_ps(inv_temp);
            int i = 0;
            for (; i + 8 <= vocab_size; i += 8) {
                __m256 v = _mm256_loadu_ps(&host_logits[i]);
                v = _mm256_sub_ps(v, vshift);
                v = _mm256_mul_ps(v, vscale);
                // Fast exp approximation using AVX2 (max error ~1.5%)
                // exp(x) = 2^(x/ln2), use polynomial for fractional part
                __m256 exp_v;
                // Clamp to [-88, 88] to avoid overflow
                v = _mm256_min_ps(v, _mm256_set1_ps(88.0f));
                v = _mm256_max_ps(v, _mm256_set1_ps(-88.0f));
                // exp(x) = 2^(x * 1.44269504) = 2^(n + f) where n = floor(x*1.44269504)
                __m256 x = _mm256_mul_ps(v, _mm256_set1_ps(1.44269504f));
                __m256i n = _mm256_cvttps_epi32(x);
                __m256 f = _mm256_sub_ps(x, _mm256_cvtepi32_ps(n));
                // Polynomial: 2^f ≈ 1 + f*(0.693147 + f*(0.240227 + f*0.055504))
                __m256 p = _mm256_fmadd_ps(f, _mm256_set1_ps(0.055504f), _mm256_set1_ps(0.240227f));
                p = _mm256_fmadd_ps(f, p, _mm256_set1_ps(0.693147f));
                p = _mm256_fmadd_ps(f, p, _mm256_set1_ps(1.0f));
                // 2^n via bit manipulation
                __m256i n_shifted = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
                exp_v = _mm256_mul_ps(p, _mm256_castsi256_ps(n_shifted));
                _mm256_storeu_ps(&probs[i], exp_v);
                vsum = _mm256_add_ps(vsum, exp_v);
            }
            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            sum = _mm_cvtss_f32(s);
            for (; i < vocab_size; ++i) {
                probs[i] = std::exp((host_logits[i] - max_val) * inv_temp);
                sum += probs[i];
            }
        }
#else
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp((host_logits[i] - max_val) / temperature);
            sum += probs[i];
        }
#endif

        // AVX2-accelerated normalization
        float inv_sum = 1.0f / sum;
#ifdef USE_AVX2
        {
            __m256 vinv = _mm256_set1_ps(inv_sum);
            int i = 0;
            for (; i + 8 <= vocab_size; i += 8) {
                __m256 v = _mm256_loadu_ps(&probs[i]);
                _mm256_storeu_ps(&probs[i], _mm256_mul_ps(v, vinv));
            }
            for (; i < vocab_size; ++i) {
                probs[i] *= inv_sum;
            }
        }
#else
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] /= sum;
        }
#endif

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
            if (cumsum >= r) return i;
        }

        return vocab_size - 1;
    }
}

void Sampler::set_config(const SamplerConfig& config) {
    config_ = config;
    if (config.seed != 0) rng_state_ = config.seed;
}

const SamplerConfig& Sampler::config() const {
    return config_;
}

void Sampler::apply_repeat_penalty(std::vector<float>& logits) const {
    float penalty = config_.repeat_penalty;
    for (int32_t tid : token_history_) {
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

} // namespace forge
