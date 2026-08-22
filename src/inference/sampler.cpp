#include "forge/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"

#include "cpu/simd.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

namespace {

// Parallel wrappers around the single-threaded SIMD sampling kernels.
// Decode logits are vocab-sized (e.g. 248K floats); the reduce/sort passes
// are pure streaming work that scales with threads. Chunked with fixed
// boundaries and ordered partial accumulation for run-to-run determinism.

constexpr int kSamplerParallelMinN = 16384;

#ifdef _OPENMP
inline void sampler_chunk(int n, int t, int nt, int& b, int& e) {
    b = static_cast<int>(static_cast<int64_t>(n) * t / nt);
    e = static_cast<int>(static_cast<int64_t>(n) * (t + 1) / nt);
}
#endif

float parallel_max_f32(const float* data, int n) {
#ifdef _OPENMP
    int nt = omp_get_max_threads();
    if (nt > 1 && n >= kSamplerParallelMinN) {
        std::vector<float> partials(nt, -1e30f);
#pragma omp parallel
        {
            int t = omp_get_thread_num();
            int b, e;
            sampler_chunk(n, t, nt, b, e);
            partials[t] = forge::cpu::max_f32(data + b, e - b);
        }
        float m = partials[0];
        for (int t = 1; t < nt; ++t) m = std::max(m, partials[t]);
        return m;
    }
#endif
    return forge::cpu::max_f32(data, n);
}

float parallel_softcap_and_max_f32(float* data, int n, float cap) {
#ifdef _OPENMP
    int nt = omp_get_max_threads();
    if (nt > 1 && n >= kSamplerParallelMinN) {
        std::vector<float> partials(nt, -1e30f);
#pragma omp parallel
        {
            int t = omp_get_thread_num();
            int b, e;
            sampler_chunk(n, t, nt, b, e);
            partials[t] = forge::cpu::softcap_and_max_f32(data + b, e - b, cap);
        }
        float m = partials[0];
        for (int t = 1; t < nt; ++t) m = std::max(m, partials[t]);
        return m;
    }
#endif
    return forge::cpu::softcap_and_max_f32(data, n, cap);
}

float parallel_exp_and_sum_f32(const float* data, float* out, int n, float max_val,
                               float inv_temp) {
#ifdef _OPENMP
    int nt = omp_get_max_threads();
    if (nt > 1 && n >= kSamplerParallelMinN) {
        std::vector<float> partials(nt, 0.0f);
#pragma omp parallel
        {
            int t = omp_get_thread_num();
            int b, e;
            sampler_chunk(n, t, nt, b, e);
            partials[t] = forge::cpu::exp_and_sum_f32(data + b, out + b, e - b, max_val, inv_temp);
        }
        float sum = 0.0f;
        for (int t = 0; t < nt; ++t) sum += partials[t];
        return sum;
    }
#endif
    return forge::cpu::exp_and_sum_f32(data, out, n, max_val, inv_temp);
}

void parallel_scale_normalize_f32(float* data, int n, float inv) {
#ifdef _OPENMP
    int nt = omp_get_max_threads();
    if (nt > 1 && n >= kSamplerParallelMinN) {
#pragma omp parallel
        {
            int t = omp_get_thread_num();
            int b, e;
            sampler_chunk(n, t, nt, b, e);
            forge::cpu::scale_normalize_f32(data + b, e - b, inv);
        }
        return;
    }
#endif
    forge::cpu::scale_normalize_f32(data, n, inv);
}

void parallel_build_pairs(std::vector<std::pair<float, int>>& indexed, const float* probs,
                          int n) {
#ifdef _OPENMP
    int nt = omp_get_max_threads();
    if (nt > 1 && n >= kSamplerParallelMinN) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) indexed[i] = {probs[i], i};
        return;
    }
#endif
    for (int i = 0; i < n; ++i) indexed[i] = {probs[i], i};
}

// Chunked top-k selection + merge. Writes the global top-k (descending) into out.
// Each chunk uses nth_element (O(N/T)) + sort of the k survivors, which is
// asymptotically cheaper than a full partial_sort per chunk.
void parallel_top_k_pairs(std::vector<std::pair<float, int>>& indexed,
                          std::vector<std::pair<float, int>>& out, int n, int k) {
    auto desc = [](const auto& a, const auto& b) { return a.first > b.first; };
#ifdef _OPENMP
    int nt = omp_get_max_threads();
    if (nt > 1 && n >= kSamplerParallelMinN && k < n) {
        out.resize(static_cast<size_t>(nt) * k);
#pragma omp parallel
        {
            int t = omp_get_thread_num();
            int b, e;
            sampler_chunk(n, t, nt, b, e);
            int c = std::min(k, e - b);
            std::nth_element(indexed.begin() + b, indexed.begin() + b + c, indexed.begin() + e,
                             desc);
            std::sort(indexed.begin() + b, indexed.begin() + b + c, desc);
            std::copy(indexed.begin() + b, indexed.begin() + b + c,
                      out.begin() + static_cast<size_t>(t) * k);
        }
        std::partial_sort(out.begin(), out.begin() + k, out.end(), desc);
        out.resize(k);
        return;
    }
#endif
    std::nth_element(indexed.begin(), indexed.begin() + k, indexed.end(), desc);
    std::sort(indexed.begin(), indexed.begin() + k, desc);
    out.assign(indexed.begin(), indexed.begin() + k);
}

// Full descending merge sort (chunked std::sort + k-way merge).
void parallel_full_sort_pairs(std::vector<std::pair<float, int>>& src,
                              std::vector<std::pair<float, int>>& dst, int n) {
    auto desc = [](const auto& a, const auto& b) { return a.first > b.first; };
#ifdef _OPENMP
    int nt = omp_get_max_threads();
    if (nt > 1 && n >= kSamplerParallelMinN) {
        std::vector<std::pair<int, int>> ranges(nt);
#pragma omp parallel
        {
            int t = omp_get_thread_num();
            int b, e;
            sampler_chunk(n, t, nt, b, e);
            ranges[t] = {b, e};
            std::sort(src.begin() + b, src.begin() + e, desc);
        }
        std::vector<int> pos(nt, 0);
        for (int i = 0; i < n; ++i) {
            int best = -1;
            std::pair<float, int> best_pair;
            for (int t = 0; t < nt; ++t) {
                int b = ranges[t].first;
                int p = pos[t];
                if (p < ranges[t].second - b) {
                    const auto& cand = src[b + p];
                    if (best < 0 || cand.first > best_pair.first) {
                        best = t;
                        best_pair = cand;
                    }
                }
            }
            dst[i] = best_pair;
            ++pos[best];
        }
        return;
    }
#endif
    std::sort(src.begin(), src.end(), desc);
    dst.assign(src.begin(), src.end());
}

}  // namespace

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

void Sampler::ensure_scratch(int vocab_size) {
    logits_scratch_.resize(vocab_size);
    probs_scratch_.resize(vocab_size);
    indexed_scratch_.resize(vocab_size);
}

int Sampler::sample_greedy(const TensorPtr& logits) {
    int vocab_size = static_cast<int>(logits->numel());

    if (logits->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        SET_PERF_CONTEXT(-1, "sampler", -1, "cuda", 1);
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

    return greedy_sample_impl(static_cast<const float*>(logits->data()), vocab_size);
}

// Shared CPU core for greedy sampling: copies `logits` into scratch, applies
// repeat penalty, then argmax (with optional softcap).
int Sampler::greedy_sample_impl(const float* logits, int vocab_size) {
    SET_PERF_CONTEXT(-1, "sampler", -1, "cpu", 1);

    ensure_scratch(vocab_size);
    std::vector<float>& host_logits = logits_scratch_;
    {
        PERF_SCOPE("sampler/logits_memcpy");
        std::memcpy(host_logits.data(), logits,
                    static_cast<size_t>(vocab_size) * sizeof(float));
    }

    if (config_.repeat_penalty != 1.0f && !token_history_.empty()) {
        PERF_SCOPE("sampler/repeat_penalty");
        apply_repeat_penalty(host_logits.data(), vocab_size);
    }

    PERF_SCOPE("sampler/argmax_cpu");
    if (config_.logit_softcapping > 0.0f) {
        return forge::cpu::softcap_and_argmax_f32(host_logits.data(), vocab_size,
                                                  config_.logit_softcapping);
    }
    return forge::cpu::argmax_f32(host_logits.data(), vocab_size);
}

int Sampler::sample_ptr(const float* logits, int vocab_size) {
    if (!config_.do_sample || config_.temperature <= 0.0f)
        return sample_greedy_ptr(logits, vocab_size);
    return sample_temperature_ptr(logits, vocab_size, config_.temperature);
}

int Sampler::sample_greedy_ptr(const float* logits, int vocab_size) {
    return greedy_sample_impl(logits, vocab_size);
}

int Sampler::sample_temperature_ptr(const float* logits, int vocab_size, float temperature) {
    return temperature_sample_impl(logits, vocab_size, temperature);
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

    // CUDA with top_k/top_p: D2H once, then run the CPU chain.
    if (logits->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        ensure_scratch(vocab_size);
        {
            PERF_SCOPE("sampler/logits_d2h");
            cudaMemcpy(logits_scratch_.data(), logits->data(), vocab_size * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
        return temperature_sample_impl(logits_scratch_.data(), vocab_size, temperature);
#endif
    }

    return temperature_sample_impl(static_cast<const float*>(logits->data()), vocab_size,
                                   temperature);
}

// Shared CPU core for temperature sampling: copies `logits` into scratch,
// applies repeat penalty + softcap, softmax(1/T), top-k and top-p filtering,
// then multinomial sampling.
int Sampler::temperature_sample_impl(const float* logits_in, int vocab_size, float temperature) {
    ensure_scratch(vocab_size);
    std::vector<float>& host_logits = logits_scratch_;
    {
        PERF_SCOPE("sampler/logits_memcpy");
        std::memcpy(host_logits.data(), logits_in,
                    static_cast<size_t>(vocab_size) * sizeof(float));
    }

    if (config_.repeat_penalty != 1.0f && !token_history_.empty()) {
        PERF_SCOPE("sampler/repeat_penalty");
        apply_repeat_penalty(host_logits.data(), vocab_size);
    }
    {
        PERF_SCOPE("sampler/softmax_sample");

        float max_val;
        bool has_softcap = (config_.logit_softcapping > 0.0f);
        if (has_softcap) {
            float cap = config_.logit_softcapping;
            max_val = parallel_softcap_and_max_f32(host_logits.data(), vocab_size, cap);
        } else {
            max_val = parallel_max_f32(host_logits.data(), vocab_size);
        }

        std::vector<float>& probs = probs_scratch_;
        float inv_temp = 1.0f / temperature;
        float sum = parallel_exp_and_sum_f32(host_logits.data(), probs.data(), vocab_size, max_val,
                                             inv_temp);
        parallel_scale_normalize_f32(probs.data(), vocab_size, 1.0f / sum);

        // Sorted (prob, index) pairs for top_k / top_p filtering.
        // After top_k, holds the (already normalized) top-k in descending order.
        std::vector<std::pair<float, int>> top_sorted;
        float inv_top = 1.0f;

        if (config_.top_k > 0) {
            std::vector<std::pair<float, int>>& indexed = indexed_scratch_;
            int k = config_.top_k;
            if (k > vocab_size) k = vocab_size;

            parallel_build_pairs(indexed, probs.data(), vocab_size);
            parallel_top_k_pairs(indexed, top_sorted, vocab_size, k);

            std::fill(probs.begin(), probs.end(), 0.0f);
            float top_sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                probs[top_sorted[i].second] = top_sorted[i].first;
                top_sum += top_sorted[i].first;
            }
            inv_top = 1.0f / top_sum;
            parallel_scale_normalize_f32(probs.data(), vocab_size, inv_top);
        }

        if (config_.top_p < 1.0f) {
            int cutoff = vocab_size;
            std::vector<std::pair<float, int>> full_sorted;

            if (!top_sorted.empty()) {
                // top_k was applied first: probs only has k non-zero entries and
                // top_sorted holds them descending (pre-normalization). Reuse it,
                // accumulating the normalized cumulative probability (cutoff <= k).
                float acc = 0.0f;
                int lim = static_cast<int>(top_sorted.size());
                for (int i = 0; i < lim; ++i) {
                    acc += top_sorted[i].first * inv_top;
                    if (acc > config_.top_p) {
                        cutoff = i + 1;
                        break;
                    }
                }
                std::fill(probs.begin(), probs.end(), 0.0f);
                float top_p_sum = 0.0f;
                for (int i = 0; i < cutoff && i < lim; ++i) {
                    probs[top_sorted[i].second] = top_sorted[i].first;
                    top_p_sum += top_sorted[i].first;
                }
                parallel_scale_normalize_f32(probs.data(), vocab_size, 1.0f / top_p_sum);
            } else {
                std::vector<std::pair<float, int>>& indexed = indexed_scratch_;
                parallel_build_pairs(indexed, probs.data(), vocab_size);
                full_sorted.resize(vocab_size);
                parallel_full_sort_pairs(indexed, full_sorted, vocab_size);

                float cumsum = 0.0f;
                for (int i = 0; i < vocab_size; ++i) {
                    cumsum += full_sorted[i].first;
                    if (cumsum > config_.top_p) {
                        cutoff = i + 1;
                        break;
                    }
                }
                std::fill(probs.begin(), probs.end(), 0.0f);
                float top_p_sum = 0.0f;
                for (int i = 0; i < cutoff; ++i) {
                    probs[full_sorted[i].second] = full_sorted[i].first;
                    top_p_sum += full_sorted[i].first;
                }
                parallel_scale_normalize_f32(probs.data(), vocab_size, 1.0f / top_p_sum);
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

void Sampler::apply_repeat_penalty(float* logits, int size) const {
    float penalty = config_.repeat_penalty;
    int n = static_cast<int>(token_history_.size());
    int last_n = config_.repeat_last_n;
    // last_n <= 0: use full history (legacy behavior); otherwise only the last N tokens.
    int start = (last_n > 0 && last_n < n) ? (n - last_n) : 0;
    for (int i = start; i < n; ++i) {
        int32_t tid = token_history_[i];
        if (tid >= 0 && tid < size) {
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
