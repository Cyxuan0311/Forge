#include <cmath>
#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/operator_attention.h"
#include "forge/perf_profiler.h"

#include "cpu/simd.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {
namespace ops {

TensorPtr scaled_dot_product_attention(const TensorPtr& q, const TensorPtr& k, const TensorPtr& v,
                                       bool causal) {
    int seq_len = static_cast<int>(q->shape()[0]);
    int num_heads = static_cast<int>(q->shape()[1]);
    int head_dim = static_cast<int>(q->shape()[2]);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto out = std::make_shared<Tensor>(DataType::FP32, q->shape(), q->device());

    if (q->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        auto q_2d =
            std::make_shared<Tensor>(q->view(std::vector<int64_t>{seq_len, num_heads * head_dim}));
        auto k_2d =
            std::make_shared<Tensor>(k->view(std::vector<int64_t>{seq_len, num_heads * head_dim}));
        auto v_2d =
            std::make_shared<Tensor>(v->view(std::vector<int64_t>{seq_len, num_heads * head_dim}));
        auto out_2d = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);

        cuda::launch_flash_attention(
            static_cast<const float*>(q_2d->data()), static_cast<const float*>(k_2d->data()),
            static_cast<const float*>(v_2d->data()), static_cast<float*>(out_2d->data()), seq_len,
            seq_len, num_heads, head_dim, nullptr, causal);

        auto out_3d = out_2d->view(std::vector<int64_t>{seq_len, num_heads, head_dim});
        *out = std::move(out_3d);
#endif
    } else {
        const float* q_data = static_cast<const float*>(q->data());
        const float* k_data = static_cast<const float*>(k->data());
        const float* v_data = static_cast<const float*>(v->data());

        std::vector<float> host_out(q->numel());

#pragma omp parallel for schedule(dynamic) collapse(2) if (seq_len * num_heads > 4)
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                std::vector<float> scores(seq_len);
                float max_val = -1e30f;
                for (int j = 0; j < seq_len; ++j) {
                    if (causal && j > i) {
                        scores[j] = -1e30f;
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_data[i * num_heads * head_dim + h * head_dim + d] *
                               k_data[j * num_heads * head_dim + h * head_dim + d];
                    }
                    scores[j] = dot * scale;
                    max_val = std::max(max_val, scores[j]);
                }
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    scores[j] = std::exp(scores[j] - max_val);
                    sum += scores[j];
                }
                float inv_sum = 1.0f / (sum + 1e-30f);
                for (int d = 0; d < head_dim; ++d) {
                    float val = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        val += scores[j] * v_data[j * num_heads * head_dim + h * head_dim + d];
                    }
                    host_out[i * num_heads * head_dim + h * head_dim + d] = val * inv_sum;
                }
            }
        }

        std::memcpy(out->data(), host_out.data(), host_out.size() * sizeof(float));
    }

    return out;
}

TensorPtr scaled_dot_product_attention_2d(const TensorPtr& q, const TensorPtr& k,
                                          const TensorPtr& v, int seq_len, int num_heads,
                                          int head_dim, const TensorPtr& mask, bool causal) {
    return scaled_dot_product_attention_2d(q, k, v, seq_len, seq_len, num_heads, head_dim, mask, causal);
}

TensorPtr scaled_dot_product_attention_2d(const TensorPtr& q, const TensorPtr& k,
                                          const TensorPtr& v, int q_len, int kv_len, int num_heads,
                                          int head_dim, const TensorPtr& mask, bool causal) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{q_len, num_heads * head_dim}, q->device());

    if (q->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        // If mask is on CPU, copy it to CUDA first
        const float* mask_ptr = nullptr;
        auto mask_cuda = mask;
        if (mask && mask->device() == DeviceType::CPU) {
            mask_cuda = std::make_shared<Tensor>(DataType::FP32, mask->shape(), DeviceType::CUDA);
            mask_cuda->copy_from(*mask);
        }
        if (mask_cuda) {
            mask_ptr = static_cast<const float*>(mask_cuda->data());
        }
        cuda::launch_flash_attention(
            static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
            static_cast<const float*>(v->data()), static_cast<float*>(out->data()), q_len, kv_len,
            num_heads, head_dim, mask_ptr, causal);
        return out;
#endif
    }
    const float* q_data = static_cast<const float*>(q->data());
    const float* k_data = static_cast<const float*>(k->data());
    const float* v_data = static_cast<const float*>(v->data());
    const float* mask_data = mask ? static_cast<const float*>(mask->data()) : nullptr;

    std::vector<float> host_out(out->numel());

    // For decode (q_len=1), causal mask is irrelevant since all KV positions
    // are visible. Use online softmax to avoid allocating a scores vector
    // per head and reduce memory traffic.
    if (q_len == 1) {
        PERF_SCOPE("attention/qk_dot");
#pragma omp parallel for schedule(static)
        for (int h = 0; h < num_heads; ++h) {
            const float* q_row = q_data + h * head_dim;
            float* out_row = host_out.data() + h * head_dim;

            // Online softmax: process one KV position at a time
            float max_val = -1e30f;
            float sum_exp = 0.0f;
            std::memset(out_row, 0, head_dim * sizeof(float));

            for (int j = 0; j < kv_len; ++j) {
                const float* k_row = k_data + j * num_heads * head_dim + h * head_dim;
                float dot = forge::cpu::dot_f32(q_row, k_row, head_dim);
                float score = dot * scale;

                // Apply mask bias for decode (single row: mask[j])
                if (mask_data) {
                    float m = mask_data[j];
                    if (m < -1e20f) continue;  // skip masked positions
                    score += m;
                }

                float new_max = std::max(max_val, score);
                float rescale = std::exp(max_val - new_max);

                // Rescale existing accumulator
                if (new_max > max_val) {
                    sum_exp *= rescale;
                    forge::cpu::scale_f32(out_row, head_dim, rescale);
                }

                float exp_score = std::exp(score - new_max);
                sum_exp += exp_score;

                const float* v_row = v_data + j * num_heads * head_dim + h * head_dim;
                forge::cpu::fmadd_f32(out_row, v_row, head_dim, exp_score);
                max_val = new_max;
            }

            // Final normalization
            float inv_sum = 1.0f / (sum_exp + 1e-30f);
            forge::cpu::scale_f32(out_row, head_dim, inv_sum);
        }
    } else {
        // General path: prefill or causal attention
        PERF_SCOPE("attention/qk_dot");
// Pre-allocate scores buffer per thread to avoid heap allocation per iteration
#pragma omp parallel
        {
            std::vector<float> scores_buf(kv_len);

#pragma omp for schedule(dynamic) collapse(2)
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < q_len; ++i) {
                    int q_pos = kv_len - q_len + i;
                    float max_val = -1e30f;
                    const float* q_row = q_data + i * num_heads * head_dim + h * head_dim;
                    for (int j = 0; j < kv_len; ++j) {
                        // Mask-aware masking: if mask is present, use it;
                        // otherwise fall back to causal mask
                        float m = 0.0f;
                        if (mask_data) {
                            m = mask_data[i * kv_len + j];
                            if (m < -1e20f) {
                                scores_buf[j] = -1e30f;
                                continue;
                            }
                        } else if (causal && j > q_pos) {
                            scores_buf[j] = -1e30f;
                            continue;
                        }
                        const float* k_row = k_data + j * num_heads * head_dim + h * head_dim;
                        float dot = forge::cpu::dot_f32(q_row, k_row, head_dim);
                        scores_buf[j] = dot * scale + m;
                        max_val = std::max(max_val, scores_buf[j]);
                    }

                    float sum = 0.0f;
                    for (int j = 0; j < kv_len; ++j) {
                        scores_buf[j] = std::exp(scores_buf[j] - max_val);
                        sum += scores_buf[j];
                    }
                    float inv_sum = 1.0f / (sum + 1e-30f);

                    float* out_row = host_out.data() + i * num_heads * head_dim + h * head_dim;
                    std::memset(out_row, 0, head_dim * sizeof(float));
                    for (int j = 0; j < kv_len; ++j) {
                        float w = scores_buf[j] * inv_sum;
                        const float* v_row = v_data + j * num_heads * head_dim + h * head_dim;
                        forge::cpu::fmadd_f32(out_row, v_row, head_dim, w);
                    }
                }
            }
        }
    }

    std::memcpy(out->data(), host_out.data(), host_out.size() * sizeof(float));

    return out;
}

TensorPtr scaled_dot_product_attention_2d_masked(const TensorPtr& q, const TensorPtr& k,
                                                 const TensorPtr& v, int seq_len, int num_heads,
                                                 int head_dim, const TensorPtr& mask) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // For masked attention, we must use CPU path (flash attention doesn't support custom masks)
    auto q_cpu = q;
    auto k_cpu = k;
    auto v_cpu = v;
    auto mask_cpu = mask;
    if (q->device() == DeviceType::CUDA) {
        q_cpu = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
        k_cpu = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
        v_cpu = std::make_shared<Tensor>(DataType::FP32, v->shape(), DeviceType::CPU);
        mask_cpu = std::make_shared<Tensor>(DataType::FP32, mask->shape(), DeviceType::CPU);
        q_cpu->copy_from(*q);
        k_cpu->copy_from(*k);
        v_cpu->copy_from(*v);
        mask_cpu->copy_from(*mask);
    }

    const float* q_data = static_cast<const float*>(q_cpu->data());
    const float* k_data = static_cast<const float*>(k_cpu->data());
    const float* v_data = static_cast<const float*>(v_cpu->data());
    const float* mask_data = static_cast<const float*>(mask_cpu->data());

    auto out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CPU);
    float* out_data = static_cast<float*>(out->data());

#pragma omp parallel for schedule(dynamic) collapse(2) if (seq_len * num_heads > 4)
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            std::vector<float> scores(seq_len);
            float max_val = -1e30f;
            for (int j = 0; j < seq_len; ++j) {
                float m = mask_data[i * seq_len + j];
                if (m < -1e20f) {
                    scores[j] = -1e30f;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_data[i * num_heads * head_dim + h * head_dim + d] *
                           k_data[j * num_heads * head_dim + h * head_dim + d];
                }
                scores[j] = dot * scale + m;
                max_val = std::max(max_val, scores[j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                scores[j] = std::exp(scores[j] - max_val);
                sum += scores[j];
            }
            float inv_sum = 1.0f / (sum + 1e-30f);
            for (int d = 0; d < head_dim; ++d) {
                float val = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    val += scores[j] * v_data[j * num_heads * head_dim + h * head_dim + d];
                }
                out_data[i * num_heads * head_dim + h * head_dim + d] = val * inv_sum;
            }
        }
    }

    return out;
}

TensorPtr scaled_dot_product_attention_2d_gqa(const TensorPtr& q, const TensorPtr& k,
                                              const TensorPtr& v, int q_len, int kv_len,
                                              int num_heads, int num_kv_heads, int head_dim,
                                              const TensorPtr& mask, bool causal) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{q_len, num_heads * head_dim}, q->device());

    // CUDA path: use GQA flash attention kernels
    if (q->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        const float* q_data = static_cast<const float*>(q->data());
        const float* k_data = static_cast<const float*>(k->data());
        const float* v_data = static_cast<const float*>(v->data());
        float* out_data = static_cast<float*>(out->data());

        // If mask is on CPU, copy it to CUDA first
        const float* mask_ptr = nullptr;
        auto mask_cuda = mask;
        if (mask && mask->device() == DeviceType::CPU) {
            mask_cuda = std::make_shared<Tensor>(DataType::FP32, mask->shape(), DeviceType::CUDA);
            mask_cuda->copy_from(*mask);
        }
        if (mask_cuda) {
            mask_ptr = static_cast<const float*>(mask_cuda->data());
        }

        bool cuda_decode_supported = (head_dim == 64 || head_dim == 96 || head_dim == 128 ||
                                       head_dim == 256 || head_dim == 512);

        if (q_len == 1 && cuda_decode_supported) {
            // Decode path (supported head_dim)
            cuda::launch_flash_attention_gqa_decode(q_data, k_data, v_data, out_data,
                                                     kv_len, num_heads, num_kv_heads, head_dim,
                                                     mask_ptr);
            return out;
        } else if (q_len > 1) {
            // Prefill path
            cuda::launch_flash_attention_gqa(q_data, k_data, v_data, out_data,
                                              q_len, kv_len, num_heads, num_kv_heads, head_dim,
                                              mask_ptr, causal);
            return out;
        }
        // Fall through to CPU path for unsupported head_dim in decode mode
#endif
    }

    // CPU path
    const float* q_data = static_cast<const float*>(q->data());
    const float* k_data = static_cast<const float*>(k->data());
    const float* v_data = static_cast<const float*>(v->data());
    const float* mask_data = mask ? static_cast<const float*>(mask->data()) : nullptr;

    std::vector<float> host_out(out->numel());

    int kv_group_size = num_heads / num_kv_heads;  // e.g., 32/4 = 8

    if (q_len == 1) {
        // Decode path: online softmax with direct GQA head mapping
        // No need to physically expand K/V
        PERF_SCOPE("attention/qk_dot");
#pragma omp parallel for schedule(static)
        for (int h = 0; h < num_heads; ++h) {
            const float* q_row = q_data + h * head_dim;
            float* out_row = host_out.data() + h * head_dim;

            // Map query head to its corresponding KV head
            int kv_h = h / kv_group_size;

            float max_val = -1e30f;
            float sum_exp = 0.0f;
            std::memset(out_row, 0, head_dim * sizeof(float));

            for (int j = 0; j < kv_len; ++j) {
                // Direct GQA mapping: use kv_h instead of h
                const float* k_row = k_data + j * num_kv_heads * head_dim + kv_h * head_dim;
                float dot = forge::cpu::dot_f32(q_row, k_row, head_dim);
                float score = dot * scale;

                // Apply mask bias for decode (single row: mask[j])
                if (mask_data) {
                    float m = mask_data[j];
                    if (m < -1e20f) continue;  // skip masked positions
                    score += m;
                }

                float new_max = std::max(max_val, score);
                float rescale = std::exp(max_val - new_max);

                if (new_max > max_val) {
                    sum_exp *= rescale;
                    forge::cpu::scale_f32(out_row, head_dim, rescale);
                }

                float exp_score = std::exp(score - new_max);
                sum_exp += exp_score;

                const float* v_row = v_data + j * num_kv_heads * head_dim + kv_h * head_dim;
                forge::cpu::fmadd_f32(out_row, v_row, head_dim, exp_score);
                max_val = new_max;
            }

            float inv_sum = 1.0f / (sum_exp + 1e-30f);
            forge::cpu::scale_f32(out_row, head_dim, inv_sum);
        }
    } else {
        // Prefill path with GQA
        PERF_SCOPE("attention/qk_dot");
#pragma omp parallel
        {
            std::vector<float> scores_buf(kv_len);
#pragma omp for schedule(dynamic) collapse(2)
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < q_len; ++i) {
                    int kv_h = h / kv_group_size;
                    int q_pos = kv_len - q_len + i;
                    float max_val = -1e30f;
                    const float* q_row = q_data + i * num_heads * head_dim + h * head_dim;
                    for (int j = 0; j < kv_len; ++j) {
                        // Mask-aware masking: if mask is present, use it;
                        // otherwise fall back to causal mask
                        float m = 0.0f;
                        if (mask_data) {
                            m = mask_data[i * kv_len + j];
                            if (m < -1e20f) {
                                scores_buf[j] = -1e30f;
                                continue;
                            }
                        } else if (causal && j > q_pos) {
                            scores_buf[j] = -1e30f;
                            continue;
                        }
                        const float* k_row = k_data + j * num_kv_heads * head_dim + kv_h * head_dim;
                        float dot = forge::cpu::dot_f32(q_row, k_row, head_dim);
                        scores_buf[j] = dot * scale + m;
                        max_val = std::max(max_val, scores_buf[j]);
                    }

                    float sum = 0.0f;
                    for (int j = 0; j < kv_len; ++j) {
                        scores_buf[j] = std::exp(scores_buf[j] - max_val);
                        sum += scores_buf[j];
                    }
                    float inv_sum = 1.0f / (sum + 1e-30f);

                    float* out_row = host_out.data() + i * num_heads * head_dim + h * head_dim;
                    std::memset(out_row, 0, head_dim * sizeof(float));
                    for (int j = 0; j < kv_len; ++j) {
                        float w = scores_buf[j] * inv_sum;
                        const float* v_row = v_data + j * num_kv_heads * head_dim + kv_h * head_dim;
                        forge::cpu::fmadd_f32(out_row, v_row, head_dim, w);
                    }
                }
            }
        }
    }

    std::memcpy(out->data(), host_out.data(), host_out.size() * sizeof(float));

    return out;
}

}  // namespace ops
}  // namespace forge
