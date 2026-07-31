#include "forge/inference/layers/rope_executor.h"

#include <cmath>

#include "forge/cuda_kernels.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {

// ============================================================================
// CPU RoPE helpers (行为与 GenericEngine 内原 file-local 版本一致)
// ============================================================================

static void apply_rope_standard_cpu(const float* q_data, const float* k_data, float* q_out,
                                    float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                    int head_dim, int64_t start_pos, float theta) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = (start_pos + s) * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int q_idx0 = s * q_stride + h * head_dim + d;
                int q_idx1 = q_idx0 + half_dim;

                q_out[q_idx0] = q_data[q_idx0] * cos_a - q_data[q_idx1] * sin_a;
                q_out[q_idx1] = q_data[q_idx0] * sin_a + q_data[q_idx1] * cos_a;

                if (h < num_kv_heads) {
                    int k_idx0 = s * k_stride + h * head_dim + d;
                    int k_idx1 = k_idx0 + half_dim;

                    k_out[k_idx0] = k_data[k_idx0] * cos_a - k_data[k_idx1] * sin_a;
                    k_out[k_idx1] = k_data[k_idx0] * sin_a + k_data[k_idx1] * cos_a;
                }
            }
        }
    }
}

static void apply_rope_neox_cpu(const float* q_data, const float* k_data, float* q_out,
                                float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                int head_dim, int64_t start_pos, float theta) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = pos * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int q_idx0 = s * q_stride + h * head_dim + d;
                int q_idx1 = q_idx0 + half_dim;

                q_out[q_idx0] = q_data[q_idx0] * cos_a - q_data[q_idx1] * sin_a;
                q_out[q_idx1] = q_data[q_idx0] * sin_a + q_data[q_idx1] * cos_a;

                if (h < num_kv_heads) {
                    int k_idx0 = s * k_stride + h * head_dim + d;
                    int k_idx1 = k_idx0 + half_dim;

                    k_out[k_idx0] = k_data[k_idx0] * cos_a - k_data[k_idx1] * sin_a;
                    k_out[k_idx1] = k_data[k_idx0] * sin_a + k_data[k_idx1] * cos_a;
                }
            }
        }
    }
}

static void apply_rope_mrope_cpu(const float* q_data, const float* k_data, float* q_out,
                                 float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                 int head_dim, int n_rot, int64_t start_pos, float theta) {
    int half_rot = n_rot / 2;
    float theta_scale = 1.0f / theta;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;

#pragma omp parallel for schedule(static) if (seq_len > 1)
    for (int s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;

        for (int h = 0; h < num_heads; ++h) {
            const float* q_src = q_data + s * q_stride + h * head_dim;
            float* q_dst = q_out + s * q_stride + h * head_dim;

            for (int d = 0; d < half_rot; ++d) {
                float freq = std::pow(theta_scale, 2.0f * d / n_rot);
                float angle = pos * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float x0 = q_src[d];
                float x1 = q_src[d + half_rot];

                q_dst[d] = x0 * cos_a - x1 * sin_a;
                q_dst[d + half_rot] = x0 * sin_a + x1 * cos_a;
            }
            for (int d = n_rot; d < head_dim; ++d) {
                q_dst[d] = q_src[d];
            }
        }

        for (int h = 0; h < num_kv_heads; ++h) {
            const float* k_src = k_data + s * k_stride + h * head_dim;
            float* k_dst = k_out + s * k_stride + h * head_dim;

            for (int d = 0; d < half_rot; ++d) {
                float freq = std::pow(theta_scale, 2.0f * d / n_rot);
                float angle = pos * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float x0 = k_src[d];
                float x1 = k_src[d + half_rot];

                k_dst[d] = x0 * cos_a - x1 * sin_a;
                k_dst[d + half_rot] = x0 * sin_a + x1 * cos_a;
            }
            for (int d = n_rot; d < head_dim; ++d) {
                k_dst[d] = k_src[d];
            }
        }
    }
}

// Proportional RoPE: same as NeoX but with per-dimension frequency scaling factors.
static void apply_rope_proportional_cpu(const float* q_data, const float* k_data, float* q_out,
                                        float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                        int head_dim, int64_t start_pos, float theta,
                                        const float* freq_factors) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float base_freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float freq = freq_factors ? base_freq / freq_factors[d] : base_freq;
                float angle = pos * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int q_idx0 = s * q_stride + h * head_dim + d;
                int q_idx1 = q_idx0 + half_dim;

                q_out[q_idx0] = q_data[q_idx0] * cos_a - q_data[q_idx1] * sin_a;
                q_out[q_idx1] = q_data[q_idx0] * sin_a + q_data[q_idx1] * cos_a;

                if (h < num_kv_heads) {
                    int k_idx0 = s * k_stride + h * head_dim + d;
                    int k_idx1 = k_idx0 + half_dim;

                    k_out[k_idx0] = k_data[k_idx0] * cos_a - k_data[k_idx1] * sin_a;
                    k_out[k_idx1] = k_data[k_idx0] * sin_a + k_data[k_idx1] * cos_a;
                }
            }
        }
    }
}

const float* RopeExecutor::freq_factors_cpu() {
    if (!rope_freqs_) return nullptr;
    if (rope_freqs_->device() == DeviceType::CUDA) {
        rope_freqs_cpu_ =
            std::make_shared<Tensor>(DataType::FP32, rope_freqs_->shape(), DeviceType::CPU);
        rope_freqs_cpu_->copy_from(*rope_freqs_);
    } else {
        rope_freqs_cpu_ = rope_freqs_;
    }
    return static_cast<const float*>(rope_freqs_cpu_->data());
}

RopeExecutor::Result RopeExecutor::apply(const TensorPtr& q, const TensorPtr& k,
                                        const ModelConfig& cfg, int64_t start_pos, int seq_len,
                                        DeviceType dev) {
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;

    // Determine RopeType
    RopeType rope_type = cfg.rope_type;
    if (cfg.use_mrope) {
        rope_type = RopeType::MRoPE;
    } else if (cfg.use_neox_rope) {
        rope_type = RopeType::NeoX;
    }

    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), dev);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), dev);

    int n_rot = cfg.use_mrope ? cfg.rope_dimension_count : head_dim;
    if (n_rot <= 0) n_rot = head_dim;

    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        // For MRoPE with partial rotation or when QK-norm forced Q/K to CPU,
        // fall back to CPU RoPE
        bool use_cpu_rope = (rope_type == RopeType::MRoPE && n_rot < head_dim) ||
                            q->device() != DeviceType::CUDA || k->device() != DeviceType::CUDA;

        if (use_cpu_rope) {
            auto to_cpu = [](const TensorPtr& t) {
                if (t->device() == DeviceType::CPU) return t;
                auto c = std::make_shared<Tensor>(DataType::FP32, t->shape(), DeviceType::CPU);
                c->copy_from(*t);
                return c;
            };
            auto q_cpu = to_cpu(q);
            auto k_cpu = to_cpu(k);
            auto q_rope_cpu =
                std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
            auto k_rope_cpu =
                std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);

            if (rope_type == RopeType::MRoPE) {
                apply_rope_mrope_cpu(static_cast<const float*>(q_cpu->data()),
                                     static_cast<const float*>(k_cpu->data()),
                                     static_cast<float*>(q_rope_cpu->data()),
                                     static_cast<float*>(k_rope_cpu->data()), seq_len, num_heads,
                                     num_kv_heads, head_dim, n_rot, start_pos, cfg.rope_theta);
            } else if (rope_type == RopeType::NeoX) {
                apply_rope_neox_cpu(static_cast<const float*>(q_cpu->data()),
                                    static_cast<const float*>(k_cpu->data()),
                                    static_cast<float*>(q_rope_cpu->data()),
                                    static_cast<float*>(k_rope_cpu->data()), seq_len, num_heads,
                                    num_kv_heads, head_dim, start_pos, cfg.rope_theta);
            } else if (rope_type == RopeType::Proportional) {
                apply_rope_proportional_cpu(
                    static_cast<const float*>(q_cpu->data()),
                    static_cast<const float*>(k_cpu->data()),
                    static_cast<float*>(q_rope_cpu->data()),
                    static_cast<float*>(k_rope_cpu->data()), seq_len, num_heads, num_kv_heads,
                    head_dim, start_pos, cfg.rope_theta, freq_factors_cpu());
            } else if (rope_type == RopeType::None) {
                q_rope_cpu->copy_from(*q_cpu);
                k_rope_cpu->copy_from(*k_cpu);
            } else {
                apply_rope_standard_cpu(static_cast<const float*>(q_cpu->data()),
                                        static_cast<const float*>(k_cpu->data()),
                                        static_cast<float*>(q_rope_cpu->data()),
                                        static_cast<float*>(k_rope_cpu->data()), seq_len, num_heads,
                                        num_kv_heads, head_dim, start_pos, cfg.rope_theta);
            }
            q_rope->copy_from(*q_rope_cpu);
            k_rope->copy_from(*k_rope_cpu);
        } else {
            // CUDA kernel path (standard/NeoX — same kernel handles both)
            cuda::launch_rope_gqa(
                static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), num_heads,
                num_kv_heads, head_dim, seq_len, start_pos, cfg.rope_theta);
        }
#else
        // CUDA not compiled — use CPU fallback
        apply_rope_standard_cpu(static_cast<const float*>(q->data()),
                                static_cast<const float*>(k->data()),
                                static_cast<float*>(q_rope->data()),
                                static_cast<float*>(k_rope->data()), seq_len, num_heads,
                                num_kv_heads, head_dim, start_pos, cfg.rope_theta);
#endif
    } else {
        // CPU path
        switch (rope_type) {
            case RopeType::None:
                q_rope->copy_from(*q);
                k_rope->copy_from(*k);
                break;
            case RopeType::NeoX:
                apply_rope_neox_cpu(static_cast<const float*>(q->data()),
                                    static_cast<const float*>(k->data()),
                                    static_cast<float*>(q_rope->data()),
                                    static_cast<float*>(k_rope->data()), seq_len, num_heads,
                                    num_kv_heads, head_dim, start_pos, cfg.rope_theta);
                break;
            case RopeType::MRoPE:
                apply_rope_mrope_cpu(static_cast<const float*>(q->data()),
                                     static_cast<const float*>(k->data()),
                                     static_cast<float*>(q_rope->data()),
                                     static_cast<float*>(k_rope->data()), seq_len, num_heads,
                                     num_kv_heads, head_dim, n_rot, start_pos, cfg.rope_theta);
                break;
            case RopeType::Proportional:
                apply_rope_proportional_cpu(
                    static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                    static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                    seq_len, num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta,
                    freq_factors_cpu());
                break;
            default:
                // Standard, LinearScaling, NTK_Scaled all share the standard kernel
                apply_rope_standard_cpu(static_cast<const float*>(q->data()),
                                        static_cast<const float*>(k->data()),
                                        static_cast<float*>(q_rope->data()),
                                        static_cast<float*>(k_rope->data()), seq_len, num_heads,
                                        num_kv_heads, head_dim, start_pos, cfg.rope_theta);
                break;
        }
    }

    // Apply Q scaling after RoPE (Gemma/Gemma2/Gemma4 pattern: rope_q_scale > 0)
    if (cfg.rope_q_scale > 0.0f) {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        bool was_cuda = (q_rope->device() == DeviceType::CUDA);
        if (was_cuda) {
            auto q_cpu = std::make_shared<Tensor>(DataType::FP32, q_rope->shape(), DeviceType::CPU);
            q_cpu->copy_from(*q_rope);
            q_rope = q_cpu;
        }
        int n = static_cast<int>(q_rope->numel());
        float* data = static_cast<float*>(q_rope->data());
        for (int i = 0; i < n; ++i) {
            data[i] *= scale;
        }
        if (was_cuda) {
            auto q_back =
                std::make_shared<Tensor>(DataType::FP32, q_rope->shape(), DeviceType::CUDA);
            q_back->copy_from(*q_rope);
            q_rope = q_back;
        }
    }

    return {q_rope, k_rope};
}

RopeExecutor::Result RopeExecutor::apply_standard(const TensorPtr& q, const TensorPtr& k,
                                                 int num_heads, int num_kv_heads, int head_dim,
                                                 int seq_len, int64_t start_pos, float theta,
                                                 DeviceType dev) {
    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), dev);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), dev);

    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_rope_gqa(
            static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
            static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), num_heads,
            num_kv_heads, head_dim, seq_len, start_pos, theta);
#endif
    } else {
        apply_rope_standard_cpu(
            static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
            static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), seq_len,
            num_heads, num_kv_heads, head_dim, start_pos, theta);
    }

    return {q_rope, k_rope};
}

}  // namespace forge
