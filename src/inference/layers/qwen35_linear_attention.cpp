#include "forge/inference/layers/qwen35_linear_attention.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/logger.h"
#include "forge/operators.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

namespace {

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    float f;
    if (exp == 0) {
        f = mant == 0 ? 0.0f : std::ldexp(static_cast<float>(mant), -24);
    } else if (exp == 31) {
        f = mant == 0 ? std::numeric_limits<float>::infinity()
                      : std::numeric_limits<float>::quiet_NaN();
    } else {
        f = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
    }
    return sign ? -f : f;
}

// conv1d 权重展开为 FP32 [conv_channels, d_conv]。非 FP32/FP16 时保持全 0,
// 与重构前一致。
std::vector<float> load_conv_weight(const TensorPtr& w, int d_conv, int conv_channels) {
    std::vector<float> out(static_cast<size_t>(d_conv) * conv_channels, 0.0f);
    if (!w)
        return out;
    auto cpu = ensure_cpu(w);
    if (cpu->dtype() == DataType::FP32) {
        std::memcpy(out.data(), cpu->data(), out.size() * sizeof(float));
    } else if (cpu->dtype() == DataType::FP16) {
        const uint16_t* fp16 = static_cast<const uint16_t*>(cpu->data());
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = fp16_to_fp32(fp16[i]);
    }
    return out;
}

}  // namespace

void Qwen35LinearAttention::gated_delta_net_step(const float* q, const float* k, const float* v,
                                                 const float* gate, const float* beta, float* state,
                                                 float* output, int head_k_dim, int head_v_dim,
                                                 int num_k_heads, int num_v_heads) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    int head_repeat = num_v_heads / num_k_heads;

#pragma omp parallel for schedule(static) if (num_v_heads > 1)
    for (int hv = 0; hv < num_v_heads; ++hv) {
        int hk = hv / head_repeat;

        float* S = state + hv * head_v_dim * head_v_dim;
        const float* q_h = q + hk * head_k_dim;
        const float* k_h = k + hk * head_k_dim;
        const float* v_h = v + hv * head_v_dim;
        float g = std::exp(gate[hv]);
        float b = beta[hv];

        for (int i = 0; i < head_v_dim * head_v_dim; ++i) {
            S[i] *= g;
        }

        // delta = (v - S^T @ k) * beta, 采用 llama.cpp 的 sk[i] = sum_j(S[j,i] * k[j]) 约定
        std::vector<float> delta(head_v_dim);
        for (int i = 0; i < head_v_dim; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < head_k_dim; ++j) {
                sum += S[j * head_v_dim + i] * k_h[j];
            }
            delta[i] = (v_h[i] - sum) * b;
        }

        // S[j,i] += k[j] * delta[i]
        for (int j = 0; j < head_v_dim; ++j) {
            for (int i = 0; i < head_k_dim; ++i) {
                S[j * head_v_dim + i] += k_h[j] * delta[i];
            }
        }

        // o[i] = sum_j(S[j,i] * q[j]) * scale
        float* o = output + hv * head_v_dim;
        for (int i = 0; i < head_v_dim; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < head_k_dim; ++j) {
                sum += S[j * head_v_dim + i] * q_h[j];
            }
            o[i] = sum * scale;
        }
    }
}

void Qwen35LinearAttention::conv1d_cpu(const float* x_data, const float* weight_data, float* y_data,
                                       float* conv_state, int seq_len, int conv_channels,
                                       int d_conv) {
    int state_len = d_conv - 1;

    for (int s = 0; s < seq_len; ++s) {
        const float* x_row = x_data + s * conv_channels;
        float* y_row = y_data + s * conv_channels;

#pragma omp parallel for schedule(static) if (conv_channels > 64)
        for (int c = 0; c < conv_channels; ++c) {
            float val = 0.0f;
            for (int k = 0; k < d_conv; ++k) {
                float x_val;
                if (k < state_len) {
                    x_val = conv_state[k * conv_channels + c];
                } else {
                    x_val = x_row[c];
                }
                val += x_val * weight_data[c * d_conv + k];
            }
            y_row[c] = val;

            for (int k = 0; k < state_len - 1; ++k) {
                conv_state[k * conv_channels + c] = conv_state[(k + 1) * conv_channels + c];
            }
            conv_state[(state_len - 1) * conv_channels + c] = x_row[c];
        }
    }
}

TensorPtr Qwen35LinearAttention::apply(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& lw = lctx.weights;
    if (!lw.attn_qkv() || !lw.attn_gate() || !lw.ssm_conv1d() || !lw.ssm_alpha() ||
        !lw.ssm_beta() || !lw.ssm_norm() || !lw.ssm_out()) {
        LOG_ERROR("Qwen35LinearAttention: layer " + std::to_string(lctx.layer_idx) +
                  " missing required SSM weights");
        return nullptr;
    }
    if (lctx.device == DeviceType::CUDA) {
#ifdef USE_CUDA
        return apply_cuda(normed, lctx);
#else
        LOG_WARN("CUDA requested but not compiled; falling back to CPU for SSM layer " +
                 std::to_string(lctx.layer_idx));
#endif
    }
    return apply_cpu(normed, lctx);
}

TensorPtr Qwen35LinearAttention::apply_cpu(const TensorPtr& normed,
                                           const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const auto& dims = memory_.dims();
    const int seq_len = lctx.seq_len();
    const int layer_idx = lctx.layer_idx;
    const int seq_id = lctx.request.seq_id;

    const int head_k_dim = dims.head_k_dim();
    const int num_k_heads = dims.num_k_heads();
    const int num_v_heads = dims.num_v_heads();
    const int head_v_dim = dims.head_v_dim;
    const int d_conv = dims.d_conv;
    const int conv_channels = dims.conv_channels;
    const int key_dim = dims.key_dim();
    const int value_dim = dims.value_dim();

    // 输入投影后统一拉回 CPU: 递推部分是标量循环。
    // 四个投影共享同一输入 (normed): 若 qkv 为 Q4_K/Q6_K 且 z/alpha/beta 为 Q4_K
    // 则走融合内核, 一次 Q8_K 量化 + 单个 OpenMP 区完成 qkv/z/alpha/beta。
    TensorPtr qkv_mixed, z, alpha, beta;
    auto qkv_dt = lw.attn_qkv()->dtype();
    bool qkv_z_ab_fused = (qkv_dt == DataType::Q4_K || qkv_dt == DataType::Q6_K) &&
                          lw.attn_gate()->dtype() == DataType::Q4_K &&
                          lw.ssm_alpha()->dtype() == DataType::Q4_K &&
                          lw.ssm_beta()->dtype() == DataType::Q4_K;
    if (qkv_z_ab_fused) {
        auto normed_cpu = ensure_cpu(normed);
        auto fused = ops::matmul_transB_fused_qkv_z_ab(normed_cpu, lw.attn_qkv(), lw.attn_gate(),
                                                       lw.ssm_alpha(), lw.ssm_beta());
        qkv_mixed = fused.qkv;
        z = fused.z;
        alpha = fused.alpha;
        beta = fused.beta;
    } else {
        qkv_mixed = ensure_cpu(ops::matmul_transB(normed, lw.attn_qkv()));
        z = ensure_cpu(ops::matmul_transB(normed, lw.attn_gate()));
        alpha = ensure_cpu(ops::matmul_transB(normed, lw.ssm_alpha()));
        beta = ensure_cpu(ops::matmul_transB(normed, lw.ssm_beta()));
    }

    float* alpha_data = static_cast<float*>(alpha->data());
    float* beta_data = static_cast<float*>(beta->data());

    // alpha: + dt_bias -> softplus -> * ssm_a
    if (lw.ssm_dt()) {
        auto dt_cpu = ensure_cpu(lw.ssm_dt());
        const float* dt_bias = static_cast<const float*>(dt_cpu->data());
        for (int s = 0; s < seq_len; ++s) {
            for (int j = 0; j < num_v_heads; ++j) {
                alpha_data[s * num_v_heads + j] += dt_bias[j];
            }
        }
    }
    for (int i = 0; i < seq_len * num_v_heads; ++i) {
        alpha_data[i] = std::log(1.0f + std::exp(alpha_data[i]));
    }
    if (lw.ssm_a()) {
        auto a_cpu = ensure_cpu(lw.ssm_a());
        const float* a_data = static_cast<const float*>(a_cpu->data());
        for (int s = 0; s < seq_len; ++s) {
            for (int j = 0; j < num_v_heads; ++j) {
                alpha_data[s * num_v_heads + j] *= a_data[j];
            }
        }
    }
    for (int i = 0; i < seq_len * num_v_heads; ++i) {
        beta_data[i] = 1.0f / (1.0f + std::exp(-beta_data[i]));
    }

    // causal conv1d + SiLU
    auto conv_w = load_conv_weight(lw.ssm_conv1d(), d_conv, conv_channels);
    auto conv_out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len, conv_channels}, DeviceType::CPU);
    conv1d_cpu(static_cast<const float*>(qkv_mixed->data()), conv_w.data(),
               static_cast<float*>(conv_out->data()), memory_.conv_state_cpu(seq_id, layer_idx),
               seq_len, conv_channels, d_conv);

    float* conv_data = static_cast<float*>(conv_out->data());
    for (int i = 0; i < seq_len * conv_channels; ++i) {
        float x = conv_data[i];
        conv_data[i] = x / (1.0f + std::exp(-x));
    }

    // split -> Q/K/V
    auto q_conv = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, key_dim},
                                           DeviceType::CPU);
    auto k_conv = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, key_dim},
                                           DeviceType::CPU);
    auto v_conv = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, value_dim},
                                           DeviceType::CPU);
    float* q_conv_data = static_cast<float*>(q_conv->data());
    float* k_conv_data = static_cast<float*>(k_conv->data());
    float* v_conv_data = static_cast<float*>(v_conv->data());

    for (int s = 0; s < seq_len; ++s) {
        const float* src = conv_data + s * conv_channels;
        std::memcpy(q_conv_data + s * key_dim, src, key_dim * sizeof(float));
        std::memcpy(k_conv_data + s * key_dim, src + key_dim, key_dim * sizeof(float));
        std::memcpy(v_conv_data + s * value_dim, src + 2 * key_dim, value_dim * sizeof(float));
    }

    // per-head L2 normalize Q/K
#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_k_heads > 4)
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_k_heads; ++h) {
            float* q_head = q_conv_data + s * key_dim + h * head_k_dim;
            float norm = 0.0f;
            for (int d = 0; d < head_k_dim; ++d)
                norm += q_head[d] * q_head[d];
            float inv_norm = 1.0f / std::sqrt(norm + cfg.rms_norm_eps);
            for (int d = 0; d < head_k_dim; ++d)
                q_head[d] *= inv_norm;

            float* k_head = k_conv_data + s * key_dim + h * head_k_dim;
            norm = 0.0f;
            for (int d = 0; d < head_k_dim; ++d)
                norm += k_head[d] * k_head[d];
            inv_norm = 1.0f / std::sqrt(norm + cfg.rms_norm_eps);
            for (int d = 0; d < head_k_dim; ++d)
                k_head[d] *= inv_norm;
        }
    }

    // Gated Delta Net 逐 token 递推
    auto delta_net_out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len, value_dim}, DeviceType::CPU);
    float* dn_out_data = static_cast<float*>(delta_net_out->data());
    float* state = memory_.ssm_state_cpu(seq_id, layer_idx);

    for (int s = 0; s < seq_len; ++s) {
        gated_delta_net_step(q_conv_data + s * key_dim, k_conv_data + s * key_dim,
                             v_conv_data + s * value_dim, alpha_data + s * num_v_heads,
                             beta_data + s * num_v_heads, state, dn_out_data + s * value_dim,
                             head_k_dim, head_v_dim, num_k_heads, num_v_heads);
    }

    // gated norm: per-head RMSNorm + SiLU(z)
    const float* z_data = static_cast<const float*>(z->data());
    std::vector<float> ssm_norm_w(head_v_dim, 1.0f);
    {
        auto norm_cpu = ensure_cpu(lw.ssm_norm());
        if (norm_cpu->dtype() == DataType::FP32) {
            std::memcpy(ssm_norm_w.data(), norm_cpu->data(), head_v_dim * sizeof(float));
        }
    }

#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_v_heads > 4)
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_v_heads; ++h) {
            float* out_head = dn_out_data + s * value_dim + h * head_v_dim;
            const float* z_head = z_data + s * value_dim + h * head_v_dim;

            float norm_sq = 0.0f;
            for (int d = 0; d < head_v_dim; ++d)
                norm_sq += out_head[d] * out_head[d];
            float inv_rms = 1.0f / std::sqrt(norm_sq / head_v_dim + cfg.rms_norm_eps);
            for (int d = 0; d < head_v_dim; ++d)
                out_head[d] *= inv_rms * ssm_norm_w[d];

            for (int d = 0; d < head_v_dim; ++d) {
                float gz = z_head[d] / (1.0f + std::exp(-z_head[d]));
                out_head[d] *= gz;
            }
        }
    }

    return ops::matmul_transB(restore_device(delta_net_out, lctx.device), lw.ssm_out());
}

#ifdef USE_CUDA
TensorPtr Qwen35LinearAttention::apply_cuda(const TensorPtr& normed,
                                            const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const auto& dims = memory_.dims();
    const int seq_len = lctx.seq_len();
    const int layer_idx = lctx.layer_idx;
    const int seq_id = lctx.request.seq_id;

    const int head_k_dim = dims.head_k_dim();
    const int num_k_heads = dims.num_k_heads();
    const int num_v_heads = dims.num_v_heads();
    const int head_v_dim = dims.head_v_dim;
    const int d_conv = dims.d_conv;
    const int conv_channels = dims.conv_channels;
    const int key_dim = dims.key_dim();
    const int value_dim = dims.value_dim();

    auto qkv_mixed = ops::matmul_transB(normed, lw.attn_qkv());
    auto z = ops::matmul_transB(normed, lw.attn_gate());
    auto alpha = ops::matmul_transB(normed, lw.ssm_alpha());
    auto beta = ops::matmul_transB(normed, lw.ssm_beta());

    auto make_gpu = [](std::vector<int64_t> shape) {
        return std::make_shared<Tensor>(DataType::FP32, std::move(shape), DeviceType::CUDA);
    };
    auto gate_out = make_gpu({seq_len, num_v_heads});
    auto beta_out = make_gpu({seq_len, num_v_heads});
    auto conv_out = make_gpu({seq_len, conv_channels});
    auto q_conv = make_gpu({seq_len, key_dim});
    auto k_conv = make_gpu({seq_len, key_dim});
    auto v_conv = make_gpu({seq_len, value_dim});
    auto delta_net_out = make_gpu({seq_len, value_dim});

    // SSM 权重需要 GPU 上的 FP32 视图; holder 保持临时张量存活到 kernel 结束。
    auto ensure_gpu_fp32 = [](const TensorPtr& t) -> std::pair<const float*, TensorPtr> {
        if (!t)
            return {nullptr, nullptr};
        if (t->device() == DeviceType::CUDA && t->dtype() == DataType::FP32) {
            return {static_cast<const float*>(t->data()), nullptr};
        }
        auto gpu = std::make_shared<Tensor>(DataType::FP32, t->shape(), DeviceType::CUDA);
        gpu->copy_from(*t);
        return {static_cast<const float*>(gpu->data()), gpu};
    };
    auto [conv_weight, conv_holder] = ensure_gpu_fp32(lw.ssm_conv1d());
    auto [dt_bias, dt_holder] = ensure_gpu_fp32(lw.ssm_dt());
    auto [ssm_a_ptr, sa_holder] = ensure_gpu_fp32(lw.ssm_a());
    auto [ssm_norm_w, sn_holder] = ensure_gpu_fp32(lw.ssm_norm());

    cudaStream_t stream = 0;

    float* gate_out_data = static_cast<float*>(gate_out->data());
    float* beta_out_data = static_cast<float*>(beta_out->data());
    forge::cuda::launch_ssm_preprocess(static_cast<const float*>(alpha->data()),
                                       static_cast<const float*>(beta->data()), dt_bias, ssm_a_ptr,
                                       gate_out_data, beta_out_data, seq_len, num_v_heads, stream);

    float* conv_out_data = static_cast<float*>(conv_out->data());
    forge::cuda::launch_ssm_conv1d(static_cast<const float*>(qkv_mixed->data()), conv_weight,
                                   memory_.conv_state_gpu(seq_id, layer_idx), conv_out_data,
                                   seq_len, conv_channels, d_conv, stream);

    float* q_conv_data = static_cast<float*>(q_conv->data());
    float* k_conv_data = static_cast<float*>(k_conv->data());
    float* v_conv_data = static_cast<float*>(v_conv->data());
    forge::cuda::launch_ssm_silu_split(conv_out_data, q_conv_data, k_conv_data, v_conv_data,
                                       seq_len, key_dim, value_dim, stream);

    forge::cuda::launch_ssm_per_head_l2norm(q_conv_data, seq_len, num_k_heads, head_k_dim,
                                            cfg.rms_norm_eps, stream);
    forge::cuda::launch_ssm_per_head_l2norm(k_conv_data, seq_len, num_k_heads, head_k_dim,
                                            cfg.rms_norm_eps, stream);

    float* dn_out_data = static_cast<float*>(delta_net_out->data());
    forge::cuda::launch_ssm_gated_delta_net(q_conv_data, k_conv_data, v_conv_data, gate_out_data,
                                            beta_out_data, memory_.ssm_state_gpu(seq_id, layer_idx),
                                            dn_out_data, seq_len, head_k_dim, head_v_dim,
                                            num_k_heads, num_v_heads, stream);

    forge::cuda::launch_ssm_gated_norm(dn_out_data, static_cast<const float*>(z->data()),
                                       ssm_norm_w, seq_len, head_v_dim, num_v_heads,
                                       cfg.rms_norm_eps, stream);

    return ops::matmul_transB(delta_net_out, lw.ssm_out());
}
#endif

}  // namespace forge
