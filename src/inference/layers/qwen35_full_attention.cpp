#include "forge/inference/layers/qwen35_full_attention.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/inference/layers/attention_executor.h"
#include "forge/inference/layers/qwen35_rope.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/logger.h"
#include "forge/operators.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

namespace {

// 读取 per-head norm 权重到 CPU vector。非 FP32 权重退化为全 1,
// 与重构前的行为一致。
std::vector<float> load_head_norm_weight(const TensorPtr& w, int head_dim) {
    std::vector<float> out(head_dim, 1.0f);
    if (!w || w->dtype() != DataType::FP32) return out;
    auto cpu = ensure_cpu(w);
    std::memcpy(out.data(), cpu->data(), head_dim * sizeof(float));
    return out;
}

void per_head_rms_norm_cpu(float* data, const std::vector<float>& weight, int seq_len,
                           int num_heads, int head_dim, float eps) {
#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_heads > 4)
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            float* head_ptr = data + s * num_heads * head_dim + h * head_dim;
            float norm_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d) norm_sq += head_ptr[d] * head_ptr[d];
            float inv_rms = 1.0f / std::sqrt(norm_sq / head_dim + eps);
            for (int d = 0; d < head_dim; ++d) head_ptr[d] *= inv_rms * weight[d];
        }
    }
}

}  // namespace

TensorPtr Qwen35FullAttention::attend(const TensorPtr& normed,
                                      const LayerExecutionContext& lctx) {
    const auto& lw = lctx.weights;
    if (!lw.attn_q() || !lw.attn_k() || !lw.attn_v() || !lw.attn_output()) {
        LOG_ERROR("Qwen35FullAttention: layer " + std::to_string(lctx.layer_idx) +
                  " missing required attention weights");
        return nullptr;
    }
    if (lctx.device == DeviceType::CUDA) {
#ifdef USE_CUDA
        return attend_cuda(normed, lctx);
#else
        LOG_WARN("CUDA requested but not compiled; falling back to CPU for FullAttn layer " +
                 std::to_string(lctx.layer_idx));
#endif
    }
    return attend_cpu(normed, lctx);
}

TensorPtr Qwen35FullAttention::attend_cpu(const TensorPtr& normed,
                                          const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;
    const int q_dim = num_heads * head_dim;

    // Q 投影输出 [num_heads * head_dim * 2]: 每个 head 是 [Q | gate] 拼接。
    auto q_full = ensure_cpu(ops::matmul_transB(normed, lw.attn_q()));

    auto q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, q_dim},
                                      DeviceType::CPU);
    auto gate = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, q_dim},
                                        DeviceType::CPU);

    const float* qf_data = static_cast<const float*>(q_full->data());
    float* q_data = static_cast<float*>(q->data());
    float* g_data = static_cast<float*>(gate->data());

#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_heads > 4)
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            const float* src = qf_data + s * num_heads * head_dim * 2 + h * head_dim * 2;
            std::memcpy(q_data + s * q_dim + h * head_dim, src, head_dim * sizeof(float));
            std::memcpy(g_data + s * q_dim + h * head_dim, src + head_dim,
                        head_dim * sizeof(float));
        }
    }

    // K/V 投影后拉回 CPU 做 per-head 操作 (RMSNorm + RoPE)。
    auto k = ensure_cpu(ops::matmul_transB(normed, lw.attn_k()));
    auto v = ensure_cpu(ops::matmul_transB(normed, lw.attn_v()));

    if (lw.attn_q_norm()) {
        per_head_rms_norm_cpu(static_cast<float*>(q->data()),
                              load_head_norm_weight(lw.attn_q_norm(), head_dim), seq_len, num_heads,
                              head_dim, cfg.rms_norm_eps);
    }
    if (lw.attn_k_norm()) {
        per_head_rms_norm_cpu(static_cast<float*>(k->data()),
                              load_head_norm_weight(lw.attn_k_norm(), head_dim), seq_len,
                              num_kv_heads, head_dim, cfg.rms_norm_eps);
    }

    const int n_rot = Qwen35Rope::rot_dim(cfg);
    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
    Qwen35Rope::apply_cpu(static_cast<const float*>(q->data()),
                          static_cast<const float*>(k->data()),
                          static_cast<float*>(q_rope->data()),
                          static_cast<float*>(k_rope->data()), seq_len, num_heads, num_kv_heads,
                          head_dim, n_rot, lctx.start_pos(), cfg.rope_theta);

    kv_cache_.update(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), k_rope, v, seq_len);
    if (kv_cache_.kv_dtype() == KVCacheDType::Q4_0) {
        kv_cache_.dequantize_layer(lctx.layer_idx);
    }

    const int total_len = kv_cache_.filled(lctx.layer_idx);
    TensorPtr k_expanded = kv_cache_.get_key_filled(lctx.layer_idx);
    TensorPtr v_expanded = kv_cache_.get_value_filled(lctx.layer_idx);
    if (num_kv_heads < num_heads) {
        k_expanded = AttentionExecutor::expand_kv_heads(ensure_cpu(k_expanded), total_len,
                                                        num_heads, num_kv_heads, head_dim,
                                                        DeviceType::CPU);
        v_expanded = AttentionExecutor::expand_kv_heads(ensure_cpu(v_expanded), total_len,
                                                        num_heads, num_kv_heads, head_dim,
                                                        DeviceType::CPU);
    }

    // dev 为 CPU 时这三个搬运都是 no-op; 保留是为了兼容"CUDA 但走 CPU 前处理"的情形。
    auto q_attn = restore_device(q_rope, lctx.device);
    k_expanded = restore_device(k_expanded, lctx.device);
    v_expanded = restore_device(v_expanded, lctx.device);

    auto attn_out = ops::scaled_dot_product_attention_2d(q_attn, k_expanded, v_expanded, seq_len,
                                                         total_len, num_heads, head_dim, nullptr,
                                                         true);

    // Gated attention: output = sigmoid(gate) * attn_out
    auto attn_out_cpu = ensure_cpu(attn_out);
    float* attn_data = static_cast<float*>(attn_out_cpu->data());
    const float* gate_data = static_cast<const float*>(gate->data());
    for (int i = 0; i < seq_len * q_dim; ++i) {
        attn_data[i] *= 1.0f / (1.0f + std::exp(-gate_data[i]));
    }

    return ops::matmul_transB(restore_device(attn_out_cpu, lctx.device), lw.attn_output());
}

#ifdef USE_CUDA
TensorPtr Qwen35FullAttention::attend_cuda(const TensorPtr& normed,
                                           const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;
    const int q_dim = num_heads * head_dim;

    cudaStream_t stream = 0;

    auto to_gpu = [](const TensorPtr& t) { return restore_device(t, DeviceType::CUDA); };

    auto q_full = ops::matmul_transB(normed, lw.attn_q());
    auto k = ops::matmul_transB(normed, lw.attn_k());
    auto v = ops::matmul_transB(normed, lw.attn_v());

    auto q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, q_dim},
                                      DeviceType::CUDA);
    auto gate = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, q_dim},
                                        DeviceType::CUDA);
    forge::cuda::launch_split_q_gate(static_cast<const float*>(q_full->data()),
                                     static_cast<float*>(q->data()),
                                     static_cast<float*>(gate->data()), seq_len, num_heads,
                                     head_dim, stream);

    if (lw.attn_q_norm()) {
        auto q_norm_w = to_gpu(lw.attn_q_norm());
        forge::cuda::launch_rms_norm(static_cast<const float*>(q->data()),
                                     static_cast<const float*>(q_norm_w->data()),
                                     static_cast<float*>(q->data()), seq_len * num_heads, head_dim,
                                     cfg.rms_norm_eps, stream);
    }
    if (lw.attn_k_norm()) {
        auto k_norm_w = to_gpu(lw.attn_k_norm());
        forge::cuda::launch_rms_norm(static_cast<const float*>(k->data()),
                                     static_cast<const float*>(k_norm_w->data()),
                                     static_cast<float*>(k->data()), seq_len * num_kv_heads,
                                     head_dim, cfg.rms_norm_eps, stream);
    }

    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CUDA);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CUDA);
    forge::cuda::launch_rope_gqa(static_cast<const float*>(q->data()),
                                 static_cast<const float*>(k->data()),
                                 static_cast<float*>(q_rope->data()),
                                 static_cast<float*>(k_rope->data()), num_heads, num_kv_heads,
                                 head_dim, seq_len, lctx.start_pos(), cfg.rope_theta, stream);

    kv_cache_.update(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), k_rope, v, seq_len);
    if (kv_cache_.kv_dtype() == KVCacheDType::Q4_0) {
        kv_cache_.dequantize_layer(lctx.layer_idx);
    }

    const int total_len = kv_cache_.filled(lctx.layer_idx);
    TensorPtr k_expanded = kv_cache_.get_key_filled(lctx.layer_idx);
    TensorPtr v_expanded = kv_cache_.get_value_filled(lctx.layer_idx);
    if (num_kv_heads < num_heads) {
        k_expanded = AttentionExecutor::expand_kv_heads(k_expanded, total_len, num_heads,
                                                        num_kv_heads, head_dim, DeviceType::CUDA);
        v_expanded = AttentionExecutor::expand_kv_heads(v_expanded, total_len, num_heads,
                                                        num_kv_heads, head_dim, DeviceType::CUDA);
    }

    auto attn_out = ops::scaled_dot_product_attention_2d(q_rope, k_expanded, v_expanded, seq_len,
                                                         total_len, num_heads, head_dim, nullptr,
                                                         true);

    forge::cuda::launch_sigmoid_multiply(static_cast<const float*>(gate->data()),
                                         static_cast<float*>(attn_out->data()), seq_len * q_dim,
                                         stream);

    return ops::matmul_transB(attn_out, lw.attn_output());
}
#endif

}  // namespace forge
