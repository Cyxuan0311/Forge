#include "cpu/common/thread_pool.h"
#include "forge/inference/layers/qwen35_full_attention.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "forge/cuda_kernels.h"  // fused + paged decode launches (F16/Q8_0/Q4_0/FP8)
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
    if (!w || w->dtype() != DataType::FP32)
        return out;
    auto cpu = ensure_cpu(w);
    std::memcpy(out.data(), cpu->data(), head_dim * sizeof(float));
    return out;
}

void per_head_rms_norm_cpu(float* data, const std::vector<float>& weight, int seq_len,
                           int num_heads, int head_dim, float eps) {
    cpu::parallel_for2(seq_len, num_heads, [&](int s, int h) {
            float* head_ptr = data + s * num_heads * head_dim + h * head_dim;
            float norm_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d)
                norm_sq += head_ptr[d] * head_ptr[d];
            float inv_rms = 1.0f / std::sqrt(norm_sq / head_dim + eps);
            for (int d = 0; d < head_dim; ++d)
                head_ptr[d] *= inv_rms * weight[d];
        });
}

}  // namespace

TensorPtr Qwen35FullAttention::attend(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& lw = lctx.weights;
    if (!lw.attn_q() || !lw.attn_k() || !lw.attn_v() || !lw.attn_output()) {
        LOG_ERROR("Qwen35FullAttention: layer " + std::to_string(lctx.layer_idx) +
                  " missing required attention weights");
        return nullptr;
    }
    // Keep the attention backend's storage binding in sync with the live KV
    // memory (paged vs contiguous). The mode is fixed for the engine's lifetime,
    // but binding it here makes the layer robust to construction ordering.
    if (kv_memory_) {
        attn_.set_storage(kv_memory_->is_paged() ? &kv_memory_->storage() : nullptr);
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
    TensorPtr q_full, k, v;
    if (lw.attn_q()->dtype() == DataType::Q4_K && lw.attn_k()->dtype() == DataType::Q4_K &&
        lw.attn_v()->dtype() == DataType::Q4_K) {
        // 三个投影共享同一输入: 融合为单次量化 + 单 OpenMP 区 (省 2 个 gemm 调用)。
        // 返回拼接 [q|k|v], 按 N 切分 (M 很小, 拷贝开销可忽略)。
        auto qkv_fused =
            ops::matmul_transB_fused_qkv_q4_k(normed, lw.attn_q(), lw.attn_k(), lw.attn_v());
        int Nq = static_cast<int>(lw.attn_q()->shape()[0]);
        int Nk = static_cast<int>(lw.attn_k()->shape()[0]);
        int Nv = static_cast<int>(lw.attn_v()->shape()[0]);
        const int row_stride = Nq + Nk + Nv;
        const float* src = static_cast<const float*>(qkv_fused->data());
        q_full = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, Nq},
                                          DeviceType::CPU);
        k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, Nk},
                                     DeviceType::CPU);
        v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, Nv},
                                     DeviceType::CPU);
        for (int s = 0; s < seq_len; ++s) {
            const float* row = src + s * row_stride;
            std::memcpy(static_cast<float*>(q_full->data()) + s * Nq, row, Nq * sizeof(float));
            std::memcpy(static_cast<float*>(k->data()) + s * Nk, row + Nq, Nk * sizeof(float));
            std::memcpy(static_cast<float*>(v->data()) + s * Nv, row + Nq + Nk, Nv * sizeof(float));
        }
    } else {
        q_full = ensure_cpu(ops::matmul_transB(normed, lw.attn_q()));
        k = ensure_cpu(ops::matmul_transB(normed, lw.attn_k()));
        v = ensure_cpu(ops::matmul_transB(normed, lw.attn_v()));
    }

    auto q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, q_dim},
                                      DeviceType::CPU);
    auto gate = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, q_dim},
                                         DeviceType::CPU);

    const float* qf_data = static_cast<const float*>(q_full->data());
    float* q_data = static_cast<float*>(q->data());
    float* g_data = static_cast<float*>(gate->data());

    cpu::parallel_for2(seq_len, num_heads, [&](int s, int h) {
            const float* src = qf_data + s * num_heads * head_dim * 2 + h * head_dim * 2;
            std::memcpy(q_data + s * q_dim + h * head_dim, src, head_dim * sizeof(float));
            std::memcpy(g_data + s * q_dim + h * head_dim, src + head_dim,
                        head_dim * sizeof(float));
        });

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
                          static_cast<const float*>(k->data()), static_cast<float*>(q_rope->data()),
                          static_cast<float*>(k_rope->data()), seq_len, num_heads, num_kv_heads,
                          head_dim, n_rot, lctx.rope_pos(), cfg.rope_theta);

    // ---- KV write (storage-aware: paged PagedKVStorage or contiguous KVCache) ----
    if (kv_memory_) {
        if (!kv_memory_->write_kv(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), seq_len, k_rope,
                                  v)) {
            throw std::runtime_error(
                "qwen35: KV write failed at layer " + std::to_string(lctx.layer_idx) +
                " (paged page pool exhausted or invalid seq — reduce --ctx-size / free VRAM)");
        }
    } else {
        if (kv_cache_.update(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), k_rope, v, seq_len) <
            0) {
            throw std::runtime_error(
                "qwen35: KV cache overflow at layer " + std::to_string(lctx.layer_idx) +
                ": write_pos=" + std::to_string(lctx.start_pos()) +
                " seq_len=" + std::to_string(seq_len) +
                " exceeds max_seq_len=" + std::to_string(kv_cache_.max_seq_len()));
        }
    }

    // ---- Attention (delegated to the shared, storage-aware executor) ----
    TensorPtr attn_out = attn_.attend(q_rope, cfg, lctx.layer_idx, seq_len, DeviceType::CPU,
                                      nullptr, lctx.seq_id(), /*causal=*/true);

    // Gated attention: output = sigmoid(gate) * attn_out
    float* attn_data = static_cast<float*>(attn_out->data());
    const float* gate_data = static_cast<const float*>(gate->data());
    for (int i = 0; i < seq_len * q_dim; ++i) {
        attn_data[i] *= 1.0f / (1.0f + std::exp(-gate_data[i]));
    }

    auto proj_out = ops::matmul_transB(attn_out, lw.attn_output());
    return proj_out;
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
    forge::cuda::launch_split_q_gate(
        static_cast<const float*>(q_full->data()), static_cast<float*>(q->data()),
        static_cast<float*>(gate->data()), seq_len, num_heads, head_dim, stream);

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
    forge::cuda::launch_rope_gqa(
        static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
        static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), num_heads,
        num_kv_heads, head_dim, seq_len, lctx.rope_pos(), cfg.rope_theta, stream);

    // ---- KV write (storage-aware: paged PagedKVStorage or contiguous KVCache) ----
    if (kv_memory_) {
        if (!kv_memory_->write_kv(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), seq_len, k_rope,
                                  v)) {
            throw std::runtime_error(
                "qwen35: KV write failed at layer " + std::to_string(lctx.layer_idx) +
                " (paged page pool exhausted or invalid seq — reduce --ctx-size / free VRAM)");
        }
    } else {
        if (kv_cache_.update(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), k_rope, v, seq_len) <
            0) {
            throw std::runtime_error(
                "qwen35: KV cache overflow at layer " + std::to_string(lctx.layer_idx) +
                ": write_pos=" + std::to_string(lctx.start_pos()) +
                " seq_len=" + std::to_string(seq_len) +
                " exceeds max_seq_len=" + std::to_string(kv_cache_.max_seq_len()));
        }
    }

    // ---- Attention (delegated to the shared, storage-aware executor) ----
    TensorPtr attn_out = attn_.attend(q_rope, cfg, lctx.layer_idx, seq_len, DeviceType::CUDA,
                                      nullptr, lctx.seq_id(), /*causal=*/true);

    // Gated attention: output = sigmoid(gate) * attn_out
    forge::cuda::launch_sigmoid_multiply(static_cast<const float*>(gate->data()),
                                         static_cast<float*>(attn_out->data()), seq_len * q_dim,
                                         stream);

    return ops::matmul_transB(attn_out, lw.attn_output());
}
#endif

}  // namespace forge
