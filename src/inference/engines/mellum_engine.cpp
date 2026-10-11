#include "forge/engines/mellum_engine.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cpu/common/thread_pool.h"
#include "forge/cuda_kernels.h"
#include "forge/inference/layers/mellum_moe.h"
#include "forge/inference/layers/per_head_norm.h"
#include "forge/inference/layers/yarn_rope.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace forge {

namespace {

// YaRN rope (CPU): theta_i = pos * inv_freq_i * pair_scale[d], cos/sin 乘 mscale。
// half-split 配对, 与 TransformerEngine::apply_rope_standard / CUDA rope_yarn_kernel
// 完全同构。
void apply_rope_yarn_cpu(const float* q_data, const float* k_data, float* q_out, float* k_out,
                         int seq_len, int num_heads, int num_kv_heads, int head_dim,
                         int64_t start_pos, float theta, const float* pair_scale, float mscale) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = pos * freq * pair_scale[d];
                float cos_a = std::cos(angle) * mscale;
                float sin_a = std::sin(angle) * mscale;

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

}  // namespace

// ============================================================================
// Construction / yarn tables
// ============================================================================

MellumEngine::MellumEngine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx), attn_(kv_cache_) {
    if (!init_weights())
        throw std::runtime_error("MellumEngine: failed to initialize weights");
}

bool MellumEngine::init_weights() {
    build_yarn_table();
    return weights_.init(model_.weights(), model_.config());
}

// Per-dimension YaRN 频率 scale (镜像 ggml rope_yarn + ggml_rope_yarn_corr_dims)。
// 只被全注意力层消费; SWA 层用 plain rope。
void MellumEngine::build_yarn_table() {
    const auto& cfg = model_.config();
    yarn_s_.clear();
    yarn_pair_s_.clear();
    yarn_mscale_ = 1.0f;
    yarn_pair_s_dev_ = nullptr;

    // mellum 在 rope_dimension_count 未设置时回退到 head_dim (与 xing4 不同)。
    const int dim = cfg.rope_dimension_count > 0 ? cfg.rope_dimension_count : cfg.head_dim;
    if (!build_yarn_freq_table(cfg, dim, yarn_s_, yarn_mscale_))
        return;

    // 导出 per-pair 表: half-split 配对 (d, d+half) 对应 ggml dim 对 (2d, 2d+1)。
    yarn_pair_s_ = yarn_pair_table(yarn_s_);

    LOG_INFO("mellum: yarn table built (dim " + std::to_string(dim) + ", mscale " +
             std::to_string(yarn_mscale_) + ")");
}

const float* MellumEngine::yarn_pair_s_dev_ptr() {
    return yarn_table_dev_ptr(yarn_pair_s_, yarn_pair_s_dev_);
}

// ============================================================================
// forward_layer: attn (dual-mode rope) + routed MoE
// ============================================================================

TensorPtr MellumEngine::forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;

    bool is_swa = (lctx.layer_idx < (int)cfg.swa_layers.size()) && cfg.swa_layers[lctx.layer_idx] == 1;

    // ---- Attention ----
    TensorPtr normed;
    {
        PERF_SCOPE("layer/attn_norm");
        normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);
    }

    const DeviceType dev = lctx.device.type;
    TensorPtr q, k, v;
    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        PERF_SCOPE("layer/attn_proj");
        q = ops::matmul_transB(normed, lw.wq());
        k = ops::matmul_transB(normed, lw.wk());
        v = ops::matmul_transB(normed, lw.wv());
        cudaStream_t stream = 0;
        if (lw.attn_q_norm()) {
            auto q_norm_w = restore_device(lw.attn_q_norm(), DeviceType::CUDA);
            cuda::launch_rms_norm(static_cast<const float*>(q->data()),
                                  static_cast<const float*>(q_norm_w->data()),
                                  static_cast<float*>(q->data()), seq_len * num_heads, head_dim,
                                  cfg.rms_norm_eps, stream);
        }
        if (lw.attn_k_norm()) {
            auto k_norm_w = restore_device(lw.attn_k_norm(), DeviceType::CUDA);
            cuda::launch_rms_norm(static_cast<const float*>(k->data()),
                                  static_cast<const float*>(k_norm_w->data()),
                                  static_cast<float*>(k->data()), seq_len * num_kv_heads,
                                  head_dim, cfg.rms_norm_eps, stream);
        }
#endif
    } else {
        PERF_SCOPE("layer/attn_proj");
        q = ensure_cpu(ops::matmul_transB(normed, lw.wq()));
        k = ensure_cpu(ops::matmul_transB(normed, lw.wk()));
        v = ensure_cpu(ops::matmul_transB(normed, lw.wv()));
        if (lw.attn_q_norm())
            per_head_rms_norm_cpu(static_cast<float*>(q->data()),
                                  load_head_norm_weight(lw.attn_q_norm(), head_dim), seq_len,
                                  num_heads, head_dim, cfg.rms_norm_eps);
        if (lw.attn_k_norm())
            per_head_rms_norm_cpu(static_cast<float*>(k->data()),
                                  load_head_norm_weight(lw.attn_k_norm(), head_dim), seq_len,
                                  num_kv_heads, head_dim, cfg.rms_norm_eps);
    }

    // ---- RoPE: SWA 层 plain (freq_base_swa) / 全注意力层 yarn (freq_base) ----
    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), dev);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), dev);
    if (is_swa || !cfg.rope_yarn_enabled) {
        const float theta = is_swa ? cfg.rope_theta_swa : cfg.rope_theta;
        if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
            cuda::launch_rope_gqa(static_cast<const float*>(q->data()),
                                  static_cast<const float*>(k->data()),
                                  static_cast<float*>(q_rope->data()),
                                  static_cast<float*>(k_rope->data()), num_heads, num_kv_heads,
                                  head_dim, seq_len, lctx.rope_pos(), theta);
#endif
        } else {
            apply_rope_standard(static_cast<const float*>(q->data()),
                                static_cast<const float*>(k->data()),
                                static_cast<float*>(q_rope->data()),
                                static_cast<float*>(k_rope->data()), seq_len, num_heads,
                                num_kv_heads, head_dim, lctx.rope_pos(), theta);
        }
    } else {
        if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
            cuda::launch_rope_gqa_yarn(
                static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                num_heads, num_kv_heads, head_dim, seq_len, lctx.rope_pos(), cfg.rope_theta,
                yarn_pair_s_dev_ptr(), yarn_mscale_);
#endif
        } else {
            apply_rope_yarn_cpu(static_cast<const float*>(q->data()),
                                static_cast<const float*>(k->data()),
                                static_cast<float*>(q_rope->data()),
                                static_cast<float*>(k_rope->data()), seq_len, num_heads,
                                num_kv_heads, head_dim, lctx.rope_pos(), cfg.rope_theta,
                                yarn_pair_s_.data(), yarn_mscale_);
        }
    }

    // ---- KV write (storage-aware: paged PagedKVStorage or contiguous KVCache) ----
    if (kv_memory_) {
        if (!kv_memory_->write_kv(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), seq_len, k_rope,
                                  v)) {
            throw std::runtime_error("mellum: KV write failed at layer " +
                                     std::to_string(lctx.layer_idx) +
                                     " (paged page pool exhausted or invalid seq)");
        }
    } else {
        if (kv_cache_.update(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), k_rope, v,
                             seq_len) < 0) {
            throw std::runtime_error("mellum: KV cache overflow at layer " +
                                     std::to_string(lctx.layer_idx) + ": write_pos=" +
                                     std::to_string(lctx.start_pos()) + " seq_len=" +
                                     std::to_string(seq_len));
        }
    }

    // ---- Attention (shared, storage-aware executor; scale 1/sqrt(head_dim) 内置) ----
    // Bind the paged storage (if any) so attend reads K/V from kv_memory_ when
    // paged; mirrors Qwen35FullAttention::attend.
    if (kv_memory_) {
        attn_.set_storage(kv_memory_->is_paged() ? &kv_memory_->storage() : nullptr);
    }
    PERF_SCOPE("layer/attention");
    TensorPtr attn_out =
        attn_.attend(q_rope, cfg, lctx.layer_idx, seq_len, dev, nullptr, lctx.seq_id(),
                     /*causal=*/true);

    // ---- wo 投影 + 残差 ----
    {
        PERF_SCOPE("layer/attn_proj");
        attn_out = ops::matmul_transB(attn_out, lw.wo());
    }
    auto attn_residual = ops::add(hidden, attn_out);

    // ---- 路由-only SiLU MoE + 残差 ----
    TensorPtr moe_out = MellumMoe::apply(attn_residual, lctx, this);
    return ops::add(attn_residual, moe_out);
}

}  // namespace forge
