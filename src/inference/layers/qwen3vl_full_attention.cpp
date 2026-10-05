#include "forge/inference/layers/qwen3vl_full_attention.h"

#include "forge/inference/layers/norm_executor.h"
#include "forge/inference/layers/rope_executor.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/logger.h"
#include "forge/operators.h"

namespace forge {

TensorPtr Qwen3vlFullAttention::attend(const TensorPtr& normed,
                                      const LayerExecutionContext& lctx) {
    const auto& lw = lctx.weights;
    if (!lw.wq() || !lw.wk() || !lw.wv() || !lw.wo()) {
        LOG_ERROR("Qwen3vlFullAttention: layer " + std::to_string(lctx.layer_idx) +
                  " missing required attention weights");
        return nullptr;
    }

    const auto& cfg = lctx.config;
    const int seq_len = lctx.seq_len();
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;
    const DeviceType dev = lctx.device.type;

    // QKV projection. Must go through AttentionExecutor::project_qkv — the
    // exact entry point GenericEngine::forward_layer uses — because for decode
    // (seq_len==1) it dispatches to fused per-dtype QKV kernels whose reduction
    // order differs from three separate ops::matmul_transB calls. Skipping the
    // executor here made the graph path diverge from the imperative baseline
    // at greedy top-2 edges within a few decode steps.
    bool has_qkv_bias = (lw.bq() && lw.bq()->numel() > 0);
    auto qkv_result = AttentionExecutor::project_qkv(normed, lw, has_qkv_bias, dev, seq_len);
    TensorPtr q = qkv_result.q;
    TensorPtr k = qkv_result.k;
    TensorPtr v = qkv_result.v;

    // Per-head QK-Norm (exact ordering of GenericEngine::forward_layer).
    if (cfg.use_qk_norm) {
        if (lw.attn_q_norm()) {
            q = NormExecutor::apply_qk_norm(q, lw.attn_q_norm(), num_heads, head_dim,
                                           cfg.rms_norm_eps, dev);
        }
        if (lw.attn_k_norm()) {
            k = NormExecutor::apply_qk_norm(k, lw.attn_k_norm(), num_kv_heads, head_dim,
                                           cfg.rms_norm_eps, dev);
        }
    }

    // MRoPE. For Qwen3-VL the rotary head dimension is conceptually split into
    // three sections, but the codebase applies the "single-position partial
    // rotation" (t = h = w = pos) via RopeExecutor::apply, which (for
    // n_rot < head_dim) routes to the CPU rotary path. We use the same executor
    // and device decision as GenericEngine so outputs are identical.
    auto roped = rope_executor_.apply(q, k, cfg, lctx.rope_pos(), seq_len, dev);
    TensorPtr q_rope = roped.q_rope;
    TensorPtr k_rope = roped.k_rope;

    // KV cache update (contiguous backend). k_rope may be on CPU after MRoPE;
    // KVCache::update handles the device, exactly like GenericEngine.
    if (kv_cache_.update(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), k_rope, v,
                         seq_len) < 0) {
        throw std::runtime_error(
            "qwen3vl: KV cache overflow at layer " + std::to_string(lctx.layer_idx) +
            ": write_pos=" + std::to_string(lctx.start_pos()) +
            " seq_len=" + std::to_string(seq_len) +
            " exceeds max_seq_len=" + std::to_string(kv_cache_.max_seq_len()));
    }

    // Attention. AttentionExecutor takes the layer device and uploads q_rope
    // internally if it landed on CPU after MRoPE (same as GenericEngine).
    TensorPtr attn_out = attn_.attend(q_rope, cfg, lctx.layer_idx, seq_len, dev, nullptr,
                                      lctx.seq_id(), /*causal=*/true);

    // Output projection.
    return ops::matmul_transB(attn_out, lw.wo());
}

}  // namespace forge
