#include "forge/inference/layers/deepseek_layer_executor.h"

#include "forge/inference/layers/attention_executor.h"
#include "forge/inference/layers/rope_executor.h"
#include "forge/operators.h"

namespace forge {

TensorPtr DeepSeekLayerExecutor::attend_gqa(const TensorPtr& normed,
                                            const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int layer_idx = lctx.layer_idx;
    const int seq_len = lctx.seq_len();
    const DeviceType dev = lctx.device;
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;

    auto q = ops::matmul_transB(normed, lw.wq(), lw.bq());
    auto k = ops::matmul_transB(normed, lw.wk(), lw.bk());
    auto v = ops::matmul_transB(normed, lw.wv(), lw.bv());

    auto rope = RopeExecutor::apply_standard(q, k, num_heads, num_kv_heads, head_dim, seq_len,
                                             lctx.start_pos(), cfg.rope_theta, dev);

    kv_cache_.update(layer_idx, lctx.seq_id(), lctx.start_pos(), rope.k_rope, v, seq_len);

    if (kv_cache_.kv_dtype() == KVCacheDType::Q4_0) {
        kv_cache_.dequantize_layer(layer_idx);
    }

    int total_len = kv_cache_.filled(layer_idx);

    TensorPtr k_sliced = kv_cache_.get_key_filled(layer_idx);
    TensorPtr v_sliced = kv_cache_.get_value_filled(layer_idx);

    if (dev == DeviceType::CUDA && k_sliced->device() == DeviceType::CPU) {
        auto k_cuda = std::make_shared<Tensor>(DataType::FP32, k_sliced->shape(), DeviceType::CUDA);
        k_cuda->copy_from(*k_sliced);
        k_sliced = k_cuda;

        auto v_cuda = std::make_shared<Tensor>(DataType::FP32, v_sliced->shape(), DeviceType::CUDA);
        v_cuda->copy_from(*v_sliced);
        v_sliced = v_cuda;
    }

    TensorPtr k_expanded = k_sliced;
    TensorPtr v_expanded = v_sliced;
    if (num_kv_heads < num_heads) {
        k_expanded = AttentionExecutor::expand_kv_heads(k_sliced, total_len, num_heads, num_kv_heads,
                                                        head_dim, dev);
        v_expanded = AttentionExecutor::expand_kv_heads(v_sliced, total_len, num_heads, num_kv_heads,
                                                        head_dim, dev);
    }

    return ops::scaled_dot_product_attention_2d(rope.q_rope, k_expanded, v_expanded, seq_len,
                                                total_len, num_heads, head_dim, nullptr, true);
}

TensorPtr DeepSeekLayerExecutor::execute(const TensorPtr& hidden,
                                         const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;

    auto normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);

    auto attn_out = (lw.layer_type == LayerType::MLA) ? mla_.attend(normed, lctx)
                                                      : attend_gqa(normed, lctx);

    auto attn_proj = ops::matmul_transB(attn_out, lw.wo());
    auto hidden_after_attn = ops::add(hidden, attn_proj);

    auto ffn_normed = ops::rms_norm(hidden_after_attn, lw.ffn_norm(), cfg.rms_norm_eps);
    auto gate = ops::matmul_transB(ffn_normed, lw.w1());
    auto up = ops::matmul_transB(ffn_normed, lw.w3());
    auto silu_gate = ops::silu(gate);
    auto ffn_mid = ops::multiply(silu_gate, up);
    auto ffn_out = ops::matmul_transB(ffn_mid, lw.w2());

    return ops::add(hidden_after_attn, ffn_out);
}

}  // namespace forge
