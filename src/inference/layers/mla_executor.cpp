#include "forge/inference/layers/mla_executor.h"

#include "forge/inference/layers/rope_executor.h"
#include "forge/operators.h"

namespace forge {

TensorPtr MlaExecutor::attend(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int layer_idx = lctx.layer_idx;
    const int seq_len = lctx.seq_len();
    const int64_t start_pos = lctx.start_pos();
    const DeviceType dev = lctx.device;
    const int num_heads = cfg.num_heads;
    const int head_dim = cfg.head_dim;
    const int kv_lora_rank = cfg.kv_lora_rank;

    // Q: 可选的 low-rank 分解 (wq_a -> wq_b)。
    TensorPtr q;
    if (lw.wq_a() && lw.wq_b()) {
        auto q_latent = ops::matmul_transB(normed, lw.wq_a());
        q = ops::matmul_transB(q_latent, lw.wq_b());
    } else {
        q = ops::matmul_transB(normed, lw.wq_a() ? lw.wq_a() : lw.wq_b());
    }

    // KV 压缩: latent K 直接复用 compressed_kv, latent V 再过一次 kv_b_proj。
    auto compressed_kv = ops::matmul_transB(normed, lw.kv_a_proj());
    auto k_latent = compressed_kv;
    auto v_latent = ops::matmul_transB(compressed_kv, lw.kv_b_proj());

    // MLA 的 K 只有一个 latent head。
    auto rope = RopeExecutor::apply_standard(q, k_latent, num_heads, /*num_kv_heads=*/1, head_dim,
                                             seq_len, start_pos, cfg.rope_theta, dev);

    kv_cache_.update(layer_idx, lctx.seq_id(), start_pos, rope.k_rope, v_latent, seq_len);

    if (kv_cache_.kv_dtype() == KVCacheDType::Q4_0) {
        kv_cache_.dequantize_layer(layer_idx);
    }

    int total_len = kv_cache_.filled(layer_idx);

    TensorPtr k_all = kv_cache_.get_key(layer_idx);
    TensorPtr v_all = kv_cache_.get_value(layer_idx);

    TensorPtr k_sliced, v_sliced;
    if (total_len < kv_cache_.max_seq_len()) {
        k_sliced = std::make_shared<Tensor>(k_all->slice(0, 0, total_len));
        v_sliced = std::make_shared<Tensor>(v_all->slice(0, 0, total_len));
    } else {
        k_sliced = k_all;
        v_sliced = v_all;
    }

    if (dev == DeviceType::CUDA && k_sliced->device() == DeviceType::CPU) {
        auto k_cuda = std::make_shared<Tensor>(DataType::FP32, k_sliced->shape(), DeviceType::CUDA);
        k_cuda->copy_from(*k_sliced);
        k_sliced = k_cuda;

        auto v_cuda = std::make_shared<Tensor>(DataType::FP32, v_sliced->shape(), DeviceType::CUDA);
        v_cuda->copy_from(*v_sliced);
        v_sliced = v_cuda;
    }

    // attention 在 latent 维度 (kv_lora_rank) 上进行。
    return ops::scaled_dot_product_attention_2d(rope.q_rope, k_sliced, v_sliced, seq_len, total_len,
                                                num_heads, kv_lora_rank, nullptr, true);
}

}  // namespace forge
