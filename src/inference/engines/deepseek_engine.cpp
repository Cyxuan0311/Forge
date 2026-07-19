#include "forge/engines/deepseek_engine.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/operators.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

DeepSeekEngine::DeepSeekEngine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx) {
    if (!init_weights()) {
        throw std::runtime_error("DeepSeekEngine: failed to initialize weights");
    }
}

bool DeepSeekEngine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

TensorPtr DeepSeekEngine::forward_layer(const TensorPtr& hidden, int layer_idx, int seq_len,
                                        int64_t start_pos, DeviceType dev) {
    const auto& lw = weights_.layers[layer_idx];
    if (lw.layer_type == LayerType::MLA) {
        return forward_layer_mla(hidden, layer_idx, seq_len, start_pos, dev);
    } else {
        return forward_layer_gqa(hidden, layer_idx, seq_len, start_pos, dev);
    }
}

TensorPtr DeepSeekEngine::forward_layer_gqa(const TensorPtr& hidden, int layer_idx, int seq_len,
                                            int64_t start_pos, DeviceType dev) {
    const auto& cfg = model_.config();
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;
    const auto& lw = weights_.layers[layer_idx];

    auto normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);

    auto q = ops::matmul_transB(normed, lw.wq(), lw.bq());
    auto k = ops::matmul_transB(normed, lw.wk(), lw.bk());
    auto v = ops::matmul_transB(normed, lw.wv(), lw.bv());

    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), dev);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), dev);

    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_rope_gqa(
            static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
            static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), num_heads,
            num_kv_heads, head_dim, seq_len, start_pos, cfg.rope_theta);
#endif
    } else {
        apply_rope_standard(
            static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
            static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), seq_len,
            num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta);
    }

    kv_cache_.update(layer_idx, /*seq_id=*/0, start_pos, k_rope, v, seq_len);

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

    TensorPtr k_expanded, v_expanded;
    if (num_kv_heads < num_heads) {
        k_expanded = expand_kv_heads(k_sliced, total_len, num_heads, num_kv_heads, head_dim, dev);
        v_expanded = expand_kv_heads(v_sliced, total_len, num_heads, num_kv_heads, head_dim, dev);
    } else {
        k_expanded = k_sliced;
        v_expanded = v_sliced;
    }

    auto attn_out = ops::scaled_dot_product_attention_2d(q_rope, k_expanded, v_expanded, seq_len,
                                                         total_len, num_heads, head_dim, true);

    auto attn_proj = ops::matmul_transB(attn_out, lw.wo());

    auto hidden_after_attn = ops::add(hidden, attn_proj);

    auto ffn_normed = ops::rms_norm(hidden_after_attn, lw.ffn_norm(), cfg.rms_norm_eps);
    auto gate = ops::matmul_transB(ffn_normed, lw.w1());
    auto up = ops::matmul_transB(ffn_normed, lw.w3());
    auto silu_gate = ops::silu(gate);
    auto ffn_mid = ops::multiply(silu_gate, up);

    auto ffn_out = ops::matmul_transB(ffn_mid, lw.w2());

    auto output = ops::add(hidden_after_attn, ffn_out);

    return output;
}

TensorPtr DeepSeekEngine::forward_layer_mla(const TensorPtr& hidden, int layer_idx, int seq_len,
                                            int64_t start_pos, DeviceType dev) {
    const auto& cfg = model_.config();
    int num_heads = cfg.num_heads;
    int head_dim = cfg.head_dim;
    int kv_lora_rank = cfg.kv_lora_rank;
    const auto& lw = weights_.layers[layer_idx];

    auto normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);

    TensorPtr q;
    if (lw.wq_a() && lw.wq_b()) {
        auto q_latent = ops::matmul_transB(normed, lw.wq_a());
        q = ops::matmul_transB(q_latent, lw.wq_b());
    } else {
        q = ops::matmul_transB(normed, lw.wq_a() ? lw.wq_a() : lw.wq_b());
    }

    auto compressed_kv = ops::matmul_transB(normed, lw.kv_a_proj());

    auto k_latent = compressed_kv;
    auto v_latent = ops::matmul_transB(compressed_kv, lw.kv_b_proj());

    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), dev);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k_latent->shape(), dev);

    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_rope_gqa(
            static_cast<const float*>(q->data()), static_cast<const float*>(k_latent->data()),
            static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), num_heads, 1,
            head_dim, seq_len, start_pos, cfg.rope_theta);
#endif
    } else {
        apply_rope_standard(
            static_cast<const float*>(q->data()), static_cast<const float*>(k_latent->data()),
            static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), seq_len,
            num_heads, 1, head_dim, start_pos, cfg.rope_theta);
    }

    kv_cache_.update(layer_idx, /*seq_id=*/0, start_pos, k_rope, v_latent, seq_len);

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

    auto attn_out = ops::scaled_dot_product_attention_2d(q_rope, k_sliced, v_sliced, seq_len,
                                                         total_len, num_heads, kv_lora_rank, true);

    auto attn_proj = ops::matmul_transB(attn_out, lw.wo());

    auto hidden_after_attn = ops::add(hidden, attn_proj);

    auto ffn_normed = ops::rms_norm(hidden_after_attn, lw.ffn_norm(), cfg.rms_norm_eps);
    auto gate = ops::matmul_transB(ffn_normed, lw.w1());
    auto up = ops::matmul_transB(ffn_normed, lw.w3());
    auto silu_gate = ops::silu(gate);
    auto ffn_mid = ops::multiply(silu_gate, up);

    auto ffn_out = ops::matmul_transB(ffn_mid, lw.w2());

    auto output = ops::add(hidden_after_attn, ffn_out);

    return output;
}

}  // namespace forge
