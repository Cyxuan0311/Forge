#include "forge/engines/falcon_engine.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

FalconEngine::FalconEngine(Model& model, InferenceContext& ctx) : TransformerEngine(model, ctx) {
    if (!init_weights()) {
        throw std::runtime_error("FalconEngine: failed to initialize weights");
    }
}

bool FalconEngine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

TensorPtr FalconEngine::forward(const TensorPtr& input_ids, int64_t start_pos) {
    const auto& cfg = model_.config();
    int seq_len = static_cast<int>(input_ids->numel());

    init_kv_cache(cfg);

    DeviceType first_dev = layer_device(0);
    auto ids_on_dev = transfer_hidden(input_ids, first_dev);

    auto token_emb = model_.weights().get("token_embedding");
    TensorPtr hidden;
    {
        PERF_SCOPE("forward/embedding");
        hidden = ops::embedding(token_emb, ids_on_dev, weights_.token_embedding_fp32);
    }

    return forward_layers(hidden, seq_len, start_pos);
}

TensorPtr FalconEngine::forward_layer(const TensorPtr& hidden, int layer_idx, int seq_len,
                                      int64_t start_pos, DeviceType dev) {
    const auto& cfg = model_.config();
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;
    const auto& lw = weights_.layers[layer_idx];

    // ---- Pre-attention LayerNorm (with bias) ----
    TensorPtr normed;
    {
        PERF_SCOPE("layer/attn_norm");
        auto norm_w = lw.attn_norm();
        auto norm_b = lw.get("attn_norm_bias");
        normed = ops::layer_norm(hidden, norm_w, norm_b, cfg.layer_norm_eps);
    }

    // Determine QKV input norm (Falcon-40B uses attn_norm_2, 7B reuses same)
    TensorPtr qkv_input = normed;
    auto attn_norm_2_w = lw.get("attn_norm_2");
    if (attn_norm_2_w) {
        auto attn_norm_2_b = lw.get("attn_norm_2_bias");
        PERF_SCOPE("layer/attn_norm_2");
        qkv_input = ops::layer_norm(normed, attn_norm_2_w, attn_norm_2_b, cfg.layer_norm_eps);
    }

    // ---- QKV Projection ----
    TensorPtr q, k, v;
    {
        PERF_SCOPE("layer/qkv_proj");
        q = ops::matmul_transB(qkv_input, lw.wq());
        k = ops::matmul_transB(qkv_input, lw.wk());
        v = ops::matmul_transB(qkv_input, lw.wv());
    }

    // ---- RoPE (NeoX, full head dim) ----
    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), dev);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), dev);
    {
        PERF_SCOPE("layer/rope");
        apply_rope_neox_cpu(
            static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
            static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), seq_len,
            num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta);
    }

    // ---- KV Cache ----
    {
        PERF_SCOPE("layer/kv_cache_update");
        kv_cache_.update(layer_idx, k_rope, v, seq_len);
    }

    int total_len = kv_cache_.filled(layer_idx);
    auto k_sliced = kv_cache_.get_key_filled(layer_idx);
    auto v_sliced = kv_cache_.get_value_filled(layer_idx);

    if (dev == DeviceType::CUDA && k_sliced->device() == DeviceType::CPU) {
        auto k_cuda = std::make_shared<Tensor>(DataType::FP32, k_sliced->shape(), DeviceType::CUDA);
        k_cuda->copy_from(*k_sliced);
        k_sliced = k_cuda;
        auto v_cuda = std::make_shared<Tensor>(DataType::FP32, v_sliced->shape(), DeviceType::CUDA);
        v_cuda->copy_from(*v_sliced);
        v_sliced = v_cuda;
    }

    // ---- Attention ----
    TensorPtr attn_out;
    {
        PERF_SCOPE("layer/attention");
        if (num_kv_heads < num_heads) {
            attn_out = ops::scaled_dot_product_attention_2d_gqa(q_rope, k_sliced, v_sliced, seq_len,
                                                                total_len, num_heads, num_kv_heads,
                                                                head_dim, true);
        } else {
            attn_out = ops::scaled_dot_product_attention_2d(q_rope, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, head_dim, true);
        }
    }

    // ---- Attention output projection ----
    TensorPtr attn_proj;
    {
        PERF_SCOPE("layer/attn_proj");
        attn_proj = ops::matmul_transB(attn_out, lw.wo());
    }

    // ---- FFN (from the SAME norm, no separate ffn_norm) ----
    // Falcon uses simple GELU FFN: down(GELU(up(x)))
    TensorPtr ffn_out;
    {
        PERF_SCOPE("layer/ffn");
        auto up = ops::matmul_transB(normed, lw.w3());
        auto activated = ops::gelu(up);
        ffn_out = ops::matmul_transB(activated, lw.w2());
    }

    // ---- Parallel residual: attn_proj + ffn_out + hidden ----
    TensorPtr output;
    {
        PERF_SCOPE("layer/residual");
        output = ops::add(attn_proj, ffn_out);
        output = ops::add(output, hidden);
    }

    return output;
}

void FalconEngine::apply_rope_neox_cpu(const float* q_data, const float* k_data, float* q_out,
                                       float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                       int head_dim, int64_t start_pos, float theta) {
    ensure_rope_freqs(head_dim, theta);
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float angle = pos * rope_freqs_[d];
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

// Engine registration
namespace {
EngineAutoRegister _reg_falcon("falcon", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<FalconEngine>(model, ctx);
});
}  // anonymous namespace

}  // namespace forge