#include "forge/engines/gemma_engine.h"
#include "forge/operators.h"
#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace forge {

GemmaEngine::GemmaEngine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx) {
    is_gemma2_ = (model.config().arch_type == "gemma2");
    if (!init_weights()) {
        throw std::runtime_error("GemmaEngine: failed to initialize weights");
    }
}

bool GemmaEngine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

TensorPtr GemmaEngine::forward(const TensorPtr& input_ids, int64_t start_pos) {
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

    // Gemma embedding scaling: hidden *= sqrt(n_embd)
    {
        PERF_SCOPE("forward/emb_scale");
        float scale = std::sqrt(static_cast<float>(cfg.hidden_dim));
        int n = static_cast<int>(hidden->numel());
        float* data = static_cast<float*>(hidden->data());
        for (int i = 0; i < n; ++i) {
            data[i] *= scale;
        }
    }

    auto logits = forward_layers(hidden, seq_len, start_pos);

    // Gemma2 final logit softcapping
    if (is_gemma2_ && cfg.f_final_logit_softcapping > 0.0f && logits) {
        PERF_SCOPE("forward/logit_softcap");
        float cap = cfg.f_final_logit_softcapping;
        logits = ops::tanh_act(logits);
        int n = static_cast<int>(logits->numel());
        float* data = static_cast<float*>(logits->data());
        for (int i = 0; i < n; ++i) {
            data[i] *= cap;
        }
    }

    return logits;
}

TensorPtr GemmaEngine::forward_layer(const TensorPtr& hidden,
                                      int layer_idx,
                                      int seq_len, int64_t start_pos,
                                      DeviceType dev) {
    const auto& cfg = model_.config();
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;
    const auto& lw = weights_.layers[layer_idx];

    // ---- Pre-attention RMSNorm ----
    TensorPtr normed;
    {
        PERF_SCOPE("layer/attn_norm");
        normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);
    }

    // ---- QKV Projection ----
    TensorPtr q, k, v;
    {
        PERF_SCOPE("layer/qkv_proj");
        q = ops::matmul_transB(normed, lw.wq());
        k = ops::matmul_transB(normed, lw.wk());
        v = ops::matmul_transB(normed, lw.wv());
    }

    // ---- RoPE (NeoX style) + Q-scaling ----
    float rcp = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), dev);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), dev);
    {
        PERF_SCOPE("layer/rope");
        ensure_rope_freqs(head_dim, cfg.rope_theta);
        const float* q_data = static_cast<const float*>(q->data());
        const float* k_data = static_cast<const float*>(k->data());
        float* qo = static_cast<float*>(q_rope->data());
        float* ko = static_cast<float*>(k_rope->data());

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

                    float q0 = q_data[q_idx0];
                    float q1 = q_data[q_idx1];
                    qo[q_idx0] = (q0 * cos_a - q1 * sin_a) * rcp;
                    qo[q_idx1] = (q0 * sin_a + q1 * cos_a) * rcp;

                    if (h < num_kv_heads) {
                        int k_idx0 = s * k_stride + h * head_dim + d;
                        int k_idx1 = k_idx0 + half_dim;

                        ko[k_idx0] = k_data[k_idx0] * cos_a - k_data[k_idx1] * sin_a;
                        ko[k_idx1] = k_data[k_idx0] * sin_a + k_data[k_idx1] * cos_a;
                    }
                }
            }
        }
    }

    // ---- KV Cache update ----
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
            attn_out = ops::scaled_dot_product_attention_2d_gqa(
                q_rope, k_sliced, v_sliced, seq_len, total_len,
                num_heads, num_kv_heads, head_dim, true);
        } else {
            attn_out = ops::scaled_dot_product_attention_2d(
                q_rope, k_sliced, v_sliced, seq_len, total_len,
                num_heads, head_dim, true);
        }
    }

    // ---- Attention output projection ----
    {
        PERF_SCOPE("layer/attn_proj");
        attn_out = ops::matmul_transB(attn_out, lw.wo());
    }

    // ---- Post-attention norm (Gemma2 only) ----
    auto post_attn_norm = lw.get("post_attention_norm");
    if (post_attn_norm) {
        PERF_SCOPE("layer/post_attn_norm");
        attn_out = ops::rms_norm(attn_out, post_attn_norm, cfg.rms_norm_eps);
    }

    // ---- First residual add ----
    auto hidden_after_attn = ops::add(hidden, attn_out);

    // ---- Pre-FFN RMSNorm ----
    TensorPtr ffn_normed;
    {
        PERF_SCOPE("layer/ffn_norm");
        ffn_normed = ops::rms_norm(hidden_after_attn, lw.ffn_norm(), cfg.rms_norm_eps);
    }

    // ---- GeGLU FFN (gate + up with GELU gate) ----
    TensorPtr ffn_out;
    {
        PERF_SCOPE("layer/ffn");
        auto gate = ops::matmul_transB(ffn_normed, lw.w1());
        auto up = ops::matmul_transB(ffn_normed, lw.w3());
        auto gated = ops::gelu_multiply(gate, up);
        ffn_out = ops::matmul_transB(gated, lw.w2());
    }

    // ---- Post-FFN norm (Gemma2 only) ----
    auto post_ffn_norm = lw.get("post_ffn_norm");
    if (post_ffn_norm) {
        PERF_SCOPE("layer/post_ffn_norm");
        ffn_out = ops::rms_norm(ffn_out, post_ffn_norm, cfg.rms_norm_eps);
    }

    // ---- Second residual add ----
    auto output = ops::add(hidden_after_attn, ffn_out);

    return output;
}

// Engine registrations
namespace {
EngineAutoRegister _reg_gemma("gemma", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<GemmaEngine>(model, ctx);
});
EngineAutoRegister _reg_gemma2("gemma2", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<GemmaEngine>(model, ctx);
});
} // anonymous namespace

} // namespace forge