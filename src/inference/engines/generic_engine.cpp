#include "forge/engines/generic_engine.h"

#include <cmath>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/model.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"
#include "forge/quant_traits.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {

GenericEngine::GenericEngine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx),
      attention_executor_(kv_cache_, kv_memory_ ? &kv_memory_->storage() : nullptr) {
    if (!init_weights()) {
        throw std::runtime_error("GenericEngine: failed to initialize weights");
    }
}

bool GenericEngine::init_weights() {
    if (!weights_.init(model_.weights(), model_.config())) {
        return false;
    }
    // Load proportional RoPE frequency factors if present
    const auto& cfg = model_.config();
    if (cfg.rope_type == RopeType::Proportional) {
        auto rope_freqs = model_.weights().get("rope_freqs");
        if (!rope_freqs) {
            for (int i = 0; i < cfg.num_layers; ++i) {
                rope_freqs = weights_.layers[i].rope_freqs();
                if (rope_freqs) break;
            }
        }
        if (rope_freqs && is_quantized_type(rope_freqs->dtype())) {
            rope_freqs = ops::dequantize_weight(rope_freqs);
        }
        rope_executor_.set_rope_freqs(rope_freqs);
    }
    return true;
}

// ============================================================================
// forward_request() — override to handle embedding scaling and logit softcapping
// ============================================================================

TensorPtr GenericEngine::forward_request(const ForwardRequest& req) {
    const auto& cfg = model_.config();

    // Decode path: use fewer threads (memory-bandwidth bound)
#ifdef _OPENMP
    omp_set_num_threads(ctx_.params().n_threads);
#endif

    init_kv_cache(cfg);

    DeviceType first_dev = layer_device(0);
    auto ids_on_dev = transfer_hidden(req.input_ids, first_dev);

    auto token_emb = model_.weights().get("token_embedding");
    TensorPtr hidden;
    {
        PERF_SCOPE("forward/embedding");
        hidden = ops::embedding(token_emb, ids_on_dev, weights_.token_embedding_fp32);
    }

    // Embedding scaling (Gemma/Gemma2/Gemma4: hidden *= sqrt(n_embd))
    if (plan_.use_embedding_scale) {
        PERF_SCOPE("forward/emb_scale");
        float scale = std::sqrt(static_cast<float>(cfg.hidden_dim));
        bool was_cuda = (hidden->device() == DeviceType::CUDA);
        if (was_cuda) {
            auto hidden_cpu =
                std::make_shared<Tensor>(DataType::FP32, hidden->shape(), DeviceType::CPU);
            hidden_cpu->copy_from(*hidden);
            hidden = hidden_cpu;
        }
        int n = static_cast<int>(hidden->numel());
        float* data = static_cast<float*>(hidden->data());
        for (int i = 0; i < n; ++i) {
            data[i] *= scale;
        }
        if (was_cuda) {
            auto hidden_back =
                std::make_shared<Tensor>(DataType::FP32, hidden->shape(), DeviceType::CUDA);
            hidden_back->copy_from(*hidden);
            hidden = hidden_back;
        }
    }

    auto logits = forward_layers(hidden, req);

    // Logit softcapping (Gemma2/Gemma4)
    if (cfg.f_final_logit_softcapping > 0.0f && logits) {
        PERF_SCOPE("forward/logit_softcap");
        DeviceType logits_dev = logits->device();
        if (logits_dev == DeviceType::CUDA) {
            auto logits_cpu =
                std::make_shared<Tensor>(DataType::FP32, logits->shape(), DeviceType::CPU);
            logits_cpu->copy_from(*logits);
            logits = logits_cpu;
        }
        float cap_val = cfg.f_final_logit_softcapping;
        int n = static_cast<int>(logits->numel());
        float* data = static_cast<float*>(logits->data());
        for (int i = 0; i < n; ++i) {
            data[i] = std::tanh(data[i] / cap_val) * cap_val;
        }
        if (logits_dev == DeviceType::CUDA) {
            auto logits_back =
                std::make_shared<Tensor>(DataType::FP32, logits->shape(), DeviceType::CUDA);
            logits_back->copy_from(*logits);
            logits = logits_back;
        }
    }

    return logits;
}

// ============================================================================
// forward_layer — orchestration only; operators live in inference/layers/*
// ============================================================================

TensorPtr GenericEngine::forward_layer(const TensorPtr& hidden,
                                       const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const int layer_idx = lctx.layer_idx;
    const int seq_len = lctx.seq_len();
    const int64_t start_pos = lctx.start_pos();
    const DeviceType dev = lctx.device.type;
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;
    const auto& lw = lctx.weights;

    // Determine norm type and epsilon
    NormType norm_type = cfg.norm_type;
    float eps = (norm_type == NormType::LayerNorm) ? cfg.layer_norm_eps : cfg.rms_norm_eps;

    // Determine if QKV has bias
    bool has_qkv_bias = (lw.bq() && lw.bq()->numel() > 0);

    // ---- 1. Pre-attention norm ----
    TensorPtr pre_attn_norm;
    {
        PERF_SCOPE("layer/attn_norm");
        auto norm_w = lw.attn_norm();
        auto norm_b = (norm_type == NormType::LayerNorm) ? lw.get("attn_norm_bias") : nullptr;
        pre_attn_norm = NormExecutor::apply(hidden, norm_w, norm_b, norm_type, eps);
    }

    // ---- 2. QKV projection ----
    TensorPtr q, k, v;
    {
        PERF_SCOPE("layer/qkv_proj");
        auto qkv_result =
            AttentionExecutor::project_qkv(pre_attn_norm, lw, has_qkv_bias, dev, seq_len);
        q = qkv_result.q;
        k = qkv_result.k;
        v = qkv_result.v;
    }

    // ---- 3. QK-Norm (per-head RMSNorm) ----
    if (cfg.use_qk_norm) {
        PERF_SCOPE("layer/qk_norm");
        if (lw.attn_q_norm()) {
            q = NormExecutor::apply_qk_norm(q, lw.attn_q_norm(), num_heads, head_dim,
                                            cfg.rms_norm_eps, dev);
        }
        if (lw.attn_k_norm()) {
            k = NormExecutor::apply_qk_norm(k, lw.attn_k_norm(), num_kv_heads, head_dim,
                                            cfg.rms_norm_eps, dev);
        }
    }

    // ---- 4. RoPE ----
    TensorPtr q_rope, k_rope;
    {
        PERF_SCOPE("layer/rope");
        auto rope_result = rope_executor_.apply(q, k, cfg, start_pos, seq_len, dev);
        q_rope = rope_result.q_rope;
        k_rope = rope_result.k_rope;
    }

    // ---- 5. KV cache update ----
    {
        PERF_SCOPE("layer/kv_cache_update");

        bool paged = kv_memory_ && kv_memory_->is_paged();

        if (paged) {
            // Paged mode: write K/V to PagedKVStorage.
            // CUDA-backed paged storage takes device pointers directly (no D2H);
            // CPU-backed paged storage needs the data on CPU.
            DeviceType storage_dev = kv_memory_->storage().device();
            const float* k_data;
            const float* v_data;
            TensorPtr k_cpu, v_cpu;  // keep alive for CPU path
            if (storage_dev == DeviceType::CUDA) {
                k_data = static_cast<const float*>(k_rope->data());
                v_data = static_cast<const float*>(v->data());
            } else {
                k_cpu = k_rope;
                v_cpu = v;
                if (k_cpu->device() == DeviceType::CUDA) {
                    k_cpu = std::make_shared<Tensor>(k_cpu->dtype(), k_cpu->shape(), DeviceType::CPU);
                    k_cpu->copy_from(*k_rope);
                }
                if (v_cpu->device() == DeviceType::CUDA) {
                    v_cpu = std::make_shared<Tensor>(v_cpu->dtype(), v_cpu->shape(), DeviceType::CPU);
                    v_cpu->copy_from(*v);
                }
                k_data = static_cast<const float*>(k_cpu->data());
                v_data = static_cast<const float*>(v_cpu->data());
            }
            kv_memory_->storage().write_kv_seq(layer_idx, lctx.seq_id(), start_pos,
                                               seq_len, k_data, v_data);
        } else {
            // Contiguous mode: write to KVCache directly
            kv_cache_.update(layer_idx, lctx.seq_id(), start_pos, k_rope, v, seq_len);

            // Fused decode path: skip dequantize_layer() when the fused attention
            // kernel will read the quantized KV cache directly (Q4_0/F16/Q8_0 on CUDA, seq_len==1).
            const auto& kv_cfg = kv_cache_.kv_config();
            bool use_fused_decode =
                (dev == DeviceType::CUDA && seq_len == 1 &&
                 kv_cache_.d_q_key_cache(layer_idx) != nullptr &&
                 ((kv_cfg.type_k == KVCacheDType::Q4_0 && kv_cfg.type_v == KVCacheDType::Q4_0) ||
                  (kv_cfg.type_k == KVCacheDType::F16 && kv_cfg.type_v == KVCacheDType::F16) ||
                  (kv_cfg.type_k == KVCacheDType::Q8_0 && kv_cfg.type_v == KVCacheDType::Q8_0)));

            // For non-fused paths with quantized KV, dequantize the layer into FP32 shadow cache.
            if (!use_fused_decode && kv_cache_.kv_dtype() != KVCacheDType::FP32) {
                kv_cache_.dequantize_layer(layer_idx);
            }
        }
    }

    // ---- 6. Attention ----
    TensorPtr attn_out;
    {
        PERF_SCOPE("layer/attention");
        attn_out = attention_executor_.attend(q_rope, cfg, layer_idx, seq_len, dev,
                                              nullptr, lctx.seq_id());
    }

    // ---- 7. Attention output projection (optionally fused with residual) ----
    TensorPtr attn_proj;
    bool fused_attn_residual = false;
    {
        PERF_SCOPE("layer/attn_proj");
        if (dev == DeviceType::CPU && seq_len == 1 && lw.wo()->dtype() == DataType::Q4_0) {
            attn_proj = ops::matmul_transB_fused_attn_proj_residual_q4_0(attn_out, lw.wo(), hidden);
            fused_attn_residual = true;
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.wo()->dtype() == DataType::Q4_K) {
            attn_proj = ops::matmul_transB_fused_attn_proj_residual_q4_k(attn_out, lw.wo(), hidden);
            fused_attn_residual = true;
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.wo()->dtype() == DataType::Q5_K) {
            attn_proj = ops::matmul_transB_fused_attn_proj_residual_q5_k(attn_out, lw.wo(), hidden);
            fused_attn_residual = true;
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.wo()->dtype() == DataType::Q6_K) {
            attn_proj = ops::matmul_transB_fused_attn_proj_residual_q6_k(attn_out, lw.wo(), hidden);
            fused_attn_residual = true;
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.wo()->dtype() == DataType::Q2_K) {
            attn_proj = ops::matmul_transB_fused_attn_proj_residual_q2_k(attn_out, lw.wo(), hidden);
            fused_attn_residual = true;
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.wo()->dtype() == DataType::Q3_K) {
            attn_proj = ops::matmul_transB_fused_attn_proj_residual_q3_k(attn_out, lw.wo(), hidden);
            fused_attn_residual = true;
        } else if (dev == DeviceType::CUDA && seq_len == 1 && lw.wo()->dtype() == DataType::Q5_K) {
#ifdef USE_CUDA
            int K_wo = static_cast<int>(lw.wo()->shape()[1]);
            int N_wo = static_cast<int>(lw.wo()->shape()[0]);
            attn_proj = std::make_shared<Tensor>(DataType::FP32,
                                                 std::vector<int64_t>{1, N_wo},
                                                 DeviceType::CUDA);
            cuda::launch_attn_proj_q5_k_cooperative(
                static_cast<const float*>(attn_out->data()), lw.wo()->data(),
                static_cast<float*>(attn_proj->data()), K_wo, N_wo);
#endif
        } else {
            attn_proj = ops::matmul_transB(attn_out, lw.wo());
        }
    }

    // ---- 8. Post-attention norm (Gemma2) ----
    if (cfg.has_post_attention_norm) {
        PERF_SCOPE("layer/post_attn_norm");
        auto post_attn_norm = lw.attn_post_norm();
        if (!post_attn_norm) post_attn_norm = lw.post_attention_norm();
        if (post_attn_norm) {
            attn_proj = NormExecutor::apply(attn_proj, post_attn_norm, nullptr, NormType::RMSNorm,
                                            cfg.rms_norm_eps);
        }
    }

    // ---- 9. Compute output based on residual style ----
    TensorPtr output;

    if (cfg.use_parallel_residual) {
        // Falcon-style: ffn from the same norm, parallel add
        // Determine FFN input: use attn_norm_2 if present (Falcon-40B), else pre_attn_norm
        TensorPtr ffn_in = pre_attn_norm;
        auto attn_norm_2_w = lw.get("attn_norm_2");
        if (attn_norm_2_w) {
            PERF_SCOPE("layer/attn_norm_2");
            auto attn_norm_2_b = lw.get("attn_norm_2_bias");
            ffn_in = NormExecutor::apply(pre_attn_norm, attn_norm_2_w, attn_norm_2_b,
                                         NormType::LayerNorm, cfg.layer_norm_eps);
        }

        TensorPtr ffn_out;
        {
            PERF_SCOPE("layer/ffn");
            ffn_out = FfnExecutor::apply(ffn_in, hidden, cfg, lw, seq_len, dev);
        }

        // Parallel residual: attn_proj + ffn_out + hidden
        // If attn_proj already has hidden fused in, skip second add.
        {
            PERF_SCOPE("layer/residual");
            output = ops::add(attn_proj, ffn_out);
            if (!fused_attn_residual)
                output = ops::add(output, hidden);
        }
    } else {
        // Sequential: norm→attn→residual→norm→ffn→residual
        auto hidden_after_attn =
            fused_attn_residual ? attn_proj : ops::add(hidden, attn_proj);

        TensorPtr ffn_normed;
        {
            PERF_SCOPE("layer/ffn_norm");
            ffn_normed =
                NormExecutor::apply(hidden_after_attn, lw.ffn_norm(), nullptr, norm_type, eps);
        }

        TensorPtr ffn_out;
        {
            PERF_SCOPE("layer/ffn");
            ffn_out = FfnExecutor::apply(ffn_normed, hidden_after_attn, cfg, lw, seq_len, dev);
        }

        // Post-FFN norm (Gemma2)
        if (cfg.has_post_ffn_norm) {
            PERF_SCOPE("layer/post_ffn_norm");
            auto post_ffn_norm = lw.post_ffn_norm();
            if (post_ffn_norm) {
                ffn_out = NormExecutor::apply(ffn_out, post_ffn_norm, nullptr, NormType::RMSNorm,
                                              cfg.rms_norm_eps);
            }
        }

        // Only skip residual add if a fused SiLUGated kernel already added it
        if (FfnExecutor::residual_fused(cfg, lw, seq_len, dev)) {
            output = ffn_out;  // residual already added in fused kernel
        } else {
            output = ops::add(hidden_after_attn, ffn_out);
        }
    }

    return output;
}

// ============================================================================
// init_kv_cache override (standard — same as base class)
// ============================================================================

void GenericEngine::init_kv_cache(const ModelConfig& cfg) {
    // Use the base class implementation (no per-layer dims needed for
    // Llama/Gemma/Gemma2/Falcon architectures)
    TransformerEngine::init_kv_cache(cfg);
}

}  // namespace forge
