#include "forge/engines/gemma4_engine.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

namespace forge {

// Helper: ensure tensor data is accessible on CPU for scalar operations.
// If the tensor is on CUDA, copies to CPU and returns the CPU version.
static TensorPtr ensure_cpu(const TensorPtr& t) {
    if (!t || t->device() == DeviceType::CPU) return t;
    auto cpu = std::make_shared<Tensor>(t->dtype(), t->shape(), DeviceType::CPU);
    cpu->copy_from(*t);
    return cpu;
}

// Helper: transfer a tensor back to the original device after CPU operations.
static TensorPtr restore_device(const TensorPtr& t, DeviceType target) {
    if (!t || t->device() == target) return t;
    auto on_dev = std::make_shared<Tensor>(t->dtype(), t->shape(), target);
    on_dev->copy_from(*t);
    return on_dev;
}

Gemma4Engine::Gemma4Engine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx) {
    if (!init_weights()) {
        throw std::runtime_error("Gemma4Engine: failed to initialize weights");
    }
}

bool Gemma4Engine::init_weights() {
    if (!weights_.init(model_.weights(), model_.config())) {
        return false;
    }
    // Load proportional RoPE frequency factors for full-attention layers
    // The GGUF stores rope_freqs per full-attention layer (e.g., blk.4.rope_freqs.weight),
    // NOT as a global tensor. Layer 0 is SWA and has no rope_freqs, so we must search
    // for the first full-attention layer that has it.
    rope_freqs_ = model_.weights().get("rope_freqs");
    if (!rope_freqs_) {
        const auto& cfg = model_.config();
        for (int i = 0; i < cfg.num_layers; ++i) {
            bool is_swa = (i < (int)cfg.swa_layers.size() && cfg.swa_layers[i] == 1);
            if (!is_swa) {
                // This is a full-attention layer — check if it has rope_freqs
                rope_freqs_ = weights_.layers[i].get("rope_freqs");
                if (rope_freqs_) break;
            }
        }
    }
    if (rope_freqs_) {
        // Dequantize if needed
        if (is_quantized_type(rope_freqs_->dtype())) {
            rope_freqs_ = ops::dequantize_weight(rope_freqs_);
        }
        LOG_INFO("rope_freqs loaded: shape=" + std::to_string(rope_freqs_->shape()[0]));
    } else {
        LOG_WARN("rope_freqs NOT found in weights for Gemma4");
    }

    return true;
}

void Gemma4Engine::init_kv_cache(const ModelConfig& cfg) {
    if (kv_cache_initialized_)
        return;

    int kv_max_seq = cfg.max_seq_len;
    const int KV_MAX_SEQ_CAP = 4096;
    if (kv_max_seq > KV_MAX_SEQ_CAP) {
        LOG_INFO("Capping KV cache max_seq_len from " + std::to_string(kv_max_seq) + " to " +
                 std::to_string(KV_MAX_SEQ_CAP) + " to avoid OOM");
        kv_max_seq = KV_MAX_SEQ_CAP;
    }

    DeviceType kv_dev = (gpu_layers_ >= cfg.num_layers) ? DeviceType::CUDA : DeviceType::CPU;

    // Gemma4: per-layer KV cache with different head dimensions
    // Full-attention layers use head_dim, SWA layers use head_dim_swa
    // SWA layers without own KV reuse the nearest full-attention layer's KV
    std::vector<int> kv_dims(cfg.num_layers, 0);
    for (int i = 0; i < cfg.num_layers; ++i) {
        if (i < cfg.n_layer_kv_from_start) {
            bool is_swa = (i < (int)cfg.swa_layers.size() && cfg.swa_layers[i] == 1);
            int hd = is_swa ? cfg.head_dim_swa : cfg.head_dim;
            int n_kv = is_swa ? cfg.num_kv_heads_swa : cfg.num_kv_heads;
            kv_dims[i] = n_kv * hd;
        }
        // else: no own KV, will reuse from another layer; kv_dims[i] stays 0
    }

    LOG_INFO("KV cache init (Gemma4 per-layer): layers=" + std::to_string(cfg.num_layers) +
             ", max_seq_len=" + std::to_string(kv_max_seq) +
             ", dev=" + (kv_dev == DeviceType::CUDA ? "CUDA" : "CPU") +
             ", head_dim=" + std::to_string(cfg.head_dim) +
             ", head_dim_swa=" + std::to_string(cfg.head_dim_swa));

    kv_cache_.init_per_layer(cfg.num_layers, kv_dims, kv_max_seq, kv_dev);
    kv_cache_initialized_ = true;

    LOG_INFO("KV cache initialized successfully, actual size: " +
             std::to_string(kv_cache_.nbytes() / (1024 * 1024)) + " MB");
}

TensorPtr Gemma4Engine::forward(const TensorPtr& input_ids, int64_t start_pos) {
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
        hidden = ensure_cpu(hidden);
        int n = static_cast<int>(hidden->numel());
        float* data = static_cast<float*>(hidden->data());
        for (int i = 0; i < n; ++i) {
            data[i] *= scale;
        }
        hidden = restore_device(hidden, first_dev);
    }

    // Gemma4 per-layer embedding projection
    if (cfg.n_embd_per_layer > 0 && weights_.per_layer_model_proj) {
        PERF_SCOPE("forward/per_layer_proj");
        auto proj = ops::matmul_transB(hidden, weights_.per_layer_model_proj);

        // Scale projection (need CPU access for scalar loop)
        proj = ensure_cpu(proj);
        float proj_scale = 1.0f / std::sqrt(static_cast<float>(cfg.hidden_dim));
        float* proj_data = static_cast<float*>(proj->data());
        int proj_n = static_cast<int>(proj->numel());
        for (int i = 0; i < proj_n; ++i) {
            proj_data[i] *= proj_scale;
        }

        // Apply per-layer proj norm
        if (weights_.per_layer_proj_norm) {
            int n_layer = cfg.num_layers;
            int n_per = cfg.n_embd_per_layer;
            auto norm_w = ensure_cpu(weights_.per_layer_proj_norm);
            for (int s = 0; s < seq_len; ++s) {
                for (int l = 0; l < n_layer; ++l) {
                    float* chunk = proj_data + s * n_layer * n_per + l * n_per;
                    float ss = 0.0f;
                    for (int d = 0; d < n_per; ++d) ss += chunk[d] * chunk[d];
                    float inv_rms = 1.0f / (std::sqrt(ss / n_per + cfg.rms_norm_eps));
                    const float* nw = static_cast<const float*>(norm_w->data());
                    for (int d = 0; d < n_per; ++d) chunk[d] = chunk[d] * inv_rms * nw[d];
                }
            }
        }

        per_layer_proj_cache_ = proj;
    }

    // Compute per-layer token embeddings if present
    if (cfg.n_embd_per_layer > 0 && weights_.per_layer_tok_embd) {
        PERF_SCOPE("forward/per_layer_embd");
        int n_layer = cfg.num_layers;
        int n_per = cfg.n_embd_per_layer;
        float embd_scale = std::sqrt(static_cast<float>(n_per));

        // Use ops::embedding for efficient row-wise dequantization of Q6_K
        auto ple = ops::embedding(weights_.per_layer_tok_embd, ids_on_dev, nullptr);

        // Scale by sqrt(n_embd_per_layer) (need CPU access for scalar loop)
        ple = ensure_cpu(ple);
        float* ple_data = static_cast<float*>(ple->data());
        int64_t total = ple->numel();
        for (int64_t j = 0; j < total; ++j) {
            ple_data[j] *= embd_scale;
        }

        // Add per-layer projection cache
        if (per_layer_proj_cache_) {
            auto proj_cpu = ensure_cpu(per_layer_proj_cache_);
            float* proj_data = static_cast<float*>(proj_cpu->data());
            float input_scale = 1.0f / std::sqrt(2.0f);
            for (int64_t j = 0; j < total; ++j) {
                ple_data[j] = (ple_data[j] + proj_data[j]) * input_scale;
            }
        }

        per_layer_input_cache_ = ple;
    }

    auto logits = forward_layers(hidden, seq_len, start_pos);

    // Gemma4 final logit softcapping (need CPU access for scalar loop)
    if (cfg.f_final_logit_softcapping > 0.0f && logits) {
        PERF_SCOPE("forward/logit_softcap");
        DeviceType logits_dev = logits->device();
        logits = ensure_cpu(logits);
        float cap = cfg.f_final_logit_softcapping;
        int n = static_cast<int>(logits->numel());
        float* data = static_cast<float*>(logits->data());
        for (int i = 0; i < n; ++i) {
            data[i] = std::tanh(data[i] / cap) * cap;
        }
        // Suppress tokens: set their logits to -INFINITY
        if (!cfg.suppress_tokens.empty()) {
            int vocab_size = logits->shape().back();
            int num_rows = logits->numel() / vocab_size;
            for (int tok_id : cfg.suppress_tokens) {
                if (tok_id >= 0 && tok_id < vocab_size) {
                    for (int r = 0; r < num_rows; ++r) {
                        data[r * vocab_size + tok_id] = -INFINITY;
                    }
                }
            }
        }
        logits = restore_device(logits, logits_dev);
    } else if (!cfg.suppress_tokens.empty() && logits) {
        DeviceType logits_dev = logits->device();
        logits = ensure_cpu(logits);
        float* logits_data = static_cast<float*>(logits->data());
        int vocab_size = logits->shape().back();
        int num_rows = logits->numel() / vocab_size;
        for (int tok_id : cfg.suppress_tokens) {
            if (tok_id >= 0 && tok_id < vocab_size) {
                for (int r = 0; r < num_rows; ++r) {
                    logits_data[r * vocab_size + tok_id] = -INFINITY;
                }
            }
        }
        logits = restore_device(logits, logits_dev);
    }

    return logits;
}

TensorPtr Gemma4Engine::forward_layer(const TensorPtr& hidden, int layer_idx, int seq_len,
                                      int64_t start_pos, DeviceType dev) {
    const auto& cfg = model_.config();
    const auto& lw = weights_.layers[layer_idx];

    // Per-layer head dimensions: SWA layers use different head_dim than full-attention
    bool is_swa_layer = (layer_idx < (int)cfg.swa_layers.size() && cfg.swa_layers[layer_idx] == 1);
    bool has_kv = (layer_idx < cfg.n_layer_kv_from_start);

    int head_dim = is_swa_layer ? cfg.head_dim_swa : cfg.head_dim;
    int num_heads = is_swa_layer ? cfg.num_heads_swa : cfg.num_heads;
    int num_kv_heads = is_swa_layer ? cfg.num_kv_heads_swa : cfg.num_kv_heads;

    // ---- Pre-attention RMSNorm ----
    TensorPtr normed;
    {
        PERF_SCOPE("layer/attn_norm");
        normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);
    }

    // ---- Q Projection ----
    TensorPtr q = ops::matmul_transB(normed, lw.wq());

    // ---- Q-Norm (per-head RMSNorm) ----
    // Q-Norm, K-Norm, V-Norm, and RoPE use CPU scalar loops, so ensure data is on CPU
    if (lw.attn_q_norm()) {
        PERF_SCOPE("layer/q_norm");
        q = ensure_cpu(q);
        auto qn_w = ensure_cpu(lw.attn_q_norm());
        float* q_data = static_cast<float*>(q->data());
        const float* qn_w_data = static_cast<const float*>(qn_w->data());
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads; ++h) {
                float* head = q_data + s * num_heads * head_dim + h * head_dim;
                float ss = 0.0f;
                for (int d = 0; d < head_dim; ++d) ss += head[d] * head[d];
                float inv_rms = 1.0f / (std::sqrt(ss / head_dim + cfg.rms_norm_eps));
                for (int d = 0; d < head_dim; ++d) head[d] = head[d] * inv_rms * qn_w_data[d];
            }
        }
    }

    // ---- RoPE (NeoX style) ----
    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
    TensorPtr k_rope, v;

    if (has_kv) {
        // ---- K, V Projection ----
        TensorPtr k = ops::matmul_transB(normed, lw.wk());
        if (lw.wv()) {
            v = ops::matmul_transB(normed, lw.wv());
        } else {
            v = k;
        }

        // ---- K-Norm (per-head RMSNorm) ----
        if (lw.attn_k_norm()) {
            PERF_SCOPE("layer/k_norm");
            k = ensure_cpu(k);
            auto kn_w = ensure_cpu(lw.attn_k_norm());
            float* k_data = static_cast<float*>(k->data());
            const float* kn_w_data = static_cast<const float*>(kn_w->data());
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < num_kv_heads; ++h) {
                    float* head = k_data + s * num_kv_heads * head_dim + h * head_dim;
                    float ss = 0.0f;
                    for (int d = 0; d < head_dim; ++d) ss += head[d] * head[d];
                    float inv_rms = 1.0f / (std::sqrt(ss / head_dim + cfg.rms_norm_eps));
                    for (int d = 0; d < head_dim; ++d) head[d] = head[d] * inv_rms * kn_w_data[d];
                }
            }
        }

        // ---- V-Norm (RMSNorm without learned weight, matching llama.cpp ggml_rms_norm) ----
        {
            PERF_SCOPE("layer/v_norm");
            v = ensure_cpu(v);
            float* v_data = static_cast<float*>(v->data());
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < num_kv_heads; ++h) {
                    float* head = v_data + s * num_kv_heads * head_dim + h * head_dim;
                    float ss = 0.0f;
                    for (int d = 0; d < head_dim; ++d) ss += head[d] * head[d];
                    float inv_rms = 1.0f / (std::sqrt(ss / head_dim + cfg.rms_norm_eps));
                    for (int d = 0; d < head_dim; ++d) head[d] *= inv_rms;
                }
            }
        }

        // Apply RoPE to Q and K (all on CPU)
        q = ensure_cpu(q);
        k = ensure_cpu(k);
        k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
        {
            PERF_SCOPE("layer/rope");
            const float* q_data = static_cast<const float*>(q->data());
            const float* k_data = static_cast<const float*>(k->data());
            float* qo = static_cast<float*>(q_rope->data());
            float* ko = static_cast<float*>(k_rope->data());

            int half_dim = head_dim / 2;
            int q_stride = num_heads * head_dim;
            int k_stride = num_kv_heads * head_dim;
            float q_scale = std::sqrt(static_cast<float>(head_dim));

            float theta = is_swa_layer ? cfg.rope_theta_swa : cfg.rope_theta;
            const float* freq_factors = nullptr;
            if (!is_swa_layer && rope_freqs_) {
                auto cpu_rf = ensure_cpu(rope_freqs_);
                rope_freqs_cpu_ = cpu_rf;
                freq_factors = static_cast<const float*>(rope_freqs_cpu_->data());
            }

            for (int s = 0; s < seq_len; ++s) {
                int64_t pos = start_pos + s;
                for (int h = 0; h < num_heads; ++h) {
                    for (int d = 0; d < half_dim; ++d) {
                        float base_freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                        float freq = freq_factors ? base_freq / freq_factors[d] : base_freq;
                        float angle = pos * freq;
                        float cos_a = std::cos(angle);
                        float sin_a = std::sin(angle);

                        int q_idx0 = s * q_stride + h * head_dim + d;
                        int q_idx1 = q_idx0 + half_dim;

                        float q0 = q_data[q_idx0];
                        float q1 = q_data[q_idx1];
                        qo[q_idx0] = (q0 * cos_a - q1 * sin_a) * q_scale;
                        qo[q_idx1] = (q0 * sin_a + q1 * cos_a) * q_scale;

                        if (h < num_kv_heads) {
                            int k_idx0 = s * k_stride + h * head_dim + d;
                            int k_idx1 = k_idx0 + half_dim;

                            float k0 = k_data[k_idx0];
                            float k1 = k_data[k_idx1];
                            ko[k_idx0] = k0 * cos_a - k1 * sin_a;
                            ko[k_idx1] = k0 * sin_a + k1 * cos_a;
                        }
                    }
                }
            }
        }

        // ---- KV Cache update (on CPU, then copy to cache device) ----
        {
            PERF_SCOPE("layer/kv_cache_update");
            kv_cache_.update(layer_idx, /*seq_id=*/0, start_pos, k_rope, v, seq_len);
        }
    } else {
        // Non-KV layer: reuse KV cache from the last two KV layers
        // Apply RoPE to Q only (with sqrt(d) pre-scaling for QK-Norm attention)
        q = ensure_cpu(q);
        {
            PERF_SCOPE("layer/rope_q_only");
            const float* q_data = static_cast<const float*>(q->data());
            float* qo = static_cast<float*>(q_rope->data());

            int half_dim = head_dim / 2;
            int q_stride = num_heads * head_dim;
            float q_scale = std::sqrt(static_cast<float>(head_dim));

            float theta = is_swa_layer ? cfg.rope_theta_swa : cfg.rope_theta;
            const float* freq_factors = nullptr;
            if (!is_swa_layer && rope_freqs_) {
                auto cpu_rf = ensure_cpu(rope_freqs_);
                rope_freqs_cpu_ = cpu_rf;
                freq_factors = static_cast<const float*>(rope_freqs_cpu_->data());
            }

            for (int s = 0; s < seq_len; ++s) {
                int64_t pos = start_pos + s;
                for (int h = 0; h < num_heads; ++h) {
                    for (int d = 0; d < half_dim; ++d) {
                        float base_freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                        float freq = freq_factors ? base_freq / freq_factors[d] : base_freq;
                        float angle = pos * freq;
                        float cos_a = std::cos(angle);
                        float sin_a = std::sin(angle);

                        int q_idx0 = s * q_stride + h * head_dim + d;
                        int q_idx1 = q_idx0 + half_dim;

                        float q0 = q_data[q_idx0];
                        float q1 = q_data[q_idx1];
                        qo[q_idx0] = (q0 * cos_a - q1 * sin_a) * q_scale;
                        qo[q_idx1] = (q0 * sin_a + q1 * cos_a) * q_scale;
                    }
                }
            }
        }
    }

    // Transfer Q, K, V back to the layer device for attention computation
    q_rope = restore_device(q_rope, dev);
    if (v) v = restore_device(v, dev);
    if (k_rope) k_rope = restore_device(k_rope, dev);

    // ---- Attention ----
    TensorPtr attn_out;
    {
        PERF_SCOPE("layer/attention");
        int total_len = 0;
        TensorPtr k_sliced, v_sliced;
        int attn_num_kv_heads = num_kv_heads;
        int attn_head_dim = head_dim;

        if (has_kv) {
            total_len = kv_cache_.filled(layer_idx);
            k_sliced = kv_cache_.get_key_filled(layer_idx);
            v_sliced = kv_cache_.get_value_filled(layer_idx);
        } else {
            // Non-KV layers reuse KV from the last two KV layers:
            // - SWA layers reuse from n_layer_kv_from_start - 2 (SWA layer, head_dim_swa)
            // - Full-attention layers reuse from n_layer_kv_from_start - 1 (full layer, head_dim)
            // This matches the llama.cpp reference implementation.
            int reuse_layer = is_swa_layer
                ? (cfg.n_layer_kv_from_start - 2)
                : (cfg.n_layer_kv_from_start - 1);
            total_len = kv_cache_.filled(reuse_layer);
            k_sliced = kv_cache_.get_key_filled(reuse_layer);
            v_sliced = kv_cache_.get_value_filled(reuse_layer);
            // Use the reused KV layer's dimensions for attention
            bool reuse_is_swa = (reuse_layer < (int)cfg.swa_layers.size() && cfg.swa_layers[reuse_layer] == 1);
            attn_num_kv_heads = reuse_is_swa ? cfg.num_kv_heads_swa : cfg.num_kv_heads;
            attn_head_dim = reuse_is_swa ? cfg.head_dim_swa : cfg.head_dim;
        }

        // ---- SWA sliding window: restrict KV cache to last n_swa positions ----
        if (is_swa_layer && cfg.n_swa > 0 && total_len > cfg.n_swa) {
            int swa_start = total_len - cfg.n_swa;
            // Slice K and V to only include positions [swa_start, total_len)
            if (k_sliced) k_sliced = std::make_shared<Tensor>(k_sliced->slice(0, swa_start, total_len));
            if (v_sliced) v_sliced = std::make_shared<Tensor>(v_sliced->slice(0, swa_start, total_len));
            total_len = cfg.n_swa;
        }

        if (dev == DeviceType::CUDA && k_sliced && k_sliced->device() == DeviceType::CPU) {
            auto k_cuda = std::make_shared<Tensor>(DataType::FP32, k_sliced->shape(), DeviceType::CUDA);
            k_cuda->copy_from(*k_sliced);
            k_sliced = k_cuda;
            auto v_cuda = std::make_shared<Tensor>(DataType::FP32, v_sliced->shape(), DeviceType::CUDA);
            v_cuda->copy_from(*v_sliced);
            v_sliced = v_cuda;
        }

        if (attn_num_kv_heads < num_heads) {
            attn_out = ops::scaled_dot_product_attention_2d_gqa(q_rope, k_sliced, v_sliced, seq_len,
                                                                total_len, num_heads, attn_num_kv_heads,
                                                                attn_head_dim, true);
        } else {
            attn_out = ops::scaled_dot_product_attention_2d(q_rope, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, attn_head_dim, true);
        }
    }

    // ---- Attention output projection ----
    {
        PERF_SCOPE("layer/attn_proj");
        attn_out = ops::matmul_transB(attn_out, lw.wo());
    }

    // ---- Post-attention norm ----
    if (lw.attn_post_norm()) {
        PERF_SCOPE("layer/attn_post_norm");
        attn_out = ops::rms_norm(attn_out, lw.attn_post_norm(), cfg.rms_norm_eps);
    }

    // ---- First residual add ----
    auto attn_residual = ops::add(hidden, attn_out);

    // ---- FFN ----
    TensorPtr ffn_out;
    bool is_moe = (lw.ffn_gate_inp() != nullptr);

    if (is_moe) {
        // Gemma4 MoE: shared expert + routed experts
        PERF_SCOPE("layer/moe");

        // Shared expert (standard FFN with GeGLU)
        TensorPtr shared_out;
        {
            auto cur_mlp = ops::rms_norm(attn_residual, lw.ffn_norm(), cfg.rms_norm_eps);
            auto gate = ops::matmul_transB(cur_mlp, lw.w1());
            auto up = ops::matmul_transB(cur_mlp, lw.w3());
            auto gated = ops::gelu_multiply(gate, up);
            shared_out = ops::matmul_transB(gated, lw.w2());
        }
        if (lw.ffn_post_norm_1()) {
            shared_out = ops::rms_norm(shared_out, lw.ffn_post_norm_1(), cfg.rms_norm_eps);
        }

        // Routed experts (all scalar operations done on CPU)
        TensorPtr expert_out;
        {
            auto cur_moe = ops::rms_norm(attn_residual, lw.ffn_pre_norm_2(), cfg.rms_norm_eps);

            // Router: compute expert logits
            TensorPtr router_input = attn_residual;
            if (lw.ffn_gate_inp_s()) {
                // Custom scaling: rms_norm then multiply by scale
                auto ones = std::make_shared<Tensor>(DataType::FP32,
                    std::vector<int64_t>{1, cfg.hidden_dim}, DeviceType::CPU);
                float* ones_data = static_cast<float*>(ones->data());
                std::fill_n(ones_data, cfg.hidden_dim, 1.0f);
                auto normed_for_router = ops::rms_norm(attn_residual, ones, cfg.rms_norm_eps);
                normed_for_router = ensure_cpu(normed_for_router);
                float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(cfg.hidden_dim));
                float* nr_data = static_cast<float*>(normed_for_router->data());
                int nr_n = static_cast<int>(normed_for_router->numel());
                for (int i = 0; i < nr_n; ++i) nr_data[i] *= inv_sqrt;
                auto s_cpu = ensure_cpu(lw.ffn_gate_inp_s());
                const float* s_data = static_cast<const float*>(s_cpu->data());
                // Per-row element-wise multiply: apply scale per token row
                for (int s = 0; s < seq_len; ++s) {
                    float* row = nr_data + s * cfg.hidden_dim;
                    for (int d = 0; d < cfg.hidden_dim; ++d) {
                        row[d] *= s_data[d];
                    }
                }
                router_input = normed_for_router;
            }
            auto router_logits = ops::matmul_transB(router_input, lw.ffn_gate_inp());

            // Top-K expert selection with softmax (on CPU)
            int n_expert = cfg.n_expert;
            int n_expert_used = cfg.n_expert_used > 0 ? cfg.n_expert_used : 1;
            int n_tokens = seq_len;

            expert_out = std::make_shared<Tensor>(DataType::FP32,
                std::vector<int64_t>{seq_len, cfg.hidden_dim}, DeviceType::CPU);
            float* expert_out_data = static_cast<float*>(expert_out->data());
            std::fill_n(expert_out_data, seq_len * cfg.hidden_dim, 0.0f);

            auto router_logits_cpu = ensure_cpu(router_logits);
            auto cur_moe_cpu = ensure_cpu(cur_moe);
            const float* logits_data = static_cast<const float*>(router_logits_cpu->data());

            for (int s = 0; s < n_tokens; ++s) {
                // Softmax over experts
                std::vector<float> probs(n_expert);
                float max_logit = -std::numeric_limits<float>::infinity();
                for (int e = 0; e < n_expert; ++e) {
                    probs[e] = logits_data[s * n_expert + e];
                    if (probs[e] > max_logit) max_logit = probs[e];
                }
                float sum_exp = 0.0f;
                for (int e = 0; e < n_expert; ++e) {
                    probs[e] = std::exp(probs[e] - max_logit);
                    sum_exp += probs[e];
                }
                for (int e = 0; e < n_expert; ++e) {
                    probs[e] /= sum_exp;
                }

                // Top-K selection
                std::vector<int> indices(n_expert);
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(indices.begin(), indices.begin() + n_expert_used, indices.end(),
                    [&](int a, int b) { return probs[a] > probs[b]; });

                // Renormalize top-K probs
                float topk_sum = 0.0f;
                for (int k = 0; k < n_expert_used; ++k) {
                    topk_sum += probs[indices[k]];
                }

                for (int k = 0; k < n_expert_used; ++k) {
                    int expert_idx = indices[k];
                    float weight = probs[expert_idx] / topk_sum;

                    // Extract token hidden state (on CPU)
                    TensorPtr token_hidden = std::make_shared<Tensor>(DataType::FP32,
                        std::vector<int64_t>{1, cfg.hidden_dim}, DeviceType::CPU);
                    const float* moe_in = static_cast<const float*>(cur_moe_cpu->data()) + s * cfg.hidden_dim;
                    std::memcpy(token_hidden->data(), moe_in, cfg.hidden_dim * sizeof(float));

                    TensorPtr expert_result;

                    auto extract_expert_2d = [&](const TensorPtr& w3d) -> TensorPtr {
                        if (!w3d) return nullptr;
                        auto& shp = w3d->shape();
                        if (shp.size() < 2) return w3d;
                        int64_t d1 = shp[0];
                        int64_t d2 = shp[1];
                        if (shp.size() == 3 && expert_idx < shp[2]) {
                            TensorPtr fp32_w = w3d;
                            if (is_quantized_type(w3d->dtype())) {
                                fp32_w = ops::dequantize_weight(w3d);
                            }
                            fp32_w = ensure_cpu(fp32_w);
                            int64_t expert_stride = d1 * d2;
                            const float* src = static_cast<const float*>(fp32_w->data()) +
                                               expert_idx * expert_stride;
                            auto slice = std::make_shared<Tensor>(DataType::FP32,
                                std::vector<int64_t>{d1, d2}, DeviceType::CPU);
                            std::memcpy(slice->data(), src, expert_stride * sizeof(float));
                            return slice;
                        }
                        return w3d;
                    };

                    if (lw.ffn_gate_up_exps()) {
                        auto gate_up_w = extract_expert_2d(lw.ffn_gate_up_exps());
                        auto down_w = extract_expert_2d(lw.ffn_down_exps());
                        if (gate_up_w && down_w) {
                            auto gate_up = ops::matmul_transB(token_hidden, gate_up_w);
                            int half_out = static_cast<int>(gate_up->shape()[1]) / 2;
                            auto gate_t = std::make_shared<Tensor>(DataType::FP32,
                                std::vector<int64_t>{1, half_out}, DeviceType::CPU);
                            auto up_t = std::make_shared<Tensor>(DataType::FP32,
                                std::vector<int64_t>{1, half_out}, DeviceType::CPU);
                            auto gate_up_cpu = ensure_cpu(gate_up);
                            const float* gu_data = static_cast<const float*>(gate_up_cpu->data());
                            std::memcpy(gate_t->data(), gu_data, half_out * sizeof(float));
                            std::memcpy(up_t->data(), gu_data + half_out, half_out * sizeof(float));
                            auto gated = ops::gelu_multiply(gate_t, up_t);
                            expert_result = ops::matmul_transB(gated, down_w);
                        }
                    } else if (lw.ffn_gate_exps() && lw.ffn_up_exps()) {
                        auto gate_w = extract_expert_2d(lw.ffn_gate_exps());
                        auto up_w = extract_expert_2d(lw.ffn_up_exps());
                        auto down_w = extract_expert_2d(lw.ffn_down_exps());
                        if (gate_w && up_w && down_w) {
                            auto gate_t = ops::matmul_transB(token_hidden, gate_w);
                            auto up_t = ops::matmul_transB(token_hidden, up_w);
                            auto gated = ops::gelu_multiply(gate_t, up_t);
                            expert_result = ops::matmul_transB(gated, down_w);
                        }
                    }

                    if (expert_result) {
                        auto er_cpu = ensure_cpu(expert_result);
                        const float* er_data = static_cast<const float*>(er_cpu->data());
                        float* out_row = expert_out_data + s * cfg.hidden_dim;
                        for (int d = 0; d < cfg.hidden_dim; ++d) {
                            out_row[d] += weight * er_data[d];
                        }
                    }
                }
            }
        }
        if (lw.ffn_post_norm_2()) {
            expert_out = ops::rms_norm(expert_out, lw.ffn_post_norm_2(), cfg.rms_norm_eps);
        }
        // Transfer expert_out back to device for the add operation
        expert_out = restore_device(expert_out, dev);

        // Combine shared + routed experts
        ffn_out = ops::add(shared_out, expert_out);
    } else {
        // Standard GeGLU FFN
        PERF_SCOPE("layer/ffn");
        auto ffn_normed = ops::rms_norm(attn_residual, lw.ffn_norm(), cfg.rms_norm_eps);
        auto gate = ops::matmul_transB(ffn_normed, lw.w1());
        auto up = ops::matmul_transB(ffn_normed, lw.w3());
        auto gated = ops::gelu_multiply(gate, up);
        ffn_out = ops::matmul_transB(gated, lw.w2());
    }

    // ---- Post-FFN norm ----
    if (lw.post_ffn_norm()) {
        PERF_SCOPE("layer/post_ffn_norm");
        ffn_out = ops::rms_norm(ffn_out, lw.post_ffn_norm(), cfg.rms_norm_eps);
    }

    // ---- Second residual add ----
    auto output = ops::add(attn_residual, ffn_out);

    // ---- Per-layer embeddings ----
    if (lw.per_layer_inp_gate() && lw.per_layer_proj() && per_layer_input_cache_) {
        PERF_SCOPE("layer/per_layer_embd");
        int n_per = cfg.n_embd_per_layer;
        int n_layer = cfg.num_layers;

        // Input gate projection
        auto gated = ops::matmul_transB(output, lw.per_layer_inp_gate());
        // GELU activation (on CPU)
        DeviceType gated_dev = gated->device();
        gated = ensure_cpu(gated);
        float* gated_data = static_cast<float*>(gated->data());
        for (int i = 0; i < static_cast<int>(gated->numel()); ++i) {
            float x = gated_data[i];
            gated_data[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        }

        // Element-wise multiply with per-layer embedding for this specific layer
        auto ple_cpu = ensure_cpu(per_layer_input_cache_);
        const float* ple_data = static_cast<const float*>(ple_cpu->data());
        for (int s = 0; s < seq_len; ++s) {
            const float* layer_embd = ple_data + s * n_per * n_layer + layer_idx * n_per;
            float* gated_row = gated_data + s * n_per;
            for (int d = 0; d < n_per; ++d) {
                gated_row[d] *= layer_embd[d];
            }
        }

        // Restore gated to original device for matmul
        gated = restore_device(gated, gated_dev);

        // Project back to hidden dim
        auto pe_out = ops::matmul_transB(gated, lw.per_layer_proj());
        if (lw.per_layer_post_norm()) {
            pe_out = ops::rms_norm(pe_out, lw.per_layer_post_norm(), cfg.rms_norm_eps);
        }

        // Residual add
        output = ops::add(output, pe_out);
    }

    // ---- Layer output scale ----
    if (lw.layer_out_scale()) {
        auto scale_cpu = ensure_cpu(lw.layer_out_scale());
        output = ensure_cpu(output);
        const float* scale_data = static_cast<const float*>(scale_cpu->data());
        float* out_data = static_cast<float*>(output->data());
        float s = scale_data[0];
        int n = static_cast<int>(output->numel());
        for (int i = 0; i < n; ++i) {
            out_data[i] *= s;
        }
        output = restore_device(output, dev);
    }

    return output;
}

}  // namespace forge
