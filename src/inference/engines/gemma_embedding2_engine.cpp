#include "forge/engines/gemma_embedding2_engine.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/inference_batch.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

namespace forge {

namespace {

// per-head RMSNorm with optional learned weight (weight == nullptr -> plain
// RMSNorm, used for the V-norm). Same helper as gemma4_attention.cpp.
void per_head_rms_norm_cpu(float* data, const float* weight, int seq_len, int num_heads,
                           int head_dim, float eps) {
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            float* head = data + s * num_heads * head_dim + h * head_dim;
            float ss = 0.0f;
            for (int d = 0; d < head_dim; ++d) ss += head[d] * head[d];
            float inv_rms = 1.0f / (std::sqrt(ss / head_dim + eps));
            if (weight) {
                for (int d = 0; d < head_dim; ++d) head[d] = head[d] * inv_rms * weight[d];
            } else {
                for (int d = 0; d < head_dim; ++d) head[d] *= inv_rms;
            }
        }
    }
}

// Gemma4 RoPE: half-split rotation + sqrt(head_dim) Q pre-scaling. The SDPA
// applies 1/sqrt(head_dim) internally, so the net attention scale is 1.0 --
// exactly llama.cpp's f_attention_scale = 1.0 semantics for this arch.
void rope_gemma4_cpu(const float* q_data, const float* k_data, float* q_out, float* k_out,
                     int seq_len, int num_heads, int num_kv_heads, int head_dim, int64_t start_pos,
                     float theta) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    float q_scale = std::sqrt(static_cast<float>(head_dim));

    for (int s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = pos * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int q_idx0 = s * q_stride + h * head_dim + d;
                int q_idx1 = q_idx0 + half_dim;

                float q0 = q_data[q_idx0];
                float q1 = q_data[q_idx1];
                q_out[q_idx0] = (q0 * cos_a - q1 * sin_a) * q_scale;
                q_out[q_idx1] = (q0 * sin_a + q1 * cos_a) * q_scale;

                if (k_data && h < num_kv_heads) {
                    int k_idx0 = s * k_stride + h * head_dim + d;
                    int k_idx1 = k_idx0 + half_dim;

                    float k0 = k_data[k_idx0];
                    float k1 = k_data[k_idx1];
                    k_out[k_idx0] = k0 * cos_a - k1 * sin_a;
                    k_out[k_idx1] = k0 * sin_a + k1 * cos_a;
                }
            }
        }
    }
}

}  // namespace

GemmaEmbedding2Engine::GemmaEmbedding2Engine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx) {
    if (!init_weights()) {
        throw std::runtime_error("GemmaEmbedding2Engine: failed to initialize weights");
    }
}

bool GemmaEmbedding2Engine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

void GemmaEmbedding2Engine::init_kv_cache(const ModelConfig& cfg) {
    // Embedding model: the whole prompt attends in one pass, no KV cache.
    // Mark initialized so init_memory()/reset() stay harmless no-ops.
    (void)cfg;
    set_kv_cache_initialized(true);
}

TensorPtr GemmaEmbedding2Engine::forward_request(const ForwardRequest& req_in) {
    const auto& cfg = model_.config();

    // return_all_logits=true keeps forward_layers() from collapsing the sparse
    // prefill projection to the last row -- pooling needs every token row.
    ForwardRequest req = req_in;
    req.return_all_logits = true;

    DeviceType first_dev = layer_device(0);
    auto ids_on_dev = transfer_hidden(req.input_ids, first_dev);
    auto token_emb = model_.weights().get("token_embedding");

    auto hidden = embedding_.embed(ids_on_dev, token_emb, weights_, cfg, first_dev, req.n_tokens);

    // forward_layers() runs the block stack plus the shared tail:
    // rms_norm(output_norm) -> output projection [seq, n_embd_out].
    auto embd = forward_layers(hidden, req);
    if (!embd)
        return nullptr;

    // Mean pooling over tokens -> [1, n_embd_out] (pooling_type = MEAN;
    // projecting per token is equivalent to projecting after mean pooling).
    embd = ensure_cpu(embd);
    int n_out = static_cast<int>(embd->shape().back());
    int n_tok = static_cast<int>(embd->shape()[0]);
    const float* src = static_cast<const float*>(embd->data());
    auto pooled =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, n_out}, DeviceType::CPU);
    float* dst = static_cast<float*>(pooled->data());
    std::memset(dst, 0, n_out * sizeof(float));
    for (int s = 0; s < n_tok; ++s) {
        const float* row = src + static_cast<size_t>(s) * n_out;
        for (int d = 0; d < n_out; ++d) dst[d] += row[d];
    }
    float inv = 1.0f / static_cast<float>(n_tok);
    for (int d = 0; d < n_out; ++d) dst[d] *= inv;
    return pooled;
}

TensorPtr GemmaEmbedding2Engine::forward_batch(const InferenceBatch& batch) {
    if (batch.empty())
        return nullptr;

    int n_seq = static_cast<int>(batch.size());
    int n_out = -1;

    struct SeqEmb {
        int idx;
        std::vector<float> data;
    };
    std::vector<SeqEmb> results;

    for (int i = 0; i < n_seq; ++i) {
        const auto& item = batch.items[i];
        int seq_len = static_cast<int>(item.tokens.size());
        auto input_ids = std::make_shared<Tensor>(DataType::INT32,
                                                  std::vector<int64_t>{seq_len}, DeviceType::CPU);
        std::memcpy(input_ids->data(), item.tokens.data(), seq_len * sizeof(int32_t));

        auto embd = forward_request(ForwardRequest::from_ids(input_ids, item.start_pos, item.seq_id));
        if (!embd)
            continue;

        embd = ensure_cpu(embd);
        n_out = static_cast<int>(embd->shape().back());
        const float* src = static_cast<const float*>(embd->data());
        results.push_back({i, std::vector<float>(src, src + n_out)});
    }

    if (results.empty() || n_out <= 0)
        return nullptr;

    auto result = std::make_shared<Tensor>(DataType::FP32,
                                           std::vector<int64_t>{n_seq, n_out}, DeviceType::CPU);
    float* dst = static_cast<float*>(result->data());
    std::memset(dst, 0, static_cast<size_t>(n_seq) * n_out * sizeof(float));
    for (auto& r : results) {
        std::memcpy(dst + static_cast<size_t>(r.idx) * n_out, r.data.data(), n_out * sizeof(float));
    }
    return result;
}

TensorPtr GemmaEmbedding2Engine::forward_layer(const TensorPtr& hidden,
                                               const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int layer_idx = lctx.layer_idx;
    const int seq_len = lctx.seq_len();
    const int64_t rope_pos = lctx.rope_pos();
    const DeviceType dev = lctx.device.type;

    const bool is_swa_layer =
        (layer_idx < static_cast<int>(cfg.swa_layers.size()) && cfg.swa_layers[layer_idx] == 1);

    const int head_dim = is_swa_layer ? cfg.head_dim_swa : cfg.head_dim;
    const int num_heads = is_swa_layer ? cfg.num_heads_swa : cfg.num_heads;
    const int num_kv_heads = is_swa_layer ? cfg.num_kv_heads_swa : cfg.num_kv_heads;

    // ---- Attention (bidirectional, no KV cache) ----
    TensorPtr normed;
    {
        PERF_SCOPE("layer/attn_norm");
        normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);
    }

    TensorPtr q = ops::matmul_transB(normed, lw.wq());
    TensorPtr k = ops::matmul_transB(normed, lw.wk());
    TensorPtr v = ops::matmul_transB(normed, lw.wv());

    // Q-Norm (per-head RMSNorm with learned weight)
    if (lw.attn_q_norm()) {
        PERF_SCOPE("layer/q_norm");
#ifdef USE_CUDA
        if (q->device() == DeviceType::CUDA && lw.attn_q_norm()->device() == DeviceType::CUDA) {
            auto q_out = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CUDA);
            cuda::launch_rms_norm(static_cast<const float*>(q->data()),
                                  static_cast<const float*>(lw.attn_q_norm()->data()),
                                  static_cast<float*>(q_out->data()), seq_len * num_heads, head_dim,
                                  cfg.rms_norm_eps);
            q = q_out;
        } else
#endif
        {
            q = ensure_cpu(q);
            auto qn_w = ensure_cpu(lw.attn_q_norm());
            per_head_rms_norm_cpu(static_cast<float*>(q->data()),
                                  static_cast<const float*>(qn_w->data()), seq_len, num_heads,
                                  head_dim, cfg.rms_norm_eps);
        }
    }

    // K-Norm (per-head RMSNorm with learned weight)
    if (lw.attn_k_norm()) {
        PERF_SCOPE("layer/k_norm");
#ifdef USE_CUDA
        if (k->device() == DeviceType::CUDA && lw.attn_k_norm()->device() == DeviceType::CUDA) {
            auto k_out = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CUDA);
            cuda::launch_rms_norm(static_cast<const float*>(k->data()),
                                  static_cast<const float*>(lw.attn_k_norm()->data()),
                                  static_cast<float*>(k_out->data()), seq_len * num_kv_heads,
                                  head_dim, cfg.rms_norm_eps);
            k = k_out;
        } else
#endif
        {
            k = ensure_cpu(k);
            auto kn_w = ensure_cpu(lw.attn_k_norm());
            per_head_rms_norm_cpu(static_cast<float*>(k->data()),
                                  static_cast<const float*>(kn_w->data()), seq_len, num_kv_heads,
                                  head_dim, cfg.rms_norm_eps);
        }
    }

    // V-Norm (plain RMSNorm, no learned weight)
    {
        PERF_SCOPE("layer/v_norm");
#ifdef USE_CUDA
        if (v->device() == DeviceType::CUDA) {
            auto v_out = std::make_shared<Tensor>(DataType::FP32, v->shape(), DeviceType::CUDA);
            cuda::launch_rms_norm_unweighted(static_cast<const float*>(v->data()),
                                             static_cast<float*>(v_out->data()),
                                             seq_len * num_kv_heads, head_dim, cfg.rms_norm_eps);
            v = v_out;
        } else
#endif
        {
            v = ensure_cpu(v);
            per_head_rms_norm_cpu(static_cast<float*>(v->data()), nullptr, seq_len, num_kv_heads,
                                  head_dim, cfg.rms_norm_eps);
        }
    }

    // RoPE (half-split rotation; theta differs between SWA and global layers)
    float theta = is_swa_layer ? cfg.rope_theta_swa : cfg.rope_theta;
    TensorPtr q_rope, k_rope;
#ifdef USE_CUDA
    if (q->device() == DeviceType::CUDA && k->device() == DeviceType::CUDA) {
        PERF_SCOPE("layer/rope");
        q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CUDA);
        k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CUDA);
        cuda::launch_rope_gemma4_gqa(
            static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
            static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), num_heads,
            num_kv_heads, head_dim, seq_len, rope_pos, theta, nullptr);
    } else
#endif
    {
        PERF_SCOPE("layer/rope");
        q = ensure_cpu(q);
        k = ensure_cpu(k);
        q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
        k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
        rope_gemma4_cpu(static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                        static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                        seq_len, num_heads, num_kv_heads, head_dim, rope_pos, theta);
    }
    q_rope = restore_device(q_rope, dev);
    k_rope = restore_device(k_rope, dev);

    // Symmetric SWA mask: attention is allowed when |i - j| <= n_swa / 2.
    // Global layers see every position (no mask, causal=false).
    TensorPtr mask;
    if (is_swa_layer && cfg.n_swa > 0 && seq_len > 1) {
        PERF_SCOPE("layer/swa_mask");
        const int half = cfg.n_swa / 2;
        mask = std::make_shared<Tensor>(DataType::FP32,
                                        std::vector<int64_t>{seq_len, seq_len}, DeviceType::CPU);
        float* m = static_cast<float*>(mask->data());
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                m[i * seq_len + j] = (std::abs(i - j) <= half) ? 0.0f : -INFINITY;
            }
        }
        if (dev == DeviceType::CUDA) {
            auto mask_dev = std::make_shared<Tensor>(DataType::FP32,
                                                     std::vector<int64_t>{seq_len, seq_len},
                                                     DeviceType::CUDA);
            mask_dev->copy_from(*mask);
            mask = mask_dev;
        }
    }

    TensorPtr attn;
    {
        PERF_SCOPE("layer/attention");
        attn = ops::scaled_dot_product_attention_2d_gqa(q_rope, k_rope, v, seq_len, seq_len,
                                                        num_heads, num_kv_heads, head_dim, mask,
                                                        /*causal=*/false);
    }

    {
        PERF_SCOPE("layer/attn_proj");
        attn = ops::matmul_transB(attn, lw.wo());
    }
    if (lw.attn_post_norm()) {
        PERF_SCOPE("layer/attn_post_norm");
        attn = ops::rms_norm(attn, lw.attn_post_norm(), cfg.rms_norm_eps);
    }

    auto attn_residual = ops::add(hidden, attn);

    // ---- FFN (exact-erf GeGLU: gelu(gate) * up; NOT the tanh approximation
    //      in ops::gelu_multiply) ----
    TensorPtr ffn_out;
    {
        PERF_SCOPE("layer/ffn");
        auto ffn_normed = ops::rms_norm(attn_residual, lw.ffn_norm(), cfg.rms_norm_eps);
        auto gate = ops::matmul_transB(ffn_normed, lw.w1());
        auto up = ops::matmul_transB(ffn_normed, lw.w3());
        auto gated = ops::multiply(ops::gelu(gate), up);
        ffn_out = ops::matmul_transB(gated, lw.w2());
    }
    if (lw.post_ffn_norm()) {
        PERF_SCOPE("layer/post_ffn_norm");
        ffn_out = ops::rms_norm(ffn_out, lw.post_ffn_norm(), cfg.rms_norm_eps);
    }

    auto output = ops::add(attn_residual, ffn_out);

    // ---- Per-layer embedding residual ----
    output = embedding_.apply_per_layer(output, lctx);

    // ---- Layer output scale ----
    if (lw.layer_out_scale()) {
        auto scale_cpu = ensure_cpu(lw.layer_out_scale());
        float s = static_cast<const float*>(scale_cpu->data())[0];
#ifdef USE_CUDA
        if (output->device() == DeviceType::CUDA) {
            cuda::launch_scale(static_cast<float*>(output->data()), s,
                               static_cast<int>(output->numel()));
        } else
#endif
        {
            output = ensure_cpu(output);
            float* out_data = static_cast<float*>(output->data());
            int n = static_cast<int>(output->numel());
            for (int i = 0; i < n; ++i) {
                out_data[i] *= s;
            }
        }
    }

    return output;
}

}  // namespace forge
