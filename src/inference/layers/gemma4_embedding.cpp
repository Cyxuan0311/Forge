#include "forge/inference/layers/gemma4_embedding.h"

#include <cmath>

#include "forge/cuda_kernels.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

namespace forge {

TensorPtr Gemma4Embedding::embed(const TensorPtr& ids_on_dev, const TensorPtr& token_emb,
                                const ModelWeights& weights, const ModelConfig& cfg,
                                DeviceType dev, int seq_len) {
    TensorPtr hidden;
    {
        PERF_SCOPE("forward/embedding");
        hidden = ops::embedding(token_emb, ids_on_dev, weights.token_embedding_fp32);
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
        hidden = restore_device(hidden, dev);
    }

    // Gemma4 per-layer embedding projection
    if (cfg.n_embd_per_layer > 0 && weights.per_layer_model_proj) {
        PERF_SCOPE("forward/per_layer_proj");
        auto proj = ops::matmul_transB(hidden, weights.per_layer_model_proj);

        proj = ensure_cpu(proj);
        float proj_scale = 1.0f / std::sqrt(static_cast<float>(cfg.hidden_dim));
        float* proj_data = static_cast<float*>(proj->data());
        int proj_n = static_cast<int>(proj->numel());
        for (int i = 0; i < proj_n; ++i) {
            proj_data[i] *= proj_scale;
        }

        if (weights.per_layer_proj_norm) {
            int n_layer = cfg.num_layers;
            int n_per = cfg.n_embd_per_layer;
            auto norm_w = ensure_cpu(weights.per_layer_proj_norm);
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

    // per-layer token embeddings
    if (cfg.n_embd_per_layer > 0 && weights.per_layer_tok_embd) {
        PERF_SCOPE("forward/per_layer_embd");
        int n_per = cfg.n_embd_per_layer;
        float embd_scale = std::sqrt(static_cast<float>(n_per));

        // ops::embedding 会按行反量化 (Q6_K), 比整体反量化便宜。
        auto ple = ops::embedding(weights.per_layer_tok_embd, ids_on_dev, nullptr);

        ple = ensure_cpu(ple);
        float* ple_data = static_cast<float*>(ple->data());
        int64_t total = ple->numel();
        for (int64_t j = 0; j < total; ++j) {
            ple_data[j] *= embd_scale;
        }

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

    return hidden;
}

TensorPtr Gemma4Embedding::apply_per_layer(const TensorPtr& output,
                                          const LayerExecutionContext& lctx) {
    const auto& lw = lctx.weights;
    if (!lw.per_layer_inp_gate() || !lw.per_layer_proj() || !per_layer_input_cache_) {
        return output;
    }

    PERF_SCOPE("layer/per_layer_embd");
    const auto& cfg = lctx.config;
    const int n_per = cfg.n_embd_per_layer;
    const int n_layer = cfg.num_layers;
    const int seq_len = lctx.seq_len();
    const int layer_idx = lctx.layer_idx;
    const DeviceType dev = lctx.device;

    auto gated = ops::matmul_transB(output, lw.per_layer_inp_gate());

#ifdef USE_CUDA
    if (gated->device() == DeviceType::CUDA &&
        per_layer_input_cache_->device() == DeviceType::CUDA) {
        cuda::launch_gelu_tanh_multiply(
            static_cast<float*>(gated->data()),
            static_cast<const float*>(per_layer_input_cache_->data()), n_per, n_layer, layer_idx,
            seq_len);
    } else
#endif
    {
        gated = ensure_cpu(gated);
        float* gated_data = static_cast<float*>(gated->data());
        for (int i = 0; i < static_cast<int>(gated->numel()); ++i) {
            float x = gated_data[i];
            gated_data[i] =
                0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        }
        auto ple_cpu = ensure_cpu(per_layer_input_cache_);
        const float* ple_data = static_cast<const float*>(ple_cpu->data());
        for (int s = 0; s < seq_len; ++s) {
            const float* layer_embd = ple_data + s * n_per * n_layer + layer_idx * n_per;
            float* gated_row = gated_data + s * n_per;
            for (int d = 0; d < n_per; ++d) {
                gated_row[d] *= layer_embd[d];
            }
        }
        gated = restore_device(gated, dev);
    }

    auto pe_out = ops::matmul_transB(gated, lw.per_layer_proj());
    if (lw.per_layer_post_norm()) {
        pe_out = ops::rms_norm(pe_out, lw.per_layer_post_norm(), cfg.rms_norm_eps);
    }

    return ops::add(output, pe_out);
}

}  // namespace forge