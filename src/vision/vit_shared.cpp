// ============================================================================
// Shared ViT toolkit implementations (P2)
// ============================================================================
// Generic sub-operations extracted from SiglipViTEncoder so that future ViT
// encoders can reuse them. The logic is identical to the original SigLIP code;
// only the function boundaries changed (methods -> free functions).
// ============================================================================
#include "forge/vit_shared.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "forge/logger.h"
#include "forge/operators.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {

// ---- shared helpers (moved from vision_encoder.cpp) ----

TensorPtr vit_dequant_to_fp32_cpu(const TensorPtr& w) {
    if (!w)
        return nullptr;
    if (w->dtype() == DataType::FP32 && w->device() == DeviceType::CPU)
        return w;

    auto out = std::make_shared<Tensor>(DataType::FP32, w->shape(), DeviceType::CPU);
    if (w->dtype() == DataType::FP32) {
        out->copy_from(*w);
    } else if (w->dtype() == DataType::FP16) {
        const uint16_t* fp16 = static_cast<const uint16_t*>(w->data());
        float* dst = static_cast<float*>(out->data());
        size_t n = out->numel();
        for (size_t i = 0; i < n; ++i) {
            uint16_t h = fp16[i];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            float f;
            if (exp == 0)
                f = mant == 0 ? 0.0f : std::ldexp(static_cast<float>(mant), -24);
            else if (exp == 31)
                f = mant == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
            else
                f = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
            dst[i] = sign ? -f : f;
        }
    }
    return out;
}

TensorPtr vit_to_device(const TensorPtr& t, DeviceType dev) {
    if (!t || t->device() == dev)
        return t;
    auto out = std::make_shared<Tensor>(t->dtype(), t->shape(), dev);
    out->copy_from(*t);
    return out;
}

// ---- vit_forward_block ----

TensorPtr vit_forward_block(const TensorPtr& hidden, const ViTLayerWeights& lw,
                            const VisionConfig& cfg, int num_heads, int head_dim,
                            ViTActivation act) {
    int num_patches = static_cast<int>(hidden->shape()[0]);

    // LayerNorm 1
    auto normed = ops::layer_norm(hidden, lw.ln1_weight, lw.ln1_bias, cfg.layer_norm_epsilon);

    // QKV projections
    auto q = ops::matmul_transB(normed, lw.attn_q_weight, lw.attn_q_bias);
    auto k = ops::matmul_transB(normed, lw.attn_k_weight, lw.attn_k_bias);
    auto v = ops::matmul_transB(normed, lw.attn_v_weight, lw.attn_v_bias);

    // Self-attention (non-causal for ViT)
    auto attn_out = ops::scaled_dot_product_attention_2d(q, k, v, num_patches, num_heads,
                                                         head_dim, nullptr, false);

    // Output projection
    auto attn_proj = ops::matmul_transB(attn_out, lw.attn_out_weight, lw.attn_out_bias);

    // Residual
    auto res1 = ops::add(hidden, attn_proj);

    // LayerNorm 2
    auto normed2 = ops::layer_norm(res1, lw.ln2_weight, lw.ln2_bias, cfg.layer_norm_epsilon);

    // FFN up + activation
    auto ffn_up = ops::matmul_transB(normed2, lw.ffn_up_weight, lw.ffn_up_bias);
    if (act == ViTActivation::GELU_Erf)
        ffn_up = ops::gelu(ffn_up);
    else
        ffn_up = ops::gelu_tanh(ffn_up);

    // FFN down
    auto ffn_down = ops::matmul_transB(ffn_up, lw.ffn_down_weight, lw.ffn_down_bias);

    // Residual
    return ops::add(res1, ffn_down);
}

// ---- vit_patch_embed ----

TensorPtr vit_patch_embed(const TensorPtr& pixel_values, const VisionConfig& cfg,
                          const TensorPtr& patch_weight, const TensorPtr& patch_bias) {
    int img_size = cfg.image_size;
    int ps = cfg.patch_size;
    int embed_dim = cfg.embedding_length;
    int num_patches_h = img_size / ps;
    int num_patches_w = img_size / ps;
    int num_patches = num_patches_h * num_patches_w;

    auto output = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{num_patches, embed_dim}, DeviceType::CPU);

    const float* px = static_cast<const float*>(pixel_values->data());
    float* out = static_cast<float*>(output->data());

    auto weight = vit_dequant_to_fp32_cpu(patch_weight);
    const float* w_data = static_cast<const float*>(weight->data());

    std::vector<float> bias(embed_dim, 0.0f);
    if (patch_bias) {
        auto b = vit_dequant_to_fp32_cpu(patch_bias);
        std::memcpy(bias.data(), b->data(), embed_dim * sizeof(float));
    }

#pragma omp parallel for schedule(dynamic) if (num_patches > 4)
    for (int patch_idx = 0; patch_idx < num_patches; ++patch_idx) {
        int ph = patch_idx / num_patches_w;
        int pw = patch_idx % num_patches_w;
        float* patch_out = out + patch_idx * embed_dim;
        std::memcpy(patch_out, bias.data(), embed_dim * sizeof(float));
        for (int dy = 0; dy < ps; ++dy) {
            for (int dx = 0; dx < ps; ++dx) {
                int img_y = ph * ps + dy;
                int img_x = pw * ps + dx;
                for (int c = 0; c < 3; ++c) {
                    float pixel_val = px[c * img_size * img_size + img_y * img_size + img_x];
                    // GGUF stores conv weight as [KW, KH, IC, OC] in column-major.
                    // After shape reversal, logical shape is [OC, IC, KH, KW].
                    for (int d = 0; d < embed_dim; ++d) {
                        patch_out[d] +=
                            pixel_val * w_data[d * 3 * ps * ps + c * ps * ps + dy * ps + dx];
                    }
                }
            }
        }
    }
    return output;
}

// ---- vit_add_pos_embed_2d ----

TensorPtr vit_add_pos_embed_2d(const TensorPtr& patches, const VisionConfig& cfg,
                               const TensorPtr& pos_weight) {
    if (!pos_weight)
        return patches;

    int num_patches = static_cast<int>(patches->shape()[0]);
    int dim = cfg.embedding_length;
    float* h_data = static_cast<float*>(patches->data());

    auto pos = vit_dequant_to_fp32_cpu(pos_weight);
    const float* pd = static_cast<const float*>(pos->data());
    int grid_size = cfg.image_size / cfg.patch_size;
    int n_embd_buckets = cfg.n_embd_buckets;

#pragma omp parallel for schedule(static) if (num_patches > 4)
    for (int i = 0; i < num_patches; ++i) {
        int patch_h = i / grid_size;
        int patch_w = i % grid_size;
        int bucket_h = static_cast<int>(
            std::floor(static_cast<double>(n_embd_buckets) * patch_h / grid_size));
        int bucket_w = static_cast<int>(
            std::floor(static_cast<double>(n_embd_buckets) * patch_w / grid_size));
        int bucket_idx = bucket_h * n_embd_buckets + bucket_w;
        for (int d = 0; d < dim; ++d) {
            h_data[i * dim + d] += pd[bucket_idx * dim + d];
        }
    }
    return patches;
}

}  // namespace forge
