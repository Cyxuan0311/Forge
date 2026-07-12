#include "forge/vision_encoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>

#include "forge/logger.h"
#include "forge/model.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {

// Forward declaration
static TensorPtr dequant_to_fp32_cpu(const TensorPtr& w);

// Helper: dequantize a weight tensor to FP32 on CPU
static TensorPtr dequant_to_fp32_cpu(const TensorPtr& w) {
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

// Helper: move tensor to device
static TensorPtr to_device(const TensorPtr& t, DeviceType dev) {
    if (!t || t->device() == dev)
        return t;
    auto out = std::make_shared<Tensor>(t->dtype(), t->shape(), dev);
    out->copy_from(*t);
    return out;
}

bool VisionWeights::init(const WeightStore& store, const VisionConfig& config) {
    patch_embd_weight = store.get("v.patch_embd.weight");
    patch_embd_bias = store.get("v.patch_embd.bias");
    position_embd_weight = store.get("v.position_embd.weight");
    post_ln_weight = store.get("v.post_ln.weight");
    post_ln_bias = store.get("v.post_ln.bias");

    merger_ln1_weight = store.get("v.vit_merger.ln1.weight");
    merger_ln1_bias = store.get("v.vit_merger.ln1.bias");
    merger_attn_q_weight = store.get("v.vit_merger.attn_q.weight");
    merger_attn_q_bias = store.get("v.vit_merger.attn_q.bias");
    merger_attn_k_weight = store.get("v.vit_merger.attn_k.weight");
    merger_attn_k_bias = store.get("v.vit_merger.attn_k.bias");
    merger_attn_v_weight = store.get("v.vit_merger.attn_v.weight");
    merger_attn_v_bias = store.get("v.vit_merger.attn_v.bias");
    merger_attn_out_weight = store.get("v.vit_merger.attn_out.weight");
    merger_attn_out_bias = store.get("v.vit_merger.attn_out.bias");
    merger_ds_ln_weight = store.get("v.vit_merger.ds_ln.weight");
    merger_ds_ln_bias = store.get("v.vit_merger.ds_ln.bias");
    merger_ds_ffn_up_weight = store.get("v.vit_merger.ds_ffn_up.weight");
    merger_ds_ffn_up_bias = store.get("v.vit_merger.ds_ffn_up.bias");
    merger_ds_ffn_down_weight = store.get("v.vit_merger.ds_ffn_down.weight");
    merger_ds_ffn_down_bias = store.get("v.vit_merger.ds_ffn_down.bias");

    mm_input_norm_weight = store.get("mm.input_norm.weight");
    mm_input_norm_bias = store.get("mm.input_norm.bias");
    mm_up_weight = store.get("mm.up.weight");
    mm_up_bias = store.get("mm.up.bias");
    mm_down_weight = store.get("mm.down.weight");
    mm_down_bias = store.get("mm.down.bias");

    layers.resize(config.block_count);
    for (int i = 0; i < config.block_count; ++i) {
        std::string blk = "v.blk." + std::to_string(i);
        layers[i].ln1_weight = store.get(blk + ".ln1.weight");
        layers[i].ln1_bias = store.get(blk + ".ln1.bias");
        layers[i].ln2_weight = store.get(blk + ".ln2.weight");
        layers[i].ln2_bias = store.get(blk + ".ln2.bias");
        layers[i].attn_q_weight = store.get(blk + ".attn_q.weight");
        layers[i].attn_q_bias = store.get(blk + ".attn_q.bias");
        layers[i].attn_k_weight = store.get(blk + ".attn_k.weight");
        layers[i].attn_k_bias = store.get(blk + ".attn_k.bias");
        layers[i].attn_v_weight = store.get(blk + ".attn_v.weight");
        layers[i].attn_v_bias = store.get(blk + ".attn_v.bias");
        layers[i].attn_out_weight = store.get(blk + ".attn_out.weight");
        layers[i].attn_out_bias = store.get(blk + ".attn_out.bias");
        layers[i].ffn_up_weight = store.get(blk + ".ffn_up.weight");
        layers[i].ffn_up_bias = store.get(blk + ".ffn_up.bias");
        layers[i].ffn_down_weight = store.get(blk + ".ffn_down.weight");
        layers[i].ffn_down_bias = store.get(blk + ".ffn_down.bias");
    }

    if (!patch_embd_weight) {
        LOG_ERROR("VisionWeights: missing patch_embd_weight");
        return false;
    }
    return true;
}

bool VisionEncoder::init(const WeightStore& store, const VisionConfig& config) {
    config_ = config;
    if (!weights_.init(store, config)) {
        LOG_ERROR("VisionEncoder: failed to initialize weights");
        return false;
    }
    initialized_ = true;
    LOG_INFO("VisionEncoder: initialized with " + std::to_string(config.block_count) +
             " ViT blocks, image_size=" + std::to_string(config.image_size) +
             ", patch_size=" + std::to_string(config.patch_size));
    return true;
}

void VisionEncoder::prepare_weights(DeviceType dev) {
    if (weights_prepared_ && weights_device_ == dev)
        return;

    auto to_dev = [dev](const TensorPtr& t) -> TensorPtr {
        auto fp32 = dequant_to_fp32_cpu(t);
        return fp32 ? to_device(fp32, dev) : nullptr;
    };

    // Per-layer weights
    cached_layers_.resize(config_.block_count);
    for (int i = 0; i < config_.block_count; ++i) {
        const auto& src = weights_.layers[i];
        auto& dst = cached_layers_[i];
        dst.ln1_weight = to_dev(src.ln1_weight);
        dst.ln1_bias = to_dev(src.ln1_bias);
        dst.ln2_weight = to_dev(src.ln2_weight);
        dst.ln2_bias = to_dev(src.ln2_bias);
        dst.attn_q_weight = to_dev(src.attn_q_weight);
        dst.attn_q_bias = to_dev(src.attn_q_bias);
        dst.attn_k_weight = to_dev(src.attn_k_weight);
        dst.attn_k_bias = to_dev(src.attn_k_bias);
        dst.attn_v_weight = to_dev(src.attn_v_weight);
        dst.attn_v_bias = to_dev(src.attn_v_bias);
        dst.attn_out_weight = to_dev(src.attn_out_weight);
        dst.attn_out_bias = to_dev(src.attn_out_bias);
        dst.ffn_up_weight = to_dev(src.ffn_up_weight);
        dst.ffn_up_bias = to_dev(src.ffn_up_bias);
        dst.ffn_down_weight = to_dev(src.ffn_down_weight);
        dst.ffn_down_bias = to_dev(src.ffn_down_bias);
    }

    // Position embedding (always stay on CPU, used before hidden moves to CUDA)
    cached_pos_embd_ = dequant_to_fp32_cpu(weights_.position_embd_weight);

    // Post-LN (on target device)
    pst_ln_w_ = to_dev(weights_.post_ln_weight);
    pst_ln_b_ = to_dev(weights_.post_ln_bias);

    // Merger weights (on target device)
    m_ln1_w_ = to_dev(weights_.merger_ln1_weight);
    m_ln1_b_ = to_dev(weights_.merger_ln1_bias);
    m_q_w_ = to_dev(weights_.merger_attn_q_weight);
    m_q_b_ = to_dev(weights_.merger_attn_q_bias);
    m_k_w_ = to_dev(weights_.merger_attn_k_weight);
    m_k_b_ = to_dev(weights_.merger_attn_k_bias);
    m_v_w_ = to_dev(weights_.merger_attn_v_weight);
    m_v_b_ = to_dev(weights_.merger_attn_v_bias);
    m_o_w_ = to_dev(weights_.merger_attn_out_weight);
    m_o_b_ = to_dev(weights_.merger_attn_out_bias);
    m_ds_ln_w_ = to_dev(weights_.merger_ds_ln_weight);
    m_ds_ln_b_ = to_dev(weights_.merger_ds_ln_bias);
    m_ds_up_w_ = to_dev(weights_.merger_ds_ffn_up_weight);
    m_ds_up_b_ = to_dev(weights_.merger_ds_ffn_up_bias);
    m_ds_down_w_ = to_dev(weights_.merger_ds_ffn_down_weight);
    m_ds_down_b_ = to_dev(weights_.merger_ds_ffn_down_bias);

    // Projector weights (on target device)
    p_in_norm_w_ = to_dev(weights_.mm_input_norm_weight);
    p_in_norm_b_ = to_dev(weights_.mm_input_norm_bias);
    p_up_w_ = to_dev(weights_.mm_up_weight);
    p_up_b_ = to_dev(weights_.mm_up_bias);
    p_down_w_ = to_dev(weights_.mm_down_weight);
    p_down_b_ = to_dev(weights_.mm_down_bias);

    weights_device_ = dev;
    weights_prepared_ = true;
}

TensorPtr VisionEncoder::preprocess(const float* rgb_data, int width, int height,
                                    int channels) const {
    int img_size = config_.image_size;
    auto pixels = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{3, img_size, img_size}, DeviceType::CPU);
    float* out = static_cast<float*>(pixels->data());

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < img_size; ++y) {
            for (int x = 0; x < img_size; ++x) {
                float src_y = (y + 0.5f) * height / img_size - 0.5f;
                float src_x = (x + 0.5f) * width / img_size - 0.5f;
                int y0 = std::max(0, std::min(height - 1, static_cast<int>(src_y)));
                int x0 = std::max(0, std::min(width - 1, static_cast<int>(src_x)));
                int y1 = std::min(height - 1, y0 + 1);
                int x1 = std::min(width - 1, x0 + 1);
                float fy = src_y - y0;
                float fx = src_x - x0;
                float val = 0.0f;
                if (channels == 3) {
                    float v00 = rgb_data[(y0 * width + x0) * 3 + c];
                    float v01 = rgb_data[(y0 * width + x1) * 3 + c];
                    float v10 = rgb_data[(y1 * width + x0) * 3 + c];
                    float v11 = rgb_data[(y1 * width + x1) * 3 + c];
                    val = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy +
                          v11 * fx * fy;
                }
                val = val / 255.0f;
                val = (val - config_.image_mean[c]) / config_.image_std[c];
                out[c * img_size * img_size + y * img_size + x] = val;
            }
        }
    }
    return pixels;
}

TensorPtr VisionEncoder::patch_embedding(const TensorPtr& pixel_values) {
    int img_size = config_.image_size;
    int ps = config_.patch_size;
    int embed_dim = config_.embedding_length;
    int num_patches_h = img_size / ps;
    int num_patches_w = img_size / ps;
    int num_patches = num_patches_h * num_patches_w;

    auto output = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{num_patches, embed_dim}, DeviceType::CPU);

    const float* px = static_cast<const float*>(pixel_values->data());
    float* out = static_cast<float*>(output->data());

    auto weight = dequant_to_fp32_cpu(weights_.patch_embd_weight);
    const float* w_data = static_cast<const float*>(weight->data());

    std::vector<float> bias(embed_dim, 0.0f);
    if (weights_.patch_embd_bias) {
        auto b = dequant_to_fp32_cpu(weights_.patch_embd_bias);
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
                    // After shape reversal, logical shape is [OC, IC, KH, KW] = [1152, 3, 14, 14].
                    // Data layout in memory (column-major equivalent to row-major of reversed
                    // shape): w_data[d_out * (IC*KH*KW) + c * (KH*KW) + dy * KW + dx]
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

TensorPtr VisionEncoder::forward_vit_block(const TensorPtr& hidden, const ViTLayerWeights& lw,
                                           int num_heads, int head_dim) {
    int num_patches = static_cast<int>(hidden->shape()[0]);

    // LayerNorm 1
    auto normed = ops::layer_norm(hidden, lw.ln1_weight, lw.ln1_bias, config_.layer_norm_epsilon);

    // QKV projections
    auto q = ops::matmul_transB(normed, lw.attn_q_weight, lw.attn_q_bias);
    auto k = ops::matmul_transB(normed, lw.attn_k_weight, lw.attn_k_bias);
    auto v = ops::matmul_transB(normed, lw.attn_v_weight, lw.attn_v_bias);

    // Self-attention (non-causal for ViT)
    auto attn_out =
        ops::scaled_dot_product_attention_2d(q, k, v, num_patches, num_heads, head_dim, false);

    // Output projection
    auto attn_proj = ops::matmul_transB(attn_out, lw.attn_out_weight, lw.attn_out_bias);

    // Residual
    auto res1 = ops::add(hidden, attn_proj);

    // LayerNorm 2
    auto normed2 = ops::layer_norm(res1, lw.ln2_weight, lw.ln2_bias, config_.layer_norm_epsilon);

    // FFN up + GELU (tanh-based)
    auto ffn_up = ops::matmul_transB(normed2, lw.ffn_up_weight, lw.ffn_up_bias);
    ffn_up = ops::gelu_tanh(ffn_up);

    // FFN down
    auto ffn_down = ops::matmul_transB(ffn_up, lw.ffn_down_weight, lw.ffn_down_bias);

    // Residual
    return ops::add(res1, ffn_down);
}

TensorPtr VisionEncoder::merge_tokens(const TensorPtr& hidden, DeviceType dev) {
    int num_patches = static_cast<int>(hidden->shape()[0]);
    int dim = config_.embedding_length;

    // Step 1: Merger self-attention (windowed attention with 2x2 windows)
    auto residual = hidden;
    auto normed = hidden;

    if (m_ln1_w_) {
        auto merger_normed =
            ops::layer_norm(hidden, m_ln1_w_, m_ln1_b_, config_.layer_norm_epsilon);

        int num_heads = config_.head_count;
        int head_dim = dim / num_heads;
        int grid_size = static_cast<int>(std::sqrt(num_patches));

        auto q = ops::matmul_transB(merger_normed, m_q_w_, m_q_b_);
        auto k = ops::matmul_transB(merger_normed, m_k_w_, m_k_b_);
        auto v = ops::matmul_transB(merger_normed, m_v_w_, m_v_b_);

        // Window attention: reorder tokens into 2x2 windows, apply block-diagonal mask
        auto q_cpu = q;
        auto k_cpu = k;
        auto v_cpu = v;
        if (dev == DeviceType::CUDA) {
            q_cpu = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
            k_cpu = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
            v_cpu = std::make_shared<Tensor>(DataType::FP32, v->shape(), DeviceType::CPU);
            q_cpu->copy_from(*q);
            k_cpu->copy_from(*k);
            v_cpu->copy_from(*v);
        }

        // Build window reorder indices
        std::vector<int> window_idx(num_patches);
        std::vector<int> inv_window_idx(num_patches);
        int idx = 0;
        for (int gh = 0; gh < grid_size / 2; ++gh) {
            for (int gw = 0; gw < grid_size / 2; ++gw) {
                for (int dh = 0; dh < 2; ++dh) {
                    for (int dw = 0; dw < 2; ++dw) {
                        int orig_idx = (gh * 2 + dh) * grid_size + (gw * 2 + dw);
                        window_idx[idx] = orig_idx;
                        inv_window_idx[orig_idx] = idx;
                        idx++;
                    }
                }
            }
        }

        // Reorder Q, K, V according to window indices
        auto q_reordered =
            std::make_shared<Tensor>(DataType::FP32, q_cpu->shape(), DeviceType::CPU);
        auto k_reordered =
            std::make_shared<Tensor>(DataType::FP32, k_cpu->shape(), DeviceType::CPU);
        auto v_reordered =
            std::make_shared<Tensor>(DataType::FP32, v_cpu->shape(), DeviceType::CPU);
        const float* q_data = static_cast<const float*>(q_cpu->data());
        const float* k_data = static_cast<const float*>(k_cpu->data());
        const float* v_data = static_cast<const float*>(v_cpu->data());
        float* qr_data = static_cast<float*>(q_reordered->data());
        float* kr_data = static_cast<float*>(k_reordered->data());
        float* vr_data = static_cast<float*>(v_reordered->data());

#pragma omp parallel for schedule(static)
        for (int i = 0; i < num_patches; ++i) {
            int src = window_idx[i];
            std::memcpy(qr_data + i * num_heads * head_dim, q_data + src * num_heads * head_dim,
                        num_heads * head_dim * sizeof(float));
            std::memcpy(kr_data + i * num_heads * head_dim, k_data + src * num_heads * head_dim,
                        num_heads * head_dim * sizeof(float));
            std::memcpy(vr_data + i * num_heads * head_dim, v_data + src * num_heads * head_dim,
                        num_heads * head_dim * sizeof(float));
        }

        // Build block-diagonal mask
        auto mask = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{num_patches, num_patches}, DeviceType::CPU);
        float* mask_data = static_cast<float*>(mask->data());
        std::fill(mask_data, mask_data + num_patches * num_patches, -INFINITY);
        for (int w = 0; w < num_patches; w += 4) {
            for (int i = w; i < w + 4 && i < num_patches; ++i) {
                for (int j = w; j < w + 4 && j < num_patches; ++j) {
                    mask_data[i * num_patches + j] = 0.0f;
                }
            }
        }

        // Apply masked attention
        auto attn_out = ops::scaled_dot_product_attention_2d_masked(
            q_reordered, k_reordered, v_reordered, num_patches, num_heads, head_dim, mask);

        // Inverse reorder: restore original token order
        auto attn_restored =
            std::make_shared<Tensor>(DataType::FP32, attn_out->shape(), DeviceType::CPU);
        const float* ao_data = static_cast<const float*>(attn_out->data());
        float* ar_data = static_cast<float*>(attn_restored->data());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < num_patches; ++i) {
            int dst = window_idx[i];
            std::memcpy(ar_data + dst * num_heads * head_dim, ao_data + i * num_heads * head_dim,
                        num_heads * head_dim * sizeof(float));
        }

        // Output projection (using cached weight m_o_w_)
        if (dev == DeviceType::CUDA) {
            auto tmp =
                std::make_shared<Tensor>(DataType::FP32, attn_restored->shape(), DeviceType::CUDA);
            tmp->copy_from(*attn_restored);
            attn_restored = tmp;
        }
        auto attn_proj = ops::matmul_transB(attn_restored, m_o_w_, m_o_b_);

        // Residual connection
        normed = ops::add(attn_proj, residual);
    }

    // Step 2: 2x2 spatial downsample with mean residual
    auto normed_cpu = normed;
    if (dev == DeviceType::CUDA) {
        normed_cpu = std::make_shared<Tensor>(DataType::FP32, normed->shape(), DeviceType::CPU);
        normed_cpu->copy_from(*normed);
    }

    int grid_size = static_cast<int>(std::sqrt(num_patches));
    int new_grid = grid_size / 2;
    int new_num_patches = new_grid * new_grid;
    int concat_dim = dim * 4;

    auto concat_tokens = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{new_num_patches, concat_dim}, DeviceType::CPU);
    auto mean_res = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{new_num_patches, dim}, DeviceType::CPU);
    float* concat_data = static_cast<float*>(concat_tokens->data());
    float* mean_data = static_cast<float*>(mean_res->data());
    const float* norm_data = static_cast<const float*>(normed_cpu->data());

#pragma omp parallel for schedule(dynamic) if (new_num_patches > 4)
    for (int new_idx = 0; new_idx < new_num_patches; ++new_idx) {
        int gh = new_idx / new_grid;
        int gw = new_idx % new_grid;
        float* concat_ptr = concat_data + new_idx * concat_dim;
        float* mean_ptr = mean_data + new_idx * dim;
        std::fill(mean_ptr, mean_ptr + dim, 0.0f);
        for (int dh = 0; dh < 2; ++dh) {
            for (int dw = 0; dw < 2; ++dw) {
                int orig_idx = (gh * 2 + dh) * grid_size + (gw * 2 + dw);
                const float* src = norm_data + orig_idx * dim;
                int offset = (dh * 2 + dw) * dim;
                std::memcpy(concat_ptr + offset, src, dim * sizeof(float));
                for (int d = 0; d < dim; ++d) {
                    mean_ptr[d] += src[d];
                }
            }
        }
        for (int d = 0; d < dim; ++d) {
            mean_ptr[d] *= 0.25f;
        }
    }

    // DS FFN: concat(4*dim) -> LayerNorm -> FFN -> dim, then add mean_res
    if (m_ds_ln_w_) {
        concat_tokens = to_device(concat_tokens, dev);
        mean_res = to_device(mean_res, dev);

        auto ds_normed =
            ops::layer_norm(concat_tokens, m_ds_ln_w_, m_ds_ln_b_, config_.layer_norm_epsilon);

        auto ffn_up = ops::matmul_transB(ds_normed, m_ds_up_w_, m_ds_up_b_);
        ffn_up = ops::gelu_tanh(ffn_up);

        auto ds_out = ops::matmul_transB(ffn_up, m_ds_down_w_, m_ds_down_b_);

        return ops::add(ds_out, mean_res);
    }

    return concat_tokens;
}

TensorPtr VisionEncoder::project(const TensorPtr& hidden, DeviceType dev) {
    int num_tokens = static_cast<int>(hidden->shape()[0]);
    int dim = static_cast<int>(hidden->shape()[1]);

    // Final merger: 2x2 spatial downsample + LayerNorm + FFN(GELU_ERF)
    auto hidden_cpu = hidden;
    if (dev == DeviceType::CUDA) {
        hidden_cpu = std::make_shared<Tensor>(DataType::FP32, hidden->shape(), DeviceType::CPU);
        hidden_cpu->copy_from(*hidden);
    }

    int grid_size = static_cast<int>(std::sqrt(num_tokens));
    int new_grid = grid_size / 2;
    int new_num = new_grid * new_grid;
    int concat_dim = dim * 4;

    auto concat_tokens = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{new_num, concat_dim}, DeviceType::CPU);
    float* concat_data = static_cast<float*>(concat_tokens->data());
    const float* h_data = static_cast<const float*>(hidden_cpu->data());

#pragma omp parallel for schedule(dynamic) if (new_num > 4)
    for (int idx = 0; idx < new_num; ++idx) {
        int gh = idx / new_grid;
        int gw = idx % new_grid;
        float* out_ptr = concat_data + idx * concat_dim;
        for (int dh = 0; dh < 2; ++dh) {
            for (int dw = 0; dw < 2; ++dw) {
                int orig_idx = (gh * 2 + dh) * grid_size + (gw * 2 + dw);
                const float* src = h_data + orig_idx * dim;
                int offset = (dh * 2 + dw) * dim;
                std::memcpy(out_ptr + offset, src, dim * sizeof(float));
            }
        }
    }

    // LayerNorm (using cached weights)
    auto h = to_device(concat_tokens, dev);
    auto normed = ops::layer_norm(h, p_in_norm_w_, p_in_norm_b_, config_.layer_norm_epsilon);

    // FFN with GELU (erf-based)
    auto up = ops::matmul_transB(normed, p_up_w_, p_up_b_);
    up = ops::gelu(up);

    return ops::matmul_transB(up, p_down_w_, p_down_b_);
}

std::vector<float> VisionEncoder::encode(const float* rgb_data, int width, int height,
                                         int channels) {
    if (!initialized_) {
        LOG_ERROR("VisionEncoder: not initialized");
        return {};
    }

    PERF_SCOPE("vision.encode");

    // Step 1: Preprocess
    TensorPtr pixels;
    {
        PERF_SCOPE("vision.preprocess");
        pixels = preprocess(rgb_data, width, height, channels);
    }

    // Step 2: Patch embedding
    TensorPtr hidden;
    {
        PERF_SCOPE("vision.patch_embd");
        hidden = patch_embedding(pixels);
    }

    // Step 3: Add position embedding (CPU, dequant once per encode, negligible overhead)
    {
        PERF_SCOPE("vision.pos_embd");
        int num_patches = static_cast<int>(hidden->shape()[0]);
        int dim = config_.embedding_length;
        float* h_data = static_cast<float*>(hidden->data());

        if (weights_.position_embd_weight) {
            auto pos = dequant_to_fp32_cpu(weights_.position_embd_weight);
            const float* pd = static_cast<const float*>(pos->data());
            int grid_size = config_.image_size / config_.patch_size;
            int n_embd_buckets = 70;
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
        }
    }

    // Move to CUDA for ViT blocks if available
    DeviceType dev = DeviceType::CPU;
#ifdef USE_CUDA
    try {
        hidden = to_device(hidden, DeviceType::CUDA);
        dev = DeviceType::CUDA;
        LOG_DEBUG("Vision: moved hidden to CUDA for ViT blocks");
    } catch (...) {
        LOG_DEBUG("Vision: CUDA not available, using CPU for ViT blocks");
        dev = DeviceType::CPU;
    }
#else
    dev = DeviceType::CPU;
#endif

    // Prepare cached weights on target device (one-time dequant + copy)
    {
        PERF_SCOPE("vision.prepare_weights");
        prepare_weights(dev);
    }

    int num_heads = config_.head_count;
    int head_dim = config_.embedding_length / num_heads;

    // Step 4-6: ViT blocks, merge, project
    int insert_lid = config_.insert_layer_id;
    if (insert_lid < 0 || insert_lid >= config_.block_count) {
        // Path 1: all ViT blocks -> merge -> project
        {
            PERF_SCOPE("vision.vit_blocks");
            for (int i = 0; i < config_.block_count; ++i) {
                hidden = forward_vit_block(hidden, cached_layers_[i], num_heads, head_dim);
            }
        }

        TensorPtr merged;
        {
            PERF_SCOPE("vision.merge_tokens");
            merged = merge_tokens(hidden, dev);
        }

        TensorPtr projected;
        {
            PERF_SCOPE("vision.project");
            projected = project(merged, dev);
        }

        // Return as flat vector
        auto proj_cpu = projected;
        if (proj_cpu->device() == DeviceType::CUDA) {
            proj_cpu =
                std::make_shared<Tensor>(DataType::FP32, projected->shape(), DeviceType::CPU);
            proj_cpu->copy_from(*projected);
        }
        int num_tokens = static_cast<int>(proj_cpu->shape()[0]);
        int proj_dim = static_cast<int>(proj_cpu->shape()[1]);
        const float* proj_data = static_cast<const float*>(proj_cpu->data());
        return std::vector<float>(proj_data, proj_data + num_tokens * proj_dim);
    }

    // Path 2: with merger insertion (MiniCPM-V 4.6)
    {
        PERF_SCOPE("vision.vit_phase1");
        for (int i = 0; i <= insert_lid; ++i) {
            hidden = forward_vit_block(hidden, cached_layers_[i], num_heads, head_dim);
        }
    }

    TensorPtr merged;
    {
        PERF_SCOPE("vision.merge_tokens");
        merged = merge_tokens(hidden, dev);
    }

    {
        PERF_SCOPE("vision.vit_phase2");
        for (int i = insert_lid + 1; i < config_.block_count; ++i) {
            merged = forward_vit_block(merged, cached_layers_[i], num_heads, head_dim);
        }
    }

    // Post LayerNorm (using cached weights)
    if (pst_ln_w_) {
        PERF_SCOPE("vision.post_ln");
        merged = ops::layer_norm(merged, pst_ln_w_, pst_ln_b_, config_.layer_norm_epsilon);
    }

    TensorPtr projected;
    {
        PERF_SCOPE("vision.project");
        projected = project(merged, dev);
    }

    // Return as flat vector (move to CPU first)
    auto proj_cpu = projected;
    if (proj_cpu->device() == DeviceType::CUDA) {
        proj_cpu = std::make_shared<Tensor>(DataType::FP32, projected->shape(), DeviceType::CPU);
        proj_cpu->copy_from(*projected);
    }

    int num_tokens = static_cast<int>(proj_cpu->shape()[0]);
    int proj_dim = static_cast<int>(proj_cpu->shape()[1]);
    const float* proj_data = static_cast<const float*>(proj_cpu->data());

    return std::vector<float>(proj_data, proj_data + num_tokens * proj_dim);
}

}  // namespace forge
