// ============================================================================
// Qwen3VL Vision Encoder — Qwen3-VL ViT + DeepStack + MLP Projector
// ============================================================================
// Implements the full Qwen3-VL vision forward pass:
//   1. Two-part conv2d patch embedding (split weights)
//   2. Spatial merge (2x2 interleave) after patch embedding
//   3. Learned position embedding (additive, spatially merged)
//   4. ViT blocks with fused QKV + M-RoPE (4-section multimodal rotary)
//   5. DeepStack side features at specific layers
//   6. Post LayerNorm
//   7. Final 2x2 spatial merge + mm projector (mm.0 + mm.2 GELU MLP)
// ============================================================================
#include "forge/vision_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "forge/logger.h"
#include "forge/model.h"
#include "forge/operators.h"
#include "cpu/common/thread_pool.h"
#include "forge/perf_profiler.h"
#include "forge/vit_shared.h"
#include "forge/vision_registry.h"

#ifdef USE_CUDA
#    include "forge/cuda_kernels.h"
#endif

namespace forge {

// ---- helpers ----

static TensorPtr dequant_to_fp32_cpu(const TensorPtr& w) {
    return vit_dequant_to_fp32_cpu(w);
}

static TensorPtr to_device(const TensorPtr& t, DeviceType dev) {
    return vit_to_device(t, dev);
}

// ---- init ----

bool Qwen3VLViTEncoder::init(const WeightStore& store, const VisionConfig& config) {
    config_ = config;
    const auto& mapping = VisionWeightMapper::get_mapping(name());

    // Core weights
    weights_.patch_embd_weight = store.get(mapping.patch_embd_weight);
    weights_.patch_embd_bias = store.get(mapping.patch_embd_bias);
    weights_.position_embd_weight = store.get(mapping.position_embd_weight);
    weights_.post_ln_weight = store.get(mapping.post_ln_weight);
    weights_.post_ln_bias = store.get(mapping.post_ln_bias);

    // Qwen3-VL second patch embedding weight
    weights_.patch_embd_weight_1 = store.get(mapping.patch_embd_weight + ".1");

    // mm projector (mm.0 / mm.2)
    const std::string& pp = mapping.projector_prefix;
    weights_.mm0_weight = store.get(pp + ".0.weight");
    weights_.mm0_bias = store.get(pp + ".0.bias");
    weights_.mm2_weight = store.get(pp + ".2.weight");
    weights_.mm2_bias = store.get(pp + ".2.bias");

    // Per-layer weights
    weights_.layers.resize(config.block_count);
    for (int i = 0; i < config.block_count; ++i) {
        std::string blk = VisionWeightMapper::format_layer_prefix(mapping.block_prefix, i);
        auto& lw = weights_.layers[i];
        lw.ln1_weight = store.get(blk + mapping.blk_ln1_w);
        lw.ln1_bias = store.get(blk + mapping.blk_ln1_b);
        lw.ln2_weight = store.get(blk + mapping.blk_ln2_w);
        lw.ln2_bias = store.get(blk + mapping.blk_ln2_b);

        // Fused QKV (Qwen3-VL uses attn_qkv, not separate q/k/v)
        lw.attn_qkv_weight = store.get(blk + ".attn_qkv.weight");
        lw.attn_qkv_bias = store.get(blk + ".attn_qkv.bias");
        if (!lw.attn_qkv_weight) {
            // Fallback: try separate Q/K/V
            lw.attn_q_weight = store.get(blk + mapping.blk_q_w);
            lw.attn_q_bias = store.get(blk + mapping.blk_q_b);
            lw.attn_k_weight = store.get(blk + mapping.blk_k_w);
            lw.attn_k_bias = store.get(blk + mapping.blk_k_b);
            lw.attn_v_weight = store.get(blk + mapping.blk_v_w);
            lw.attn_v_bias = store.get(blk + mapping.blk_v_b);
        }

        lw.attn_out_weight = store.get(blk + mapping.blk_o_w);
        lw.attn_out_bias = store.get(blk + mapping.blk_o_b);
        lw.ffn_up_weight = store.get(blk + mapping.blk_ffn_up_w);
        lw.ffn_up_bias = store.get(blk + mapping.blk_ffn_up_b);
        lw.ffn_down_weight = store.get(blk + mapping.blk_ffn_down_w);
        lw.ffn_down_bias = store.get(blk + mapping.blk_ffn_down_b);
    }

    // Deepstack weights
    for (int layer_idx : config.deepstack_layers) {
        VisionWeights::DeepstackWeights dw;
        std::string ds_prefix = "v.deepstack." + std::to_string(layer_idx);
        dw.norm_weight = store.get(ds_prefix + ".norm.weight");
        dw.norm_bias = store.get(ds_prefix + ".norm.bias");
        dw.fc1_weight = store.get(ds_prefix + ".fc1.weight");
        dw.fc1_bias = store.get(ds_prefix + ".fc1.bias");
        dw.fc2_weight = store.get(ds_prefix + ".fc2.weight");
        dw.fc2_bias = store.get(ds_prefix + ".fc2.bias");
        weights_.deepstack_weights[layer_idx] = std::move(dw);
    }

    if (!weights_.patch_embd_weight) {
        LOG_ERROR("Qwen3VLViTEncoder: missing patch_embd_weight");
        return false;
    }
    if (!weights_.position_embd_weight) {
        LOG_WARN("Qwen3VLViTEncoder: no position_embd_weight (will skip pos embd)");
    }

    initialized_ = true;
    LOG_INFO("Qwen3VLViTEncoder: initialized with " + std::to_string(config.block_count) +
             " ViT blocks, image_size=" + std::to_string(config.image_size) +
             ", patch_size=" + std::to_string(config.patch_size) +
             ", deepstack_layers=" + std::to_string(config.deepstack_layers.size()));
    return true;
}

// ---- prepare_weights ----

// FP32 staging policy
// --------------------
// The mmproj stores the tower in FP16 (792 MB for Qwen3-VL-4B), but the op chain
// here (matmul_transB, layer_norm, gelu) is written against FP32, so the naive
// "dequantize everything once" costs 1583 MB of VRAM -- permanently resident,
// and more than a 6 GB card can spare next to the KV cache. Measured on a
// 6 GB RTX 4050: a default-context server had 4201 MiB after startup and jumped
// to 5874 MiB on the first image request, leaving 258 MiB and pushing the whole
// process into allocator thrash.
//
// So: keep the FP16 masters resident (they ARE the file's precision, so this is
// lossless), plus the FP32 tensors the GGUF already stores as FP32 (norms and
// biases, ~1 MB total), and materialize FP32 copies of the big F16 weights into
// one reusable scratch buffer right before the block that needs them. The tower
// runs its blocks in order and never revisits one, so a single slot is enough:
// ~50 MB instead of 1583 MB, at the cost of one device-side F16->FP32 conversion
// per block (~24 x 0.1 ms per image).
void Qwen3VLViTEncoder::prepare_weights(DeviceType dev) {
    if (weights_prepared_ && weights_device_ == dev)
        return;

    // Resident: the FP16 masters themselves, plus everything the GGUF already
    // stores as FP32. No FP32 copies of the F16 weights are made here.
    auto stage_native = [dev](const TensorPtr& t) -> TensorPtr {
        return t ? to_device(t, dev) : nullptr;
    };

    cached_layers_.resize(config_.block_count);
    for (int i = 0; i < config_.block_count; ++i) {
        const auto& src = weights_.layers[i];
        auto& dst = cached_layers_[i];
        // FP16 masters (converted to FP32 on demand by load_block).
        dst.attn_qkv_weight = stage_native(src.attn_qkv_weight);
        dst.attn_out_weight = stage_native(src.attn_out_weight);
        dst.ffn_up_weight = stage_native(src.ffn_up_weight);
        dst.ffn_down_weight = stage_native(src.ffn_down_weight);
        // Separate q/k/v weights, for towers that do not use the fused layout.
        dst.attn_q_weight = stage_native(src.attn_q_weight);
        dst.attn_k_weight = stage_native(src.attn_k_weight);
        dst.attn_v_weight = stage_native(src.attn_v_weight);
        // FP32 in the GGUF already: norms and biases are tiny, keep them as-is.
        dst.ln1_weight = stage_native(src.ln1_weight);
        dst.ln1_bias = stage_native(src.ln1_bias);
        dst.ln2_weight = stage_native(src.ln2_weight);
        dst.ln2_bias = stage_native(src.ln2_bias);
        dst.attn_qkv_bias = stage_native(src.attn_qkv_bias);
        dst.attn_q_bias = stage_native(src.attn_q_bias);
        dst.attn_k_bias = stage_native(src.attn_k_bias);
        dst.attn_v_bias = stage_native(src.attn_v_bias);
        dst.attn_out_bias = stage_native(src.attn_out_bias);
        dst.ffn_up_bias = stage_native(src.ffn_up_bias);
        dst.ffn_down_bias = stage_native(src.ffn_down_bias);
    }

    // Patch embedding / position embedding stay on the host: both are consumed
    // by CPU kernels before the hidden state moves to the device.
    cached_patch_w0_ = dequant_to_fp32_cpu(weights_.patch_embd_weight);
    cached_patch_w1_ = dequant_to_fp32_cpu(weights_.patch_embd_weight_1);
    cached_patch_b_ = dequant_to_fp32_cpu(weights_.patch_embd_bias);
    cached_pos_embd_ = dequant_to_fp32_cpu(weights_.position_embd_weight);

    // Post-LN and the projector biases are FP32 already; the two projector
    // matrices stay FP16 masters and are materialized per image in project().
    pst_ln_w_ = stage_native(weights_.post_ln_weight);
    pst_ln_b_ = stage_native(weights_.post_ln_bias);
    mm0_b_ = stage_native(weights_.mm0_bias);
    mm2_b_ = stage_native(weights_.mm2_bias);
    mm0_master_ = stage_native(weights_.mm0_weight);
    mm2_master_ = stage_native(weights_.mm2_weight);

    // DeepStack side features are computed but never injected (encode() runs the
    // ViT blocks only), so staging its FP32 copies was 312 MB of pure waste on a
    // 4B tower. forward_deepstack() materializes its own weights on demand.

    weights_device_ = dev;
    weights_prepared_ = true;
    loaded_block_ = -1;
}

// Materialize `master` (FP16, resident) as an FP32 tensor on `dev`, reusing
// `scratch` when its shape already matches so a 24-block tower converts into the
// same ~50 MB of device memory rather than allocating per block.
TensorPtr Qwen3VLViTEncoder::fp32_from_master(const TensorPtr& master, TensorPtr& scratch,
                                              DeviceType dev) {
    if (!master)
        return nullptr;
    if (master->dtype() == DataType::FP32)
        return to_device(master, dev);

    if (!scratch || scratch->shape() != master->shape())
        scratch = std::make_shared<Tensor>(DataType::FP32, master->shape(), dev);

    if (dev == DeviceType::CUDA && master->device() == DeviceType::CUDA) {
        // Device-side conversion: the master is already resident, so this is a
        // pure D2D pass (~25 MB read + 50 MB write per block).
#ifdef USE_CUDA
        const int64_t rows = master->shape()[0];
        const int64_t cols = master->numel() / std::max<int64_t>(rows, 1);
        cuda::launch_dequant_f16_matrix(master->data(), static_cast<float*>(scratch->data()),
                                        static_cast<int>(rows), static_cast<int>(cols));
#else
        scratch->copy_from(*master);
#endif
    } else {
        auto fp32 = dequant_to_fp32_cpu(master);
        scratch->copy_from(*fp32);
    }
    return scratch;
}

// Fill `out` with block `i`'s weights, converting its FP16 masters into the
// shared scratch. Every block has the same shapes, so the scratch is allocated
// once and refilled; blocks are visited in ascending order, so tracking the
// last-loaded index is all the bookkeeping needed.
const ViTLayerWeights& Qwen3VLViTEncoder::load_block(int i, DeviceType dev) {
    if (loaded_block_ == i)
        return active_layer_;

    auto& src = cached_layers_[i];
    auto& dst = active_layer_;

    dst.ln1_weight = src.ln1_weight;
    dst.ln1_bias = src.ln1_bias;
    dst.ln2_weight = src.ln2_weight;
    dst.ln2_bias = src.ln2_bias;
    dst.attn_qkv_bias = src.attn_qkv_bias;
    dst.attn_q_bias = src.attn_q_bias;
    dst.attn_k_bias = src.attn_k_bias;
    dst.attn_v_bias = src.attn_v_bias;
    dst.attn_out_bias = src.attn_out_bias;
    dst.ffn_up_bias = src.ffn_up_bias;
    dst.ffn_down_bias = src.ffn_down_bias;

    dst.attn_qkv_weight = fp32_from_master(src.attn_qkv_weight, scratch_qkv_, dev);
    dst.attn_out_weight = fp32_from_master(src.attn_out_weight, scratch_out_, dev);
    dst.ffn_up_weight = fp32_from_master(src.ffn_up_weight, scratch_ffn_up_, dev);
    dst.ffn_down_weight = fp32_from_master(src.ffn_down_weight, scratch_ffn_down_, dev);
    dst.attn_q_weight = fp32_from_master(src.attn_q_weight, scratch_q_, dev);
    dst.attn_k_weight = fp32_from_master(src.attn_k_weight, scratch_k_, dev);
    dst.attn_v_weight = fp32_from_master(src.attn_v_weight, scratch_v_, dev);

    loaded_block_ = i;
    return dst;
}

// ---- preprocess ----

TensorPtr Qwen3VLViTEncoder::preprocess(const float* rgb_data, int width, int height,
                                         int channels) const {
    int img_size = config_.image_size;
    auto pixels = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{3, img_size, img_size}, DeviceType::CPU);
    float* out = static_cast<float*>(pixels->data());

    // Qwen3-VL normalization: ImageNet mean/std
    const float mean[3] = {0.48145466f, 0.4578125f, 0.40821073f};
    const float std_[3] = {0.26862954f, 0.26130258f, 0.27577711f};

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
                    val = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                          v10 * (1 - fx) * fy + v11 * fx * fy;
                }
                val = val / 255.0f;
                val = (val - mean[c]) / std_[c];
                out[c * img_size * img_size + y * img_size + x] = val;
            }
        }
    }
    return pixels;
}

// ---- patch_embedding ----
// Two full-output-channel conv2d: conv0(pixel_values) + conv1(pixel_values) + bias
// Matches llama.cpp's build_inp_with_temporal_merge for single image.
// Both GGUF weights have shape [KW, KH, IC, OC] = [16, 16, 3, 1024],
// producing full OC=1024 output channels each. For single images, both
// convs process the same input and their outputs are summed (temporal merge
// with a single frame). For video, conv0 processes frame0, conv1 frame1.

TensorPtr Qwen3VLViTEncoder::patch_embedding(const TensorPtr& pixel_values) {
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

    // Dequantize conv weights — both have full OC=embed_dim
    const float* w0_data = nullptr;
    if (cached_patch_w0_) {
        w0_data = static_cast<const float*>(cached_patch_w0_->data());
    }
    const float* w1_data = nullptr;
    if (cached_patch_w1_) {
        w1_data = static_cast<const float*>(cached_patch_w1_->data());
    }

    // Bias
    std::vector<float> bias(embed_dim, 0.0f);
    if (cached_patch_b_) {
        std::memcpy(bias.data(), cached_patch_b_->data(), embed_dim * sizeof(float));
    }

    // Weight layout: [KW, KH, IC, OC] = [ps, ps, 3, embed_dim] in ggml notation.
    // Element [kw, kh, ic, oc] at offset: kw + kh*ps + ic*ps*ps + oc*ps*ps*3
    // Conv2d with stride=ps: each patch at grid (ph, pw) covers pixels
    //   [ph*ps..ph*ps+ps-1, pw*ps..pw*ps+ps-1]

    forge::cpu::parallel_for_chunks_dynamic(num_patches, 1, [&](int p0, int p1) {
        for (int patch_idx = p0; patch_idx < p1; ++patch_idx) {
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
                        int kv_idx = c * ps * ps + dy * ps + dx;
                        if (w0_data) {
                            for (int d = 0; d < embed_dim; ++d) {
                                patch_out[d] += pixel_val * w0_data[d * 3 * ps * ps + kv_idx];
                            }
                        }
                        if (w1_data) {
                            for (int d = 0; d < embed_dim; ++d) {
                                patch_out[d] += pixel_val * w1_data[d * 3 * ps * ps + kv_idx];
                            }
                        }
                    }
                }
            }
        }
    });
    return output;
}

// ---- spatial_merge ----
// Reorder patches from row-major to 2x2-block-major order.
// This matches llama.cpp's spatial merge in qwen2vl/qwen3vl:
//   permute(1,2,0,3) → cont_4d(n_embd*2, nx/2, ny, 1)
//   → reshape_4d(n_embd*2, nx/2, 2, ny/2) → permute(0,2,1,3)
//   → cont_3d(n_embd, nx*ny, 1)
//
// The result is that every 4 consecutive tokens form a 2x2 spatial block,
// which is required for the projection step (reshape to [n_embd*4, n_patches/4]).
// Feature values are unchanged — only token ordering is permuted.
//
// Token mapping: for original patch at grid position (x, y):
//   new_index = (y//2) * (nx//2) * 4 + (x//2) * 4 + (x%2)*2 + (y%2)

TensorPtr Qwen3VLViTEncoder::spatial_merge(const TensorPtr& patches, int nx, int ny) {
    int num_patches = nx * ny;
    int dim = config_.embedding_length;
    int half_nx = nx / 2;
    int half_ny = ny / 2;

    auto output = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{num_patches, dim}, DeviceType::CPU);
    const float* src = static_cast<const float*>(patches->data());
    float* dst = static_cast<float*>(output->data());

    // Build the permutation: for each new token index, which original index?
    // new_idx = block_y * half_nx * 4 + block_x * 4 + (x%2)*2 + (y%2)
    // orig_idx = x * ny + y  (row-major with x as column, y as row)
    forge::cpu::parallel_for_chunks(num_patches, 1, [&](int n0, int n1) {
        for (int new_idx = n0; new_idx < n1; ++new_idx) {
            int block_y = new_idx / (half_nx * 4);
            int rem = new_idx % (half_nx * 4);
            int block_x = rem / 4;
            int within = rem % 4;
            int x = block_x * 2 + (within / 2);
            int y = block_y * 2 + (within % 2);
            int orig_idx = x * ny + y;
            std::memcpy(dst + new_idx * dim, src + orig_idx * dim, dim * sizeof(float));
        }
    });
    return output;
}

// ---- add_position_embedding ----
// Learned position embedding: pos_embd has shape [dim, max_patches]
// For Qwen3-VL, it needs to be spatially merged the same way as patches,
// then added. For simplicity (matching spatial_merge TODO above), we add
// directly without the merge step.

TensorPtr Qwen3VLViTEncoder::add_position_embedding(const TensorPtr& patches, int nx, int ny) {
    if (!cached_pos_embd_)
        return patches;

    int num_patches = nx * ny;
    int dim = config_.embedding_length;
    float* h_data = static_cast<float*>(patches->data());
    const float* pd = static_cast<const float*>(cached_pos_embd_->data());

    // Position embedding shape: GGUF stores as [dim, max_patches] or [max_patches, dim]
    // From the GGUF metadata: v.position_embd.weight dims=[1024, 2304]
    // This is [dim, num_patches] in GGUF notation (reversed from typical)
    auto pos_shape = cached_pos_embd_->shape();
    int64_t pos_total = cached_pos_embd_->numel();

    // Determine if position embedding is [dim, max_patches] or [max_patches, dim]
    bool transposed = (pos_shape.size() == 2 && pos_shape[0] == dim);

    if (transposed) {
        // [dim, max_patches] — each column is a position's embedding
        int max_patches = static_cast<int>(pos_shape[1]);
        forge::cpu::parallel_for_chunks(num_patches, 1, [&](int i0, int i1) {
            for (int i = i0; i < i1; ++i) {
                int pos_idx = std::min(i, max_patches - 1);
                for (int d = 0; d < dim; ++d) {
                    h_data[i * dim + d] += pd[d * max_patches + pos_idx];
                }
            }
        });
    } else {
        // [max_patches, dim] — standard row-major
        int max_patches = static_cast<int>(pos_total / dim);
        forge::cpu::parallel_for_chunks(num_patches, 1, [&](int i0, int i1) {
            for (int i = i0; i < i1; ++i) {
                int pos_idx = std::min(i, max_patches - 1);
                for (int d = 0; d < dim; ++d) {
                    h_data[i * dim + d] += pd[pos_idx * dim + d];
                }
            }
        });
    }
    return patches;
}

// ---- apply_mrope ----
// M-RoPE (Multimodal Rotary Position Embedding) for Vision (GGML_ROPE_TYPE_VISION).
//
// Matches ggml's rotate_pairs(ne0=head_dim, n_dims=head_dim/2) with
// ggml_mrope_cache_init(indep_sects=true). Key differences from M-RoPE:
//   1. Half-rotation pairs: (d, d+n_dims) for d=0..n_dims-1, NOT interleaved (j,j+1)
//   2. Independent section frequencies: theta resets at section boundaries
//   3. Only first 2 sections are used (sections 2-3 are ignored per ggml spec)
//
// Example with head_dim=8, sections=[2,2,0,0], n_dims=4:
//   Pair layout: [yyyyxxxx] — pairs (0,4)(1,5) use y, pairs (2,6)(3,7) use x
//
// For Qwen3-VL with head_dim=64, sections=[16,16,16,16], n_dims=32:
//   Pairs (0,32)..(15,47) use section 0 position (row/height)
//   Pairs (16,48)..(31,63) use section 1 position (column/width)
//   Sections 2 and 3 are computed in cache but never accessed by rotate_pairs

void Qwen3VLViTEncoder::apply_mrope(float* data, int num_patches, int num_heads, int head_dim,
                                      const int32_t* positions) {
    // data layout: [num_patches, num_heads, head_dim]
    // positions layout: [num_patches * 4] (4 position IDs per patch)
    int n_dims = head_dim / 2;       // 32 for head_dim=64
    int section_dim = head_dim / 4;  // 16 for head_dim=64
    float base = 10000.0f;
    float theta_scale = std::pow(base, -2.0f / n_dims);  // freq_base^(-2/n_dims)

    forge::cpu::parallel_for_chunks(num_patches, 1, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            float pos_base[4] = {
                static_cast<float>(positions[i * 4 + 0]),
                static_cast<float>(positions[i * 4 + 1]),
                static_cast<float>(positions[i * 4 + 2]),
                static_cast<float>(positions[i * 4 + 3]),
            };

            for (int h = 0; h < num_heads; ++h) {
                float* head_ptr = data + (i * num_heads + h) * head_dim;

                float theta[4] = {pos_base[0], pos_base[1], pos_base[2], pos_base[3]};

                for (int d = 0; d < n_dims; ++d) {
                    int sector = d % head_dim;

                    if (sector == 0) {
                        theta[0] = pos_base[0];
                    } else if (sector == section_dim) {
                        theta[1] = pos_base[1];
                    } else if (sector == 2 * section_dim) {
                        theta[2] = pos_base[2];
                    } else if (sector == 3 * section_dim) {
                        theta[3] = pos_base[3];
                    }

                    float pos_theta;
                    if (sector >= 3 * section_dim) {
                        pos_theta = theta[3];
                    } else if (sector >= 2 * section_dim) {
                        pos_theta = theta[2];
                    } else if (sector >= section_dim) {
                        pos_theta = theta[1];
                    } else {
                        pos_theta = theta[0];
                    }

                    float cos_val = std::cos(pos_theta);
                    float sin_val = std::sin(pos_theta);

                    float x0 = head_ptr[d];
                    float x1 = head_ptr[d + n_dims];
                    head_ptr[d] = x0 * cos_val - x1 * sin_val;
                    head_ptr[d + n_dims] = x0 * sin_val + x1 * cos_val;

                    theta[0] *= theta_scale;
                    theta[1] *= theta_scale;
                    theta[2] *= theta_scale;
                    theta[3] *= theta_scale;
                }
            }
        }
    });
}

// ---- forward_vit_block ----
// ViT block with fused QKV + M-RoPE.

TensorPtr Qwen3VLViTEncoder::forward_vit_block(const TensorPtr& hidden, const ViTLayerWeights& lw,
                                                 int num_heads, int head_dim,
                                                 const int32_t* positions) {
    int num_patches = static_cast<int>(hidden->shape()[0]);
    int dim = num_heads * head_dim;

    // LayerNorm 1
    auto normed = ops::layer_norm(hidden, lw.ln1_weight, lw.ln1_bias, config_.layer_norm_epsilon);

    TensorPtr q, k, v;

    if (lw.attn_qkv_weight) {
        // Fused QKV: [num_patches, 3*dim]
        auto qkv = ops::matmul_transB(normed, lw.attn_qkv_weight, lw.attn_qkv_bias);

        // Split Q, K, V from fused output
        auto qkv_cpu = qkv;
        bool was_cuda = (qkv->device() == DeviceType::CUDA);
        if (was_cuda) {
            qkv_cpu = std::make_shared<Tensor>(DataType::FP32, qkv->shape(), DeviceType::CPU);
            qkv_cpu->copy_from(*qkv);
        }
        const float* qkv_data = static_cast<const float*>(qkv_cpu->data());

        q = std::make_shared<Tensor>(DataType::FP32,
                                      std::vector<int64_t>{num_patches, dim}, DeviceType::CPU);
        k = std::make_shared<Tensor>(DataType::FP32,
                                      std::vector<int64_t>{num_patches, dim}, DeviceType::CPU);
        v = std::make_shared<Tensor>(DataType::FP32,
                                      std::vector<int64_t>{num_patches, dim}, DeviceType::CPU);
        float* q_data = static_cast<float*>(q->data());
        float* k_data = static_cast<float*>(k->data());
        float* v_data = static_cast<float*>(v->data());

        // Fused QKV layout: [num_patches, 3*dim]
        // Q: [0, dim), K: [dim, 2*dim), V: [2*dim, 3*dim)
        forge::cpu::parallel_for_chunks(num_patches, 1, [&](int i0, int i1) {
            for (int i = i0; i < i1; ++i) {
                const float* row = qkv_data + i * 3 * dim;
                std::memcpy(q_data + i * dim, row, dim * sizeof(float));
                std::memcpy(k_data + i * dim, row + dim, dim * sizeof(float));
                std::memcpy(v_data + i * dim, row + 2 * dim, dim * sizeof(float));
            }
        });

        if (was_cuda) {
            q = to_device(q, DeviceType::CUDA);
            k = to_device(k, DeviceType::CUDA);
            v = to_device(v, DeviceType::CUDA);
        }
    } else {
        // Separate Q, K, V
        q = ops::matmul_transB(normed, lw.attn_q_weight, lw.attn_q_bias);
        k = ops::matmul_transB(normed, lw.attn_k_weight, lw.attn_k_bias);
        v = ops::matmul_transB(normed, lw.attn_v_weight, lw.attn_v_bias);
    }

    // Reshape Q, K for M-RoPE: [num_patches, num_heads, head_dim]
    // We need to apply M-RoPE on CPU
    auto q_cpu = q;
    auto k_cpu = k;
    bool q_was_cuda = (q->device() == DeviceType::CUDA);
    if (q_was_cuda) {
        q_cpu = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
        k_cpu = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
        q_cpu->copy_from(*q);
        k_cpu->copy_from(*k);
    }

    // Reshape from [num_patches, dim] to [num_patches, num_heads, head_dim]
    // and apply M-RoPE
    {
        // The data in q_cpu is [num_patches, dim] in row-major.
        // We need to treat it as [num_patches, num_heads, head_dim].
        float* q_data = static_cast<float*>(q_cpu->data());
        float* k_data = static_cast<float*>(k_cpu->data());

        apply_mrope(q_data, num_patches, num_heads, head_dim, positions);
        apply_mrope(k_data, num_patches, num_heads, head_dim, positions);
    }

    if (q_was_cuda) {
        q = to_device(q_cpu, DeviceType::CUDA);
        k = to_device(k_cpu, DeviceType::CUDA);
    } else {
        q = q_cpu;
        k = k_cpu;
    }

    // Self-attention
    auto attn_out = ops::scaled_dot_product_attention_2d(q, k, v, num_patches, num_heads,
                                                         head_dim, nullptr, false);

    // Output projection
    auto attn_proj = ops::matmul_transB(attn_out, lw.attn_out_weight, lw.attn_out_bias);

    // Residual
    auto res1 = ops::add(hidden, attn_proj);

    // LayerNorm 2
    auto normed2 = ops::layer_norm(res1, lw.ln2_weight, lw.ln2_bias, config_.layer_norm_epsilon);

    // FFN
    auto ffn_up = ops::matmul_transB(normed2, lw.ffn_up_weight, lw.ffn_up_bias);
    ffn_up = ops::gelu(ffn_up);
    auto ffn_down = ops::matmul_transB(ffn_up, lw.ffn_down_weight, lw.ffn_down_bias);

    return ops::add(res1, ffn_down);
}

// ---- forward_deepstack ----

TensorPtr Qwen3VLViTEncoder::forward_deepstack(const TensorPtr& hidden, int layer_idx,
                                                 int num_patches, DeviceType dev) {
    // Dead code in the current forward pass (encode() runs the ViT blocks only),
    // and prepare_weights deliberately does not stage these -- 312 MB of FP32
    // for a 4B tower for features nobody consumes. Materialize on demand so the
    // function stays correct if deepstack injection is ever wired up.
    auto it = weights_.deepstack_weights.find(layer_idx);
    if (it == weights_.deepstack_weights.end())
        return nullptr;
    if (cached_deepstack_.find(layer_idx) == cached_deepstack_.end()) {
        const auto& dw = it->second;
        CachedDeepstack cd;
        cd.norm_w = dw.norm_weight ? to_device(dw.norm_weight, dev) : nullptr;
        cd.norm_b = dw.norm_bias ? to_device(dw.norm_bias, dev) : nullptr;
        cd.fc1_b = dw.fc1_bias ? to_device(dw.fc1_bias, dev) : nullptr;
        cd.fc2_b = dw.fc2_bias ? to_device(dw.fc2_bias, dev) : nullptr;
        // FP16 masters stay on the host: the FP32 forms go through the scratch.
        cd.fc1_w = dw.fc1_weight;
        cd.fc2_w = dw.fc2_weight;
        cached_deepstack_[layer_idx] = std::move(cd);
    }

    const auto& cd_it = cached_deepstack_.find(layer_idx);
    CachedDeepstack& cd = cd_it->second;
    TensorPtr fc1_w = fp32_from_master(cd.fc1_w, cd.scratch_fc1_, dev);
    TensorPtr fc2_w = fp32_from_master(cd.fc2_w, cd.scratch_fc2_, dev);

    int dim = config_.embedding_length;
    int merge_factor = config_.spatial_merge_size * config_.spatial_merge_size;  // 4

    // 2x2 spatial merge: [num_patches, dim] -> [num_patches/4, dim*4]
    int new_num = num_patches / merge_factor;
    int concat_dim = dim * merge_factor;

    auto hidden_cpu = hidden;
    if (dev == DeviceType::CUDA) {
        hidden_cpu = std::make_shared<Tensor>(DataType::FP32, hidden->shape(), DeviceType::CPU);
        hidden_cpu->copy_from(*hidden);
    }
    const float* h_data = static_cast<const float*>(hidden_cpu->data());

    int grid_size = static_cast<int>(std::sqrt(num_patches));
    int new_grid = grid_size / config_.spatial_merge_size;

    auto concat_tokens = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{new_num, concat_dim}, DeviceType::CPU);
    float* concat_data = static_cast<float*>(concat_tokens->data());

    cpu::parallel_for_dynamic(new_num, 1, [&](int idx) {
        int gh = idx / new_grid;
        int gw = idx % new_grid;
        float* out_ptr = concat_data + idx * concat_dim;
        for (int dh = 0; dh < config_.spatial_merge_size; ++dh) {
            for (int dw = 0; dw < config_.spatial_merge_size; ++dw) {
                int orig_idx =
                    (gh * config_.spatial_merge_size + dh) * grid_size +
                    (gw * config_.spatial_merge_size + dw);
                const float* src = h_data + orig_idx * dim;
                int offset = (dh * config_.spatial_merge_size + dw) * dim;
                std::memcpy(out_ptr + offset, src, dim * sizeof(float));
            }
        }
        });

    // LayerNorm + FC1 + GELU + FC2
    auto h = to_device(concat_tokens, dev);
    if (cd.norm_w) {
        h = ops::layer_norm(h, cd.norm_w, cd.norm_b, config_.layer_norm_epsilon);
    }
    auto fc1_out = ops::matmul_transB(h, fc1_w, cd.fc1_b);
    fc1_out = ops::gelu(fc1_out);
    return ops::matmul_transB(fc1_out, fc2_w, cd.fc2_b);
}

// ---- project ----
// Final 2x2 spatial merge + mm projector (mm.0 + GELU + mm.2)
// Since tokens are already in 2x2-block-major order after spatial_merge(),
// every 4 consecutive tokens form a 2x2 block — just concatenate them.
// This matches llama.cpp's: ggml_reshape_3d(embeddings, n_embd*4, n_pos/4, batch)

TensorPtr Qwen3VLViTEncoder::project(const TensorPtr& hidden, int num_patches, int nx, int ny,
                                      DeviceType dev) {
    int dim = config_.embedding_length;
    int merge_factor = config_.spatial_merge_size * config_.spatial_merge_size;  // 4

    // 2x2 spatial merge: concatenate every 4 consecutive tokens
    int new_num = num_patches / merge_factor;
    int concat_dim = dim * merge_factor;

    auto hidden_cpu = hidden;
    if (dev == DeviceType::CUDA) {
        hidden_cpu = std::make_shared<Tensor>(DataType::FP32, hidden->shape(), DeviceType::CPU);
        hidden_cpu->copy_from(*hidden);
    }
    const float* h_data = static_cast<const float*>(hidden_cpu->data());

    auto concat_tokens = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{new_num, concat_dim}, DeviceType::CPU);
    float* concat_data = static_cast<float*>(concat_tokens->data());

    forge::cpu::parallel_for_chunks(new_num, 1, [&](int n0, int n1) {
        for (int idx = n0; idx < n1; ++idx) {
            float* out_ptr = concat_data + idx * concat_dim;
            for (int k = 0; k < merge_factor; ++k) {
                const float* src = h_data + (idx * merge_factor + k) * dim;
                std::memcpy(out_ptr + k * dim, src, dim * sizeof(float));
            }
        }
    });

    // mm projector: mm.0 (GELU) + mm.2. The matrices are FP16 masters staged in
    // prepare_weights; their FP32 forms are materialized here, once per image,
    // rather than held resident (~104 MB for the 4B tower).
    auto h = to_device(concat_tokens, dev);
    auto mm0_w = fp32_from_master(mm0_master_, scratch_mm0_, dev);
    auto mm2_w = fp32_from_master(mm2_master_, scratch_mm2_, dev);
    if (mm0_w && mm2_w) {
        auto up = ops::matmul_transB(h, mm0_w, mm0_b_);
        up = ops::gelu(up);
        h = ops::matmul_transB(up, mm2_w, mm2_b_);
    }

    return h;
}

// ---- encode ----

std::vector<float> Qwen3VLViTEncoder::encode(const float* rgb_data, int width, int height,
                                               int channels) {
    if (!initialized_) {
        LOG_ERROR("Qwen3VLViTEncoder: not initialized");
        return {};
    }

    PERF_SCOPE("vision.encode_qwen3vl");

    // Determine target device early — needed before patch_embedding which
    // uses cached weights that must be prepared first.
    DeviceType dev = DeviceType::CPU;
#ifdef USE_CUDA
    try {
        // Test CUDA availability with a small allocation
        auto test = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1}, DeviceType::CUDA);
        (void)test;
        dev = DeviceType::CUDA;
    } catch (...) {
        dev = DeviceType::CPU;
    }
#endif

    // Prepare cached weights on target device (must happen BEFORE patch_embedding
    // and add_position_embedding, which both depend on cached CPU weights)
    {
        PERF_SCOPE("vision.prepare_weights");
        prepare_weights(dev);
    }

    // Step 1: Preprocess
    TensorPtr pixels;
    {
        PERF_SCOPE("vision.preprocess");
        pixels = preprocess(rgb_data, width, height, channels);
    }

    // Step 2: Patch embedding (two-part conv)
    TensorPtr hidden;
    {
        PERF_SCOPE("vision.patch_embd");
        hidden = patch_embedding(pixels);
    }

    // Grid dimensions
    int nx = config_.image_size / config_.patch_size;  // 48
    int ny = nx;
    int num_patches = nx * ny;  // 2304

    // Step 3: Add position embedding (in original row-major order)
    // Spatial merge is a permutation, so adding position embedding before or
    // after the merge is equivalent — both operands get the same permutation.
    {
        PERF_SCOPE("vision.pos_embd");
        hidden = add_position_embedding(hidden, nx, ny);
    }

    // Step 4: Spatial merge (reorder patches to 2x2-block-major)
    // After this, every 4 consecutive tokens form a 2x2 spatial block.
    {
        PERF_SCOPE("vision.spatial_merge");
        hidden = spatial_merge(hidden, nx, ny);
    }

    // Compute M-RoPE position IDs based on the spatially-merged token order.
    // For Vision M-RoPE, section 0 uses row/y position, section 1 uses column/x
    // (matching ggml's pos_h=h, pos_w=w layout; only first 2 sections are used).
    // For each token at new index i, recover the original grid position (x, y).
    int half_nx = nx / 2;
    std::vector<int32_t> positions(num_patches * 4);
    for (int i = 0; i < num_patches; ++i) {
        int block_y = i / (half_nx * 4);
        int rem = i % (half_nx * 4);
        int block_x = rem / 4;
        int within = rem % 4;
        int x = block_x * 2 + (within / 2);
        int y = block_y * 2 + (within % 2);
        positions[i * 4 + 0] = y;       // section 0: row/height
        positions[i * 4 + 1] = x;       // section 1: column/width
        positions[i * 4 + 2] = 0;       // section 2: unused
        positions[i * 4 + 3] = 0;       // section 3: unused
    }

    // Move to CUDA for ViT blocks if available
    if (dev == DeviceType::CUDA) {
        hidden = to_device(hidden, DeviceType::CUDA);
    }

    int num_heads = config_.head_count;
    int head_dim = config_.embedding_length / num_heads;

    // Step 5: ViT blocks with DeepStack
    {
        PERF_SCOPE("vision.vit_blocks");
        for (int i = 0; i < config_.block_count; ++i) {
            // Materializes block i's FP32 weights into the shared scratch
            // (~50 MB) instead of holding an FP32 copy of all 24 blocks.
            const ViTLayerWeights& lw = load_block(i, dev);
            hidden = forward_vit_block(hidden, lw, num_heads, head_dim, positions.data());

            // DeepStack: collect side features at specific layers
            bool is_deepstack = false;
            for (int dl : config_.deepstack_layers) {
                if (dl == i) { is_deepstack = true; break; }
            }
            // Note: deepstack features are computed but not returned in the
            // main encode() path. They would be used for layer-wise injection
            // into the LLM, which is a future optimization.
            // For now, we just run through all ViT blocks.
        }
    }

    // Step 6: Post LayerNorm
    if (pst_ln_w_) {
        PERF_SCOPE("vision.post_ln");
        hidden = ops::layer_norm(hidden, pst_ln_w_, pst_ln_b_, config_.layer_norm_epsilon);
    }

    // Step 7: Final 2x2 spatial merge + mm projector
    TensorPtr projected;
    {
        PERF_SCOPE("vision.project");
        projected = project(hidden, num_patches, nx, ny, dev);
    }

    // Return as flat vector
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
