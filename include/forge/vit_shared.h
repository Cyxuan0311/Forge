#pragma once

#include <string>

#include "forge/tensor.h"
#include "forge/types.h"
#include "forge/vision_encoder.h"

namespace forge {

// ============================================================================
// Shared ViT toolkit functions (P2)
// ============================================================================
// Generic sub-operations shared across ViT-family vision encoders. Each
// function is parameterized by config (norm epsilon, activation, head count)
// so that SiglipViTEncoder and future encoders (Gemma3, Qwen2VL, ...) can
// compose them without duplicating the block / patch-embed / pos-embed logic.
// ============================================================================

// Activation type for ViT FFN
enum class ViTActivation : int {
    GELU_Tanh = 0,  // tanh approximation (default for SigLIP ViT blocks)
    GELU_Erf = 1,   // erf-based (used in MiniCPM-V 4.6 projector)
};

// Generic ViT block forward:
//   LN1 -> QKV -> SDPA -> Out proj -> Residual
//   LN2 -> FFN(up + act + down) -> Residual
// `num_heads` / `head_dim` override config values to support merger phases
// that operate on downsampled token counts.
TensorPtr vit_forward_block(const TensorPtr& hidden, const ViTLayerWeights& lw,
                            const VisionConfig& cfg, int num_heads, int head_dim,
                            ViTActivation act = ViTActivation::GELU_Tanh);

// Generic patch embedding: conv2d-style patch projection on normalized pixels.
// Input:  pixel_values [3, image_size, image_size] (CPU FP32)
// Output: [num_patches, embedding_length] (CPU FP32)
TensorPtr vit_patch_embed(const TensorPtr& pixel_values, const VisionConfig& cfg,
                          const TensorPtr& patch_weight, const TensorPtr& patch_bias);

// Generic 2D bucket position embedding addition (SigLIP-style).
// Adds learned position embeddings to patch tokens in-place, using 2D bucket
// indexing with n_embd_buckets per axis.
TensorPtr vit_add_pos_embed_2d(const TensorPtr& patches, const VisionConfig& cfg,
                               const TensorPtr& pos_weight);

// Dequantize a weight tensor to FP32 on CPU (shared helper).
TensorPtr vit_dequant_to_fp32_cpu(const TensorPtr& w);

// Move tensor to device (shared helper).
TensorPtr vit_to_device(const TensorPtr& t, DeviceType dev);

}  // namespace forge
