#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "forge/tensor.h"
#include "forge/types.h"

namespace forge {

class WeightStore;

// ============================================================================
// Position embedding type (inferred from projector_type / arch)
// ============================================================================
enum class PosEmbedType : int {
    None = 0,        // no position embedding
    Learned2D = 1,   // learned 2D positional embedding (SigLIP-style, bucket indexing)
    Learned1D = 2,   // learned 1D positional embedding (CLIP-style)
};

// Configuration for vision encoder (driven by mmproj GGUF metadata)
struct VisionConfig {
    int image_size = 0;            // input image size (parsed from metadata)
    int patch_size = 0;            // patch size
    int embedding_length = 0;      // hidden dim of vision encoder
    int feed_forward_length = 0;   // FFN intermediate dim
    int block_count = 0;           // number of ViT blocks
    int head_count = 0;            // number of attention heads
    float layer_norm_epsilon = 1e-6f;
    std::vector<float> image_mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> image_std = {0.5f, 0.5f, 0.5f};
    int projection_dim = 0;        // output projection dim (LLM hidden dim)
    int scale_factor = 4;          // vit merger scale factor
    bool use_gelu = true;          // use GELU activation
    std::string projector_type = "none";
    int insert_layer_id = -1;      // ViT merger insertion layer (-1 = after all layers)
    PosEmbedType pos_embed_type = PosEmbedType::None;
    // SigLIP 2D bucket position encoding parameter (n_embd_buckets per axis)
    int n_embd_buckets = 70;
    // Qwen3-VL spatial merge size (2 = 2x2 merge)
    int spatial_merge_size = 1;
    // Qwen3-VL deepstack layer indices (layers with extra FC processing)
    std::vector<int> deepstack_layers;
    // Qwen3-VL uses fused QKV + M-RoPE (separate Q/K/V weights not present)
    bool use_fused_qkv = false;
};

// Per-layer weights for ViT
struct ViTLayerWeights {
    TensorPtr ln1_weight;
    TensorPtr ln1_bias;
    TensorPtr ln2_weight;
    TensorPtr ln2_bias;
    TensorPtr attn_q_weight;
    TensorPtr attn_q_bias;
    TensorPtr attn_k_weight;
    TensorPtr attn_k_bias;
    TensorPtr attn_v_weight;
    TensorPtr attn_v_bias;
    TensorPtr attn_out_weight;
    TensorPtr attn_out_bias;
    TensorPtr ffn_up_weight;
    TensorPtr ffn_up_bias;
    TensorPtr ffn_down_weight;
    TensorPtr ffn_down_bias;
    // Qwen3-VL fused QKV (when use_fused_qkv=true, q/k/v weights are empty)
    TensorPtr attn_qkv_weight;
    TensorPtr attn_qkv_bias;
};

// Vision Encoder weights
struct VisionWeights {
    TensorPtr patch_embd_weight;
    TensorPtr patch_embd_bias;
    TensorPtr position_embd_weight;
    TensorPtr post_ln_weight;
    TensorPtr post_ln_bias;

    // Qwen3-VL: second half of split patch embedding conv
    TensorPtr patch_embd_weight_1;

    // ViT merger (token compression)
    TensorPtr merger_ln1_weight;
    TensorPtr merger_ln1_bias;
    TensorPtr merger_attn_q_weight;
    TensorPtr merger_attn_q_bias;
    TensorPtr merger_attn_k_weight;
    TensorPtr merger_attn_k_bias;
    TensorPtr merger_attn_v_weight;
    TensorPtr merger_attn_v_bias;
    TensorPtr merger_attn_out_weight;
    TensorPtr merger_attn_out_bias;
    TensorPtr merger_ds_ln_weight;
    TensorPtr merger_ds_ln_bias;
    TensorPtr merger_ds_ffn_up_weight;
    TensorPtr merger_ds_ffn_up_bias;
    TensorPtr merger_ds_ffn_down_weight;
    TensorPtr merger_ds_ffn_down_bias;

    // Projector (mm.*)
    TensorPtr mm_input_norm_weight;
    TensorPtr mm_input_norm_bias;
    TensorPtr mm_up_weight;
    TensorPtr mm_up_bias;
    TensorPtr mm_down_weight;
    TensorPtr mm_down_bias;

    // Qwen3-VL: mm.0 / mm.2 projector (2-layer MLP with GELU)
    TensorPtr mm0_weight;
    TensorPtr mm0_bias;
    TensorPtr mm2_weight;
    TensorPtr mm2_bias;

    // Qwen3-VL deepstack per-layer weights
    struct DeepstackWeights {
        TensorPtr norm_weight;
        TensorPtr norm_bias;
        TensorPtr fc1_weight;
        TensorPtr fc1_bias;
        TensorPtr fc2_weight;
        TensorPtr fc2_bias;
    };
    std::unordered_map<int, DeepstackWeights> deepstack_weights;

    std::vector<ViTLayerWeights> layers;

    bool init(const WeightStore& store, const VisionConfig& config,
              const struct VisionWeightMapping& mapping);
};

// ============================================================================
// VisionEncoder — abstract base class
// ============================================================================
// Concrete encoders (SiglipViTEncoder, future Gemma3ViTEncoder, etc.) implement
// this interface. Callers always hold VisionEncoder* / unique_ptr<VisionEncoder>
// and dispatch through the virtual interface, so adding a new encoder requires
// no changes at call sites — only a new registration (see vision_registry.h).
// ============================================================================
class VisionEncoder {
public:
    virtual ~VisionEncoder() = default;

    virtual bool init(const WeightStore& store, const VisionConfig& config) = 0;

    // Full forward pass: image pixels -> vision embeddings
    // Returns: [num_output_tokens, projection_dim]
    virtual std::vector<float> encode(const float* rgb_data, int width, int height,
                                      int channels) = 0;

    virtual const VisionConfig& config() const = 0;
    virtual std::string name() const = 0;
};

// ============================================================================
// SiglipViTEncoder — SigLIP-L + MiniCPM-V 4.6 merger/projector
// ============================================================================
// Implements the SigLIP ViT architecture with optional ViT merger insertion
// (MiniCPM-V 4.6 style) and a final 2x2 spatial downsample projector.
// ============================================================================
class SiglipViTEncoder : public VisionEncoder {
public:
    SiglipViTEncoder() = default;

    bool init(const WeightStore& store, const VisionConfig& config) override;
    std::vector<float> encode(const float* rgb_data, int width, int height,
                              int channels) override;
    const VisionConfig& config() const override { return config_; }
    std::string name() const override { return "siglip"; }

private:
    // Preprocess raw image pixels -> normalized tensor
    // input: [3, image_size, image_size] float32 RGB
    // output: [num_patches, embedding_length]
    TensorPtr preprocess(const float* rgb_data, int width, int height, int channels) const;

    // Patch embedding
    TensorPtr patch_embedding(const TensorPtr& pixel_values);

    // Single ViT block (uses cached device weights in cached_layers_)
    TensorPtr forward_vit_block(const TensorPtr& hidden, const ViTLayerWeights& lw, int num_heads,
                                int head_dim);

    // Vision token merger (downsample) — uses cached merger weights
    TensorPtr merge_tokens(const TensorPtr& hidden, DeviceType dev);

    // Projector: vision dim -> LLM dim — uses cached projector weights
    TensorPtr project(const TensorPtr& hidden, DeviceType dev);

    // Pre-dequantize all weights to FP32 and move to target device
    void prepare_weights(DeviceType dev);

    VisionConfig config_;
    VisionWeights weights_;
    bool initialized_ = false;

    // Cached FP32 weights on target device (built once by prepare_weights)
    DeviceType weights_device_ = DeviceType::CPU;
    bool weights_prepared_ = false;
    std::vector<ViTLayerWeights> cached_layers_;
    TensorPtr cached_pos_embd_;
    // cached merger
    TensorPtr m_ln1_w_, m_ln1_b_;
    TensorPtr m_q_w_, m_q_b_, m_k_w_, m_k_b_, m_v_w_, m_v_b_, m_o_w_, m_o_b_;
    TensorPtr m_ds_ln_w_, m_ds_ln_b_;
    TensorPtr m_ds_up_w_, m_ds_up_b_, m_ds_down_w_, m_ds_down_b_;
    // cached projector
    TensorPtr p_in_norm_w_, p_in_norm_b_;
    TensorPtr p_up_w_, p_up_b_, p_down_w_, p_down_b_;
    // cached post_ln
    TensorPtr pst_ln_w_, pst_ln_b_;
};

// ============================================================================
// Qwen3VLViTEncoder — Qwen3-VL ViT + DeepStack + MLP Projector
// ============================================================================
// Implements the Qwen3-VL vision encoder which differs from SigLIP:
//   - Two-part split conv2d patch embedding (patch_embd.weight + .weight.1)
//   - Spatial merge (2x2 interleave) after patch embedding
//   - Learned position embedding (additive, with same spatial merge)
//   - Fused QKV attention with M-RoPE (4-section multimodal rotary)
//   - DeepStack layers at specific positions (optional side features)
//   - Simple 2-layer MLP projector (mm.0 + mm.2)
// ============================================================================
class Qwen3VLViTEncoder : public VisionEncoder {
public:
    Qwen3VLViTEncoder() = default;

    bool init(const WeightStore& store, const VisionConfig& config) override;
    std::vector<float> encode(const float* rgb_data, int width, int height,
                              int channels) override;
    const VisionConfig& config() const override { return config_; }
    std::string name() const override { return "qwen3vl"; }

private:
    // Preprocess raw image -> normalized [3, H, W]
    TensorPtr preprocess(const float* rgb_data, int width, int height, int channels) const;

    // Split conv2d patch embedding (two conv parts summed)
    TensorPtr patch_embedding(const TensorPtr& pixel_values);

    // Spatial merge: interleave 2x2 adjacent patch features
    TensorPtr spatial_merge(const TensorPtr& patches, int nx, int ny);

    // Add learned position embedding (with spatial merge applied to pos embd too)
    TensorPtr add_position_embedding(const TensorPtr& patches, int nx, int ny);

    // Single ViT block with fused QKV + M-RoPE
    TensorPtr forward_vit_block(const TensorPtr& hidden, const ViTLayerWeights& lw,
                                int num_heads, int head_dim, const int32_t* positions);

    // Apply M-RoPE to Q or K tensor (4-section rotary embedding)
    void apply_mrope(float* data, int num_patches, int num_heads, int head_dim,
                     const int32_t* positions);

    // Deepstack processing at a specific layer
    TensorPtr forward_deepstack(const TensorPtr& hidden, int layer_idx, int num_patches,
                                DeviceType dev);

    // Final 2x2 spatial merge + mm projector
    TensorPtr project(const TensorPtr& hidden, int num_patches, int nx, int ny, DeviceType dev);

    // Pre-dequantize all weights to FP32
    void prepare_weights(DeviceType dev);

    VisionConfig config_;
    VisionWeights weights_;
    bool initialized_ = false;

    // Cached FP32 weights on target device
    DeviceType weights_device_ = DeviceType::CPU;
    bool weights_prepared_ = false;
    std::vector<ViTLayerWeights> cached_layers_;
    TensorPtr cached_pos_embd_;
    TensorPtr cached_patch_w0_, cached_patch_w1_, cached_patch_b_;
    // cached mm projector
    TensorPtr mm0_w_, mm0_b_, mm2_w_, mm2_b_;
    // cached post_ln
    TensorPtr pst_ln_w_, pst_ln_b_;
    // cached deepstack
    struct CachedDeepstack {
        TensorPtr norm_w, norm_b;
        TensorPtr fc1_w, fc1_b;
        TensorPtr fc2_w, fc2_b;
    };
    std::unordered_map<int, CachedDeepstack> cached_deepstack_;
};

}  // namespace forge
