#pragma once

#include "nanoinfer/tensor.h"
#include "nanoinfer/types.h"
#include <memory>
#include <vector>
#include <string>

namespace nanoinfer {

class WeightStore;

// Configuration for CLIP vision encoder (from mmproj GGUF metadata)
struct VisionConfig {
    int image_size = 448;           // input image size
    int patch_size = 14;            // patch size
    int embedding_length = 1152;    // hidden dim of vision encoder
    int feed_forward_length = 4304; // FFN intermediate dim
    int block_count = 27;           // number of ViT blocks
    int head_count = 16;            // number of attention heads
    float layer_norm_epsilon = 1e-6;
    std::vector<float> image_mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> image_std = {0.5f, 0.5f, 0.5f};
    int projection_dim = 1024;      // output projection dim (LLM hidden dim)
    int scale_factor = 4;           // vit merger scale factor
    bool use_gelu = true;           // use GELU activation
    std::string projector_type = "minicpmv4_6";
    int insert_layer_id = -1;       // ViT merger insertion layer (-1 = after all layers)
};

// Per-layer weights for CLIP ViT
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
};

// Vision Encoder weights
struct VisionWeights {
    TensorPtr patch_embd_weight;
    TensorPtr patch_embd_bias;
    TensorPtr position_embd_weight;
    TensorPtr post_ln_weight;
    TensorPtr post_ln_bias;

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

    std::vector<ViTLayerWeights> layers;

    bool init(const WeightStore& store, const VisionConfig& config);
};

class VisionEncoder {
public:
    VisionEncoder() = default;

    bool init(const WeightStore& store, const VisionConfig& config);

    // Preprocess raw image pixels -> normalized tensor
    // input: [3, image_size, image_size] float32 RGB
    // output: [num_patches, embedding_length]
    TensorPtr preprocess(const float* rgb_data, int width, int height, int channels) const;

    // Full forward pass: image pixels -> vision embeddings
    // Returns: [num_output_tokens, projection_dim]
    std::vector<float> encode(const float* rgb_data, int width, int height, int channels);

    const VisionConfig& config() const { return config_; }

private:
    // Patch embedding
    TensorPtr patch_embedding(const TensorPtr& pixel_values);

    // Single ViT block (uses cached device weights in cached_layers_)
    TensorPtr forward_vit_block(const TensorPtr& hidden, const ViTLayerWeights& lw,
                                int num_heads, int head_dim);

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

} // namespace nanoinfer
