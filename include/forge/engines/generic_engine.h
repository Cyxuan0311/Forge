#pragma once

// GenericEngine: unified transformer layer implementation driven by ModelConfig.
// Replaces LlamaEngine, GemmaEngine, FalconEngine with a single class that
// dispatches to 5 strategy sub-functions based on configuration.
//
// Inspired by llama.cpp's llm_graph_context which provides build_norm(),
// build_ffn(), build_attn(), build_qkv(), build_rope_ext() for all models.

#include "forge/engines/transformer_engine.h"

namespace forge {

class GenericEngine : public TransformerEngine {
public:
    GenericEngine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "generic"; }

    // Override forward() to handle embedding scaling and logit softcapping
    TensorPtr forward(const TensorPtr& input_ids, int64_t start_pos, int seq_id = 0) override;

protected:
    bool init_weights() override;

    // ---- 5 strategy sub-functions ----

    // 1. Norm: RMSNorm or LayerNorm, dispatches CPU/CUDA
    TensorPtr norm_forward(const TensorPtr& x, const TensorPtr& weight,
                           const TensorPtr& bias, NormType type, float eps, DeviceType dev);

    // 2. QKV projection: fused Q4_0/Q4_K fast paths or separate matmul
    struct QKVProjResult { TensorPtr q, k, v; };
    QKVProjResult qkv_proj_forward(const TensorPtr& x, int layer_idx,
                                   bool has_bias, DeviceType dev, int seq_len);

    // 3. RoPE: dispatches by RopeType (None/Standard/NeoX/MRoPE/Proportional)
    struct RopeResult { TensorPtr q_rope, k_rope; };
    RopeResult rope_forward(const TensorPtr& q, const TensorPtr& k,
                            int layer_idx, int64_t start_pos, int seq_len,
                            DeviceType dev);

    // 4. Attention: CUDA flash/GQA decode or CPU SDPA, with SWA window support
    // Optional mask: [q_len, kv_len] additive bias (0=attend, -inf=mask out)
    TensorPtr attention_forward(const TensorPtr& q, const TensorPtr& k,
                                const TensorPtr& v, int layer_idx,
                                int64_t start_pos, int seq_len, DeviceType dev,
                                const TensorPtr& mask = nullptr);

    // 5. FFN: SiLUGated/GeGLU/SimpleGELU/MoE, with fused quantized fast paths
    TensorPtr ffn_forward(const TensorPtr& x, const TensorPtr& residual,
                          int layer_idx, int seq_len, DeviceType dev);

    // ---- QK-Norm helper (shared by Llama/Qwen3VL/Gemma4) ----
    TensorPtr qk_norm_forward(const TensorPtr& x, const TensorPtr& norm_weight,
                              int num_heads, int head_dim, DeviceType dev);

    // ---- Main forward_layer (dispatches by ModelConfig) ----
    TensorPtr forward_layer(const TensorPtr& hidden, int layer_idx, int seq_len,
                            int64_t start_pos, DeviceType dev, int seq_id = 0) override;

    // ---- Gemma4-specific overrides ----
    void init_kv_cache(const ModelConfig& cfg) override;

private:
    // Gemma4 proportional RoPE frequency factors
    TensorPtr rope_freqs_;
    TensorPtr rope_freqs_cpu_;
    // Gemma4 per-layer embedding caches
    TensorPtr per_layer_proj_cache_;
    TensorPtr per_layer_input_cache_;
};

}  // namespace forge
