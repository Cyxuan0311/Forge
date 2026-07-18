#pragma once

#include "transformer_engine.h"

namespace forge {

class Gemma4Engine : public TransformerEngine {
public:
    Gemma4Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "Gemma4Engine"; }

    TensorPtr forward(const TensorPtr& input_ids, int64_t start_pos) override;

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, int layer_idx, int seq_len, int64_t start_pos,
                            DeviceType dev) override;
    bool init_weights() override;
    void init_kv_cache(const ModelConfig& cfg) override;

private:
    // Per-layer embedding projection cache
    TensorPtr per_layer_proj_cache_;  // [n_embd_per_layer, n_layer, n_tokens]
    TensorPtr per_layer_input_cache_; // [n_embd_per_layer, n_layer, n_tokens]

    // Proportional RoPE frequency factors for full-attention layers
    TensorPtr rope_freqs_;            // [head_dim/2] per-dimension frequency scale
    TensorPtr rope_freqs_cpu_;        // CPU copy for RoPE computation when rope_freqs_ is on CUDA
};

}  // namespace forge
