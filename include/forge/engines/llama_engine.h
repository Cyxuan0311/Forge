#pragma once

#include "forge/engines/transformer_engine.h"

namespace forge {

class LlamaEngine : public TransformerEngine {
public:
    explicit LlamaEngine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "llama"; }

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, int layer_idx, int seq_len, int64_t start_pos,
                            DeviceType dev) override;
    bool init_weights() override;

private:
    void apply_rope_neox_cpu(const float* q_data, const float* k_data, float* q_out, float* k_out,
                             int seq_len, int num_heads, int num_kv_heads, int head_dim,
                             int64_t start_pos, float theta);
};

}  // namespace forge
