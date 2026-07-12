#pragma once

#include "forge/engines/transformer_engine.h"

namespace forge {

class DeepSeekEngine : public TransformerEngine {
public:
    explicit DeepSeekEngine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "deepseek"; }

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, int layer_idx,
                            int seq_len, int64_t start_pos, DeviceType dev) override;
    bool init_weights() override;

private:
    TensorPtr forward_layer_gqa(const TensorPtr& hidden, int layer_idx,
                                int seq_len, int64_t start_pos, DeviceType dev);
    TensorPtr forward_layer_mla(const TensorPtr& hidden, int layer_idx,
                                int seq_len, int64_t start_pos, DeviceType dev);
};

} // namespace forge
