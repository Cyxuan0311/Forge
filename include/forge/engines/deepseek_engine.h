#pragma once

#include "forge/engines/transformer_engine.h"
#include "forge/inference/layers/deepseek_layer_executor.h"

namespace forge {

// DeepSeekEngine: 只做编排。MLA / GQA 的层内实现位于 DeepSeekLayerExecutor。
class DeepSeekEngine : public TransformerEngine {
public:
    explicit DeepSeekEngine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "deepseek"; }

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;
    bool init_weights() override;

private:
    DeepSeekLayerExecutor layer_executor_;
};

}  // namespace forge
