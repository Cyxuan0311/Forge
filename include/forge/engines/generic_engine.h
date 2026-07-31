#pragma once

// GenericEngine: unified transformer layer implementation driven by ModelConfig.
// Replaces LlamaEngine, GemmaEngine, FalconEngine with a single class.
//
// 阶段 3 之后本类只做编排: 创建 executor、按 ModelConfig 决定调用顺序、
// 管理 KV cache 与 embedding/output。具体算子实现位于 inference/layers/*。

#include "forge/engines/transformer_engine.h"
#include "forge/inference/layers/attention_executor.h"
#include "forge/inference/layers/ffn_executor.h"
#include "forge/inference/layers/norm_executor.h"
#include "forge/inference/layers/rope_executor.h"

namespace forge {

class GenericEngine : public TransformerEngine {
public:
    GenericEngine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "generic"; }

    // Override forward_request() to handle embedding scaling and logit softcapping
    TensorPtr forward_request(const ForwardRequest& req) override;

protected:
    bool init_weights() override;

    // ---- Main forward_layer (dispatches by ModelConfig) ----
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;

    // ---- Gemma4-specific overrides ----
    void init_kv_cache(const ModelConfig& cfg) override;

private:
    RopeExecutor rope_executor_;
    AttentionExecutor attention_executor_;
};

}  // namespace forge
