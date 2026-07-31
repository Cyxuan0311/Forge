#pragma once

#include "forge/engines/transformer_engine.h"
#include "forge/inference/layers/qwen35_full_attention.h"
#include "forge/inference/layers/qwen35_linear_attention.h"

namespace forge {

class Qwen35Engine : public TransformerEngine {
public:
    explicit Qwen35Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "qwen35"; }

    void reset() override;

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;
    bool init_weights() override;

private:
    // FFN 对 FullAttention 与 LinearAttention 两种层型完全相同。
    TensorPtr apply_ffn(const TensorPtr& hidden_after_attn, const LayerExecutionContext& lctx);

    // recurrent state 由 InferenceContext 的 HybridMemory 持有, engine 只引用。
    Qwen35RecurrentMemory& recurrent_memory_;
    Qwen35FullAttention full_attention_;
    Qwen35LinearAttention linear_attention_;
};

}  // namespace forge
