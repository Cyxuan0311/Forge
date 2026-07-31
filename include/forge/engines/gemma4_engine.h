#pragma once

#include "forge/inference/layers/gemma4_attention.h"
#include "forge/inference/layers/gemma4_embedding.h"
#include "transformer_engine.h"

namespace forge {

class Gemma4Engine : public TransformerEngine {
public:
    Gemma4Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "Gemma4Engine"; }

    TensorPtr forward_request(const ForwardRequest& req) override;

    // Gemma4 has custom embedding scaling + per-layer projection;
    // override forward_batch to use per-sequence forward() instead of
    // the flat hidden state path (which doesn't apply Gemma4-specific transforms).
    TensorPtr forward_batch(const InferenceBatch& batch) override;

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;
    bool init_weights() override;
    void init_kv_cache(const ModelConfig& cfg) override;

private:
    // final logit softcapping + suppress tokens。
    void apply_logit_postprocess(TensorPtr& logits, const ModelConfig& cfg);

    Gemma4Embedding embedding_;
    Gemma4Attention attention_;

    // GPU-side cache for suppress tokens (used by logit softcap kernel)
    TensorPtr suppress_tokens_gpu_;   // [num_suppress] int32 on CUDA
};

}  // namespace forge
