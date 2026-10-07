#pragma once

// GemmaEmbedding2Engine (arch gemma-embedding2, EmbeddingGemma 2):
// Gemma4 block stack configured as a text-embedding encoder. Differences from
// the generative Gemma4Engine (llama.cpp gemma-embedding2.cpp):
//   - bidirectional attention (no causal mask), symmetric SWA window ±n_swa/2
//   - no KV cache: the whole prompt attends in one pass
//   - exact-erf GELU in the FFN and per-layer gates (no tanh approximation)
//   - tail: rms_norm(output_norm) -> output projection to n_embd_out (768)
//           -> mean pooling -> [1, n_embd_out]

#include "forge/inference/layers/gemma4_embedding.h"
#include "transformer_engine.h"

namespace forge {

class GemmaEmbedding2Engine : public TransformerEngine {
public:
    GemmaEmbedding2Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "GemmaEmbedding2Engine"; }

    // Returns the pooled text embedding [1, n_embd_out] (CPU FP32).
    TensorPtr forward_request(const ForwardRequest& req) override;

    // Per-sequence embedding; returns [n_seq, n_embd_out] (CPU FP32).
    TensorPtr forward_batch(const InferenceBatch& batch) override;

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;
    bool init_weights() override;
    void init_kv_cache(const ModelConfig& cfg) override;

private:
    Gemma4Embedding embedding_;
};

}  // namespace forge
