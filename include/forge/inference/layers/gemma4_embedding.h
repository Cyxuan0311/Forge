#pragma once

// Gemma4Embedding: Gemma4 的 token embedding 与 per-layer embedding。
//
// 从 Gemma4Engine::forward_request 的 embedding 段和 forward_layer 末尾的
// per-layer embedding 残差抽出。两处共用 per_layer_input_cache_, 因此放在同一组件,
// 由 engine 持有, layer executor 只引用。

#include "forge/inference/layer_execution_context.h"
#include "forge/model.h"
#include "forge/model_weights.h"

namespace forge {

class Gemma4Embedding {
public:
    // token embedding + sqrt(n_embd) 缩放, 并填充 per-layer proj / input 缓存。
    TensorPtr embed(const TensorPtr& ids_on_dev, const TensorPtr& token_emb,
                    const ModelWeights& weights, const ModelConfig& cfg, DeviceType dev,
                    int seq_len);

    // 层输出上的 per-layer embedding 残差。无 per-layer 权重时原样返回。
    TensorPtr apply_per_layer(const TensorPtr& output, const LayerExecutionContext& lctx);

private:
    TensorPtr per_layer_proj_cache_;   // [n_tokens, n_layer * n_embd_per_layer]
    TensorPtr per_layer_input_cache_;  // [n_tokens, n_layer * n_embd_per_layer]
};

}  // namespace forge
