#pragma once

// Gemma4Moe: Gemma4 的 MoE FFN (shared expert + routed experts)。
//
// 从 Gemma4Engine::forward_layer 的 MoE 分支抽出。包含 router 缩放、top-k 选择、
// GPU 融合 expert GEMV 与 CPU 逐 token fallback 两条路径。
//
// 输入为第一次残差之后的 hidden, 返回 shared + routed 的合并结果。

#include "forge/inference/layer_execution_context.h"
#include "forge/model.h"

namespace forge {

class TransformerEngine;  // MoE expert-activation hook (P0+); see transformer_engine.h

class Gemma4Moe {
public:
    // engine may be null; when provided, the router's top-k experts are reported
    // via engine->sync_experts_resident() so later phases can page experts.
    static TensorPtr apply(const TensorPtr& attn_residual, const LayerExecutionContext& lctx,
                           TransformerEngine* engine = nullptr);
};

}  // namespace forge
