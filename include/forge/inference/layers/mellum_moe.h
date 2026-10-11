#pragma once

// MellumMoe: Mellum 的路由-only MoE FFN (无共享专家)。
//
// 计算图对齐 llama.cpp src/models/mellum.cpp:
//   normed = rms_norm(residual, ffn_norm)
//   router = softmax-all(normed @ ffn_gate_inp) -> top-k, weight = p[e]/topk_sum
//   out    = sum_k weight_k * expert_k(normed)
//   expert_k: silu(normed @ gate_e) * (normed @ up_e) -> @ down_e
//
// 与 Gemma4Moe 的差异: 无共享专家 / 无 router 缩放 / SiLU 激活 / 输入 norm 是
// ffn_norm 本身。输入为残差 hidden, 返回 MoE 输出 (不含残差, 由 engine 相加)。

#include "forge/inference/layer_execution_context.h"
#include "forge/model.h"

namespace forge {

class TransformerEngine;  // router top-k hook; see transformer_engine.h

class MellumMoe {
public:
    static TensorPtr apply(const TensorPtr& attn_residual, const LayerExecutionContext& lctx,
                           TransformerEngine* engine = nullptr);
};

}  // namespace forge
