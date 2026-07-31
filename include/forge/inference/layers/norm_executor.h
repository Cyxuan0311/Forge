#pragma once

// NormExecutor: 归一化算子策略。
//
// 从 GenericEngine::norm_forward / qk_norm_forward 抽出, 不持有任何运行时状态,
// 因此全部为静态方法, 可以独立单测。

#include "forge/inference/layer_execution_context.h"
#include "forge/model.h"

namespace forge {

class NormExecutor {
public:
    // LayerNorm 或 RMSNorm。bias 仅 LayerNorm 使用。
    static TensorPtr apply(const TensorPtr& x, const TensorPtr& weight, const TensorPtr& bias,
                           NormType type, float eps);

    // per-head RMSNorm (Llama/Qwen3VL/Gemma4 的 QK-Norm)。
    static TensorPtr apply_qk_norm(const TensorPtr& x, const TensorPtr& norm_weight, int num_heads,
                                   int head_dim, float eps, DeviceType dev);
};

}  // namespace forge
