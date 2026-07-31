#pragma once

// FfnExecutor: FFN 策略 (SiLUGated / GeGLU / SimpleGELU)。
//
// 从 GenericEngine::ffn_forward 抽出。无持久状态。注意 SiLUGated 的量化融合
// 快路径会把 residual 加进结果, 是否已融合由调用方按同一组条件判断。

#include "forge/inference/layer_execution_context.h"
#include "forge/model.h"

namespace forge {

class FfnExecutor {
public:
    static TensorPtr apply(const TensorPtr& x, const TensorPtr& residual, const ModelConfig& cfg,
                           const LayerWeights& lw, int seq_len, DeviceType dev);

    // SiLUGated 的融合 down-proj 是否已经把 residual 加进输出。
    static bool residual_fused(const ModelConfig& cfg, const LayerWeights& lw, int seq_len,
                               DeviceType dev);
};

}  // namespace forge
