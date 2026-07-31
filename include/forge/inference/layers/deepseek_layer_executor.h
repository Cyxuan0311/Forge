#pragma once

// DeepSeekLayerExecutor: DeepSeek 单层前向。
//
// 从 DeepSeekEngine::forward_layer_gqa / forward_layer_mla 抽出。按 LayerType
// 选择 GQA 或 MLA attention, 之后的输出投影、残差和 FFN 两种层型共用。
// DeepSeekEngine 因此只保留 embedding/output/层遍历的编排职责。

#include "forge/inference/layer_execution_context.h"
#include "forge/inference/layers/mla_executor.h"
#include "forge/kv_cache.h"
#include "forge/model.h"

namespace forge {

class DeepSeekLayerExecutor {
public:
    explicit DeepSeekLayerExecutor(KVCache& kv_cache) : kv_cache_(kv_cache), mla_(kv_cache) {}

    TensorPtr execute(const TensorPtr& hidden, const LayerExecutionContext& lctx);

private:
    // GQA attention: 标准 QKV 投影 + RoPE + KV cache + 展开 KV heads。
    TensorPtr attend_gqa(const TensorPtr& normed, const LayerExecutionContext& lctx);

    KVCache& kv_cache_;
    MlaExecutor mla_;
};

}  // namespace forge
