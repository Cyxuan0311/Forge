#pragma once

// MlaExecutor: DeepSeek MLA (Multi-head Latent Attention) 的 attention 部分。
//
// 从 DeepSeekEngine::forward_layer_mla 抽出。负责 Q 的 low-rank 投影、
// KV 压缩投影、RoPE、KV cache 更新和 latent 空间上的 attention。
// 输出为 attention 结果 (未过 wo), 由 layer executor 负责后续投影与残差。
//
// 算子调用顺序与重构前保持一致, 本阶段不改变数值行为。

#include "forge/inference/layer_execution_context.h"
#include "forge/kv_cache.h"
#include "forge/model.h"

namespace forge {

class MlaExecutor {
public:
    explicit MlaExecutor(KVCache& kv_cache) : kv_cache_(kv_cache) {}

    // normed: 已过 attn_norm 的 hidden。返回 [seq_len, num_heads * kv_lora_rank]。
    TensorPtr attend(const TensorPtr& normed, const LayerExecutionContext& lctx);

private:
    KVCache& kv_cache_;
};

}  // namespace forge
