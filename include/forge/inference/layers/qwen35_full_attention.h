#pragma once

// Qwen35FullAttention: Qwen3.5 Full Attention 层 (gated Q + Q/K norm + MRoPE)。
//
// 从 Qwen35Engine::forward_full_attn_layer_{cpu,cuda} 抽出。输入为已过 attn_norm
// 的 hidden, 返回已过 attn_output 投影的 attention 输出 (不含残差)。
// CPU/GPU 两条路径的分派规则收敛在本组件内, engine 不再关心。

#include "forge/inference/layer_execution_context.h"
#include "forge/inference/layers/attention_executor.h"
#include "forge/kv_cache.h"
#include "forge/kv_memory.h"
#include "forge/model.h"

namespace forge {

class Qwen35FullAttention {
public:
    // kv_memory may be null (legacy / test paths); when non-null the attention
    // layer routes KV writes through it so it works under both the contiguous
    // KVCache and the paged PagedKVStorage backends without any coupling.
    explicit Qwen35FullAttention(KVCache& kv_cache, KVMemory* kv_memory = nullptr)
        : kv_cache_(kv_cache),
          kv_memory_(kv_memory),
          attn_(kv_cache,
                (kv_memory && kv_memory->is_paged()) ? &kv_memory->storage() : nullptr) {}

    // 权重缺失时返回 nullptr, 由 engine 决定如何降级。
    TensorPtr attend(const TensorPtr& normed, const LayerExecutionContext& lctx);

private:
    TensorPtr attend_cpu(const TensorPtr& normed, const LayerExecutionContext& lctx);
#ifdef USE_CUDA
    TensorPtr attend_cuda(const TensorPtr& normed, const LayerExecutionContext& lctx);
#endif

    KVCache& kv_cache_;
    KVMemory* kv_memory_ = nullptr;
    // Shared, storage-aware attention backend. Handles contiguous + paged,
    // CPU + CUDA, prefill + decode uniformly — the only place that needs to
    // know about the KV backend.
    AttentionExecutor attn_;
};

}  // namespace forge
