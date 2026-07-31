#pragma once

// Gemma4Attention: Gemma4 的 attention。
//
// 从 Gemma4Engine::forward_layer 的 Q/K/V 投影、Q/K/V norm、Gemma4 RoPE、
// KV cache 更新和 attention 段抽出。SWA 与 full-attention 层的 head 维度差异、
// 无自有 KV 层复用其他层 KV 的规则都收敛在这里。
//
// 返回 attention 输出 (未过 wo)。

#include "forge/inference/layer_execution_context.h"
#include "forge/kv_cache.h"
#include "forge/model.h"

namespace forge {

class Gemma4Attention {
public:
    explicit Gemma4Attention(KVCache& kv_cache) : kv_cache_(kv_cache) {}

    // full-attention 层的 Proportional RoPE 频率因子, 由 engine 在权重加载后注入。
    void set_rope_freqs(TensorPtr freqs) { rope_freqs_ = std::move(freqs); }

    TensorPtr attend(const TensorPtr& normed, const LayerExecutionContext& lctx);

private:
    KVCache& kv_cache_;
    TensorPtr rope_freqs_;
    TensorPtr rope_freqs_cpu_;
};

}  // namespace forge
