#pragma once

// AttentionExecutor: QKV 投影与 attention 计算。
//
// 从 GenericEngine::qkv_proj_forward / attention_forward 抽出。持有 KVCache 引用,
// 因为 attention 需要读取已填充的 K/V slice; KV cache 的 update 仍由 layer 编排层调用,
// 保持与重构前相同的调用顺序。

#include "forge/inference/layer_execution_context.h"
#include "forge/kv_cache.h"
#include "forge/kv_storage.h"
#include "forge/model.h"

namespace forge {

class AttentionExecutor {
public:
    explicit AttentionExecutor(KVCache& kv_cache, KVStorage* storage = nullptr)
        : kv_cache_(kv_cache), storage_(storage) {}

    struct QKVResult {
        TensorPtr q, k, v;
    };

    // QKV 投影, 含融合量化快路径。has_bias 由调用方根据权重是否存在决定。
    static QKVResult project_qkv(const TensorPtr& x, const LayerWeights& lw, bool has_bias,
                                 DeviceType dev, int seq_len);

    // Attention。K/V 从 KV cache 读取, 传入的 q 已经过 RoPE。
    // mask: 可选 [q_len, kv_len] additive bias。
    // seq_id: 当前序列 id (paged CUDA decode 用 per-seq page table; -1 = legacy)
    TensorPtr attend(const TensorPtr& q, const ModelConfig& cfg, int layer_idx, int seq_len,
                     DeviceType dev, const TensorPtr& mask = nullptr, int seq_id = -1,
                     bool causal = true);

    // GQA 的 KV head 复制。DeepSeek/Qwen3.5 的非 flash-attention 路径需要显式展开,
    // 因此放在这里共享, 不再由 TransformerEngine 提供 protected helper。
    static TensorPtr expand_kv_heads(const TensorPtr& kv, int seq_len, int num_heads,
                                     int num_kv_heads, int head_dim, DeviceType dev);

    void set_storage(KVStorage* s) { storage_ = s; }

private:
    KVCache& kv_cache_;
    KVStorage* storage_;  // optional: when non-null and paged, read K/V from storage
};

}  // namespace forge
