#pragma once

// InferenceMemory: 推理期可变状态的唯一所有者接口。
//
// 重构前 KV cache 有两个所有者: InferenceContext 持有一个 unique_ptr<KVCache>,
// TransformerEngine 又持有一个 KVCache 成员。两者都会各自 init, 结果是 CLI 调用
// ctx.init_kv_cache() 会白白多分配一份显存(context.cpp 原有注释已记录该问题),
// 且 reset 语义分散在两处。
//
// 现在 InferenceContext 是唯一 owner, engine 只持有引用。engine 的 kv_cache()
// 保留为兼容转发接口, 现有 scheduler / generator / bindings / CLI 不受影响。
//
// 阶段 8 将 Qwen3.5 的 recurrent state 所有权也收归 context: 同时需要 KV cache
// 和 recurrent state 的架构使用 HybridMemory, 单纯 KV cache 的继续用 KvMemory。

#include "forge/inference/layers/qwen35_recurrent_memory.h"
#include "forge/kv_cache.h"

namespace forge {

class InferenceMemory {
public:
    virtual ~InferenceMemory() = default;

    // 释放内容并回到未初始化状态, 下一次 forward 会按当前配置重新分配。
    virtual void reset() = 0;

    // 是否已完成分配。分配本身由 engine 执行, 因为只有 engine 知道该架构需要的
    // 层数、per-layer KV 维度和 ring buffer 配置。
    virtual bool initialized() const = 0;
    virtual void set_initialized(bool v) = 0;

    // 该 memory 承载的 KV cache。没有 KV cache 的实现返回 nullptr。
    virtual KVCache* kv() = 0;
    virtual const KVCache* kv() const = 0;

    // 该 memory 承载的 recurrent state (Qwen3.5 linear attention)。没有
    // recurrent state 的架构返回 nullptr。
    virtual Qwen35RecurrentMemory* recurrent() { return nullptr; }
    virtual const Qwen35RecurrentMemory* recurrent() const { return nullptr; }
};

// 标准 KV cache memory。Gemma4 的 per-layer KV 布局同样使用它,
// 区别只在于 engine 调用 init_per_layer() 而不是 init_quantized()。
class KvMemory final : public InferenceMemory {
public:
    void reset() override {
        cache_.reset();
        initialized_ = false;
    }

    bool initialized() const override { return initialized_; }
    void set_initialized(bool v) override { initialized_ = v; }

    KVCache* kv() override { return &cache_; }
    const KVCache* kv() const override { return &cache_; }

private:
    KVCache cache_;
    bool initialized_ = false;
};

// 同时持有 KV cache 和 Qwen3.5 recurrent state 的 memory, 供 Qwen3.5 hybrid 架构使用。
class HybridMemory final : public InferenceMemory {
public:
    void reset() override {
        cache_.reset();
        recurrent_.reset();
        initialized_ = false;
    }

    bool initialized() const override { return initialized_; }
    void set_initialized(bool v) override { initialized_ = v; }

    KVCache* kv() override { return &cache_; }
    const KVCache* kv() const override { return &cache_; }

    Qwen35RecurrentMemory* recurrent() override { return &recurrent_; }
    const Qwen35RecurrentMemory* recurrent() const override { return &recurrent_; }

private:
    KVCache cache_;
    Qwen35RecurrentMemory recurrent_;
    bool initialized_ = false;
};

}  // namespace forge
