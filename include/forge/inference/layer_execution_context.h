#pragma once

// LayerExecutionContext: 单层前向所需的全部显式输入。
//
// 重构前 forward_layer(hidden, layer_idx, seq_len, start_pos, dev, seq_id) 用六个
// 位置参数传递运行时状态, 每个 engine 都要自己把它们再逐层转发给 attention / FFN /
// RoPE, 参数顺序写错不会被编译器发现。LayerExecutionContext 把这些收敛为一个对象,
// 并直接引用阶段 1 的 ForwardRequest, 保证 start_pos / seq_id 全程只有一个来源。
//
// 本阶段不改变执行语义: engine 内部仍然是同一套算子调用顺序, 只是参数传递方式变化。
// 阶段 3 拆分 executor 时, 各 executor 直接接收本对象而不是重新推导。

#include "forge/inference/forward_request.h"
#include "forge/tensor.h"

namespace forge {

struct ModelConfig;
struct LayerWeights;

struct LayerExecutionContext {
    // 模型级配置。
    const ModelConfig& config;

    // 当前层的权重。engine 不再需要用 layer_idx 反查 weights_.layers[]。
    const LayerWeights& weights;

    // 本次前向请求。start_pos / seq_id / n_tokens 的唯一来源。
    const ForwardRequest& request;

    // 当前层索引。用于 KV cache 定位和层级日志。
    int layer_idx = 0;

    // 本层的执行设备。可能与其他层不同(gpu_layers offload)。
    // DeviceTarget carries both device type and GPU device_id for multi-GPU setups.
    DeviceTarget device = DeviceTarget::cpu();

    // ---- 便捷访问 ----
    int seq_len() const { return request.n_tokens; }
    int64_t start_pos() const { return request.start_pos; }
    int seq_id() const { return request.seq_id; }
    bool is_prefill() const { return request.is_prefill; }
};

}  // namespace forge
