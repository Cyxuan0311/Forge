#pragma once

// GraphRuntime: graph 模式的构建、缓存与执行。
//
// 重构前这部分逻辑内联在 TransformerEngine::forward_layers_graph 里, 缓存判别只有
// 三个字段, 且 start_pos 被构建时固化。GraphRuntime 把它们收敛为一个组件:
//   - 缓存判别用完整的 GraphKey;
//   - start_pos / seq_id 每次执行前注入 GraphRuntimeState 与 op_params;
//   - engine 只需提供权重和 per-layer 设备, 不再关心图的生命周期。

#include <memory>
#include <string>
#include <vector>

#include "forge/compute_graph.h"
#include "forge/inference/forward_request.h"
#include "forge/inference/graph/graph_key.h"
#include "forge/inference/graph/layer_graph_builder.h"
#include "forge/kv_cache.h"
#include "forge/model.h"

namespace forge {

class GraphRuntime {
public:
    GraphRuntime() : runtime_state_(std::make_shared<GraphRuntimeState>()) {}

    void set_builder(std::shared_ptr<LayerGraphBuilder> builder) {
        builder_ = std::move(builder);
        invalidate();
    }
    bool has_builder() const { return builder_ != nullptr; }
    const LayerGraphBuilder* builder() const { return builder_.get(); }

    void invalidate() {
        graph_.reset();
        cached_key_ = GraphKey{};
        start_pos_slots_.clear();
    }

    // 执行一次前向。图不可复用时先重建。返回 logits。
    TensorPtr run(const TensorPtr& hidden, const ForwardRequest& req, const GraphKey& key,
                  const ModelConfig& cfg, ModelWeights& weights, KVCache& kv_cache,
                  const std::vector<DeviceType>& layer_devices, DeviceType default_device);

    int num_nodes() const { return graph_ ? graph_->num_nodes() : 0; }

private:
    void build(const TensorPtr& hidden, const ModelConfig& cfg, ModelWeights& weights,
               KVCache& kv_cache, const std::vector<DeviceType>& layer_devices,
               DeviceType default_device, int seq_len);

    // 把本次请求的 start_pos / seq_id 写入图: 闭包节点通过 runtime_state_ 读,
    // OpDispatch 节点通过回填 op_params 读。
    void inject_runtime(const ForwardRequest& req);

    std::shared_ptr<LayerGraphBuilder> builder_;
    std::unique_ptr<ComputeGraph> graph_;
    GraphKey cached_key_;
    GraphRuntimeStatePtr runtime_state_;
    std::vector<StartPosSlot> start_pos_slots_;
};

}  // namespace forge
