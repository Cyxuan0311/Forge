#pragma once

// LayerGraphBuilder: 按架构声明单层计算图。
//
// 与重构前的差别有三点:
//   1. 位置参数收敛为 GraphBuildContext, 与 LayerExecutionContext 保持同一形态,
//      新增一个运行期量不再需要修改所有 builder 的签名。
//   2. start_pos 不再由 builder 捕获。builder 只在 op_params 里留出槽位并登记,
//      具体值由 GraphRuntime 在每次执行前回填, 因此缓存图可以跨 start_pos 复用。
//   3. 注册从静态对象改为显式函数调用, 不再依赖静态库的 force-link 行为。

#include <memory>
#include <string>
#include <vector>

#include "forge/compute_graph.h"
#include "forge/inference/graph/graph_runtime_state.h"
#include "forge/kv_cache.h"
#include "forge/model.h"

namespace forge {

struct GraphBuildContext {
    ComputeGraph& graph;
    const ModelConfig& config;
    const LayerWeights& weights;
    KVCache& kv_cache;

    // 执行期共享状态。builder 把它按值捕获进 lambda(shared_ptr 拷贝),
    // 节点执行时读到的是 engine 最新写入的 start_pos / seq_id。
    GraphRuntimeStatePtr runtime;

    // builder 登记需要执行前回填 start_pos 的 op_params 槽。
    std::vector<StartPosSlot>* start_pos_slots = nullptr;

    int hidden_idx = 0;
    int layer_idx = 0;
    int seq_len = 0;
    DeviceType device = DeviceType::CPU;

    void record_start_pos_slot(int node_idx, int int32_offset) const {
        if (start_pos_slots) {
            start_pos_slots->push_back(StartPosSlot{node_idx, int32_offset});
        }
    }
};

class LayerGraphBuilder {
public:
    virtual ~LayerGraphBuilder() = default;

    virtual std::string name() const = 0;

    // 返回本层输出节点在图中的下标。
    virtual int build_layer(const GraphBuildContext& bctx) = 0;

    // 输出头(output norm + projection)。不涉及运行期状态, 保持简单签名。
    virtual int build_output(ComputeGraph& graph, int hidden_idx, const ModelWeights& weights,
                            const ModelConfig& cfg) = 0;
};

// 显式创建内建 graph builder。返回 nullptr 表示该架构不支持 graph 执行。
//
// 重构前依赖静态初始化对象注册 + GraphBuilderRegistry 单例查表, 而 builder 位于
// 静态库中: 若没有其他符号引用该 translation unit, 链接器会整体丢弃, 注册表在运行
// 时为空, 表现为"某架构突然不支持 graph"。阶段 8 删除了 registry, 改为显式工厂函数,
// ExecutionPlan 在构建期直接调用, 消除对链接顺序的隐式依赖。
std::shared_ptr<LayerGraphBuilder> create_graph_builder(const std::string& arch);

}  // namespace forge
