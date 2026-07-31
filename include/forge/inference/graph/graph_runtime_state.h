#pragma once

// GraphRuntimeState: 同一 graph 拓扑在不同调用间变化的运行期量。
//
// 重构前 start_pos 和 seq_id 被 graph builder 以值捕获进 lambda 和 op_params,
// 于是缓存图一旦复用就永久停在构建时的位置上, 交错 decode 或不同 start_pos 会读到
// 错误的 KV slot。这些量不进 GraphKey (不影响拓扑), 而是放在共享状态里,
// 由 engine 在每次 execute 前写入, 图内节点执行时再读。

#include <cstdint>
#include <memory>

namespace forge {

struct GraphRuntimeState {
    int64_t start_pos = 0;
    int seq_id = 0;
};

using GraphRuntimeStatePtr = std::shared_ptr<GraphRuntimeState>;

// 走 OpDispatch 的节点(如 ROPE)从固定大小的 op_params 读取 start_pos, 无法通过
// 闭包访问 GraphRuntimeState。构建时记录参数槽位置, 执行前由 GraphRuntime 回填。
struct StartPosSlot {
    int node_idx = -1;
    int int32_offset = 0;  // op_params 中 int64_t start_pos 的起始下标
};

}  // namespace forge
