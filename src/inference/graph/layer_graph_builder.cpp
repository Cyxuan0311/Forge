#include "forge/inference/graph/layer_graph_builder.h"

#include "forge/inference/graph/deepseek_graph_builder.h"
#include "forge/inference/graph/llama_graph_builder.h"

namespace forge {

// 阶段 8: 删除了 GraphBuilderRegistry 单例, 改为直接工厂函数。
// ExecutionPlan 在构建期调用此函数, 不再经过注册表查表。
std::shared_ptr<LayerGraphBuilder> create_graph_builder(const std::string& arch) {
    // GQA 系列: llama / mistral / qwen / qwen2 / yi 共用同一 builder。
    if (arch == "llama" || arch == "mistral" || arch == "qwen" ||
        arch == "qwen2" || arch == "yi") {
        return std::make_shared<LlamaGraphBuilder>();
    }

    // DeepSeek 系列: deepseek / deepseek_v2 / deepseek_v3 共用同一 builder。
    if (arch == "deepseek" || arch == "deepseek_v2" || arch == "deepseek_v3") {
        return std::make_shared<DeepSeekGraphBuilder>();
    }

    // Qwen3.5 的循环状态尚未建模为 graph 节点, 因此不支持 graph 执行。
    // ExecutionPlan 会据此把 supports_graph 置为 false, 走 imperative 路径,
    // 而不是注册一个返回 -1 的 placeholder builder 再在执行期回退。
    return nullptr;
}

}  // namespace forge
