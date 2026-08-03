#include "forge/inference/graph/layer_graph_builder.h"

#include "forge/inference/graph/deepseek_graph_builder.h"
#include "forge/inference/graph/llama_graph_builder.h"
#include "forge/inference/graph/qwen35_graph_builder.h"

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

    // Qwen3.5 (SSM hybrid: FullAttention + LinearAttention with Gated Delta Net)。
    if (arch == "qwen35") {
        return std::make_shared<Qwen35GraphBuilder>();
    }

    return nullptr;
}

}  // namespace forge
