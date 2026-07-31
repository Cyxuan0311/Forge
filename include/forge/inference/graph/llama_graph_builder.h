#pragma once

#include "forge/inference/graph/layer_graph_builder.h"

namespace forge {

// GQA 系架构(llama / mistral / qwen / qwen2 / yi)的单层图。
class LlamaGraphBuilder : public LayerGraphBuilder {
public:
    std::string name() const override { return "llama"; }

    int build_layer(const GraphBuildContext& bctx) override;

    int build_output(ComputeGraph& graph, int hidden_idx, const ModelWeights& weights,
                     const ModelConfig& cfg) override;
};

}  // namespace forge
