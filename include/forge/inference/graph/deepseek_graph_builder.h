#pragma once

#include "forge/inference/graph/layer_graph_builder.h"

namespace forge {

// DeepSeek 单层图。按 LayerType 在 GQA 与 MLA 两种 attention 之间分派。
class DeepSeekGraphBuilder : public LayerGraphBuilder {
public:
    std::string name() const override { return "deepseek"; }

    int build_layer(const GraphBuildContext& bctx) override;

    int build_output(ComputeGraph& graph, int hidden_idx, const ModelWeights& weights,
                     const ModelConfig& cfg) override;

private:
    int build_gqa_layer(const GraphBuildContext& bctx);
    int build_mla_layer(const GraphBuildContext& bctx);
};

}  // namespace forge
