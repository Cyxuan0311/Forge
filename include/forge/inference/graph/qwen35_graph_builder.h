#pragma once

#include "forge/inference/graph/layer_graph_builder.h"

namespace forge {

// Qwen3.5 hybrid architecture: FullAttention + LinearAttention (Gated Delta Net).
// Dispatches to build_full_attn_layer or build_linear_attn_layer based on layer_type.
class Qwen35GraphBuilder : public LayerGraphBuilder {
public:
    std::string name() const override { return "qwen35"; }

    int build_layer(const GraphBuildContext& bctx) override;

    int build_output(ComputeGraph& graph, int hidden_idx, const ModelWeights& weights,
                     const ModelConfig& cfg) override;

private:
    int build_full_attn_layer(const GraphBuildContext& bctx);
    int build_linear_attn_layer(const GraphBuildContext& bctx);
};

}  // namespace forge
