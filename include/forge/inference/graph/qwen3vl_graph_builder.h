#pragma once

#include "forge/inference/graph/layer_graph_builder.h"

namespace forge {

// Qwen3-VL: standard GQA decoder with per-head QK-Norm and MRoPE.
// The attention (QK-Norm + MRoPE + KV write + SDPA + output proj) is
// encapsulated as a single node (Qwen3vlFullAttention) reusing the engine's
// exact executors, while the FFN is decomposed into primitive CUDA ops for the
// real speedup. This mirrors Qwen35GraphBuilder's structure.
class Qwen3vlGraphBuilder : public LayerGraphBuilder {
public:
    std::string name() const override { return "qwen3vl"; }

    int build_layer(const GraphBuildContext& bctx) override;

    int build_output(ComputeGraph& graph, int hidden_idx, const ModelWeights& weights,
                     const ModelConfig& cfg, DeviceType dev) override;
};

}  // namespace forge
