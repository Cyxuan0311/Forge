#pragma once

#include "nanoinfer/engines/layer_graph_builder.h"

namespace nanoinfer {

class DeepSeekGraphBuilder : public LayerGraphBuilder {
public:
    std::string name() const override { return "deepseek"; }

    int build_layer_graph(ComputeGraph& graph,
                          int hidden_idx,
                          const LayerWeights& lw,
                          const ModelConfig& cfg,
                          int layer_idx,
                          int seq_len,
                          int64_t start_pos,
                          DeviceType dev,
                          KVCache& kv_cache) override;

    int build_output_graph(ComputeGraph& graph,
                           int hidden_idx,
                           const ModelWeights& weights,
                           const ModelConfig& cfg) override;

private:
    int build_gqa_layer(ComputeGraph& graph, int hidden_idx,
                        const LayerWeights& lw, const ModelConfig& cfg,
                        int layer_idx, int seq_len, int64_t start_pos,
                        DeviceType dev, KVCache& kv_cache);

    int build_mla_layer(ComputeGraph& graph, int hidden_idx,
                        const LayerWeights& lw, const ModelConfig& cfg,
                        int layer_idx, int seq_len, int64_t start_pos,
                        DeviceType dev, KVCache& kv_cache);
};

} // namespace nanoinfer
