#include "nanoinfer/engines/qwen35_graph_builder.h"
#include "nanoinfer/operators.h"
#include "nanoinfer/logger.h"

namespace nanoinfer {

int Qwen35GraphBuilder::build_layer_graph(ComputeGraph& graph,
                                           int hidden_idx,
                                           const LayerWeights& lw,
                                           const ModelConfig& cfg,
                                           int layer_idx,
                                           int seq_len,
                                           int64_t start_pos,
                                           DeviceType dev,
                                           KVCache& kv_cache) {
    // Qwen35 has complex stateful operations (SSM, MRoPE, Gated Delta Net)
    // that are tightly coupled with Qwen35Engine's internal state.
    //
    // Current status: The graph builder for Qwen35 is a placeholder.
    // The actual layer computation is handled by Qwen35Engine::forward_layer
    // via the imperative path in TransformerEngine::forward_layers.
    //
    // Future work: Extract SSM state management into first-class graph nodes
    // so that Qwen35 can also benefit from graph-based optimizations.
    //
    // For now, when use_graph=true and arch=qwen35, the system will
    // fall back to imperative mode automatically (see TransformerEngine::forward_layers).

    LOG_WARN("Qwen35GraphBuilder: graph-based execution not yet fully supported for Qwen35");
    return -1;  // Signal that graph building is not supported
}

int Qwen35GraphBuilder::build_output_graph(ComputeGraph& graph,
                                            int hidden_idx,
                                            const ModelWeights& weights,
                                            const ModelConfig& cfg) {
    // Output norm
    int norm_idx = graph.add_node("output_norm", "rms_norm", {hidden_idx},
        [&weights, eps = cfg.rms_norm_eps](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            return ops::rms_norm(inputs[0], weights.output_norm, eps);
        });

    // Output projection
    int logits_idx = graph.add_node("output_proj", "matmul_transB", {norm_idx},
        [&weights, tie = cfg.tie_embeddings](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            auto output_weight = weights.output_weight;
            if (!output_weight && tie) {
                output_weight = weights.token_embedding;
            }
            return ops::matmul_transB(inputs[0], output_weight);
        });

    return logits_idx;
}

// Note: Qwen35 graph builder is NOT registered in GraphBuilderRegistry yet
// because full graph-based execution is not yet supported for this architecture.
// When use_graph=true and arch=qwen35, TransformerEngine will fall back to
// imperative mode automatically.
//
// To enable in the future, uncomment the following:
// static GraphBuilderAutoRegister _reg_qwen35_gb("qwen35", []() -> std::unique_ptr<LayerGraphBuilder> {
//     return std::make_unique<Qwen35GraphBuilder>();
// });

} // namespace nanoinfer
