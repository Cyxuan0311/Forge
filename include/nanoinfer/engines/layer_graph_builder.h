#pragma once

#include "nanoinfer/compute_graph.h"
#include "nanoinfer/model.h"
#include "nanoinfer/kv_cache.h"
#include <string>
#include <memory>

namespace nanoinfer {

// Forward declarations
class KVCache;

/**
 * LayerGraphBuilder - Architecture-specific graph builder for transformer layers.
 *
 * Each architecture (LLaMA, DeepSeek, Qwen35) implements this interface to
 * declare the computation graph for a single transformer layer. The graph
 * is then executed by the ComputeGraph scheduler, enabling optimizations
 * like operator fusion, memory reuse, and device offloading.
 */
class LayerGraphBuilder {
public:
    virtual ~LayerGraphBuilder() = default;

    virtual std::string name() const = 0;

    /**
     * Build the computation graph for a single transformer layer.
     *
     * @param graph      The ComputeGraph to add nodes to
     * @param hidden_idx Index of the hidden state input in the graph
     * @param lw         Layer weights (unified)
     * @param cfg        Model configuration
     * @param layer_idx  Which layer is being built
     * @param seq_len    Sequence length for this forward pass
     * @param start_pos  Start position for RoPE
     * @param dev        Target device for this layer
     * @param kv_cache   KV cache for attention
     * @return Index of the output node in the graph
     */
    virtual int build_layer_graph(ComputeGraph& graph,
                                  int hidden_idx,
                                  const LayerWeights& lw,
                                  const ModelConfig& cfg,
                                  int layer_idx,
                                  int seq_len,
                                  int64_t start_pos,
                                  DeviceType dev,
                                  KVCache& kv_cache) = 0;

    /**
     * Build the output head graph (output norm + projection).
     *
     * @param graph       The ComputeGraph to add nodes to
     * @param hidden_idx  Index of the hidden state input
     * @param weights     Model-level weights
     * @param cfg         Model configuration
     * @return Index of the output node
     */
    virtual int build_output_graph(ComputeGraph& graph,
                                   int hidden_idx,
                                   const ModelWeights& weights,
                                   const ModelConfig& cfg) = 0;
};

/**
 * Registry for LayerGraphBuilder instances, keyed by architecture name.
 */
class GraphBuilderRegistry {
public:
    using BuilderCreator = std::function<std::unique_ptr<LayerGraphBuilder>()>;

    static GraphBuilderRegistry& instance();

    void register_builder(const std::string& arch, BuilderCreator creator);
    std::unique_ptr<LayerGraphBuilder> create(const std::string& arch) const;
    std::vector<std::string> registered_archs() const;

private:
    GraphBuilderRegistry() = default;
    std::unordered_map<std::string, BuilderCreator> creators_;
};

struct GraphBuilderAutoRegister {
    GraphBuilderAutoRegister(const std::string& arch, GraphBuilderRegistry::BuilderCreator creator);
};

#define NANOINFER_REGISTER_GRAPH_BUILDER(arch, creator) \
    static ::nanoinfer::GraphBuilderAutoRegister _graph_reg_##arch(#arch, creator)

} // namespace nanoinfer
