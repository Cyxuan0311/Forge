#pragma once

#include "forge/context.h"
#include "forge/engine.h"
#include "forge/engines/layer_graph_builder.h"
#include "forge/kv_cache.h"
#include "forge/memory_pool.h"
#include "forge/model.h"

namespace forge {

struct GraphCache {
    std::unique_ptr<ComputeGraph> graph;
    int cached_seq_len = -1;
    int cached_gpu_layers = -1;
    std::string cached_arch;
    bool valid = false;

    bool can_reuse(int seq_len, int gpu_layers, const std::string& arch) const {
        return valid && graph
               && seq_len == cached_seq_len
               && gpu_layers == cached_gpu_layers
               && arch == cached_arch;
    }

    void invalidate() { valid = false; }
};

class TransformerEngine : public InferenceEngine {
public:
    explicit TransformerEngine(Model& model, InferenceContext& ctx);

    TensorPtr forward(const TensorPtr& input_ids, int64_t start_pos, int seq_id = 0) override;
    TensorPtr forward_batch(const InferenceBatch& batch) override;
    TensorPtr forward_from_hidden(const TensorPtr& hidden, int64_t start_pos) override;
    void reset() override;
    void set_gpu_layers(int gpu_layers) override;
    int gpu_layers() const override { return gpu_layers_; }
    void set_kv_cache_dtype(KVCacheDType dtype) { kv_cache_dtype_ = dtype; }
    KVCacheDType kv_cache_dtype() const { return kv_cache_dtype_; }

    KVCache* kv_cache() override { return &kv_cache_; }
    const KVCache* kv_cache() const override { return &kv_cache_; }
    KVCache& kv_cache_ref() { return kv_cache_; }
    const KVCache& kv_cache_ref() const { return kv_cache_; }

    // Access to unified model weights
    const ModelWeights& weights() const { return weights_; }
    ModelWeights& weights() { return weights_; }

    // Graph-based execution mode
    void set_use_graph(bool use_graph) { use_graph_ = use_graph; }
    bool use_graph() const { return use_graph_; }

protected:
    virtual TensorPtr forward_layer(const TensorPtr& hidden, int layer_idx, int seq_len,
                                    int64_t start_pos, DeviceType dev, int seq_id = 0) = 0;
    virtual bool init_weights() = 0;

    virtual void init_kv_cache(const ModelConfig& cfg);
    TensorPtr forward_layers(const TensorPtr& hidden, int seq_len, int64_t start_pos, int seq_id = 0);
    TensorPtr forward_layers_graph(const TensorPtr& hidden, int seq_len, int64_t start_pos,
                                   int seq_id = 0);

    DeviceType layer_device(int layer_idx) const;
    const std::vector<DeviceType>& layer_devices() const { return layer_devices_; }
    TensorPtr transfer_hidden(const TensorPtr& hidden, DeviceType target) const;

    void apply_rope_standard(const float* q_data, const float* k_data, float* q_out, float* k_out,
                             int seq_len, int num_heads, int num_kv_heads, int head_dim,
                             int64_t start_pos, float theta);

    TensorPtr expand_kv_heads(const TensorPtr& kv, int seq_len, int num_heads, int num_kv_heads,
                              int head_dim, DeviceType dev);

    Model& model_;
    InferenceContext& ctx_;
    ModelWeights weights_;
    KVCache kv_cache_;
    bool kv_cache_initialized_ = false;
    KVCacheDType kv_cache_dtype_ = KVCacheDType::FP32;
    MemoryPool workspace_pool_;
    int gpu_layers_ = -1;
    std::vector<DeviceType> layer_devices_;  // per-layer device assignment
    bool use_graph_ = false;
    std::unique_ptr<LayerGraphBuilder> graph_builder_;
    GraphCache graph_cache_;
};

}  // namespace forge
