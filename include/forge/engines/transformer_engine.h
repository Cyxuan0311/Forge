#pragma once

#include "forge/context.h"
#include "forge/engine.h"
#include "forge/inference/execution_plan.h"
#include "forge/inference/graph/graph_runtime.h"
#include "forge/inference/layer_execution_context.h"
#include "forge/kv_cache.h"
#include "forge/kv_memory.h"
#include "forge/model.h"

namespace forge {

class TransformerEngine : public InferenceEngine {
public:
    explicit TransformerEngine(Model& model, InferenceContext& ctx);

    TensorPtr forward_request(const ForwardRequest& req) override;
    TensorPtr forward_batch(const InferenceBatch& batch) override;
    TensorPtr forward_from_hidden(const TensorPtr& hidden, int64_t start_pos) override;
    void reset() override;
    void set_gpu_layers(int gpu_layers) override;
    // Multi-GPU overload: gpu_layers_per_dev[i] = layers on GPU i
    void set_gpu_layers(int gpu_layers, const std::vector<int>& gpu_layers_per_dev);
    int gpu_layers() const override { return gpu_layers_; }
    void set_kv_cache_dtype(KVCacheDType dtype) { kv_cache_dtype_ = dtype; }
    KVCacheDType kv_cache_dtype() const { return kv_cache_dtype_; }

    KVCache* kv_cache() override { return &kv_cache_; }
    const KVCache* kv_cache() const override { return &kv_cache_; }
    KVCache& kv_cache_ref() { return kv_cache_; }
    const KVCache& kv_cache_ref() const { return kv_cache_; }

    KVMemory* kv_memory() override { return kv_memory_.get(); }
    const KVMemory* kv_memory() const override { return kv_memory_.get(); }

    // InferenceContext 通过此入口在首次 forward 之前完成 KV cache 分配,
    // 使 CLI/bindings 可以在推理前读取 cache 统计信息。
    void init_memory() override { init_kv_cache(model_.config()); }

    // Access to unified model weights
    const ModelWeights& weights() const { return weights_; }
    ModelWeights& weights() { return weights_; }

    // Graph-based execution mode
    void set_use_graph(bool use_graph) { use_graph_ = use_graph; }
    bool use_graph() const { return use_graph_; }

    // CUDA Graph for decode (Phase 10). Default off.
    // Requires graph mode (use_graph_=true) and CUDA device.
    void set_cuda_graph_enabled(bool v) { graph_runtime_.set_cuda_graph_enabled(v); }
    bool cuda_graph_enabled() const { return graph_runtime_.cuda_graph_enabled(); }

    // 本 engine 的执行计划。在构造时生成一次, 执行期只读。
    const ExecutionPlan& plan() const { return plan_; }

protected:
    // 单层前向。所有运行时状态由 LayerExecutionContext 携带, 不再使用位置参数。
    virtual TensorPtr forward_layer(const TensorPtr& hidden,
                                    const LayerExecutionContext& lctx) = 0;
    virtual bool init_weights() = 0;

    virtual void init_kv_cache(const ModelConfig& cfg);

    // KV cache 是否已分配。状态存放在 context 持有的 InferenceMemory 上,
    // engine 不再自己维护一份, 避免两个所有者各自 init。
    bool kv_cache_initialized() const { return memory_.initialized(); }
    void set_kv_cache_initialized(bool v) { memory_.set_initialized(v); }

    // 按层构造 LayerExecutionContext。集中处理权重查找和 per-layer device 分配,
    // 避免每个 engine 各自实现一套。
    LayerExecutionContext make_layer_context(int layer_idx, const ForwardRequest& req,
                                             DeviceTarget dev) const;

    // Request-based layer driving. The ForwardRequest carries n_tokens, start_pos
    // and seq_id, so no layer can silently drop or reorder them.
    TensorPtr forward_layers(const TensorPtr& hidden, const ForwardRequest& req);

    DeviceType layer_device(int layer_idx) const;
    DeviceTarget layer_device_target(int layer_idx) const;
    const std::vector<DeviceTarget>& layer_devices() const { return layer_devices_; }
    TensorPtr transfer_hidden(const TensorPtr& hidden, DeviceTarget target) const;

    void apply_rope_standard(const float* q_data, const float* k_data, float* q_out, float* k_out,
                             int seq_len, int num_heads, int num_kv_heads, int head_dim,
                             int64_t start_pos, float theta);

    TensorPtr expand_kv_heads(const TensorPtr& kv, int seq_len, int num_heads, int num_kv_heads,
                              int head_dim, DeviceType dev);

    Model& model_;
    InferenceContext& ctx_;
    ExecutionPlan plan_;
    // Memory 由 InferenceContext 持有, engine 只引用。
    InferenceMemory& memory_;
    KVCache& kv_cache_;
    std::unique_ptr<KVMemory> kv_memory_;
    ModelWeights weights_;
    KVCacheDType kv_cache_dtype_ = KVCacheDType::FP32;
    int gpu_layers_ = -1;
    std::vector<DeviceTarget> layer_devices_;  // per-layer (device_type, device_id) assignment
    bool use_graph_ = false;
    // graph 的构建/缓存/执行全部由 GraphRuntime 负责, builder 来自 ExecutionPlan。
    GraphRuntime graph_runtime_;
};

}  // namespace forge
