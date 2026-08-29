#pragma once

#include "forge/context.h"
#include "forge/engine.h"
#include "forge/engines/expert_page_cache.h"
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

    // ---- Expert-level placement (MoE partial-activation, Phase P0+) ----
    // P0 brings the per-expert physical-location mapping into view and exposes
    // a sync hook; it does NOT change weight movement or computation. Later
    // phases (ExpertPageCache) use expert_device() to page experts in/out.
    //
    // Assign which experts of a MoE layer live on the GPU. Default (never
    // called) = every expert inherits the layer's device (current behavior).
    void set_expert_placement(int layer, const std::vector<int>& gpu_experts,
                              int n_expert);
    // Where a specific (layer, expert) currently resides. Falls back to the
    // layer device when no explicit placement was configured.
    DeviceTarget expert_device(int layer, int expert) const;
    bool has_expert_placement() const { return !expert_devices_.empty(); }

    // MoE expert-activation hook. Called by MoE operators right after the
    // router picks the top-k experts for a token, before any expert GEMV runs.
    // P1: records the routed experts in ExpertPageCache (and, when paging is
    // enabled, moves each per-expert weight view onto its target device).
    virtual void sync_experts_resident(int layer,
                                       const std::vector<int>& active_experts) const;

    // Enable per-expert movement (ExpertPageCache::record_active paging_enabled).
    // Off by default: P1 only collects router statistics, behavior unchanged.
    void set_expert_paging(bool on) { expert_paging_enabled_ = on; }
    bool expert_paging_enabled() const { return expert_paging_enabled_; }
    // Print the accumulated router-distribution report (--expert-stats).
    std::string expert_stats_report() const { return expert_page_cache_.format_report(); }

    // ---- P2: per-expert residency control ------------------------------------
    // VRAM budget for resident (non-CPU) experts. 0 = unbounded.
    void set_expert_budget_bytes(int64_t b) { expert_page_cache_.set_budget_bytes(b); }
    int64_t expert_budget_bytes() const { return expert_page_cache_.budget_bytes(); }
    int64_t expert_resident_bytes() const { return expert_page_cache_.resident_bytes(); }

    // Per-expert weight lookup for compute paths. Returns nullptr when paging is
    // off or the expert has not been materialised yet, so callers fall back to
    // slicing the monolithic 3D tensor (keeps the non-paging path unchanged).
    TensorPtr expert_weight(int layer, int expert, ExpertSlot slot) const {
        return expert_page_cache_.expert_weight(layer, expert, slot);
    }
    bool expert_materialized(int layer, int expert) const {
        return expert_page_cache_.is_materialized(layer, expert);
    }
    KVCacheDType kv_cache_dtype() const { return kv_cache_dtype_; }

    KVCache* kv_cache() override { return &kv_cache_; }
    const KVCache* kv_cache() const override { return &kv_cache_; }

    TensorPtr take_last_hidden() override {
        auto h = last_hidden_;
        last_hidden_ = nullptr;
        return h;
    }
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
    TensorPtr last_hidden_;  // post-final-norm hidden from last forward_request
    std::unique_ptr<KVMemory> kv_memory_;
    ModelWeights weights_;
    KVCacheDType kv_cache_dtype_ = KVCacheDType::FP32;
    int gpu_layers_ = -1;
    std::vector<DeviceTarget> layer_devices_;  // per-layer (device_type, device_id) assignment
    // Per-(layer, expert) device assignment for MoE partial activation.
    // Empty until set_expert_placement() is called for at least one layer.
    std::vector<std::vector<DeviceTarget>> expert_devices_;

    // Build the (src3d, slots, expert_dim) triple describing a layer's expert
    // weights, used for lazy materialisation. Returns false if not a MoE layer.
    bool expert_source(int layer, std::vector<TensorPtr>& src3d,
                       std::vector<ExpertSlot>& slots, int& expert_dim) const;

    // Phase P1: per-expert residency + router stats for partial activation.
    mutable ExpertPageCache expert_page_cache_;
    bool expert_paging_enabled_ = false;
    mutable int64_t expert_step_ = 0;  // generation step counter for LRU
    bool use_graph_ = false;
    // graph 的构建/缓存/执行全部由 GraphRuntime 负责, builder 来自 ExecutionPlan。
    GraphRuntime graph_runtime_;
};

}  // namespace forge
