#include "forge/inference/graph/graph_runtime.h"

#include <chrono>

#include "forge/backend_scheduler.h"
#include "forge/logger.h"

namespace forge {

void GraphRuntime::inject_runtime(const ForwardRequest& req) {
    runtime_state_->start_pos = req.start_pos;
    runtime_state_->seq_id = req.seq_id;
    for (const auto& slot : start_pos_slots_) {
        graph_->patch_node_param_i64(slot.node_idx, slot.int32_offset, req.start_pos);
    }
}

void GraphRuntime::build(const TensorPtr& hidden, const ModelConfig& cfg, ModelWeights& weights,
                         KVCache& kv_cache, const std::vector<DeviceTarget>& layer_devices,
                         DeviceType default_device, int seq_len) {
    graph_ = std::make_unique<ComputeGraph>();
    start_pos_slots_.clear();

    int cur_idx = graph_->add_input(hidden);

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        DeviceType dev = (layer < static_cast<int>(layer_devices.size())) ? layer_devices[layer].type
                                                                         : default_device;
        // 第一层 hidden_idx=0 是 graph input; 后续层 cur_idx 是节点索引,
        // 必须用负数引用(ref)以免被 execute() 误当作 graph input。
        int hidden_ref = (layer == 0) ? cur_idx : -(cur_idx + 1);
        GraphBuildContext bctx{*graph_,     cfg,   weights.layers[layer], kv_cache,
                               runtime_state_, &start_pos_slots_, hidden_ref, layer,
                               seq_len,     dev,   recurrent_memory_};
        cur_idx = builder_->build_layer(bctx);
    }

    int output_idx = builder_->build_output(*graph_, cur_idx, weights, cfg);
    (void)output_idx;

    BackendScheduler scheduler;
    SchedulingPlan sched = scheduler.schedule(*graph_);
    if (sched.valid) {
        graph_->apply_schedule(sched);
    }

    // Phase 3: insert cross-device copy nodes after device assignment.
    // This converts implicit auto_transfer() calls into explicit graph COPY nodes,
    // enabling proper lifetime tracking and per-backend buffer planning.
    int copies_inserted = graph_->insert_copy_nodes();
    if (copies_inserted > 0) {
        LOG_INFO("GraphRuntime: " + std::to_string(copies_inserted) +
                 " cross-device copies inserted");
    }
}

void GraphRuntime::set_cuda_graph_enabled(bool v) {
#ifdef USE_CUDA
    if (v && !cuda_graph_runner_) {
        cuda_graph_runner_ = std::make_unique<CudaGraphRunner>();
    }
    if (cuda_graph_runner_) {
        cuda_graph_runner_->set_enabled(v);
    }
#else
    (void)v;
#endif
}

bool GraphRuntime::cuda_graph_enabled() const {
#ifdef USE_CUDA
    return cuda_graph_runner_ && cuda_graph_runner_->enabled();
#else
    return false;
#endif
}

TensorPtr GraphRuntime::run(const TensorPtr& hidden, const ForwardRequest& req,
                            const GraphKey& key, const ModelConfig& cfg, ModelWeights& weights,
                            KVCache& kv_cache, const std::vector<DeviceTarget>& layer_devices,
                            DeviceType default_device) {
    if (!builder_) return nullptr;

    auto t0 = std::chrono::steady_clock::now();

    const bool reuse = graph_ && key == cached_key_;
    if (reuse) {
        graph_->set_input(0, hidden);
    } else {
        build(hidden, cfg, weights, kv_cache, layer_devices, default_device, req.n_tokens);
        cached_key_ = key;
#ifdef USE_CUDA
        // Graph topology changed — invalidate CUDA Graph
        if (cuda_graph_runner_) cuda_graph_runner_->invalidate();
#endif
    }

    inject_runtime(req);

    // Phase 10: Try CUDA Graph replay/capture first.
    // CUDA Graph is only used for decode (non-prefill) on compatible graphs.
    bool used_cuda_graph = false;

#ifdef USE_CUDA
    if (cuda_graph_runner_ && cuda_graph_runner_->enabled() && !key.is_prefill) {
        // Try replay or capture. The execute_fn wraps graph_->execute().
        used_cuda_graph = cuda_graph_runner_->try_replay_or_capture(
            *graph_, [this]() { graph_->execute(); }, /*stream=*/0);
    }
#endif

    TensorPtr result;
    if (!used_cuda_graph) {
        // Normal execution path (fallback or CUDA Graph disabled)
        result = graph_->execute();
    } else {
        result = graph_->last_output();
    }

    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::string mode = reuse ? "reused" : "built";
    if (used_cuda_graph) mode = "cuda_graph";
    LOG_INFO(std::string("Forward (graph, ") + mode +
             ") total: " + std::to_string((int)total_ms) + "ms (" + key.to_string() +
             ", start_pos=" + std::to_string(req.start_pos) +
             ", nodes=" + std::to_string(graph_->num_nodes()) + ")");

    return result;
}

}  // namespace forge
