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
                         KVCache& kv_cache, const std::vector<DeviceType>& layer_devices,
                         DeviceType default_device, int seq_len) {
    graph_ = std::make_unique<ComputeGraph>();
    start_pos_slots_.clear();

    int cur_idx = graph_->add_input(hidden);

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        DeviceType dev = (layer < static_cast<int>(layer_devices.size())) ? layer_devices[layer]
                                                                         : default_device;
        // 第一层 hidden_idx=0 是 graph input; 后续层 cur_idx 是节点索引,
        // 必须用负数引用(ref)以免被 execute() 误当作 graph input。
        int hidden_ref = (layer == 0) ? cur_idx : -(cur_idx + 1);
        GraphBuildContext bctx{*graph_,     cfg,   weights.layers[layer], kv_cache,
                               runtime_state_, &start_pos_slots_, hidden_ref, layer,
                               seq_len,     dev};
        cur_idx = builder_->build_layer(bctx);
    }

    int output_idx = builder_->build_output(*graph_, cur_idx, weights, cfg);
    (void)output_idx;

    BackendScheduler scheduler;
    SchedulingPlan sched = scheduler.schedule(*graph_);
    if (sched.valid) {
        graph_->apply_schedule(sched);
    }
}

TensorPtr GraphRuntime::run(const TensorPtr& hidden, const ForwardRequest& req,
                            const GraphKey& key, const ModelConfig& cfg, ModelWeights& weights,
                            KVCache& kv_cache, const std::vector<DeviceType>& layer_devices,
                            DeviceType default_device) {
    if (!builder_) return nullptr;

    auto t0 = std::chrono::steady_clock::now();

    const bool reuse = graph_ && key == cached_key_;
    if (reuse) {
        graph_->set_input(0, hidden);
    } else {
        build(hidden, cfg, weights, kv_cache, layer_devices, default_device, req.n_tokens);
        cached_key_ = key;
    }

    inject_runtime(req);

    auto result = graph_->execute();

    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG_INFO(std::string("Forward (graph, ") + (reuse ? "reused" : "built") +
             ") total: " + std::to_string((int)total_ms) + "ms (" + key.to_string() +
             ", start_pos=" + std::to_string(req.start_pos) +
             ", nodes=" + std::to_string(graph_->num_nodes()) + ")");

    return result;
}

}  // namespace forge
