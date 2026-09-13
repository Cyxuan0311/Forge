#include "forge/inference/graph/graph_runtime.h"

#include <chrono>

#include "forge/backend_scheduler.h"
#include "forge/logger.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

#ifdef USE_CUDA
GraphRuntime::~GraphRuntime() {
    // 6-A-6.4: capture stream dies with GraphRuntime, never between
    // capture and replay.
    if (capture_stream_) {
        cudaStreamDestroy(capture_stream_);
        capture_stream_ = nullptr;
    }
}

// Lazily create (once) the owned capture stream.
cudaStream_t GraphRuntime::capture_stream() {
    if (!capture_stream_) {
        cudaError_t err = cudaStreamCreate(&capture_stream_);
        if (err != cudaSuccess) {
            LOG_ERROR("GraphRuntime: cudaStreamCreate failed: " +
                      std::string(cudaGetErrorString(err)));
            capture_stream_ = nullptr;
        }
    }
    return capture_stream_;
}
#else
GraphRuntime::~GraphRuntime() = default;
#endif

void GraphRuntime::inject_runtime(ComputeGraph& graph, std::vector<StartPosSlot>& slots,
                                  const ForwardRequest& req) {
    runtime_state_->start_pos = req.start_pos;
    runtime_state_->seq_id = req.seq_id;
    for (const auto& slot : slots) {
        graph.patch_node_param_i64(slot.node_idx, slot.int32_offset, req.start_pos);
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

    // 输出段跟随最后一层设备: hidden 已驻留该设备, 落到别处会引入跨设备拷贝,
    // 且 CPU output_norm 会直接解引用 GPU 权重指针 (segmentation fault)。
    DeviceType out_dev = (!layer_devices.empty())
                             ? layer_devices.back().type
                             : default_device;
    int output_idx = builder_->build_output(*graph_, cur_idx, weights, cfg, out_dev);
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

    // decode (non-prefill, seq_len==1) 使用独立图槽, 与 prefill 图共存; 跨
    // reset_kv_cache 复用, 避免每轮首 token 重建 decode 图(1757ms 级别的一次性开销)。
    const bool is_decode = !key.is_prefill;
    std::unique_ptr<ComputeGraph>& active_graph = is_decode ? decode_graph_ : graph_;
    GraphKey& active_cached_key = is_decode ? decode_cached_key_ : cached_key_;
    std::vector<StartPosSlot>& active_slots = is_decode ? decode_start_pos_slots_ : start_pos_slots_;

    const bool reuse = active_graph && key == active_cached_key;
    if (reuse) {
        active_graph->set_input(0, hidden);
    } else {
        build(hidden, cfg, weights, kv_cache, layer_devices, default_device, req.n_tokens);
        // build() 写入 graph_ / start_pos_slots_, 搬运到当前槽位。
        active_graph = std::move(graph_);
        active_cached_key = key;
        active_slots = std::move(start_pos_slots_);
#ifdef USE_CUDA
        if (cuda_graph_runner_) cuda_graph_runner_->invalidate();
#endif
    }

    inject_runtime(*active_graph, active_slots, req);

    // Phase 10: Try CUDA Graph replay/capture first.
    // CUDA Graph is only used for decode (non-prefill) on compatible graphs.
    bool used_cuda_graph = false;

#ifdef USE_CUDA
    if (cuda_graph_runner_ && cuda_graph_runner_->enabled() && !key.is_prefill) {
        // 6-A-6.4: capture/replay on the owned capture stream; the execute_fn
        // forwards the same stream so every kernel + D2D copy is capturable.
        // Falls back to stream 0 if stream creation failed.
        cudaStream_t cs = capture_stream();
        if (cs) {
            used_cuda_graph = cuda_graph_runner_->try_replay_or_capture(
                *active_graph, [this, cs, g = active_graph.get()]() { g->execute(cs); }, cs);
        } else {
            used_cuda_graph = cuda_graph_runner_->try_replay_or_capture(
                *active_graph, [this, g = active_graph.get()]() { g->execute(); }, /*stream=*/0);
        }
    }
#endif

    TensorPtr result;
    if (!used_cuda_graph) {
        // Normal execution path (fallback or CUDA Graph disabled)
        result = active_graph->execute();
    } else {
        result = active_graph->last_output();
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
