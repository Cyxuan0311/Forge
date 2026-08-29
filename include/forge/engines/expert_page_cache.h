// ExpertPageCache: per-(layer, expert) residency + router statistics for MoE
// partial activation.
//
// Goal: run very-large MoE models by keeping only the routed experts resident
// (RAM/VRAM/disk) instead of every expert of every layer at once.
//
// P1 scope:
//   * Track where each expert currently lives (DeviceTarget) and when it was
//     last used (LRU eviction in later phases).
//   * Accumulate per-expert activation counts from sync_experts_resident() so
//     the CLI can print a router-distribution report (--expert-stats).
//   * Provide ensure_resident() that moves a per-expert weight view onto the
//     requested device. CPU<->CPU and the no-move cases are no-ops, so
//     correctness is unchanged vs. the monolithic 3D-tensor path.
//
// GPU per-expert paging (passing per-expert device pointers into the MoE cuda
// kernel instead of the whole 3D tensor) is Phase P2 -- it requires changing
// cuda_moe.cu to index an expert-pointer array. This cache already records the
// needed residency so P2 can consume it directly.
//
// Thread-safety: called from the generation thread only (single decoder), so no
// internal locking is needed.

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "forge/tensor.h"
#include "forge/types.h"

namespace forge {

struct ExpertPageInfo {
    DeviceTarget current = DeviceTarget::cpu();
    int64_t last_used_step = -1;
    int64_t activation_count = 0;
};

class ExpertPageCache {
public:
    // (re)size for a model: layers x experts-per-layer.
    void resize(int n_layers, const std::vector<int>& experts_per_layer);
    bool empty() const { return layers_ == 0; }

    // Stats for the CLI report.
    struct LayerStat {
        int n_expert;
        std::vector<int64_t> activation_counts;  // per expert
        int64_t total_activations = 0;
    };
    std::vector<LayerStat> stats() const;
    std::string format_report() const;

    // record_active mutates per-expert residency. Called from a const engine hook
    // via the mutable expert_page_cache_ member, so it is a non-const method.
    // expert_views holds the 3D expert tensors (gate/up/down) the caller can move;
    // P1 only records the target device / stats. Actual cross-device movement of a
    // per-expert slice is deferred to P2 (requires per-expert independent storage,
    // since to_device() on a slice of a monolithic 3D tensor cannot persist).
    void record_active(int layer, const std::vector<int>& active_experts, int64_t step,
                       bool paging_enabled, const std::vector<TensorPtr>& expert_views,
                       DeviceTarget target);

private:
    int layers_ = 0;
    std::vector<int> n_experts_;
    std::vector<std::vector<ExpertPageInfo>> info_;  // [layer][expert]
    mutable std::mutex mu_;
};

}  // namespace forge
