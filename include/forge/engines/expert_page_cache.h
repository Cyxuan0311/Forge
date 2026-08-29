// ExpertPageCache: per-(layer, expert) residency + router statistics for MoE
// partial activation.
//
// Goal: run very-large MoE models by keeping only the routed experts resident
// (RAM/VRAM) instead of every expert of every layer at once.
//
// P2 scope:
//   * Own INDEPENDENT per-expert weight copies so a single expert can be moved
//     between devices. This is the thing P1 could not do: to_device() on a slice
//     of a monolithic 3D tensor cannot persist, because the slice is a temporary
//     view that shares the 3D tensor's storage (moving it either does nothing or
//     corrupts the parent). Owning a real copy per expert is what makes paging
//     possible at all.
//   * ensure_resident() materialises an expert on first use and moves it onto the
//     requested device; LRU eviction keeps resident bytes under a budget.
//   * Track router activations plus paging hit/miss/eviction counters for the CLI
//     (--expert-stats) so residency behaviour is observable.
//
// Memory model:
//   An expert is "materialised" once its weights have been COPIED out of the
//   monolithic 3D tensor into an owned 2D tensor. Materialisation is LAZY: it
//   happens the first time the router selects that expert, so cold experts are
//   never copied and enabling paging costs nothing for experts never routed.
//   Before switching an expert to another device, evict_to_budget() demotes the
//   least-recently-used experts back to CPU.
//
//   NOTE: materialisation duplicates the bytes of every expert that gets routed
//   (the monolithic 3D tensor is still referenced by the non-paging paths). Peak
//   memory therefore only DROPS below the baseline once the 3D tensor itself is
//   released, which is deliberately left for a follow-up so the non-paging path
//   stays byte-identical. The win today is residency CONTROL (which experts are
//   on the compute device, and how much VRAM they occupy), not raw peak RSS.
//
// Thread-safety: mu_ guards all state so stats can be read while generating.

#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "forge/tensor.h"
#include "forge/types.h"

namespace forge {

// Expert weight slots. Separate-tensor architectures (PhiMoE) use Gate/Up/Down;
// combined architectures (Gemma4) use GateUp/Down.
enum class ExpertSlot : int {
    Gate = 0,
    Up = 1,
    Down = 2,
    GateUp = 3,
    NumSlots = 4,
};

struct ExpertPageInfo {
    DeviceTarget current = DeviceTarget::cpu();
    int64_t last_used_step = -1;
    int64_t activation_count = 0;
    // Independent per-expert weights, indexed by ExpertSlot. Empty until the
    // expert is materialised (lazy).
    std::array<TensorPtr, static_cast<size_t>(ExpertSlot::NumSlots)> w{};
    int64_t bytes = 0;
    bool materialized = false;
};

class ExpertPageCache {
public:
    // (re)size for a model: layers x experts-per-layer.
    void resize(int n_layers, const std::vector<int>& experts_per_layer);
    bool empty() const { return layers_ == 0; }

    // ---- Lazy materialisation ------------------------------------------------
    // Copy expert `e`'s weights out of `src3d` (the 3D expert tensors) into
    // independent owned tensors. `slots[i]` names what src3d[i] holds.
    // `expert_dim` is the axis that indexes experts (0 for PhiMoE, 2 for Gemma4).
    // No-op if already materialised.
    void materialize(int layer, int expert, const std::vector<TensorPtr>& src3d,
                     const std::vector<ExpertSlot>& slots, int expert_dim);

    bool is_materialized(int layer, int expert) const;

    // Resident per-expert tensor for `slot` (nullptr if not materialised).
    TensorPtr expert_weight(int layer, int expert, ExpertSlot slot) const;

    // ---- Residency -----------------------------------------------------------
    // Make `expert` resident on `target`: materialise on first use, then move if
    // the device differs. Returns true when a device move happened (a "miss").
    bool ensure_resident(int layer, int expert, DeviceTarget target, int64_t step);

    // P1 entry point kept for the stats-only path: record the routed experts and
    // (if paging is on) make each resident on `target`.
    void record_active(int layer, const std::vector<int>& active_experts, int64_t step,
                       bool paging_enabled, const std::vector<TensorPtr>& src3d,
                       const std::vector<ExpertSlot>& slots, int expert_dim,
                       DeviceTarget target);

    // ---- Budget / LRU --------------------------------------------------------
    void set_budget_bytes(int64_t b) { budget_bytes_ = b; }
    int64_t budget_bytes() const { return budget_bytes_; }
    int64_t resident_bytes() const;
    int resident_count() const;
    // Demote least-recently-used experts back to CPU until resident_bytes <=
    // budget. Experts already on CPU are skipped (nowhere cheaper to put them).
    // Returns the number of experts evicted.
    int evict_to_budget();

    // ---- Reporting -----------------------------------------------------------
    struct Stats {
        int n_layers = 0;
        std::vector<int> n_experts;                        // per layer
        std::vector<std::vector<int64_t>> activations;     // [layer][expert]
        int64_t total_activations = 0;
        int64_t hits = 0, misses = 0, evictions = 0;
        int64_t resident_bytes = 0;
        int resident_count = 0;
        int materialized_count = 0;
        int64_t budget_bytes = 0;
    };
    Stats stats() const;
    std::string format_report() const;

private:
    // Locked variants for callers that already hold mu_.
    void materialize_locked(int layer, int expert, const std::vector<TensorPtr>& src3d,
                            const std::vector<ExpertSlot>& slots, int expert_dim);
    bool ensure_resident_locked(int layer, int expert, DeviceTarget target, int64_t step);
    int64_t resident_bytes_locked() const;
    int evict_to_budget_locked();

    int layers_ = 0;
    std::vector<int> n_experts_;
    std::vector<std::vector<ExpertPageInfo>> info_;  // [layer][expert]
    // Source of truth for lazy materialisation, captured on the first
    // record_active() for a layer.
    std::vector<std::vector<TensorPtr>> src3d_;  // [layer] 3D expert tensors
    std::vector<ExpertSlot> slots_;              // what each src3d_[l][i] holds
    int expert_dim_ = 0;                         // axis indexing experts
    int64_t budget_bytes_ = 0;
    int64_t hits_ = 0, misses_ = 0, evictions_ = 0;
    mutable std::mutex mu_;
};

}  // namespace forge
