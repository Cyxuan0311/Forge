// See expert_page_cache.h for design notes.

#include "forge/engines/expert_page_cache.h"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "forge/logger.h"

namespace forge {
namespace {

// True when (shape, strides) describes a densely packed tensor. Only contiguous
// slices can be byte-copied out of the monolithic 3D expert tensor; a strided
// slice would need a gather copy, so materialisation refuses those instead of
// silently producing wrong data.
bool is_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
    if (shape.empty()) return true;
    if (strides.size() != shape.size()) return false;
    int64_t expected = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        if (strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

// The two non-expert dims of a 3D expert tensor, kept in their original order.
bool expert_2d_shape(const std::vector<int64_t>& s3, int expert_dim,
                     std::vector<int64_t>& out2d) {
    if (s3.size() != 3 || expert_dim < 0 || expert_dim > 2) return false;
    out2d.clear();
    for (int d = 0; d < 3; ++d) {
        if (d != expert_dim) out2d.push_back(s3[d]);
    }
    return out2d.size() == 2;
}

}  // namespace

void ExpertPageCache::resize(int n_layers, const std::vector<int>& experts_per_layer) {
    std::lock_guard<std::mutex> lk(mu_);
    layers_ = n_layers;
    n_experts_ = experts_per_layer;
    info_.assign(n_layers, {});
    src3d_.assign(n_layers, {});
    for (int l = 0; l < n_layers; ++l) {
        info_[l].assign(n_experts_[l], ExpertPageInfo{});
    }
}

void ExpertPageCache::materialize(int layer, int expert,
                                  const std::vector<TensorPtr>& src3d,
                                  const std::vector<ExpertSlot>& slots, int expert_dim) {
    std::lock_guard<std::mutex> lk(mu_);
    materialize_locked(layer, expert, src3d, slots, expert_dim);
}

void ExpertPageCache::materialize_locked(int layer, int expert,
                                         const std::vector<TensorPtr>& src3d,
                                         const std::vector<ExpertSlot>& slots,
                                         int expert_dim) {
    if (layer < 0 || layer >= layers_ || expert < 0) return;
    auto& row = info_[layer];
    if (expert >= static_cast<int>(row.size())) row.resize(expert + 1);
    ExpertPageInfo& p = row[expert];
    if (p.materialized) return;

    // Build into a local array so a failure leaves the expert untouched rather
    // than half-materialised.
    std::array<TensorPtr, static_cast<size_t>(ExpertSlot::NumSlots)> built{};
    int64_t total = 0;

    for (size_t i = 0; i < src3d.size() && i < slots.size(); ++i) {
        const TensorPtr& src = src3d[i];
        if (!src) continue;
        const auto& s3 = src->shape();
        if (s3.size() != 3) continue;
        if (expert >= s3[expert_dim]) continue;

        std::vector<int64_t> s2;
        if (!expert_2d_shape(s3, expert_dim, s2)) continue;

        Tensor view = src->slice(expert_dim, expert, expert + 1).view(s2);
        if (!is_contiguous(view.shape(), view.strides())) {
            LOG_WARN("expert_page_cache: expert slice is not contiguous "
                     "(layer=" + std::to_string(layer) +
                     " expert=" + std::to_string(expert) +
                     " dim=" + std::to_string(expert_dim) +
                     "); skipping materialisation for this expert");
            return;
        }
        auto owned = std::make_shared<Tensor>(view.dtype(), view.shape(), view.device());
        try {
            owned->copy_from(view);
        } catch (const std::exception& e) {
            LOG_WARN("expert_page_cache: failed to materialise expert (layer=" +
                     std::to_string(layer) + " expert=" + std::to_string(expert) +
                     "): " + e.what());
            return;
        }
        built[static_cast<size_t>(slots[i])] = owned;
        total += static_cast<int64_t>(owned->nbytes());
    }

    if (total == 0) return;
    p.w = built;
    p.bytes = total;
    p.materialized = true;
}

bool ExpertPageCache::is_materialized(int layer, int expert) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (layer < 0 || layer >= layers_) return false;
    const auto& row = info_[layer];
    if (expert < 0 || expert >= static_cast<int>(row.size())) return false;
    return row[expert].materialized;
}

TensorPtr ExpertPageCache::expert_weight(int layer, int expert, ExpertSlot slot) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (layer < 0 || layer >= layers_) return nullptr;
    const auto& row = info_[layer];
    if (expert < 0 || expert >= static_cast<int>(row.size())) return nullptr;
    return row[expert].w[static_cast<size_t>(slot)];
}

bool ExpertPageCache::ensure_resident(int layer, int expert, DeviceTarget target,
                                      int64_t step) {
    std::lock_guard<std::mutex> lk(mu_);
    return ensure_resident_locked(layer, expert, target, step);
}

bool ExpertPageCache::ensure_resident_locked(int layer, int expert, DeviceTarget target,
                                             int64_t step) {
    if (layer < 0 || layer >= layers_ || expert < 0) return false;
    auto& row = info_[layer];
    if (expert >= static_cast<int>(row.size())) row.resize(expert + 1);
    ExpertPageInfo& p = row[expert];

    // Lazy: copy the expert out of the monolithic tensor the first time the
    // router selects it. Cold experts are never copied.
    if (!p.materialized && layer < static_cast<int>(src3d_.size()) &&
        !src3d_[layer].empty()) {
        materialize_locked(layer, expert, src3d_[layer], slots_, expert_dim_);
    }
    p.last_used_step = step;
    if (!p.materialized) return false;

    if (p.current == target) {
        ++hits_;
        return false;
    }

    // Move every materialised weight of this expert onto the target device.
    // CPU->CPU (and already-resident) cases are cheap no-ops inside to_device().
    // NOTE: Tensor::to_device() takes a DeviceType, so the GPU index in
    // DeviceTarget is not honoured here; multi-GPU expert placement needs a
    // device-index-aware move (follow-up).
    for (auto& w : p.w) {
        if (w && w->device() != target.type) w->to_device(target.type);
    }
    p.current = target;
    ++misses_;
    return true;
}

void ExpertPageCache::record_active(int layer, const std::vector<int>& active_experts,
                                    int64_t step, bool paging_enabled,
                                    const std::vector<TensorPtr>& src3d,
                                    const std::vector<ExpertSlot>& slots, int expert_dim,
                                    DeviceTarget target) {
    if (layer < 0 || layer >= layers_) return;
    std::lock_guard<std::mutex> lk(mu_);

    // Remember the source tensors so ensure_resident() can materialise lazily
    // even when it is called without an explicit source.
    if (layer < static_cast<int>(src3d_.size())) {
        if (src3d_[layer].empty()) src3d_[layer] = src3d;
        if (slots_.empty()) slots_ = slots;
        expert_dim_ = expert_dim;
    }

    auto& row = info_[layer];
    for (int e : active_experts) {
        if (e < 0) continue;
        if (e >= static_cast<int>(row.size())) row.resize(e + 1);
        ExpertPageInfo& p = row[e];
        ++p.activation_count;
        if (!paging_enabled) continue;
        ensure_resident_locked(layer, e, target, step);
    }

    if (paging_enabled) evict_to_budget_locked();
}

int64_t ExpertPageCache::resident_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    return resident_bytes_locked();
}

int64_t ExpertPageCache::resident_bytes_locked() const {
    int64_t total = 0;
    for (int l = 0; l < layers_; ++l) {
        for (const auto& p : info_[l]) {
            if (p.materialized && p.current.is_cuda()) total += p.bytes;
        }
    }
    return total;
}

int ExpertPageCache::resident_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    int n = 0;
    for (int l = 0; l < layers_; ++l) {
        for (const auto& p : info_[l]) {
            if (p.materialized && p.current.is_cuda()) ++n;
        }
    }
    return n;
}

int ExpertPageCache::evict_to_budget() {
    std::lock_guard<std::mutex> lk(mu_);
    return evict_to_budget_locked();
}

int ExpertPageCache::evict_to_budget_locked() {
    if (budget_bytes_ <= 0) return 0;  // 0 = unbounded

    struct Cand {
        int layer, expert;
        int64_t step, bytes;
    };
    std::vector<Cand> cands;
    for (int l = 0; l < layers_; ++l) {
        for (int e = 0; e < static_cast<int>(info_[l].size()); ++e) {
            const auto& p = info_[l][e];
            if (p.materialized && p.current.is_cuda()) {
                cands.push_back({l, e, p.last_used_step, p.bytes});
            }
        }
    }
    // Least-recently-used first.
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.step < b.step; });

    int evicted = 0;
    int64_t cur = resident_bytes_locked();
    for (const auto& c : cands) {
        if (cur <= budget_bytes_) break;
        auto& p = info_[c.layer][c.expert];
        for (auto& w : p.w) {
            if (w && w->device() != DeviceType::CPU) w->to_device(DeviceType::CPU);
        }
        p.current = DeviceTarget::cpu();
        cur -= c.bytes;
        ++evicted;
        ++evictions_;
    }
    return evicted;
}

ExpertPageCache::Stats ExpertPageCache::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s;
    s.n_layers = layers_;
    s.hits = hits_;
    s.misses = misses_;
    s.evictions = evictions_;
    s.budget_bytes = budget_bytes_;

    s.n_experts.resize(layers_);
    s.activations.resize(layers_);
    for (int l = 0; l < layers_; ++l) {
        s.n_experts[l] = static_cast<int>(info_[l].size());
        s.activations[l].resize(info_[l].size(), 0);
        for (size_t e = 0; e < info_[l].size(); ++e) {
            const auto& p = info_[l][e];
            s.activations[l][e] = p.activation_count;
            s.total_activations += p.activation_count;
            if (p.materialized) ++s.materialized_count;
            if (p.materialized && p.current.is_cuda()) {
                s.resident_bytes += p.bytes;
                ++s.resident_count;
            }
        }
    }
    return s;
}

std::string ExpertPageCache::format_report() const {
    Stats st = stats();
    std::ostringstream os;

    if (st.n_layers == 0 || st.total_activations == 0) {
        return "  (no MoE layers observed)\n";
    }

    for (int l = 0; l < st.n_layers; ++l) {
        const int ne = st.n_experts[l];
        if (ne <= 0) continue;
        int64_t layer_total = 0;
        for (int e = 0; e < ne; ++e) layer_total += st.activations[l][e];
        if (layer_total == 0) continue;

        std::vector<int> order(ne);
        for (int e = 0; e < ne; ++e) order[e] = e;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return st.activations[l][a] > st.activations[l][b];
        });

        os << "layer " << l << ": " << ne << " experts, " << layer_total
           << " total activations\n";
        const int shown = std::min(ne, 10);
        for (int i = 0; i < shown; ++i) {
            int e = order[i];
            int64_t c = st.activations[l][e];
            if (c == 0) break;
            double pct = 100.0 * static_cast<double>(c) / static_cast<double>(layer_total);
            os << "    expert " << e << ": " << c << " (" << pct << "%)\n";
        }
    }

    // Residency summary (only meaningful once paging is enabled).
    const int64_t total = st.hits + st.misses;
    os << "paging: materialized=" << st.materialized_count
       << " resident=" << st.resident_count
       << " resident_bytes=" << st.resident_bytes
       << " budget_bytes=" << st.budget_bytes;
    if (total > 0) {
        double hit_rate = 100.0 * static_cast<double>(st.hits) / static_cast<double>(total);
        os << " hits=" << st.hits << " misses=" << st.misses
           << " hit_rate=" << hit_rate << "%";
    }
    if (st.evictions > 0) os << " evictions=" << st.evictions;
    os << "\n";

    return os.str();
}

}  // namespace forge
