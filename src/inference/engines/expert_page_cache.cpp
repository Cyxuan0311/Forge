// See expert_page_cache.h for design notes.

#include "forge/engines/expert_page_cache.h"

#include <algorithm>
#include <sstream>

#include "forge/logger.h"

namespace forge {

void ExpertPageCache::resize(int n_layers, const std::vector<int>& experts_per_layer) {
    std::lock_guard<std::mutex> lk(mu_);
    layers_ = n_layers;
    n_experts_ = experts_per_layer;
    info_.assign(n_layers, {});
    for (int l = 0; l < n_layers; ++l) {
        info_[l].assign(std::max(0, experts_per_layer[l]), ExpertPageInfo{});
    }
}

void ExpertPageCache::record_active(int layer, const std::vector<int>& active_experts,
                                    int64_t step, bool paging_enabled,
                                    const std::vector<TensorPtr>& expert_views,
                                    DeviceTarget target) {
    if (layer < 0 || layer >= layers_) return;
    std::lock_guard<std::mutex> lk(mu_);
    auto& row = info_[layer];

    for (int e : active_experts) {
        if (e < 0) continue;
        // Grow the row lazily so a layer whose expert count wasn't known at
        // resize() time (e.g. cfg.n_expert missing in metadata) is still tracked.
        if (e >= static_cast<int>(row.size())) row.resize(e + 1);
        ExpertPageInfo& p = row[e];
        ++p.activation_count;

        if (!paging_enabled) continue;  // P1: stats only until GPU paging lands
        // P1: record the intended residency. Actual movement of a per-expert slice
        // is deferred to P2, because to_device() on a slice of a monolithic 3D
        // tensor cannot persist (the slice is a temporary that frees its buffer).
        // P2 will keep per-expert weights as independent Tensors and page those.
        (void)expert_views;
        if (p.current != target) {
            p.current = target;
        }
        p.last_used_step = step;
    }
}

std::vector<ExpertPageCache::LayerStat> ExpertPageCache::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<LayerStat> out(layers_);
    for (int l = 0; l < layers_; ++l) {
        LayerStat& s = out[l];
        s.n_expert = static_cast<int>(info_[l].size());
        s.activation_counts.resize(s.n_expert);
        for (int e = 0; e < s.n_expert; ++e) {
            s.activation_counts[e] = info_[l][e].activation_count;
            s.total_activations += info_[l][e].activation_count;
        }
    }
    return out;
}

std::string ExpertPageCache::format_report() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::ostringstream os;
    int moe_layers = 0;
    for (int l = 0; l < layers_; ++l) {
        const auto& row = info_[l];
        if (row.empty()) continue;
        ++moe_layers;
        int64_t total = 0;
        for (const auto& p : row) total += p.activation_count;
        if (total == 0) continue;  // skip layers never routed (e.g. prefill only)
        os << "[expert-stats] layer " << l << ": " << row.size()
           << " experts, " << total << " total activations\n";
        // Print per-expert counts sorted by frequency (top 12) for compactness.
        std::vector<std::pair<int, int64_t>> ranked;
        ranked.reserve(row.size());
        for (int e = 0; e < static_cast<int>(row.size()); ++e)
            ranked.emplace_back(e, row[e].activation_count);
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        const int show = std::min(12, static_cast<int>(ranked.size()));
        for (int i = 0; i < show; ++i) {
            const double pct = total > 0 ? 100.0 * ranked[i].second / total : 0.0;
            os << "    expert " << ranked[i].first << ": " << ranked[i].second << " ("
               << pct << "%)\n";
        }
    }
    if (moe_layers == 0) os << "[expert-stats] no MoE layers observed\n";
    return os.str();
}

}  // namespace forge
