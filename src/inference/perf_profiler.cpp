#include "forge/perf_profiler.h"

#include <cstdio>
#include <cstdlib>

namespace forge {

PerfProfiler& PerfProfiler::instance() {
    static PerfProfiler profiler;
    return profiler;
}

void dump_phase_profile(const char* title, double wall_ms, int n_tokens,
                        const std::vector<std::pair<std::string, PerfRecord>>& rows) {
    if (rows.empty()) {
        fprintf(stderr,
                "[%s-profile] no operator data. Build with -DFORGE_PROFILING=ON and "
                "run with --perf-detail (or FORGE_PERF_PREFILL/DECODE=1) to capture it.\n",
                title);
        return;
    }

    // Split layer-scoped records (forward/layer_*) from everything else so we can
    // print layers in index order and the rest sorted by cost.
    std::vector<std::pair<std::string, PerfRecord>> layers, others;
    for (auto& r : rows) {
        if (r.first.rfind("forward/layer_", 0) == 0)
            layers.push_back(r);
        else
            others.push_back(r);
    }
    std::sort(layers.begin(), layers.end(), [](const auto& a, const auto& b) {
        int la = 0, lb = 0;
        sscanf(a.first.c_str(), "forward/layer_%d", &la);
        sscanf(b.first.c_str(), "forward/layer_%d", &lb);
        return la < lb;
    });
    std::sort(others.begin(), others.end(),
              [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });

    double layer_total = 0.0;
    for (auto& l : layers)
        layer_total += l.second.total_ms;

    fprintf(stderr, "\n===== %s PROFILE =====\n", title);
    fprintf(stderr, "[%s-profile] wall = %.1f ms (%d tokens)\n", title, wall_ms, n_tokens);
    fprintf(stderr, "[%s-profile] layers total = %.1f ms across %zu layer(s)\n", title,
            layer_total, layers.size());
    fprintf(stderr, "[%s-profile] per-layer (ms):\n", title);
    for (auto& l : layers)
        fprintf(stderr, "  %-22s %9.2f ms\n", l.first.c_str(), l.second.total_ms);
    fprintf(stderr, "[%s-profile] top other operators (by total ms):\n", title);
    // Print everything, not just the head: the interesting scopes are often the
    // small ones nested inside a big parent, and truncating the list is what made
    // dspark's append_context_kv look like a single opaque 17 ms for several
    // iterations. A single noisy run (model load, autotune) can still dominate, so
    // an env caps it when that gets in the way.
    static const size_t kMaxOps = [] {
        const char* e = std::getenv("FORGE_PERF_MAX_OPS");
        return e && e[0] ? (size_t)std::atoi(e) : (size_t)0;  // 0 = no cap
    }();
    const size_t limit = kMaxOps > 0 ? kMaxOps : others.size();
    for (size_t i = 0; i < others.size() && i < limit; ++i) {
        const auto& [name, rec] = others[i];
        double avg = rec.count > 0 ? rec.total_ms / rec.count : 0.0;
        fprintf(stderr, "  %-38s n=%-6lld total=%9.2f ms avg=%8.3f ms\n", name.c_str(),
                (long long)rec.count, rec.total_ms, avg);
    }
    fprintf(stderr, "==========================================================\n\n");
    fflush(stderr);
}

}  // namespace forge
