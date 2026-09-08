// 6-C-5.0: FP8 scale debug / trace infrastructure. See include/forge/fp8_scale_debug.h.

#include "forge/fp8_scale_debug.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "forge/quant_traits.h"  // dtype_name

namespace forge {
namespace {

std::atomic<unsigned long long> g_sidecar_count{0};
std::atomic<unsigned long long> g_registry_count{0};
std::atomic<int> g_trace_init{-1};
std::atomic<bool> g_trace_on{false};

bool trace_enabled_cached() {
    if (g_trace_init.load(std::memory_order_relaxed) < 0) {
        const char* env = std::getenv("FORGE_TRACE_FP8_SCALE");
        g_trace_on.store(env && env[0] == '1', std::memory_order_relaxed);
        g_trace_init.store(0, std::memory_order_relaxed);
    }
    return g_trace_on.load(std::memory_order_relaxed);
}

}  // namespace

bool fp8_scale_trace_enabled() { return trace_enabled_cached(); }

Fp8ScaleDebugInfo fp8_scale_debug_info(const TensorPtr& t, bool used_registry_fallback) {
    Fp8ScaleDebugInfo info;
    if (!t) return info;

    info.has_sidecar = t->has_fp8_scale();
    info.tensor_data = t->data();
    info.host_scale = t->has_fp8_scale() ? t->fp8_scale() : 0.f;
    info.used_registry_fallback = used_registry_fallback;
    // Do NOT call fp8_scale_device() here: that would allocate GPU memory as a side
    // effect of debugging. Report the mirror only if it already exists.
    info.device_scale_ptr = t->fp8_scale_device_ptr();
    info.has_device_scale = (info.device_scale_ptr != nullptr);

    if (used_registry_fallback)
        g_registry_count.fetch_add(1, std::memory_order_relaxed);
    else
        g_sidecar_count.fetch_add(1, std::memory_order_relaxed);
    return info;
}

void fp8_scale_trace(const char* where, const TensorPtr& t, bool used_registry_fallback) {
    if (!trace_enabled_cached()) return;
    const Fp8ScaleDebugInfo info = fp8_scale_debug_info(t, used_registry_fallback);
    const std::string dt = t ? dtype_name(t->dtype()) : std::string("null");
    std::printf(
        "[fp8-scale] where=%s tensor=0x%p dtype=%s scale=%.8g device_scale=0x%p "
        "source=%s registry_fallback=%s\n",
        where ? where : "?", info.tensor_data, dt.c_str(), info.host_scale, info.device_scale_ptr,
        used_registry_fallback ? "Registry" : "TensorSidecar",
        used_registry_fallback ? "YES" : "NO");
}

Fp8ScaleSourceCounts fp8_scale_source_counts() {
    Fp8ScaleSourceCounts c;
    c.sidecar = g_sidecar_count.load(std::memory_order_relaxed);
    c.registry = g_registry_count.load(std::memory_order_relaxed);
    return c;
}

void fp8_scale_reset_counts() {
    g_sidecar_count.store(0, std::memory_order_relaxed);
    g_registry_count.store(0, std::memory_order_relaxed);
}

}  // namespace forge
