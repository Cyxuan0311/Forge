// =========================================================================
// fp8_scale_debug.h — 6-C-5.0: FP8 scale debug / trace infrastructure
//
// Pure instrumentation: it does NOT change the runtime architecture. Its job is to
// make "where did this FP8 scale come from, and is it device-visible?" answerable
// without grepping pointer/registry/host-device paths by hand.
//
// The three failure modes it is built for:
//   1. scale does not take effect            -> has_sidecar / source counters
//   2. graph replay does not see a new scale -> device_scale_ptr stability
//   3. an FP8 chain silently round-trips     -> trace lines + source counters
//
// Enable with FORGE_TRACE_FP8_SCALE=1.
// =========================================================================

#pragma once

#include <cstddef>

#include "forge/tensor.h"

namespace forge {

// Provenance snapshot for ONE tensor's FP8 scale.
struct Fp8ScaleDebugInfo {
    bool has_sidecar = false;          // Tensor sidecar is set (6-C-5)
    bool has_device_scale = false;     // device mirror allocated (6-C-5.2)
    // 6-C-5.7: the legacy host registry is deleted, so this is always false for a
    // production consumer (kept only so the gate counter semantics stay visible).
    bool used_registry_fallback = false;
    const void* tensor_data = nullptr;     // identity of the tensor's data buffer
    const void* device_scale_ptr = nullptr;  // stable device address (graph-visible)
    float host_scale = 0.f;
};

// Snapshot scale provenance for a tensor.
Fp8ScaleDebugInfo fp8_scale_debug_info(const TensorPtr& t, bool used_registry_fallback);

// Emit one human-readable trace line when FORGE_TRACE_FP8_SCALE=1; no-op otherwise.
//   Tensor=0x... dtype=FP8_E4M3 scale=... device_scale=0x... source=... fallback=...
void fp8_scale_trace(const char* where, const TensorPtr& t, bool used_registry_fallback);

// How many scale lookups came from the sidecar vs the legacy registry. 6-C-5.7
// deleted the registry, so a production run MUST show  registry == 0 (the field is
// kept as a live canary -- it can never go positive again) and sidecar > 0.
struct Fp8ScaleSourceCounts {
    unsigned long long sidecar = 0;
    unsigned long long registry = 0;
};
Fp8ScaleSourceCounts fp8_scale_source_counts();
void fp8_scale_reset_counts();

bool fp8_scale_trace_enabled();

}  // namespace forge
