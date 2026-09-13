// =========================================================================
// gpu_memory_planner.h — GPU memory budget for execution policy
//
// Forge keeps two concerns apart on purpose:
//   * THIS header: how much device memory the process may claim, and what is
//     left over for *optional* performance structures.
//   * kernel_selector: which matmul path to actually run.
//
// The budget is a plain snapshot taken once at initialization; nothing here
// allocates, frees or queries the driver. Callers fill it from a single
// probe (cudaMemGetInfo + the dequant-cache accounting) and hand it to the
// policy selector.
//
// Field semantics
// ---------------
//   free_bytes              RAW device free memory at probe time. It already
//                           excludes everything this process (and any other)
//                           currently holds, so memory that is already
//                           resident must NOT be subtracted a second time.
//   kv_cache_bytes          KV memory this process still has to claim. This
//                           is MANDATORY and outranks any optional cache.
//   safety_margin_bytes     headroom for activation/workspace allocations that
//                           the runtime will make once requests start.
//   mandatory_workspace_bytes
//                           workspace the TensorCore path cannot avoid (the
//                           largest single weight's FP16 dequant scratch). It
//                           cannot be shrunk away, so it is reserved up front.
//   reclaimable_bytes       optional cache that is currently resident but can
//                           be handed back on demand. Counting it as available
//                           is what makes the plan stable: the selector decides
//                           against the *reclaimable* total and the caller then
//                           actually trims the cache down to the chosen budget.
//
// The single derived quantity is available_for_optional(): everything that a
// performance-only structure may take without ever endangering KV/workspace.
// =========================================================================
#pragma once

#include <cstddef>

namespace forge {

struct GpuMemoryBudget {
    size_t free_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t safety_margin_bytes = 0;
    size_t mandatory_workspace_bytes = 0;
    size_t reclaimable_bytes = 0;

    // Memory that must remain untouched by optional structures.
    size_t reserved_bytes() const {
        return kv_cache_bytes + safety_margin_bytes + mandatory_workspace_bytes;
    }

    // Free memory including what we are willing to take back from the cache.
    size_t effective_free_bytes() const { return free_bytes + reclaimable_bytes; }

    // Headroom left for an optional cache (never underflows).
    size_t available_for_optional() const {
        const size_t eff = effective_free_bytes();
        const size_t reserved = reserved_bytes();
        return eff > reserved ? eff - reserved : 0;
    }
};

}  // namespace forge
