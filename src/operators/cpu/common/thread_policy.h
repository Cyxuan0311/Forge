#pragma once

// ---------------------------------------------------------------------------
// Thread-pool policy — the single place where per-architecture scheduling
// differences live.
//
// The pool *mechanism* (persistent workers, generation wakeup, ggml-style chunk
// stealing, calling-thread participation) is architecture-agnostic and shared by
// every backend; see thread_pool.h. What legitimately differs between CPUs is
// only a handful of knobs, collected here:
//
//   spin_limit     PAUSE iterations before a worker parks on the condition
//                  variable — short for many cheap threads, long for few fast
//                  ones
//   pin            hard CPU pinning (off by default: it hurts P/E hybrids and
//                  VMs, see the rationale in thread_pool.h)
//
// A new backend therefore does NOT need its own pool. It either inherits these
// defaults or adds its numbers to arch_policy_defaults() below. Every value can
// also be overridden per machine at runtime through the environment, which is
// the practical way to tune a heterogeneous fleet without recompiling.
// ---------------------------------------------------------------------------

#include <cstdlib>
#include <cstring>

namespace forge {
namespace cpu {

enum class ThreadPinMode {
    None,   // let the OS scheduler place threads (default)
    First,  // pin to the leading `nth` allowed CPUs
    Last,   // pin to the trailing `nth` allowed CPUs
};

struct ThreadPoolPolicy {
    int spin_limit = 4096;
    ThreadPinMode pin = ThreadPinMode::None;
};

// ---- per-architecture defaults ---------------------------------------------
// Every backend currently shares these values: none has been benchmarked into a
// measurably different optimum, and inventing numbers would be worse than
// measuring. When data exists, adjust the matching branch — or, if a backend
// grows enough knobs to warrant it, split this into arch/<arch>/thread_policy.h
// and include it here, mirroring how cpu/simd.h dispatches to arch kernels.
inline ThreadPoolPolicy arch_policy_defaults() {
    ThreadPoolPolicy p;
    p.spin_limit = 4096;
    p.pin = ThreadPinMode::None;
#if defined(FORGE_ARCH_X86)
    // AVX2 GEMV is bandwidth-bound with short rows, so fine-grained stealing
    // wins. Hybrid P/E parts keep pin == None so workers are never parked on
    // efficiency cores.
#elif defined(FORGE_ARCH_ARM64)
    // big.LITTLE has the same asymmetry as x86 hybrids: no hard pinning.
#elif defined(FORGE_ARCH_PPC64)
    // SMT-heavy parts expose many cheap hardware threads; a shorter spin would
    // be friendlier once measured.
#else
    // generic / scalar fallback
#endif
    return p;
}

// Resolve the effective policy: architecture defaults, then environment
// overrides (FORGE_POOL_SPIN / FORGE_PIN_THREADS).
inline ThreadPoolPolicy make_policy() {
    ThreadPoolPolicy p = arch_policy_defaults();

    if (const char* v = std::getenv("FORGE_POOL_SPIN")) {
        const int n = std::atoi(v);
        if (n >= 0) {
            p.spin_limit = n;
        }
    }
    if (const char* v = std::getenv("FORGE_PIN_THREADS")) {
        if (std::strcmp(v, "first") == 0) {
            p.pin = ThreadPinMode::First;
        } else if (std::strcmp(v, "last") == 0) {
            p.pin = ThreadPinMode::Last;
        }
    }
    return p;
}

// Process-wide policy, resolved once from architecture defaults + environment.
inline const ThreadPoolPolicy& thread_policy() {
    static const ThreadPoolPolicy p = make_policy();
    return p;
}

}  // namespace cpu
}  // namespace forge
