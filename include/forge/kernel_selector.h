// =========================================================================
// kernel_selector.h — Unified Kernel Selector (Phase 4)
//
// Centralizes dispatch decisions for the matmul family (GEMV / MMQ /
// TensorCore) and MoE expert GEMM, following the attention_selector pattern:
//   - Encode problem constraints into KernelProblem
//   - select_kernel() returns the optimal KernelVariant
//   - The caller dispatches to the corresponding launch function
//
// Phase 3's per-file if-else dispatch (matmul_cuda.cpp select_prefill_path,
// gemma4_moe.cpp dtype switch) is folded in here so the policy is one place,
// testable, and ready for Phase 4-B benchmark-driven dispatch (an autotuning
// layer sits behind select_kernel() without touching launcher code).
// =========================================================================

#pragma once

#include "forge/types.h"

namespace forge {

// ---- Selection domain ----
//
// The same KernelProblem/KernelVariant vocabulary is reused across operators.
// MatMul covers A(M,K) x B(K,N) in transB layout — both the M == 1 single-row
// GEMV (decode) and the M > 1 batched prefill / MMQ cases. MoE covers the
// grouped expert quantized GEMV inside the routed experts.
enum class KernelDomain {
    MatMul,
    MoE,
};

// ---- Problem descriptor ----
//
// Dimensions that matter for dispatch. sm_arch is the compute capability
// encoded as major*10+minor (e.g. 89 for sm_89); 0 means "detect at selection
// time". Phase 4-B will additionally use the full (domain, dtype, M, N, K,
// sm_arch) tuple as the autotuner cache key.
struct KernelProblem {
    KernelDomain domain = KernelDomain::MatMul;
    DataType dtype = DataType::FP32;
    int M = 1;  // rows / tokens (1 == decode GEMV)
    int N = 0;  // output columns
    int K = 0;  // reduction (inner) dimension
    int batch = 1;  // reserved (multi-batch / paged variants)
    int sm_arch = 0;  // compute capability major*10+minor; 0 = auto-detect
};

// ---- Variant enumeration ----
//
// The selected kernel family. Callers switch on this and launch the matching
// kernel. Variants are decision *preferences*: MMQ_TENSORCORE still needs a
// runtime probe at the call site (the FP16-dequant + cuBLAS launcher reports
// availability); when it fails, the caller falls through to the DP4A chain.
enum class KernelVariant {
    // MatMul, M == 1 (decode GEMV)
    GEMV_Q8_1,           // dedicated single-row quantized GEMV (Q8_1 + dp4a)
    GEMV_BATCH,          // quantized dtype without a single-row kernel: reuse the batched GEMV with M == 1
    GEMV_DEQUANT_TRANSB, // quantized dtype without any GEMV: dequant whole B to FP32 scratch, then GEMV
    GEMV_FP32_TRANSB,    // FP32 weight, custom FP32 GEMV (small N)
    GEMV_FP32_CUBLAS,    // FP32 weight, cuBLAS (large N, output_proj)

    // MatMul, M > 1 (prefill / MMQ)
    MMQ,                     // tiled int8 dp4a kernel (legacy default path)
    MMQ_TENSORCORE,          // dequant B to FP16 once + cuBLAS FP16 GEMM (preferred; probed at launch)
    MATMUL_BATCH_GEMV,       // quantized dtype without an MMQ kernel: batched GEMV
    MATMUL_DEQUANT_CUBLAS,   // quantized dtype without MMQ/batched GEMV: FP32 dequant scratch + cuBLAS
    MATMUL_CUBLAS,           // FP32 weight, direct cuBLAS SGEMM

    // MoE
    MOE_EXPERT_GEMV,  // grouped expert quantized GEMV (Q8_1 + dp4a)

    // Fallback — callers must not launch; report a capability failure.
    UNSUPPORTED,
};

// ---- Selector function ----
//
// Pure policy: reads KernelProblem plus the documented environment overrides
// (FORGE_MATMUL_PATH / FORGE_FORCE_TC / FORGE_FORCE_MMQ) and the dispatch
// table availability, returns the best KernelVariant. Behaviour matches the
// Phase 3 / 3.5 landing state exactly.
KernelVariant select_kernel(const KernelProblem& problem);

// ---- Human-readable names (for logging/debugging) ----

inline const char* kernel_domain_name(KernelDomain domain) {
    switch (domain) {
    case KernelDomain::MatMul:
        return "matmul";
    case KernelDomain::MoE:
        return "moe";
    }
    return "?";
}

inline const char* kernel_variant_name(KernelVariant variant) {
    switch (variant) {
    case KernelVariant::GEMV_Q8_1:
        return "GEMV_Q8_1";
    case KernelVariant::GEMV_BATCH:
        return "GEMV_BATCH";
    case KernelVariant::GEMV_DEQUANT_TRANSB:
        return "GEMV_DEQUANT_TRANSB";
    case KernelVariant::GEMV_FP32_TRANSB:
        return "GEMV_FP32_TRANSB";
    case KernelVariant::GEMV_FP32_CUBLAS:
        return "GEMV_FP32_CUBLAS";
    case KernelVariant::MMQ:
        return "MMQ";
    case KernelVariant::MMQ_TENSORCORE:
        return "MMQ_TENSORCORE";
    case KernelVariant::MATMUL_BATCH_GEMV:
        return "MATMUL_BATCH_GEMV";
    case KernelVariant::MATMUL_DEQUANT_CUBLAS:
        return "MATMUL_DEQUANT_CUBLAS";
    case KernelVariant::MATMUL_CUBLAS:
        return "MATMUL_CUBLAS";
    case KernelVariant::MOE_EXPERT_GEMV:
        return "MOE_EXPERT_GEMV";
    case KernelVariant::UNSUPPORTED:
        return "UNSUPPORTED";
    }
    return "?";
}

// ===========================================================================
// Phase 4-B — measured-dispatch cache
//
// select_kernel() is a pure policy function. The hot dispatch path goes through
// select_kernel_cached(), which adds a process-local decision cache keyed on
// the documented (arch, dtype, M, N, K) signature (extended with the selection
// domain and batch). A FORGE_AUTOTUNE=1 run measures the MMQ vs TensorCore
// candidates on real tensors at the call site (cache miss only) and records the
// winner via kernel_cache_record(), overriding the Phase 5-C per-dtype
// conservative static thresholds (see src/operators/cuda/kernel_selector.cpp
// kTcMinM) for that signature. Env-forced selections (FORGE_MATMUL_PATH /
// FORGE_FORCE_TC / FORGE_FORCE_MMQ, as used by the benchmark harness) always
// bypass the cache so regression comparisons stay authoritative.
// ===========================================================================

// Phase 4-B measured key. Deliberately the *minimal sufficient signature* for
// the current dispatch slice (quantized prefill, M > 1, MMQ-vs-TensorCore on a
// fixed sm_arch) — NOT a permanent complete model of a kernel's behaviour, and
// it never replaces the runtime launcher semantics (MMQ_TENSORCORE is still
// probed at launch before the dp4a chain). Future slices needing finer keys
// must extend the signature rather than overload it; known gaps:
//   - transB / layout (A and B memory order; today fixed transB)
//   - quant format + layout (Q4_0 / Q4_K / Q8_0 block sizes and layouts are
//     distinct — never flatten several dtypes into one "dtype" bucket)
//   - activation dtype / weight dtype / compute dtype (today FP32 activation
//     in, int8 / FP16 compute; FP16/BF16 activations need their own key)
//   - workspace / accumulator mode (atomic vs non-atomic, workspace policy)
//     once batched / paged variants land
// A measured entry is already GPU-specific through sm_arch.
struct KernelSignature {
    KernelDomain domain = KernelDomain::MatMul;
    DataType dtype = DataType::FP32;
    int M = 1;
    int N = 0;
    int K = 0;
    int batch = 1;
    int sm_arch = 0;  // actual capability; a measured entry is GPU-specific
};

inline bool operator==(const KernelSignature& l, const KernelSignature& r) {
    return l.domain == r.domain && l.dtype == r.dtype && l.M == r.M && l.N == r.N &&
           l.K == r.K && l.batch == r.batch && l.sm_arch == r.sm_arch;
}

// Strict weak ordering (std::map key).
inline bool operator<(const KernelSignature& l, const KernelSignature& r) {
    if (l.domain != r.domain)
        return static_cast<int>(l.domain) < static_cast<int>(r.domain);
    if (l.dtype != r.dtype)
        return static_cast<int>(l.dtype) < static_cast<int>(r.dtype);
    if (l.M != r.M)
        return l.M < r.M;
    if (l.N != r.N)
        return l.N < r.N;
    if (l.K != r.K)
        return l.K < r.K;
    if (l.batch != r.batch)
        return l.batch < r.batch;
    return l.sm_arch < r.sm_arch;
}

inline KernelSignature make_signature(const KernelProblem& p) {
    KernelSignature s;
    s.domain = p.domain;
    s.dtype = p.dtype;
    s.M = p.M;
    s.N = p.N;
    s.K = p.K;
    s.batch = p.batch;
    s.sm_arch = p.sm_arch;
    return s;
}

struct KernelCacheStats {
    long lookups = 0;   // cacheable prefill consults
    long hits = 0;      // consults served from the cache
    long misses = 0;    // cacheable consults not found in the cache
    long measured = 0;  // autotune measurements recorded via kernel_cache_record
    long tc_selected = 0;    // recorded winners that chose the TensorCore path
    long dp4a_selected = 0;  // recorded winners that chose the MMQ (dp4a) path
};

// True when this problem lives in the measured-dispatch slice (quantized
// prefill, sm80+, TC-eligible dtype) — the only place MMQ vs TensorCore
// competes and decisions are worth caching/measuring.
bool kernel_autotune_eligible(const KernelProblem& problem);

// Hot-path entry. Consult/update the process-local cache for eligible
// problems; all other problems (decode, MoE, FP32 prefill, non-eligible
// dtypes) and env-forced selections fall straight through to select_kernel().
// *cache_hit (optional) reports whether the returned variant came from the
// cache.
KernelVariant select_kernel_cached(const KernelProblem& problem, bool* cache_hit = nullptr);

// Record an externally measured winner (autotune pass) for this problem. It
// overrides the auto rule until kernel_cache_reset() or process exit.
void kernel_cache_record(const KernelProblem& problem, KernelVariant best);

KernelCacheStats kernel_cache_stats();
void kernel_cache_reset();

}  // namespace forge
