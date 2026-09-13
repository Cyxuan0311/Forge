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

#include "forge/gpu_memory_planner.h"
#include "forge/types.h"

#include <optional>
#include <string>

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
    MATMUL_FP8_E4M3_GEMV, // 6-C-4.2/4.3: W8A8 FP8 E4M3 GEMV (M==1 decode, hand-written, no CUTLASS)

    // MatMul, M > 1 (prefill / MMQ)
    MMQ,                     // tiled int8 dp4a kernel (legacy default path)
    MMQ_TENSORCORE,          // dequant B to FP16 once + cuBLAS FP16 GEMM (preferred; probed at launch)
    MATMUL_BATCH_GEMV,       // quantized dtype without an MMQ kernel: batched GEMV
    MATMUL_DEQUANT_CUBLAS,   // quantized dtype without MMQ/batched GEMV: FP32 dequant scratch + cuBLAS
    MATMUL_CUBLAS,           // FP32 weight, direct cuBLAS SGEMM
    MATMUL_FP8_E4M3,         // 6-C-4.1d: W8A8 FP8 E4M3 GEMM (CUTLASS, Ada)

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
    case KernelVariant::MATMUL_FP8_E4M3:
        return "MATMUL_FP8_E4M3";
    case KernelVariant::MATMUL_FP8_E4M3_GEMV:
        return "MATMUL_FP8_E4M3_GEMV";
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

// ===========================================================================
// Quantized-weight prefill matmul policy (VRAM-aware dispatch)
//
// Any quantized dtype that offers a dequant-FP16 + cuBLAS TensorCore prefill
// path can join this dispatch; today only Q3_K is registered, and adding
// Q4_K / Q5_K / ... is a single register_quant_tc_dtype() call, not a rewrite of
// the machinery. The rule chain, the optional-cache ladder and the memory
// budget are all dtype-agnostic -- they operate on QuantPolicyFacts, never on a
// concrete format.
//
// Two responsibilities stay separate:
//   * GpuMemoryPlanner (forge/gpu_memory_planner.h) owns how much VRAM the
//     process may claim and what is left for an OPTIONAL dequant cache.
//   * This layer owns which matmul PATH a dtype runs (TensorCore vs MMQ) and
//     applies the shared optional cache budget. It never manages VRAM itself.
//
// KV cache + the TensorCore scratch are MANDATORY; the per-weight dequant cache
// is OPTIONAL and reclaimable. The auto rule therefore walks a fixed cache
// ladder down to 0 before ever falling back to MMQ, so a performance cache can
// never starve the KV cache.
//
// The policy is resolved ONCE after model + KV init from a VRAM-fit estimate,
// not per GEMM. Env overrides (FORGE_MATMUL_PATH / FORGE_FORCE_TC /
// FORGE_FORCE_MMQ) win and pin a value.
//
// Extensibility: the decision is a *sequence of prioritized rules* (a chain of
// responsibility), not a monolithic if-else. Each rule is a pure function of the
// gathered facts and returns an optional decision; the first concrete answer wins.
// To add a new input (GPU capability tier, batch-size awareness, a CLI flag, a
// per-request override) write a quant_rule_*() function and register it in
// kQuantPolicyRules[] (see kernel_selector.cpp) at the right priority -- zero
// edits to the dispatch body. The same machinery backs every registered dtype.
// ===========================================================================

// The matmul path a quantized weight can take. Universal across dtypes; only
// the registration (which dtypes may use it) is per-format.
enum class QuantMatmulPolicy {
    Auto,        // resolve from VRAM fit at init (default)
    TensorCore,  // force the dequant-FP16 + cuBLAS path
    MMQ,         // force the dp4a/MMQ SIMT path
};

inline const char* quant_matmul_policy_name(QuantMatmulPolicy p) {
    switch (p) {
    case QuantMatmulPolicy::Auto: return "auto";
    case QuantMatmulPolicy::TensorCore: return "tc";
    case QuantMatmulPolicy::MMQ: return "mmq";
    }
    return "?";
}

// Facts gathered once at resolution time, shared by every quantized dtype.
// Keeping them in a struct (rather than re-reading env / probing VRAM inside
// each rule) decouples the *sources of information* from the *policy logic* and
// makes every rule unit-testable with synthetic facts (no GPU needed).
struct QuantPolicyFacts {
    DataType dtype = DataType::Q3_K;  // which dtype this resolution is for (logging only)
    bool cuda_available = false;
    // Memory budget: KV + workspace are mandatory, the dequant cache is optional
    // and reclaimable. See forge/gpu_memory_planner.h for field semantics.
    GpuMemoryBudget budget;
    size_t vram_total = 0;
    size_t dequant_total_fp16 = 0;      // sum of FP16 dequant over CUDA quantized weights
    size_t configured_cache_bytes = 0;  // CLI/env cap on the per-weight dequant cache
    // env snapshot (read once)
    std::string matmul_path;       // FORGE_MATMUL_PATH ("" if unset)
    bool force_tc = false;         // FORGE_FORCE_TC == "1"
    bool force_mmq = false;        // FORGE_FORCE_MMQ == "1"
};

// The outcome of memory planning: which path to run *and* the optional cache
// budget that goes with it. Kept as a value rather than a bare enum so the
// decision carries its own rationale -- this is the main debug surface for
// VRAM-related regressions (log it verbatim). The cache budget is shared across
// all registered dtypes (the dequant cache is keyed by weight pointer); the PATH
// is recorded per-dtype via set_quant_matmul_policy().
struct QuantExecutionPlan {
    QuantMatmulPolicy path = QuantMatmulPolicy::Auto;
    size_t dequant_cache_bytes = 0;        // optional cache budget for this plan
    size_t mandatory_workspace_bytes = 0;  // FP16 dequant scratch peak (not shrinkable)
    size_t available_bytes = 0;            // headroom the decision was made against
    bool cache_reduced = false;            // cache had to shrink below the configured cap
    const char* source = "unresolved";     // winning rule / rationale, for logging
};

// Decide the execution plan for a synthetic or probed budget. Pure: no GPU
// probing, no allocation, so it is fully unit-testable without a device.
QuantExecutionPlan plan_quant_execution(const QuantPolicyFacts& facts);
// Auto path: take the largest optional cache level that still fits, walking a
// fixed ladder down to 0 before ever giving up on TensorCore.
QuantExecutionPlan plan_quant_auto(const QuantPolicyFacts& facts);
// Pinned path (env override / explicit CLI): honours the requested path and the
// configured cache cap verbatim -- never silently degrades to the other path.
QuantExecutionPlan plan_quant_pinned(QuantMatmulPolicy policy, const QuantPolicyFacts& facts,
                                     const char* source);

// Apply a plan's cache budget: for TensorCore, tighten the dequant cache budget,
// releasing entries that no longer fit so the memory is actually available to
// the mandatory allocations. (The per-dtype PATH is recorded by the resolver,
// not here -- the cache budget is a single shared working set.)
void apply_quant_execution_plan(const QuantExecutionPlan& plan);

// The shared plan (the global dequant-cache working set) decided by the most
// recent resolve_quant_execution_plan() call.
QuantExecutionPlan quant_execution_plan();

// A rule: a pure function of QuantPolicyFacts returning an optional decision.
// Carries its own name for logging. Register in kQuantPolicyRules[] (see
// kernel_selector.cpp) to add a decision input without touching the resolver.
struct QuantPolicyRule {
    const char* name;
    std::optional<QuantMatmulPolicy> (*fn)(const QuantPolicyFacts&);
};

// Built-in rules (each a focused, independently testable unit).
std::optional<QuantMatmulPolicy> quant_rule_env_matmul_path(const QuantPolicyFacts&);
std::optional<QuantMatmulPolicy> quant_rule_env_force_tc(const QuantPolicyFacts&);
std::optional<QuantMatmulPolicy> quant_rule_env_force_mmq(const QuantPolicyFacts&);
std::optional<QuantMatmulPolicy> quant_rule_vram_aware(const QuantPolicyFacts&);

struct QuantPolicyDecision {
    QuantMatmulPolicy policy;
    const char* source;  // winning rule name, for logging
};

// Apply the rule chain; returns the first concrete decision, else fallback.
QuantPolicyDecision resolve_quant_policy(const QuantPolicyFacts& facts,
                                          QuantMatmulPolicy fallback = QuantMatmulPolicy::MMQ);

// ---- Per-dtype registry: which dtypes participate in TC dispatch ----
// Registering a dataType adds it to VRAM-aware TC/MMQ selection; it then reuses
// the shared rule chain with no dtype-specific code. Q3_K is registered by
// default (see kernel_selector.cpp). To support a new format (e.g. Q4_K), call
// this once at startup -- no other change is required.
void register_quant_tc_dtype(DataType dt, const char* tag = nullptr);
bool is_quant_tc_dtype(DataType dt);
const char* quant_tc_dtype_tag(DataType dt);

// ---- Public state surface (integration point) ----
// Pin / read the resolved policy for a dtype (Auto until resolved).
void set_quant_matmul_policy(DataType dt, QuantMatmulPolicy policy);
QuantMatmulPolicy quant_matmul_policy(DataType dt);

// Resolve the execution plan once, after the weights are loaded and before the
// KV cache is (re)sized. kv_cache_bytes_missing is the KV memory the process
// still has to claim: the planner reserves it ahead of any optional dequant
// cache, so a performance cache can never starve the KV. safety_margin_bytes
// absorbs activation/workspace growth between this probe and the first forward.
//
// Ordering matters: call this BEFORE the KV sizing decision and BEFORE warmup
// has a chance to fill the dequant cache, otherwise the cache is sized against
// a free-memory reading that already excludes the KV it is supposed to host.
// Updates the shared cache budget and records the resolved path for dt.
QuantExecutionPlan resolve_quant_execution_plan(DataType dt, size_t kv_cache_bytes_missing,
                                                size_t safety_margin_bytes);

// Legacy entry point: resolve with no outstanding KV request. Prefer the
// explicit resolve_quant_execution_plan() above.
void resolve_quant_policy_vram_aware(DataType dt, size_t safety_margin_bytes);

}  // namespace forge
