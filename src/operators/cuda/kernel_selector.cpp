// =========================================================================
// kernel_selector.cpp — Unified Kernel Selector policy (Phase 4)
//
// Decision logic formerly scattered across matmul_cuda.cpp
// (select_prefill_path + table fallbacks) and gemma4_moe.cpp (dtype switch).
// Behaviour kept identical to the Phase 3 / 3.5 landing state, extended by
// Phase 5-A/5-B backend coverage and Phase 5-C per-dtype conservative
// thresholds:
//   * M == 1 decode: dedicated Q8_1 GEMV -> batched GEMV(M=1) -> whole-matrix
//     dequant + FP32 GEMV, and for FP32 weights cuBLAS when N > 4096.
//   * M >  1 prefill: quantized Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 on sm80+ prefer the
//     FP16-dequant + cuBLAS TensorCore path once M reaches the dtype's
//     conservative threshold (Phase 5-C, see kTcMinM below), or on an explicit
//     env override; otherwise MMQ -> batched GEMV -> dequant + cuBLAS. FP32
//     weights go straight to cuBLAS. Q6_K joined in Phase 5-A and Q5_K in
//     Phase 5-B (Q5_K needed its own FP16 whole-matrix dequant kernel; see
//     launch_cublas_gemm_quant_fp16). Q3_K is NOT TC-eligible by default: its
//     block/scale dequant is slower than dp4a MMQ on shipping GPUs (RTX 4050
//     measured ~2.6x slower prefill), so it stays on MMQ; FORGE_AUTOTUNE=1 can
//     still opt it into TC per-signature.
//   * MoE: only Q4_0/Q4_K/Q5_0/Q6_K/Q8_0 experts are launchable; anything else
//     is UNSUPPORTED and the caller skips silently (as before).
//
// Policy layers (lowest to highest priority):
//   1. per-dtype conservative static heuristic (Phase 5-C);
//   2. the 4-B autotuner (FORGE_AUTOTUNE=1) measures MMQ vs TensorCore on real
//      tensors and overrides per measured signature;
//   3. benchmark force envs (FORGE_MATMUL_PATH / FORGE_FORCE_TC /
//      FORGE_FORCE_MMQ) pin a path and bypass the cache entirely.
// =========================================================================

#include "forge/kernel_selector.h"

#ifdef USE_CUDA

#    include <cstdlib>
#    include <cstring>
#    include <map>
#    include <mutex>
#    include <unordered_map>

#    include <cuda_runtime.h>

#    include "forge/cuda_kernels.h"
#    include "forge/logger.h"
#    include "forge/quant_traits.h"
#    include "matmul_dispatch.h"

namespace forge {
namespace ops {
namespace internal {

cuda::GemvFn lookup_gemv(DataType dt) {
    int idx = static_cast<int>(dt);
    if (idx < 0 || idx >= static_cast<int>(sizeof(cuda::gemv_dispatch) / sizeof(cuda::gemv_dispatch[0])))
        return nullptr;
    return cuda::gemv_dispatch[idx];
}

cuda::GemvBatchFn lookup_gemv_batch(DataType dt) {
    int idx = static_cast<int>(dt);
    if (idx < 0 || idx >= static_cast<int>(sizeof(cuda::gemv_batch_dispatch) / sizeof(cuda::gemv_batch_dispatch[0])))
        return nullptr;
    return cuda::gemv_batch_dispatch[idx];
}

cuda::MmqFn lookup_mmq(DataType dt) {
    int idx = static_cast<int>(dt);
    if (idx < 0 || idx >= 20)
        return nullptr;
    return cuda::mmq_dispatch[idx];
}

DequantMatrixFn lookup_dequant_matrix(DataType dt) {
    switch (dt) {
    case DataType::Q4_0:
        return cuda::launch_dequant_q4_0_matrix;
    case DataType::Q4_1:
        return cuda::launch_dequant_q4_1_matrix;
    case DataType::Q4_K:
        return cuda::launch_dequant_q4_k_matrix;
    case DataType::Q5_0:
        return cuda::launch_dequant_q5_0_matrix;
    case DataType::Q5_1:
        return cuda::launch_dequant_q5_1_matrix;
    case DataType::Q2_K:
        return cuda::launch_dequant_q2_k_matrix;
    case DataType::Q3_K:
        return cuda::launch_dequant_q3_k_matrix;
    case DataType::Q5_K:
        return cuda::launch_dequant_q5_k_matrix;
    case DataType::Q6_K:
        return cuda::launch_dequant_q6_k_matrix;
    case DataType::Q8_0:
        return cuda::launch_dequant_q8_0_matrix;
    case DataType::IQ2_S:
        return cuda::launch_dequant_iq2_s_matrix;
    case DataType::IQ2_XS:
        return cuda::launch_dequant_iq2_xs_matrix;
    case DataType::IQ2_XXS:
        return cuda::launch_dequant_iq2_xxs_matrix;
    case DataType::IQ3_S:
        return cuda::launch_dequant_iq3_s_matrix;
    case DataType::IQ4_NL:
        return cuda::launch_dequant_iq4_nl_matrix;
    default:
        return nullptr;
    }
}

}  // namespace internal
}  // namespace ops

// Per-dtype resolved matmul policy (VRAM-aware dispatch). Auto until
// resolve_quant_execution_plan() runs for a dtype; only registered dtypes
// consult it (today: Q3_K). File-scope statics (internal linkage) shared by both
// the anonymous-namespace helpers (prefer_tensor_core) and the public API below.
static std::unordered_map<DataType, QuantMatmulPolicy> g_quant_policy;
// Registered dtypes that may take the dequant-FP16 + cuBLAS TensorCore path and
// therefore participate in VRAM-aware TC/MMQ selection. Q3_K is seeded by
// ensure_quant_tc_dtypes(); more formats are added via register_quant_tc_dtype()
// with no further change -- they reuse the shared rule chain.
static std::unordered_map<DataType, std::string> g_quant_tc_dtypes;
// The single shared execution plan (the global dequant-cache working set).
static QuantExecutionPlan g_quant_plan;

static void ensure_quant_tc_dtypes() {
    if (g_quant_tc_dtypes.empty()) {
        g_quant_tc_dtypes.emplace(DataType::Q3_K, "q3k");
        // To enable a new format (e.g. Q4_K) in VRAM-aware dispatch, add it here
        // or call register_quant_tc_dtype() at startup -- nothing else changes.
        // g_quant_tc_dtypes.emplace(DataType::Q4_K, "q4k");
    }
}

namespace {

using ops::internal::lookup_dequant_matrix;
using ops::internal::lookup_gemv;
using ops::internal::lookup_gemv_batch;
using ops::internal::lookup_mmq;

// dtypes with an FP16 whole-matrix dequant kernel
// (see launch_cublas_gemm_quant_fp16). Phase 3 whitelist (Q4_0/Q4_K/Q8_0), Q6_K
// added by Phase 5-A, Q5_K by Phase 5-B (its launch_dequant_q5_k_matrix_fp16 was
// the missing backend; everything else on this path predates it).
bool tensorcore_eligible(DataType dt) {
    switch (dt) {
    case DataType::Q4_0:
    case DataType::Q4_K:
    case DataType::Q5_K:
    case DataType::Q6_K:
    case DataType::Q8_0:
        return true;
    // Q3_K is intentionally NOT TC-eligible. Phase 6-MMQ briefly claimed Q3_K
    // wins on the dequant-FP16 + cuBLAS path at prefill M, but measured on
    // shipping GPUs (e.g. RTX 4050, Ada) the TC path is ~2.6x SLOWER than the
    // dp4a MMQ path for Q3_K prefill (and ~3.2x slower at decode, M==1) -- the
    // Q3_K block/scale dequant does not amortize well on tensor cores there.
    // Default to MMQ; FORGE_AUTOTUNE=1 still measures per-signature and can
    // opt Q3_K back into TC where it actually wins.
    default:
        return false;
    }
    // Q3_K was previously excluded because a M==1 (decode) measurement showed
    // the dequant-FP16 + cuBLAS path 3.2x slower than dp4a MMQ (the cuBLAS
    // GEMM has no tensor-core occupancy at M==1). But select_matmul only routes
    // M>1 (prefill / speculative) here, and at prefill M the FP16 GEMM on
    // tensor cores wins clearly — exactly the regime where Q4_K/Q8_0 use TC.
    // The dequant is paid once per weight and amortized over all M rows, so it
    // is negligible versus the GEMM. The launcher falls back to dp4a MMQ if the
    // TC path is unavailable, and FORGE_AUTOTUNE=1 measures MMQ vs TC per
    // signature and keeps the faster one, so there is no silent regression.
    // NOTE: on memory-constrained GPUs the per-weight FP16 dequant cache
    // (FORGE_DEQUANT_CACHE_MB, default 1024 MB) cannot hold the whole-model
    // working set, so the dequant re-runs per forward. Combined with large-M
    // activation memory pressure this can thrash (see dequant_cache_acquire
    // churn); keep an eye on prefill at high M.
    // cached. Re-enable Q3_K TC only behind a VRAM/cache-fit check plus a fast
    // dequant kernel.
}

int detect_sm_arch() {
    static int cached = -1;
    if (cached < 0) {
        int dev = 0;
        int major = 0;
        int minor = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) !=
                cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) !=
                cudaSuccess) {
            major = 0;
            minor = 0;
        }
        cached = major * 10 + minor;
    }
    return cached;
}

int effective_sm_arch(const KernelProblem& p) {
    return p.sm_arch > 0 ? p.sm_arch : detect_sm_arch();
}

// 6-C-4.1d: W8A8 FP8 GEMM (CUTLASS) requires Ada FP8 Tensor Cores (sm_89+).
static bool fp8_gemm_supported() {
    return detect_sm_arch() >= 89;
}

// Phase 5-C: per-dtype conservative static thresholds (current-profile
// heuristic, not a universal theory) for the K=4096 / N=128256 (llama-3) shape
// on RTX 4050 sm_89. Measured MMQ->TensorCore crossover bands (interleaved
// scans):
//   Q8_0 (real MMQ baseline, Phase 3.5) ~M≈48-64
//   Q5_K (Phase 5-B)                    ~M≈64-80
//   Q6_K (Phase 5-A)                    ~M≈80
//   Q4_K / Q4_0                         ~M≈96-110
// Each default threshold keeps a deliberate safety margin above its measured
// band and is NOT the crossover point itself: the rule kicks in only well
// inside TC territory, and the margin absorbs the 4-B-observed +20-40%
// run-to-run jitter at band edges. The ordering (Q8_0 < Q5_K ~ Q6_K < Q4_K ~
// Q4_0) is used only as a monotone hint for a future empirical cost model; it is
// NOT a bit-width law -- what decides is the full set of dtype features (block
// size, scale layout, dequant / MMQ tile structure, memory traffic, integer
// instruction count), not bits alone (Q5_K and Q6_K land within ~16 M of each
// other). Keep in sync with the bench-local tc_min_m() in
// forge-bench-prefill_gemm.cpp and with docs/optimization_plan_zh.md §5-C.
struct TcMinM {
    DataType dt;
    int min_m;
};
static constexpr TcMinM kTcMinM[] = {
    {DataType::Q8_0, 96},  // measured ~48-64,  margin -> 96
    {DataType::Q5_K, 96},  // measured ~64-80,  margin -> 96
    {DataType::Q6_K, 96},  // measured ~80,     margin -> 96
    {DataType::Q4_K, 128}, // measured ~96-110, margin -> 128
    {DataType::Q4_0, 128}, // measured ~96-110, margin -> 128
    {DataType::Q3_K, 256}, // Phase 6-MMQ: prefill-only TC (M==1 was 3.2x slower,
                           // so keep a wide margin above the other 4-BPW types).
};
static constexpr int kTcMinMDefault = 128;  // future eligible dtypes stay at the
                                            // most conservative level
int tc_min_m(DataType dt) {
    for (const TcMinM& e : kTcMinM) {
        if (e.dt == dt)
            return e.min_m;
    }
    return kTcMinMDefault;
}

// M == 1 single-row GEMV selection. Env overrides never apply to decode (the
// pre-Phase-4 M == 1 branch never consulted FORGE_MATMUL_PATH etc.).
KernelVariant select_gemv(const KernelProblem& p) {
    // 6-C-4.2/4.3: FP8 E4M3 decode (M == 1) -> hand-written GEMV. M > 1 FP8 stays
    // on the CUTLASS GEMM path (select_matmul never sends M == 1 here), so FP8
    // never enters CUTLASS GEMM at M == 1.
    if (p.dtype == DataType::FP8_E4M3) {
        return fp8_gemm_supported() ? KernelVariant::MATMUL_FP8_E4M3_GEMV
                                    : KernelVariant::UNSUPPORTED;
    }
    if (lookup_gemv(p.dtype))
        return KernelVariant::GEMV_Q8_1;
    if (is_quantized_type(p.dtype)) {
        if (lookup_gemv_batch(p.dtype))
            return KernelVariant::GEMV_BATCH;
        if (lookup_dequant_matrix(p.dtype))
            return KernelVariant::GEMV_DEQUANT_TRANSB;
        return KernelVariant::UNSUPPORTED;
    }
    // FP32 (and other non-quantized) weights: cuBLAS for large N (output_proj),
    // custom GEMV otherwise.
    if (p.N > 4096)
        return KernelVariant::GEMV_FP32_CUBLAS;
    return KernelVariant::GEMV_FP32_TRANSB;
}

// M > 1 quantized prefill selection. Checks (in order):
//   FORGE_MATMUL_PATH = "mmq"|"dp4a"  -> force the SIMT path
//   FORGE_MATMUL_PATH = "tc"          -> force the TensorCore path
//   FORGE_FORCE_TC    = "1"           -> same as "tc" (benchmarking alias)
//   FORGE_FORCE_MMQ   = "1"           -> same as "mmq"/"dp4a" (regression alias)
// Anything else falls through to the automatic rule below.
//
// auto (Phase 5-C landing state): an eligible dtype goes TensorCore once M
// reaches *its* conservative threshold (tc_min_m: Q8_0/Q5_K/Q6_K at 96,
// Q4_K/Q4_0 at 128); everything else DP4A/MMQ. Coverage joined across
// phases: Q8_0 after Phase 3.5 added its true M>1 MMQ instance
// (mmq_dispatch[Q8_0]); Q6_K after Phase 5-A (dequant path already existed);
// Q5_K after Phase 5-B added its missing FP16 whole-matrix dequant. Q3_K is
// intentionally NOT TC-eligible (its dequant is slower than dp4a MMQ on
// shipping GPUs at both prefill and decode M); FORGE_AUTOTUNE can override.
// Per-signature override inside / around each threshold is owned by the 4-B
// autotuner once opted in (FORGE_AUTOTUNE=1); benchmark force envs stay the
// highest priority.
bool prefer_tensor_core(int M, DataType dt) {
#if !FORGE_USE_CUBLAS
    (void)M;
    (void)dt;
    return false;
#else
    if (!tensorcore_eligible(dt))
        return false;
    const char* env = std::getenv("FORGE_MATMUL_PATH");
    if (env) {
        if (std::strcmp(env, "mmq") == 0 || std::strcmp(env, "dp4a") == 0)
            return false;
        if (std::strcmp(env, "tc") == 0)
            return true;
        // anything else: fall through to the aliases, then auto
    }
    const char* ftc = std::getenv("FORGE_FORCE_TC");
    if (ftc && ftc[0] == '1')
        return true;
    const char* fmq = std::getenv("FORGE_FORCE_MMQ");
    if (fmq && fmq[0] == '1')
        return false;
    // VRAM-aware resolved policy: a registered dtype may have been pinned to MMQ
    // (VRAM too tight for TensorCore), in which case we short-circuit so we don't
    // probe a path that would thrash under memory pressure. Env overrides above
    // already returned. Extending to more dtypes is a register_quant_tc_dtype()
    // call, not a rewrite of this function.
    if (is_quant_tc_dtype(dt) && quant_matmul_policy(dt) == QuantMatmulPolicy::MMQ)
        return false;
    // Conservative per-dtype thresholds (see kTcMinM); the eligibility gate
    // above already excluded the rest.
    return M >= tc_min_m(dt);
#endif
}

KernelVariant select_matmul(const KernelProblem& p) {
    if (p.M <= 1)
        return select_gemv(p);
    // 6-C-4.1d: W8A8 FP8 E4M3 GEMM (CUTLASS, Ada). M == 1 GEMV is 6-C-4.2.
    if (p.dtype == DataType::FP8_E4M3) {
        if (fp8_gemm_supported())
            return KernelVariant::MATMUL_FP8_E4M3;
        return KernelVariant::UNSUPPORTED;
    }
    // M > 1 (prefill / MMQ).
    if (!is_quantized_type(p.dtype))
        return KernelVariant::MATMUL_CUBLAS;
    // Quantized B. Two compute paths (Phase 3):
    //   * DP4A        — INT8 SIMT dot products (MMQ tiled kernel, or batched
    //                    GEMV for dtypes without an MMQ kernel).
    //   * TensorCore  — dequantize the whole B to FP16 once, then a cuBLAS FP16
    //                    GEMM with FP32 accumulate on sm80+ Tensor Cores.
    const bool sm80_plus = effective_sm_arch(p) >= 80;
    if (sm80_plus && prefer_tensor_core(p.M, p.dtype))
        return KernelVariant::MMQ_TENSORCORE;
    if (lookup_mmq(p.dtype))
        return KernelVariant::MMQ;
    if (lookup_gemv_batch(p.dtype))
        return KernelVariant::MATMUL_BATCH_GEMV;
    if (lookup_dequant_matrix(p.dtype))
        return KernelVariant::MATMUL_DEQUANT_CUBLAS;
    return KernelVariant::UNSUPPORTED;
}

KernelVariant select_moe(const KernelProblem& p) {
    switch (p.dtype) {
    case DataType::Q4_0:
    case DataType::Q4_K:
    case DataType::Q5_0:
    case DataType::Q6_K:
    case DataType::Q8_0:
        return KernelVariant::MOE_EXPERT_GEMV;
    default:
        // No launchable expert kernel for this dtype: caller skips silently,
        // matching the pre-Phase-4 behaviour.
        return KernelVariant::UNSUPPORTED;
    }
}

// ---- Phase 4-B: measured-dispatch cache ------------------------------------
//
// The MMQ-vs-TensorCore competition exists only for quantized prefill
// (M > 1) on sm80+ with a TC-eligible dtype. That slice is the cacheable /
// measurable one; everything else has a single deterministic winner and is not
// cached (avoids decode/MoE churn for no benefit).
bool cacheable_prefill(const KernelProblem& p) {
    if (p.domain != KernelDomain::MatMul || p.M <= 1)
        return false;
    if (!is_quantized_type(p.dtype) || !tensorcore_eligible(p.dtype))
        return false;
    return effective_sm_arch(p) >= 80;
}

// True when a benchmark-style env override pins the prefill decision.
// Mirrors the force branches of prefer_tensor_core(): FORGE_MATMUL_PATH =
// "mmq"/"dp4a"/"tc" pins a path, and FORGE_FORCE_TC/MMQ == 1 alias them. Any
// other FORGE_MATMUL_PATH value (e.g. the benchmark's "auto") falls through to
// the auto rule and must NOT count as forced.
bool env_prefill_forced() {
    const char* env = std::getenv("FORGE_MATMUL_PATH");
    if (env && (std::strcmp(env, "mmq") == 0 || std::strcmp(env, "dp4a") == 0 ||
                std::strcmp(env, "tc") == 0))
        return true;
    const char* ftc = std::getenv("FORGE_FORCE_TC");
    if (ftc && ftc[0] == '1')
        return true;
    const char* fmq = std::getenv("FORGE_FORCE_MMQ");
    return fmq && fmq[0] == '1';
}

struct CacheState {
    std::mutex m;
    std::map<KernelSignature, KernelVariant> entries;
    KernelCacheStats stats;
};

CacheState& cache_state() {
    static CacheState state;
    return state;
}

}  // namespace

// ===========================================================================
// Phase 2: Q3_K prefill matmul policy (VRAM-aware dispatch)
//
// Decision = a prioritized *chain of rules* (chain of responsibility), not a
// monolithic if-else. Each rule is a pure function of Q3KPolicyFacts returning an
// optional decision; the first concrete answer wins. Adding a new input (GPU
// capability tier, batch-size awareness, a CLI flag, a per-request override)
// means writing a q3k_rule_*() function and registering it in kQ3KPolicyRules[]
// at the right priority -- zero edits to the dispatch body. Facts are gathered
// once (gather_q3k_policy_facts) and handed to the pure rules, so the policy
// logic is unit-testable with synthetic facts (no GPU needed).
// ===========================================================================

namespace {

// Optional dequant-cache ladder (MiB). The auto planner walks it top-down and
// takes the largest level that still fits after the mandatory reservations.
// 0 means "no cache at all", which the TensorCore path still supports (it then
// falls back to the grow-only scratch), so 0 is a valid plan rather than a
// failure: trading the performance cache away is always preferable to trading
// the TensorCore path away.
// NOTE: values are MiB and are converted to bytes at the use site (see the
// ladder loop in plan_quant_auto) -- do NOT pass them to size_t-byte APIs raw.
constexpr size_t kQuantCacheStepsMib[] = {1024, 768, 512, 384, 256, 128, 64, 0};
constexpr size_t kMiB = 1024u * 1024u;

// Gather every input the planner may need exactly once. Decoupled from the
// planning logic so plan_quant_*() never reads env / probes VRAM itself (keeps
// them pure and unit-testable against synthetic budgets).
QuantPolicyFacts gather_quant_policy_facts(DataType dt, size_t kv_cache_bytes_missing,
                                           size_t safety_margin) {
    QuantPolicyFacts f;
    f.dtype = dt;
    const char* e = std::getenv("FORGE_MATMUL_PATH");
    f.matmul_path = e ? e : "";
    const char* ftc = std::getenv("FORGE_FORCE_TC");
    f.force_tc = ftc && ftc[0] == '1';
    const char* fmq = std::getenv("FORGE_FORCE_MMQ");
    f.force_mmq = fmq && fmq[0] == '1';
    f.dequant_total_fp16 = cuda::dequant_total_fp16_bytes();
    f.configured_cache_bytes = cuda::dequant_cache_budget_bytes();
    f.budget.mandatory_workspace_bytes = cuda::dequant_scratch_peak_bytes();
    f.budget.kv_cache_bytes = kv_cache_bytes_missing;
    f.budget.safety_margin_bytes = safety_margin;
    // Already-resident cache entries are optional, so they count as reclaimable
    // rather than spent: the decision is made against that total and
    // apply_quant_execution_plan() then actually hands the memory back.
    f.budget.reclaimable_bytes = cuda::dequant_cache_current_bytes();
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        f.cuda_available = true;
        f.budget.free_bytes = free_mem;
        f.vram_total = total_mem;
    }
    return f;
}

// One-line rationale, emitted verbatim so VRAM regressions are diagnosable from
// a plain server log.
std::string quant_plan_summary(const QuantPolicyFacts& f, const QuantExecutionPlan& p) {
    const size_t mib = 1024 * 1024;
    const char* tag = quant_tc_dtype_tag(f.dtype);
    return std::string(tag) + " path=" + quant_matmul_policy_name(p.path) + " source=" + p.source +
           " cache=" + std::to_string(p.dequant_cache_bytes / mib) + "MB" +
           (p.cache_reduced ? "(reduced)" : "") +
           " scratch=" + std::to_string(p.mandatory_workspace_bytes / mib) + "MB" +
           " headroom=" + std::to_string(p.available_bytes / mib) + "MB" +
           " vram_free=" + std::to_string(f.budget.free_bytes / mib) + "MB" +
           " reclaimable=" + std::to_string(f.budget.reclaimable_bytes / mib) + "MB" +
           " kv_pending=" + std::to_string(f.budget.kv_cache_bytes / mib) + "MB" +
           " safety=" + std::to_string(f.budget.safety_margin_bytes / mib) + "MB";
}

}  // namespace

std::optional<QuantMatmulPolicy> quant_rule_env_matmul_path(const QuantPolicyFacts& f) {
    const std::string& e = f.matmul_path;
    if (e == "tc")
        return QuantMatmulPolicy::TensorCore;
    if (e == "mmq" || e == "dp4a")
        return QuantMatmulPolicy::MMQ;
    return std::nullopt;  // "auto" or unset: let lower-priority rules decide
}

std::optional<QuantMatmulPolicy> quant_rule_env_force_tc(const QuantPolicyFacts& f) {
    if (f.force_tc)
        return QuantMatmulPolicy::TensorCore;
    return std::nullopt;
}

std::optional<QuantMatmulPolicy> quant_rule_env_force_mmq(const QuantPolicyFacts& f) {
    if (f.force_mmq)
        return QuantMatmulPolicy::MMQ;
    return std::nullopt;
}

// VRAM-fit rule, kept for the rule-chain API: TensorCore when some (possibly
// reduced) optional cache level still fits, MMQ otherwise. Prefer
// plan_quant_execution(), which also reports the cache budget it settled on.
std::optional<QuantMatmulPolicy> quant_rule_vram_aware(const QuantPolicyFacts& f) {
    return plan_quant_auto(f).path;
}

QuantExecutionPlan plan_quant_pinned(QuantMatmulPolicy policy, const QuantPolicyFacts& f,
                                     const char* source) {
    QuantExecutionPlan p;
    p.path = policy;
    p.mandatory_workspace_bytes = f.budget.mandatory_workspace_bytes;
    p.available_bytes = f.budget.available_for_optional();
    p.source = source;
    if (policy == QuantMatmulPolicy::TensorCore) {
        // Honour the requested cap verbatim. An explicit request must never be
        // second-guessed -- silently shrinking it would make benchmarks measure
        // something other than what was asked for.
        p.dequant_cache_bytes =
            f.configured_cache_bytes < f.dequant_total_fp16 ? f.configured_cache_bytes
                                                            : f.dequant_total_fp16;
    }
    return p;
}

QuantExecutionPlan plan_quant_auto(const QuantPolicyFacts& f) {
    QuantExecutionPlan p;
    p.mandatory_workspace_bytes = f.budget.mandatory_workspace_bytes;
    p.available_bytes = f.budget.available_for_optional();
    p.source = "vram:auto";

    if (!f.cuda_available) {
        p.path = QuantMatmulPolicy::MMQ;  // no TC path without CUDA
        p.source = "vram:no_cuda";
        return p;
    }

    // A dtype that cannot use the TensorCore path must not reserve the optional
    // dequant cache (it would sit unused and starve the KV cache). Go straight
    // to MMQ without touching the budget.
    if (!tensorcore_eligible(f.dtype)) {
        p.path = QuantMatmulPolicy::MMQ;
        p.source = "vram:not_tc_eligible";
        return p;
    }

    const size_t configured = f.configured_cache_bytes;
    auto take = [&](size_t level_bytes) {
        // The cache can never usefully exceed the whole FP16 model.
        const size_t cap = level_bytes < f.dequant_total_fp16 ? level_bytes : f.dequant_total_fp16;
        if (cap > p.available_bytes)
            return false;
        p.path = QuantMatmulPolicy::TensorCore;
        p.dequant_cache_bytes = cap;
        p.cache_reduced = cap < configured;
        return true;
    };

    // 1) The configured cap first; cache_reduced here means the model itself is
    //    smaller than the cap, which is not a memory-pressure degradation.
    if (take(configured)) {
        p.source = p.cache_reduced ? "vram:tc_capped_by_model" : "vram:tc";
        return p;
    }
    // 2) Walk the ladder down. Giving up optional cache is always preferable to
    //    giving up TensorCore, so try every level -- including 0 -- first.
    //    Ladder values are MiB; convert to bytes before comparing against the
    //    byte-sized budget (the previous raw-byte use capped the cache at ~1 KiB,
    //    defeating the persistent dequant cache and thrashing prefill).
    for (size_t step : kQuantCacheStepsMib) {
        const size_t level = step * kMiB;
        if (level >= configured)
            continue;  // already tried, or above the user's cap
        if (take(level)) {
            p.source = "vram:tc_reduced_cache";
            return p;
        }
    }
    // 3) Even a 0-byte cache does not fit: MMQ is the only safe option left.
    p.path = QuantMatmulPolicy::MMQ;
    p.dequant_cache_bytes = 0;
    p.cache_reduced = true;
    p.source = "vram:mmq_no_room";
    return p;
}

QuantExecutionPlan plan_quant_execution(const QuantPolicyFacts& f) {
    if (auto d = quant_rule_env_matmul_path(f))
        return plan_quant_pinned(*d, f, "env:FORGE_MATMUL_PATH");
    if (auto d = quant_rule_env_force_tc(f))
        return plan_quant_pinned(*d, f, "env:FORGE_FORCE_TC");
    if (auto d = quant_rule_env_force_mmq(f))
        return plan_quant_pinned(*d, f, "env:FORGE_FORCE_MMQ");
    return plan_quant_auto(f);
}

// Ordered rule chain: env overrides outrank the VRAM auto rule. First concrete
// answer wins. Register new rules here -- no dispatch-body edits needed.
static const QuantPolicyRule kQuantPolicyRules[] = {
    {"env:FORGE_MATMUL_PATH", quant_rule_env_matmul_path},
    {"env:FORGE_FORCE_TC", quant_rule_env_force_tc},
    {"env:FORGE_FORCE_MMQ", quant_rule_env_force_mmq},
    {"vram_aware", quant_rule_vram_aware},
};

QuantPolicyDecision resolve_quant_policy(const QuantPolicyFacts& facts, QuantMatmulPolicy fallback) {
    for (const auto& r : kQuantPolicyRules) {
        if (auto d = r.fn(facts))
            return {*d, r.name};
    }
    return {fallback, "fallback"};
}

// ---- Per-dtype registry ----
void register_quant_tc_dtype(DataType dt, const char* tag) {
    g_quant_tc_dtypes[dt] = tag ? tag : "";
}

bool is_quant_tc_dtype(DataType dt) {
    ensure_quant_tc_dtypes();
    return g_quant_tc_dtypes.find(dt) != g_quant_tc_dtypes.end();
}

const char* quant_tc_dtype_tag(DataType dt) {
    ensure_quant_tc_dtypes();
    auto it = g_quant_tc_dtypes.find(dt);
    if (it != g_quant_tc_dtypes.end() && !it->second.empty())
        return it->second.c_str();
    return dtype_name(dt).c_str();  // fallback: the enum name (e.g. "Q3_K")
}

// ---- State surface ----
void set_quant_matmul_policy(DataType dt, QuantMatmulPolicy policy) {
    g_quant_policy[dt] = policy;
}

QuantMatmulPolicy quant_matmul_policy(DataType dt) {
    auto it = g_quant_policy.find(dt);
    if (it != g_quant_policy.end())
        return it->second;
    return QuantMatmulPolicy::Auto;  // not yet resolved
}

// Apply only the shared cache budget. The per-dtype PATH is recorded by the
// resolver (below), not here -- the dequant cache is a single working set keyed
// by weight pointer, so its budget is format-agnostic.
void apply_quant_execution_plan(const QuantExecutionPlan& plan) {
    if (plan.path != QuantMatmulPolicy::TensorCore)
        return;  // MMQ: leave the global cap alone, other dtypes still cache via TC
    if (plan.dequant_cache_bytes >= cuda::dequant_cache_budget_bytes())
        return;  // nothing to tighten
    cuda::set_dequant_cache_budget_mb(static_cast<long>(plan.dequant_cache_bytes / (1024 * 1024)));
    // The budget is only a ceiling; the buffers are already resident. Actually
    // release them, otherwise the memory the planner just promised to the KV
    // cache stays locked in the optional cache.
    cuda::dequant_cache_trim_to(plan.dequant_cache_bytes);
}

QuantExecutionPlan resolve_quant_execution_plan(DataType dt, size_t kv_cache_bytes_missing,
                                                size_t safety_margin_bytes) {
    ensure_quant_tc_dtypes();
    QuantPolicyFacts facts = gather_quant_policy_facts(dt, kv_cache_bytes_missing, safety_margin_bytes);
    QuantExecutionPlan plan = plan_quant_execution(facts);
    g_quant_plan = plan;
    set_quant_matmul_policy(dt, plan.path);  // record the per-dtype resolved path
    apply_quant_execution_plan(plan);        // tighten the shared cache budget
    // Degradations (shrunk cache, MMQ fallback) are operationally relevant and
    // must be visible at the default log level; the happy path stays at INFO.
    const std::string msg = "execution plan: " + quant_plan_summary(facts, plan);
    if (plan.cache_reduced || plan.path != QuantMatmulPolicy::TensorCore)
        LOG_WARN(msg);
    else
        LOG_INFO(msg);
    return plan;
}

void resolve_quant_policy_vram_aware(DataType dt, size_t safety_margin_bytes) {
    (void)resolve_quant_execution_plan(dt, 0, safety_margin_bytes);
}

QuantExecutionPlan quant_execution_plan() { return g_quant_plan; }

bool kernel_autotune_eligible(const KernelProblem& problem) {
    return cacheable_prefill(problem);
}

KernelVariant select_kernel_cached(const KernelProblem& problem, bool* cache_hit) {
    if (cache_hit)
        *cache_hit = false;
    // Decode / MoE / FP32 / non-eligible dtypes and env-forced benchmark
    // selections always go straight through the pure policy.
    if (!cacheable_prefill(problem) || env_prefill_forced())
        return select_kernel(problem);

    const KernelSignature sig = make_signature(problem);
    CacheState& state = cache_state();
    {
        std::lock_guard<std::mutex> lock(state.m);
        ++state.stats.lookups;
        const auto it = state.entries.find(sig);
        if (it != state.entries.end()) {
            ++state.stats.hits;
            if (cache_hit)
                *cache_hit = true;
            return it->second;
        }
        ++state.stats.misses;
    }
    // Cache miss: deterministic auto rule (no overrides — checked above).
    const KernelVariant variant = select_kernel(problem);
    std::lock_guard<std::mutex> lock(state.m);
    state.entries.emplace(sig, variant);
    return variant;
}

void kernel_cache_record(const KernelProblem& problem, KernelVariant best) {
    if (!cacheable_prefill(problem))
        return;
    CacheState& state = cache_state();
    std::lock_guard<std::mutex> lock(state.m);
    ++state.stats.measured;
    if (best == KernelVariant::MMQ_TENSORCORE)
        ++state.stats.tc_selected;
    else if (best == KernelVariant::MMQ)
        ++state.stats.dp4a_selected;
    state.entries[make_signature(problem)] = best;
}

KernelCacheStats kernel_cache_stats() {
    CacheState& state = cache_state();
    std::lock_guard<std::mutex> lock(state.m);
    return state.stats;
}

void kernel_cache_reset() {
    CacheState& state = cache_state();
    std::lock_guard<std::mutex> lock(state.m);
    state.entries.clear();
    state.stats = KernelCacheStats{};
}

KernelVariant select_kernel(const KernelProblem& problem) {
    switch (problem.domain) {
    case KernelDomain::MatMul:
        return select_matmul(problem);
    case KernelDomain::MoE:
        return select_moe(problem);
    }
    return KernelVariant::UNSUPPORTED;
}

}  // namespace forge

#endif  // USE_CUDA
