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
//     launch_cublas_gemm_quant_fp16).
//   * MoE: only Q4_0/Q4_K/Q6_K/Q8_0 experts are launchable; anything else is
//     UNSUPPORTED and the caller skips silently (as before).
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

#    include <cuda_runtime.h>

#    include "forge/cuda_kernels.h"
#    include "forge/quant_traits.h"
#    include "matmul_dispatch.h"

namespace forge {
namespace ops {
namespace internal {

cuda::GemvFn lookup_gemv(DataType dt) {
    int idx = static_cast<int>(dt);
    if (idx < 0 || idx >= 21)
        return nullptr;
    return cuda::gemv_dispatch[idx];
}

cuda::GemvBatchFn lookup_gemv_batch(DataType dt) {
    int idx = static_cast<int>(dt);
    if (idx < 0 || idx >= 21)
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
    default:
        return false;
    }
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
// Q4_K/Q4_0 at 128); everything else DP4A/MMQ. Coverage joined across phases:
// Q8_0 after Phase 3.5 added its true M>1 MMQ instance (mmq_dispatch[Q8_0]);
// Q6_K after Phase 5-A (dequant path already existed); Q5_K after Phase 5-B
// added its missing FP16 whole-matrix dequant. Per-signature override inside /
// around each threshold is owned by the 4-B autotuner once opted in
// (FORGE_AUTOTUNE=1); benchmark force envs stay the highest priority.
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
    // Conservative per-dtype thresholds (see kTcMinM); the eligibility gate
    // above already excluded the rest.
    return M >= tc_min_m(dt);
#endif
}

KernelVariant select_matmul(const KernelProblem& p) {
    if (p.M <= 1)
        return select_gemv(p);
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
