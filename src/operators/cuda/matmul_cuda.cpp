// Phase 4: CUDA dispatch for the matmul family (launch side).
// Policy decisions (GEMV / MMQ / TensorCore / MoE) live in the unified kernel
// selector (forge/kernel_selector.{h,cpp}); this file only routes the selected
// KernelVariant to the matching kernel launch.
// Originally moved out of src/operators/cpu/matmul.cpp (commit 1: pure move).

#include "matmul_cuda.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA

#    include <algorithm>
#    include <atomic>
#    include <cmath>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <functional>
#    include <stdexcept>
#    include <vector>

#    include <cuda_runtime.h>

#    include "forge/cuda_kernels.h"
#    include "forge/cuda_mem_pool.h"
#    include "forge/dtype_contract.h"
#    include "forge/kernel_selector.h"
#    include "forge/logger.h"
#    include "forge/operator_elementwise.h"
#    include "forge/operator_matmul.h"
#    include "forge/quant_traits.h"
#    include "matmul_dispatch.h"
#    include "forge/fp8_gemm.h"
#    include "forge/fp8_gemv.h"
#    include "forge/fp8_scale_debug.h"

namespace forge {
namespace ops {

namespace {

using internal::DequantMatrixFn;
using internal::lookup_dequant_matrix;
using internal::lookup_gemv;
using internal::lookup_gemv_batch;
using internal::lookup_mmq;

// Phase 4 hard rule: when CUDA cannot handle the request, report a capability failure
// with the reason. Never silently fall back to the host (no D2H/H2D staging).
[[noreturn]] void capability_failure(const char* op, DataType dt, int M, int K, int N) {
    std::string msg = std::string("CUDA capability failure in ") + op + ": dtype=" +
                      dtype_name(dt) + " M=" + std::to_string(M) + " K=" + std::to_string(K) +
                      " N=" + std::to_string(N) +
                      " has no CUDA kernel (no GEMV, no batched GEMV, no device dequant). "
                      "Cross-device fallback to CPU is disabled.";
    LOG_ERROR(msg);
    throw std::runtime_error(msg);
}

// RAII owner for a device scratch buffer drawn from the caching pool.
// Returns the block to the pool (instead of cudaFree) when the matmul call
// is done; kernels run ordered on the same stream, so the buffer is never
// recycled while the async GEMV that reads it is still in flight.
struct DeviceScratch {
    void* ptr = nullptr;
    ~DeviceScratch() {
        if (ptr)
            forge::cuda_mem::deallocate(ptr);
    }
    DeviceScratch() = default;
    explicit DeviceScratch(void* p) : ptr(p) {}
    DeviceScratch(const DeviceScratch&) = delete;
    DeviceScratch& operator=(const DeviceScratch&) = delete;
    DeviceScratch(DeviceScratch&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    float* as_float() { return static_cast<float*>(ptr); }
};

// Dequantize B on device into a pooled device scratch buffer.
DeviceScratch dequant_b_on_device(DequantMatrixFn fn, const TensorPtr& b, int N, int K) {
    size_t fp32_bytes = (size_t)N * K * sizeof(float);
    DeviceScratch scr(forge::cuda_mem::allocate(fp32_bytes));
    if (!scr.ptr)
        throw std::runtime_error("cuda_mem::allocate failed in dequant_b_on_device");
    fn(b->data(), scr.as_float(), N, K, 0);
    return scr;
}

// 6-C-4.1d: W8A8 FP8 E4M3 GEMM (CUTLASS, Ada). A is E4M3 row-major (M,K); B is
// the transB weight [N,K] row-major, which is exactly CUTLASS column-major [K,N]
// (ldb = K), so its pointer is passed through as-is.
// 6-C-5.7: an FP8 consumer reads its per-tensor scale exclusively from the Tensor
// sidecar (Tensor::fp8_scale_device) -- graph-visible, stable, Tensor-owned device
// state that travels with the TensorPtr along ComputeGraph edges. The legacy host
// scale registry (set_fp8_tensor_scale / fp8_scale_registry) is deleted; there is
// no fallback. `ok` is false when the sidecar is unset.
static const float* fp8_scale_device_of(const TensorPtr& t, bool* ok) {
    if (t->has_fp8_scale()) {
        const float* d = t->fp8_scale_device(/*stream=*/0);
        fp8_scale_trace("fp8_consumer", t, /*used_registry_fallback=*/false);
        if (d) {
            *ok = true;
            return d;  // stable, Tensor-owned, graph-visible device scalar
        }
    }
    *ok = false;
    return nullptr;
}

static void fp8_w8a8_matmul(const TensorPtr& a, const TensorPtr& b, float* o_data, int M,
                            int K, int N) {
    const uint8_t* a_fp8 = static_cast<const uint8_t*>(a->data());
    const uint8_t* b_fp8 = static_cast<const uint8_t*>(b->data());
    bool ok_a = false, ok_b = false;
    const float* ra = fp8_scale_device_of(a, &ok_a);
    const float* rb = fp8_scale_device_of(b, &ok_b);
    if (!ok_a || !ok_b) {
        throw std::runtime_error(
            "fp8_w8a8_matmul: FP8 activation/weight scales missing (6-C-5: the scale "
            "sidecar must be set on the FP8 Tensor, e.g. by the quantize producer).");
    }

    // 6-C-5.2: no per-call cudaMalloc/cudaMemcpy -- these are the Tensors' own,
    // stable device scalars (graph-visible state).
    cuda::Fp8W8A8Scales sc;
    sc.scale_a = ra;
    sc.scale_w = rb;
    const bool ok = cuda::launch_fp8_gemm_e4m3_fp32(a_fp8, b_fp8, o_data, M, K, N, sc, 0);
    if (!ok)
        capability_failure("matmul_transB(fp8)", DataType::FP8_E4M3, M, K, N);
}

// 6-C-4.2/4.3/5.6: W8A8 FP8 E4M3 GEMV (M == 1 decode). A is E4M3 (length K); B is
// the transB weight in Forge's native row-major [N,K] form (the same buffer the
// CUTLASS FP8 GEMM reads). Scale plumbing reads the Tensor FP8 scale sidecar
// (6-C-5); the legacy host registry is deleted (6-C-5.7).
static void fp8_w8a8_gemv(const TensorPtr& a, const TensorPtr& b, float* o_row, int K, int N) {
    const uint8_t* a_fp8 = static_cast<const uint8_t*>(a->data());
    const uint8_t* b_fp8 = static_cast<const uint8_t*>(b->data());
    bool ok_a = false, ok_b = false;
    const float* ra = fp8_scale_device_of(a, &ok_a);
    const float* rb = fp8_scale_device_of(b, &ok_b);
    if (!ok_a || !ok_b) {
        throw std::runtime_error(
            "fp8_w8a8_gemv: FP8 activation/weight scales missing (6-C-5: the scale "
            "sidecar must be set on the FP8 Tensor, e.g. by the quantize producer).");
    }

    // 6-C-5.2: no per-call cudaMalloc/cudaMemcpy; the GEMV forms s = s_a * s_w on
    // device from these stable, Tensor-owned scalars (graph-visible state).
    cuda::Fp8W8A8Scales sc;
    sc.scale_a = ra;
    sc.scale_w = rb;
    const bool ok = cuda::launch_fp8_gemv_e4m3_fp32(a_fp8, b_fp8, o_row, K, N, sc, 0);
    if (!ok)
        capability_failure("matmul_transB(fp8/gemv)", DataType::FP8_E4M3, 1, K, N);
}

// M == 1 (decode) single-row dispatch. The unified selector picks the variant;
// each case launches. A null lookup after a non-UNSUPPORTED variant cannot
// normally happen (the selector checks the same tables) and is treated as a
// capability failure for defence in depth.
void gemv_dispatch_row(const TensorPtr& a, const TensorPtr& b, float* o_row, int K, int N_cols,
                       const char* op_name) {
    const float* a_data = static_cast<const float*>(a->data());
    const DataType dt = b->dtype();

    PERF_SCOPE_FMT("cuda/gemv_%s", dtype_name(dt));

    KernelProblem problem;
    problem.domain = KernelDomain::MatMul;
    problem.dtype = dt;
    problem.M = 1;
    problem.N = N_cols;
    problem.K = K;

    switch (select_kernel(problem)) {
    case KernelVariant::GEMV_Q8_1: {
        // Dedicated single-row quantized GEMV (Q8_1 + dp4a).
        auto gemv = lookup_gemv(dt);
        if (!gemv)
            capability_failure(op_name, dt, 1, K, N_cols);
        gemv(a_data, b->data(), o_row, K, N_cols, 0);
        return;
    }
    case KernelVariant::GEMV_BATCH: {
        // No dedicated M == 1 kernel: reuse the batched GEMV (on-the-fly
        // dequant) with M == 1 before dequantizing the whole weight to FP32.
        auto gemv_batch = lookup_gemv_batch(dt);
        if (!gemv_batch)
            capability_failure(op_name, dt, 1, K, N_cols);
        gemv_batch(a_data, b->data(), o_row, 1, K, N_cols, 0);
        return;
    }
    case KernelVariant::GEMV_DEQUANT_TRANSB: {
        auto dequant_matrix = lookup_dequant_matrix(dt);
        if (!dequant_matrix)
            capability_failure(op_name, dt, 1, K, N_cols);
        auto b_scratch = dequant_b_on_device(dequant_matrix, b, N_cols, K);
        cuda::launch_gemv_transB(a_data, b_scratch.as_float(), o_row, K, N_cols);
        return;
    }
    case KernelVariant::GEMV_FP32_TRANSB:
        cuda::launch_gemv_transB(a_data, static_cast<const float*>(b->data()), o_row, K, N_cols);
        return;
    case KernelVariant::GEMV_FP32_CUBLAS:
        // FP32: use cuBLAS for large N (output_proj), custom GEMV for small N.
        cuda::launch_cublas_sgemm(a_data, static_cast<const float*>(b->data()), o_row, 1, K,
                                  N_cols, true);
        return;
    case KernelVariant::MATMUL_FP8_E4M3_GEMV:
        // 6-C-4.2/4.3: W8A8 FP8 E4M3 decode GEMV (M == 1). Hand-written, no CUTLASS.
        fp8_w8a8_gemv(a, b, o_row, K, N_cols);
        return;
    case KernelVariant::UNSUPPORTED:
    default:
        capability_failure(op_name, dt, 1, K, N_cols);
    }
}

// True when a benchmark env override pins the prefill path (mirrors the
// selector's env_prefill_forced). Autotune must never overturn an explicit
// FORGE_MATMUL_PATH / FORGE_FORCE_TC / FORGE_FORCE_MMQ regression switch.
bool prefill_override_env_active() {
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

// FORGE_AUTOTUNE=1 (opt-in): one-shot real-tensor comparison of the MMQ (dp4a)
// and TensorCore candidates, on a cache miss only, for the measurable prefill
// slice (quantized Q4_0/Q4_K/Q5_K/Q6_K/Q8_0, M > 1, sm80+; Q5_K joined in Phase
// 5-B with its FP16 whole-matrix dequant). Times min-of-3 runs of
// each candidate on the actual inputs/outputs (TensorCore timing includes the
// whole-weight FP16 dequant it pays in real use), records the winner into the
// selector cache and returns it. Later calls with the same signature launch the
// cached winner directly. Off by default: opt-in experiments only.
KernelVariant autotune_prefill(const KernelProblem& problem, const float* a_data,
                               const TensorPtr& b, float* o_data, int M, int K, int N) {
    const DataType dt = b->dtype();
    // Phase 2: if the VRAM-aware policy already pinned Q3_K to MMQ, do not probe
    // the TensorCore path (it would thrash under memory pressure). Respect the
    // resolved policy instead of re-measuring on real tensors.
    if (is_quant_tc_dtype(dt) && quant_matmul_policy(dt) == QuantMatmulPolicy::MMQ)
        return select_kernel(problem);
    auto mmq = lookup_mmq(dt);
    if (!mmq)
        return select_kernel(problem);  // unreachable on the eligible slice; be safe

    // Probe the TensorCore candidate first (FP16 whole-matrix dequant + cuBLAS
    // FP16 GEMM). If it is unavailable there is no competition.
    const bool tc_ok = cuda::launch_cublas_gemm_quant_fp16(dt, a_data, b->data(), o_data, M, K, N,
                                                           /*transB=*/true, 0);
    if (!tc_ok) {
        kernel_cache_record(problem, KernelVariant::MMQ);
        return KernelVariant::MMQ;
    }

    cudaEvent_t t0 = nullptr;
    cudaEvent_t t1 = nullptr;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    auto best_ms = [&](const std::function<void()>& run) -> float {
        float best = 1e30f;
        for (int i = 0; i < 3; ++i) {
            cudaEventRecord(t0, 0);
            run();
            cudaEventRecord(t1, 0);
            cudaEventSynchronize(t1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, t0, t1);
            best = std::min(best, ms);
        }
        return best;
    };
    const float mmq_ms = best_ms([&] { mmq(a_data, b->data(), o_data, M, K, N, 0); });
    const float tc_ms = best_ms([&] {
        cuda::launch_cublas_gemm_quant_fp16(dt, a_data, b->data(), o_data, M, K, N, true, 0);
    });
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    const KernelVariant best =
        (mmq_ms <= tc_ms) ? KernelVariant::MMQ : KernelVariant::MMQ_TENSORCORE;
    kernel_cache_record(problem, best);
    return best;
}

// FORGE_TC_CHECK=1: numeric self-check of the TensorCore (dequant-FP16 +
// cuBLAS) result against the dp4a MMQ reference for the same dtype. Lets us
// confirm the Q3_K prefill tensor-core path (and any other TC-eligible dtype)
// is bit-faithful before trusting it. No overhead unless the env var is set.
static void tc_numeric_selfcheck(DataType dt, const float* a_data, const void* b_data,
                                 const float* o_tc, int M, int K, int N) {
    auto mmq = lookup_mmq(dt);
    if (!mmq) return;
    const size_t bytes = (size_t)M * N * sizeof(float);
    float* ref = nullptr;
    if (cudaMalloc(&ref, bytes) != cudaSuccess) return;
    mmq(a_data, b_data, ref, M, K, N, 0);
    float* h_tc = static_cast<float*>(malloc(bytes));
    float* h_ref = static_cast<float*>(malloc(bytes));
    if (!h_tc || !h_ref) {
        free(h_tc);
        free(h_ref);
        cudaFree(ref);
        return;
    }
    cudaMemcpy(h_tc, o_tc, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref, ref, bytes, cudaMemcpyDeviceToHost);
    float maxerr = 0.0f, maxref = 0.0f;
    const size_t n = (size_t)M * N;
    for (size_t i = 0; i < n; ++i) {
        float d = h_tc[i] - h_ref[i];
        if (d < 0.0f) d = -d;
        if (d > maxerr) maxerr = d;
        float r = h_ref[i];
        if (r < 0.0f) r = -r;
        if (r > maxref) maxref = r;
    }
    fprintf(stderr,
            "[tc-check] %s M=%d N=%d K=%d max|tc-mmq|=%.4e max|mmq|=%.4e rel=%.4e\n",
            dtype_name(dt).c_str(), M, N, K, maxerr, maxref, maxref > 0.0f ? maxerr / maxref : 0.0f);
    free(h_tc);
    free(h_ref);
    cudaFree(ref);
}

// M > 1 (prefill / speculative verification) dispatch.
void matmul_multi_row(const TensorPtr& a, const TensorPtr& b, float* o_data, int M, int K, int N) {
    const float* a_data = static_cast<const float*>(a->data());
    const DataType dt = b->dtype();

    KernelProblem problem;
    problem.domain = KernelDomain::MatMul;
    problem.dtype = dt;
    problem.M = M;
    problem.N = N;
    problem.K = K;

    // Phase 4-B: measured-dispatch cache + opt-in autotuner. With
    // FORGE_AUTOTUNE=1 the first encounter of a cacheable signature measures
    // MMQ vs TensorCore on the real tensors and records the winner, replacing
    // the M>=128 heuristic for that signature thereafter.
    bool cache_hit = false;
    KernelVariant variant = select_kernel_cached(problem, &cache_hit);
    const char* autotune_env = std::getenv("FORGE_AUTOTUNE");
    const bool autotune = autotune_env && autotune_env[0] == '1';
    if (autotune && !prefill_override_env_active() && !cache_hit &&
        kernel_autotune_eligible(problem)) {
        variant = autotune_prefill(problem, a_data, b, o_data, M, K, N);
    }

    switch (variant) {
    case KernelVariant::MMQ_TENSORCORE:
        // Phase 3 (Tensor Core): dequantize the whole B to FP16 once and run a
        // cuBLAS FP16 GEMM with FP32 accumulate. The launcher returns false (and
        // launches nothing) whenever the dtype/arch/cuBLAS combination is not
        // available, in which case we fall through to the verified SIMT kernels
        // below (identical to the pre-Phase-4 TC probe + DP4A fall-through).
        if (cuda::launch_cublas_gemm_quant_fp16(dt, a_data, b->data(), o_data, M, K, N,
                                                /*transB=*/true, 0)) {
            // FORGE_TC_CHECK=1: verify the TC result against the dp4a MMQ
            // reference for this dtype (numeric self-check, no perf impact when
            // unset). Primarily to validate the Q3_K prefill TC path.
            if (std::getenv("FORGE_TC_CHECK")) {
                tc_numeric_selfcheck(dt, a_data, b->data(), o_data, M, K, N);
            }
            return;
        }
        [[fallthrough]];
    case KernelVariant::MMQ: {
        // MMQ tiled kernel: weight scales/mins decode once per tile and are
        // reused across all M rows (dp4a int8 dot), giving sub-linear cost in
        // M. The batched-GEMV fallback re-reads every weight row once per
        // activation row (cost proportional to M).
        auto mmq = lookup_mmq(dt);
        if (!mmq)
            capability_failure("matmul_transB", dt, M, K, N);
        mmq(a_data, b->data(), o_data, M, K, N, 0);
        return;
    }
    case KernelVariant::MATMUL_BATCH_GEMV: {
        auto gemv_batch = lookup_gemv_batch(dt);
        if (!gemv_batch)
            capability_failure("matmul_transB", dt, M, K, N);
        gemv_batch(a_data, b->data(), o_data, M, K, N, 0);
        return;
    }
    case KernelVariant::MATMUL_DEQUANT_CUBLAS: {
        // No batched GEMV kernel: dequantize on device, then cuBLAS.
        auto dequant_matrix = lookup_dequant_matrix(dt);
        if (!dequant_matrix)
            capability_failure("matmul_transB", dt, M, K, N);
        auto b_scratch = dequant_b_on_device(dequant_matrix, b, N, K);
        cuda::launch_cublas_sgemm(a_data, b_scratch.as_float(), o_data, M, K, N, true);
        return;
    }
    case KernelVariant::MATMUL_FP8_E4M3:
        // 6-C-4.1d: W8A8 FP8 E4M3 GEMM (CUTLASS, Ada). Only selected on sm_89+
        // (the launcher returns false otherwise -> capability_failure below).
        fp8_w8a8_matmul(a, b, o_data, M, K, N);
        return;
    case KernelVariant::MATMUL_CUBLAS:
        cuda::launch_cublas_sgemm(a_data, static_cast<const float*>(b->data()), o_data, M, K, N,
                                  true);
        return;
    case KernelVariant::UNSUPPORTED:
    default:
        capability_failure("matmul_transB", dt, M, K, N);
    }
}

}  // namespace

void cuda_apply_bias(const TensorPtr& out, const TensorPtr& bias, int M, int N) {
    const float* bias_data = static_cast<const float*>(bias->data());
    float* o_data = static_cast<float*>(out->data());

    TensorPtr bias_cuda;
    if (bias->device() == DeviceType::CPU) {
        bias_cuda = std::make_shared<Tensor>(bias->dtype(), bias->shape(), DeviceType::CUDA);
        bias_cuda->copy_from(*bias);
        bias_data = static_cast<const float*>(bias_cuda->data());
    }

    if (bias->ndim() == 1) {
        int bias_size = static_cast<int>(bias->shape()[0]);
        for (int m = 0; m < M; ++m) {
            cuda::launch_add_bias(o_data + m * N, bias_data, o_data + m * N, bias_size);
        }
    } else {
        int total = static_cast<int>(out->numel());
        cuda::launch_add_bias(o_data, bias_data, o_data, total);
    }
}

TensorPtr cuda_dequantize_matrix(const TensorPtr& b, int rows, int cols) {
    auto dequant_matrix = lookup_dequant_matrix(b->dtype());
    if (!dequant_matrix)
        capability_failure("dequantize_matrix", b->dtype(), 0, cols, rows);
    auto b_fp32 = std::make_shared<Tensor>(DataType::FP32, b->shape(), DeviceType::CUDA);
    dequant_matrix(b->data(), static_cast<float*>(b_fp32->data()), rows, cols, 0);
    return b_fp32;
}

void cuda_matmul(const TensorPtr& a, const TensorPtr& b_fp32, const TensorPtr& out, int M, int K,
                 int N) {
    // Phase 6-A-1: explicit dtype contract at the CUDA matmul boundary.
    require_dtype_supported("cuda_matmul/a", OpType::MUL_MAT, a->dtype());
    require_dtype_supported("cuda_matmul/b", OpType::MUL_MAT, b_fp32->dtype());
    require_dtype_supported("cuda_matmul/out", OpType::MUL_MAT, out->dtype());

    for (int m = 0; m < M; ++m) {
        cuda::launch_gemv(static_cast<const float*>(a->data()) + m * K,
                          static_cast<const float*>(b_fp32->data()),
                          static_cast<float*>(out->data()) + m * N, K, N);
    }
}

void cuda_matmul_transB(const TensorPtr& a, const TensorPtr& b, const TensorPtr& out, int M, int K,
                        int N, cudaStream_t stream) {
    // Phase 6-A-1: an FP16 weight used to fall through to the FP32 path and be
    // read as float (6-A-0 #8) — silently wrong, not a crash. Reject instead.
    require_dtype_supported("cuda_matmul_transB/a", OpType::MUL_MAT_TRANSB, a->dtype());
    require_dtype_supported("cuda_matmul_transB/b", OpType::MUL_MAT_TRANSB, b->dtype());
    require_dtype_supported("cuda_matmul_transB/out", OpType::MUL_MAT_TRANSB, out->dtype());

    // A natively-BF16 weight (the GGUF loader no longer expands it) meeting the
    // FP32 activation chain: narrow the activation and run a Tensor Core GEMM
    // with FP32 accumulate and FP32 out. Checked BEFORE the all-BF16 gate below,
    // which would otherwise reject the mixed pairing.
    if (b->dtype() == DataType::BF16 && a->dtype() == DataType::FP32 &&
        out->dtype() == DataType::FP32) {
        cuda::launch_cublas_gemm_bf16_fp32(static_cast<const float*>(a->data()), b->data(),
                                          static_cast<float*>(out->data()), M, K, N,
                                          /*transB=*/true, stream);
        return;
    }

    // Phase 6-A-5: native FP16 storage -> FP16 storage GEMM (FP32 accumulate).
    // Triggered only when A and B are both FP16; the op allocates an FP16 out for
    // this branch. Everything else (FP32 / quantized) keeps the existing path, so
    // Phase 5 is untouched.
    // Phase 6-B-4: MUL_MAT_TRANSB is retired from require_no_bf16 -- BF16 A/B
    // take the native BF16 GEMM below (FP32 accumulate, BF16 out). Mixed BF16
    // pairings fall through to the float* path and are rejected loudly there
    // (BF16 bytes must never reach a float reinterpret).
    if (a->dtype() == DataType::BF16 || b->dtype() == DataType::BF16 ||
        out->dtype() == DataType::BF16) {
        if (a->dtype() != DataType::BF16 || b->dtype() != DataType::BF16 ||
            out->dtype() != DataType::BF16) {
            throw std::runtime_error(
                std::string("cuda_matmul_transB: BF16 matmul needs all-BF16 A/B/out, got '") +
                dtype_name(a->dtype()) + "' / '" + dtype_name(b->dtype()) + "' / '" +
                dtype_name(out->dtype()) + "' (mixed dtypes fail loudly).");
        }
        cuda::launch_cublas_gemm_bf16(static_cast<const __nv_bfloat16*>(a->data()),
                                      static_cast<const __nv_bfloat16*>(b->data()),
                                      static_cast<__nv_bfloat16*>(out->data()), M, K, N,
                                      /*transB=*/true, stream);
        return;
    }
    if (a->dtype() == DataType::FP16 && b->dtype() == DataType::FP16) {
        if (out->dtype() != DataType::FP16) {
            throw std::runtime_error(
                "cuda_matmul_transB: FP16 A/B require an FP16 out (6-A-5 native path).");
        }
        cuda::launch_cublas_gemm_fp16(static_cast<const __half*>(a->data()),
                                      static_cast<const __half*>(b->data()),
                                      static_cast<__half*>(out->data()), M, K, N,
                                      /*transB=*/true, stream);
        return;
    }

    float* o_data = static_cast<float*>(out->data());

    if (M > 1)
        matmul_multi_row(a, b, o_data, M, K, N);
    else
        gemv_dispatch_row(a, b, o_data, K, N, "matmul_transB");
}

void cuda_matmul_transB_dual(const TensorPtr& a, const TensorPtr& b1, const TensorPtr& b2,
                             const TensorPtr& out, int M, int K, int N1, int N2) {
    // Phase 6-A-1: same contract as cuda_matmul_transB.
    require_dtype_supported("cuda_matmul_transB_dual/a", OpType::MUL_MAT_TRANSB, a->dtype());
    require_dtype_supported("cuda_matmul_transB_dual/b1", OpType::MUL_MAT_TRANSB, b1->dtype());
    require_dtype_supported("cuda_matmul_transB_dual/b2", OpType::MUL_MAT_TRANSB, b2->dtype());
    require_dtype_supported("cuda_matmul_transB_dual/out", OpType::MUL_MAT_TRANSB, out->dtype());
    // 6-B-1: BF16 passes the gate (lowp-aware) but has no kernel -- reject loudly
    // before any float* cast below.
    require_no_bf16("cuda_matmul_transB_dual/a", OpType::MUL_MAT_TRANSB, a->dtype());
    require_no_bf16("cuda_matmul_transB_dual/b1", OpType::MUL_MAT_TRANSB, b1->dtype());
    require_no_bf16("cuda_matmul_transB_dual/b2", OpType::MUL_MAT_TRANSB, b2->dtype());
    require_no_bf16("cuda_matmul_transB_dual/out", OpType::MUL_MAT_TRANSB, out->dtype());

    (void)K;
    const int N = N1 + N2;
    float* o_data = static_cast<float*>(out->data());

    if (M == 1) {
        // Decode: write b1 into the first N1 columns and b2 into the last N2
        // columns of `out` directly. No intermediate tensor allocation and no
        // device-to-device interleave copies.
        gemv_dispatch_row(a, b1, o_data, K, N1, "matmul_transB_dual");
        gemv_dispatch_row(a, b2, o_data + N1, K, N2, "matmul_transB_dual");
        return;
    }

    // M > 1 (batched prefill / verification): the strided interleave layout
    // cannot be produced by a single contiguous GEMV write, so fall back to the
    // general allocate + interleave path, which is verified for all dtypes.
    auto out1 = ops::matmul_transB(a, b1);
    auto out2 = ops::matmul_transB(a, b2);

    const float* o1_data = static_cast<const float*>(out1->data());
    const float* o2_data = static_cast<const float*>(out2->data());
    for (int m = 0; m < M; ++m) {
        cudaMemcpyAsync(o_data + m * N, o1_data + m * N1, N1 * sizeof(float),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(o_data + m * N + N1, o2_data + m * N2, N2 * sizeof(float),
                        cudaMemcpyDeviceToDevice);
    }
}

TensorPtr cuda_ffn_up_fused(const TensorPtr& input, const TensorPtr& w1, const TensorPtr& w3,
                            const TensorPtr& out, int M, int K, int intermediate_dim) {

    auto separate_path = [&]() {
        auto gate = ops::matmul_transB(input, w1);
        auto up = ops::matmul_transB(input, w3);
        return ops::silu_multiply(gate, up);
    };

    if (w1->dtype() == DataType::Q4_0 && w3->dtype() == DataType::Q4_0) {
        if (M == 1) {
            // Decode: single-token fused GEMV kernel (Q8_1 + dp4a)
            cuda::launch_ffn_up_fused_q4_0_q8_1(static_cast<const float*>(input->data()),
                                                w1->data(), w3->data(),
                                                static_cast<float*>(out->data()), K,
                                                intermediate_dim);
        } else {
            // M > 1: MMQ-backed matmuls (sub-linear in M) + silu_multiply.
            // The old fused batched-GEMV re-read every weight row per
            // activation row, so verification batches scaled linearly with M.
            return separate_path();
        }
    } else if (w1->dtype() == DataType::Q4_K && w3->dtype() == DataType::Q4_K) {
        if (M == 1) {
            // Decode: single-token fused GEMV kernel with Q8_1 + dp4a acceleration
            cuda::launch_ffn_up_fused_q4_k_q8_1(static_cast<const float*>(input->data()),
                                                 w1->data(), w3->data(),
                                                 static_cast<float*>(out->data()), K,
                                                 intermediate_dim);
        } else {
            // M > 1: MMQ-backed matmuls (sub-linear in M) + silu_multiply
            return separate_path();
        }
    } else if (w1->dtype() == DataType::Q5_K && w3->dtype() == DataType::Q5_K) {
        if (M == 1) {
            cuda::launch_ffn_up_fused_q5_k(static_cast<const float*>(input->data()), w1->data(),
                                           w3->data(), static_cast<float*>(out->data()), K,
                                           intermediate_dim);
        } else {
            return separate_path();
        }
    } else if (w1->dtype() == DataType::IQ4_XS && w3->dtype() == DataType::IQ4_XS) {
        if (M == 1) {
            // Decode: IQ4_XS gate + up fused with shared Q8_1 quantization
            cuda::launch_ffn_up_fused_iq4_xs_q8_1(static_cast<const float*>(input->data()),
                                                 w1->data(), w3->data(),
                                                 static_cast<float*>(out->data()), K,
                                                 intermediate_dim);
        } else {
            return separate_path();
        }
    } else if (w1->dtype() == DataType::Q3_K && w3->dtype() == DataType::Q3_K) {
        if (M == 1) {
            // Decode: Q3_K gate + Q3_K up fused with shared Q8_1 quantization
            cuda::launch_ffn_up_fused_q3k_q3k(static_cast<const float*>(input->data()), w1->data(),
                                               w3->data(), static_cast<float*>(out->data()), K,
                                               intermediate_dim);
        } else {
            return separate_path();
        }
    } else if (w1->dtype() == DataType::Q3_K && w3->dtype() == DataType::Q4_K) {
        if (M == 1) {
            // Decode: Q3_K gate + Q4_K up fused with shared Q8_1 quantization
            cuda::launch_ffn_up_fused_q3k_q4k(static_cast<const float*>(input->data()), w1->data(),
                                               w3->data(), static_cast<float*>(out->data()), K,
                                               intermediate_dim);
        } else {
            return separate_path();
        }
    } else if (w1->dtype() == DataType::Q2_K && w3->dtype() == DataType::Q2_K) {
        if (M == 1) {
            // Decode: Q2_K gate + Q2_K up fused with shared Q8_1 quantization
            cuda::launch_ffn_up_fused_q2k_q2k(static_cast<const float*>(input->data()), w1->data(),
                                               w3->data(), static_cast<float*>(out->data()), K,
                                               intermediate_dim);
        } else {
            return separate_path();
        }
    } else {
        return separate_path();
    }

    return out;
}

}  // namespace ops
}  // namespace forge

#endif  // USE_CUDA
