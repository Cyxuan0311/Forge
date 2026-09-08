// =========================================================================
// cuda_fp8_gemm.cu — Forge W8A8 FP8 E4M3 GEMM production backend (6-C-4.1d)
//
// Lifted from benchmarks/fp8_cutlass_gemm.cu (6-C-4.1 / 6-C-4.1b / 6-C-4.1c).
// Two launchers are exposed in forge::cuda:
//   * launch_fp8_gemm_e4m3_fp32  -> FP32 output        (decode / prefill dequant)
//   * launch_fp8_gemm_e4m3_fp8out -> FP8 E4M3 output   (chain W8A8 layers)
//
// Graph discipline (6-C-4.1c proof): the caller owns the scale device scalars
// (graph-visible), passes an explicit stream, and may mutate a scale between
// replays WITHOUT rebuilding the graph. Workspace is 0 for these configs.
//
// Arch handling: the CUTLASS Sm89 device types are only instantiated for
// __CUDA_ARCH__ >= 890 (host pass included). Non-Ada device passes skip them,
// so this TU is safe to compile as part of a sm_86;89 build (only the sm_89
// fatbinary carries the FP8 kernels). The launchers do a runtime capability
// check and return false on pre-Ada GPUs.
// =========================================================================

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
#define FORGE_FP8_GEMM_IMPL 1
#else
#define FORGE_FP8_GEMM_IMPL 0
#endif

#include "forge/fp8_gemm.h"

#include <cstdio>

#if FORGE_FP8_GEMM_IMPL
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/epilogue/thread/activation.h"

using namespace cutlass;

// Matches benchmarks/fp8_cutlass_gemm.cu exactly (the proven configuration).
using ElementA = float_e4m3_t;     // E4M3 input  (activation / weight)
using ElementB = float_e4m3_t;
using ElementAccumulator = float;
using LayoutA = layout::RowMajor;   // A: (M,K) row-major
using LayoutB = layout::ColumnMajor;  // B: (K,N) column-major
using LayoutC = layout::RowMajor;   // C (beta source) / D

static const int kStages = 3;
static const int kAlignmentA = 16;
static const int kAlignmentB = 16;

using Gemm = gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, float, LayoutC, ElementAccumulator,
    arch::OpClassTensorOp, arch::Sm89,
    gemm::GemmShape<128, 64, 128>, gemm::GemmShape<64, 32, 128>, gemm::GemmShape<16, 8, 32>,
    epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
        epilogue::thread::Identity, float, float, 8, ElementAccumulator, ElementAccumulator>,
    gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages, kAlignmentA, kAlignmentB,
    arch::OpMultiplyAdd>;

using GemmF8 = gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, float_e4m3_t, LayoutC, ElementAccumulator,
    arch::OpClassTensorOp, arch::Sm89,
    gemm::GemmShape<128, 64, 128>, gemm::GemmShape<64, 32, 128>, gemm::GemmShape<16, 8, 32>,
    epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<epilogue::thread::Identity,
                                                                  float_e4m3_t, float_e4m3_t, 8,
                                                                  ElementAccumulator,
                                                                  ElementAccumulator>,
    gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages, kAlignmentA, kAlignmentB,
    arch::OpMultiplyAdd>;

// Generic launch helper shared by both output variants. Matrix/scale pointers are
// passed as void* (the CUTLASS Arguments ctor accepts void const*) so the same
// helper works for FP32 and FP8 element types.
template <typename GemmT>
static bool fp8_gemm_run(GemmT& op, int M, int N, int K, const void* dA, const void* dBcol,
                         void* dD, void* dAux, const void* dSrcC, void* dBias,
                         const float* dScaleA, const float* dScaleB, const float* dScaleC,
                         const float* dScaleD,                          const float* dScaleAux, float* dAbsMaxAux,
                         float* dAbsMaxD, cudaStream_t stream) {
    using EO = typename GemmT::EpilogueOutputOp;
    using ElementCompute = float;
    typename EO::Params::ActivationParams activation_params{ElementCompute(1.f),
                                                           ElementCompute(0.f)};
    typename EO::Params epilogue_params{activation_params, dScaleA, dScaleB, dScaleC, dScaleD,
                                        dScaleAux, dAbsMaxAux, dAbsMaxD};

    const size_t sz = (size_t)M * N;
    typename GemmT::Arguments args(
        gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,
        epilogue_params,
        dA, dBcol, dSrcC, dD, dAux, dBias,
        (int64_t)((size_t)M * K),   // batch stride A
        (int64_t)((size_t)K * N),   // batch stride B
        (int64_t)sz,                // batch stride C
        (int64_t)sz,                // batch stride D
        (int64_t)M,                 // batch stride Vector
        (int64_t)K,                 // lda (A row-major, MxK)
        (int64_t)K,                 // ldb (B col-major, KxN)
        (int64_t)N,                 // ldc
        (int64_t)N,                 // ldd
        (int64_t)0);                // ldr (vector, unused)

    cutlass::Status can = op.can_implement(args);
    if (can != cutlass::Status::kSuccess) {
        fprintf(stderr, "[fp8_gemm] can_implement failed: %s\n",
                cutlassGetStatusString(can));
        return false;
    }
    size_t ws_bytes = GemmT::get_workspace_size(args);
    void* ws = nullptr;
    if (ws_bytes) {
        if (cudaMalloc(&ws, ws_bytes) != cudaSuccess) return false;
    }
    cutlass::Status init = op.initialize(args, ws, stream);
    bool ok = (init == cutlass::Status::kSuccess);
    if (ok) {
        cutlass::Status run = op(stream);
        ok = (run == cutlass::Status::kSuccess);
        if (!ok) {
            fprintf(stderr, "[fp8_gemm] run failed: %s\n", cutlassGetStatusString(run));
        }
    }
    if (ws) cudaFree(ws);
    return ok;
}

static bool fp8_gemm_supported() {
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess)
        return false;
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess)
        return false;
    return (major * 10 + minor) >= 89;
}

static float* fp8_upload_scalar(float v) {
    float* d = nullptr;
    if (cudaMalloc(&d, sizeof(float)) != cudaSuccess) return nullptr;
    if (cudaMemcpy(d, &v, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return nullptr;
    }
    return d;
}

#endif  // FORGE_FP8_GEMM_IMPL

namespace forge {
namespace cuda {

bool launch_fp8_gemm_e4m3_fp32(const uint8_t* A_fp8, const uint8_t* B_fp8, float* D_fp32,
                                int M, int K, int N, const Fp8W8A8Scales& sc,
                                cudaStream_t stream) {
#if FORGE_FP8_GEMM_IMPL
    if (!fp8_gemm_supported()) return false;
    if (!sc.scale_a || !sc.scale_w) return false;

    const size_t sz = (size_t)M * N;
    float *dSrcC = nullptr, *dBias = nullptr, *dAux = nullptr;
    if (cudaMalloc(&dSrcC, sz * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dBias, N * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dAux, sz * sizeof(float)) != cudaSuccess) {
        if (dSrcC) cudaFree(dSrcC);
        if (dBias) cudaFree(dBias);
        if (dAux) cudaFree(dAux);
        return false;
    }
    cudaMemset(dSrcC, 0, sz * sizeof(float));
    cudaMemset(dBias, 0, N * sizeof(float));
    cudaMemset(dAux, 0, sz * sizeof(float));

    float one = 1.f, zero = 0.f;
    float *dC = fp8_upload_scalar(one);
    float *dDsc = fp8_upload_scalar(one);
    float *dAuxS = fp8_upload_scalar(one);
    float *dAMA = fp8_upload_scalar(zero);
    bool ok = dC && dDsc && dAuxS && dAMA;
    if (ok) {
        Gemm op;
        ok = fp8_gemm_run<Gemm>(op, M, N, K, A_fp8, B_fp8, D_fp32, dAux, dSrcC, dBias,
                                sc.scale_a, sc.scale_w, dC, dDsc, dAuxS, dAMA, dAMA, stream);
    }
    if (dC) cudaFree(dC);
    if (dDsc) cudaFree(dDsc);
    if (dAuxS) cudaFree(dAuxS);
    if (dAMA) cudaFree(dAMA);
    cudaFree(dSrcC);
    cudaFree(dBias);
    cudaFree(dAux);
    return ok;
#else
    (void)A_fp8; (void)B_fp8; (void)D_fp32; (void)M; (void)K; (void)N; (void)sc; (void)stream;
    return false;
#endif
}

bool launch_fp8_gemm_e4m3_fp8out(const uint8_t* A_fp8, const uint8_t* B_fp8, uint8_t* D_fp8,
                                  int M, int K, int N, const Fp8W8A8Scales& sc,
                                  cudaStream_t stream) {
#if FORGE_FP8_GEMM_IMPL
    if (!fp8_gemm_supported()) return false;
    if (!sc.scale_a || !sc.scale_w || !sc.scale_d || !sc.amax_out) return false;

    const size_t sz = (size_t)M * N;
    uint8_t *dSrcC = nullptr, *dAux = nullptr;
    float* dBias = nullptr;
    if (cudaMalloc(&dSrcC, sz) != cudaSuccess ||
        cudaMalloc(&dAux, sz) != cudaSuccess ||
        cudaMalloc(&dBias, N * sizeof(float)) != cudaSuccess) {
        if (dSrcC) cudaFree(dSrcC);
        if (dAux) cudaFree(dAux);
        if (dBias) cudaFree(dBias);
        return false;
    }
    cudaMemset(dSrcC, 0, sz);
    cudaMemset(dAux, 0, sz);
    cudaMemset(dBias, 0, N * sizeof(float));

    float one = 1.f, zero = 0.f;
    float *dC = fp8_upload_scalar(one);
    float *dAuxS = fp8_upload_scalar(one);
    float *dAMA = fp8_upload_scalar(zero);
    bool ok = dC && dAuxS && dAMA;
    if (ok) {
        GemmF8 op;
        ok = fp8_gemm_run<GemmF8>(op, M, N, K, A_fp8, B_fp8, D_fp8, dAux, dSrcC, dBias,
                                  sc.scale_a, sc.scale_w, dC, sc.scale_d, dAuxS, dAMA,
                                  sc.amax_out, stream);
    }
    if (dC) cudaFree(dC);
    if (dAuxS) cudaFree(dAuxS);
    if (dAMA) cudaFree(dAMA);
    cudaFree(dSrcC);
    cudaFree(dAux);
    cudaFree(dBias);
    return ok;
#else
    (void)A_fp8; (void)B_fp8; (void)D_fp8; (void)M; (void)K; (void)N; (void)sc; (void)stream;
    return false;
#endif
}

}  // namespace cuda
}  // namespace forge
