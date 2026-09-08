// 6-C-4.1 CUTLASS Ada FP8 GEMM — minimal standalone benchmark
// ---------------------------------------------------------------------------
// NOT part of the CTest suite. Build:
//   nvcc -O3 -arch=sm_89 --std=c++17 \
//        -I<forge>/include -I<forge>/third_party/cutlass/include \
//        -o fp8_cutlass_gemm fp8_cutlass_gemm.cu -lcublasLt -lcublas
//
// Paths measured (same operands, same golden):
//   CUTLASS : SM89 E4M3 x E4M3 -> FP32 accumulate -> FP32 output
//             (v1: NO scale fusion; epilogue = Identity, all scales neutral 1.0)
//   ROUTE_B : hand-written mma.sync.m16n8k32 (Forge Route B reference/fallback)
//   FP16    : cuBLASLt FP16 Tensor Core baseline
//   FP32    : cuBLASLt FP32 baseline (accuracy floor + perf floor)
//
// Golden is computed from the DECODED FP8 operands (not from the original
// random floats), so quantization error is never mistaken for kernel error:
//   C_golden[m,n] = sum_k decode(A_fp8[m,k]) * decode(B_fp8[k,n])   (FP32)
//
// M=1 (decode / GEMV) is deliberately OUT OF SCOPE here -> 6-C-4.2.

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "forge/fp8_utils.h"

using namespace forge;

// --------------------------------------------------------------------------
// CUTLASS Ada FP8 GEMM type
//   A: E4M3 row-major (M,K)   B: E4M3 column-major (K,N)   D: FP32 row-major
//   Threadblock 128x64x128, Warp 64x32x128, MMA 16x8x32, stages 3
//   Epilogue: Identity + FP32 output -> D == FP32 accumulator (no scale fusion)
// --------------------------------------------------------------------------
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementOutput = float;
using ElementAuxOutput = float;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

static int const kStages = 3;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity, ElementOutput, ElementAuxOutput, 8, ElementAccumulator,
    ElementAccumulator>;

using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>, cutlass::gemm::GemmShape<16, 8, 32>, EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages, kAlignmentA, kAlignmentB,
    cutlass::arch::OpMultiplyAdd>;

// Same mainloop, but FP8 (E4M3) output epilogue - used to check that the FP8
// packing path does NOT repeat the gated-activation packing bug.
using ElementOutputF8 = cutlass::float_e4m3_t;
using EpilogueOutputOpF8 = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity, ElementOutputF8, ElementOutputF8, 8, ElementAccumulator,
    ElementAccumulator>;
using GemmF8 = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutputF8, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>, cutlass::gemm::GemmShape<16, 8, 32>, EpilogueOutputOpF8,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages, kAlignmentA, kAlignmentB,
    cutlass::arch::OpMultiplyAdd>;

// --------------------------------------------------------------------------
// Route B: hand-written FP8 Tensor Core GEMM (Forge fallback / reference)
//   A: MxK row-major fp8 ; B: KxN row-major fp8 ; C: MxN row-major fp32
//   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32, FP32 accumulate
// --------------------------------------------------------------------------
// Canonical PTX fragment layout for mma.m16n8k32 (e4m3, f32 accumulate):
//   A (16x32): a0,a1 -> row = lane/4        , k = (lane%4)*4 + {0..3} and +16
//              a2,a3 -> row = lane/4 + 8    , k = (lane%4)*4 + {0..3} and +16
//   B (32x8) : col = lane/4 ; b0 -> k = (lane%4)*4 + {0..3}, b1 -> k = +16
//              (so B must be K-contiguous per column => column-major (k + n*K))
//   D (16x8) : d0,d1 -> C[row][col], C[row][col+1]       row = lane/4
//              d2,d3 -> C[row+8][col], C[row+8][col+1]   col = (lane%4)*2
__global__ void fp8_gemm_kernel(const uint8_t* __restrict__ A,  // MxK row-major
                                const uint8_t* __restrict__ B,  // KxN col-major (k + n*K)
                                float* __restrict__ C, int M, int N, int K, float alpha) {
    const int lane = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int wm = warpId >> 1;  // 0..3  (TILE_M = 64)
    const int wn = warpId & 1;   // 0..1  (TILE_N = 16)
    const int gRow = blockIdx.y * 64 + wm * 16;
    const int gCol = blockIdx.x * 16 + wn * 8;

    const int ra = lane >> 2;        // A row within the 16-row MMA tile (0..7)
    const int kA = (lane & 3) * 4;   // A k offset (0,4,8,12)
    const int kB = (lane >> 3) * 4;  // B k offset (0,4,8,12) -- B uses lane/8
    const int row0 = gRow + ra;
    const int row1 = gRow + ra + 8;
    const int col0 = gCol + (lane & 3) * 2;

    float d[4] = {0.f, 0.f, 0.f, 0.f};

    for (int k = 0; k < K; k += 32) {
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
        if (row0 < M) {
            __builtin_memcpy(&a0, A + (size_t)row0 * K + k + kA, 4);
            __builtin_memcpy(&a1, A + (size_t)row0 * K + k + 16 + kA, 4);
        }
        if (row1 < M) {
            __builtin_memcpy(&a2, A + (size_t)row1 * K + k + kA, 4);
            __builtin_memcpy(&a3, A + (size_t)row1 * K + k + 16 + kA, 4);
        }
        const uint8_t* Bp = B + (size_t)(gCol + (lane & 7)) * K;  // B col = lane%8
        __builtin_memcpy(&b0, Bp + k + kB, 4);
        __builtin_memcpy(&b1, Bp + k + 16 + kB, 4);

        float c[4] = {d[0], d[1], d[2], d[3]};
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]),
              "f"(c[3]));
    }

    if (col0 + 1 < N) {
        if (row0 < M) {
            C[(size_t)row0 * N + col0] = d[0] * alpha;
            C[(size_t)row0 * N + col0 + 1] = d[1] * alpha;
        }
        if (row1 < M) {
            C[(size_t)row1 * N + col0] = d[2] * alpha;
            C[(size_t)row1 * N + col0 + 1] = d[3] * alpha;
        }
    }
}

// Host-side __float2half is unreliable (observed to yield 0) => convert on device.
__global__ void f32_to_half_kernel(const float* __restrict__ src, __half* __restrict__ dst,
                                   size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __float2half(src[i]);
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
struct Sample {
    int i, j;
    float ref;
};

// Golden element from the DECODED fp8 operands (FP32 accumulation).
static float golden_scalar(const std::vector<uint8_t>& Af, const std::vector<uint8_t>& Bf, int i,
                           int j, int K, int N) {
    float acc = 0.f;
    const uint8_t* Arow = Af.data() + (size_t)i * K;
    for (int k = 0; k < K; ++k)
        acc += fp8_e4m3_to_fp32(Arow[k]) * fp8_e4m3_to_fp32(Bf[(size_t)k * N + j]);
    return acc;
}

static float bench_cuda(std::function<cudaError_t()> kernel, int iters, cudaStream_t stream,
                        bool* ok) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaError_t warm = kernel();
    cudaDeviceSynchronize();
    if (warm != cudaSuccess) {
        if (ok)
            *ok = false;
        cudaEventDestroy(s);
        cudaEventDestroy(e);
        return 0.f;
    }
    cudaEventRecord(s, stream);
    for (int i = 0; i < iters; ++i)
        kernel();
    cudaEventRecord(e, stream);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    if (ok)
        *ok = true;
    return ms / iters;
}

static double max_abs_err(const std::vector<float>& C, int N, const std::vector<Sample>& samples) {
    double m = 0;
    for (const auto& s : samples)
        m = std::max(m, (double)fabs(C[(size_t)s.i * N + s.j] - s.ref));
    return m;
}

// --------------------------------------------------------------------------
// One problem size: run all four paths, report TFLOPs / GB/s / parity
// --------------------------------------------------------------------------
static void run_one(int M, int N, int K, int iters, const std::vector<uint8_t>& Af,
                    const std::vector<uint8_t>& Bf, const std::vector<Sample>& samples,
                    cublasLtHandle_t lt, float s_a, float s_w) {
    const double flops = 2.0 * M * N * K;
    auto tflops = [&](double ms) { return flops / (ms / 1000.0) / 1e12; };
    auto gbps = [&](double ms, size_t elemSize) {
        double bytes = (double)((size_t)M * K + (size_t)K * N) * elemSize + (size_t)M * N * 4;
        return bytes / (ms / 1000.0) / 1e9;
    };

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ---------------- device operands ----------------
    // CUTLASS: A row-major (M,K) = Af ; B column-major (K,N): B[k + n*K]
    std::vector<uint8_t> Bcol((size_t)K * N);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            Bcol[(size_t)k + (size_t)n * K] = Bf[(size_t)k * N + n];

    uint8_t *dA, *dBcol;
    cudaMalloc(&dA, (size_t)M * K);
    cudaMalloc(&dBcol, Bcol.size());
    cudaMemcpy(dA, Af.data(), (size_t)M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dBcol, Bcol.data(), Bcol.size(), cudaMemcpyHostToDevice);

    float *dCcut, *dCb, *dCfp16, *dCfp32;
    size_t szC = (size_t)M * N;
    cudaMalloc(&dCcut, szC * sizeof(float));
    cudaMalloc(&dCb, szC * sizeof(float));
    cudaMalloc(&dCfp16, szC * sizeof(float));
    cudaMalloc(&dCfp32, szC * sizeof(float));

    // ---------------- CUTLASS FP8 ----------------
    // Neutral epilogue: alpha=1, beta=0, all scales=1.0, bias=0, Identity
    // => D(fp32) == FP32 accumulator == A_fp8 @ B_fp8  (no scale fusion)
    float hOne = 1.f, hZero = 0.f;
    float *dScaleA, *dScaleB, *dScaleC, *dScaleD, *dScaleAux, *dAbsMaxAux, *dAbsMaxD;
    cudaMalloc(&dScaleA, 4);
    cudaMalloc(&dScaleB, 4);
    cudaMalloc(&dScaleC, 4);
    cudaMalloc(&dScaleD, 4);
    cudaMalloc(&dScaleAux, 4);
    cudaMalloc(&dAbsMaxAux, 4);
    cudaMalloc(&dAbsMaxD, 4);
    cudaMemcpy(dScaleA, &s_a, 4, cudaMemcpyHostToDevice);  // per-tensor tensor sidecar
    cudaMemcpy(dScaleB, &s_w, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dScaleC, &hOne, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dScaleD, &hOne, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dScaleAux, &hOne, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dAbsMaxAux, &hZero, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dAbsMaxD, &hZero, 4, cudaMemcpyHostToDevice);

    float *dSrcC, *dAux, *dBias;
    cudaMalloc(&dSrcC, szC * sizeof(float));
    cudaMalloc(&dAux, szC * sizeof(float));
    cudaMalloc(&dBias, (size_t)N * sizeof(float));
    cudaMemset(dSrcC, 0, szC * sizeof(float));
    cudaMemset(dAux, 0, szC * sizeof(float));
    cudaMemset(dBias, 0, (size_t)N * sizeof(float));

    using ElementCompute = float;  // == Gemm::EpilogueOutputOp::ElementCompute
    typename Gemm::EpilogueOutputOp::Params::ActivationParams activation_params{
        ElementCompute(1.f), ElementCompute(0.f)};
    typename Gemm::EpilogueOutputOp::Params epilogue_params{
        activation_params, dScaleA, dScaleB, dScaleC, dScaleD, dScaleAux, dAbsMaxAux, dAbsMaxD};

    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {M, N, K},  // problem size
                                  1,          // batch
                                  epilogue_params,
                                  dA,
                                  dBcol,
                                  dSrcC,
                                  dCcut,
                                  dAux,
                                  dBias,
                                  (int64_t)((size_t)M * K),  // batch stride A
                                  (int64_t)((size_t)K * N),  // batch stride B
                                  (int64_t)szC,              // batch stride C
                                  (int64_t)szC,              // batch stride D
                                  M,                         // batch stride vector
                                  (int64_t)K,                // lda  (A row-major MxK)
                                  (int64_t)K,                // ldb  (B col-major KxN)
                                  (int64_t)N,                // ldc
                                  (int64_t)N,                // ldd
                                  (int64_t)0};               // ld_vector (must be 0)

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    size_t wsBytes = 0;
    bool okCut = (status == cutlass::Status::kSuccess);
    float msCut = 0;
    void* wsPtr = nullptr;
    if (okCut) {
        wsBytes = Gemm::get_workspace_size(args);
        if (wsBytes)
            cudaMalloc(&wsPtr, wsBytes);
        status = gemm_op.initialize(args, wsPtr);
        okCut = (status == cutlass::Status::kSuccess);
        if (okCut) {
            msCut = bench_cuda(
                [&]() -> cudaError_t {
                    return gemm_op() == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
                },
                iters, stream, &okCut);
        }
    }
    double errCut = -1;
    if (okCut) {
        std::vector<float> hC(szC);
        cudaMemcpy(hC.data(), dCcut, szC * sizeof(float), cudaMemcpyDeviceToHost);
        // CUTLASS fuses the scales, so its output is (raw golden) * s_a * s_w
        const double sw = (double)s_a * (double)s_w;
        errCut = 0;
        for (const auto& s : samples)
            errCut = std::max(errCut, (double)fabs(hC[(size_t)s.i * N + s.j] - s.ref * sw));
    }

    // ---------------- Route B (hand-written MMA) ----------------
    // Reuses dBcol (B column-major / K-contiguous), the same storage CUTLASS uses.
    bool okB = true;
    float msB = 0;
    double errB = -1;
    if (M >= 16) {
        dim3 block(8 * 32);
        dim3 grid((N + 15) / 16, (M + 63) / 64);
        msB = bench_cuda(
            [&]() -> cudaError_t {
                fp8_gemm_kernel<<<grid, block, 0, stream>>>(dA, dBcol, dCb, M, N, K, 1.f);
                return cudaGetLastError();
            },
            iters, stream, &okB);
        if (okB) {
            std::vector<float> hC(szC);
            cudaMemcpy(hC.data(), dCb, szC * sizeof(float), cudaMemcpyDeviceToHost);
            errB = max_abs_err(hC, N, samples);
        }
    } else {
        okB = false;
    }

    // ---------------- FP16 / FP32 cuBLASLt baselines ----------------
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    float one = 1.f, beta = 0.f;
    cublasLtMatrixLayout_t CdL;
    cublasLtMatrixLayoutCreate(&CdL, CUDA_R_32F, M, N, N);
    cublasLtMatrixLayoutSetAttribute(CdL, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    void* workspace = nullptr;
    cudaMalloc(&workspace, 1 << 22);

    // FP32 operands (decode fp8 -> fp32) on device; also the source for FP16.
    std::vector<float> Af32((size_t)M * K), Bf32((size_t)K * N);
    for (size_t t = 0; t < Af32.size(); ++t)
        Af32[t] = fp8_e4m3_to_fp32(Af[t]);
    for (size_t t = 0; t < Bf32.size(); ++t)
        Bf32[t] = fp8_e4m3_to_fp32(Bf[t]);
    float *dAf32, *dBf32;
    cudaMalloc(&dAf32, Af32.size() * 4);
    cudaMalloc(&dBf32, Bf32.size() * 4);
    cudaMemcpy(dAf32, Af32.data(), Af32.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dBf32, Bf32.data(), Bf32.size() * 4, cudaMemcpyHostToDevice);

    // FP16 operands: convert ON DEVICE (host __float2half was observed to yield 0).
    __half *dAh, *dBh;
    cudaMalloc(&dAh, Af32.size() * 2);
    cudaMalloc(&dBh, Bf32.size() * 2);
    const int tpb = 256;
    f32_to_half_kernel<<<((unsigned)Af32.size() + tpb - 1) / tpb, tpb, 0, stream>>>(dAf32, dAh,
                                                                                    Af32.size());
    f32_to_half_kernel<<<((unsigned)Bf32.size() + tpb - 1) / tpb, tpb, 0, stream>>>(dBf32, dBh,
                                                                                    Bf32.size());
    cudaStreamSynchronize(stream);

    cublasLtMatmulDesc_t descH;
    cublasLtMatmulDescCreate(&descH, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(descH, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(descH, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t AdH, BdH;
    cublasLtMatrixLayoutCreate(&AdH, CUDA_R_16F, M, K, K);
    cublasLtMatrixLayoutCreate(&BdH, CUDA_R_16F, K, N, N);
    cublasLtMatrixLayoutSetAttribute(AdH, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    cublasLtMatrixLayoutSetAttribute(BdH, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    bool okH = true;
    float msH = bench_cuda(
        [&]() -> cudaError_t {
            return cublasLtMatmul(lt, descH, &one, dAh, AdH, dBh, BdH, &beta, dCfp16, CdL, dCfp16,
                                  CdL, nullptr, workspace, 1 << 22, stream) == CUBLAS_STATUS_SUCCESS
                       ? cudaSuccess
                       : cudaErrorUnknown;
        },
        iters, stream, &okH);
    double errH = -1;
    if (okH) {
        std::vector<float> hC(szC);
        cudaMemcpy(hC.data(), dCfp16, szC * sizeof(float), cudaMemcpyDeviceToHost);
        errH = max_abs_err(hC, N, samples);
    }

    // FP32 GEMM (operands already uploaded above)
    cublasLtMatmulDesc_t descF;
    cublasLtMatmulDescCreate(&descF, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(descF, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(descF, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t AdF, BdF;
    cublasLtMatrixLayoutCreate(&AdF, CUDA_R_32F, M, K, K);
    cublasLtMatrixLayoutCreate(&BdF, CUDA_R_32F, K, N, N);
    cublasLtMatrixLayoutSetAttribute(AdF, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    cublasLtMatrixLayoutSetAttribute(BdF, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    bool okF = true;
    float msF = bench_cuda(
        [&]() -> cudaError_t {
            return cublasLtMatmul(lt, descF, &one, dAf32, AdF, dBf32, BdF, &beta, dCfp32, CdL,
                                  dCfp32, CdL, nullptr, workspace, 1 << 22,
                                  stream) == CUBLAS_STATUS_SUCCESS
                       ? cudaSuccess
                       : cudaErrorUnknown;
        },
        iters, stream, &okF);
    double errF = -1;
    if (okF) {
        std::vector<float> hC(szC);
        cudaMemcpy(hC.data(), dCfp32, szC * sizeof(float), cudaMemcpyDeviceToHost);
        errF = max_abs_err(hC, N, samples);
    }

    // ---------------- debug: raw values at a few sample positions ----------------
    if (M == 32) {
        std::vector<float> dbg(szC);
        auto at = [&](float* d, int i, int j) {
            cudaMemcpy(dbg.data(), d, szC * sizeof(float), cudaMemcpyDeviceToHost);
            return dbg[(size_t)i * N + j];
        };
        for (int s = 0; s < 3; ++s) {
            int i = samples[s].i, j = samples[s].j;
            printf(
                "    [dbg] (%d,%d) ref=%10.4f  fp32=%10.4f  fp16=%10.4f"
                "  route_b=%10.4f  cutlass=%10.4f\n",
                i, j, samples[s].ref, at(dCfp32, i, j), at(dCfp16, i, j), at(dCb, i, j),
                at(dCcut, i, j));
        }
        printf("    [dbg] A fp8[0]=%3u -> f32=%9.4f | B fp8[0]=%3u -> f32=%9.4f\n", (unsigned)Af[0],
               fp8_e4m3_to_fp32(Af[0]), (unsigned)Bf[0], fp8_e4m3_to_fp32(Bf[0]));
    }

    // ---------------- report ----------------
    printf(
        "  M=%4d | FP32 %6.2f TF %7.1f GB/s e=%.1e | FP16 %6.2f TF %7.1f GB/s e=%.1e"
        " | ROUTE_B %6.2f TF %7.1f GB/s e=%.1e | CUTLASS %6.2f TF %7.1f GB/s e=%.1e ws=%zu\n",
        M, okF ? tflops(msF) : 0.0, okF ? gbps(msF, 4) : 0.0, errF, okH ? tflops(msH) : 0.0,
        okH ? gbps(msH, 2) : 0.0, errH, okB ? tflops(msB) : 0.0, okB ? gbps(msB, 1) : 0.0, errB,
        okCut ? tflops(msCut) : 0.0, okCut ? gbps(msCut, 1) : 0.0, errCut, wsBytes);

    cudaFree(dA);
    cudaFree(dBcol);
    cudaFree(dCcut);
    cudaFree(dCb);
    cudaFree(dCfp16);
    cudaFree(dCfp32);
    cudaFree(dScaleA);
    cudaFree(dScaleB);
    cudaFree(dScaleC);
    cudaFree(dScaleD);
    cudaFree(dScaleAux);
    cudaFree(dAbsMaxAux);
    cudaFree(dAbsMaxD);
    cudaFree(dSrcC);
    cudaFree(dAux);
    cudaFree(dBias);
    cudaFree(dAh);
    cudaFree(dBh);
    cudaFree(dAf32);
    cudaFree(dBf32);
    cudaFree(workspace);
    if (wsPtr)
        cudaFree(wsPtr);
    cudaStreamDestroy(stream);
}

// ==========================================================================
// 6-C-4.1 scale / epilogue fusion suite
//
// Frozen 6-C-1 contract (S6), see docs + forge/fp8_utils.h:
//     scale = amax / 240
//     q     = fp8_encode(x / scale)
//     x    ~= decode(q) * scale
//   => D_fp32 = GEMM(decode(A_fp8), decode(W_fp8)) * s_a * s_w
//
// scale_a / scale_w are consumed by the GEMM epilogue as DEVICE-SIDE tensor
// sidecars. They are deliberately NOT wired into ComputeDTypePolicy (policy
// owns dtype only); this suite only proves the epilogue consumes tensor scale.
// ==========================================================================

template <typename GemmT>
static bool gemm_setup_run(GemmT& op, int M, int N, int K, const uint8_t* dA, const uint8_t* dBcol,
                           typename GemmT::EpilogueOutputOp::ElementOutput* dD,
                           typename GemmT::EpilogueOutputOp::ElementAuxOutput* dAux,
                           const float* dSrcC, float* dBias,  // ptr_Vector is void* (non-const)
                           float* dScaleA, float* dScaleB, float* dScaleC, float* dScaleD,
                           float* dScaleAux, float* dAbsMaxAux, float* dAbsMaxD,
                           cudaStream_t stream) {
    typename GemmT::EpilogueOutputOp::Params::ActivationParams ap{1.f, 0.f};
    typename GemmT::EpilogueOutputOp::Params ep{ap,      dScaleA,   dScaleB,    dScaleC,
                                                dScaleD, dScaleAux, dAbsMaxAux, dAbsMaxD};
    size_t sz = (size_t)M * N;
    // Parenthesised init + all 21 ctor params (ldr = ld_vector must be 0, ldaux = N)
    // to avoid brace-init narrowing and the two-overload ambiguity.
    typename GemmT::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, dA,
                                   dBcol, dSrcC, dD, dAux, dBias,
                                   (int64_t)((size_t)M * K),  // batch_stride_A
                                   (int64_t)((size_t)K * N),  // batch_stride_B
                                   (int64_t)sz,               // batch_stride_C
                                   (int64_t)sz,               // batch_stride_D
                                   (int64_t)M,                // batch_stride_Vector
                                   (int64_t)K,                // lda
                                   (int64_t)K,                // ldb
                                   (int64_t)N,                // ldc
                                   (int64_t)N,                // ldd
                                   (int64_t)0,                // ldr (ld_vector, must be 0)
                                   (int64_t)N);               // ldaux
    if (op.can_implement(args) != cutlass::Status::kSuccess)
        return false;
    size_t wsBytes = GemmT::get_workspace_size(args);
    void* ws = nullptr;
    if (wsBytes)
        cudaMalloc(&ws, wsBytes);
    bool ok = op.initialize(args, ws, stream) == cutlass::Status::kSuccess;
    if (ok)
        ok = op(stream) == cutlass::Status::kSuccess;
    cudaDeviceSynchronize();
    if (ws)
        cudaFree(ws);
    return ok;
}

struct ScaleCase {
    const char* name;
    float amax_a;  // drives s_a = amax_a / 240
    float amax_w;
};

// ==========================================================================
// 6-C-4.1c CUDA Graph capture / replay compatibility
//
//   eager -> capture -> replay xN -> bitwise compare vs eager
//   + scale mutation: mutate the DEVICE scale scalars after capture, replay the
//     SAME graphExec, and compare against a fresh eager run with the new scales.
//     Passing this proves scale_a/scale_w/scale_d are graph-visible DEVICE state
//     read at exec time, NOT host values baked into the graph at capture time.
//     (If they were baked in, the final W8A8 runtime would have to rebuild the
//      graph for every distinct tensor scale - unacceptable.)
// ==========================================================================

static inline float out_f(float v) {
    return v;
}
static inline float out_f(cutlass::float_e4m3_t v) {
    return fp8_e4m3_to_fp32(reinterpret_cast<const uint8_t&>(v));
}

template <typename GemmT>
static void run_graph_case(const char* tag, const char* outName, int M, int N, int K,
                           const std::vector<uint8_t>& Af, const std::vector<uint8_t>& Bcol,
                           float s_a, float s_w, float s_d, float* amaxOut) {
    using OutT = typename GemmT::EpilogueOutputOp::ElementOutput;
    using AuxT = typename GemmT::EpilogueOutputOp::ElementAuxOutput;
    const size_t sz = (size_t)M * N;
    const size_t obytes = sz * sizeof(OutT);

    // Capture requires a NON-BLOCKING stream: a default-flag stream is implicitly
    // joined with the legacy default stream, which makes capture unjoinable.
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    uint8_t *dA, *dBcol;
    cudaMalloc(&dA, Af.size());
    cudaMemcpy(dA, Af.data(), Af.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dBcol, Bcol.size());
    cudaMemcpy(dBcol, Bcol.data(), Bcol.size(), cudaMemcpyHostToDevice);

    OutT *dOut, *dAux;
    cudaMalloc(&dOut, obytes);
    cudaMalloc(&dAux, sz * sizeof(AuxT));
    float *dSrcC, *dBias;
    cudaMalloc(&dSrcC, sz * 4);
    cudaMemset(dSrcC, 0, sz * 4);
    cudaMalloc(&dBias, (size_t)N * 4);
    cudaMemset(dBias, 0, (size_t)N * 4);

    float *dSA, *dSW, *dSC, *dSD, *dSAux, *dAMAux, *dAMD;
    cudaMalloc(&dSA, 4);
    cudaMalloc(&dSW, 4);
    cudaMalloc(&dSC, 4);
    cudaMalloc(&dSD, 4);
    cudaMalloc(&dSAux, 4);
    cudaMalloc(&dAMAux, 4);
    cudaMalloc(&dAMD, 4);
    float one = 1.f, zero = 0.f;
    cudaMemcpy(dSA, &s_a, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSW, &s_w, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSC, &one, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSD, &s_d, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSAux, &one, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dAMAux, &zero, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dAMD, &zero, 4, cudaMemcpyHostToDevice);

    typename GemmT::EpilogueOutputOp::Params::ActivationParams ap{1.f, 0.f};
    typename GemmT::EpilogueOutputOp::Params ep{ap, dSA, dSW, dSC, dSD, dSAux, dAMAux, dAMD};
    typename GemmT::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, dA, dBcol, dSrcC, dOut, dAux,
        dBias, (int64_t)Af.size(), (int64_t)Bcol.size(), (int64_t)sz, (int64_t)sz, (int64_t)M,
        (int64_t)K, (int64_t)K, (int64_t)N, (int64_t)N, (int64_t)0, (int64_t)N);
    GemmT op;
    bool initOk = (op.can_implement(args) == cutlass::Status::kSuccess);
    size_t wsBytes = GemmT::get_workspace_size(args);
    void* ws = nullptr;
    if (wsBytes)
        cudaMalloc(&ws, wsBytes);
    if (initOk)
        initOk = (op.initialize(args, ws, stream) == cutlass::Status::kSuccess);

    std::vector<OutT> hEager(sz), hR(sz), hPrev(sz), hMut(sz), hEager2(sz);
    bool capOk = false, allEq = true, stable = true, mutMatch = false, mutChanged = false;

    if (initOk) {
        // ---- eager reference (also performs any lazy host-side setup) ----
        op(stream);
        cudaStreamSynchronize(stream);
        cudaMemcpy(hEager.data(), dOut, obytes, cudaMemcpyDeviceToHost);
        if (amaxOut) {
            float mx = 0;
            for (size_t t = 0; t < sz; ++t)
                mx = std::max(mx, fabsf(out_f(hEager[t])));
            *amaxOut = mx;
        }

        // ---- capture (op must launch on `stream`, or we capture an empty graph) ----
        cudaGraph_t graph = nullptr;
        cudaError_t e1 = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
        cutlass::Status st = op(stream);
        cudaError_t e2 = cudaStreamEndCapture(stream, &graph);
        capOk = (e1 == cudaSuccess) && (st == cutlass::Status::kSuccess) && (e2 == cudaSuccess) &&
                (graph != nullptr);
        size_t nNodes = 0;
        if (capOk) {
            cudaGraphGetNodes(graph, nullptr, &nNodes);  // NULL dst returns count
            capOk = (nNodes > 0);  // an empty capture would trivially pass replay
        }

        cudaGraphExec_t exec = nullptr;
        cudaError_t e3 = cudaSuccess;
        if (capOk) {
            e3 = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            capOk = (e3 == cudaSuccess);
        }
        if (!capOk)
            printf("      [diag] begin=%s(%d) cutlass=%d end=%s(%d) inst=%s(%d) nodes=%zu\n",
                   cudaGetErrorString(e1), (int)e1, (int)st, cudaGetErrorString(e2), (int)e2,
                   cudaGetErrorString(e3), (int)e3, nNodes);

        if (capOk) {
            // ---- replay x3 ----
            for (int i = 0; i < 3; ++i) {
                cudaGraphLaunch(exec, stream);
                cudaStreamSynchronize(stream);
                cudaMemcpy(hR.data(), dOut, obytes, cudaMemcpyDeviceToHost);
                if (memcmp(hR.data(), hEager.data(), obytes) != 0)
                    allEq = false;
                if (i > 0 && memcmp(hR.data(), hPrev.data(), obytes) != 0)
                    stable = false;
                hPrev = hR;
            }
            // ---- scale mutation: same graphExec, new DEVICE scale contents ----
            // Async copies on the SAME stream keep ordering vs the replay.
            float s_a2 = s_a * 2.0f, s_w2 = s_w * 2.0f;
            cudaMemcpyAsync(dSA, &s_a2, 4, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dSW, &s_w2, 4, cudaMemcpyHostToDevice, stream);
            cudaGraphLaunch(exec, stream);
            cudaStreamSynchronize(stream);
            cudaMemcpy(hMut.data(), dOut, obytes, cudaMemcpyDeviceToHost);
            mutChanged = (memcmp(hMut.data(), hEager.data(), obytes) != 0);
            // fresh eager with the same (mutated) device scales -> must match
            op(stream);
            cudaStreamSynchronize(stream);
            cudaMemcpy(hEager2.data(), dOut, obytes, cudaMemcpyDeviceToHost);
            mutMatch = (memcmp(hMut.data(), hEager2.data(), obytes) == 0);

            cudaGraphExecDestroy(exec);
        }
        if (graph)
            cudaGraphDestroy(graph);
    }

    printf(
        "  %-9s %-5s ws=%zu | capture=%s | replay==eager=%s | stable=%s"
        " | scale_mut(match=%s changed=%s)%s\n",
        tag, outName, wsBytes, capOk ? "OK" : "FAIL", allEq ? "YES" : "NO", stable ? "YES" : "NO",
        mutMatch ? "YES" : "NO", mutChanged ? "YES" : "NO", initOk ? "" : " [INIT_FAIL]");

    cudaFree(dA);
    cudaFree(dBcol);
    cudaFree(dOut);
    cudaFree(dAux);
    cudaFree(dSrcC);
    cudaFree(dBias);
    cudaFree(dSA);
    cudaFree(dSW);
    cudaFree(dSC);
    cudaFree(dSD);
    cudaFree(dSAux);
    cudaFree(dAMAux);
    cudaFree(dAMD);
    if (ws)
        cudaFree(ws);
    cudaStreamDestroy(stream);
}

// E4M3 ULP (3 mantissa bits); below the first normal the step is the fixed
// subnormal quantum 2^-9.
static float e4m3_ulp(float v) {
    float a = fabsf(v);
    if (a < exp2f(-6.0f))
        return exp2f(-9.0f);
    int e = (int)floorf(log2f(a));
    return exp2f((float)e - 3.0f);
}

static void run_scale_case(int M, int N, int K, const ScaleCase& c, unsigned seed) {
    // 1) real fp32 operands, normalized so that max|x| == target amax
    std::vector<float> Ar((size_t)M * K), Wr((size_t)K * N);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (auto& v : Ar)
        v = u(rng);
    for (auto& v : Wr)
        v = u(rng);
    float mxA = 0, mxW = 0;
    for (auto v : Ar)
        mxA = std::max(mxA, fabsf(v));
    for (auto v : Wr)
        mxW = std::max(mxW, fabsf(v));
    for (auto& v : Ar)
        v *= (c.amax_a / mxA);
    for (auto& v : Wr)
        v *= (c.amax_w / mxW);
    mxA = 0;
    mxW = 0;
    for (auto v : Ar)
        mxA = std::max(mxA, fabsf(v));
    for (auto v : Wr)
        mxW = std::max(mxW, fabsf(v));
    const float s_a = mxA / 240.0f;  // frozen contract
    const float s_w = mxW / 240.0f;

    // 2) quantize: q = fp8_encode(x / scale)
    std::vector<uint8_t> Af((size_t)M * K), Bf((size_t)K * N);
    for (size_t t = 0; t < Af.size(); ++t)
        Af[t] = fp32_to_fp8_e4m3(Ar[t] / s_a);
    for (size_t t = 0; t < Bf.size(); ++t)
        Bf[t] = fp32_to_fp8_e4m3(Wr[t] / s_w);
    std::vector<uint8_t> Bcol((size_t)K * N);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            Bcol[(size_t)k + (size_t)n * K] = Bf[(size_t)k * N + n];

    // 3) sampled goldens: (a) scaled fp8 golden, (b) true real fp32 matmul
    const int NS = 1024;
    std::vector<int> si(NS), sj(NS);
    std::vector<float> gScaled(NS), gReal(NS);
    std::mt19937 srg(777);
    std::uniform_int_distribution<int> diM(0, M - 1), diN(0, N - 1);
    for (int s = 0; s < NS; ++s) {
        int i = diM(srg), j = diN(srg);
        si[s] = i;
        sj[s] = j;
        const float* arow = Ar.data() + (size_t)i * K;
        const uint8_t* afrow = Af.data() + (size_t)i * K;
        float raw = 0, real = 0;
        for (int k = 0; k < K; ++k) {
            raw += fp8_e4m3_to_fp32(afrow[k]) * fp8_e4m3_to_fp32(Bf[(size_t)k * N + j]);
            real += arow[k] * Wr[(size_t)k * N + j];
        }
        gScaled[s] = raw * ((double)s_a * (double)s_w);
        gReal[s] = real;
    }

    // 4) device
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const size_t sz = (size_t)M * N;
    uint8_t *dA, *dBcol;
    cudaMalloc(&dA, (size_t)M * K);
    cudaMalloc(&dBcol, Bcol.size());
    cudaMemcpy(dA, Af.data(), (size_t)M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dBcol, Bcol.data(), Bcol.size(), cudaMemcpyHostToDevice);

    float *dSrcC, *dBias, *dAuxF, *dDfused, *dDraw;
    cudaMalloc(&dSrcC, sz * 4);
    cudaMalloc(&dBias, (size_t)N * 4);
    cudaMalloc(&dAuxF, sz * 4);
    cudaMalloc(&dDfused, sz * 4);
    cudaMalloc(&dDraw, sz * 4);
    cudaMemset(dSrcC, 0, sz * 4);
    cudaMemset(dBias, 0, (size_t)N * 4);
    cudaMemset(dAuxF, 0, sz * 4);
    ElementOutputF8 *dDf8, *dAuxF8;
    cudaMalloc(&dDf8, sz);
    cudaMalloc(&dAuxF8, sz);

    float one = 1.f, zero = 0.f;
    float *dSA, *dSW, *dSC, *dSD, *dSAux, *dAMAux, *dAMD, *dSA1, *dSW1;
    cudaMalloc(&dSA, 4);
    cudaMalloc(&dSW, 4);
    cudaMalloc(&dSC, 4);
    cudaMalloc(&dSD, 4);
    cudaMalloc(&dSAux, 4);
    cudaMalloc(&dAMAux, 4);
    cudaMalloc(&dAMD, 4);
    cudaMalloc(&dSA1, 4);
    cudaMalloc(&dSW1, 4);
    cudaMemcpy(dSA, &s_a, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSW, &s_w, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSA1, &one, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSW1, &one, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSC, &one, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSD, &one, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSAux, &one, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dAMAux, &zero, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dAMD, &zero, 4, cudaMemcpyHostToDevice);

    Gemm gF;
    bool okF = gemm_setup_run(gF, M, N, K, dA, dBcol, dDfused, dAuxF, dSrcC, dBias, dSA, dSW, dSC,
                              dSD, dSAux, dAMAux, dAMD, stream);
    Gemm gR;
    bool okR = gemm_setup_run(gR, M, N, K, dA, dBcol, dDraw, dAuxF, dSrcC, dBias, dSA1, dSW1, dSC,
                              dSD, dSAux, dAMAux, dAMD, stream);

    std::vector<float> hFused(sz), hRaw(sz);
    cudaMemcpy(hFused.data(), dDfused, sz * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hRaw.data(), dDraw, sz * 4, cudaMemcpyDeviceToHost);

    // Output scale: bring D into the E4M3 legal range before the FP8 epilogue,
    // otherwise the FP8 output simply saturates at 240 and the test is vacuous.
    float amaxD = 0;
    for (size_t t = 0; t < sz; ++t)
        amaxD = std::max(amaxD, fabsf(hFused[t]));
    const float s_d = (amaxD > 0.f) ? (240.0f / amaxD) : 1.f;
    cudaMemcpy(dSD, &s_d, 4, cudaMemcpyHostToDevice);

    GemmF8 g8;
    bool ok8 = gemm_setup_run(g8, M, N, K, dA, dBcol, dDf8, dAuxF8, dSrcC, dBias, dSA, dSW, dSC,
                              dSD, dSAux, dAMAux, dAMD, stream);

    std::vector<uint8_t> hF8(sz);
    cudaMemcpy(hF8.data(), dDf8, sz, cudaMemcpyDeviceToHost);

    double eGolden = 0, eReal = 0, eRawScaled = 0, eF8 = 0, maxUlp = 0;
    int badBit = 0, badFar = 0, nonFinite = 0;
    const double sw = (double)s_a * (double)s_w;
    double refGolden = 0, refReal = 0;
    for (int s = 0; s < NS; ++s) {
        refGolden = std::max(refGolden, (double)fabsf(gScaled[s]));
        refReal = std::max(refReal, (double)fabsf(gReal[s]));
    }
    for (int s = 0; s < NS; ++s) {
        size_t idx = (size_t)si[s] * N + sj[s];
        float f = hFused[idx];
        if (!std::isfinite(f))
            ++nonFinite;
        eGolden = std::max(eGolden, (double)fabsf(f - gScaled[s]));
        eReal = std::max(eReal, (double)fabsf(f - gReal[s]));
        eRawScaled = std::max(eRawScaled, (double)fabsf(f - hRaw[idx] * sw));
        uint8_t expect = fp32_to_fp8_e4m3(f * s_d);
        if (expect != hF8[idx]) {
            ++badBit;
            // Is it a 1-ULP rounding difference, or a real packing/layout bug?
            float dv = fp8_e4m3_to_fp32(hF8[idx]);
            float ev = fp8_e4m3_to_fp32(expect);
            double ratio = (double)fabsf(dv - ev) / (double)e4m3_ulp(ev);
            maxUlp = std::max(maxUlp, ratio);
            if (ratio > 1.5)
                ++badFar;
        }
        eF8 = std::max(eF8, (double)fabsf(fp8_e4m3_to_fp32(hF8[idx]) - f * s_d));
    }
    const double rG = refGolden > 0 ? eGolden / refGolden : 0.0;
    const double rR = refReal > 0 ? eReal / refReal : 0.0;
    const double rS = refGolden > 0 ? eRawScaled / refGolden : 0.0;
    const double rF = eF8 / 240.0;

    printf(
        "  %-9s s_a=%.3e s_w=%.3e s_d=%.3e | golden abs=%.2e rel=%.2e"
        " | real rel=%.2e | raw*sw rel=%.2e | fp8 bad=%4d/%d maxULP=%.2f far=%d"
        " | nonfin=%d%s%s%s\n",
        c.name, s_a, s_w, s_d, eGolden, rG, rR, rS, badBit, NS, maxUlp, badFar, nonFinite,
        okF ? "" : " [FUSED_FAIL]", okR ? "" : " [RAW_FAIL]", ok8 ? "" : " [F8_FAIL]");

    cudaFree(dA);
    cudaFree(dBcol);
    cudaFree(dSrcC);
    cudaFree(dBias);
    cudaFree(dAuxF);
    cudaFree(dDfused);
    cudaFree(dDraw);
    cudaFree(dDf8);
    cudaFree(dAuxF8);
    cudaFree(dSA);
    cudaFree(dSW);
    cudaFree(dSC);
    cudaFree(dSD);
    cudaFree(dSAux);
    cudaFree(dAMAux);
    cudaFree(dAMD);
    cudaFree(dSA1);
    cudaFree(dSW1);
    cudaStreamDestroy(stream);
}

int main() {
    printf("6-C-4.1 CUTLASS Ada FP8 GEMM (RTX 4050 sm_89)\n");
    int devMajor = 0, devMinor = 0;
    cudaDeviceGetAttribute(&devMajor, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&devMinor, cudaDevAttrComputeCapabilityMinor, 0);
    printf("compute capability = %d.%d\n", devMajor, devMinor);
    if (devMajor < 8 || (devMajor == 8 && devMinor < 9)) {
        printf("ERROR: CUTLASS Ada FP8 GEMM requires sm_89 or higher.\n");
        return 1;
    }

    cublasLtHandle_t lt;
    cublasLtCreate(&lt);

    // ---- scale / epilogue fusion suite ----
    {
        const int M = 256, N = 4096, K = 4096;
        printf("\n=== scale/epilogue fusion (contract: scale=amax/240, q=fp8(x/scale)) ===\n");
        printf("  M=%d N=%d K=%d | D_fp32 = GEMM(decode(A),decode(W)) * s_a*s_w\n", M, N, K);
        ScaleCase cases[] = {
            {"scale==1", 240.0f, 240.0f},  // s_a = s_w = 1 exactly
            {"normal", 8.0f, 16.0f},
            {"tiny", 1e-4f, 1e-4f},
            {"huge", 1e4f, 1e4f},
        };
        for (auto& c : cases)
            run_scale_case(M, N, K, c, 4242);
    }

    // ---- 6-C-4.1c graph capture / replay ----
    {
        const int M = 256, N = 4096, K = 4096;
        printf(
            "\n=== 6-C-4.1c graph capture/replay "
            "(eager -> capture -> replay x3 -> scale mutation) ===\n");
        printf("  M=%d N=%d K=%d | FP32: D=GEMM*sa*sw | FP8: q=encode(D*s_d), s_d=240/amax(D)\n", M,
               N, K);
        ScaleCase cases[] = {
            {"scale==1", 240.0f, 240.0f},
            {"normal", 8.0f, 16.0f},
            {"tiny", 1e-4f, 1e-4f},
            {"huge", 1e4f, 1e4f},
        };
        for (auto& c : cases) {
            // operands under the amax/240 contract: q = fp8_encode(x / scale)
            std::vector<float> Ar((size_t)M * K), Wr((size_t)K * N);
            std::mt19937 rng(4242);
            std::uniform_real_distribution<float> u(-1.f, 1.f);
            for (auto& v : Ar)
                v = u(rng);
            for (auto& v : Wr)
                v = u(rng);
            float mxA = 0, mxW = 0;
            for (auto v : Ar)
                mxA = std::max(mxA, fabsf(v));
            for (auto v : Wr)
                mxW = std::max(mxW, fabsf(v));
            for (auto& v : Ar)
                v *= (c.amax_a / mxA);
            for (auto& v : Wr)
                v *= (c.amax_w / mxW);
            mxA = 0;
            mxW = 0;
            for (auto v : Ar)
                mxA = std::max(mxA, fabsf(v));
            for (auto v : Wr)
                mxW = std::max(mxW, fabsf(v));
            const float s_a = mxA / 240.0f, s_w = mxW / 240.0f;
            std::vector<uint8_t> Af((size_t)M * K), Bf((size_t)K * N);
            for (size_t t = 0; t < Af.size(); ++t)
                Af[t] = fp32_to_fp8_e4m3(Ar[t] / s_a);
            for (size_t t = 0; t < Bf.size(); ++t)
                Bf[t] = fp32_to_fp8_e4m3(Wr[t] / s_w);
            std::vector<uint8_t> Bcol((size_t)K * N);
            for (int k = 0; k < K; ++k)
                for (int n = 0; n < N; ++n)
                    Bcol[(size_t)k + (size_t)n * K] = Bf[(size_t)k * N + n];

            float amaxD32 = 0;
            run_graph_case<Gemm>(c.name, "fp32", M, N, K, Af, Bcol, s_a, s_w, 1.f, &amaxD32);
            const float s_d = (amaxD32 > 0.f) ? (240.0f / amaxD32) : 1.f;
            run_graph_case<GemmF8>(c.name, "fp8", M, N, K, Af, Bcol, s_a, s_w, s_d, nullptr);
        }
    }

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::vector<int> Ms = {32, 64, 128, 256, 512, 1024};  // M=1 -> 6-C-4.2
    const std::vector<std::pair<int, int>> NKs = {{4096, 4096}, {8192, 8192}};
    const int NSAMP = 2048;

    for (auto& nk : NKs) {
        const int N = nk.first, K = nk.second;
        const int Mpad = 1024;
        std::vector<uint8_t> Af((size_t)Mpad * K), Bf((size_t)K * N);
        for (size_t t = 0; t < Af.size(); ++t)
            Af[t] = fp32_to_fp8_e4m3(dist(rng));
        for (size_t t = 0; t < Bf.size(); ++t)
            Bf[t] = fp32_to_fp8_e4m3(dist(rng));

        // (1,1) = no-fusion baseline ; (0.02,0.03) = realistic amax/240 per-tensor
        // scales, to confirm scale fusion costs nothing in throughput.
        const std::pair<float, float> scaleCfg[] = {{1.f, 1.f}, {0.02f, 0.03f}};
        for (auto& sc : scaleCfg) {
            printf("\n=== N=%d K=%d  s_a=%g s_w=%g (golden = decoded FP8, FP32 acc) ===\n", N, K,
                   sc.first, sc.second);
            std::mt19937 srg(98765);
            for (int M : Ms) {
                // Sampled golden: O(S*K) instead of O(M*N*K)
                std::vector<Sample> samples;
                samples.reserve(NSAMP);
                std::uniform_int_distribution<int> diM(0, M - 1);
                std::uniform_int_distribution<int> diN(0, N - 1);
                for (int s = 0; s < NSAMP; ++s) {
                    int i = diM(srg), j = diN(srg);
                    samples.push_back({i, j, golden_scalar(Af, Bf, i, j, K, N)});
                }
                // Use Af's first M rows only
                std::vector<uint8_t> AfM((size_t)M * K);
                std::memcpy(AfM.data(), Af.data(), (size_t)M * K);
                run_one(M, N, K, M <= 128 ? 100 : 20, AfM, Bf, samples, lt, sc.first, sc.second);
            }
        }
    }

    cublasLtDestroy(lt);
    return 0;
}
