// 6-C-4.0 FP8 W8A8 Matmul Backend Audit
// ---------------------------------------------------------------------------
// Standalone audit harness (NOT part of the CTest suite). Measures the three
// candidate routes on the real target (RTX 4050, sm_89) and reports actual
// throughput + numeric parity vs a CPU FP32 reference.
//
//   Route A : cuBLASLt FP8 GEMM   (CUDA_R_8F_E4M3 A/B, CUBLAS_COMPUTE_32F)
//   Route C : cuBLASLt FP16 GEMM  (baseline / fallback)
//   Route B : custom mma.m16n8k32  (wmma experimental fp8, FP32 accumulate)
//
// Scale contract (per forge 6-C-1): per-tensor, E4M3 weights & activations,
// FP32 accumulation, dequant folded as C = (s_a*s_w) * (A_fp8 @ B_fp8).
//
// Build:
//   nvcc -O3 -arch=sm_89 -I<forge>/include \
//        -o fp8_matmul_audit fp8_matmul_audit.cu -lcublasLt -lcublas
//
// The custom FP8 MMA kernel is guarded for __CUDA_ARCH__ >= 890 (sm_86 has no
// FP8 Tensor Core); this audit only ever runs on sm_89.

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "forge/fp8_utils.h"

using namespace forge;

// --------------------------------------------------------------------------
// Timing helper
// --------------------------------------------------------------------------
static float bench_cuda(const char* name, int iters, cudaStream_t stream,
                        std::function<cudaError_t()> kernel, bool* ok) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaError_t warm = kernel();
    cudaDeviceSynchronize();
    if (warm != cudaSuccess) {
        if (ok)
            *ok = false;
        printf("  [launch error in %s: %s]\n", name, cudaGetErrorString(warm));
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

// --------------------------------------------------------------------------
// Reference (CPU FP32) scalar element for parity sampling.
//   C[i,j] = alpha * sum_k A[i,k]*B[k,j]
// We evaluate only a small set of SAMPLED (i,j) positions (O(S*K)) instead of
// the full O(M*N*K) naive GEMM, so the audit stays fast even at N=K=8192.
// --------------------------------------------------------------------------
struct Sample {
    int i, j;
    float ref;
};

static float ref_scalar(const std::vector<uint8_t>& Af, const std::vector<uint8_t>& Bf, int i,
                        int j, int K, int N, float alpha) {
    float acc = 0.f;
    const uint8_t* Arow = Af.data() + (size_t)i * K;
    const uint8_t* Bcol = Bf.data() + (size_t)j;  // B is KxN row-major
    for (int k = 0; k < K; ++k)
        acc += fp8_e4m3_to_fp32(Arow[k]) * fp8_e4m3_to_fp32(Bcol[(size_t)k * N]);
    return acc * alpha;
}

// --------------------------------------------------------------------------
// Route B: custom FP8 Tensor-Core GEMM via wmma experimental fp8.
// A: MxK row-major fp8 ; B: KxN row-major fp8 ; C: MxN row-major fp32.
// C = alpha * (A @ B), FP32 accumulate inside the MMA.
// --------------------------------------------------------------------------
// Route B: hand-written FP8 Tensor-Core GEMM via raw PTX
//   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// A: MxK row-major fp8 (1B/elem) ; B: KxN row-major fp8 ; C: MxN fp32.
// C = alpha * (A @ B), FP32 accumulate inside the MMA.
// Operand packing (canonical CP layout, matches CUTLASS):
//   A 16x32 per lane: row = lane%16, K-seg = (lane/16)*16 + {0..15}; 4 regs x4 e4m3.
//   B  8x32 per lane: col = gCol + lane%8, K-seg = (lane/8)*8 + {0..7}; 2 regs x4 e4m3.
//   D 16x8 f32 per lane: 4 regs, cols = gCol + (lane/16)*4 + {0..3}.
__global__ void fp8_gemm_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K, float alpha) {
    const int lane = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int wm = warpId >> 1;  // 0..3  (TILE_M = 64)
    const int wn = warpId & 1;   // 0..1  (TILE_N = 16)
    const int gRow = blockIdx.y * 64 + wm * 16;
    const int gCol = blockIdx.x * 16 + wn * 8;
    const int r = gRow + (lane & 15);

    float d[4] = {0.f, 0.f, 0.f, 0.f};

    const uint8_t* Arow = A + (size_t)r * K;
    const int aK = (lane >> 4) * 16;  // 0 or 16 within the 32-K step
    const uint8_t* Bbase = B + (size_t)gCol + (lane & 7);
    const int bK = (lane >> 3) & 31;  // (lane/8)*8 -> 0/8/16/24

    for (int k = 0; k < K; k += 32) {
        if (r >= M)
            continue;
        uint32_t ra[4], rb[2];
        __builtin_memcpy(&ra[0], Arow + k + aK + 0, 4);
        __builtin_memcpy(&ra[1], Arow + k + aK + 4, 4);
        __builtin_memcpy(&ra[2], Arow + k + aK + 8, 4);
        __builtin_memcpy(&ra[3], Arow + k + aK + 12, 4);
        __builtin_memcpy(&rb[0], Bbase + (size_t)(k + bK + 0) * N, 4);
        __builtin_memcpy(&rb[1], Bbase + (size_t)(k + bK + 4) * N, 4);
        float c[4] = {d[0], d[1], d[2], d[3]};
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]), "f"(c[0]),
              "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }

    if (r < M) {
        float* Cptr = C + (size_t)r * N + gCol + (lane >> 4) * 4;
        int col0 = gCol + (lane >> 4) * 4;
#pragma unroll
        for (int t = 0; t < 4; ++t)
            if (col0 + t < N)
                Cptr[t] = d[t] * alpha;
    }
}

// --------------------------------------------------------------------------
// Host-side: allocate + randomize FP8 operands, run routes, report.
// --------------------------------------------------------------------------
struct Problem {
    int M, N, K;
};

// --------------------------------------------------------------------------
// Layout repacking for cuBLASLt FP8 (Ada): A/C/D must be CUBLASLT_ORDER_COL32,
// B must be CUBLASLT_ORDER_COL. Our operands are row-major, so repack them.
// --------------------------------------------------------------------------
static void pack_col32(std::vector<uint8_t>& dst, const uint8_t* src, int m, int n) {
    int ld = (m + 31) & ~31;
    dst.assign((size_t)ld * n, 0);
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < m; ++r)
            dst[(size_t)c * ld + r] = src[(size_t)r * n + c];
}
static void pack_col(std::vector<uint8_t>& dst, const uint8_t* src, int m, int n) {
    dst.assign((size_t)m * n, 0);
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < m; ++r)
            dst[(size_t)r + (size_t)c * m] = src[(size_t)r * n + c];
}
static void unpack_col32(std::vector<float>& dst, const float* src, int m, int n) {
    int ld = (m + 31) & ~31;
    dst.assign((size_t)m * n, 0.f);
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < m; ++r)
            dst[(size_t)r * n + c] = src[(size_t)c * ld + r];
}

static void run_one(const Problem& p, float s_a, float s_w, int iters,
                    const std::vector<uint8_t>& Af, const std::vector<uint8_t>& Bf,
                    const std::vector<Sample>& samples) {
    const int M = p.M, N = p.N, K = p.K;
    const float alpha = s_a * s_w;
    const double flops = 2.0 * M * N * K;
    const double bytesA = (double)((size_t)M * K + (size_t)K * N) * 1 + (size_t)M * N * 4;
    const double bytesC = (double)((size_t)M * K + (size_t)K * N) * 2 + (size_t)M * N * 4;

    uint8_t *dA, *dB;
    float* dC;
    cudaMalloc(&dA, (size_t)M * K);
    cudaMalloc(&dB, (size_t)K * N);
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));
    cudaMemcpy(dA, Af.data(), (size_t)M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bf.data(), (size_t)K * N, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasLtHandle_t lt;
    cublasLtCreate(&lt);
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    void* workspace = nullptr;
    cudaMalloc(&workspace, 1 << 22);

    // ---- Route A: cuBLASLt FP8 ----
    // Ada FP8 Tensor Core requires A/C/D in CUBLASLT_ORDER_COL32 and B in
    // CUBLASLT_ORDER_COL, plus per-tensor device scale pointers.
    //   D = scaleD * (scaleA*A @ scaleB*B) + beta*C
    // We set scaleA=s_a, scaleB=s_w, scaleD=1 -> D = (s_a*s_w)(Afp8@Bfp8) = Cref.
    const int ldA = (M + 31) & ~31;
    const int ldC = (M + 31) & ~31;
    std::vector<uint8_t> Af32, Bfc;
    pack_col32(Af32, Af.data(), M, K);
    pack_col(Bfc, Bf.data(), K, N);
    uint8_t *dA32, *dBc;
    cudaMalloc(&dA32, Af32.size());
    cudaMalloc(&dBc, Bfc.size());
    cudaMemcpy(dA32, Af32.data(), Af32.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dBc, Bfc.data(), Bfc.size(), cudaMemcpyHostToDevice);
    size_t szC32 = (size_t)ldC * N;
    float* dC32;
    cudaMalloc(&dC32, szC32 * sizeof(float));

    cublasLtMatmulDesc_t descA;
    cublasLtMatmulDescCreate(&descA, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(descA, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(descA, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    float hScaleA = s_a, hScaleB = s_w, hScaleD = 1.f;
    float *dScaleA, *dScaleB, *dScaleD;
    cudaMalloc(&dScaleA, 4);
    cudaMalloc(&dScaleB, 4);
    cudaMalloc(&dScaleD, 4);
    cudaMemcpy(dScaleA, &hScaleA, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dScaleB, &hScaleB, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dScaleD, &hScaleD, 4, cudaMemcpyHostToDevice);
    cublasLtMatmulDescSetAttribute(descA, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dScaleA,
                                   sizeof(dScaleA));
    cublasLtMatmulDescSetAttribute(descA, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dScaleB,
                                   sizeof(dScaleB));
    cublasLtMatmulDescSetAttribute(descA, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &dScaleD,
                                   sizeof(dScaleD));
    float hScaleC = 0.f;  // beta=0 -> C input unused, but FP8 desc needs the pointer
    float* dScaleC;
    cudaMalloc(&dScaleC, 4);
    cudaMemcpy(dScaleC, &hScaleC, 4, cudaMemcpyHostToDevice);
    cublasLtMatmulDescSetAttribute(descA, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &dScaleC,
                                   sizeof(dScaleC));
    cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t col = CUBLASLT_ORDER_COL;
    cublasLtMatrixLayout_t AdA, BdA, CdA;
    cublasLtMatrixLayoutCreate(&AdA, CUDA_R_8F_E4M3, M, K, ldA);
    cublasLtMatrixLayoutCreate(&BdA, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&CdA, CUDA_R_32F, M, N, ldC);  // C/D in COL32
    cublasLtMatrixLayoutSetAttribute(AdA, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32));
    cublasLtMatrixLayoutSetAttribute(BdA, CUBLASLT_MATRIX_LAYOUT_ORDER, &col, sizeof(col));
    cublasLtMatrixLayoutSetAttribute(CdA, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32));
    float alphaA = 1.f, beta = 0.f;
    bool okA = true;
    // Use the documented heuristic to obtain a valid algorithm (passing nullptr
    // as algo can yield CUBLAS_STATUS_INVALID_VALUE on some configs).
    cublasLtMatmulHeuristicResult_t resA;
    cublasLtMatmulPreference_t prefA;
    cublasLtMatmulPreferenceCreate(&prefA);
    size_t wsBytes = 1 << 22;
    cublasLtMatmulPreferenceSetAttribute(prefA, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes,
                                         sizeof(wsBytes));
    int returned = 0;
    cublasStatus_t hst =
        cublasLtMatmulAlgoGetHeuristic(lt, descA, AdA, BdA, CdA, CdA, prefA, 1, &resA, &returned);
    cublasLtMatmulPreferenceDestroy(prefA);
    if (hst != CUBLAS_STATUS_SUCCESS || returned == 0) {
        printf("  [heuristic status=%d returned=%d -> no FP8 algo]\n", (int)hst, returned);
        okA = false;
    }
    auto runA = [&]() -> cudaError_t {
        cublasStatus_t st =
            cublasLtMatmul(lt, descA, &alphaA, dA32, AdA, dBc, BdA, &beta, dC32, CdA, dC32, CdA,
                           returned ? &resA.algo : nullptr, workspace, 1 << 22, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("  [cublasLt FP8 error status=%d]\n", (int)st);
            return cudaErrorUnknown;
        }
        return cudaSuccess;
    };
    float msA = bench_cuda("A", iters, stream, runA, &okA);
    std::vector<float> CA((size_t)M * N);
    double maxAbsA = 0;
    if (okA) {
        std::vector<float> C32(szC32);
        cudaMemcpy(C32.data(), dC32, szC32 * sizeof(float), cudaMemcpyDeviceToHost);
        unpack_col32(CA, C32.data(), M, N);
        for (const auto& s : samples)
            maxAbsA = std::max(maxAbsA, (double)fabs(CA[(size_t)s.i * N + s.j] - s.ref));
    }

    // ---- Route C: cuBLASLt FP16 (baseline) ----
    std::vector<uint16_t> Ah((size_t)M * K), Bh((size_t)K * N);
    for (size_t t = 0; t < Ah.size(); ++t)
        Ah[t] = __float2half(fp8_e4m3_to_fp32(Af[t]) * s_a);
    for (size_t t = 0; t < Bh.size(); ++t)
        Bh[t] = __float2half(fp8_e4m3_to_fp32(Bf[t]) * s_w);
    uint16_t *dAh, *dBh;
    cudaMalloc(&dAh, Ah.size() * 2);
    cudaMalloc(&dBh, Bh.size() * 2);
    cudaMemcpy(dAh, Ah.data(), Ah.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dBh, Bh.data(), Bh.size() * 2, cudaMemcpyHostToDevice);
    cublasLtMatmulDesc_t descC;
    cublasLtMatmulDescCreate(&descC, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(descC, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(descC, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t AdC, BdC, CdC;
    cublasLtMatrixLayoutCreate(&AdC, CUDA_R_16F, M, K, K);
    cublasLtMatrixLayoutCreate(&BdC, CUDA_R_16F, K, N, N);
    cublasLtMatrixLayoutCreate(&CdC, CUDA_R_32F, M, N, N);  // Route C output: row-major
    cublasLtMatrixLayoutSetAttribute(AdC, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    cublasLtMatrixLayoutSetAttribute(BdC, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    cublasLtMatrixLayoutSetAttribute(CdC, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    float one = 1.f;
    bool okC = true;
    auto runC = [&]() -> cudaError_t {
        cublasStatus_t st = cublasLtMatmul(lt, descC, &one, dAh, AdC, dBh, BdC, &beta, dC, CdC, dC,
                                           CdC, nullptr, workspace, 1 << 22, stream);
        return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
    };
    float msC = bench_cuda("C", iters, stream, runC, &okC);
    std::vector<float> CC((size_t)M * N);
    double maxAbsC = 0;
    if (okC) {
        cudaMemcpy(CC.data(), dC, CC.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (const auto& s : samples)
            maxAbsC = std::max(maxAbsC, (double)fabs(CC[(size_t)s.i * N + s.j] - s.ref));
    }

    // ---- Route B: custom mma.m16n8k32 (only on sm_89) ----
    float msB = 0;
    double maxAbsB = -1;
    bool okB = true;
    bool ranB = false;
    if (M >= 16) {
        ranB = true;
        dim3 block(8 * 32);  // 8 warps (TILE_M=64, TILE_N=16)
        dim3 grid((N + 15) / 16, (M + 63) / 64);
        auto runB = [&]() -> cudaError_t {
            fp8_gemm_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
            return cudaGetLastError();
        };
        msB = bench_cuda("B", iters, stream, runB, &okB);
        std::vector<float> CB((size_t)M * N);
        if (okB) {
            cudaMemcpy(CB.data(), dC, CB.size() * sizeof(float), cudaMemcpyDeviceToHost);
            for (const auto& s : samples)
                maxAbsB = std::max(maxAbsB, (double)fabs(CB[(size_t)s.i * N + s.j] - s.ref));
        }
    }

    printf(
        "  M=%5d N=%5d K=%5d | A(FP8) %6.2f TFLOPs %6.1f GB/s par=%.2e"
        " | C(FP16) %6.2f TFLOPs %6.1f GB/s par=%.2e",
        M, N, K, flops / (msA / 1000.0) / 1e12, bytesA / (msA / 1000.0) / 1e9, maxAbsA,
        flops / (msC / 1000.0) / 1e12, bytesC / (msC / 1000.0) / 1e9, maxAbsC);
    if (ranB && okB)
        printf(" | B(mma) %6.2f TFLOPs %6.1f GB/s par=%.2e\n", flops / (msB / 1000.0) / 1e12,
               bytesA / (msB / 1000.0) / 1e9, maxAbsB);
    else
        printf(" | B(mma) n/a (M<16 or launch err)\n");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dAh);
    cudaFree(dBh);
    cudaFree(dA32);
    cudaFree(dBc);
    cudaFree(dC32);
    cudaFree(dScaleA);
    cudaFree(dScaleB);
    cudaFree(dScaleC);
    cudaFree(dScaleD);
    cudaFree(workspace);
    cublasLtDestroy(lt);
    cudaStreamDestroy(stream);
}

int main() {
    printf("6-C-4.0 FP8 W8A8 Matmul Backend Audit (RTX 4050 sm_89)\n");
    int devMajor = 0;
    cudaDeviceGetAttribute(&devMajor, cudaDevAttrComputeCapabilityMajor, 0);
    printf("compute capability major = %d\n", devMajor);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::vector<int> Ms = {1, 32, 64, 128, 256, 512, 1024};
    const std::vector<std::pair<int, int>> NKs = {{4096, 4096}, {8192, 8192}};
    const float s_a = 0.02f, s_w = 0.03f;

    for (auto& nk : NKs) {
        int N = nk.first, K = nk.second;
        int Mpad = 1024;  // max M tested; allocated once, reused per M
        std::vector<uint8_t> Af((size_t)Mpad * K), Bf((size_t)K * N);
        for (size_t t = 0; t < Af.size(); ++t)
            Af[t] = fp32_to_fp8_e4m3(dist(rng));
        for (size_t t = 0; t < Bf.size(); ++t)
            Bf[t] = fp32_to_fp8_e4m3(dist(rng));
        std::mt19937 srg(98765);
        const int NSAMP = 2048;
        printf("\n=== N=%d K=%d (per-tensor s_a=%.2f s_w=%.2f) ===\n", N, K, s_a, s_w);
        for (int M : Ms) {
            std::vector<Sample> samples;
            samples.reserve(NSAMP);
            std::uniform_int_distribution<int> diM(0, M - 1);
            std::uniform_int_distribution<int> diN(0, N - 1);
            for (int s = 0; s < NSAMP; ++s) {
                int i = diM(srg), j = diN(srg);
                samples.push_back({i, j, ref_scalar(Af, Bf, i, j, K, N, s_a * s_w)});
            }
            run_one({M, N, K}, s_a, s_w, M <= 64 ? 100 : 20, Af, Bf, samples);
        }
    }
    return 0;
}
