// =========================================================================
// cuda_fp8_gemv.cu — Forge W8A8 FP8 E4M3 GEMV production backend (6-C-4.2/4.3)
//
// Hand-written decode GEMV for M == 1 decode. Does NOT use CUTLASS: sm_89 has no
// fp8 GEMV Tensor-Core instruction, and the operator is memory-bandwidth-bound
// (fp8 = 1 byte/element), so a decode (shared 256-entry LUT) + fp32-accum kernel
// is the right shape.
//
// Layout (6-C-5.6 correctness fix): B is the Forge transB weight in its NATIVE
// row-major [N, K] form (row n contiguous, W[n*K + k]) -- the identical buffer
// consumed by the CUTLASS FP8 GEMM (column-major [K,N], ldb=K), the FP32 GEMV and
// the CPU reference. One warp computes one output row (lanes sweep k, one
// coalesced 32B sector per step), then a warp reduce. This is what makes the SAME
// weight Tensor correct for both M == 1 (GEMV) and M > 1 (GEMM) dispatch. The
// 6-C-4.2 bandwidth audit (tests/test_fp8_gemv_cuda.cu) explored VEC widths on an
// artificial k-major layout; VEC=4 was the best there, but the production kernel
// MUST consume the real row-major weight layout, so it uses the row-warp shape
// above (identical structure to the proven FP32 gemv_transB_kernel).
//
// Graph discipline (6-C-4.1c proof): the caller owns the scale device scalars
// (graph-visible); the decoder LUT is a static device buffer allocated once (a
// constant address, also graph-visible). Workspace is 0.
//
// Arch handling: this is pure CUDA (no CUTLASS), so it always compiles for the
// configured sm_89+ target. The launcher does a runtime capability check and
// returns false on pre-Ada GPUs.
// =========================================================================

#include "forge/fp8_gemv.h"
#include "forge/fp8_gemm.h"  // Fp8W8A8Scales
#include "forge/fp8_utils.h"  // fp8_e4m3_to_fp32

#include <cuda_runtime.h>

#include <cstdio>

namespace forge {
namespace cuda {

namespace {

// --- FP8 E4M3 decode LUT (shared memory, 256 entries) ----------------------
// Value = fp8_e4m3_to_fp32(byte), so it is bit-identical to the host golden.
// Allocated once into a static device buffer (constant address, graph-visible).
const float* get_device_decode_lut() {
    static float* d_lut = nullptr;
    if (d_lut == nullptr) {
        float host[256];
        for (int i = 0; i < 256; ++i) host[i] = fp8_e4m3_to_fp32(static_cast<uint8_t>(i));
        if (cudaMalloc(&d_lut, sizeof(host)) != cudaSuccess) return nullptr;
        if (cudaMemcpy(d_lut, host, sizeof(host), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_lut);
            d_lut = nullptr;
            return nullptr;
        }
    }
    return d_lut;
}

bool fp8_gemv_supported() {
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess)
        return false;
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess)
        return false;
    return (major * 10 + minor) >= 89;
}

// W is the transB weight in Forge's native layout: row-major [N, K] with row n
// (output channel) contiguous, i.e. W[n*K + k] -- the SAME buffer that the
// CUTLASS W8A8 GEMM (column-major [K,N], ldb=K) and the FP32 GEMV
// (gemv_transB_kernel) consume. One output row per warp; the 32 lanes sweep k so
// each warp step reads one coalesced 32B sector of the row.
__global__ void gemv_fp8_e4m3(float* __restrict__ D, const uint8_t* __restrict__ A,
                              const uint8_t* __restrict__ W, int K, int N,
                              const float* __restrict__ scale_a,
                              const float* __restrict__ scale_w,
                              const float* __restrict__ lut) {
    extern __shared__ __align__(16) char smem[];
    float* LUT = reinterpret_cast<float*>(smem);
    uint8_t* Ash = reinterpret_cast<uint8_t*>(smem + 256 * sizeof(float));
    const int tid = threadIdx.x;

    for (int i = tid; i < 256; i += blockDim.x) LUT[i] = lut[i];
    for (int k = tid; k < K; k += blockDim.x) Ash[k] = A[k];
    __syncthreads();

    // 6-C-5.2: the product is formed ON DEVICE from the two per-tensor scale
    // scalars (both graph-visible, owned by their Tensors). No host round-trip and
    // no per-call allocation, so a captured graph keeps reading the same addresses
    // and a scale mutation is visible on replay without rebuilding the graphExec.
    const float s = (*scale_a) * (*scale_w);

    const int warp = blockIdx.x * (blockDim.x / 32) + (tid >> 5);
    if (warp >= N) return;
    const int lane = tid & 31;

    const uint8_t* row = W + (size_t)warp * K;
    float acc = 0.f;
    for (int k = lane; k < K; k += 32) {
        const float af = LUT[Ash[k]];
        const float wf = LUT[row[k]];
        acc += af * wf;
    }
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0) D[warp] = acc * s;
}

}  // namespace

bool launch_fp8_gemv_e4m3_fp32(const uint8_t* A_fp8, const uint8_t* B_fp8, float* D_fp32,
                               int K, int N, const Fp8W8A8Scales& sc, cudaStream_t stream) {
    if (!fp8_gemv_supported()) return false;
    if (!sc.scale_a || !sc.scale_w) return false;

    const float* lut = get_device_decode_lut();
    if (!lut) return false;

    // 6-C-5.2/5.4: the scales are consumed ON DEVICE. No cudaMalloc, no D2H, no H2D
    // per call -- both scalars are stable, Tensor-owned, graph-visible addresses.
    const int BDIM = 256;
    const int warps_per_block = BDIM / 32;
    const int grid = (N + warps_per_block - 1) / warps_per_block;
    const size_t shm = 256 * sizeof(float) + static_cast<size_t>(K);

    gemv_fp8_e4m3<<<grid, BDIM, shm, stream>>>(D_fp32, A_fp8, B_fp8, K, N, sc.scale_a,
                                               sc.scale_w, lut);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace cuda
}  // namespace forge
