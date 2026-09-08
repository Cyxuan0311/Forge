// =========================================================================
// fp8_gemv.h — Forge W8A8 FP8 E4M3 GEMV production backend (6-C-4.2 / 6-C-4.3)
//
// Hand-written decode GEMV (independent of the 6-C-4.1d CUTLASS GEMM path).
// Audited in 6-C-4.2 (tests/test_fp8_gemv_cuda.cu) and corrected for the real
// Forge weight layout in 6-C-5.6 (see below). CUTLASS GEMM (MATMUL_FP8_E4M3)
// stays the prefill / M>1 path.
//
//   A: E4M3 activation, length K (1 token, M == 1)
//   B: E4M3 weight in Forge's NATIVE row-major [N, K] form -- row n (output
//      channel) contiguous, W[n*K + k] -- the exact buffer read by the CUTLASS
//      FP8 GEMM (column-major [K,N], ldb=K), the FP32 GEMV and the CPU
//      reference. 6-C-5.6: a single weight Tensor must be correct for both
//      M == 1 (GEMV) and M > 1 (GEMM), which only holds in this layout.
//   D: FP32 row (N),  D[n] = (s_a * s_w) * sum_k decode(A[k]) * decode(W[n,k])
//
// Per 6-C-1 contract S6: scale = amax / 240; the per-tensor scales are DEVICE
// float scalars (graph-visible state, 6-C-4.1c). All pointers in Fp8W8A8Scales
// must be device pointers the caller owns and keeps alive across replays.
//
// Requires sm_89+ (Ada). On pre-Ada devices the launcher returns false and
// launches nothing, so callers fall back to an FP32 path.
// =========================================================================

#pragma once

#include <cuda_runtime.h>

#include "forge/fp8_gemm.h"  // Fp8W8A8Scales

namespace forge {
namespace cuda {

// W8A8 FP8 E4M3 GEMV (M == 1) -> FP32 output:
//   D = (s_a * s_w) * (decode(A_fp8)^T * decode(W_fp8))
// Returns false (and launches nothing) on pre-Ada devices.
bool launch_fp8_gemv_e4m3_fp32(const uint8_t* A_fp8, const uint8_t* B_fp8, float* D_fp32,
                                int K, int N, const Fp8W8A8Scales& sc,
                                cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
