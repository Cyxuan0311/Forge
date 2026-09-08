// =========================================================================
// fp8_gemm.h — Forge W8A8 FP8 E4M3 GEMM production backend (6-C-4.1d)
//
// Wraps the proven CUTLASS Ada FP8 GEMM (benchmarks/fp8_cutlass_gemm.cu,
// 6-C-4.1 / 6-C-4.1b / 6-C-4.1c) as a Forge operator launch surface.
//
//   A: E4M3 row-major  (M, K)
//   B: E4M3 column-major (K, N)  -- i.e. Forge's transB weight [N, K] row-major
//                                    reinterpreted as column-major [K, N] (ldb = K)
//   D: FP32 row-major (M, N) , fused  D = GEMM * (s_a * s_w)
//   (FP8-output variant: D = quantize(GEMM * (s_a*s_w) * s_d); absmax -> amax_out)
//
// Per 6-C-1 contract S6: scale = amax / 240; the per-tensor scales are DEVICE
// float scalars (graph-visible state, 6-C-4.1c). All pointers in Fp8W8A8Scales
// must be device pointers the caller owns and keeps alive across replays.
//
// Requires sm_89+ (Ada FP8 Tensor Cores). On pre-Ada devices the launchers
// return false and launch nothing, so callers can fall back to an FP32 path.
// =========================================================================

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace forge {
namespace cuda {

// Per-tensor DEVICE-side scale sidecars for a W8A8 FP8 GEMM (6-C-1 contract S3).
// scale_a / scale_w are required; scale_d / amax_out only for the FP8-output
// variant. Every pointer is a device float scalar so CUDA Graph capture/replay
// sees the scale as graph-visible state (6-C-4.1c: mutation needs no rebuild).
struct Fp8W8A8Scales {
    const float* scale_a = nullptr;  // device float scalar (activation amax/240)
    const float* scale_w = nullptr;  // device float scalar (weight amax/240)
    const float* scale_d = nullptr;  // device float scalar, FP8-out only (1/s_d)
    float* amax_out = nullptr;       // device float scalar, FP8-out only (absmax of D)
};

// W8A8 FP8 E4M3 GEMM -> FP32 output (fused D = GEMM * (s_a * s_w)).
// Returns false (and launches nothing) on pre-Ada devices.
bool launch_fp8_gemm_e4m3_fp32(const uint8_t* A_fp8, const uint8_t* B_fp8, float* D_fp32, int M,
                               int K, int N, const Fp8W8A8Scales& sc, cudaStream_t stream = 0);

// W8A8 FP8 E4M3 GEMM -> FP8 (E4M3) output:
//   D = quantize( GEMM * (s_a*s_w) * s_d );  absmax of D -> sc.amax_out
// Returns false (and launches nothing) on pre-Ada devices.
bool launch_fp8_gemm_e4m3_fp8out(const uint8_t* A_fp8, const uint8_t* B_fp8, uint8_t* D_fp8, int M,
                                 int K, int N, const Fp8W8A8Scales& sc, cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
