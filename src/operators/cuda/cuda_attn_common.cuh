// =========================================================================
// cuda_attn_common.cuh — Shared helpers for all CUDA attention kernels
//
// Warp-level reduction primitives and cross-warp merge logic used by
// both cuda_flash_attn.cu (FP32 KV) and cuda_fused_attn.cu (quantized KV).
// =========================================================================

#pragma once

#include <cmath>

namespace forge {
namespace cuda {

// ---- Warp-level reduction primitives ----

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

// ---- Cross-warp merge via shared memory ----
//
// After warp-level reduce, each warp has (warp_max, warp_sum, warp_acc[HEAD_DIM]).
// Warp 0 merges all warps' results into global (max, sum, acc) and writes output.
//
// Template parameter HEAD_DIM must match the kernel's head dimension.
// NUM_WARPS = blockDim.x / 32.
//
// This function MUST be called with __syncthreads() already executed after
// s_warp_max/s_warp_sum/s_warp_acc are populated by all warps.
//
// On entry, only warp_id==0 threads participate. Other warps return immediately.

template <int HEAD_DIM, int NUM_WARPS>
__device__ __forceinline__ void cross_warp_merge_and_write(
    const float* __restrict__ s_warp_max,
    const float* __restrict__ s_warp_sum,
    const float* __restrict__ s_warp_acc,
    float* __restrict__ O,
    int h,  // head index for output writing
    int lane) {

    // Only warp 0 does the final merge
    if (threadIdx.x >> 5 != 0) return;

    // Find global max across warps
    float global_max = -1e30f;
    for (int w = 0; w < NUM_WARPS; ++w) {
        global_max = fmaxf(global_max, s_warp_max[w]);
    }

    // Merge sum and acc with LSE correction
    float global_sum = 0.0f;
    float global_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) global_acc[d] = 0.0f;

    for (int w = 0; w < NUM_WARPS; ++w) {
        float corr = expf(s_warp_max[w] - global_max);
        global_sum += s_warp_sum[w] * corr;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            global_acc[d] += s_warp_acc[w * HEAD_DIM + d] * corr;
        }
    }

    // Write output
    float inv_sum = 1.0f / (global_sum + 1e-30f);
    for (int d = lane; d < HEAD_DIM; d += 32) {
        O[h * HEAD_DIM + d] = global_acc[d] * inv_sum;
    }
}

}  // namespace cuda
}  // namespace forge
