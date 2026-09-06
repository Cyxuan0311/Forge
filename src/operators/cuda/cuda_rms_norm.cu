#include <algorithm>
#include <cmath>

#include <cuda_fp16.h>

#include "cuda_common.h"
#include "cuda_rms_norm.h"

namespace forge {
namespace cuda {

__global__ void rms_norm_kernel(const float* x, const float* weight, float* out, int rows, int cols,
                                float eps) {
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum_sq += x_row[i] * x_row[i];
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_sums[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        s_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp reduces across all warps
    float final_sum = (threadIdx.x < blockDim.x / 32) ? s_sums[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
    }

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(final_sum / cols + eps);
    }
    __syncthreads();

    float rms = s_rms;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = x_row[i] * rms * weight[i];
    }
}

void launch_rms_norm(const float* x, const float* weight, float* out, int rows, int cols, float eps,
                     cudaStream_t stream) {
    int threads = std::min(cols, 1024);
    rms_norm_kernel<<<rows, threads, 0, stream>>>(x, weight, out, rows, cols, eps);
}

// Phase 6-A-2: real native FP16 RMSNorm. FP16 storage/arithmetic with an FP32
// reduction (square-sum -> rsqrt -> scale) so numerical stability is preserved.
// This is a native FP16 *path*, not "all intermediates in half".
__global__ void rms_norm_kernel_fp16(const __half* x, const __half* weight, __half* out, int rows,
                                     int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const __half* x_row = x + row * cols;
    __half* out_row = out + row * cols;

    // FP32 reduction of squares (FP16 inputs widened to float first).
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_sums[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        s_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    float final_sum = (threadIdx.x < blockDim.x / 32) ? s_sums[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
    }

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(final_sum / cols + eps);
    }
    __syncthreads();

    float rms = s_rms;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(v * rms * w);
    }
}

void launch_rms_norm_fp16(const __half* x, const __half* weight, __half* out, int rows, int cols,
                          float eps, cudaStream_t stream) {
    int threads = std::min(cols, 1024);
    rms_norm_kernel_fp16<<<rows, threads, 0, stream>>>(x, weight, out, rows, cols, eps);
}

// Unweighted RMSNorm: out = x * rsqrt(mean(x^2) + eps) — no learned weight
__global__ void rms_norm_unweighted_kernel(const float* x, float* out, int rows, int cols,
                                           float eps) {
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum_sq += x_row[i] * x_row[i];
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_sums[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        s_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp reduces across all warps
    float final_sum = (threadIdx.x < blockDim.x / 32) ? s_sums[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
    }

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(final_sum / cols + eps);
    }
    __syncthreads();

    float rms = s_rms;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = x_row[i] * rms;
    }
}

void launch_rms_norm_unweighted(const float* x, float* out, int rows, int cols, float eps,
                                cudaStream_t stream) {
    int threads = std::min(cols, 1024);
    rms_norm_unweighted_kernel<<<rows, threads, 0, stream>>>(x, out, rows, cols, eps);
}

// Unweighted FP16 RMSNorm (6-A-2): FP16 storage/arithmetic with an FP32 reduction
// (out = x * rsqrt(mean(x^2) + eps)). Native FP16 path, not "all-half".
__global__ void rms_norm_unweighted_kernel_fp16(const __half* x, __half* out, int rows, int cols,
                                                float eps) {
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const __half* x_row = x + row * cols;
    __half* out_row = out + row * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float s_sums[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        s_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    float final_sum = (threadIdx.x < blockDim.x / 32) ? s_sums[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
    }

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(final_sum / cols + eps);
    }
    __syncthreads();

    float rms = s_rms;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = __float2half(__half2float(x_row[i]) * rms);
    }
}

void launch_rms_norm_unweighted_fp16(const __half* x, __half* out, int rows, int cols, float eps,
                                     cudaStream_t stream) {
    int threads = std::min(cols, 1024);
    rms_norm_unweighted_kernel_fp16<<<rows, threads, 0, stream>>>(x, out, rows, cols, eps);
}

}  // namespace cuda
}  // namespace forge
