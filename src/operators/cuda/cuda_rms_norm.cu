#include <algorithm>
#include <cmath>

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

void launch_rms_norm_fp16(const void* x, const void* weight, void* out, int rows, int cols,
                          float eps, cudaStream_t stream) {
    launch_rms_norm(static_cast<const float*>(x), static_cast<const float*>(weight),
                    static_cast<float*>(out), rows, cols, eps, stream);
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

}  // namespace cuda
}  // namespace forge
