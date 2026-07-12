#include "cuda_rms_norm.h"
#include "cuda_common.h"
#include <algorithm>
#include <cmath>

namespace nanoinfer {
namespace cuda {

__global__ void rms_norm_kernel(const float* x, const float* weight, float* out,
                                 int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum_sq += x_row[i] * x_row[i];
    }

    __shared__ float s_sum;
    s_sum = 0.0f;
    __syncthreads();

    atomicAdd(&s_sum, sum_sq);
    __syncthreads();

    float rms = rsqrtf(s_sum / cols + eps);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = x_row[i] * rms * weight[i];
    }
}

void launch_rms_norm(const float* x, const float* weight, float* out,
                     int rows, int cols, float eps, cudaStream_t stream) {
    int threads = std::min(cols, 1024);
    rms_norm_kernel<<<rows, threads, 0, stream>>>(x, weight, out, rows, cols, eps);
}

void launch_rms_norm_fp16(const void* x, const void* weight, void* out,
                          int rows, int cols, float eps, cudaStream_t stream) {
    launch_rms_norm(static_cast<const float*>(x), static_cast<const float*>(weight),
                    static_cast<float*>(out), rows, cols, eps, stream);
}

} // namespace cuda
} // namespace nanoinfer
