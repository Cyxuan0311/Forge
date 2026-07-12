#include <cmath>

#include "cuda_activation.h"

namespace forge {
namespace cuda {

__global__ void silu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = v / (1.0f + expf(-v));
    }
}

void launch_silu(const float* x, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, stream>>>(x, out, n);
}

void launch_silu_fp16(const void* x, void* out, int n, cudaStream_t stream) {
    launch_silu(static_cast<const float*>(x), static_cast<float*>(out), n, stream);
}

__global__ void gelu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float cdf = 0.5f * (1.0f + erff(v * 0.7071067811865475f));
        out[idx] = v * cdf;
    }
}

void launch_gelu(const float* x, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(x, out, n);
}

__global__ void gelu_tanh_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        out[idx] = v * cdf;
    }
}

void launch_gelu_tanh(const float* x, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_tanh_kernel<<<blocks, threads, 0, stream>>>(x, out, n);
}

__global__ void tanh_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(x[idx]);
    }
}

void launch_tanh(const float* x, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    tanh_kernel<<<blocks, threads, 0, stream>>>(x, out, n);
}

}  // namespace cuda
}  // namespace forge
