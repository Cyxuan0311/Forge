#include "cuda_elementwise.h"
#include <cmath>

namespace nanoinfer {
namespace cuda {

__global__ void add_bias_kernel(const float* data, const float* bias, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = data[idx] + bias[idx];
    }
}

void launch_add_bias(const float* data, const float* bias, float* out,
                     int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(data, bias, out, n);
}

__global__ void multiply_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

void launch_multiply(const float* a, const float* b, float* out,
                     int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    multiply_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

__global__ void silu_multiply_kernel(const float* gate, const float* up, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = gate[idx];
        float silu_v = v / (1.0f + expf(-v));
        out[idx] = silu_v * up[idx];
    }
}

void launch_silu_multiply(const float* gate, const float* up, float* out,
                          int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_multiply_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

__global__ void gelu_multiply_kernel(const float* gate, const float* up, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = gate[idx];
        float gelu_v = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
        out[idx] = gelu_v * up[idx];
    }
}

void launch_gelu_multiply(const float* gate, const float* up, float* out,
                           int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_multiply_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

} // namespace cuda
} // namespace nanoinfer
