#include <cmath>

#include "cuda_elementwise.h"

namespace forge {
namespace cuda {

__global__ void add_bias_kernel(const float* data, const float* bias, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = data[idx] + bias[idx];
    }
}

void launch_add_bias(const float* data, const float* bias, float* out, int n, cudaStream_t stream) {
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

void launch_multiply(const float* a, const float* b, float* out, int n, cudaStream_t stream) {
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

void launch_silu_multiply(const float* gate, const float* up, float* out, int n,
                          cudaStream_t stream) {
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

void launch_gelu_multiply(const float* gate, const float* up, float* out, int n,
                          cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_multiply_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

__global__ void scale_kernel(float* data, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= s;
    }
}

void launch_scale(float* data, float s, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads, 0, stream>>>(data, s, n);
}

// GELU(tanh) activation + element-wise multiply with per-layer embedding slice
// x[i] = gelu_tanh(x[i]) * y[offset + i] for each token row
__global__ void gelu_tanh_multiply_kernel(
    float* __restrict__ x,
    const float* __restrict__ y,
    int n_per,          // elements per token row in x
    int n_layer,        // number of layers
    int layer_idx,      // which layer's embedding slice to use
    int seq_len,        // number of token rows
    int n_total)        // total elements in x (seq_len * n_per)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;

    int s = idx / n_per;     // which token row
    int d = idx % n_per;     // position within the row

    float v = x[idx];
    float gelu = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    x[idx] = gelu * y[s * n_per * n_layer + layer_idx * n_per + d];
}

void launch_gelu_tanh_multiply(float* x, const float* y, int n_per, int n_layer,
                               int layer_idx, int seq_len, cudaStream_t stream) {
    int n_total = seq_len * n_per;
    int threads = 256;
    int blocks = (n_total + threads - 1) / threads;
    gelu_tanh_multiply_kernel<<<blocks, threads, 0, stream>>>(
        x, y, n_per, n_layer, layer_idx, seq_len, n_total);
}

}  // namespace cuda
}  // namespace forge
