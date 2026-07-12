#include "forge/cuda_kernels.h"
#include "cuda_common.h"
#include <cmath>

namespace forge {
namespace cuda {

// ---- RMS Norm ----

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

// ---- SiLU ----

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

// ---- GELU ----

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

// ---- GELU Tanh ----

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

// ---- Add Bias ----

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

// ---- Element-wise Multiply ----

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

// ---- SiLU * Up (fused for FFN) ----

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

} // namespace cuda
} // namespace forge
