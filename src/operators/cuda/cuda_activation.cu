#include <cmath>

#include <cuda_fp16.h>

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

// Phase 6-A-3: real native FP16 SiLU. Compute model keeps storage precision
// (FP16) decoupled from the transcendental compute precision (FP32):
//     FP16 x -> float widen -> sigmoid(x)=1/(1+exp(-x)) -> x*sigmoid -> FP16 out
// so numerical behaviour matches the FP32 kernel up to FP16 output store
// rounding (exactly the 6-A-2 RMSNorm policy: low-precision storage, required
// FP32 compute).
__global__ void silu_kernel_fp16(const __half* x, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __half2float(x[idx]);
        out[idx] = __float2half(v / (1.0f + expf(-v)));
    }
}

void launch_silu_fp16(const __half* x, __half* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_kernel_fp16<<<blocks, threads, 0, stream>>>(x, out, n);
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

// Phase 6-A-4: real native FP16 GELU / GELU-tanh. Storage FP16, compute FP32
// (erf / tanh in float), output FP16 -- same policy as 6-A-2/6-A-3.
__global__ void gelu_kernel_fp16(const __half* x, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __half2float(x[idx]);
        float cdf = 0.5f * (1.0f + erff(v * 0.7071067811865475f));
        out[idx] = __float2half(v * cdf);
    }
}

void launch_gelu_fp16(const __half* x, __half* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel_fp16<<<blocks, threads, 0, stream>>>(x, out, n);
}

__global__ void gelu_tanh_kernel_fp16(const __half* x, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __half2float(x[idx]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        out[idx] = __float2half(v * cdf);
    }
}

void launch_gelu_tanh_fp16(const __half* x, __half* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_tanh_kernel_fp16<<<blocks, threads, 0, stream>>>(x, out, n);
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
