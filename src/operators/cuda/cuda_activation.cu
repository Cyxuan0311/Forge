#include <cmath>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cuda_activation.h"
#include "forge/fp8_utils.h"

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

// Phase 6-B-3: real native BF16 SiLU. Mirrors the 6-A-3 FP16 kernel: BF16
// storage decoupled from the FP32 transcendental compute, BF16 output.
__global__ void silu_kernel_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(x[idx]);
        out[idx] = __float2bfloat16(v / (1.0f + expf(-v)));
    }
}

void launch_silu_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_kernel_bf16<<<blocks, threads, 0, stream>>>(x, out, n);
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

// Phase 6-B-3: real native BF16 GELU / GELU-tanh. Storage BF16, compute FP32
// (erf / tanh in float), output BF16 -- same policy as 6-A-4.
__global__ void gelu_kernel_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(x[idx]);
        float cdf = 0.5f * (1.0f + erff(v * 0.7071067811865475f));
        out[idx] = __float2bfloat16(v * cdf);
    }
}

void launch_gelu_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel_bf16<<<blocks, threads, 0, stream>>>(x, out, n);
}

__global__ void gelu_tanh_kernel_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(x[idx]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        out[idx] = __float2bfloat16(v * cdf);
    }
}

void launch_gelu_tanh_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_tanh_kernel_bf16<<<blocks, threads, 0, stream>>>(x, out, n);
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

// Phase 6-C-3: native FP8 E4M3 activation family. Mirrors the 6-A-3/6-A-4 /
// 6-B-3 unary policy (FP8 storage + FP32 transcendental compute + FP8 output)
// and reuses the 6-C-2 / 6-C-2-perf hardware FP8 codec + vec16 vectorization.
// The helpers below are named with an `_act` suffix on purpose: cuda_rms_norm.cu
// already defines the same-named inline codec helpers in this namespace, and two
// non-inline definitions of the same symbol would be an ODR violation at link
// time. (They are functionally identical to the RMSNorm ones.)
__host__ __device__ inline float fp8_hw_load_byte_act(uint8_t b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(hr));
}

__host__ __device__ inline uint8_t fp8_hw_store_byte_act(float x) {
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

// Paired sm_89 FP8 codecs used by the vectorized kernels (mirrors
// cuda_rms_norm.cu's vec16 helpers): fp8 -> f16 is exact, so the decode is
// bit-identical to fp8_hw_load_byte_act per element; the encode applies the same
// per-element single-step RN (SATFINITE), also bit-identical.
__device__ inline float4 fp8x4_to_float4_act(uint32_t packed) {
    __half2_raw h01 =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed & 0xFFFFu), __NV_E4M3);
    __half2_raw h23 =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed >> 16), __NV_E4M3);
    float2 f01 = __half22float2(__half2(h01));
    float2 f23 = __half22float2(__half2(h23));
    return make_float4(f01.x, f01.y, f23.x, f23.y);
}

// Encode via four independent scalar hardware cvt stores. (The paired
// __nv_cvt_float2_to_fp8x2 into a uint32 packs the HI half unreliably across
// optimization levels, so we build the uint32 from four single-step RN bytes --
// bit-identical to the scalar fp8_hw_store_byte_act reference path.)
__device__ inline uint32_t float4_to_fp8x4_act(float4 f) {
    uint32_t b0 = static_cast<uint32_t>(fp8_hw_store_byte_act(f.x));
    uint32_t b1 = static_cast<uint32_t>(fp8_hw_store_byte_act(f.y));
    uint32_t b2 = static_cast<uint32_t>(fp8_hw_store_byte_act(f.z));
    uint32_t b3 = static_cast<uint32_t>(fp8_hw_store_byte_act(f.w));
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__device__ inline float4 silu_f4(float4 v) {
    v.x /= (1.0f + expf(-v.x));
    v.y /= (1.0f + expf(-v.y));
    v.z /= (1.0f + expf(-v.z));
    v.w /= (1.0f + expf(-v.w));
    return v;
}

__device__ inline float4 gelu_f4(float4 v) {
    v.x *= 0.5f * (1.0f + erff(v.x * 0.7071067811865475f));
    v.y *= 0.5f * (1.0f + erff(v.y * 0.7071067811865475f));
    v.z *= 0.5f * (1.0f + erff(v.z * 0.7071067811865475f));
    v.w *= 0.5f * (1.0f + erff(v.w * 0.7071067811865475f));
    return v;
}

__device__ inline float4 gelu_tanh_f4(float4 v) {
    v.x = v.x * 0.5f * (1.0f + tanhf(0.7978845608028654f * (v.x + 0.044715f * v.x * v.x * v.x)));
    v.y = v.y * 0.5f * (1.0f + tanhf(0.7978845608028654f * (v.y + 0.044715f * v.y * v.y * v.y)));
    v.z = v.z * 0.5f * (1.0f + tanhf(0.7978845608028654f * (v.z + 0.044715f * v.z * v.z * v.z)));
    v.w = v.w * 0.5f * (1.0f + tanhf(0.7978845608028654f * (v.w + 0.044715f * v.w * v.w * v.w)));
    return v;
}

// ---- scalar unary FP8 (exact-match decode/encode to the vec16 path) ----
__global__ void silu_kernel_fp8(const uint8_t* x, uint8_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = fp8_hw_load_byte_act(x[idx]);
        out[idx] = fp8_hw_store_byte_act(v / (1.0f + expf(-v)));
    }
}

__global__ void gelu_kernel_fp8(const uint8_t* x, uint8_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = fp8_hw_load_byte_act(x[idx]);
        out[idx] = fp8_hw_store_byte_act(0.5f * v * (1.0f + erff(v * 0.7071067811865475f)));
    }
}

__global__ void gelu_tanh_kernel_fp8(const uint8_t* x, uint8_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = fp8_hw_load_byte_act(x[idx]);
        out[idx] = fp8_hw_store_byte_act(
            v * 0.5f * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))));
    }
}

// ---- vec16 unary FP8 (128-bit load/store + paired hw cvt, mirrors 6-C-2-perf) ----
// Per-element results are bit-identical to the scalar kernels (same hw RN codec;
// only the FP32 transcendental order differs trivially). ngroups = n / 16; the
// launcher runs the scalar kernel over the [ngroups*16, n) tail.
__global__ void silu_kernel_fp8_vec16(const uint8_t* x, uint8_t* out, int ngroups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups)
        return;
    const uint4* xv = reinterpret_cast<const uint4*>(x) + g;
    float4 f0 = fp8x4_to_float4_act(xv->x);
    float4 f1 = fp8x4_to_float4_act(xv->y);
    float4 f2 = fp8x4_to_float4_act(xv->z);
    float4 f3 = fp8x4_to_float4_act(xv->w);
    uint4 ov;
    ov.x = float4_to_fp8x4_act(silu_f4(f0));
    ov.y = float4_to_fp8x4_act(silu_f4(f1));
    ov.z = float4_to_fp8x4_act(silu_f4(f2));
    ov.w = float4_to_fp8x4_act(silu_f4(f3));
    reinterpret_cast<uint4*>(out)[g] = ov;
}

__global__ void gelu_kernel_fp8_vec16(const uint8_t* x, uint8_t* out, int ngroups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups)
        return;
    const uint4* xv = reinterpret_cast<const uint4*>(x) + g;
    float4 f0 = fp8x4_to_float4_act(xv->x);
    float4 f1 = fp8x4_to_float4_act(xv->y);
    float4 f2 = fp8x4_to_float4_act(xv->z);
    float4 f3 = fp8x4_to_float4_act(xv->w);
    uint4 ov;
    ov.x = float4_to_fp8x4_act(gelu_f4(f0));
    ov.y = float4_to_fp8x4_act(gelu_f4(f1));
    ov.z = float4_to_fp8x4_act(gelu_f4(f2));
    ov.w = float4_to_fp8x4_act(gelu_f4(f3));
    reinterpret_cast<uint4*>(out)[g] = ov;
}

__global__ void gelu_tanh_kernel_fp8_vec16(const uint8_t* x, uint8_t* out, int ngroups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups)
        return;
    const uint4* xv = reinterpret_cast<const uint4*>(x) + g;
    float4 f0 = fp8x4_to_float4_act(xv->x);
    float4 f1 = fp8x4_to_float4_act(xv->y);
    float4 f2 = fp8x4_to_float4_act(xv->z);
    float4 f3 = fp8x4_to_float4_act(xv->w);
    uint4 ov;
    ov.x = float4_to_fp8x4_act(gelu_tanh_f4(f0));
    ov.y = float4_to_fp8x4_act(gelu_tanh_f4(f1));
    ov.z = float4_to_fp8x4_act(gelu_tanh_f4(f2));
    ov.w = float4_to_fp8x4_act(gelu_tanh_f4(f3));
    reinterpret_cast<uint4*>(out)[g] = ov;
}

void launch_silu_fp8(const uint8_t* x, uint8_t* out, int n, cudaStream_t stream) {
    int threads = 256;
    int ngroups = n / 16;
    if (ngroups > 0) {
        int blocks = (ngroups + threads - 1) / threads;
        silu_kernel_fp8_vec16<<<blocks, threads, 0, stream>>>(x, out, ngroups);
    }
    int tail = n - ngroups * 16;
    if (tail > 0) {
        int blocks = (tail + threads - 1) / threads;
        silu_kernel_fp8<<<blocks, threads, 0, stream>>>(x + ngroups * 16, out + ngroups * 16, tail);
    }
}

void launch_gelu_fp8(const uint8_t* x, uint8_t* out, int n, cudaStream_t stream) {
    int threads = 256;
    int ngroups = n / 16;
    if (ngroups > 0) {
        int blocks = (ngroups + threads - 1) / threads;
        gelu_kernel_fp8_vec16<<<blocks, threads, 0, stream>>>(x, out, ngroups);
    }
    int tail = n - ngroups * 16;
    if (tail > 0) {
        int blocks = (tail + threads - 1) / threads;
        gelu_kernel_fp8<<<blocks, threads, 0, stream>>>(x + ngroups * 16, out + ngroups * 16, tail);
    }
}

void launch_gelu_tanh_fp8(const uint8_t* x, uint8_t* out, int n, cudaStream_t stream) {
    int threads = 256;
    int ngroups = n / 16;
    if (ngroups > 0) {
        int blocks = (ngroups + threads - 1) / threads;
        gelu_tanh_kernel_fp8_vec16<<<blocks, threads, 0, stream>>>(x, out, ngroups);
    }
    int tail = n - ngroups * 16;
    if (tail > 0) {
        int blocks = (tail + threads - 1) / threads;
        gelu_tanh_kernel_fp8<<<blocks, threads, 0, stream>>>(x + ngroups * 16, out + ngroups * 16,
                                                              tail);
    }
}

}  // namespace cuda
}  // namespace forge
