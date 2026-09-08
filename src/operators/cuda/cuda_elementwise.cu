#include <cmath>
#include <cstdio>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cuda_elementwise.h"
#include "forge/fp8_utils.h"

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

// Phase 6-A-4: real native FP16 gated activations (FP16 storage, FP32 compute of
// silu/gelu, FP16 output). gate reuses the SILU / GELU dtype-contract gate.
__global__ void silu_multiply_kernel_fp16(const __half* gate, const __half* up, __half* out,
                                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __half2float(gate[idx]);
        float silu_v = v / (1.0f + expf(-v));
        out[idx] = __float2half(silu_v * __half2float(up[idx]));
    }
}

void launch_silu_multiply_fp16(const __half* gate, const __half* up, __half* out, int n,
                               cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_multiply_kernel_fp16<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

__global__ void gelu_multiply_kernel_fp16(const __half* gate, const __half* up, __half* out,
                                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __half2float(gate[idx]);
        float gelu_v = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
        out[idx] = __float2half(gelu_v * __half2float(up[idx]));
    }
}

void launch_gelu_multiply_fp16(const __half* gate, const __half* up, __half* out, int n,
                               cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_multiply_kernel_fp16<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

// Phase 6-B-3: real native BF16 gated activations (BF16 storage, FP32 compute of
// silu/gelu, BF16 output). gate reuses the SILU / GELU dtype-contract gate.
__global__ void silu_multiply_kernel_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                          __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(gate[idx]);
        float silu_v = v / (1.0f + expf(-v));
        out[idx] = __float2bfloat16(silu_v * __bfloat162float(up[idx]));
    }
}

void launch_silu_multiply_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                               __nv_bfloat16* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_multiply_kernel_bf16<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

__global__ void gelu_multiply_kernel_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                          __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(gate[idx]);
        float gelu_v = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
        out[idx] = __float2bfloat16(gelu_v * __bfloat162float(up[idx]));
    }
}

void launch_gelu_multiply_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                               __nv_bfloat16* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_multiply_kernel_bf16<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

// Phase 6-A-8.0: native FP16 elementwise ADD / MUL (FP16 storage, FP32 compute
// of each element, FP16 output) -- mirrors the 6-A-4 gated-activation policy.
__global__ void add_fp16_kernel(const __half* a, const __half* b, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

void launch_add_fp16(const __half* a, const __half* b, __half* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_fp16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

__global__ void multiply_fp16_kernel(const __half* a, const __half* b, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(a[idx]) * __half2float(b[idx]));
    }
}

void launch_multiply_fp16(const __half* a, const __half* b, __half* out, int n,
                          cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    multiply_fp16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

// Phase 6-B-5: native BF16 elementwise ADD / MUL (BF16 storage, FP32 compute
// of each element, BF16 output) -- mirrors the 6-A-8.0 FP16 policy. These are
// the residual-add and gate*up/scale nodes of BF16 graphs; ADD / MUL retire
// from require_no_bf16 with this step.
__global__ void add_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

void launch_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int n,
                     cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

__global__ void multiply_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                     __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) * __bfloat162float(b[idx]));
    }
}

void launch_multiply_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int n,
                           cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    multiply_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
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

__global__ void scale_accumulate_kernel(const float* src, float* dst, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += s * src[idx];
    }
}

void launch_scale_accumulate(const float* src, float* dst, float s, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_accumulate_kernel<<<blocks, threads, 0, stream>>>(src, dst, s, n);
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

__global__ void split_q_gate_kernel(const float* q_full, float* q, float* gate,
                                     int seq_len, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h = (idx / head_dim) % num_heads;
    int s = idx / (num_heads * head_dim);

    int src_idx = s * num_heads * head_dim * 2 + h * head_dim * 2 + d;
    q[idx] = q_full[src_idx];
    gate[idx] = q_full[src_idx + head_dim];
}

void launch_split_q_gate(const float* q_full, float* q, float* gate,
                         int seq_len, int num_heads, int head_dim,
                         cudaStream_t stream) {
    int n = seq_len * num_heads * head_dim;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    split_q_gate_kernel<<<blocks, threads, 0, stream>>>(q_full, q, gate, seq_len, num_heads, head_dim);
}

__global__ void sigmoid_multiply_kernel(const float* gate, float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 1.0f / (1.0f + expf(-gate[idx]));
    }
}

void launch_sigmoid_multiply(const float* gate, float* data, int n,
                             cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_multiply_kernel<<<blocks, threads, 0, stream>>>(gate, data, n);
}

// Phase 6-C-3: native FP8 E4M3 gated activations (fused single-kernel SiLU/GeGLU).
// Storage FP8, FP32 compute of silu/gelu, FP8 output -- mirrors the 6-A-4 /
// 6-B-3 gated policy and reuses the 6-C-2-perf vec16 FP8 codec. Helper names
// carry `_act` to avoid an ODR clash with cuda_rms_norm.cu's same-named inline
// codec helpers in this namespace.
__host__ __device__ inline float fp8_hw_load_byte_act(uint8_t b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(hr));
}

__host__ __device__ inline uint8_t fp8_hw_store_byte_act(float x) {
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

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

__device__ inline float silu_mul_f(float g, float u) {
    return (g / (1.0f + expf(-g))) * u;
}

__device__ inline float gelu_mul_f(float g, float u) {
    return (0.5f * g * (1.0f + erff(g * 0.7071067811865475f))) * u;
}

// ---- scalar gated FP8 ----
__global__ void silu_multiply_kernel_fp8(const uint8_t* gate, const uint8_t* up, uint8_t* out,
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = fp8_hw_load_byte_act(gate[idx]);
        float sv = v / (1.0f + expf(-v));
        out[idx] = fp8_hw_store_byte_act(sv * fp8_hw_load_byte_act(up[idx]));
    }
}

__global__ void gelu_multiply_kernel_fp8(const uint8_t* gate, const uint8_t* up, uint8_t* out,
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = fp8_hw_load_byte_act(gate[idx]);
        float gv = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
        out[idx] = fp8_hw_store_byte_act(gv * fp8_hw_load_byte_act(up[idx]));
    }
}

// ---- vec16 gated FP8 (128-bit load/store of gate+up, paired hw cvt) ----
// Per-element results are bit-identical to the scalar kernels. ngroups = n / 16;
// the launcher runs the scalar kernel over the [ngroups*16, n) tail.
__global__ void silu_multiply_kernel_fp8_vec16(const uint8_t* gate, const uint8_t* up,
                                                uint8_t* out, int ngroups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups)
        return;
    uint4 gv = reinterpret_cast<const uint4*>(gate)[g];
    uint4 uv = reinterpret_cast<const uint4*>(up)[g];
    float4 g0 = fp8x4_to_float4_act(gv.x), g1 = fp8x4_to_float4_act(gv.y),
            g2 = fp8x4_to_float4_act(gv.z), g3 = fp8x4_to_float4_act(gv.w);
    float4 u0 = fp8x4_to_float4_act(uv.x), u1 = fp8x4_to_float4_act(uv.y),
            u2 = fp8x4_to_float4_act(uv.z), u3 = fp8x4_to_float4_act(uv.w);
    float4 o0 = make_float4(silu_mul_f(g0.x, u0.x), silu_mul_f(g0.y, u0.y), silu_mul_f(g0.z, u0.z),
                            silu_mul_f(g0.w, u0.w));
    float4 o1 = make_float4(silu_mul_f(g1.x, u1.x), silu_mul_f(g1.y, u1.y), silu_mul_f(g1.z, u1.z),
                            silu_mul_f(g1.w, u1.w));
    float4 o2 = make_float4(silu_mul_f(g2.x, u2.x), silu_mul_f(g2.y, u2.y), silu_mul_f(g2.z, u2.z),
                            silu_mul_f(g2.w, u2.w));
    float4 o3 = make_float4(silu_mul_f(g3.x, u3.x), silu_mul_f(g3.y, u3.y), silu_mul_f(g3.z, u3.z),
                            silu_mul_f(g3.w, u3.w));
    uint4 ov;
    ov.x = float4_to_fp8x4_act(o0);
    ov.y = float4_to_fp8x4_act(o1);
    ov.z = float4_to_fp8x4_act(o2);
    ov.w = float4_to_fp8x4_act(o3);
    reinterpret_cast<uint4*>(out)[g] = ov;
}

__global__ void gelu_multiply_kernel_fp8_vec16(const uint8_t* gate, const uint8_t* up,
                                                uint8_t* out, int ngroups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups)
        return;
    uint4 gv = reinterpret_cast<const uint4*>(gate)[g];
    uint4 uv = reinterpret_cast<const uint4*>(up)[g];
    float4 g0 = fp8x4_to_float4_act(gv.x), g1 = fp8x4_to_float4_act(gv.y),
            g2 = fp8x4_to_float4_act(gv.z), g3 = fp8x4_to_float4_act(gv.w);
    float4 u0 = fp8x4_to_float4_act(uv.x), u1 = fp8x4_to_float4_act(uv.y),
            u2 = fp8x4_to_float4_act(uv.z), u3 = fp8x4_to_float4_act(uv.w);
    float4 o0 = make_float4(gelu_mul_f(g0.x, u0.x), gelu_mul_f(g0.y, u0.y), gelu_mul_f(g0.z, u0.z),
                            gelu_mul_f(g0.w, u0.w));
    float4 o1 = make_float4(gelu_mul_f(g1.x, u1.x), gelu_mul_f(g1.y, u1.y), gelu_mul_f(g1.z, u1.z),
                            gelu_mul_f(g1.w, u1.w));
    float4 o2 = make_float4(gelu_mul_f(g2.x, u2.x), gelu_mul_f(g2.y, u2.y), gelu_mul_f(g2.z, u2.z),
                            gelu_mul_f(g2.w, u2.w));
    float4 o3 = make_float4(gelu_mul_f(g3.x, u3.x), gelu_mul_f(g3.y, u3.y), gelu_mul_f(g3.z, u3.z),
                            gelu_mul_f(g3.w, u3.w));
    uint4 ov;
    ov.x = float4_to_fp8x4_act(o0);
    ov.y = float4_to_fp8x4_act(o1);
    ov.z = float4_to_fp8x4_act(o2);
    ov.w = float4_to_fp8x4_act(o3);
    reinterpret_cast<uint4*>(out)[g] = ov;
}

void launch_silu_multiply_fp8(const uint8_t* gate, const uint8_t* up, uint8_t* out, int n,
                              cudaStream_t stream) {
    int threads = 256;
    int ngroups = n / 16;
    if (ngroups > 0) {
        int blocks = (ngroups + threads - 1) / threads;
        silu_multiply_kernel_fp8_vec16<<<blocks, threads, 0, stream>>>(gate, up, out, ngroups);
    }
    int tail = n - ngroups * 16;
    if (tail > 0) {
        int blocks = (tail + threads - 1) / threads;
        silu_multiply_kernel_fp8<<<blocks, threads, 0, stream>>>(
            gate + ngroups * 16, up + ngroups * 16, out + ngroups * 16, tail);
    }
}

void launch_gelu_multiply_fp8(const uint8_t* gate, const uint8_t* up, uint8_t* out, int n,
                              cudaStream_t stream) {
    int threads = 256;
    int ngroups = n / 16;
    if (ngroups > 0) {
        int blocks = (ngroups + threads - 1) / threads;
        gelu_multiply_kernel_fp8_vec16<<<blocks, threads, 0, stream>>>(gate, up, out, ngroups);
    }
    int tail = n - ngroups * 16;
    if (tail > 0) {
        int blocks = (tail + threads - 1) / threads;
        gelu_multiply_kernel_fp8<<<blocks, threads, 0, stream>>>(
            gate + ngroups * 16, up + ngroups * 16, out + ngroups * 16, tail);
    }
}

}  // namespace cuda
}  // namespace forge
