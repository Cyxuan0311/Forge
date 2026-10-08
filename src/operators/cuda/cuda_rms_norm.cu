#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cuda_common.h"
#include "cuda_rms_norm.h"
#include "forge/fp8_utils.h"

namespace forge {
namespace cuda {

// 6-C-2: hardware FP8 E4M3 codec (cuda_fp8.h). On sm_89 these compile to the
// FP8 cvt instructions; on pre-89 archs cuda_fp8.h falls back to software --
// both paths stay bit-consistent with the forge E4M3 codec for |x| <= 240
// (6-C-1 contract S6: forge never emits exp=1111 patterns; the amax/240 scale
// convention keeps all stored values inside [-240, 240]). The only divergences
// are unreachable under the contract: values in (240, 448] and the 0x7F NaN
// encoding. The tie-break differs (hardware RNE vs forge round-half-away), but
// both pick a nearest bucket, so the relative error stays within one half
// step (1/16). The forge software codec remains the boundary/reference codec
// (host paths, tests) -- it must never appear in a kernel hot loop.
__host__ __device__ inline float fp8_hw_load_byte(uint8_t b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);  // exact (fp8 in half)
    return __half2float(__half(hr));
}

__host__ __device__ inline uint8_t fp8_hw_store_byte(float x) {
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);  // single-step RN
}

// 6-C-2.7: paired sm_89 FP8 codecs used by the vectorized RMSNorm kernels.
// Decode: cvt.rn.f16x2.e4m3x2 (fp8 -> f16 is exact, so this is bit-identical to
// fp8_hw_load_byte per element). Encode: cvt.rn.satfinite.e4m3x2.f32 applies
// the same per-element single-step RN as fp8_hw_store_byte -- also bit-identical.
// Byte order: the LOW byte of a packed pair is the element at the lower address.
__device__ inline float4 fp8x4_to_float4(uint32_t packed) {
    __half2_raw h01 =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed & 0xFFFFu), __NV_E4M3);
    __half2_raw h23 =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed >> 16), __NV_E4M3);
    float2 f01 = __half22float2(__half2(h01));
    float2 f23 = __half22float2(__half2(h23));
    return make_float4(f01.x, f01.y, f23.x, f23.y);
}

__device__ inline uint32_t float4_to_fp8x4(float4 f) {
    __nv_fp8x2_storage_t lo =
        __nv_cvt_float2_to_fp8x2(make_float2(f.x, f.y), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8x2_storage_t hi =
        __nv_cvt_float2_to_fp8x2(make_float2(f.z, f.w), __NV_SATFINITE, __NV_E4M3);
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}


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

// Phase 6-B-2: real native BF16 RMSNorm. Mirrors the 6-A-2 FP16 kernel line by
// line: BF16 storage/arithmetic with an FP32 reduction (square-sum -> rsqrt ->
// scale) so numerical stability is preserved. This is a native BF16 *path*,
// not "all intermediates in bfloat16".
__global__ void rms_norm_kernel_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                     __nv_bfloat16* out, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const __nv_bfloat16* x_row = x + row * cols;
    __nv_bfloat16* out_row = out + row * cols;

    // FP32 reduction of squares (BF16 inputs widened to float first).
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __bfloat162float(x_row[i]);
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
        float v = __bfloat162float(x_row[i]);
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16(v * rms * w);
    }
}

void launch_rms_norm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out,
                           int rows, int cols, float eps, cudaStream_t stream) {
    int threads = std::min(cols, 1024);
    rms_norm_kernel_bf16<<<rows, threads, 0, stream>>>(x, weight, out, rows, cols, eps);
}

// Unweighted BF16 RMSNorm (6-B-2): BF16 storage/arithmetic with an FP32 reduction
// (out = x * rsqrt(mean(x^2) + eps)). Native BF16 path, not "all-bfloat16".
__global__ void rms_norm_unweighted_kernel_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int rows,
                                                int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const __nv_bfloat16* x_row = x + row * cols;
    __nv_bfloat16* out_row = out + row * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __bfloat162float(x_row[i]);
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
        out_row[i] = __float2bfloat16(__bfloat162float(x_row[i]) * rms);
    }
}

void launch_rms_norm_unweighted_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int rows, int cols,
                                     float eps, cudaStream_t stream) {
    int threads = std::min(cols, 1024);
    rms_norm_unweighted_kernel_bf16<<<rows, threads, 0, stream>>>(x, out, rows, cols, eps);
}

// Weighted FP8 E4M3 RMSNorm (6-C-2): FP8 storage with an FP32 reduction/math
// (out = x * rsqrt(mean(x^2) + eps) * w). Hardware codec (fp8_hw_* -- see the
// S6 bit-consistency note above); forge software codec stays on the
// host/boundary side. Weight arrives as FP8 and is decoded per element; no
// FP8 arithmetic happens anywhere (6-A discipline: storage precision !=
// reduction/compute precision).
__global__ void rms_norm_kernel_fp8(const uint8_t* x, const uint8_t* weight, uint8_t* out, int rows,
                                    int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const uint8_t* x_row = x + static_cast<size_t>(row) * cols;
    uint8_t* out_row = out + static_cast<size_t>(row) * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = fp8_hw_load_byte(x_row[i]);
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
        float v = fp8_hw_load_byte(x_row[i]);
        float w = fp8_hw_load_byte(weight[i]);
        out_row[i] = fp8_hw_store_byte(v * rms * w);
    }
}

// 6-C-2.7: vectorized weighted FP8 RMSNorm. One uint4 (16 fp8 bytes) per lane
// step: LDG.128 + 8 paired hw cvt for decode, FP32 square-sum/math, 4 paired
// single-step RN cvt + STG.128 for the store. Per-element results are
// bit-identical to rms_norm_kernel_fp8 (same hw RN codec); only the FP32
// accumulation order differs (16 consecutive elements per lane). The launcher
// guarantees cols % 16 == 0 and 16B alignment of all three pointers.
__global__ void rms_norm_kernel_fp8_vec16(const uint8_t* x, const uint8_t* weight, uint8_t* out,
                                          int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const int vecs = cols / 16;
    const uint4* x_vec = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * cols);
    uint4* out_vec = reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * cols);
    const uint4* w_vec = reinterpret_cast<const uint4*>(weight);

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
        uint4 xv = x_vec[i];
        float4 f0 = fp8x4_to_float4(xv.x);
        float4 f1 = fp8x4_to_float4(xv.y);
        float4 f2 = fp8x4_to_float4(xv.z);
        float4 f3 = fp8x4_to_float4(xv.w);
        sum_sq += f0.x * f0.x + f0.y * f0.y + f0.z * f0.z + f0.w * f0.w +
                  f1.x * f1.x + f1.y * f1.y + f1.z * f1.z + f1.w * f1.w +
                  f2.x * f2.x + f2.y * f2.y + f2.z * f2.z + f2.w * f2.w +
                  f3.x * f3.x + f3.y * f3.y + f3.z * f3.z + f3.w * f3.w;
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
    for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
        uint4 xv = x_vec[i];
        uint4 wv = w_vec[i];
        float4 x0 = fp8x4_to_float4(xv.x);
        float4 x1 = fp8x4_to_float4(xv.y);
        float4 x2 = fp8x4_to_float4(xv.z);
        float4 x3 = fp8x4_to_float4(xv.w);
        float4 w0 = fp8x4_to_float4(wv.x);
        float4 w1 = fp8x4_to_float4(wv.y);
        float4 w2 = fp8x4_to_float4(wv.z);
        float4 w3 = fp8x4_to_float4(wv.w);
        uint4 ov;
        ov.x = float4_to_fp8x4(make_float4(x0.x * rms * w0.x, x0.y * rms * w0.y,
                                           x0.z * rms * w0.z, x0.w * rms * w0.w));
        ov.y = float4_to_fp8x4(make_float4(x1.x * rms * w1.x, x1.y * rms * w1.y,
                                           x1.z * rms * w1.z, x1.w * rms * w1.w));
        ov.z = float4_to_fp8x4(make_float4(x2.x * rms * w2.x, x2.y * rms * w2.y,
                                           x2.z * rms * w2.z, x2.w * rms * w2.w));
        ov.w = float4_to_fp8x4(make_float4(x3.x * rms * w3.x, x3.y * rms * w3.y,
                                           x3.z * rms * w3.z, x3.w * rms * w3.w));
        out_vec[i] = ov;
    }
}

void launch_rms_norm_fp8(const uint8_t* x, const uint8_t* weight, uint8_t* out, int rows, int cols,
                         float eps, cudaStream_t stream) {
    // 6-C-2.7: dispatch to the vectorized kernel only when every row start
    // keeps 16B alignment (cols % 16 == 0 plus aligned bases); otherwise the
    // scalar kernel stays the fallback for arbitrary layouts.
    bool vec_ok = (cols % 16 == 0) && (reinterpret_cast<uintptr_t>(x) % 16 == 0) &&
                  (reinterpret_cast<uintptr_t>(out) % 16 == 0) &&
                  (reinterpret_cast<uintptr_t>(weight) % 16 == 0);
    if (vec_ok) {
        int vecs = cols / 16;
        int threads = std::min((vecs + 31) & ~31, 1024);
        rms_norm_kernel_fp8_vec16<<<rows, threads, 0, stream>>>(x, weight, out, rows, cols, eps);
    } else {
        int threads = std::min(cols, 1024);
        rms_norm_kernel_fp8<<<rows, threads, 0, stream>>>(x, weight, out, rows, cols, eps);
    }
}

// Unweighted FP8 RMSNorm (6-C-2): out = x * rsqrt(mean(x^2) + eps), same FP32
// reduction, FP8 store. Dedicated kernel (no host ones vector may reach the
// device -- the 6-A-2 lesson).
__global__ void rms_norm_unweighted_kernel_fp8(const uint8_t* x, uint8_t* out, int rows, int cols,
                                               float eps) {
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const uint8_t* x_row = x + static_cast<size_t>(row) * cols;
    uint8_t* out_row = out + static_cast<size_t>(row) * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = fp8_hw_load_byte(x_row[i]);
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
        float v = fp8_hw_load_byte(x_row[i]);
        out_row[i] = fp8_hw_store_byte(v * rms);
    }
}

// 6-C-2.7: vectorized unweighted FP8 RMSNorm (mirrors rms_norm_kernel_fp8_vec16
// without the weight decode).
__global__ void rms_norm_unweighted_kernel_fp8_vec16(const uint8_t* x, uint8_t* out, int rows,
                                                     int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const int vecs = cols / 16;
    const uint4* x_vec = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * cols);
    uint4* out_vec = reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * cols);

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
        uint4 xv = x_vec[i];
        float4 f0 = fp8x4_to_float4(xv.x);
        float4 f1 = fp8x4_to_float4(xv.y);
        float4 f2 = fp8x4_to_float4(xv.z);
        float4 f3 = fp8x4_to_float4(xv.w);
        sum_sq += f0.x * f0.x + f0.y * f0.y + f0.z * f0.z + f0.w * f0.w +
                  f1.x * f1.x + f1.y * f1.y + f1.z * f1.z + f1.w * f1.w +
                  f2.x * f2.x + f2.y * f2.y + f2.z * f2.z + f2.w * f2.w +
                  f3.x * f3.x + f3.y * f3.y + f3.z * f3.z + f3.w * f3.w;
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
    for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
        uint4 xv = x_vec[i];
        float4 x0 = fp8x4_to_float4(xv.x);
        float4 x1 = fp8x4_to_float4(xv.y);
        float4 x2 = fp8x4_to_float4(xv.z);
        float4 x3 = fp8x4_to_float4(xv.w);
        uint4 ov;
        ov.x = float4_to_fp8x4(make_float4(x0.x * rms, x0.y * rms, x0.z * rms, x0.w * rms));
        ov.y = float4_to_fp8x4(make_float4(x1.x * rms, x1.y * rms, x1.z * rms, x1.w * rms));
        ov.z = float4_to_fp8x4(make_float4(x2.x * rms, x2.y * rms, x2.z * rms, x2.w * rms));
        ov.w = float4_to_fp8x4(make_float4(x3.x * rms, x3.y * rms, x3.z * rms, x3.w * rms));
        out_vec[i] = ov;
    }
}

void launch_rms_norm_unweighted_fp8(const uint8_t* x, uint8_t* out, int rows, int cols, float eps,
                                    cudaStream_t stream) {
    bool vec_ok = (cols % 16 == 0) && (reinterpret_cast<uintptr_t>(x) % 16 == 0) &&
                  (reinterpret_cast<uintptr_t>(out) % 16 == 0);
    if (vec_ok) {
        int vecs = cols / 16;
        int threads = std::min((vecs + 31) & ~31, 1024);
        rms_norm_unweighted_kernel_fp8_vec16<<<rows, threads, 0, stream>>>(x, out, rows, cols, eps);
    } else {
        int threads = std::min(cols, 1024);
        rms_norm_unweighted_kernel_fp8<<<rows, threads, 0, stream>>>(x, out, rows, cols, eps);
    }
}

// =========================================================================
// LayerNorm (FP32) -- two-pass mean/variance, one block per row.
// =========================================================================
// Arithmetic mirrors the CPU reference in operators/cpu/norm.cpp exactly:
// mean first, then the variance measured around that mean, then
// (x - mean) * rsqrt(var/cols + eps) * w + b. Only the summation order differs
// (tree reduction here vs a serial loop on the host), so results agree to FP32
// round-off rather than bit-exactly.
//
// This exists because ops::layer_norm had no device kernel at all: it copied x
// D2H, normalized on the host, and copied back. A vision tower calls it twice
// per block, so a 2304-token / 1024-wide tower paid ~900 MB of round-trip
// traffic per image for the two norms alone.
__device__ __forceinline__ float block_reduce_sum_ln(float val, float* s_partials) {
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    if (lane == 0)
        s_partials[warp_id] = val;
    __syncthreads();
    // First warp folds the per-warp partials. Every thread reads all of them so
    // the result is broadcast without a second __syncthreads.
    val = (threadIdx.x < num_warps) ? s_partials[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0)
            s_partials[0] = val;
    }
    __syncthreads();
    return s_partials[0];
}

__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                                  const float* __restrict__ bias, float* __restrict__ out, int rows,
                                  int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows)
        return;

    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* out_row = out + static_cast<size_t>(row) * cols;

    __shared__ float s_partials[32];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        sum += x_row[i];
    const float mean = block_reduce_sum_ln(sum, s_partials) / cols;
    __syncthreads();  // s_partials is reused by the variance pass

    float var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float d = x_row[i] - mean;
        var += d * d;
    }
    const float inv_std = rsqrtf(block_reduce_sum_ln(var, s_partials) / cols + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = (x_row[i] - mean) * inv_std;
        if (weight)
            v *= weight[i];
        if (bias)
            v += bias[i];
        out_row[i] = v;
    }
}

void launch_layer_norm(const float* x, const float* weight, const float* bias, float* out,
                       int rows, int cols, float eps, cudaStream_t stream) {
    // One block per row; 1024 threads covers the widest head dim the vision
    // towers use, and fewer threads just means each one strides further.
    const int threads = std::min(cols, 1024);
    layer_norm_kernel<<<rows, threads, 0, stream>>>(x, weight, bias, out, rows, cols, eps);
}

}  // namespace cuda
}  // namespace forge
