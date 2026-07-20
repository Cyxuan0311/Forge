#pragma once

// Templated GEMV kernels for quantized types.
// Inspired by llama.cpp's mmq.cuh — a single template kernel is instantiated
// for each quantized DataType via CudaQuantTraits<DT>::dot_block().
//
// Code reduction: 8 separate kernels × ~150 lines → 1 template + traits.

#include "cuda_common.h"
#include "forge/types.h"  // DataType enum + QuantTraits

namespace forge {
namespace cuda {

// ============================================================================
// CudaQuantTraits<DT> — per-type __device__ dot_block() for GEMV templating
// ============================================================================

template <DataType DT>
struct CudaQuantTraits;

// ---- Q4_0: 32 elements/block, 18 bytes/block ----
// Layout: scale(f16, 2B) + qs[16]
template <>
struct CudaQuantTraits<DataType::Q4_0> {
    static constexpr int block_elements = QuantTraits<DataType::Q4_0>::block_elements;  // 32
    static constexpr int block_size = QuantTraits<DataType::Q4_0>::block_size;           // 18

    static __device__ __forceinline__ float dot_block(
        const uint8_t* block_ptr, const float* x, int base, int K) {
        uint16_t scale_bits;
        memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
        float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
        const uint8_t* qs = block_ptr + sizeof(uint16_t);

        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < block_elements; ++j) {
            if (base + j >= K) break;
            sum += x[base + j] * (static_cast<float>(q4_unpack(qs, j)) * scale);
        }
        return sum;
    }
};

// ---- Q4_1: 32 elements/block, 20 bytes/block ----
// Layout: d(f16, 2B) + m(f16, 2B) + qs[16]
template <>
struct CudaQuantTraits<DataType::Q4_1> {
    static constexpr int block_elements = QuantTraits<DataType::Q4_1>::block_elements;  // 32
    static constexpr int block_size = QuantTraits<DataType::Q4_1>::block_size;           // 20

    static __device__ __forceinline__ float dot_block(
        const uint8_t* block_ptr, const float* x, int base, int K) {
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, block_ptr, sizeof(uint16_t));
        memcpy(&m_bits, block_ptr + 2, sizeof(uint16_t));
        float d_val = __half2float(reinterpret_cast<const __half&>(d_bits));
        float m_val = __half2float(reinterpret_cast<const __half&>(m_bits));
        const uint8_t* qs = block_ptr + 4;

        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < block_elements; ++j) {
            if (base + j >= K) break;
            sum += x[base + j] * (static_cast<float>(q4_unpack_unsigned(qs, j)) * d_val + m_val);
        }
        return sum;
    }
};

// ---- Q8_0: 32 elements/block, 34 bytes/block ----
// Layout: scale(f16, 2B) + qs[32] (int8)
template <>
struct CudaQuantTraits<DataType::Q8_0> {
    static constexpr int block_elements = QuantTraits<DataType::Q8_0>::block_elements;  // 32
    static constexpr int block_size = QuantTraits<DataType::Q8_0>::block_size;           // 34

    static __device__ __forceinline__ float dot_block(
        const uint8_t* block_ptr, const float* x, int base, int K) {
        uint16_t scale_bits;
        memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
        float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + 2);

        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < block_elements; ++j) {
            if (base + j >= K) break;
            sum += x[base + j] * (static_cast<float>(qs[j]) * scale);
        }
        return sum;
    }
};

// ---- Q4_K: 256 elements/block, 144 bytes/block ----
// Layout: d(f16, 2B) + dmin(f16, 2B) + scales[12] + qs[128]
template <>
struct CudaQuantTraits<DataType::Q4_K> {
    static constexpr int block_elements = QuantTraits<DataType::Q4_K>::block_elements;  // 256
    static constexpr int block_size = QuantTraits<DataType::Q4_K>::block_size;           // 144

    static __device__ __forceinline__ float dot_block(
        const uint8_t* block_ptr, const float* x, int base, int K) {
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));
        float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qs = block_ptr + 16;

        float sum = 0.0f;
        int is = 0;
        for (int j = 0; j < block_elements; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * sc1;
            float m1_val = dmin * m1;
            float d2 = d * sc2;
            float m2_val = dmin * m2;

            for (int l = 0; l < 32; ++l) {
                if (base + j + l < K)
                    sum += x[base + j + l] * (d1 * (qs[l] & 0xF) - m1_val);
            }
            for (int l = 0; l < 32; ++l) {
                if (base + j + 32 + l < K)
                    sum += x[base + j + 32 + l] * (d2 * (qs[l] >> 4) - m2_val);
            }
            qs += 32;
            is += 2;
        }
        return sum;
    }
};

// ---- Q6_K: 256 elements/block, 210 bytes/block ----
// Layout: ql[128] + qh[64] + sc[16+2]
template <>
struct CudaQuantTraits<DataType::Q6_K> {
    static constexpr int block_elements = QuantTraits<DataType::Q6_K>::block_elements;  // 256
    static constexpr int block_size = QuantTraits<DataType::Q6_K>::block_size;           // 210

    static __device__ __forceinline__ float dot_block(
        const uint8_t* block_ptr, const float* x, int base, int K) {
        const uint8_t* ql = block_ptr;
        const uint8_t* qh = ql + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(qh + 64);
        uint16_t d_bits;
        memcpy(&d_bits, sc + 16, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));

        const uint8_t* ql_cur = ql;
        const uint8_t* qh_cur = qh;
        const int8_t* sc_cur = sc;

        float sum = 0.0f;
        for (int n = 0; n < block_elements; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is_ = l / 16;
                int8_t q1 = (int8_t)((ql_cur[l + 0] & 0xF) | (((qh_cur[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql_cur[l + 32] & 0xF) | (((qh_cur[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql_cur[l + 0] >> 4) | (((qh_cur[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql_cur[l + 32] >> 4) | (((qh_cur[l] >> 6) & 3) << 4)) - 32;

                if (base + n + l + 0 < K)
                    sum += x[base + n + l + 0] * d * static_cast<float>(sc_cur[is_ + 0]) * static_cast<float>(q1);
                if (base + n + l + 32 < K)
                    sum += x[base + n + l + 32] * d * static_cast<float>(sc_cur[is_ + 2]) * static_cast<float>(q2);
                if (base + n + l + 64 < K)
                    sum += x[base + n + l + 64] * d * static_cast<float>(sc_cur[is_ + 4]) * static_cast<float>(q3);
                if (base + n + l + 96 < K)
                    sum += x[base + n + l + 96] * d * static_cast<float>(sc_cur[is_ + 6]) * static_cast<float>(q4);
            }
            ql_cur += 64;
            qh_cur += 32;
            sc_cur += 8;
        }
        return sum;
    }
};

// ---- Q3_K: 256 elements/block, 110 bytes/block ----
// Layout: hmask[32] + qs[64] + scales[12] + d[2]
template <>
struct CudaQuantTraits<DataType::Q3_K> {
    static constexpr int block_elements = QuantTraits<DataType::Q3_K>::block_elements;  // 256
    static constexpr int block_size = QuantTraits<DataType::Q3_K>::block_size;           // 110

    static __device__ __forceinline__ float dot_block(
        const uint8_t* block_ptr, const float* x, int base, int K) {
        const uint8_t* hm = block_ptr;
        const uint8_t* qs = block_ptr + 32;
        const uint8_t* scales_raw = block_ptr + 96;
        uint16_t d_bits;
        memcpy(&d_bits, block_ptr + 108, 2);
        float d_all = __half2float(reinterpret_cast<const __half&>(d_bits));

        int8_t scales[16];
        q3_k_unpack_scales(scales_raw, scales);

        float sum = 0.0f;
        int is = 0;
        uint8_t m = 1;
        const uint8_t* q = qs;

        for (int n = 0; n < block_elements; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int idx = base + n + j * 32 + l;
                    if (idx < K) {
                        int8_t q_val = static_cast<int8_t>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                        sum += x[idx] * dl * static_cast<float>(q_val);
                    }
                }

                dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int idx = base + n + j * 32 + 16 + l;
                    if (idx < K) {
                        int8_t q_val = static_cast<int8_t>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                        sum += x[idx] * dl * static_cast<float>(q_val);
                    }
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
        return sum;
    }
};

// ============================================================================
// Template GEMV kernels
// ============================================================================

// M=1 decode GEMV: one warp per output row
template <DataType DT>
__global__ void gemv_typed_transB_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ q_weight,
    float* __restrict__ out, int K, int N)
{
    constexpr int BE = CudaQuantTraits<DT>::block_elements;
    constexpr int BS = CudaQuantTraits<DT>::block_size;
    int num_blocks_row = (K + BE - 1) / BE;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N) return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const uint8_t* block_ptr = row_ptr + bi * BS;
        int base = bi * BE;
        sum += CudaQuantTraits<DT>::dot_block(block_ptr, x, base, K);
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) out[warp_id] = sum;
}

// M>1 batch GEMV: one warp per (m, n) output element
template <DataType DT>
__global__ void gemv_typed_transB_batch_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ q_weight,
    float* __restrict__ out, int M, int K, int N)
{
    constexpr int BE = CudaQuantTraits<DT>::block_elements;
    constexpr int BS = CudaQuantTraits<DT>::block_size;
    int num_blocks_row = (K + BE - 1) / BE;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int total_warps = M * N;
    if (warp_id >= total_warps) return;

    int m = warp_id / N;
    int n = warp_id % N;

    const float* x_row = x + (size_t)m * K;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const uint8_t* block_ptr = w_row + bi * BS;
        int base = bi * BE;
        sum += CudaQuantTraits<DT>::dot_block(block_ptr, x_row, base, K);
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) out[m * N + n] = sum;
}

// ============================================================================
// Template launch wrappers
// ============================================================================

template <DataType DT>
void launch_gemv_typed_transB(const float* x, const void* q_weight, float* out,
                               int K, int N, cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_typed_transB_kernel<DT><<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N);
}

template <DataType DT>
void launch_gemv_typed_transB_batch(const float* x, const void* q_weight, float* out,
                                     int M, int K, int N, cudaStream_t stream) {
    int total_warps = M * N;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_typed_transB_batch_kernel<DT><<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// Function pointer dispatch tables (defined in cuda_gemv_instances.cu)
// ============================================================================

using GemvFn = void (*)(const float*, const void*, float*, int, int, cudaStream_t);
using GemvBatchFn = void (*)(const float*, const void*, float*, int, int, int, cudaStream_t);

extern const GemvFn gemv_dispatch[18];
extern const GemvBatchFn gemv_batch_dispatch[18];

}  // namespace cuda
}  // namespace forge
