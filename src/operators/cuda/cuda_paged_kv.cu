// =========================================================================
// cuda_paged_kv.cu — Phase 4: CUDA paged KV cache kernels
//
// Two responsibilities:
//   1. launch_kv_scatter(): batch-writes FP32 K/V rows directly into pages,
//      quantizing to the target dtype in-place. No host staging.
//   2. launch_paged_flash_attention_gqa_decode_{q4_0,f16,q8_0}(): paged
//      attention decode kernels that traverse the sequence page table
//      directly, never materializing a contiguous K/V tensor.
//
// Page data layout: each page is a fixed-size device buffer of
// (page_size * row_bytes) bytes. A device pointer table (page_ptrs[page_id])
// gives the base of each page. The sequence page table (page_ids[j])
// maps logical page index -> physical page id.
//
// Row pointer for logical KV position j:
//   page_id   = page_ids[j / page_size]
//   offset    = j % page_size
//   row_ptr   = page_ptrs[page_id] + offset * row_bytes
// =========================================================================

#include <cstdio>

#include "cuda_attn_common.cuh"
#include "cuda_common.h"
#include "forge/cuda_kernels.h"
#include "forge/fp8_utils.h"
#include "forge/kv_cache.h"  // KVCacheDType full definition (for switch in launch_kv_scatter)

namespace forge {
namespace cuda {

// =========================================================================
// Device helper: resolve a logical KV row pointer via the page table.
// When page_ids == nullptr, falls back to contiguous layout (base + j*row_bytes).
// =========================================================================
__device__ __forceinline__ const uint8_t* paged_kv_row(const uint8_t* base,
                                                       const int32_t* __restrict__ page_ids,
                                                       void* const* __restrict__ page_ptrs,
                                                       int page_size, size_t row_bytes, int j) {
    if (page_ids == nullptr) {
        return base + (size_t)j * row_bytes;
    }
    int page_idx = j / page_size;
    int offset = j - page_idx * page_size;
    int page_id = page_ids[page_idx];
    const uint8_t* page_base = static_cast<const uint8_t*>(page_ptrs[page_id]);
    return page_base + (size_t)offset * row_bytes;
}

__device__ __forceinline__ uint8_t* paged_kv_row_mut(uint8_t* base,
                                                     const int32_t* __restrict__ page_ids,
                                                     void* const* __restrict__ page_ptrs,
                                                     int page_size, size_t row_bytes, int j) {
    if (page_ids == nullptr) {
        return base + (size_t)j * row_bytes;
    }
    int page_idx = j / page_size;
    int offset = j - page_idx * page_size;
    int page_id = page_ids[page_idx];
    uint8_t* page_base = static_cast<uint8_t*>(page_ptrs[page_id]);
    return page_base + (size_t)offset * row_bytes;
}

// =========================================================================
// Row quantization device functions.
// Each writes one row of `n` FP32 elements into the target dtype buffer `dst`.
// Thread cooperation: blockDim.x threads cooperate per row.
// =========================================================================

// ---- F16 row: FP32 -> FP16 ----
__device__ __forceinline__ void quantize_row_f16(const float* __restrict__ src,
                                                 __half* __restrict__ dst, int n) {
    int tid = threadIdx.x;
    int bs = blockDim.x;
    for (int i = tid; i < n; i += bs) {
        dst[i] = __float2half(src[i]);
    }
}

// ---- FP32 row: memcpy ----
__device__ __forceinline__ void copy_row_fp32(const float* __restrict__ src,
                                              float* __restrict__ dst, int n) {
    int tid = threadIdx.x;
    int bs = blockDim.x;
    for (int i = tid; i < n; i += bs) {
        dst[i] = src[i];
    }
}

// ---- Q8_0 row: fp16 scale + int8 qs[32], 34 bytes per 32-elem block ----
__device__ __forceinline__ void quantize_row_q8_0(const float* __restrict__ src,
                                                  uint8_t* __restrict__ dst, int n) {
    const int QK = 32;
    const int Q8_0_BLOCK_SIZE = 34;
    int num_blocks = (n + QK - 1) / QK;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    for (int bi = tid; bi < num_blocks; bi += bs) {
        int start = bi * QK;
        int end = min(start + QK, n);

        float amax = 0.0f;
        for (int i = start; i < end; ++i) {
            amax = fmaxf(amax, fabsf(src[i]));
        }
        float d = amax / 127.0f;
        if (d == 0.0f)
            d = 1.0f;  // avoid div-by-zero; scale becomes irrelevant
        float id = 1.0f / d;

        uint8_t* block_ptr = dst + bi * Q8_0_BLOCK_SIZE;
        // Store the fp16 scale as a single 16-bit store (see cuda_quant.cu
        // quantize_q8_0_matrix_kernel for why memcpy/byte-stores into a
        // uint8_t* get clobbered by the subsequent aliased int8_t writes).
        {
            __half scale_half = __float2half(d);
            uint16_t s;
            memcpy(&s, &scale_half, sizeof(uint16_t));
            *reinterpret_cast<uint16_t*>(block_ptr) = s;
        }
        int8_t* qs = reinterpret_cast<int8_t*>(block_ptr + 2);

        for (int i = 0; i < QK; ++i) {
            int idx = start + i;
            float v = (idx < end) ? src[idx] * id : 0.0f;
            int q = (int)roundf(v);
            q = max(-128, min(127, q));
            qs[i] = static_cast<int8_t>(q);
        }
    }
}

// ---- Q4_0 row: fp16 scale + 4-bit packed nibbles, 18 bytes per 32-elem block ----
__device__ __forceinline__ void quantize_row_q4_0(const float* __restrict__ src,
                                                  uint8_t* __restrict__ dst, int n) {
    const int QK = 32;
    const int Q4_0_BLOCK_SIZE = 18;
    int num_blocks = (n + QK - 1) / QK;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    for (int bi = tid; bi < num_blocks; bi += bs) {
        int start = bi * QK;
        int end = min(start + QK, n);

        float amax = 0.0f;
        float max_val = 0.0f;
        for (int i = start; i < end; ++i) {
            float a = fabsf(src[i]);
            if (amax < a) {
                amax = a;
                max_val = src[i];
            }
        }
        float d = max_val / -8.0f;
        if (d == 0.0f)
            d = -1.0f;
        float id = 1.0f / d;

        uint8_t* block_ptr = dst + bi * Q4_0_BLOCK_SIZE;
        // Store fp16 scale as a 16-bit store (see cuda_quant.cu
        // quantize_q8_0_matrix_kernel comment for why memcpy/byte-stores into a
        // uint8_t* get clobbered by subsequent aliased int8/uint8 writes).
        {
            __half scale_half = __float2half(d);
            uint16_t s;
            memcpy(&s, &scale_half, sizeof(uint16_t));
            *reinterpret_cast<uint16_t*>(block_ptr) = s;
        }
        uint8_t* qs = block_ptr + 2;

        for (int i = 0; i < 16; ++i) {
            int idx0 = start + i;
            int idx1 = start + i + 16;
            float x0 = (idx0 < end) ? src[idx0] * id : 0.0f;
            float x1 = (idx1 < end) ? src[idx1] * id : 0.0f;
            uint8_t qi0 = min(15, max(0, (int)(x0 + 8.5f)));
            uint8_t qi1 = min(15, max(0, (int)(x1 + 8.5f)));
            qs[i] = (qi0 & 0x0F) | ((qi1 & 0x0F) << 4);
        }
    }
}

// =========================================================================
// kv_scatter_kernel — one block per token, writes K and V rows for that token.
//
// Each block quantizes one K row and one V row into the target page slot.
// KV dtype is selected at launch time via separate kernel instantiations to
// keep the inner loop branch-free.
//
// pos:        starting logical position for this batch of tokens.
// page_size:  tokens per page.
// kv_dim:     elements per K/V row (= num_kv_heads * head_dim).
// =========================================================================

template <typename RowFn>  // RowFn: void(const float* src, uint8_t* dst, int n)
__global__ void kv_scatter_kernel(const float* __restrict__ k_src, const float* __restrict__ v_src,
                                  void* const* __restrict__ k_page_ptrs,
                                  void* const* __restrict__ v_page_ptrs,
                                  const int32_t* __restrict__ page_ids, int n_tokens, int64_t pos,
                                  int page_size, int kv_dim, size_t k_row_bytes,
                                  size_t v_row_bytes) {
    int tok = blockIdx.x;
    if (tok >= n_tokens)
        return;

    int64_t abs_pos = pos + tok;
    int page_idx = (int)(abs_pos / page_size);
    int offset = (int)(abs_pos - (int64_t)page_idx * page_size);
    int page_id = page_ids[page_idx];

    uint8_t* k_dst = static_cast<uint8_t*>(k_page_ptrs[page_id]) + (size_t)offset * k_row_bytes;
    uint8_t* v_dst = static_cast<uint8_t*>(v_page_ptrs[page_id]) + (size_t)offset * v_row_bytes;

    const float* k_row_src = k_src + (size_t)tok * kv_dim;
    const float* v_row_src = v_src + (size_t)tok * kv_dim;

    RowFn()(k_row_src, k_dst, kv_dim);
    __syncthreads();
    RowFn()(v_row_src, v_dst, kv_dim);
}

// Row functor types for template dispatch
struct RowFnF16 {
    __device__ void operator()(const float* s, uint8_t* d, int n) {
        quantize_row_f16(s, reinterpret_cast<__half*>(d), n);
    }
};
struct RowFnFP32 {
    __device__ void operator()(const float* s, uint8_t* d, int n) {
        copy_row_fp32(s, reinterpret_cast<float*>(d), n);
    }
};
struct RowFnQ8_0 {
    __device__ void operator()(const float* s, uint8_t* d, int n) { quantize_row_q8_0(s, d, n); }
};
struct RowFnQ4_0 {
    __device__ void operator()(const float* s, uint8_t* d, int n) { quantize_row_q4_0(s, d, n); }
};

// =========================================================================
// launch_kv_scatter — public entry point.
// Writes n_tokens K/V rows starting at logical position `pos` into the paged
// KV cache, quantizing to `dtype` in-place on device.
// =========================================================================
void launch_kv_scatter(const float* k_src, const float* v_src, void* const* k_page_ptrs,
                       void* const* v_page_ptrs, const int32_t* page_ids, int n_tokens, int64_t pos,
                       int page_size, int kv_dim, size_t k_row_bytes, size_t v_row_bytes,
                       KVCacheDType dtype, cudaStream_t stream) {
    if (n_tokens <= 0)
        return;
    int threads = 256;

    switch (dtype) {
    case KVCacheDType::F16:
        kv_scatter_kernel<RowFnF16><<<n_tokens, threads, 0, stream>>>(
            k_src, v_src, k_page_ptrs, v_page_ptrs, page_ids, n_tokens, pos, page_size, kv_dim,
            k_row_bytes, v_row_bytes);
        break;
    case KVCacheDType::FP32:
        kv_scatter_kernel<RowFnFP32><<<n_tokens, threads, 0, stream>>>(
            k_src, v_src, k_page_ptrs, v_page_ptrs, page_ids, n_tokens, pos, page_size, kv_dim,
            k_row_bytes, v_row_bytes);
        break;
    case KVCacheDType::Q8_0:
        kv_scatter_kernel<RowFnQ8_0><<<n_tokens, threads, 0, stream>>>(
            k_src, v_src, k_page_ptrs, v_page_ptrs, page_ids, n_tokens, pos, page_size, kv_dim,
            k_row_bytes, v_row_bytes);
        break;
    case KVCacheDType::Q4_0:
        kv_scatter_kernel<RowFnQ4_0><<<n_tokens, threads, 0, stream>>>(
            k_src, v_src, k_page_ptrs, v_page_ptrs, page_ids, n_tokens, pos, page_size, kv_dim,
            k_row_bytes, v_row_bytes);
        break;
    default:
        fprintf(stderr, "[ERROR] launch_kv_scatter: unsupported dtype=%d\n",
                static_cast<int>(dtype));
        break;
    }
}

// =========================================================================
// Paged Flash Attention GQA Decode kernels.
//
// Identical math to the fused (contiguous) decode kernels in cuda_fused_attn.cu,
// except the K/V row pointer is resolved through the page table. Single-pass
// online softmax, works for all HEAD_DIM (64..512).
// =========================================================================

// ---- Q4_0 paged decode ----
template <int HEAD_DIM, int NUM_WARPS>
__global__ void paged_attn_q4_0_decode_kernel(const float* __restrict__ Q,
                                              void* const* __restrict__ k_page_ptrs,
                                              void* const* __restrict__ v_page_ptrs,
                                              const int32_t* __restrict__ page_ids,
                                              float* __restrict__ O, int kv_len, int num_heads,
                                              int num_kv_heads, int page_size, size_t q_row_size,
                                              const float* __restrict__ mask_row) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    constexpr int BLOCK_ELEMS = 32;
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int NUM_BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_ELEMS;

    const size_t head_byte_offset = (size_t)kv_h * NUM_BLOCKS_PER_HEAD * Q4_0_BLOCK_SIZE;

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size) {
        s_q[d] = Q[h * HEAD_DIM + d];
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        if (mask_row != nullptr && mask_row[j] < -1e20f)
            continue;

        const uint8_t* k_row =
            paged_kv_row(nullptr, page_ids, k_page_ptrs, page_size, q_row_size, j) +
            head_byte_offset;
        const uint8_t* v_row =
            paged_kv_row(nullptr, page_ids, v_page_ptrs, page_size, q_row_size, j) +
            head_byte_offset;

        float dot = 0.0f;
#pragma unroll
        for (int bi = 0; bi < NUM_BLOCKS_PER_HEAD; ++bi) {
            const uint8_t* block_ptr = k_row + bi * Q4_0_BLOCK_SIZE;
            uint16_t scale_bits;
            memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
            float k_scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
            const uint8_t* qs = block_ptr + sizeof(__half);

#pragma unroll
            for (int i = 0; i < 16; ++i) {
                int val_lo = (qs[i] & 0x0F) - 8;
                int val_hi = ((qs[i] >> 4) & 0x0F) - 8;
                int idx_lo = bi * 32 + i;
                int idx_hi = bi * 32 + i + 16;
                dot += s_q[idx_lo] * (val_lo * k_scale);
                dot += s_q[idx_hi] * (val_hi * k_scale);
            }
        }
        dot *= scale;
        if (mask_row != nullptr)
            dot += mask_row[j];

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int bi = 0; bi < NUM_BLOCKS_PER_HEAD; ++bi) {
            const uint8_t* block_ptr = v_row + bi * Q4_0_BLOCK_SIZE;
            uint16_t scale_bits;
            memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
            float v_scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
            const uint8_t* qs = block_ptr + sizeof(__half);

#pragma unroll
            for (int i = 0; i < 16; ++i) {
                int val_lo = (qs[i] & 0x0F) - 8;
                int val_hi = ((qs[i] >> 4) & 0x0F) - 8;
                int idx_lo = bi * 32 + i;
                int idx_hi = bi * 32 + i + 16;
                local_acc[idx_lo] = local_acc[idx_lo] * correction + weight * (val_lo * v_scale);
                local_acc[idx_hi] = local_acc[idx_hi] * correction + weight * (val_hi * v_scale);
            }
        }
        local_max = new_max;
    }

    float warp_max = warp_reduce_max(local_max);
    float rescale = expf(local_max - warp_max);
    float warp_sum = warp_reduce_sum(local_sum * rescale);
    float warp_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        warp_acc[d] = local_acc[d] * rescale;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        warp_acc[d] = warp_reduce_sum(warp_acc[d]);

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_warp_acc[NUM_WARPS * HEAD_DIM];

    if (lane == 0) {
        s_warp_max[warp_id] = warp_max;
        s_warp_sum[warp_id] = warp_sum;
    }
    for (int d = lane; d < HEAD_DIM; d += 32) {
        s_warp_acc[warp_id * HEAD_DIM + d] = warp_acc[d];
    }
    __syncthreads();

    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(s_warp_max, s_warp_sum, s_warp_acc, O, h, lane);
}

// ---- F16 paged decode ----
template <int HEAD_DIM, int NUM_WARPS>
__global__ void paged_attn_f16_decode_kernel(const float* __restrict__ Q,
                                             void* const* __restrict__ k_page_ptrs,
                                             void* const* __restrict__ v_page_ptrs,
                                             const int32_t* __restrict__ page_ids,
                                             float* __restrict__ O, int kv_len, int num_heads,
                                             int num_kv_heads, int page_size, size_t q_row_size,
                                             const float* __restrict__ mask_row) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const size_t head_byte_offset = (size_t)kv_h * HEAD_DIM * sizeof(__half);

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size) {
        s_q[d] = Q[h * HEAD_DIM + d];
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        if (mask_row != nullptr && mask_row[j] < -1e20f)
            continue;

        const __half* k_row = reinterpret_cast<const __half*>(
            paged_kv_row(nullptr, page_ids, k_page_ptrs, page_size, q_row_size, j) +
            head_byte_offset);
        const __half* v_row = reinterpret_cast<const __half*>(
            paged_kv_row(nullptr, page_ids, v_page_ptrs, page_size, q_row_size, j) +
            head_byte_offset);

        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_q[d] * __half2float(k_row[d]);
        }
        dot *= scale;
        if (mask_row != nullptr)
            dot += mask_row[j];

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            local_acc[d] = local_acc[d] * correction + weight * __half2float(v_row[d]);
        }
        local_max = new_max;
    }

    float warp_max = warp_reduce_max(local_max);
    float rescale = expf(local_max - warp_max);
    float warp_sum = warp_reduce_sum(local_sum * rescale);
    float warp_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        warp_acc[d] = local_acc[d] * rescale;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        warp_acc[d] = warp_reduce_sum(warp_acc[d]);

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_warp_acc[NUM_WARPS * HEAD_DIM];

    if (lane == 0) {
        s_warp_max[warp_id] = warp_max;
        s_warp_sum[warp_id] = warp_sum;
    }
    for (int d = lane; d < HEAD_DIM; d += 32) {
        s_warp_acc[warp_id * HEAD_DIM + d] = warp_acc[d];
    }
    __syncthreads();

    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(s_warp_max, s_warp_sum, s_warp_acc, O, h, lane);
}

// ---- Q8_0 paged decode ----
template <int HEAD_DIM, int NUM_WARPS>
__global__ void paged_attn_q8_0_decode_kernel(const float* __restrict__ Q,
                                              void* const* __restrict__ k_page_ptrs,
                                              void* const* __restrict__ v_page_ptrs,
                                              const int32_t* __restrict__ page_ids,
                                              float* __restrict__ O, int kv_len, int num_heads,
                                              int num_kv_heads, int page_size, size_t q_row_size,
                                              const float* __restrict__ mask_row) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    constexpr int BLOCK_ELEMS = 32;
    constexpr int Q8_0_BLOCK_SIZE = 34;
    constexpr int NUM_BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_ELEMS;

    const size_t head_byte_offset = (size_t)kv_h * NUM_BLOCKS_PER_HEAD * Q8_0_BLOCK_SIZE;

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size) {
        s_q[d] = Q[h * HEAD_DIM + d];
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        if (mask_row != nullptr && mask_row[j] < -1e20f)
            continue;

        const uint8_t* k_row =
            paged_kv_row(nullptr, page_ids, k_page_ptrs, page_size, q_row_size, j) +
            head_byte_offset;
        const uint8_t* v_row =
            paged_kv_row(nullptr, page_ids, v_page_ptrs, page_size, q_row_size, j) +
            head_byte_offset;

        float dot = 0.0f;
#pragma unroll
        for (int bi = 0; bi < NUM_BLOCKS_PER_HEAD; ++bi) {
            const uint8_t* block_ptr = k_row + bi * Q8_0_BLOCK_SIZE;
            uint16_t scale_bits;
            memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
            float k_scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
            const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + sizeof(__half));

#pragma unroll
            for (int i = 0; i < BLOCK_ELEMS; ++i) {
                dot += s_q[bi * 32 + i] * (qs[i] * k_scale);
            }
        }
        dot *= scale;
        if (mask_row != nullptr)
            dot += mask_row[j];

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int bi = 0; bi < NUM_BLOCKS_PER_HEAD; ++bi) {
            const uint8_t* block_ptr = v_row + bi * Q8_0_BLOCK_SIZE;
            uint16_t scale_bits;
            memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
            float v_scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
            const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + sizeof(__half));

#pragma unroll
            for (int i = 0; i < BLOCK_ELEMS; ++i) {
                int idx = bi * 32 + i;
                local_acc[idx] = local_acc[idx] * correction + weight * (qs[i] * v_scale);
            }
        }
        local_max = new_max;
    }

    float warp_max = warp_reduce_max(local_max);
    float rescale = expf(local_max - warp_max);
    float warp_sum = warp_reduce_sum(local_sum * rescale);
    float warp_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        warp_acc[d] = local_acc[d] * rescale;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        warp_acc[d] = warp_reduce_sum(warp_acc[d]);

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_warp_acc[NUM_WARPS * HEAD_DIM];

    if (lane == 0) {
        s_warp_max[warp_id] = warp_max;
        s_warp_sum[warp_id] = warp_sum;
    }
    for (int d = lane; d < HEAD_DIM; d += 32) {
        s_warp_acc[warp_id * HEAD_DIM + d] = warp_acc[d];
    }
    __syncthreads();

    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(s_warp_max, s_warp_sum, s_warp_acc, O, h, lane);
}

// =========================================================================
// Launch functions for paged decode kernels
// =========================================================================

void launch_paged_flash_attention_gqa_decode_q4_0(const float* Q, void* const* k_page_ptrs,
                                                  void* const* v_page_ptrs, const int32_t* page_ids,
                                                  float* O, int kv_len, int num_heads,
                                                  int num_kv_heads, int head_dim, int page_size,
                                                  size_t q_row_size, const float* mask_row,
                                                  cudaStream_t stream) {
    int blocks = num_heads;

#define LAUNCH(HD)                                                                                \
    paged_attn_q4_0_decode_kernel<HD, 4>                                                          \
        <<<blocks, 128, 0, stream>>>(Q, k_page_ptrs, v_page_ptrs, page_ids, O, kv_len, num_heads, \
                                     num_kv_heads, page_size, q_row_size, mask_row)

    switch (head_dim) {
    case 64:
        LAUNCH(64);
        break;
    case 96:
        LAUNCH(96);
        break;
    case 128:
        LAUNCH(128);
        break;
    case 256:
        LAUNCH(256);
        break;
    case 512:
        LAUNCH(512);
        break;
    default:
        fprintf(stderr, "[ERROR] paged_attn_q4_0_decode: unsupported head_dim=%d\n", head_dim);
        break;
    }
#undef LAUNCH
}

void launch_paged_flash_attention_gqa_decode_f16(const float* Q, void* const* k_page_ptrs,
                                                 void* const* v_page_ptrs, const int32_t* page_ids,
                                                 float* O, int kv_len, int num_heads,
                                                 int num_kv_heads, int head_dim, int page_size,
                                                 size_t q_row_size, const float* mask_row,
                                                 cudaStream_t stream) {
    int blocks = num_heads;

#define LAUNCH(HD)                                                                                \
    paged_attn_f16_decode_kernel<HD, 4>                                                           \
        <<<blocks, 128, 0, stream>>>(Q, k_page_ptrs, v_page_ptrs, page_ids, O, kv_len, num_heads, \
                                     num_kv_heads, page_size, q_row_size, mask_row)

    switch (head_dim) {
    case 64:
        LAUNCH(64);
        break;
    case 96:
        LAUNCH(96);
        break;
    case 128:
        LAUNCH(128);
        break;
    case 256:
        LAUNCH(256);
        break;
    case 512:
        LAUNCH(512);
        break;
    default:
        fprintf(stderr, "[ERROR] paged_attn_f16_decode: unsupported head_dim=%d\n", head_dim);
        break;
    }
#undef LAUNCH
}

void launch_paged_flash_attention_gqa_decode_q8_0(const float* Q, void* const* k_page_ptrs,
                                                  void* const* v_page_ptrs, const int32_t* page_ids,
                                                  float* O, int kv_len, int num_heads,
                                                  int num_kv_heads, int head_dim, int page_size,
                                                  size_t q_row_size, const float* mask_row,
                                                  cudaStream_t stream) {
    int blocks = num_heads;

#define LAUNCH(HD)                                                                                \
    paged_attn_q8_0_decode_kernel<HD, 4>                                                          \
        <<<blocks, 128, 0, stream>>>(Q, k_page_ptrs, v_page_ptrs, page_ids, O, kv_len, num_heads, \
                                     num_kv_heads, page_size, q_row_size, mask_row)

    switch (head_dim) {
    case 64:
        LAUNCH(64);
        break;
    case 96:
        LAUNCH(96);
        break;
    case 128:
        LAUNCH(128);
        break;
    case 256:
        LAUNCH(256);
        break;
    case 512:
        LAUNCH(512);
        break;
    default:
        fprintf(stderr, "[ERROR] paged_attn_q8_0_decode: unsupported head_dim=%d\n", head_dim);
        break;
    }
#undef LAUNCH
}

// =========================================================================
// FP8 Paged GQA Decode Kernel（E4M3 / E5M2）
// 与 paged f16 decode 同构，K/V 以 fp8（1 字节/元素）在线反量化。
// =========================================================================

template <typename Fp8Fmt, int HEAD_DIM, int NUM_WARPS>
__global__ void paged_attn_fp8_decode_kernel(const float* __restrict__ Q,
                                             void* const* __restrict__ k_page_ptrs,
                                             void* const* __restrict__ v_page_ptrs,
                                             const int32_t* __restrict__ page_ids,
                                             float* __restrict__ O, int kv_len, int num_heads,
                                             int num_kv_heads, int page_size, size_t q_row_size,
                                             const float* __restrict__ mask_row) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const size_t head_byte_offset = (size_t)kv_h * HEAD_DIM;

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size) {
        s_q[d] = Q[h * HEAD_DIM + d];
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        if (mask_row != nullptr && mask_row[j] < -1e20f)
            continue;

        const uint8_t* k_row =
            paged_kv_row(nullptr, page_ids, k_page_ptrs, page_size, q_row_size, j) +
            head_byte_offset;
        const uint8_t* v_row =
            paged_kv_row(nullptr, page_ids, v_page_ptrs, page_size, q_row_size, j) +
            head_byte_offset;

        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_q[d] * fp8_load<Fp8Fmt>(k_row, d);
        }
        dot *= scale;
        if (mask_row != nullptr)
            dot += mask_row[j];

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            local_acc[d] = local_acc[d] * correction + weight * fp8_load<Fp8Fmt>(v_row, d);
        }
        local_max = new_max;
    }

    float warp_max = warp_reduce_max(local_max);
    float rescale = expf(local_max - warp_max);
    float warp_sum = warp_reduce_sum(local_sum * rescale);
    float warp_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        warp_acc[d] = local_acc[d] * rescale;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        warp_acc[d] = warp_reduce_sum(warp_acc[d]);

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_warp_acc[NUM_WARPS * HEAD_DIM];

    if (lane == 0) {
        s_warp_max[warp_id] = warp_max;
        s_warp_sum[warp_id] = warp_sum;
    }
    for (int d = lane; d < HEAD_DIM; d += 32) {
        s_warp_acc[warp_id * HEAD_DIM + d] = warp_acc[d];
    }
    __syncthreads();

    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(s_warp_max, s_warp_sum, s_warp_acc, O, h, lane);
}

#define LAUNCH_PAGED_FP8(FMT, HD)                                                                 \
    paged_attn_fp8_decode_kernel<FMT, HD, 4>                                                      \
        <<<blocks, 128, 0, stream>>>(Q, k_page_ptrs, v_page_ptrs, page_ids, O, kv_len, num_heads, \
                                     num_kv_heads, page_size, q_row_size, mask_row)

void launch_paged_flash_attention_gqa_decode_fp8_e4m3(const float* Q, void* const* k_page_ptrs,
                                                      void* const* v_page_ptrs,
                                                      const int32_t* page_ids, float* O, int kv_len,
                                                      int num_heads, int num_kv_heads, int head_dim,
                                                      int page_size, size_t q_row_size,
                                                      const float* mask_row, cudaStream_t stream) {
    int blocks = num_heads;
    switch (head_dim) {
    case 64:
        LAUNCH_PAGED_FP8(Fp8E4M3, 64);
        break;
    case 96:
        LAUNCH_PAGED_FP8(Fp8E4M3, 96);
        break;
    case 128:
        LAUNCH_PAGED_FP8(Fp8E4M3, 128);
        break;
    case 256:
        LAUNCH_PAGED_FP8(Fp8E4M3, 256);
        break;
    case 512:
        LAUNCH_PAGED_FP8(Fp8E4M3, 512);
        break;
    default:
        fprintf(stderr, "[ERROR] paged_attn_fp8_e4m3_decode: unsupported head_dim=%d\n", head_dim);
        break;
    }
}

void launch_paged_flash_attention_gqa_decode_fp8_e5m2(const float* Q, void* const* k_page_ptrs,
                                                      void* const* v_page_ptrs,
                                                      const int32_t* page_ids, float* O, int kv_len,
                                                      int num_heads, int num_kv_heads, int head_dim,
                                                      int page_size, size_t q_row_size,
                                                      const float* mask_row, cudaStream_t stream) {
    int blocks = num_heads;
    switch (head_dim) {
    case 64:
        LAUNCH_PAGED_FP8(Fp8E5M2, 64);
        break;
    case 96:
        LAUNCH_PAGED_FP8(Fp8E5M2, 96);
        break;
    case 128:
        LAUNCH_PAGED_FP8(Fp8E5M2, 128);
        break;
    case 256:
        LAUNCH_PAGED_FP8(Fp8E5M2, 256);
        break;
    case 512:
        LAUNCH_PAGED_FP8(Fp8E5M2, 512);
        break;
    default:
        fprintf(stderr, "[ERROR] paged_attn_fp8_e5m2_decode: unsupported head_dim=%d\n", head_dim);
        break;
    }
#undef LAUNCH_PAGED_FP8
}

}  // namespace cuda
}  // namespace forge
