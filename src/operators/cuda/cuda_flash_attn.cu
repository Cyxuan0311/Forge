#include <cmath>
#include <cstdio>

#include "cuda_attn_common.cuh"
#include "cuda_flash_attn.h"

namespace forge {
namespace cuda {

// =========================================================================
// Prefill kernel (q_len > 1) — SINGLE-PASS online softmax
//
// Each thread iterates over its assigned KV positions, maintaining
// local max/sum/acc incrementally. After the loop:
//   - Warp-level reduce (max → sum → acc[d])
//   - Cross-warp reduce via shared memory with LSE merge
//   QK computed only ONCE, NO atomicAdd.
// =========================================================================

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_attn_kernel(const float* Q, const float* K, const float* V, float* O,
                                  int q_len, int kv_len, int num_heads,
                                  const float* mask, bool causal) {
    int h = blockIdx.y;
    int qi = blockIdx.x;
    if (h >= num_heads || qi >= q_len) return;

    int q_pos = kv_len - q_len + qi;
    const float* q_row = Q + qi * num_heads * HEAD_DIM + h * HEAD_DIM;
    float* o_row = O + qi * num_heads * HEAD_DIM + h * HEAD_DIM;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        float m = (mask != nullptr) ? mask[qi * kv_len + j] : 0.0f;
        if (m < -1e20f) continue;
        if (!mask && causal && j > q_pos) continue;

        float dot = 0.0f;
        const float* k_row = K + j * num_heads * HEAD_DIM + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) dot += q_row[d] * k_row[d];
        dot *= scale;
        dot += m;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

        const float* v_row = V + j * num_heads * HEAD_DIM + h * HEAD_DIM;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            local_acc[d] = local_acc[d] * correction + weight * v_row[d];
        local_max = new_max;
    }

    float w_max = warp_reduce_max(local_max);
    float rescale = expf(local_max - w_max);
    float w_sum = warp_reduce_sum(local_sum * rescale);
    float w_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) w_acc[d] = local_acc[d] * rescale;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) w_acc[d] = warp_reduce_sum(w_acc[d]);

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_warp_acc[NUM_WARPS * HEAD_DIM];
    if (lane == 0) { s_warp_max[warp_id] = w_max; s_warp_sum[warp_id] = w_sum; }
    for (int d = lane; d < HEAD_DIM; d += 32) s_warp_acc[warp_id * HEAD_DIM + d] = w_acc[d];
    __syncthreads();

    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(
        s_warp_max, s_warp_sum, s_warp_acc, o_row, 0, lane);
}

void launch_flash_attention(const float* Q, const float* K, const float* V, float* O, int q_len,
                            int kv_len, int num_heads, int head_dim,
                            const float* mask, bool causal, cudaStream_t stream) {
    dim3 grid(q_len, num_heads);
    int threads = 128;
#define LAUNCH(HD) flash_attn_kernel<HD, 128><<<grid, threads, 0, stream>>>(Q, K, V, O, q_len, kv_len, num_heads, mask, causal)
    switch (head_dim) {
    case 16:  LAUNCH(16);  break;
    case 32:  LAUNCH(32);  break;
    case 64:  LAUNCH(64);  break;
    case 72:  LAUNCH(72);  break;
    case 96:  LAUNCH(96);  break;
    case 128: LAUNCH(128); break;
    case 256: LAUNCH(256); break;
    case 512: LAUNCH(512); break;
    default:  fprintf(stderr, "[ERROR] flash_attn: unsupported head_dim=%d\n", head_dim); break;
    }
#undef LAUNCH
}

// =========================================================================
// GQA Prefill kernel — SINGLE-PASS online softmax (same structure, GQA mapping)
// =========================================================================

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_attn_gqa_kernel(const float* Q, const float* K, const float* V, float* O,
                                      int q_len, int kv_len, int num_heads, int num_kv_heads,
                                      const float* mask, bool causal) {
    int h = blockIdx.y;
    int qi = blockIdx.x;
    if (h >= num_heads || qi >= q_len) return;

    int q_pos = kv_len - q_len + qi;
    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    const float* q_row = Q + qi * num_heads * HEAD_DIM + h * HEAD_DIM;
    float* o_row = O + qi * num_heads * HEAD_DIM + h * HEAD_DIM;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        float m_val = (mask != nullptr) ? mask[qi * kv_len + j] : 0.0f;
        if (m_val < -1e20f) continue;
        if (!mask && causal && j > q_pos) continue;

        float dot = 0.0f;
        const float* k_row = K + j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) dot += q_row[d] * k_row[d];
        dot *= scale;
        dot += m_val;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

        const float* v_row = V + j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            local_acc[d] = local_acc[d] * correction + weight * v_row[d];
        local_max = new_max;
    }

    float w_max = warp_reduce_max(local_max);
    float rescale = expf(local_max - w_max);
    float w_sum = warp_reduce_sum(local_sum * rescale);
    float w_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) w_acc[d] = local_acc[d] * rescale;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) w_acc[d] = warp_reduce_sum(w_acc[d]);

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_warp_acc[NUM_WARPS * HEAD_DIM];
    if (lane == 0) { s_warp_max[warp_id] = w_max; s_warp_sum[warp_id] = w_sum; }
    for (int d = lane; d < HEAD_DIM; d += 32) s_warp_acc[warp_id * HEAD_DIM + d] = w_acc[d];
    __syncthreads();

    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(
        s_warp_max, s_warp_sum, s_warp_acc, o_row, 0, lane);
}

void launch_flash_attention_gqa(const float* Q, const float* K, const float* V, float* O, int q_len,
                                int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                const float* mask, bool causal, cudaStream_t stream) {
    dim3 grid(q_len, num_heads);
    int threads = 128;
#define LAUNCH(HD) flash_attn_gqa_kernel<HD, 128><<<grid, threads, 0, stream>>>(Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads, mask, causal)
    switch (head_dim) {
    case 16:  LAUNCH(16);  break;
    case 32:  LAUNCH(32);  break;
    case 64:  LAUNCH(64);  break;
    case 72:  LAUNCH(72);  break;
    case 96:  LAUNCH(96);  break;
    case 128: LAUNCH(128); break;
    case 256: LAUNCH(256); break;
    case 512: LAUNCH(512); break;
    default:  fprintf(stderr, "[ERROR] flash_attn_gqa: unsupported head_dim=%d\n", head_dim); break;
    }
#undef LAUNCH
}

// =========================================================================
// GQA Decode kernel (HEAD_DIM <= 128)
// Per-thread register arrays + warp-level reduction
// SINGLE-PASS online softmax with float4 vectorized QK/V access
// =========================================================================

template <int HEAD_DIM, int NUM_WARPS>
__global__ void flash_attn_gqa_decode_kernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V, float* __restrict__ O,
                                             int kv_len, int num_heads, int num_kv_heads,
                                             const float* __restrict__ mask_row) {
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size) s_q[d] = Q[h * HEAD_DIM + d];
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
    constexpr int VEC_COUNT = HEAD_DIM / 4;

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const float* k_row = K + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        const float* v_row = V + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;

        float dot = 0.0f;
#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 kv = *reinterpret_cast<const float4*>(k_row + vi * 4);
            float4 qv = *reinterpret_cast<const float4*>(s_q + vi * 4);
            dot += kv.x * qv.x + kv.y * qv.y + kv.z * qv.z + kv.w * qv.w;
        }
        dot *= scale;
        if (mask_row != nullptr) dot += mask_row[j];

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 vf = *reinterpret_cast<const float4*>(v_row + vi * 4);
            local_acc[vi * 4 + 0] = local_acc[vi * 4 + 0] * correction + weight * vf.x;
            local_acc[vi * 4 + 1] = local_acc[vi * 4 + 1] * correction + weight * vf.y;
            local_acc[vi * 4 + 2] = local_acc[vi * 4 + 2] * correction + weight * vf.z;
            local_acc[vi * 4 + 3] = local_acc[vi * 4 + 3] * correction + weight * vf.w;
        }
        local_max = new_max;
    }

    float w_max = warp_reduce_max(local_max);
    float rescale = expf(local_max - w_max);
    float w_sum = warp_reduce_sum(local_sum * rescale);
    float w_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) w_acc[d] = local_acc[d] * rescale;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) w_acc[d] = warp_reduce_sum(w_acc[d]);

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_warp_acc[NUM_WARPS * HEAD_DIM];
    if (lane == 0) { s_warp_max[warp_id] = w_max; s_warp_sum[warp_id] = w_sum; }
    for (int d = lane; d < HEAD_DIM; d += 32) s_warp_acc[warp_id * HEAD_DIM + d] = w_acc[d];
    __syncthreads();

    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(
        s_warp_max, s_warp_sum, s_warp_acc, O, h, lane);
}

// =========================================================================
// GQA Decode kernel for LARGE head_dim (256, 512) — SINGLE-PASS online softmax
//
// Use 4 warps (128 threads) so each thread can keep local_acc[HEAD_DIM]
// in registers. NO QK recomputation, NO atomicAdd.
// =========================================================================

template <int HEAD_DIM, int NUM_WARPS>
__global__ void flash_attn_gqa_decode_large_online_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int kv_len, int num_heads, int num_kv_heads,
    const float* __restrict__ mask_row) {

    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size) s_q[d] = Q[h * HEAD_DIM + d];
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const float* k_row = K + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        const float* v_row = V + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;

        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) dot += s_q[d] * k_row[d];
        dot *= scale;
        if (mask_row != nullptr) dot += mask_row[j];

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            local_acc[d] = local_acc[d] * correction + weight * v_row[d];
        local_max = new_max;
    }

    float w_max = warp_reduce_max(local_max);
    float rescale = expf(local_max - w_max);
    float w_sum = warp_reduce_sum(local_sum * rescale);
    float w_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) w_acc[d] = local_acc[d] * rescale;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) w_acc[d] = warp_reduce_sum(w_acc[d]);

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_warp_acc[NUM_WARPS * HEAD_DIM];
    if (lane == 0) { s_warp_max[warp_id] = w_max; s_warp_sum[warp_id] = w_sum; }
    for (int d = lane; d < HEAD_DIM; d += 32) s_warp_acc[warp_id * HEAD_DIM + d] = w_acc[d];
    __syncthreads();

    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(
        s_warp_max, s_warp_sum, s_warp_acc, O, h, lane);
}

// =========================================================================
// Launch function for GQA decode
// =========================================================================

void launch_flash_attention_gqa_decode(const float* Q, const float* K, const float* V, float* O,
                                       int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                       const float* mask_row, cudaStream_t stream) {
    int blocks = num_heads;

#define LAUNCH_SMALL(HD, NW)              \
    flash_attn_gqa_decode_kernel<HD, NW>  \
        <<<blocks, 128, 0, stream>>>(Q, K, V, O, kv_len, num_heads, num_kv_heads, mask_row)

#define LAUNCH_LARGE(HD, NW)                           \
    flash_attn_gqa_decode_large_online_kernel<HD, NW>  \
        <<<blocks, (NW)*32, 0, stream>>>(Q, K, V, O, kv_len, num_heads, num_kv_heads, mask_row)

    switch (head_dim) {
    case 16:  LAUNCH_SMALL(16, 4);   break;
    case 32:  LAUNCH_SMALL(32, 4);   break;
    case 64:  LAUNCH_SMALL(64, 4);   break;
    case 96:  LAUNCH_SMALL(96, 4);   break;
    case 128: LAUNCH_SMALL(128, 4);  break;
    case 256: LAUNCH_LARGE(256, 4);  break;
    case 512: LAUNCH_LARGE(512, 4);  break;
    default:  fprintf(stderr, "[ERROR] flash_attn_gqa_decode: unsupported head_dim=%d\n", head_dim); break;
    }
#undef LAUNCH_SMALL
#undef LAUNCH_LARGE
}

}  // namespace cuda
}  // namespace forge
