#include <cmath>
#include <cstdio>

#include <mma.h>

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

// =========================================================================
// GQA Prefill kernel — VECTORIZED single-pass online softmax (Phase 2-C base)
//
// Same algorithm / numerics as flash_attn_gqa_kernel (Scalar), but:
//   - Q is loaded ONCE into shared memory (Q tile) and reused across all KV
//     positions. The Scalar kernel re-reads Q from global memory on every KV
//     iteration (q_len * kv_len redundant global fetches per block).
//   - K/V are loaded with float4 vectorized accesses (HEAD_DIM must be %4==0).
// The d-accumulation order is identical, so outputs are bit-exact vs Scalar.
// (Full KV-tiling to eliminate local_acc[HEAD_DIM] is deferred — see plan 2-C.)
// =========================================================================

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_attn_gqa_kernel_vectorized(const float* Q, const float* K, const float* V,
                                                  float* O, int q_len, int kv_len, int num_heads,
                                                  int num_kv_heads, const float* mask, bool causal) {
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
    constexpr int VEC_COUNT = HEAD_DIM / 4;
    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    // Q tile in shared memory — loaded cooperatively once, reused per KV position.
    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) s_q[d] = q_row[d];
    __syncthreads();

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        float m_val = (mask != nullptr) ? mask[qi * kv_len + j] : 0.0f;
        if (m_val < -1e20f) continue;
        if (!mask && causal && j > q_pos) continue;

        const float* k_row = K + j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        const float* v_row = V + j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;

        float dot = 0.0f;
#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 kv = *reinterpret_cast<const float4*>(k_row + vi * 4);
            float4 qv = *reinterpret_cast<const float4*>(s_q + vi * 4);
            dot += kv.x * qv.x + kv.y * qv.y + kv.z * qv.z + kv.w * qv.w;
        }
        dot *= scale;
        dot += m_val;

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

void launch_flash_attention_gqa_vectorized(const float* Q, const float* K, const float* V, float* O,
                                            int q_len, int kv_len, int num_heads, int num_kv_heads,
                                            int head_dim, const float* mask, bool causal,
                                            cudaStream_t stream) {
    dim3 grid(q_len, num_heads);
    int threads = 128;
#define LAUNCH(HD) flash_attn_gqa_kernel_vectorized<HD, 128><<<grid, threads, 0, stream>>>(Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads, mask, causal)
    switch (head_dim) {
    case 16:  LAUNCH(16);  break;
    case 32:  LAUNCH(32);  break;
    case 64:  LAUNCH(64);  break;
    case 72:  LAUNCH(72);  break;
    case 96:  LAUNCH(96);  break;
    case 128: LAUNCH(128); break;
    case 256: LAUNCH(256); break;
    case 512: LAUNCH(512); break;
    default:  fprintf(stderr, "[ERROR] flash_attn_gqa_vectorized: unsupported head_dim=%d\n", head_dim); break;
    }
#undef LAUNCH
}

// =========================================================================
// Streaming GQA Decode kernel (HEAD_DIM <= 128) — REFERENCE fast path.
// (bench output column "legacy" == this kernel; not an obsolete impl.)
// Per-thread register arrays + warp-level reduction; SINGLE-PASS online
// softmax with float4 vectorized QK/V access. Each thread owns a strided
// run of KV rows (j = tid; j < kv_len; j += block_size): warp-local
// streaming with no cross-warp KV sharing; a single __syncthreads happens
// only after the whole KV loop.
//
// Status (2026-09-05, RTX 4050 sm_89 + ptxas -v): REG 255 (cap) with real
// spill (628B st / 388B ld) at ~12% theoretical occupancy, yet it is the
// FASTEST decode here. On this workload decode is latency/instruction
// limited, NOT occupancy/DRAM limited — do not rewrite to tiled staging
// in the default path (see docs/optimization_plan_zh.md, Phase 2 收敛).
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

// =========================================================================
// 6-A-7: native FP16 streaming GQA decode (HEAD_DIM 16/32/64/96/128).
//
// Structural mirror of flash_attn_gqa_decode_kernel above: 1 block/head,
// Q in smem, streaming KV, single __syncthreads after the KV loop.
// ONLY the dtype boundary changes:
//   storage: __half Q/K/V in, __half O out
//   compute: __half2float on load, then FP32 dot/scale/online-softmax/PV,
//            __float2half on the final store (see the __half* overload of
//            cross_warp_merge_and_write).
// K/V loads stay 16B (float4-width = 8 halves); Q is read from the float
// smem tile. No half2, no tiling redesign, no TensorCore (Phase 2 iron rule:
// rank by measured latency, never by occupancy/registers).
// =========================================================================

template <int HEAD_DIM, int NUM_WARPS>
__global__ void flash_attn_gqa_decode_kernel_fp16(const __half* __restrict__ Q,
                                                  const __half* __restrict__ K,
                                                  const __half* __restrict__ V,
                                                  __half* __restrict__ O, int kv_len,
                                                  int num_heads, int num_kv_heads,
                                                  const __half* __restrict__ mask_row) {
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size)
        s_q[d] = __half2float(Q[h * HEAD_DIM + d]);
    __syncthreads();

    // Same scale convention as the FP32 kernels: 1/sqrt(HD), applied once to
    // the accumulated dot (score = dot(q,k) * rsqrt(head_dim)).
    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
    constexpr int VEC_COUNT = HEAD_DIM / 8;  // 8 halves per 16B transaction

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        float m_val = 0.0f;
        if (mask_row != nullptr) {
            m_val = __half2float(mask_row[j]);
            if (m_val < -1e20f) continue;
        }

        const __half* k_row = K + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        const __half* v_row = V + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;

        float dot = 0.0f;
#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 k8 = *reinterpret_cast<const float4*>(k_row + vi * 8);
            const __half* kh = reinterpret_cast<const __half*>(&k8);
            const float* qf = s_q + vi * 8;
            dot += __half2float(kh[0]) * qf[0] + __half2float(kh[1]) * qf[1] +
                   __half2float(kh[2]) * qf[2] + __half2float(kh[3]) * qf[3] +
                   __half2float(kh[4]) * qf[4] + __half2float(kh[5]) * qf[5] +
                   __half2float(kh[6]) * qf[6] + __half2float(kh[7]) * qf[7];
        }
        dot *= scale;
        dot += m_val;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 v8 = *reinterpret_cast<const float4*>(v_row + vi * 8);
            const __half* vh = reinterpret_cast<const __half*>(&v8);
            local_acc[vi * 8 + 0] = local_acc[vi * 8 + 0] * correction + weight * __half2float(vh[0]);
            local_acc[vi * 8 + 1] = local_acc[vi * 8 + 1] * correction + weight * __half2float(vh[1]);
            local_acc[vi * 8 + 2] = local_acc[vi * 8 + 2] * correction + weight * __half2float(vh[2]);
            local_acc[vi * 8 + 3] = local_acc[vi * 8 + 3] * correction + weight * __half2float(vh[3]);
            local_acc[vi * 8 + 4] = local_acc[vi * 8 + 4] * correction + weight * __half2float(vh[4]);
            local_acc[vi * 8 + 5] = local_acc[vi * 8 + 5] * correction + weight * __half2float(vh[5]);
            local_acc[vi * 8 + 6] = local_acc[vi * 8 + 6] * correction + weight * __half2float(vh[6]);
            local_acc[vi * 8 + 7] = local_acc[vi * 8 + 7] * correction + weight * __half2float(vh[7]);
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
// Phase 6-B-6: BF16-Q decode kernel (decode-first; prefill stays loud).
//
// Mixed storage contract (NOT "swap __half for __nv_bfloat16"): Q arrives
// BF16 (query projection output), K/V arrive FP32 (authoritative KV store --
// bf16_policy.kv stays FP32 until 6-B-6 decides otherwise), all math stays
// FP32 exactly as in the 6-A-7 kernel above (1/sqrt(HD) scale applied once,
// fused online softmax, FP32 PV), and only the final store converts
// (__float2bfloat16 via the __nv_bfloat16 overload of
// cross_warp_merge_and_write).
// Load widths follow the storage: Q is 8xbf16 per 16B (paired converts),
// K/V are 4xfloat per 16B (direct, no conversion). No bf162, no tiling
// redesign, no TensorCore (same iron rule as 6-A-7).
// =========================================================================

template <int HEAD_DIM, int NUM_WARPS>
__global__ void flash_attn_gqa_decode_kernel_bf16q(const __nv_bfloat16* __restrict__ Q,
                                                   const float* __restrict__ K,
                                                   const float* __restrict__ V,
                                                   __nv_bfloat16* __restrict__ O, int kv_len,
                                                   int num_heads, int num_kv_heads,
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
    for (int d = tid; d < HEAD_DIM; d += block_size)
        s_q[d] = __bfloat162float(Q[h * HEAD_DIM + d]);
    __syncthreads();

    // Same scale convention as the FP32/FP16 kernels: 1/sqrt(HD), applied once
    // to the accumulated dot.
    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
    constexpr int VEC_COUNT = HEAD_DIM / 4;  // 4 floats per 16B transaction

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        float m_val = 0.0f;
        if (mask_row != nullptr) {
            m_val = mask_row[j];
            if (m_val < -1e20f) continue;
        }

        const float* k_row = K + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        const float* v_row = V + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;

        float dot = 0.0f;
#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 k4 = *reinterpret_cast<const float4*>(k_row + vi * 4);
            const float* qf = s_q + vi * 4;
            dot += k4.x * qf[0] + k4.y * qf[1] + k4.z * qf[2] + k4.w * qf[3];
        }
        dot *= scale;
        dot += m_val;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 v4 = *reinterpret_cast<const float4*>(v_row + vi * 4);
            local_acc[vi * 4 + 0] = local_acc[vi * 4 + 0] * correction + weight * v4.x;
            local_acc[vi * 4 + 1] = local_acc[vi * 4 + 1] * correction + weight * v4.y;
            local_acc[vi * 4 + 2] = local_acc[vi * 4 + 2] * correction + weight * v4.z;
            local_acc[vi * 4 + 3] = local_acc[vi * 4 + 3] * correction + weight * v4.w;
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
// Phase 2 round 2: __half2 decode candidate.
//
// Same algorithm, same grid/block config, SAME FP32 math contract as the
// 6-A-7 baseline above (flash_attn_gqa_decode_kernel_fp16): FP16 storage,
// FP32 QK^T / online-softmax / PV, FP16 output. HFMA2 arithmetic is NOT
// used -- half2 multiply would round products to FP16 and break the locked
// contract. The candidate's changes are instruction-level only:
//   * K/V 16B chunks reinterpreted as __half2 pairs and converted via
//     __half22float2 (exact, paired) instead of 8x scalar __half2float;
//   * V-row 16B loads hoisted ahead of the QK dot math (overlap load
//     latency with compute);
//   * summation order identical to the baseline. NOTE: outputs are NOT
//     bitwise-equal -- FMA-contraction decisions differ between the two
//     compilations (baseline reads K/V from global; the candidate reads the
//     hoisted v8[] register array), so a few products round differently.
//     Measured deviation vs baseline is <= 1 FP16 output ULP; the test gates
//     on max_abs < 1e-3 (not bitwise) for this reason.
// Baseline stays untouched for A/B (Phase 2 iron rule: rank by measured
// latency, never by occupancy/registers).
// =========================================================================

template <int HEAD_DIM, int NUM_WARPS>
__global__ void flash_attn_gqa_decode_kernel_fp16_h2(const __half* __restrict__ Q,
                                                     const __half* __restrict__ K,
                                                     const __half* __restrict__ V,
                                                     __half* __restrict__ O, int kv_len,
                                                     int num_heads, int num_kv_heads,
                                                     const __half* __restrict__ mask_row) {
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size)
        s_q[d] = __half2float(Q[h * HEAD_DIM + d]);
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
    constexpr int VEC_COUNT = HEAD_DIM / 8;  // 8 halves per 16B transaction

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        float m_val = 0.0f;
        if (mask_row != nullptr) {
            m_val = __half2float(mask_row[j]);
            if (m_val < -1e20f) continue;
        }

        const __half* k_row = K + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        const __half* v_row = V + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;

        // Hoisted V loads: issue all 16B V transactions before the QK math so
        // their latency overlaps with the dot-product compute. Values are
        // converted later in the PV loop -- conversions are pure functions of
        // the input, so accumulation order and results are unchanged.
        float4 v8[VEC_COUNT];
#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi)
            v8[vi] = *reinterpret_cast<const float4*>(v_row + vi * 8);

        float dot = 0.0f;
#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 k8 = *reinterpret_cast<const float4*>(k_row + vi * 8);
            const __half2* k2 = reinterpret_cast<const __half2*>(&k8);
            float2 k01 = __half22float2(k2[0]);
            float2 k23 = __half22float2(k2[1]);
            float2 k45 = __half22float2(k2[2]);
            float2 k67 = __half22float2(k2[3]);
            const float* qf = s_q + vi * 8;
            // Same term order as the baseline: t0+t1+...+t7, then += dot.
            dot += k01.x * qf[0] + k01.y * qf[1] + k23.x * qf[2] + k23.y * qf[3] +
                   k45.x * qf[4] + k45.y * qf[5] + k67.x * qf[6] + k67.y * qf[7];
        }
        dot *= scale;
        dot += m_val;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            const __half2* v2 = reinterpret_cast<const __half2*>(&v8[vi]);
            float2 v01 = __half22float2(v2[0]);
            float2 v23 = __half22float2(v2[1]);
            float2 v45 = __half22float2(v2[2]);
            float2 v67 = __half22float2(v2[3]);
            local_acc[vi * 8 + 0] = local_acc[vi * 8 + 0] * correction + weight * v01.x;
            local_acc[vi * 8 + 1] = local_acc[vi * 8 + 1] * correction + weight * v01.y;
            local_acc[vi * 8 + 2] = local_acc[vi * 8 + 2] * correction + weight * v23.x;
            local_acc[vi * 8 + 3] = local_acc[vi * 8 + 3] * correction + weight * v23.y;
            local_acc[vi * 8 + 4] = local_acc[vi * 8 + 4] * correction + weight * v45.x;
            local_acc[vi * 8 + 5] = local_acc[vi * 8 + 5] * correction + weight * v45.y;
            local_acc[vi * 8 + 6] = local_acc[vi * 8 + 6] * correction + weight * v67.x;
            local_acc[vi * 8 + 7] = local_acc[vi * 8 + 7] * correction + weight * v67.y;
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
// 6-A-7.2: native FP16 vectorized GQA prefill.
//
// Mirror of flash_attn_gqa_kernel_vectorized above: Q tile in smem (as
// float), float4-width K/V loads (8 halves), FP32 dot/scale/online-softmax/
// PV, __half output store. local_acc[HEAD_DIM] is KEPT (no tiling redesign —
// 7.2 is "FP32 Vectorized kernel with only the dtype boundary replaced").
// No half2, no TensorCore.
// =========================================================================

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_attn_gqa_kernel_vectorized_fp16(const __half* Q, const __half* K,
                                                      const __half* V, __half* O, int q_len,
                                                      int kv_len, int num_heads, int num_kv_heads,
                                                      const __half* mask, bool causal) {
    int h = blockIdx.y;
    int qi = blockIdx.x;
    if (h >= num_heads || qi >= q_len) return;

    int q_pos = kv_len - q_len + qi;
    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    const __half* q_row = Q + (size_t)qi * num_heads * HEAD_DIM + h * HEAD_DIM;
    __half* o_row = O + (size_t)qi * num_heads * HEAD_DIM + h * HEAD_DIM;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int VEC_COUNT = HEAD_DIM / 8;  // 8 halves per 16B transaction
    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    // Q tile in shared memory — loaded cooperatively once, reused per KV position.
    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) s_q[d] = __half2float(q_row[d]);
    __syncthreads();

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        float m_val = (mask != nullptr) ? __half2float(mask[(size_t)qi * kv_len + j]) : 0.0f;
        if (m_val < -1e20f) continue;
        if (mask == nullptr && causal && j > q_pos) continue;

        const __half* k_row = K + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        const __half* v_row = V + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;

        float dot = 0.0f;
#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 k8 = *reinterpret_cast<const float4*>(k_row + vi * 8);
            const __half* kh = reinterpret_cast<const __half*>(&k8);
            const float* qf = s_q + vi * 8;
            dot += __half2float(kh[0]) * qf[0] + __half2float(kh[1]) * qf[1] +
                   __half2float(kh[2]) * qf[2] + __half2float(kh[3]) * qf[3] +
                   __half2float(kh[4]) * qf[4] + __half2float(kh[5]) * qf[5] +
                   __half2float(kh[6]) * qf[6] + __half2float(kh[7]) * qf[7];
        }
        dot *= scale;
        dot += m_val;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 v8 = *reinterpret_cast<const float4*>(v_row + vi * 8);
            const __half* vh = reinterpret_cast<const __half*>(&v8);
            local_acc[vi * 8 + 0] = local_acc[vi * 8 + 0] * correction + weight * __half2float(vh[0]);
            local_acc[vi * 8 + 1] = local_acc[vi * 8 + 1] * correction + weight * __half2float(vh[1]);
            local_acc[vi * 8 + 2] = local_acc[vi * 8 + 2] * correction + weight * __half2float(vh[2]);
            local_acc[vi * 8 + 3] = local_acc[vi * 8 + 3] * correction + weight * __half2float(vh[3]);
            local_acc[vi * 8 + 4] = local_acc[vi * 8 + 4] * correction + weight * __half2float(vh[4]);
            local_acc[vi * 8 + 5] = local_acc[vi * 8 + 5] * correction + weight * __half2float(vh[5]);
            local_acc[vi * 8 + 6] = local_acc[vi * 8 + 6] * correction + weight * __half2float(vh[6]);
            local_acc[vi * 8 + 7] = local_acc[vi * 8 + 7] * correction + weight * __half2float(vh[7]);
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

    // o_row points at this (qi,h) row: merge helper writes O[0*HD+d] relative
    // to the row pointer (same convention as the FP32 vectorized kernel).
    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(
        s_warp_max, s_warp_sum, s_warp_acc, o_row, 0, lane);
}

// =========================================================================
// 6-B-7.0: native BF16 vectorized prefill (E2E unlock; decode-only scope ends).
//
// Mirror of flash_attn_gqa_kernel_vectorized_fp16 above: Q tile in smem (as
// float via __bfloat162float), 16B K/V loads (8xbf16), FP32 dot/scale/online-
// softmax/PV, BF16 output store. local_acc[HEAD_DIM] is KEPT (no tiling
// redesign -- same "only the dtype boundary replaced" discipline as 6-A-7.2).
// No bf162, no TensorCore. Causal convention matches the FP32/FP16 kernels:
// q_pos = kv_len - q_len + qi.
// =========================================================================

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_attn_gqa_kernel_vectorized_bf16(const __nv_bfloat16* Q,
                                                      const __nv_bfloat16* K,
                                                      const __nv_bfloat16* V,
                                                      __nv_bfloat16* O, int q_len, int kv_len,
                                                      int num_heads, int num_kv_heads,
                                                      const __nv_bfloat16* mask, bool causal) {
    int h = blockIdx.y;
    int qi = blockIdx.x;
    if (h >= num_heads || qi >= q_len) return;

    int q_pos = kv_len - q_len + qi;
    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    const __nv_bfloat16* q_row = Q + (size_t)qi * num_heads * HEAD_DIM + h * HEAD_DIM;
    __nv_bfloat16* o_row = O + (size_t)qi * num_heads * HEAD_DIM + h * HEAD_DIM;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int VEC_COUNT = HEAD_DIM / 8;  // 8 bf16 per 16B transaction
    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    // Q tile in shared memory -- loaded cooperatively once, reused per KV position.
    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) s_q[d] = __bfloat162float(q_row[d]);
    __syncthreads();

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        float m_val = (mask != nullptr) ? __bfloat162float(mask[(size_t)qi * kv_len + j]) : 0.0f;
        if (m_val < -1e20f) continue;
        if (mask == nullptr && causal && j > q_pos) continue;

        const __nv_bfloat16* k_row = K + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        const __nv_bfloat16* v_row = V + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;

        float dot = 0.0f;
#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 k8 = *reinterpret_cast<const float4*>(k_row + vi * 8);
            const __nv_bfloat16* kh = reinterpret_cast<const __nv_bfloat16*>(&k8);
            const float* qf = s_q + vi * 8;
            dot += __bfloat162float(kh[0]) * qf[0] + __bfloat162float(kh[1]) * qf[1] +
                   __bfloat162float(kh[2]) * qf[2] + __bfloat162float(kh[3]) * qf[3] +
                   __bfloat162float(kh[4]) * qf[4] + __bfloat162float(kh[5]) * qf[5] +
                   __bfloat162float(kh[6]) * qf[6] + __bfloat162float(kh[7]) * qf[7];
        }
        dot *= scale;
        dot += m_val;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 v8 = *reinterpret_cast<const float4*>(v_row + vi * 8);
            const __nv_bfloat16* vh = reinterpret_cast<const __nv_bfloat16*>(&v8);
            local_acc[vi * 8 + 0] =
                local_acc[vi * 8 + 0] * correction + weight * __bfloat162float(vh[0]);
            local_acc[vi * 8 + 1] =
                local_acc[vi * 8 + 1] * correction + weight * __bfloat162float(vh[1]);
            local_acc[vi * 8 + 2] =
                local_acc[vi * 8 + 2] * correction + weight * __bfloat162float(vh[2]);
            local_acc[vi * 8 + 3] =
                local_acc[vi * 8 + 3] * correction + weight * __bfloat162float(vh[3]);
            local_acc[vi * 8 + 4] =
                local_acc[vi * 8 + 4] * correction + weight * __bfloat162float(vh[4]);
            local_acc[vi * 8 + 5] =
                local_acc[vi * 8 + 5] * correction + weight * __bfloat162float(vh[5]);
            local_acc[vi * 8 + 6] =
                local_acc[vi * 8 + 6] * correction + weight * __bfloat162float(vh[6]);
            local_acc[vi * 8 + 7] =
                local_acc[vi * 8 + 7] * correction + weight * __bfloat162float(vh[7]);
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

    // o_row points at this (qi,h) row: merge helper writes O[0*HD+d] relative
    // to the row pointer (same convention as the FP16 vectorized kernel).
    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(
        s_warp_max, s_warp_sum, s_warp_acc, o_row, 0, lane);
}

// =========================================================================
// Phase 2 round 3/4: TensorCore (wmma) prefill candidate.
//
// First MMA candidate: wmma 16x16x16 (f16 inputs, f32 accumulator) -- the
// official mma.sync wrapper, chosen over hand-written mma+ldmatrix to keep
// fragment-layout risk low while still measuring the REAL TensorCore lever.
// Round 4 parameterizes the tile shape for the config scan:
//   BR   in {32, 64}   query rows per block (warp owns BR/NUM_WARPS rows,
//                      must be a multiple of 16 => Br=32 & 4 warps invalid)
//   BC   in {16,32,64} KV rows per tile
//   NUM_WARPS in {2,4} blockDim = NUM_WARPS*32
// HEAD_DIM=64 only. Shared memory is carved from dynamic smem (configs with
// BR=64 & BC=64 need 66-68KB, above the 48KB static limit; sm_89 opt-in max
// is 99KB, both fit).
//
// Contract analysis (vs the locked 6-A-7 "QK^T / online-softmax / PV 全 FP32"):
//   * QK^T: f16 x f16 -> f32 mma. Each product of two f16 values is EXACT in
//     f32 (11+11 mantissa bits < 24), accumulation is f32 => same numerical
//     semantics as the baseline (different, still-f32 summation tree).
//   * online softmax: explicit FP32 CUDA C on the stored score tile
//     (scale / causal mask / max / exp / sum) -- unchanged.
//   * PV: mma requires f16 OPERANDS. P is stored to smem as f16 (the ONE new
//     rounding vs the baseline -- documented CONTRACT AMENDMENT; products
//     remain exact, accumulation stays f32).
//   * Output store: f16 (unchanged).
//
// Grid changes are inherent to MMA (M >= 16): one block now covers a BR-row
// query tile instead of the baseline's one-block-per-query-row.
// Bench/test-only entries; NOT wired into selector/dispatch.
// =========================================================================

namespace {

template <int HEAD_DIM, int BR, int BC, int NUM_WARPS>
__host__ __device__ constexpr size_t mma_smem_bytes() {
    return sizeof(__half) * (BR * HEAD_DIM)        // s_q
           + sizeof(__half) * (BC * HEAD_DIM) * 2  // s_k, s_v
           + sizeof(__half) * (BR * BC)            // s_p
           + sizeof(float) * (BR * BC)             // s_s (score tiles, f32)
           + sizeof(float) * (BR * HEAD_DIM)       // s_o (f32 O accumulator)
           + sizeof(float) * (NUM_WARPS * 256)     // s_oscratch (per warp 16x16)
           + sizeof(float) * BR * 2;               // s_row_max, s_row_sum
}

template <int HEAD_DIM, int BR, int BC, int NUM_WARPS>
struct MmaSmem {
    __half* q;
    __half* k;
    __half* v;
    __half* p;
    float* s;
    float* o;
    float* osc;
    float* rmax;
    float* rsum;

    static __device__ MmaSmem carve(unsigned char* base) {
        MmaSmem m;
        unsigned char* p = base;
        auto take = [&](size_t bytes) {
            void* r = p;
            p += bytes;
            return r;
        };
        m.q = static_cast<__half*>(take(sizeof(__half) * BR * HEAD_DIM));
        m.k = static_cast<__half*>(take(sizeof(__half) * BC * HEAD_DIM));
        m.v = static_cast<__half*>(take(sizeof(__half) * BC * HEAD_DIM));
        m.p = static_cast<__half*>(take(sizeof(__half) * BR * BC));
        m.s = static_cast<float*>(take(sizeof(float) * BR * BC));
        m.o = static_cast<float*>(take(sizeof(float) * BR * HEAD_DIM));
        m.osc = static_cast<float*>(take(sizeof(float) * NUM_WARPS * 256));
        m.rmax = static_cast<float*>(take(sizeof(float) * BR));
        m.rsum = static_cast<float*>(take(sizeof(float) * BR));
        return m;
    }
};

}  // namespace

// Device body shared by the static-smem and dynamic-smem kernel wrappers
// below. Round 4 finding: the same (32,32,2) config compiled against dynamic
// smem (extern base + carved pointers) measured ~20-25% slower at q=1024 than
// the round-3 static-smem original, while pure clock drift was ~3% -- so all
// configs that fit the 48KB static limit get the static-address path, and
// only (Br=64, Bc=64) pays the dynamic-smem tax.
template <int HEAD_DIM, int BR, int BC, int NUM_WARPS>
__device__ __forceinline__ void mma_prefill_body(const __half* __restrict__ Q,
                                                 const __half* __restrict__ K,
                                                 const __half* __restrict__ V,
                                                 __half* __restrict__ O, int q_len, int kv_len,
                                                 int num_heads, int num_kv_heads, bool causal,
                                                 const MmaSmem<HEAD_DIM, BR, BC, NUM_WARPS>& sm) {
    using namespace nvcuda;

    constexpr int M_FRAGS = BR / (NUM_WARPS * 16);  // Q fragments (16 rows) per warp
    constexpr int TPR_Q = NUM_WARPS * 32 / BR;      // threads per Q row (staging/store)
    constexpr int TPR_K = NUM_WARPS * 32 / BC;      // threads per K/V row (staging)
    constexpr int HD_CHUNKS = HEAD_DIM / 16;
    constexpr int SOFT_I = BC / 2;                  // score cols per softmax lane
    static_assert(HEAD_DIM % 16 == 0, "mma candidate requires HEAD_DIM % 16 == 0");
    static_assert(BR % (NUM_WARPS * 16) == 0, "each warp must own whole 16-row fragments");
    static_assert(BC % 16 == 0 && BC >= 16, "BC must be a multiple of 16");
    static_assert(BR <= NUM_WARPS * 32 && BC <= NUM_WARPS * 32,
                  "staging assumes at least one thread per tile row");

    const int h = blockIdx.y;
    if (h >= num_heads) return;
    const int q0 = blockIdx.x * BR;
    const int kv_groups = num_heads / num_kv_heads;
    const int kv_h = h / kv_groups;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    const int kv_dim = num_kv_heads * HEAD_DIM;

    // Stage Q tile (f16, raw storage -- exact). Rows >= q_len zero-filled.
    // Row assignment is same-warp (rows r of warp w go to threads of warp w),
    // so a __syncwarp() is enough before the collective q_frag load.
    {
        const int r = tid / TPR_Q;
        const int dc = (tid % TPR_Q) * (HEAD_DIM / TPR_Q);
        const __half* src = nullptr;
        if (q0 + r < q_len) src = Q + (size_t)(q0 + r) * num_heads * HEAD_DIM + h * HEAD_DIM;
        for (int i = 0; i < HEAD_DIM / TPR_Q; ++i)
            sm.q[r * HEAD_DIM + dc + i] = src ? src[dc + i] : __float2half(0.0f);
    }
    for (int i = tid; i < BR * HEAD_DIM; i += blockDim.x) sm.o[i] = 0.0f;
    if (tid < BR) {
        sm.rmax[tid] = -1e30f;
        sm.rsum[tid] = 0.0f;
    }
    __syncwarp();  // Q staging (same-warp rows) -> collective q_frag load below

    // Per-warp fragments: Q kept in registers across the whole kv loop.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag[M_FRAGS][HD_CHUNKS];
#pragma unroll
    for (int m = 0; m < M_FRAGS; ++m)
#pragma unroll
        for (int c = 0; c < HD_CHUNKS; ++c)
            wmma::load_matrix_sync(q_frag[m][c],
                                   sm.q + (warp * M_FRAGS * 16 + m * 16) * HEAD_DIM + c * 16,
                                   HEAD_DIM);

    const int num_tiles = (kv_len + BC - 1) / BC;
    for (int t = 0; t < num_tiles; ++t) {
        const int j0 = t * BC;
        // Stage K/V tile (zero-fill rows >= kv_len so P=0 cols cannot hit NaN).
        {
            const int r = tid / TPR_K;
            const int dc = (tid % TPR_K) * (HEAD_DIM / TPR_K);
            const int j = j0 + r;
            const __half* ks = nullptr;
            const __half* vs = nullptr;
            if (j < kv_len) {
                ks = K + (size_t)j * kv_dim + kv_h * HEAD_DIM;
                vs = V + (size_t)j * kv_dim + kv_h * HEAD_DIM;
            }
            for (int i = 0; i < HEAD_DIM / TPR_K; ++i) {
                sm.k[r * HEAD_DIM + dc + i] = ks ? ks[dc + i] : __float2half(0.0f);
                sm.v[r * HEAD_DIM + dc + i] = vs ? vs[dc + i] : __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- S = Q x K^T (TensorCore, f32 acc) -> per-warp score tile ----
        float* s_warp = sm.s + warp * (M_FRAGS * 16 * BC);
#pragma unroll
        for (int m = 0; m < M_FRAGS; ++m) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[BC / 16];
#pragma unroll
            for (int t2 = 0; t2 < BC / 16; ++t2) wmma::fill_fragment(s_frag[t2], 0.0f);
#pragma unroll
            for (int t2 = 0; t2 < BC / 16; ++t2) {
#pragma unroll
                for (int c = 0; c < HD_CHUNKS; ++c) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
                    // col_major B(k=d, n=j) at ptr[n*ld + k] == K_smem[j][d]
                    wmma::load_matrix_sync(k_frag, sm.k + t2 * 16 * HEAD_DIM + c * 16, HEAD_DIM);
                    wmma::mma_sync(s_frag[t2], q_frag[m][c], k_frag, s_frag[t2]);
                }
            }
#pragma unroll
            for (int t2 = 0; t2 < BC / 16; ++t2)
                wmma::store_matrix_sync(s_warp + m * 16 * BC + t2 * 16, s_frag[t2], BC,
                                        wmma::mem_row_major);
        }
        __syncwarp();

        // ---- online softmax (explicit FP32) + P store (f16) ----
        // lane pairs cover one score row: r_local = lane>>1; each lane holds
        // half the row (SOFT_I cols).
        {
            const int r_local = lane >> 1;
            const int half = lane & 1;
#pragma unroll
            for (int m = 0; m < M_FRAGS; ++m) {
                const int r = warp * M_FRAGS * 16 + m * 16 + r_local;
                const int qi = q0 + r;
                const int q_pos = kv_len - q_len + qi;
                const float* s_row = s_warp + m * 16 * BC + r_local * BC;

                float part_max = -INFINITY;
#pragma unroll
                for (int i = 0; i < SOFT_I; ++i) {
                    const int col = half * SOFT_I + i;
                    const int j = j0 + col;
                    bool valid = j < kv_len;
                    if (causal && j > q_pos) valid = false;
                    if (valid) part_max = fmaxf(part_max, s_row[col]);
                }
                const float tile_max_pair =
                    fmaxf(part_max, __shfl_xor_sync(0xFFFFFFFF, part_max, 1));
                const float prev_max = sm.rmax[r];
                const float new_max = fmaxf(prev_max, tile_max_pair);
                const float correction = expf(prev_max - new_max);  // exp(0)=1 when tile all masked

                float part_sum = 0.0f;
#pragma unroll
                for (int i = 0; i < SOFT_I; ++i) {
                    const int col = half * SOFT_I + i;
                    const int j = j0 + col;
                    bool valid = j < kv_len;
                    if (causal && j > q_pos) valid = false;
                    float p = 0.0f;
                    if (valid) {
                        const float sc = s_row[col] * scale;
                        p = expf(sc - new_max);
                    }
                    sm.p[r * BC + col] = __float2half(p);
                    part_sum += p;
                }
                const float tile_sum = part_sum + __shfl_xor_sync(0xFFFFFFFF, part_sum, 1);
                if (half == 0) {
                    sm.rmax[r] = new_max;
                    sm.rsum[r] = sm.rsum[r] * correction + tile_sum;
                }
                // rescale this row's accumulated O before adding the new PV term
                for (int i = 0; i < HEAD_DIM / 2; ++i)
                    sm.o[r * HEAD_DIM + half * (HEAD_DIM / 2) + i] *= correction;
            }
        }
        __syncwarp();

        // ---- O += P x V (TensorCore, f32 acc; P operand is f16) ----
#pragma unroll
        for (int m = 0; m < M_FRAGS; ++m) {
            const int r0 = warp * M_FRAGS * 16 + m * 16;
#pragma unroll
            for (int c = 0; c < HD_CHUNKS; ++c) {
#pragma unroll
                for (int p2 = 0; p2 < BC / 16; ++p2) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
                    wmma::load_matrix_sync(p_frag, sm.p + r0 * BC + p2 * 16, BC);
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
                    // B(k=j, n=d) at ptr[k*ld + n] == V_smem[j][d]
                    wmma::load_matrix_sync(v_frag, sm.v + p2 * 16 * HEAD_DIM + c * 16, HEAD_DIM);
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;
                    wmma::fill_fragment(o_frag, 0.0f);
                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                    wmma::store_matrix_sync(sm.osc + warp * 256, o_frag, 16, wmma::mem_row_major);
                    __syncwarp();
                    // add 16x16 scratch into this warp's O rows
                    const int r = lane >> 1;
                    const int col0 = (lane & 1) * 8;
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        sm.o[(r0 + r) * HEAD_DIM + c * 16 + col0 + i] +=
                            sm.osc[warp * 256 + r * 16 + col0 + i];
                }
            }
        }
        __syncthreads();  // K/V/P/s smem reused next tile
    }

    // ---- normalize + store (skip rows >= q_len) ----
    {
        const int r = tid / TPR_Q;
        const int dc = (tid % TPR_Q) * (HEAD_DIM / TPR_Q);
        const int qi = q0 + r;
        if (qi < q_len) {
            const float inv_sum = 1.0f / (sm.rsum[r] + 1e-30f);
            __half* dst = O + (size_t)qi * num_heads * HEAD_DIM + h * HEAD_DIM;
            for (int i = 0; i < HEAD_DIM / TPR_Q; ++i)
                dst[dc + i] = __float2half(sm.o[r * HEAD_DIM + dc + i] * inv_sum);
        }
    }
}

// Static-smem wrapper: compile-time known shared addresses (what the round-3
// kernel used). Requires the config's smem footprint to fit the 48KB static
// limit -- enforced by the launcher below.
template <int HEAD_DIM, int BR, int BC, int NUM_WARPS>
__global__ void flash_attn_gqa_kernel_mma_fp16(const __half* __restrict__ Q,
                                               const __half* __restrict__ K,
                                               const __half* __restrict__ V,
                                               __half* __restrict__ O, int q_len, int kv_len,
                                               int num_heads, int num_kv_heads, bool causal) {
    static_assert(mma_smem_bytes<HEAD_DIM, BR, BC, NUM_WARPS>() <= 48 * 1024,
                  "static-smem wrapper requires <= 48KB (use the dyn wrapper)");
    __shared__ __align__(16) __half s_q[BR * HEAD_DIM];
    __shared__ __align__(16) __half s_k[BC * HEAD_DIM];
    __shared__ __align__(16) __half s_v[BC * HEAD_DIM];
    __shared__ __align__(16) __half s_p[BR * BC];
    __shared__ __align__(16) float s_s[BR * BC];
    __shared__ __align__(16) float s_o[BR * HEAD_DIM];
    __shared__ __align__(16) float s_osc[NUM_WARPS * 256];
    __shared__ float s_rmax[BR];
    __shared__ float s_rsum[BR];
    const MmaSmem<HEAD_DIM, BR, BC, NUM_WARPS> sm = {s_q, s_k, s_v, s_p,
                                                     s_s, s_o, s_osc, s_rmax, s_rsum};
    mma_prefill_body<HEAD_DIM, BR, BC, NUM_WARPS>(Q, K, V, O, q_len, kv_len, num_heads,
                                                  num_kv_heads, causal, sm);
}

// Dynamic-smem wrapper: only for configs whose footprint exceeds the 48KB
// static limit ((Br=64, Bc=64): 66-68KB; sm_89 opt-in max is 99KB).
template <int HEAD_DIM, int BR, int BC, int NUM_WARPS>
__global__ void flash_attn_gqa_kernel_mma_fp16_dyn(const __half* __restrict__ Q,
                                                   const __half* __restrict__ K,
                                                   const __half* __restrict__ V,
                                                   __half* __restrict__ O, int q_len, int kv_len,
                                                   int num_heads, int num_kv_heads, bool causal) {
    extern __shared__ __align__(16) unsigned char forge_mma_smem_raw[];
    const MmaSmem<HEAD_DIM, BR, BC, NUM_WARPS> sm =
        MmaSmem<HEAD_DIM, BR, BC, NUM_WARPS>::carve(forge_mma_smem_raw);
    mma_prefill_body<HEAD_DIM, BR, BC, NUM_WARPS>(Q, K, V, O, q_len, kv_len, num_heads,
                                                  num_kv_heads, causal, sm);
}

// Config-dispatch helper: <=48KB configs launch the static-smem kernel; the
// (64,64) configs launch the dynamic-smem kernel with a one-time opt-in.
template <int HEAD_DIM, int BR, int BC, int NUM_WARPS>
void launch_mma_prefill_cfg_impl(const __half* Q, const __half* K, const __half* V, __half* O,
                                 int q_len, int kv_len, int num_heads, int num_kv_heads,
                                 bool causal, cudaStream_t stream) {
    constexpr size_t SMEM = mma_smem_bytes<HEAD_DIM, BR, BC, NUM_WARPS>();
    constexpr bool NEEDS_DYN = SMEM > 48 * 1024;
    static const bool smem_attr_ok = [] {
        if constexpr (NEEDS_DYN) {
            cudaError_t e = cudaFuncSetAttribute(
                flash_attn_gqa_kernel_mma_fp16_dyn<HEAD_DIM, BR, BC, NUM_WARPS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(SMEM));
            if (e != cudaSuccess)
                fprintf(stderr,
                        "[ERROR] mma prefill cfg: cudaFuncSetAttribute(%zu bytes) failed: %s\n",
                        SMEM, cudaGetErrorString(e));
            return e == cudaSuccess;
        } else {
            return true;
        }
    }();
    (void)smem_attr_ok;
    dim3 grid((q_len + BR - 1) / BR, num_heads);
    if constexpr (NEEDS_DYN) {
        flash_attn_gqa_kernel_mma_fp16_dyn<HEAD_DIM, BR, BC, NUM_WARPS>
            <<<grid, NUM_WARPS * 32, SMEM, stream>>>(Q, K, V, O, q_len, kv_len, num_heads,
                                                     num_kv_heads, causal);
    } else {
        flash_attn_gqa_kernel_mma_fp16<HEAD_DIM, BR, BC, NUM_WARPS>
            <<<grid, NUM_WARPS * 32, 0, stream>>>(Q, K, V, O, q_len, kv_len, num_heads,
                                                  num_kv_heads, causal);
    }
}

// Phase 2 round 3: wmma prefill candidate, round-3 default shape
// (Br=32/Bc=32/2 warps). Round 5 promoted the wmma route into the production
// entry launch_flash_attention_gqa_prefill_fp16 with the round-4 winning
// config; this entry remains for the round-3/4 measurement history.
// HEAD_DIM=64 only (the 6-A shape family);
// other dims and arbitrary masks are loudly rejected.
void launch_flash_attention_gqa_prefill_mma_fp16(const __half* Q, const __half* K,
                                                 const __half* V, __half* O, int q_len,
                                                 int kv_len, int num_heads, int num_kv_heads,
                                                 int head_dim, bool causal, cudaStream_t stream) {
    if (head_dim != 64) {
        fprintf(stderr,
                "[ERROR] flash_attn_gqa_prefill_mma_fp16: candidate supports head_dim=64 only "
                "(got %d)\n",
                head_dim);
        return;
    }
    launch_mma_prefill_cfg_impl<64, 32, 32, 2>(Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads,
                                               causal, stream);
}

// Phase 2 round 4: explicit-config scan entry (same
// restrictions as above). Invalid combos are loudly rejected on the host.
// wmma M=16 excludes Br=32 & 4 warps (8 rows/warp); the 4-warp axis is
// covered by Br=64. The scan winner (64,16,4) is wired into the production
// entry launch_flash_attention_gqa_prefill_fp16 since round 5.
void launch_flash_attention_gqa_prefill_mma_cfg(const __half* Q, const __half* K, const __half* V,
                                                __half* O, int q_len, int kv_len, int num_heads,
                                                int num_kv_heads, int head_dim, bool causal,
                                                int br, int bc, int warps,
                                                cudaStream_t stream) {
    if (head_dim != 64) {
        fprintf(stderr,
                "[ERROR] flash_attn_gqa_prefill_mma_cfg: supports head_dim=64 only (got %d)\n",
                head_dim);
        return;
    }
#define FORGE_MMA_CFG(BR_, BC_, NW_)                                                      \
    if (br == BR_ && bc == BC_ && warps == NW_) {                                         \
        launch_mma_prefill_cfg_impl<64, BR_, BC_, NW_>(Q, K, V, O, q_len, kv_len,         \
                                                       num_heads, num_kv_heads, causal,   \
                                                       stream);                           \
        return;                                                                           \
    }
    FORGE_MMA_CFG(32, 16, 2)
    FORGE_MMA_CFG(32, 32, 2)
    FORGE_MMA_CFG(32, 64, 2)
    FORGE_MMA_CFG(64, 16, 2)
    FORGE_MMA_CFG(64, 32, 2)
    FORGE_MMA_CFG(64, 64, 2)
    FORGE_MMA_CFG(64, 16, 4)
    FORGE_MMA_CFG(64, 32, 4)
    FORGE_MMA_CFG(64, 64, 4)
#undef FORGE_MMA_CFG
    fprintf(stderr,
            "[ERROR] flash_attn_gqa_prefill_mma_cfg: unsupported config Br=%d Bc=%d warps=%d "
            "(scan set: Br{32,64} x Bc{16,32,64} x warps{2,4}, minus Br=32&4w)\n",
            br, bc, warps);
}

// 6-A-7.2: the vectorized FP16 prefill kernel, now reachable through an
// explicit entry. Phase 2 round 5 promoted the wmma prefill into the
// production entry below (hd=64, causal, no mask -- exactly the round-3/4
// gated scope); this entry keeps the frozen baseline measurable and serves
// as the fallback for every shape/mask combination outside that scope.
void launch_flash_attention_gqa_prefill_fp16_vectorized(const __half* Q, const __half* K,
                                                        const __half* V, __half* O, int q_len,
                                                        int kv_len, int num_heads,
                                                        int num_kv_heads, int head_dim,
                                                        const __half* mask, bool causal,
                                                        cudaStream_t stream) {
    dim3 grid(q_len, num_heads);
    int threads = 128;
#define LAUNCH_FP16_PREFILL(HD) flash_attn_gqa_kernel_vectorized_fp16<HD, 128> \
    <<<grid, threads, 0, stream>>>(Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads, mask, causal)
    switch (head_dim) {
    case 16:  LAUNCH_FP16_PREFILL(16);  break;
    case 32:  LAUNCH_FP16_PREFILL(32);  break;
    case 64:  LAUNCH_FP16_PREFILL(64);  break;
    case 96:  LAUNCH_FP16_PREFILL(96);  break;
    case 128: LAUNCH_FP16_PREFILL(128); break;
    // 6-A-7 scope: larger dims rejected host-side, never silently reinterpreted.
    default:  fprintf(stderr, "[ERROR] flash_attn_gqa_prefill_fp16: unsupported head_dim=%d (6-A-7: 16/32/64/96/128 only)\n", head_dim); break;
    }
#undef LAUNCH_FP16_PREFILL
}

// 6-B-7.0: native BF16 vectorized prefill launch (E2E unlock). All-BF16 QKV,
// FP32 math, BF16 output, causal convention identical to the FP16 entry.
// HD 16/32/64/96/128; larger dims rejected host-side, never fallback.
// No wmma/TensorCore (decode-track parity: correctness first, tuning later).
void launch_flash_attention_gqa_prefill_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                             const __nv_bfloat16* V, __nv_bfloat16* O, int q_len,
                                             int kv_len, int num_heads, int num_kv_heads,
                                             int head_dim, const __nv_bfloat16* mask, bool causal,
                                             cudaStream_t stream) {
    dim3 grid(q_len, num_heads);
    int threads = 128;
#define LAUNCH_BF16_PREFILL(HD) flash_attn_gqa_kernel_vectorized_bf16<HD, 128> \
    <<<grid, threads, 0, stream>>>(Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads, mask, causal)
    switch (head_dim) {
    case 16:  LAUNCH_BF16_PREFILL(16);  break;
    case 32:  LAUNCH_BF16_PREFILL(32);  break;
    case 64:  LAUNCH_BF16_PREFILL(64);  break;
    case 96:  LAUNCH_BF16_PREFILL(96);  break;
    case 128: LAUNCH_BF16_PREFILL(128); break;
    // 6-B-7 scope: larger dims rejected host-side, never silently reinterpreted.
    default:  fprintf(stderr, "[ERROR] flash_attn_gqa_prefill_bf16: unsupported head_dim=%d (6-B-7: 16/32/64/96/128 only)\n", head_dim); break;
    }
#undef LAUNCH_BF16_PREFILL
}

// Phase 2 round 5 (production): wmma prefill promoted per the round-4 scan
// (Br=64/Bc=16/4 warps: q=64 2.44x, q=256 2.35x, q=1024 1.45x vs this entry's
// vectorized fallback, best-of-5 interleaved A/B). Contract amendment vs the
// vectorized baseline: PV's P operand stored f16 (documented); QK^T products
// stay exact in f32. Dispatch scope is exactly the gated scope -- hd=64,
// causal, no mask; anything else (other head dims, additive masks, non-causal)
// falls back to the 6-A-7.2 vectorized kernel unchanged.
void launch_flash_attention_gqa_prefill_fp16(const __half* Q, const __half* K, const __half* V,
                                             __half* O, int q_len, int kv_len, int num_heads,
                                             int num_kv_heads, int head_dim, const __half* mask,
                                             bool causal, cudaStream_t stream) {
    if (head_dim == 64 && causal && mask == nullptr) {
        launch_mma_prefill_cfg_impl<64, 64, 16, 4>(Q, K, V, O, q_len, kv_len, num_heads,
                                                   num_kv_heads, causal, stream);
        return;
    }
    launch_flash_attention_gqa_prefill_fp16_vectorized(Q, K, V, O, q_len, kv_len, num_heads,
                                                       num_kv_heads, head_dim, mask, causal,
                                                       stream);
}

void launch_flash_attention_gqa_decode_fp16(const __half* Q, const __half* K, const __half* V,
                                            __half* O, int kv_len, int num_heads,
                                            int num_kv_heads, int head_dim,
                                            const __half* mask_row_or_null,
                                            cudaStream_t stream) {
    int blocks = num_heads;

#define LAUNCH_FP16(HD, NW)              \
    flash_attn_gqa_decode_kernel_fp16<HD, NW>  \
        <<<blocks, 128, 0, stream>>>(Q, K, V, O, kv_len, num_heads, num_kv_heads, mask_row_or_null)

    switch (head_dim) {
    case 16:  LAUNCH_FP16(16, 4);   break;
    case 32:  LAUNCH_FP16(32, 4);   break;
    case 64:  LAUNCH_FP16(64, 4);   break;
    case 96:  LAUNCH_FP16(96, 4);   break;
    case 128: LAUNCH_FP16(128, 4);  break;
    // 6-A-7 scope: 256/512 loudly rejected host-side (op_kernels / executor),
    // never silently reinterpreted. This fprintf is unreachable defense-in-depth.
    default:  fprintf(stderr, "[ERROR] flash_attn_gqa_decode_fp16: unsupported head_dim=%d (6-A-7: 16/32/64/96/128 only)\n", head_dim); break;
    }
#undef LAUNCH_FP16
}

// =========================================================================
// Phase 6-B-6.7: all-BF16 decode kernel (BF16 KV extension; decode-only).
//
// Storage contract: Q/K/V all __nv_bfloat16 (BF16 authoritative KV -- the
// 6-B-6 decode slice proved KV traffic is the binding variable, so the KV
// side now stores BF16 too), all math FP32 exactly as the kernels above,
// output BF16. Structural twin of flash_attn_gqa_decode_kernel_bf16q except
// the K/V/mask loads convert (__bfloat162float) instead of reading float
// direct; 16B = 8xbf16 on every side. No bf162, no tiling redesign, no
// TensorCore.
// =========================================================================

template <int HEAD_DIM, int NUM_WARPS>
__global__ void flash_attn_gqa_decode_kernel_bf16(const __nv_bfloat16* __restrict__ Q,
                                                  const __nv_bfloat16* __restrict__ K,
                                                  const __nv_bfloat16* __restrict__ V,
                                                  __nv_bfloat16* __restrict__ O, int kv_len,
                                                  int num_heads, int num_kv_heads,
                                                  const __nv_bfloat16* __restrict__ mask_row) {
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size)
        s_q[d] = __bfloat162float(Q[h * HEAD_DIM + d]);
    __syncthreads();

    // Same scale convention as the FP32/FP16/BF16-Q kernels: 1/sqrt(HD).
    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
    constexpr int VEC_COUNT = HEAD_DIM / 8;  // 8 bf16 per 16B transaction

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        float m_val = 0.0f;
        if (mask_row != nullptr) {
            m_val = __bfloat162float(mask_row[j]);
            if (m_val < -1e20f) continue;
        }

        const __nv_bfloat16* k_row = K + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        const __nv_bfloat16* v_row = V + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;

        float dot = 0.0f;
#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 k8 = *reinterpret_cast<const float4*>(k_row + vi * 8);
            const __nv_bfloat16* kh = reinterpret_cast<const __nv_bfloat16*>(&k8);
            const float* qf = s_q + vi * 8;
            dot += __bfloat162float(kh[0]) * qf[0] + __bfloat162float(kh[1]) * qf[1] +
                   __bfloat162float(kh[2]) * qf[2] + __bfloat162float(kh[3]) * qf[3] +
                   __bfloat162float(kh[4]) * qf[4] + __bfloat162float(kh[5]) * qf[5] +
                   __bfloat162float(kh[6]) * qf[6] + __bfloat162float(kh[7]) * qf[7];
        }
        dot *= scale;
        dot += m_val;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 v8 = *reinterpret_cast<const float4*>(v_row + vi * 8);
            const __nv_bfloat16* vh = reinterpret_cast<const __nv_bfloat16*>(&v8);
            local_acc[vi * 8 + 0] =
                local_acc[vi * 8 + 0] * correction + weight * __bfloat162float(vh[0]);
            local_acc[vi * 8 + 1] =
                local_acc[vi * 8 + 1] * correction + weight * __bfloat162float(vh[1]);
            local_acc[vi * 8 + 2] =
                local_acc[vi * 8 + 2] * correction + weight * __bfloat162float(vh[2]);
            local_acc[vi * 8 + 3] =
                local_acc[vi * 8 + 3] * correction + weight * __bfloat162float(vh[3]);
            local_acc[vi * 8 + 4] =
                local_acc[vi * 8 + 4] * correction + weight * __bfloat162float(vh[4]);
            local_acc[vi * 8 + 5] =
                local_acc[vi * 8 + 5] * correction + weight * __bfloat162float(vh[5]);
            local_acc[vi * 8 + 6] =
                local_acc[vi * 8 + 6] * correction + weight * __bfloat162float(vh[6]);
            local_acc[vi * 8 + 7] =
                local_acc[vi * 8 + 7] * correction + weight * __bfloat162float(vh[7]);
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

// Phase 6-B-6.7: all-BF16 decode launch (BF16 KV extension; decode-only).
// Q/K/V all __nv_bfloat16, FP32 math, BF16 output. HD 16/32/64/96/128;
// larger dims rejected host-side, never fallback.
void launch_flash_attention_gqa_decode_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                            const __nv_bfloat16* V, __nv_bfloat16* O, int kv_len,
                                            int num_heads, int num_kv_heads, int head_dim,
                                            const __nv_bfloat16* mask_row_or_null,
                                            cudaStream_t stream) {
    int blocks = num_heads;

#define LAUNCH_BF16(HD, NW)              \
    flash_attn_gqa_decode_kernel_bf16<HD, NW>  \
        <<<blocks, 128, 0, stream>>>(Q, K, V, O, kv_len, num_heads, num_kv_heads, mask_row_or_null)

    switch (head_dim) {
    case 16:  LAUNCH_BF16(16, 4);   break;
    case 32:  LAUNCH_BF16(32, 4);   break;
    case 64:  LAUNCH_BF16(64, 4);   break;
    case 96:  LAUNCH_BF16(96, 4);   break;
    case 128: LAUNCH_BF16(128, 4);  break;
    // Same scope as the BF16-Q launch: 256/512 loudly rejected host-side.
    // This fprintf is unreachable defense-in-depth.
    default:  fprintf(stderr, "[ERROR] flash_attn_gqa_decode_bf16: unsupported head_dim=%d (6-B-6.7: 16/32/64/96/128 only)\n", head_dim); break;
    }
#undef LAUNCH_BF16
}

// Phase 6-B-6: BF16-Q decode launch (decode-first production path for BF16
// queries over the authoritative FP32 KV store). HD 16/32/64/96/128; larger
// dims rejected host-side (op_kernels / executor), never fallback.
void launch_flash_attention_gqa_decode_bf16q(const __nv_bfloat16* Q, const float* K, const float* V,
                                             __nv_bfloat16* O, int kv_len, int num_heads,
                                             int num_kv_heads, int head_dim,
                                             const float* mask_row_or_null,
                                             cudaStream_t stream) {
    int blocks = num_heads;

#define LAUNCH_BF16Q(HD, NW)              \
    flash_attn_gqa_decode_kernel_bf16q<HD, NW>  \
        <<<blocks, 128, 0, stream>>>(Q, K, V, O, kv_len, num_heads, num_kv_heads, mask_row_or_null)

    switch (head_dim) {
    case 16:  LAUNCH_BF16Q(16, 4);   break;
    case 32:  LAUNCH_BF16Q(32, 4);   break;
    case 64:  LAUNCH_BF16Q(64, 4);   break;
    case 96:  LAUNCH_BF16Q(96, 4);   break;
    case 128: LAUNCH_BF16Q(128, 4);  break;
    // 6-B-6 scope mirrors 6-A-7: 256/512 loudly rejected host-side, never
    // silently reinterpreted. This fprintf is unreachable defense-in-depth.
    default:  fprintf(stderr, "[ERROR] flash_attn_gqa_decode_bf16q: unsupported head_dim=%d (6-B-6: 16/32/64/96/128 only)\n", head_dim); break;
    }
#undef LAUNCH_BF16Q
}

// Phase 2 round 2: __half2 decode candidate launch. Bench/test-only entry --
// NOT wired into the selector or dispatch; the frozen 6-A-7 baseline stays
// the production path until this candidate wins a measured A/B.
void launch_flash_attention_gqa_decode_fp16_h2(const __half* Q, const __half* K,
                                               const __half* V, __half* O, int kv_len,
                                               int num_heads, int num_kv_heads, int head_dim,
                                               const __half* mask_row_or_null,
                                               cudaStream_t stream) {
    int blocks = num_heads;

#define LAUNCH_FP16_H2(HD, NW)              \
    flash_attn_gqa_decode_kernel_fp16_h2<HD, NW>  \
        <<<blocks, 128, 0, stream>>>(Q, K, V, O, kv_len, num_heads, num_kv_heads, mask_row_or_null)

    switch (head_dim) {
    case 16:  LAUNCH_FP16_H2(16, 4);   break;
    case 32:  LAUNCH_FP16_H2(32, 4);   break;
    case 64:  LAUNCH_FP16_H2(64, 4);   break;
    case 96:  LAUNCH_FP16_H2(96, 4);   break;
    case 128: LAUNCH_FP16_H2(128, 4);  break;
    default:  fprintf(stderr, "[ERROR] flash_attn_gqa_decode_fp16_h2: unsupported head_dim=%d (candidate mirrors 6-A-7: 16/32/64/96/128 only)\n", head_dim); break;
    }
#undef LAUNCH_FP16_H2
}

}  // namespace cuda
}  // namespace forge
