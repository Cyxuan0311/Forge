#include <cmath>
#include <cstdio>

#include "cuda_flash_attn.h"

namespace forge {
namespace cuda {

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_attn_kernel(const float* Q, const float* K, const float* V, float* O,
                                  int q_len, int kv_len, int num_heads, bool causal) {
    int h = blockIdx.y;
    int qi = blockIdx.x;

    if (h >= num_heads || qi >= q_len)
        return;

    int q_pos = kv_len - q_len + qi;

    const float* q_row = Q + qi * num_heads * HEAD_DIM + h * HEAD_DIM;
    float* o_row = O + qi * num_heads * HEAD_DIM + h * HEAD_DIM;

    __shared__ float s_sum;
    __shared__ float s_acc[HEAD_DIM];

    float thread_max = -1e30f;

    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        if (causal && j > q_pos)
            continue;
        float dot = 0.0f;
        const float* k_row = K + j * num_heads * HEAD_DIM + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += q_row[d] * k_row[d];
        }
        dot *= 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
        thread_max = fmaxf(thread_max, dot);
    }

    __shared__ float s_block_max[BLOCK_SIZE];
    s_block_max[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_block_max[threadIdx.x] =
                fmaxf(s_block_max[threadIdx.x], s_block_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = s_block_max[0];

    float thread_sum = 0.0f;
    float thread_acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d)
        thread_acc[d] = 0.0f;

    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        if (causal && j > q_pos)
            continue;
        float dot = 0.0f;
        const float* k_row = K + j * num_heads * HEAD_DIM + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += q_row[d] * k_row[d];
        }
        dot *= 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

        float weight = expf(dot - max_val);
        thread_sum += weight;

        const float* v_row = V + j * num_heads * HEAD_DIM + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) {
            thread_acc[d] += weight * v_row[d];
        }
    }

    if (threadIdx.x == 0) {
        s_sum = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d)
            s_acc[d] = 0.0f;
    }
    __syncthreads();

    atomicAdd(&s_sum, thread_sum);
    for (int d = 0; d < HEAD_DIM; ++d) {
        atomicAdd(&s_acc[d], thread_acc[d]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float inv_sum = 1.0f / (s_sum + 1e-30f);
        for (int d = 0; d < HEAD_DIM; ++d) {
            o_row[d] = s_acc[d] * inv_sum;
        }
    }
}

void launch_flash_attention(const float* Q, const float* K, const float* V, float* O, int q_len,
                            int kv_len, int num_heads, int head_dim, bool causal,
                            cudaStream_t stream) {
    dim3 grid(q_len, num_heads);
    int threads = 128;

#define LAUNCH_FLASH_ATTN(HD)  \
    flash_attn_kernel<HD, 128> \
        <<<grid, threads, 0, stream>>>(Q, K, V, O, q_len, kv_len, num_heads, causal)

    switch (head_dim) {
    case 16:
        LAUNCH_FLASH_ATTN(16);
        break;
    case 32:
        LAUNCH_FLASH_ATTN(32);
        break;
    case 64:
        LAUNCH_FLASH_ATTN(64);
        break;
    case 72:
        LAUNCH_FLASH_ATTN(72);
        break;
    case 96:
        LAUNCH_FLASH_ATTN(96);
        break;
    case 128:
        LAUNCH_FLASH_ATTN(128);
        break;
    case 256:
        LAUNCH_FLASH_ATTN(256);
        break;
    default:
        fprintf(stderr, "[ERROR] flash_attn_kernel: unsupported head_dim=%d\n", head_dim);
        break;
    }
#undef LAUNCH_FLASH_ATTN
}

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_attn_gqa_kernel(const float* Q, const float* K, const float* V, float* O,
                                      int q_len, int kv_len, int num_heads, int num_kv_heads,
                                      bool causal) {
    int h = blockIdx.y;
    int qi = blockIdx.x;

    if (h >= num_heads || qi >= q_len)
        return;

    int q_pos = kv_len - q_len + qi;
    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    const float* q_row = Q + qi * num_heads * HEAD_DIM + h * HEAD_DIM;
    float* o_row = O + qi * num_heads * HEAD_DIM + h * HEAD_DIM;

    __shared__ float s_sum;
    __shared__ float s_acc[HEAD_DIM];

    float thread_max = -1e30f;

    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        if (causal && j > q_pos)
            continue;
        float dot = 0.0f;
        const float* k_row = K + j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += q_row[d] * k_row[d];
        }
        dot *= 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
        thread_max = fmaxf(thread_max, dot);
    }

    __shared__ float s_block_max[BLOCK_SIZE];
    s_block_max[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_block_max[threadIdx.x] =
                fmaxf(s_block_max[threadIdx.x], s_block_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = s_block_max[0];

    float thread_sum = 0.0f;
    float thread_acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d)
        thread_acc[d] = 0.0f;

    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        if (causal && j > q_pos)
            continue;
        float dot = 0.0f;
        const float* k_row = K + j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += q_row[d] * k_row[d];
        }
        dot *= 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

        float weight = expf(dot - max_val);
        thread_sum += weight;

        const float* v_row = V + j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) {
            thread_acc[d] += weight * v_row[d];
        }
    }

    if (threadIdx.x == 0) {
        s_sum = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d)
            s_acc[d] = 0.0f;
    }
    __syncthreads();

    atomicAdd(&s_sum, thread_sum);
    for (int d = 0; d < HEAD_DIM; ++d) {
        atomicAdd(&s_acc[d], thread_acc[d]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float inv_sum = 1.0f / (s_sum + 1e-30f);
        for (int d = 0; d < HEAD_DIM; ++d) {
            o_row[d] = s_acc[d] * inv_sum;
        }
    }
}

void launch_flash_attention_gqa(const float* Q, const float* K, const float* V, float* O, int q_len,
                                int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                bool causal, cudaStream_t stream) {
    dim3 grid(q_len, num_heads);
    int threads = 128;

#define LAUNCH_FLASH_ATTN_GQA(HD)  \
    flash_attn_gqa_kernel<HD, 128> \
        <<<grid, threads, 0, stream>>>(Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads, causal)

    switch (head_dim) {
    case 16:
        LAUNCH_FLASH_ATTN_GQA(16);
        break;
    case 32:
        LAUNCH_FLASH_ATTN_GQA(32);
        break;
    case 64:
        LAUNCH_FLASH_ATTN_GQA(64);
        break;
    case 72:
        LAUNCH_FLASH_ATTN_GQA(72);
        break;
    case 96:
        LAUNCH_FLASH_ATTN_GQA(96);
        break;
    case 128:
        LAUNCH_FLASH_ATTN_GQA(128);
        break;
    case 256:
        LAUNCH_FLASH_ATTN_GQA(256);
        break;
    default:
        fprintf(stderr, "[ERROR] flash_attn_gqa_kernel: unsupported head_dim=%d\n", head_dim);
        break;
    }
#undef LAUNCH_FLASH_ATTN_GQA
}

template <int HEAD_DIM, int NUM_WARPS>
__global__ void flash_attn_gqa_decode_kernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V, float* __restrict__ O,
                                             int kv_len, int num_heads, int num_kv_heads) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size) {
        s_q[d] = Q[h * HEAD_DIM + d];
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
    constexpr int VEC_SIZE = 4;
    constexpr int VEC_COUNT = HEAD_DIM / VEC_SIZE;

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        const float* k_row = K + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;
        const float* v_row = V + (size_t)j * num_kv_heads * HEAD_DIM + kv_h * HEAD_DIM;

        float dot = 0.0f;
#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 kv = *reinterpret_cast<const float4*>(k_row + vi * VEC_SIZE);
            float4 qv = *reinterpret_cast<const float4*>(s_q + vi * VEC_SIZE);
            dot += kv.x * qv.x + kv.y * qv.y + kv.z * qv.z + kv.w * qv.w;
        }
        dot *= scale;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);

        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int vi = 0; vi < VEC_COUNT; ++vi) {
            float4 vf = *reinterpret_cast<const float4*>(v_row + vi * VEC_SIZE);
            local_acc[vi * VEC_SIZE + 0] =
                local_acc[vi * VEC_SIZE + 0] * correction + weight * vf.x;
            local_acc[vi * VEC_SIZE + 1] =
                local_acc[vi * VEC_SIZE + 1] * correction + weight * vf.y;
            local_acc[vi * VEC_SIZE + 2] =
                local_acc[vi * VEC_SIZE + 2] * correction + weight * vf.z;
            local_acc[vi * VEC_SIZE + 3] =
                local_acc[vi * VEC_SIZE + 3] * correction + weight * vf.w;
        }
        local_max = new_max;
    }

    float warp_max = local_max;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
    }
    warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0);

    float rescale = expf(local_max - warp_max);
    float warp_sum = local_sum * rescale;
    float warp_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        warp_acc[d] = local_acc[d] * rescale;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
    }
    warp_sum = __shfl_sync(0xFFFFFFFF, warp_sum, 0);

#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        float val = warp_acc[d];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        warp_acc[d] = __shfl_sync(0xFFFFFFFF, val, 0);
    }

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

    if (warp_id == 0) {
        float global_max = -1e30f;
        for (int w = 0; w < NUM_WARPS; ++w) {
            global_max = fmaxf(global_max, s_warp_max[w]);
        }

        float global_sum = 0.0f;
        float global_acc[HEAD_DIM];
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            global_acc[d] = 0.0f;

        for (int w = 0; w < NUM_WARPS; ++w) {
            float corr = expf(s_warp_max[w] - global_max);
            global_sum += s_warp_sum[w] * corr;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                global_acc[d] += s_warp_acc[w * HEAD_DIM + d] * corr;
            }
        }

        float inv_sum = 1.0f / (global_sum + 1e-30f);
        for (int d = lane; d < HEAD_DIM; d += 32) {
            O[h * HEAD_DIM + d] = global_acc[d] * inv_sum;
        }
    }
}

void launch_flash_attention_gqa_decode(const float* Q, const float* K, const float* V, float* O,
                                       int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                       cudaStream_t stream) {
    int threads = 128;
    int blocks = num_heads;

#define LAUNCH_DECODE(HD, NW)            \
    flash_attn_gqa_decode_kernel<HD, NW> \
        <<<blocks, threads, 0, stream>>>(Q, K, V, O, kv_len, num_heads, num_kv_heads)

    switch (head_dim) {
    case 64:
        LAUNCH_DECODE(64, 4);
        break;
    case 96:
        LAUNCH_DECODE(96, 4);
        break;
    case 128:
        LAUNCH_DECODE(128, 4);
        break;
    default:
        fprintf(stderr, "[ERROR] flash_attn_gqa_decode: unsupported head_dim=%d\n", head_dim);
        break;
    }
#undef LAUNCH_DECODE
}

}  // namespace cuda
}  // namespace forge
