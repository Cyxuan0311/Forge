// =========================================================================
// cuda_fused_attn.cu — Fused Flash Attention Kernel for Quantized KV Cache
//
// 设计目标：在 decode 阶段直接读取 Q4_0/F16/Q8_0 量化的 KV cache，
// 在 kernel 内部按需反量化，避免 dequantize_layer() 把整个 KV cache
// 复制成 FP32（节省 FP32 带宽开销）。
//
// 参考：llama.cpp 的 fattn-vec.cuh 模板化 <D, ncols, type_K, type_V>
// =========================================================================

#include "cuda_common.h"
#include "forge/cuda_kernels.h"

#include <cstdio>

namespace forge {
namespace cuda {

// =========================================================================
// 共用：Q4_0 块布局常量
//   fp16 d (2B) + 16 packed bytes (32 nibbles) = 18B per 32 elements
//   qs[i] low  nibble  → element i      (0..15)
//   qs[i] high nibble  → element i + 16 (16..31)
// =========================================================================

// =========================================================================
// Q4_0 融合 GQA Decode Kernel（HEAD_DIM <= 128）
// 沿用 flash_attn_gqa_decode_kernel 结构：
//   - 每个 block 处理一个 query head
//   - 每个线程处理若干 KV 位置，使用 local_acc[HEAD_DIM] 寄存器数组
//   - 在线 softmax（per-thread），warp/block 归约
// 区别：K/V 直接读取 Q4_0 块，按需反量化
// =========================================================================

template <int HEAD_DIM, int NUM_WARPS>
__global__ void fused_attn_q4_0_decode_kernel(
    const float* __restrict__ Q,
    const uint8_t* __restrict__ q_K,
    const uint8_t* __restrict__ q_V,
    float* __restrict__ O,
    int kv_len, int num_heads, int num_kv_heads,
    size_t q_row_size,
    const float* __restrict__ mask_row) {

    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    constexpr int BLOCK_ELEMS = 32;
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int NUM_BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_ELEMS;

    // 当前 kv_h 在 KV 行中的字节偏移（每个 head 占 NUM_BLOCKS_PER_HEAD 个块）
    const size_t head_byte_offset = (size_t)kv_h * NUM_BLOCKS_PER_HEAD * Q4_0_BLOCK_SIZE;

    // 将 Q 加载到共享内存（每个 head 一份）
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
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    // 主循环：每个线程处理 stride=block_size 的若干 KV 位置
    for (int j = tid; j < kv_len; j += block_size) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const uint8_t* k_row = q_K + (size_t)j * q_row_size + head_byte_offset;
        const uint8_t* v_row = q_V + (size_t)j * q_row_size + head_byte_offset;

        // ---- 计算 Q·K 点积（K 在线反量化）----
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
        if (mask_row != nullptr) dot += mask_row[j];

        // ---- 在线 softmax 更新 ----
        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

        // ---- 累加加权 V（V 在线反量化）----
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
                local_acc[idx_lo] =
                    local_acc[idx_lo] * correction + weight * (val_lo * v_scale);
                local_acc[idx_hi] =
                    local_acc[idx_hi] * correction + weight * (val_hi * v_scale);
            }
        }
        local_max = new_max;
    }

    // ---- Warp 内归约 max ----
    float warp_max = local_max;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
    }
    warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0);

    // 用 warp_max 对本线程的 sum/acc 做 rescale
    float rescale = expf(local_max - warp_max);
    float warp_sum = local_sum * rescale;
    float warp_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) warp_acc[d] = local_acc[d] * rescale;

    // ---- Warp 内归约 sum ----
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
    }
    warp_sum = __shfl_sync(0xFFFFFFFF, warp_sum, 0);

    // ---- Warp 内归约 acc[d] ----
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        float val = warp_acc[d];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        warp_acc[d] = __shfl_sync(0xFFFFFFFF, val, 0);
    }

    // ---- 跨 Warp 归约（共享内存）----
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
        for (int d = 0; d < HEAD_DIM; ++d) global_acc[d] = 0.0f;

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

// =========================================================================
// Q4_0 融合 GQA Decode Kernel（HEAD_DIM = 256, 512）
// 沿用 flash_attn_gqa_decode_large_kernel 的两阶段 tiled 方案：
//   Pass 1: 计算 per-thread max（在线 softmax）
//   Cross-thread max 归约（共享内存）
//   Pass 2: 用 global max 重新计算 Q·K，加权 V 累加到 s_acc（atomicAdd）
//   最终归一化输出
// =========================================================================

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void fused_attn_q4_0_decode_large_kernel(
    const float* __restrict__ Q,
    const uint8_t* __restrict__ q_K,
    const uint8_t* __restrict__ q_V,
    float* __restrict__ O,
    int kv_len, int num_heads, int num_kv_heads,
    size_t q_row_size,
    const float* __restrict__ mask_row) {

    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;
    int tid = threadIdx.x;

    constexpr int BLOCK_ELEMS = 32;
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int NUM_BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_ELEMS;

    const size_t head_byte_offset = (size_t)kv_h * NUM_BLOCKS_PER_HEAD * Q4_0_BLOCK_SIZE;

    __shared__ float s_q[HEAD_DIM];
    __shared__ float s_acc[HEAD_DIM];
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum;

    // 加载 Q
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        s_q[d] = Q[h * HEAD_DIM + d];
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    // ---- Pass 1: 找全局 max ----
    float local_max = -1e30f;
    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const uint8_t* k_row = q_K + (size_t)j * q_row_size + head_byte_offset;

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
                dot += s_q[bi * 32 + i] * (val_lo * k_scale);
                dot += s_q[bi * 32 + i + 16] * (val_hi * k_scale);
            }
        }
        dot *= scale;
        if (mask_row != nullptr) dot += mask_row[j];
        local_max = fmaxf(local_max, dot);
    }

    // 归约 max
    s_max[tid] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        __syncthreads();
    }
    float global_max = s_max[0];

    // ---- Pass 2: 累加加权 V ----
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) s_acc[d] = 0.0f;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const uint8_t* k_row = q_K + (size_t)j * q_row_size + head_byte_offset;
        const uint8_t* v_row = q_V + (size_t)j * q_row_size + head_byte_offset;

        // 重新计算 Q·K
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
                dot += s_q[bi * 32 + i] * (val_lo * k_scale);
                dot += s_q[bi * 32 + i + 16] * (val_hi * k_scale);
            }
        }
        dot *= scale;
        if (mask_row != nullptr) dot += mask_row[j];

        float weight = expf(dot - global_max);
        local_sum += weight;

        // 累加加权 V（在线反量化，atomicAdd 到共享内存）
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
                atomicAdd(&s_acc[bi * 32 + i], weight * (val_lo * v_scale));
                atomicAdd(&s_acc[bi * 32 + i + 16], weight * (val_hi * v_scale));
            }
        }
    }

    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    // 归一化输出
    float inv_sum = 1.0f / (s_sum + 1e-30f);
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        O[h * HEAD_DIM + d] = s_acc[d] * inv_sum;
    }
}

// =========================================================================
// Launch function: Q4_0 融合 GQA Decode
// =========================================================================

void launch_fused_flash_attention_gqa_decode_q4_0(
    const float* Q, const void* q_K, const void* q_V, float* O,
    int kv_len, int num_heads, int num_kv_heads, int head_dim,
    size_t q_row_size, const float* mask_row,
    cudaStream_t stream) {

    int blocks = num_heads;

#define LAUNCH_FUSED_Q4_0_DECODE(HD, NW)                                        \
    fused_attn_q4_0_decode_kernel<HD, NW>                                       \
        <<<blocks, 128, 0, stream>>>(Q, static_cast<const uint8_t*>(q_K),       \
                                     static_cast<const uint8_t*>(q_V), O,       \
                                     kv_len, num_heads, num_kv_heads,           \
                                     q_row_size, mask_row)

#define LAUNCH_FUSED_Q4_0_DECODE_LARGE(HD)                                       \
    fused_attn_q4_0_decode_large_kernel<HD, 256>                                 \
        <<<blocks, 256, 0, stream>>>(Q, static_cast<const uint8_t*>(q_K),        \
                                     static_cast<const uint8_t*>(q_V), O,        \
                                     kv_len, num_heads, num_kv_heads,            \
                                     q_row_size, mask_row)

    switch (head_dim) {
    case 64:  LAUNCH_FUSED_Q4_0_DECODE(64, 4);  break;
    case 96:  LAUNCH_FUSED_Q4_0_DECODE(96, 4);  break;
    case 128: LAUNCH_FUSED_Q4_0_DECODE(128, 4); break;
    case 256: LAUNCH_FUSED_Q4_0_DECODE_LARGE(256); break;
    case 512: LAUNCH_FUSED_Q4_0_DECODE_LARGE(512); break;
    default:
        fprintf(stderr,
                "[ERROR] fused_attn_q4_0_decode: unsupported head_dim=%d\n",
                head_dim);
        break;
    }
#undef LAUNCH_FUSED_Q4_0_DECODE
#undef LAUNCH_FUSED_Q4_0_DECODE_LARGE
}

// =========================================================================
// F16 融合 GQA Decode Kernel（HEAD_DIM <= 128）
// K/V 以 half 格式存储（每元素 2 字节），无块结构
// =========================================================================

template <int HEAD_DIM, int NUM_WARPS>
__global__ void fused_attn_f16_decode_kernel(
    const float* __restrict__ Q,
    const uint8_t* __restrict__ q_K,
    const uint8_t* __restrict__ q_V,
    float* __restrict__ O,
    int kv_len, int num_heads, int num_kv_heads,
    size_t q_row_size,
    const float* __restrict__ mask_row) {

    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // 当前 kv_h 在 KV 行中的字节偏移
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
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const __half* k_row = reinterpret_cast<const __half*>(
            q_K + (size_t)j * q_row_size + head_byte_offset);
        const __half* v_row = reinterpret_cast<const __half*>(
            q_V + (size_t)j * q_row_size + head_byte_offset);

        // Q·K 点积（F16 在线反量化）
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_q[d] * __half2float(k_row[d]);
        }
        dot *= scale;
        if (mask_row != nullptr) dot += mask_row[j];

        // 在线 softmax
        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

        // 累加加权 V
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            local_acc[d] = local_acc[d] * correction + weight * __half2float(v_row[d]);
        }
        local_max = new_max;
    }

    // ---- Warp 内归约 max ----
    float warp_max = local_max;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
    }
    warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0);

    float rescale = expf(local_max - warp_max);
    float warp_sum = local_sum * rescale;
    float warp_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) warp_acc[d] = local_acc[d] * rescale;

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

    // ---- 跨 Warp 归约 ----
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
        for (int d = 0; d < HEAD_DIM; ++d) global_acc[d] = 0.0f;

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

// =========================================================================
// F16 融合 GQA Decode Kernel（HEAD_DIM = 256, 512）
// 两阶段 tiled 方案
// =========================================================================

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void fused_attn_f16_decode_large_kernel(
    const float* __restrict__ Q,
    const uint8_t* __restrict__ q_K,
    const uint8_t* __restrict__ q_V,
    float* __restrict__ O,
    int kv_len, int num_heads, int num_kv_heads,
    size_t q_row_size,
    const float* __restrict__ mask_row) {

    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;
    int tid = threadIdx.x;

    const size_t head_byte_offset = (size_t)kv_h * HEAD_DIM * sizeof(__half);

    __shared__ float s_q[HEAD_DIM];
    __shared__ float s_acc[HEAD_DIM];
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum;

    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        s_q[d] = Q[h * HEAD_DIM + d];
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    // Pass 1: 找全局 max
    float local_max = -1e30f;
    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const __half* k_row = reinterpret_cast<const __half*>(
            q_K + (size_t)j * q_row_size + head_byte_offset);

        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_q[d] * __half2float(k_row[d]);
        }
        dot *= scale;
        if (mask_row != nullptr) dot += mask_row[j];
        local_max = fmaxf(local_max, dot);
    }

    s_max[tid] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        __syncthreads();
    }
    float global_max = s_max[0];

    // Pass 2: 累加加权 V
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) s_acc[d] = 0.0f;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const __half* k_row = reinterpret_cast<const __half*>(
            q_K + (size_t)j * q_row_size + head_byte_offset);
        const __half* v_row = reinterpret_cast<const __half*>(
            q_V + (size_t)j * q_row_size + head_byte_offset);

        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_q[d] * __half2float(k_row[d]);
        }
        dot *= scale;
        if (mask_row != nullptr) dot += mask_row[j];

        float weight = expf(dot - global_max);
        local_sum += weight;

#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            atomicAdd(&s_acc[d], weight * __half2float(v_row[d]));
        }
    }

    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    float inv_sum = 1.0f / (s_sum + 1e-30f);
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        O[h * HEAD_DIM + d] = s_acc[d] * inv_sum;
    }
}

// =========================================================================
// Launch function: F16 融合 GQA Decode
// =========================================================================

void launch_fused_flash_attention_gqa_decode_f16(
    const float* Q, const void* q_K, const void* q_V, float* O,
    int kv_len, int num_heads, int num_kv_heads, int head_dim,
    size_t q_row_size, const float* mask_row,
    cudaStream_t stream) {

    int blocks = num_heads;

#define LAUNCH_FUSED_F16_DECODE(HD, NW)                                         \
    fused_attn_f16_decode_kernel<HD, NW>                                        \
        <<<blocks, 128, 0, stream>>>(Q, static_cast<const uint8_t*>(q_K),       \
                                     static_cast<const uint8_t*>(q_V), O,       \
                                     kv_len, num_heads, num_kv_heads,           \
                                     q_row_size, mask_row)

#define LAUNCH_FUSED_F16_DECODE_LARGE(HD)                                        \
    fused_attn_f16_decode_large_kernel<HD, 256>                                  \
        <<<blocks, 256, 0, stream>>>(Q, static_cast<const uint8_t*>(q_K),        \
                                     static_cast<const uint8_t*>(q_V), O,        \
                                     kv_len, num_heads, num_kv_heads,            \
                                     q_row_size, mask_row)

    switch (head_dim) {
    case 64:  LAUNCH_FUSED_F16_DECODE(64, 4);  break;
    case 96:  LAUNCH_FUSED_F16_DECODE(96, 4);  break;
    case 128: LAUNCH_FUSED_F16_DECODE(128, 4); break;
    case 256: LAUNCH_FUSED_F16_DECODE_LARGE(256); break;
    case 512: LAUNCH_FUSED_F16_DECODE_LARGE(512); break;
    default:
        fprintf(stderr,
                "[ERROR] fused_attn_f16_decode: unsupported head_dim=%d\n",
                head_dim);
        break;
    }
#undef LAUNCH_FUSED_F16_DECODE
#undef LAUNCH_FUSED_F16_DECODE_LARGE
}

// =========================================================================
// Q8_0 融合 GQA Decode Kernel（HEAD_DIM <= 128）
// Block: fp16 d (2B) + int8 qs[32] (32B) = 34B per 32 elements
// =========================================================================

template <int HEAD_DIM, int NUM_WARPS>
__global__ void fused_attn_q8_0_decode_kernel(
    const float* __restrict__ Q,
    const uint8_t* __restrict__ q_K,
    const uint8_t* __restrict__ q_V,
    float* __restrict__ O,
    int kv_len, int num_heads, int num_kv_heads,
    size_t q_row_size,
    const float* __restrict__ mask_row) {

    int h = blockIdx.x;
    if (h >= num_heads) return;

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
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const uint8_t* k_row = q_K + (size_t)j * q_row_size + head_byte_offset;
        const uint8_t* v_row = q_V + (size_t)j * q_row_size + head_byte_offset;

        // Q·K 点积（Q8_0 在线反量化）
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
        if (mask_row != nullptr) dot += mask_row[j];

        // 在线 softmax
        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

        // 累加加权 V（Q8_0 在线反量化）
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

    // ---- Warp 内归约 max ----
    float warp_max = local_max;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
    }
    warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0);

    float rescale = expf(local_max - warp_max);
    float warp_sum = local_sum * rescale;
    float warp_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) warp_acc[d] = local_acc[d] * rescale;

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

    // ---- 跨 Warp 归约 ----
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
        for (int d = 0; d < HEAD_DIM; ++d) global_acc[d] = 0.0f;

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

// =========================================================================
// Q8_0 融合 GQA Decode Kernel（HEAD_DIM = 256, 512）
// =========================================================================

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void fused_attn_q8_0_decode_large_kernel(
    const float* __restrict__ Q,
    const uint8_t* __restrict__ q_K,
    const uint8_t* __restrict__ q_V,
    float* __restrict__ O,
    int kv_len, int num_heads, int num_kv_heads,
    size_t q_row_size,
    const float* __restrict__ mask_row) {

    int h = blockIdx.x;
    if (h >= num_heads) return;

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;
    int tid = threadIdx.x;

    constexpr int BLOCK_ELEMS = 32;
    constexpr int Q8_0_BLOCK_SIZE = 34;
    constexpr int NUM_BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_ELEMS;

    const size_t head_byte_offset = (size_t)kv_h * NUM_BLOCKS_PER_HEAD * Q8_0_BLOCK_SIZE;

    __shared__ float s_q[HEAD_DIM];
    __shared__ float s_acc[HEAD_DIM];
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum;

    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        s_q[d] = Q[h * HEAD_DIM + d];
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    // Pass 1: 找全局 max
    float local_max = -1e30f;
    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const uint8_t* k_row = q_K + (size_t)j * q_row_size + head_byte_offset;

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
        if (mask_row != nullptr) dot += mask_row[j];
        local_max = fmaxf(local_max, dot);
    }

    s_max[tid] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        __syncthreads();
    }
    float global_max = s_max[0];

    // Pass 2: 累加加权 V
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) s_acc[d] = 0.0f;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < kv_len; j += BLOCK_SIZE) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const uint8_t* k_row = q_K + (size_t)j * q_row_size + head_byte_offset;
        const uint8_t* v_row = q_V + (size_t)j * q_row_size + head_byte_offset;

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
        if (mask_row != nullptr) dot += mask_row[j];

        float weight = expf(dot - global_max);
        local_sum += weight;

#pragma unroll
        for (int bi = 0; bi < NUM_BLOCKS_PER_HEAD; ++bi) {
            const uint8_t* block_ptr = v_row + bi * Q8_0_BLOCK_SIZE;
            uint16_t scale_bits;
            memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
            float v_scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
            const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + sizeof(__half));

#pragma unroll
            for (int i = 0; i < BLOCK_ELEMS; ++i) {
                atomicAdd(&s_acc[bi * 32 + i], weight * (qs[i] * v_scale));
            }
        }
    }

    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    float inv_sum = 1.0f / (s_sum + 1e-30f);
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        O[h * HEAD_DIM + d] = s_acc[d] * inv_sum;
    }
}

// =========================================================================
// Launch function: Q8_0 融合 GQA Decode
// =========================================================================

void launch_fused_flash_attention_gqa_decode_q8_0(
    const float* Q, const void* q_K, const void* q_V, float* O,
    int kv_len, int num_heads, int num_kv_heads, int head_dim,
    size_t q_row_size, const float* mask_row,
    cudaStream_t stream) {

    int blocks = num_heads;

#define LAUNCH_FUSED_Q8_0_DECODE(HD, NW)                                        \
    fused_attn_q8_0_decode_kernel<HD, NW>                                       \
        <<<blocks, 128, 0, stream>>>(Q, static_cast<const uint8_t*>(q_K),       \
                                     static_cast<const uint8_t*>(q_V), O,       \
                                     kv_len, num_heads, num_kv_heads,           \
                                     q_row_size, mask_row)

#define LAUNCH_FUSED_Q8_0_DECODE_LARGE(HD)                                       \
    fused_attn_q8_0_decode_large_kernel<HD, 256>                                 \
        <<<blocks, 256, 0, stream>>>(Q, static_cast<const uint8_t*>(q_K),        \
                                     static_cast<const uint8_t*>(q_V), O,        \
                                     kv_len, num_heads, num_kv_heads,            \
                                     q_row_size, mask_row)

    switch (head_dim) {
    case 64:  LAUNCH_FUSED_Q8_0_DECODE(64, 4);  break;
    case 96:  LAUNCH_FUSED_Q8_0_DECODE(96, 4);  break;
    case 128: LAUNCH_FUSED_Q8_0_DECODE(128, 4); break;
    case 256: LAUNCH_FUSED_Q8_0_DECODE_LARGE(256); break;
    case 512: LAUNCH_FUSED_Q8_0_DECODE_LARGE(512); break;
    default:
        fprintf(stderr,
                "[ERROR] fused_attn_q8_0_decode: unsupported head_dim=%d\n",
                head_dim);
        break;
    }
#undef LAUNCH_FUSED_Q8_0_DECODE
#undef LAUNCH_FUSED_Q8_0_DECODE_LARGE
}

}  // namespace cuda
}  // namespace forge
