#include "cuda_common.h"
#include "cuda_elementwise.h"
#include "cuda_fused.h"
#include "cuda_quant.h"
#include "cuda_gemv_tmpl.cuh"

namespace forge {
namespace cuda {

// ---- FFN Up Fused Q4_0 (shared memory, M=1, decode) ----
// Fuses gate + up projections: out = SiLU(x @ w1^T) * (x @ w3^T)
// with shared memory to reduce x vector reads from 2x to 1x.
// x vector is loaded once into shared memory, then reused for both
// gate and up weight rows.

__global__ void ffn_up_fused_q4_0_kernel(const float* __restrict__ x,
                                         const uint8_t* __restrict__ q_w1,
                                         const uint8_t* __restrict__ q_w3, float* __restrict__ out,
                                         int K, int N) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    extern __shared__ float smem_x[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const float4* x_vec = reinterpret_cast<const float4*>(x);
    int num_float4 = K / 4;
    for (int i = tid; i < num_float4; i += block_size) {
        float4 val = x_vec[i];
        smem_x[i * 4 + 0] = val.x;
        smem_x[i * 4 + 1] = val.y;
        smem_x[i * 4 + 2] = val.z;
        smem_x[i * 4 + 3] = val.w;
    }
    for (int i = num_float4 * 4 + tid; i < K; i += block_size) {
        smem_x[i] = x[i];
    }
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    const uint8_t* w1_row = q_w1 + (size_t)warp_id * num_blocks_row * Q4_0_BLOCK_SIZE;
    const uint8_t* w3_row = q_w3 + (size_t)warp_id * num_blocks_row * Q4_0_BLOCK_SIZE;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int bi = lane; bi < num_blocks_row; bi += 32) {
        int base = bi * BLOCK_ELEMS;

        const uint8_t* w1_block = w1_row + bi * Q4_0_BLOCK_SIZE;
        uint16_t w1_scale_bits;
        memcpy(&w1_scale_bits, w1_block, sizeof(uint16_t));
        float w1_scale = __half2float(reinterpret_cast<const __half&>(w1_scale_bits));
        const uint8_t* w1_qs = w1_block + sizeof(uint16_t);

        const uint8_t* w3_block = w3_row + bi * Q4_0_BLOCK_SIZE;
        uint16_t w3_scale_bits;
        memcpy(&w3_scale_bits, w3_block, sizeof(uint16_t));
        float w3_scale = __half2float(reinterpret_cast<const __half&>(w3_scale_bits));
        const uint8_t* w3_qs = w3_block + sizeof(uint16_t);

#pragma unroll
        for (int j = 0; j < BLOCK_ELEMS; ++j) {
            if (base + j >= K)
                break;
            float x_val = smem_x[base + j];
            gate_sum += x_val * (static_cast<float>(q4_unpack(w1_qs, j)) * w1_scale);
            up_sum += x_val * (static_cast<float>(q4_unpack(w3_qs, j)) * w3_scale);
        }
    }

    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 16);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 8);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 4);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 2);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 1);

    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 16);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 8);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 4);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 2);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 1);

    if (lane == 0) {
        float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        out[warp_id] = silu_gate * up_sum;
    }
}

void launch_ffn_up_fused_q4_0(const float* x, const void* q_w1, const void* q_w3, float* out, int K,
                              int intermediate_dim, cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    size_t smem_bytes = K * sizeof(float);
    ffn_up_fused_q4_0_kernel<<<blocks, threads, smem_bytes, stream>>>(
        x, static_cast<const uint8_t*>(q_w1), static_cast<const uint8_t*>(q_w3), out, K,
        intermediate_dim);
}

// ---- FFN Up Fused Q4_0 Batch GEMV (M > 1, small batch) ----

__global__ void ffn_up_fused_q4_0_batch_gemv_kernel(const float* __restrict__ x,
                                                    const uint8_t* __restrict__ q_w1,
                                                    const uint8_t* __restrict__ q_w3,
                                                    float* __restrict__ out, int M, int K, int N) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int total_warps = M * N;
    if (warp_id >= total_warps)
        return;

    int m = warp_id / N;
    int n = warp_id % N;

    const float* x_row = x + m * K;
    const uint8_t* w1_row = q_w1 + n * num_blocks_row * Q4_0_BLOCK_SIZE;
    const uint8_t* w3_row = q_w3 + n * num_blocks_row * Q4_0_BLOCK_SIZE;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int bi = lane; bi < num_blocks_row; bi += 32) {
        int base = bi * BLOCK_ELEMS;

        const uint8_t* w1_block = w1_row + bi * Q4_0_BLOCK_SIZE;
        uint16_t w1_scale_bits;
        memcpy(&w1_scale_bits, w1_block, sizeof(uint16_t));
        float w1_scale = __half2float(reinterpret_cast<const __half&>(w1_scale_bits));
        const uint8_t* w1_qs = w1_block + sizeof(uint16_t);

        const uint8_t* w3_block = w3_row + bi * Q4_0_BLOCK_SIZE;
        uint16_t w3_scale_bits;
        memcpy(&w3_scale_bits, w3_block, sizeof(uint16_t));
        float w3_scale = __half2float(reinterpret_cast<const __half&>(w3_scale_bits));
        const uint8_t* w3_qs = w3_block + sizeof(uint16_t);

#pragma unroll
        for (int j = 0; j < BLOCK_ELEMS; ++j) {
            if (base + j >= K)
                break;
            float x_val = x_row[base + j];
            gate_sum += x_val * (static_cast<float>(q4_unpack(w1_qs, j)) * w1_scale);
            up_sum += x_val * (static_cast<float>(q4_unpack(w3_qs, j)) * w3_scale);
        }
    }

    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 16);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 8);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 4);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 2);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 1);

    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 16);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 8);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 4);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 2);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 1);

    if (lane == 0) {
        float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        out[m * N + n] = silu_gate * up_sum;
    }
}

void launch_ffn_up_fused_q4_0_batch_gemv(const float* x, const void* q_w1, const void* q_w3,
                                         float* out, int M, int K, int N, cudaStream_t stream) {
    int total_warps = M * N;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q4_0_batch_gemv_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_w1), static_cast<const uint8_t*>(q_w3), out, M, K, N);
}

// ---- FFN Up Fused Q4_0 Batch (M > 1, prefill, dequant + cublas) ----

void launch_ffn_up_fused_q4_0_batch(const float* x, const void* q_w1, const void* q_w3, float* out,
                                    int M, int K, int intermediate_dim, cudaStream_t stream) {
    size_t fp32_bytes = (size_t)intermediate_dim * K * sizeof(float);
    size_t result_bytes = (size_t)M * intermediate_dim * sizeof(float);
    size_t total_bytes = fp32_bytes * 2 + result_bytes * 2;

    float* base = static_cast<float*>(scratch_pool().ensure(total_bytes));
    float* w1_fp32 = base;
    float* w3_fp32 = w1_fp32 + intermediate_dim * K;
    float* gate_buf = w3_fp32 + intermediate_dim * K;
    float* up_buf = gate_buf + M * intermediate_dim;

    launch_dequant_q4_0_matrix(q_w1, w1_fp32, intermediate_dim, K, stream);
    launch_dequant_q4_0_matrix(q_w3, w3_fp32, intermediate_dim, K, stream);

    launch_cublas_sgemm(x, w1_fp32, gate_buf, M, K, intermediate_dim, true, stream);
    launch_cublas_sgemm(x, w3_fp32, up_buf, M, K, intermediate_dim, true, stream);

    launch_silu_multiply(gate_buf, up_buf, out, M * intermediate_dim, stream);
}

// ---- FFN Up Fused Q4_K Batch (M > 1, prefill) ----

void launch_ffn_up_fused_q4_k_batch(const float* x, const void* q_w1, const void* q_w3, float* out,
                                    int M, int K, int intermediate_dim, cudaStream_t stream) {
    size_t fp32_bytes = (size_t)intermediate_dim * K * sizeof(float);
    size_t result_bytes = (size_t)M * intermediate_dim * sizeof(float);
    size_t total_bytes = fp32_bytes * 2 + result_bytes * 2;

    float* base = static_cast<float*>(scratch_pool().ensure(total_bytes));
    float* w1_fp32 = base;
    float* w3_fp32 = w1_fp32 + intermediate_dim * K;
    float* gate_buf = w3_fp32 + intermediate_dim * K;
    float* up_buf = gate_buf + M * intermediate_dim;

    launch_dequant_q4_k_matrix(q_w1, w1_fp32, intermediate_dim, K, stream);
    launch_dequant_q4_k_matrix(q_w3, w3_fp32, intermediate_dim, K, stream);

    launch_cublas_sgemm(x, w1_fp32, gate_buf, M, K, intermediate_dim, true, stream);
    launch_cublas_sgemm(x, w3_fp32, up_buf, M, K, intermediate_dim, true, stream);

    launch_silu_multiply(gate_buf, up_buf, out, M * intermediate_dim, stream);
}

// ---- Q4_K GEMV (M=1, decode) ----

__global__ void gemv_q4_k_transB_kernel(const float* __restrict__ x,
                                        const uint8_t* __restrict__ q_weight,
                                        float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    const uint8_t* row_ptr = q_weight + warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;

    float sum = 0.0f;

    int super_blocks_per_thread = (blocks_per_row + 31) / 32;

    for (int sb = 0; sb < super_blocks_per_thread; ++sb) {
        int bi = sb * 32 + lane;
        if (bi >= blocks_per_row)
            break;

        const uint8_t* block_ptr = row_ptr + bi * Q4_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));
        float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qs = block_ptr + 16;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * sc1;
            float m1_val = dmin * m1;
            float d2 = d * sc2;
            float m2_val = dmin * m2;

            int base = bi * QK_K + j;
            for (int l = 0; l < 32; ++l) {
                int idx0 = base + l;
                int idx1 = base + 32 + l;
                if (idx0 < K)
                    sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                if (idx1 < K)
                    sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
            }
            qs += 32;
            is += 2;
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        out[warp_id] = sum;
    }
}

// ---- Q4_K Split-K GEMV (M=1, decode, for large K) ----

__global__ void gemv_q4_k_splitK_kernel(const float* __restrict__ x,
                                        const uint8_t* __restrict__ q_weight,
                                        float* __restrict__ out, int K, int N, int warps_per_row) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int row = global_warp_id / warps_per_row;
    int sub_warp = global_warp_id % warps_per_row;

    if (row >= N)
        return;

    const uint8_t* row_ptr = q_weight + row * blocks_per_row * Q4_K_BLOCK_SIZE;

    int blocks_per_sub = (blocks_per_row + warps_per_row - 1) / warps_per_row;
    int start_block = sub_warp * blocks_per_sub;
    int end_block = min(start_block + blocks_per_sub, blocks_per_row);

    float sum = 0.0f;

    for (int bi = start_block + lane; bi < end_block; bi += 32) {
        const uint8_t* block_ptr = row_ptr + bi * Q4_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));
        float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qs = block_ptr + 16;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * sc1;
            float m1_val = dmin * m1;
            float d2 = d * sc2;
            float m2_val = dmin * m2;

            int base = bi * QK_K + j;
            for (int l = 0; l < 32; ++l) {
                int idx0 = base + l;
                int idx1 = base + 32 + l;
                if (idx0 < K)
                    sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                if (idx1 < K)
                    sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
            }
            qs += 32;
            is += 2;
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        atomicAdd(&out[row], sum);
    }
}

// ---- Q4_K FFN Up Fused (M=1, decode) ----
// Computes SiLU(x @ w1^T) * (x @ w3^T) in a single kernel,
// sharing the x vector read across gate and up projections.

__global__ void ffn_up_fused_q4_k_kernel(const float* __restrict__ x,
                                         const uint8_t* __restrict__ q_w1,
                                         const uint8_t* __restrict__ q_w3, float* __restrict__ out,
                                         int K, int N) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    const uint8_t* w1_row = q_w1 + warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;
    const uint8_t* w3_row = q_w3 + warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    int super_blocks_per_thread = (blocks_per_row + 31) / 32;

    for (int sb = 0; sb < super_blocks_per_thread; ++sb) {
        int bi = sb * 32 + lane;
        if (bi >= blocks_per_row)
            break;

        // Process w1 (gate) block
        {
            const uint8_t* block_ptr = w1_row + bi * Q4_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qs = block_ptr + 16;

            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(is, scales, &sc1, &m1);
                get_scale_min_k4(is + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int base = bi * QK_K + j;
                for (int l = 0; l < 32; ++l) {
                    int idx0 = base + l;
                    int idx1 = base + 32 + l;
                    if (idx0 < K)
                        gate_sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                    if (idx1 < K)
                        gate_sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
                }
                qs += 32;
                is += 2;
            }
        }

        // Process w3 (up) block
        {
            const uint8_t* block_ptr = w3_row + bi * Q4_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qs = block_ptr + 16;

            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(is, scales, &sc1, &m1);
                get_scale_min_k4(is + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int base = bi * QK_K + j;
                for (int l = 0; l < 32; ++l) {
                    int idx0 = base + l;
                    int idx1 = base + 32 + l;
                    if (idx0 < K)
                        up_sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                    if (idx1 < K)
                        up_sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
                }
                qs += 32;
                is += 2;
            }
        }
    }

    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 16);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 8);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 4);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 2);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 1);

    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 16);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 8);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 4);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 2);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 1);

    if (lane == 0) {
        float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        out[warp_id] = silu_gate * up_sum;
    }
}

// ---- Q4_K FFN Up Fused Split-K (M=1, decode, for large K) ----
// Multiple warps per output row, each computing a partial sum.
// Results are accumulated via atomicAdd into a shared output buffer.

__global__ void ffn_up_fused_q4_k_splitK_kernel(const float* __restrict__ x,
                                                const uint8_t* __restrict__ q_w1,
                                                const uint8_t* __restrict__ q_w3,
                                                float* __restrict__ out, int K, int N,
                                                int warps_per_row) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int row = global_warp_id / warps_per_row;
    int sub_warp = global_warp_id % warps_per_row;

    if (row >= N)
        return;

    const uint8_t* w1_row = q_w1 + row * blocks_per_row * Q4_K_BLOCK_SIZE;
    const uint8_t* w3_row = q_w3 + row * blocks_per_row * Q4_K_BLOCK_SIZE;

    int blocks_per_sub = (blocks_per_row + warps_per_row - 1) / warps_per_row;
    int start_block = sub_warp * blocks_per_sub;
    int end_block = min(start_block + blocks_per_sub, blocks_per_row);

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int bi = start_block + lane; bi < end_block; bi += 32) {
        // Process w1 (gate) block
        {
            const uint8_t* block_ptr = w1_row + bi * Q4_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qs = block_ptr + 16;

            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(is, scales, &sc1, &m1);
                get_scale_min_k4(is + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int base = bi * QK_K + j;
                for (int l = 0; l < 32; ++l) {
                    int idx0 = base + l;
                    int idx1 = base + 32 + l;
                    if (idx0 < K)
                        gate_sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                    if (idx1 < K)
                        gate_sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
                }
                qs += 32;
                is += 2;
            }
        }

        // Process w3 (up) block
        {
            const uint8_t* block_ptr = w3_row + bi * Q4_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qs = block_ptr + 16;

            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(is, scales, &sc1, &m1);
                get_scale_min_k4(is + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int base = bi * QK_K + j;
                for (int l = 0; l < 32; ++l) {
                    int idx0 = base + l;
                    int idx1 = base + 32 + l;
                    if (idx0 < K)
                        up_sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                    if (idx1 < K)
                        up_sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
                }
                qs += 32;
                is += 2;
            }
        }
    }

    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 16);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 8);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 4);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 2);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 1);

    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 16);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 8);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 4);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 2);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 1);

    if (lane == 0) {
        float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        atomicAdd(&out[row], silu_gate * up_sum);
    }
}

// ---- Q4_K FFN Up Fused v2 (coalesced x reads) ----
// Same coalescing strategy as gemv_q4_k_transB_v2_kernel:
// all 32 threads process the same super-block, giving coalesced x reads.
// Both gate and up projections share the same x values per super-block.

__global__ void ffn_up_fused_q4_k_v2_kernel(const float* __restrict__ x,
                                            const uint8_t* __restrict__ q_w1,
                                            const uint8_t* __restrict__ q_w3,
                                            float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    const uint8_t* w1_row = q_w1 + (size_t)warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;
    const uint8_t* w3_row = q_w3 + (size_t)warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        // Read x values for this super-block (shared by gate and up)
        float x_vals[8];  // 4 groups * 2 values per group
        {
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                int idx0 = bi * QK_K + j + lane;
                int idx1 = bi * QK_K + j + 32 + lane;
                x_vals[is * 2 + 0] = (idx0 < K) ? x[idx0] : 0.0f;
                x_vals[is * 2 + 1] = (idx1 < K) ? x[idx1] : 0.0f;
                is++;
            }
        }

        // Process w1 (gate) block
        {
            const uint8_t* block_ptr = w1_row + bi * Q4_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qs = block_ptr + 16;

            int si = 0;  // scale index (0,2,4,6)
            int xi = 0;  // x_vals index (0,1,2,3)
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(si, scales, &sc1, &m1);
                get_scale_min_k4(si + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int q_low = qs[lane] & 0xF;
                int q_high = qs[lane] >> 4;

                gate_sum += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
                gate_sum += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

                qs += 32;
                si += 2;
                xi += 1;
            }
        }

        // Process w3 (up) block
        {
            const uint8_t* block_ptr = w3_row + bi * Q4_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qs = block_ptr + 16;

            int si = 0;  // scale index (0,2,4,6)
            int xi = 0;  // x_vals index (0,1,2,3)
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(si, scales, &sc1, &m1);
                get_scale_min_k4(si + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int q_low = qs[lane] & 0xF;
                int q_high = qs[lane] >> 4;

                up_sum += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
                up_sum += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

                qs += 32;
                si += 2;
                xi += 1;
            }
        }
    }

    // Warp reduce
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 16);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 8);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 4);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 2);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 1);

    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 16);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 8);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 4);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 2);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 1);

    if (lane == 0) {
        float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        out[warp_id] = silu_gate * up_sum;
    }
}

void launch_ffn_up_fused_q4_k(const float* x, const void* q_w1, const void* q_w3, float* out, int K,
                              int intermediate_dim, cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q4_k_v2_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_w1), static_cast<const uint8_t*>(q_w3), out, K,
        intermediate_dim);
}

// ============================================================================
// FFN Up Fused Q4_K (Q8_1 pre-quantization + dp4a, M=1, decode)
// ============================================================================
// Replaces the FP32-domain ffn_up_fused_q4_k_v2_kernel with int8 dp4a.
// Strategy: quantize FP32 x → Q8_1 once, then use dp4a for int8×int8 dot product.
//
// Q4_K block: 256 elements per super-block, 144 bytes/block
//   Layout: dm(half2, 4B) + scales(12B) + qs(128B)
// Q8_1 block: 32 elements, 36 bytes/block (half2 ds + int8 qs[32])

// Constants for Q4_K dp4a (matching cuda_gemv.cu)
#define FUSED_Q4K_BE 256   // elements per Q4_K super-block
#define FUSED_Q4K_BS 144   // bytes per Q4_K block
#define FUSED_QR4_K 2
#define FUSED_QI4_K (FUSED_Q4K_BE / (4 * FUSED_QR4_K))  // 32
#define FUSED_QI8_1 8

struct block_q8_1_fused {
    half2 ds;
    int8_t qs[32];
};

// Quantize FP32 x to Q8_1 format (same as cuda_gemv.cu)
__global__ void quantize_q8_1_fused_kernel(const float* __restrict__ x,
                                            block_q8_1_fused* __restrict__ y, int k) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = (k + 31) / 32;
    if (bi >= num_blocks) return;

    int base = bi * 32;
    int end = min(base + 32, k);

    float amax = 0.0f;
    for (int j = base; j < end; ++j) {
        float ax = fabsf(x[j]);
        if (ax > amax) amax = ax;
    }

    float d_val = (amax > 1e-10f) ? (amax / 127.0f) : (1.0f / 127.0f);
    float inv_d = 1.0f / d_val;

    y[bi].ds = __halves2half2(__float2half(d_val), __float2half(0.0f));

    int8_t* qs = y[bi].qs;
    for (int j = 0; j < end - base; ++j) {
        float v = x[base + j] * inv_d;
        v = fminf(fmaxf(v, -128.0f), 127.0f);
        qs[j] = static_cast<int8_t>(__float2int_rn(v));
    }
}

// vec_dot for Q4_K × Q8_1 (duplicated from cuda_gemv.cu for kernel fusion)
static __device__ __forceinline__ float vec_dot_q4_k_q8_1_fused(
    const void* __restrict__ vbq, const block_q8_1_fused* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    const uint8_t* bq4_K = (const uint8_t*)vbq + kbx * FUSED_Q4K_BS;

    int v[2];
    int u[2 * FUSED_QR4_K];
    float d8[FUSED_QR4_K];

    const int bq8_offset = FUSED_QR4_K * ((iqs / 2) / (FUSED_QI8_1 / 2));

    const int* q4 = (const int*)(bq4_K + 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t* scales = (const uint16_t*)(bq4_K + 4);
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t* sc = (const uint8_t*)aux;
    const uint8_t* m  = sc + 2;

#pragma unroll
    for (int i = 0; i < FUSED_QR4_K; ++i) {
        const block_q8_1_fused* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < FUSED_QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

        const int dot1 = forge_dp4a(v1i, u[2 * i + 1], forge_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = forge_dp4a(0x01010101, u[2 * i + 1], forge_dp4a(0x01010101, u[2 * i + 0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm4f = __half22float2(*(const half2*)bq4_K);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// FFN up fused Q4_K kernel: one warp per output row, Q8_1 + dp4a
__global__ void ffn_up_fused_q4_k_q8_1_kernel(
    const block_q8_1_fused* __restrict__ x_q8,
    const uint8_t* __restrict__ q_w1,
    const uint8_t* __restrict__ q_w3,
    float* __restrict__ out, int K, int N)
{
    int num_blocks_row = (K + FUSED_Q4K_BE - 1) / FUSED_Q4K_BE;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;

    const uint8_t* w1_row = q_w1 + (size_t)warp_id * num_blocks_row * FUSED_Q4K_BS;
    const uint8_t* w3_row = q_w3 + (size_t)warp_id * num_blocks_row * FUSED_Q4K_BS;

    const int q8_stride_per_q4k = FUSED_QR4_K * FUSED_QI4_K / FUSED_QI8_1;  // = 8

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_fused* row_q8 = x_q8 + (size_t)bi * q8_stride_per_q4k;

        // Process gate (w1) block
        for (int iqs = 0; iqs < FUSED_QI4_K; iqs += 2) {
            gate_sum += vec_dot_q4_k_q8_1_fused(w1_row, row_q8, bi, iqs);
        }

        // Process up (w3) block
        for (int iqs = 0; iqs < FUSED_QI4_K; iqs += 2) {
            up_sum += vec_dot_q4_k_q8_1_fused(w3_row, row_q8, bi, iqs);
        }
    }

    // Warp reduce
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 16);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 8);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 4);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 2);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 1);

    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 16);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 8);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 4);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 2);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 1);

    if (lane == 0) {
        float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        out[warp_id] = silu_gate * up_sum;
    }
}

void launch_ffn_up_fused_q4_k_q8_1(const float* x, const void* q_w1, const void* q_w3,
                                     float* out, int K, int intermediate_dim, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 format
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_fused);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_fused*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_fused_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);

    // Step 2: Launch fused gate+up+SiLU kernel
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int grid_blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q4_k_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_w1), static_cast<const uint8_t*>(q_w3),
        out, K, intermediate_dim);
}

// ============================================================================
// FFN gate+up fused kernel with GELU(tanh) activation (for Gemma4)
// Identical to ffn_up_fused_q4_k_v2_kernel but uses GELU_tanh instead of SiLU
// ============================================================================

__global__ void ffn_up_fused_q4_k_geglu_kernel(const float* __restrict__ x,
                                                 const uint8_t* __restrict__ q_w1,
                                                 const uint8_t* __restrict__ q_w3,
                                                 float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    const uint8_t* w1_row = q_w1 + (size_t)warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;
    const uint8_t* w3_row = q_w3 + (size_t)warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        // Read x values for this super-block (shared by gate and up)
        float x_vals[8];
        {
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                int idx0 = bi * QK_K + j + lane;
                int idx1 = bi * QK_K + j + 32 + lane;
                x_vals[is * 2 + 0] = (idx0 < K) ? x[idx0] : 0.0f;
                x_vals[is * 2 + 1] = (idx1 < K) ? x[idx1] : 0.0f;
                is++;
            }
        }

        // Process w1 (gate) block
        {
            const uint8_t* block_ptr = w1_row + bi * Q4_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qs = block_ptr + 16;

            int si = 0;  // scale index (0,2,4,6)
            int xi = 0;  // x_vals index (0,1,2,3)
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(si, scales, &sc1, &m1);
                get_scale_min_k4(si + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int q_low = qs[lane] & 0xF;
                int q_high = qs[lane] >> 4;

                gate_sum += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
                gate_sum += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

                qs += 32;
                si += 2;
                xi += 1;
            }
        }

        // Process w3 (up) block
        {
            const uint8_t* block_ptr = w3_row + bi * Q4_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qs = block_ptr + 16;

            int si = 0;  // scale index (0,2,4,6)
            int xi = 0;  // x_vals index (0,1,2,3)
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(si, scales, &sc1, &m1);
                get_scale_min_k4(si + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int q_low = qs[lane] & 0xF;
                int q_high = qs[lane] >> 4;

                up_sum += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
                up_sum += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

                qs += 32;
                si += 2;
                xi += 1;
            }
        }
    }

    // Warp reduce
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 16);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 8);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 4);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 2);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 1);

    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 16);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 8);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 4);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 2);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 1);

    if (lane == 0) {
        // GELU(tanh) activation instead of SiLU
        float x_tmp = gate_sum;
        float tanh_in = 0.7978845608f * (x_tmp + 0.044715f * x_tmp * x_tmp * x_tmp);
        float gelu_gate = 0.5f * x_tmp * (1.0f + tanhf(tanh_in));
        out[warp_id] = gelu_gate * up_sum;
    }
}

void launch_ffn_up_fused_q4_k_geglu(const float* x, const void* q_w1, const void* q_w3,
                                     float* out, int K, int intermediate_dim,
                                     cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q4_k_geglu_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_w1), static_cast<const uint8_t*>(q_w3),
        out, K, intermediate_dim);
}

// ---- Fused FFN Up Q5_K (M=1, decode) ----
// Shares x_vals across w1 (gate) and w3 (up) — halves x global read.

__global__ void ffn_up_fused_q5_k_kernel(const float* __restrict__ x,
                                         const uint8_t* __restrict__ q_w1,
                                         const uint8_t* __restrict__ q_w3,
                                         float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q5_K_BLOCK_SIZE = 176;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N) return;

    const uint8_t* w1_row = q_w1 + (size_t)warp_id * blocks_per_row * Q5_K_BLOCK_SIZE;
    const uint8_t* w3_row = q_w3 + (size_t)warp_id * blocks_per_row * Q5_K_BLOCK_SIZE;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        float x_vals[8];
        {
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                int idx0 = bi * QK_K + j + lane;
                int idx1 = bi * QK_K + j + 32 + lane;
                x_vals[is * 2 + 0] = (idx0 < K) ? x[idx0] : 0.0f;
                x_vals[is * 2 + 1] = (idx1 < K) ? x[idx1] : 0.0f;
                is++;
            }
        }

        // Process w1 (gate) block
        {
            const uint8_t* block_ptr = w1_row + bi * Q5_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qh = block_ptr + 16;
            const uint8_t* ql = block_ptr + 48;

            int si = 0;
            int xi = 0;
            uint16_t u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(si, scales, &sc1, &m1);
                get_scale_min_k4(si + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int q_low = (ql[lane] & 0xF) + ((qh[lane] & u1) ? 16 : 0);
                int q_high = (ql[lane] >> 4) + ((qh[lane] & u2) ? 16 : 0);

                gate_sum += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
                gate_sum += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

                ql += 32;
                si += 2;
                xi += 1;
                u1 <<= 2;
                u2 <<= 2;
            }
        }

        // Process w3 (up) block
        {
            const uint8_t* block_ptr = w3_row + bi * Q5_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qh = block_ptr + 16;
            const uint8_t* ql = block_ptr + 48;

            int si = 0;
            int xi = 0;
            uint16_t u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(si, scales, &sc1, &m1);
                get_scale_min_k4(si + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int q_low = (ql[lane] & 0xF) + ((qh[lane] & u1) ? 16 : 0);
                int q_high = (ql[lane] >> 4) + ((qh[lane] & u2) ? 16 : 0);

                up_sum += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
                up_sum += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

                ql += 32;
                si += 2;
                xi += 1;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 16);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 8);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 4);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 2);
    gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, 1);

    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 16);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 8);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 4);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 2);
    up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, 1);

    if (lane == 0) {
        float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        out[warp_id] = silu_gate * up_sum;
    }
}

void launch_ffn_up_fused_q5_k(const float* x, const void* q_w1, const void* q_w3, float* out, int K,
                               int intermediate_dim, cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q5_k_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_w1), static_cast<const uint8_t*>(q_w3),
        out, K, intermediate_dim);
}

template <int ROWS_PER_WARP>
__global__ void ffn_down_fused_q4_0_tiled_kernel(const float* __restrict__ ffn_mid,
                                                 const uint8_t* __restrict__ q_w2,
                                                 const float* __restrict__ residual,
                                                 float* __restrict__ out, int K, int N) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int first_row = global_warp_id * ROWS_PER_WARP;
    if (first_row >= N)
        return;

    // Accumulate partial sums for ROWS_PER_WARP output rows in registers
    float sums[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        sums[r] = 0.0f;

    // Each lane processes different Q4_0 blocks; ffn_mid values are loaded once
    // and reused across all ROWS_PER_WARP output rows
    for (int bi = lane; bi < num_blocks_row; bi += 32) {
        int base = bi * BLOCK_ELEMS;

        // Load ffn_mid values for this block (shared across all output rows)
        float x_vals[BLOCK_ELEMS];
#pragma unroll
        for (int j = 0; j < BLOCK_ELEMS; ++j) {
            x_vals[j] = (base + j < K) ? ffn_mid[base + j] : 0.0f;
        }

// Compute contribution to each output row using the same x_vals
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = first_row + r;
            if (row >= N)
                break;

            const uint8_t* row_ptr = q_w2 + (size_t)row * num_blocks_row * Q4_0_BLOCK_SIZE;
            const uint8_t* block_ptr = row_ptr + bi * Q4_0_BLOCK_SIZE;

            uint16_t scale_bits;
            memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
            float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));

            const uint8_t* qs = block_ptr + sizeof(uint16_t);

#pragma unroll
            for (int j = 0; j < BLOCK_ELEMS; ++j) {
                int val = q4_unpack(qs, j);
                sums[r] += x_vals[j] * (static_cast<float>(val) * scale);
            }
        }
    }

// Warp reduce and write results for each row
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = first_row + r;
        if (row >= N)
            break;

        float s = sums[r];
        s += __shfl_down_sync(0xFFFFFFFF, s, 16);
        s += __shfl_down_sync(0xFFFFFFFF, s, 8);
        s += __shfl_down_sync(0xFFFFFFFF, s, 4);
        s += __shfl_down_sync(0xFFFFFFFF, s, 2);
        s += __shfl_down_sync(0xFFFFFFFF, s, 1);

        if (lane == 0) {
            out[row] = s + residual[row];
        }
    }
}

void launch_ffn_down_fused_q4_0(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream) {
    const int ROWS_PER_WARP = 4;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int num_warps = (hidden_dim + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    ffn_down_fused_q4_0_tiled_kernel<ROWS_PER_WARP><<<blocks, threads, 0, stream>>>(
        ffn_mid, static_cast<const uint8_t*>(q_w2), residual, out, K, hidden_dim);
}

// ---- FFN Down Fused Q4_K (M=1, decode) ----

template <int ROWS_PER_WARP>
__global__ void ffn_down_fused_q4_k_tiled_kernel(const float* __restrict__ ffn_mid,
                                                  const uint8_t* __restrict__ q_w2,
                                                  const float* __restrict__ residual,
                                                  float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int first_row = global_warp_id * ROWS_PER_WARP;
    if (first_row >= N)
        return;

    float sums[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        sums[r] = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        float x_vals[8];
        {
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                int idx0 = bi * QK_K + j + lane;
                int idx1 = bi * QK_K + j + 32 + lane;
                x_vals[is * 2 + 0] = (idx0 < K) ? ffn_mid[idx0] : 0.0f;
                x_vals[is * 2 + 1] = (idx1 < K) ? ffn_mid[idx1] : 0.0f;
                is++;
            }
        }

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = first_row + r;
            if (row >= N)
                break;

            const uint8_t* row_ptr = q_w2 + (size_t)row * blocks_per_row * Q4_K_BLOCK_SIZE;
            const uint8_t* block_ptr = row_ptr + bi * Q4_K_BLOCK_SIZE;

            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qs = block_ptr + 16;

            int si = 0;
            int xi = 0;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(si, scales, &sc1, &m1);
                get_scale_min_k4(si + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int q_low = qs[lane] & 0xF;
                int q_high = qs[lane] >> 4;

                sums[r] += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
                sums[r] += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

                qs += 32;
                si += 2;
                xi += 1;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = first_row + r;
        if (row >= N)
            break;

        float s = sums[r];
        s += __shfl_down_sync(0xFFFFFFFF, s, 16);
        s += __shfl_down_sync(0xFFFFFFFF, s, 8);
        s += __shfl_down_sync(0xFFFFFFFF, s, 4);
        s += __shfl_down_sync(0xFFFFFFFF, s, 2);
        s += __shfl_down_sync(0xFFFFFFFF, s, 1);

        if (lane == 0) {
            out[row] = s + residual[row];
        }
    }
}

void launch_ffn_down_fused_q4_k(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream) {
    const int ROWS_PER_WARP = 4;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int num_warps = (hidden_dim + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    ffn_down_fused_q4_k_tiled_kernel<ROWS_PER_WARP><<<blocks, threads, 0, stream>>>(
        ffn_mid, static_cast<const uint8_t*>(q_w2), residual, out, K, hidden_dim);
}

// ============================================================================
// FFN Down Fused Q4_K (Q8_1 pre-quantization + dp4a, M=1, decode)
// ============================================================================

template <int ROWS_PER_WARP>
__global__ void ffn_down_fused_q4_k_q8_1_tiled_kernel(
    const block_q8_1_fused* __restrict__ ffn_mid_q8,
    const uint8_t* __restrict__ q_w2,
    const float* __restrict__ residual,
    float* __restrict__ out, int K, int N)
{
    int num_blocks_row = (K + FUSED_Q4K_BE - 1) / FUSED_Q4K_BE;
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int first_row = global_warp_id * ROWS_PER_WARP;
    if (first_row >= N) return;

    // Q8_1 blocks per Q4_K super-block: 256/32 = 8
    const int q8_stride_per_q4k = FUSED_QR4_K * FUSED_QI4_K / FUSED_QI8_1;  // = 8

    float sums[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) sums[r] = 0.0f;

    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        // Offset Q8_1 pointer for this super-block
        const block_q8_1_fused* row_q8 = ffn_mid_q8 + (size_t)bi * q8_stride_per_q4k;

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = first_row + r;
            if (row >= N) break;
            const uint8_t* w2_row = q_w2 + (size_t)row * num_blocks_row * FUSED_Q4K_BS;
            float dot = 0.0f;
            for (int iqs = 0; iqs < FUSED_QI4_K; iqs += 2)
                dot += vec_dot_q4_k_q8_1_fused(w2_row, row_q8, bi, iqs);
            sums[r] += dot;
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = first_row + r;
        if (row >= N) break;
        float s = sums[r];
        s += __shfl_down_sync(0xFFFFFFFF, s, 16);
        s += __shfl_down_sync(0xFFFFFFFF, s, 8);
        s += __shfl_down_sync(0xFFFFFFFF, s, 4);
        s += __shfl_down_sync(0xFFFFFFFF, s, 2);
        s += __shfl_down_sync(0xFFFFFFFF, s, 1);
        if (lane == 0) out[row] = s + residual[row];
    }
}

void launch_ffn_down_fused_q4_k_q8_1(const float* ffn_mid, const void* q_w2,
                                       const float* residual, float* out,
                                       int K, int hidden_dim, cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_fused);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* ffn_mid_q8 = static_cast<block_q8_1_fused*>(q8_buf);
    int q8_threads = 256;
    int q8_blocks_grid = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_fused_kernel<<<q8_blocks_grid, q8_threads, 0, stream>>>(ffn_mid, ffn_mid_q8, K);

    const int ROWS_PER_WARP = 4;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int num_warps = (hidden_dim + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    ffn_down_fused_q4_k_q8_1_tiled_kernel<ROWS_PER_WARP><<<blocks, threads, 0, stream>>>(
        ffn_mid_q8, static_cast<const uint8_t*>(q_w2), residual, out, K, hidden_dim);
}

// ---- FFN Down Fused Q5_K (M=1, decode) ----

template <int ROWS_PER_WARP>
__global__ void ffn_down_fused_q5_k_tiled_kernel(const float* __restrict__ ffn_mid,
                                                  const uint8_t* __restrict__ q_w2,
                                                  const float* __restrict__ residual,
                                                  float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q5_K_BLOCK_SIZE = 176;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int first_row = global_warp_id * ROWS_PER_WARP;
    if (first_row >= N)
        return;

    float sums[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        sums[r] = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        float x_vals[8];
        {
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                int idx0 = bi * QK_K + j + lane;
                int idx1 = bi * QK_K + j + 32 + lane;
                x_vals[is * 2 + 0] = (idx0 < K) ? ffn_mid[idx0] : 0.0f;
                x_vals[is * 2 + 1] = (idx1 < K) ? ffn_mid[idx1] : 0.0f;
                is++;
            }
        }

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = first_row + r;
            if (row >= N)
                break;

            const uint8_t* row_ptr = q_w2 + (size_t)row * blocks_per_row * Q5_K_BLOCK_SIZE;
            const uint8_t* block_ptr = row_ptr + bi * Q5_K_BLOCK_SIZE;

            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));
            float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qh_fixed = block_ptr + 16;
            const uint8_t* ql = block_ptr + 48;

            int si = 0;
            int xi = 0;
            uint16_t u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64) {

                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(si, scales, &sc1, &m1);
                get_scale_min_k4(si + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int q_low_val = (ql[lane] & 0xF) + ((qh_fixed[lane] & u1) ? 16 : 0);
                int q_high_val = (ql[lane] >> 4) + ((qh_fixed[lane] & u2) ? 16 : 0);

                sums[r] += x_vals[xi * 2 + 0] * (d1 * q_low_val - m1_val);
                sums[r] += x_vals[xi * 2 + 1] * (d2 * q_high_val - m2_val);

                ql += 32;
                si += 2;
                xi += 1;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = first_row + r;
        if (row >= N)
            break;

        float s = sums[r];
        s += __shfl_down_sync(0xFFFFFFFF, s, 16);
        s += __shfl_down_sync(0xFFFFFFFF, s, 8);
        s += __shfl_down_sync(0xFFFFFFFF, s, 4);
        s += __shfl_down_sync(0xFFFFFFFF, s, 2);
        s += __shfl_down_sync(0xFFFFFFFF, s, 1);

        if (lane == 0) {
            out[row] = s + residual[row];
        }
    }
}

void launch_ffn_down_fused_q5_k(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream) {
    const int ROWS_PER_WARP = 4;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int num_warps = (hidden_dim + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    ffn_down_fused_q5_k_tiled_kernel<ROWS_PER_WARP><<<blocks, threads, 0, stream>>>(
        ffn_mid, static_cast<const uint8_t*>(q_w2), residual, out, K, hidden_dim);
}

// ---- FFN Down Fused Q6_K (M=1, decode) ----

template <int ROWS_PER_WARP>
__global__ void ffn_down_fused_q6_k_tiled_kernel(const float* __restrict__ ffn_mid,
                                                  const uint8_t* __restrict__ q_w2,
                                                  const float* __restrict__ residual,
                                                  float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q6_K_BLOCK_SIZE = 210;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int first_row = global_warp_id * ROWS_PER_WARP;
    if (first_row >= N)
        return;

    float sums[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        sums[r] = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        float x_vals[8];
        {
            int base = bi * QK_K;
            if (base + lane < K) x_vals[0] = ffn_mid[base + lane];
            else x_vals[0] = 0.0f;
            if (base + 32 + lane < K) x_vals[1] = ffn_mid[base + 32 + lane];
            else x_vals[1] = 0.0f;
            if (base + 64 + lane < K) x_vals[2] = ffn_mid[base + 64 + lane];
            else x_vals[2] = 0.0f;
            if (base + 96 + lane < K) x_vals[3] = ffn_mid[base + 96 + lane];
            else x_vals[3] = 0.0f;
            if (base + 128 + lane < K) x_vals[4] = ffn_mid[base + 128 + lane];
            else x_vals[4] = 0.0f;
            if (base + 160 + lane < K) x_vals[5] = ffn_mid[base + 160 + lane];
            else x_vals[5] = 0.0f;
            if (base + 192 + lane < K) x_vals[6] = ffn_mid[base + 192 + lane];
            else x_vals[6] = 0.0f;
            if (base + 224 + lane < K) x_vals[7] = ffn_mid[base + 224 + lane];
            else x_vals[7] = 0.0f;
        }

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = first_row + r;
            if (row >= N)
                break;

            const uint8_t* row_ptr = q_w2 + (size_t)row * blocks_per_row * Q6_K_BLOCK_SIZE;
            const uint8_t* block_ptr = row_ptr + bi * Q6_K_BLOCK_SIZE;

            const uint8_t* ql = block_ptr;
            const uint8_t* qh = ql + 128;
            const int8_t* sc = reinterpret_cast<const int8_t*>(qh + 64);
            uint16_t d_bits;
            memcpy(&d_bits, sc + 16, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));

            for (int n = 0; n < QK_K; n += 128) {
                int x_off = n >> 5;
                const uint8_t* ql_cur = ql + (n >> 1);
                const uint8_t* qh_cur = qh + (n >> 2);
                const int8_t* sc_cur = sc + (n >> 4);

                int is_ = lane / 16;

                int8_t q1 = (int8_t)((ql_cur[lane] & 0xF) | (((qh_cur[lane] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql_cur[lane + 32] & 0xF) | (((qh_cur[lane] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql_cur[lane] >> 4) | (((qh_cur[lane] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql_cur[lane + 32] >> 4) | (((qh_cur[lane] >> 6) & 3) << 4)) - 32;

                sums[r] += x_vals[x_off + 0] * d * sc_cur[is_ + 0] * static_cast<float>(q1);
                sums[r] += x_vals[x_off + 1] * d * sc_cur[is_ + 2] * static_cast<float>(q2);
                sums[r] += x_vals[x_off + 2] * d * sc_cur[is_ + 4] * static_cast<float>(q3);
                sums[r] += x_vals[x_off + 3] * d * sc_cur[is_ + 6] * static_cast<float>(q4);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = first_row + r;
        if (row >= N)
            break;

        float s = sums[r];
        s += __shfl_down_sync(0xFFFFFFFF, s, 16);
        s += __shfl_down_sync(0xFFFFFFFF, s, 8);
        s += __shfl_down_sync(0xFFFFFFFF, s, 4);
        s += __shfl_down_sync(0xFFFFFFFF, s, 2);
        s += __shfl_down_sync(0xFFFFFFFF, s, 1);

        if (lane == 0) {
            out[row] = s + residual[row];
        }
    }
}

void launch_ffn_down_fused_q6_k(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream) {
    const int ROWS_PER_WARP = 4;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int num_warps = (hidden_dim + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    ffn_down_fused_q6_k_tiled_kernel<ROWS_PER_WARP><<<blocks, threads, 0, stream>>>(
        ffn_mid, static_cast<const uint8_t*>(q_w2), residual, out, K, hidden_dim);
}

// ---- FFN Down Fused Q3_K (M=1, decode) ----
// Q3_K block layout: hmask[32] + qs[64] + scales[12] + d(f16,2B) = 110 bytes.
// Each element j (0..255): q_lo = 2-bit from qs, h = 1-bit from hmask,
// scale = (scales[j/16] - 32) (6-bit signed), value = d * scale * (q_lo - 4*h).

template <int ROWS_PER_WARP>
__global__ void ffn_down_fused_q3_k_tiled_kernel(const float* __restrict__ ffn_mid,
                                                  const uint8_t* __restrict__ q_w2,
                                                  const float* __restrict__ residual,
                                                  float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q3_K_BLOCK_SIZE = 110;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int first_row = global_warp_id * ROWS_PER_WARP;
    if (first_row >= N)
        return;

    float sums[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        sums[r] = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        float x_vals[8];
        {
            int base = bi * QK_K;
            if (base + lane < K) x_vals[0] = ffn_mid[base + lane];
            else x_vals[0] = 0.0f;
            if (base + 32 + lane < K) x_vals[1] = ffn_mid[base + 32 + lane];
            else x_vals[1] = 0.0f;
            if (base + 64 + lane < K) x_vals[2] = ffn_mid[base + 64 + lane];
            else x_vals[2] = 0.0f;
            if (base + 96 + lane < K) x_vals[3] = ffn_mid[base + 96 + lane];
            else x_vals[3] = 0.0f;
            if (base + 128 + lane < K) x_vals[4] = ffn_mid[base + 128 + lane];
            else x_vals[4] = 0.0f;
            if (base + 160 + lane < K) x_vals[5] = ffn_mid[base + 160 + lane];
            else x_vals[5] = 0.0f;
            if (base + 192 + lane < K) x_vals[6] = ffn_mid[base + 192 + lane];
            else x_vals[6] = 0.0f;
            if (base + 224 + lane < K) x_vals[7] = ffn_mid[base + 224 + lane];
            else x_vals[7] = 0.0f;
        }

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = first_row + r;
            if (row >= N)
                break;

            const uint8_t* row_ptr = q_w2 + (size_t)row * blocks_per_row * Q3_K_BLOCK_SIZE;
            const uint8_t* block_ptr = row_ptr + bi * Q3_K_BLOCK_SIZE;

            const uint8_t* hmask = block_ptr;    // 32 bytes (1 bit per element)
            const uint8_t* qs = hmask + 32;       // 64 bytes (2 bits per element)
            const uint8_t* scales_raw = qs + 64;  // 12 bytes (16 packed 6-bit scales)

            uint16_t d_bits;
            memcpy(&d_bits, scales_raw + 12, 2);
            float d = __half2float(reinterpret_cast<const __half&>(d_bits));

            int8_t scales[16];
            q3_k_unpack_scales(scales_raw, scales);

            // Each lane covers 8 elements: j = lane + 32*n (n = 0..7)
            //   q_lo = (qs[(j/128)*32 + j%32] >> (2*((j%128)/32))) & 3
            //        = (qs[(n>>2)*32 + lane] >> (2*(n&3))) & 3
            //   h    = 1 - ((hmask[j%32] >> (j/32)) & 1)  (hmask is stored inverted)
            //   is   = j/16 = lane/16 + 2n
            const int l2 = lane & 3;    // lane%4 (unused, qs shift uses n%4)
            const int is0 = lane >> 4;  // lane/16

#pragma unroll
            for (int n = 0; n < 8; ++n) {
                int q_lo = (qs[((n >> 2) << 5) + lane] >> (2 * (n & 3))) & 3;
                int h = 1 - ((hmask[lane] >> n) & 1);
                int is = is0 + 2 * n;
                float w_val = d * static_cast<float>(scales[is] - 32) *
                              static_cast<float>(q_lo - 4 * h);
                sums[r] += x_vals[n] * w_val;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = first_row + r;
        if (row >= N)
            break;

        float s = sums[r];
        s += __shfl_down_sync(0xFFFFFFFF, s, 16);
        s += __shfl_down_sync(0xFFFFFFFF, s, 8);
        s += __shfl_down_sync(0xFFFFFFFF, s, 4);
        s += __shfl_down_sync(0xFFFFFFFF, s, 2);
        s += __shfl_down_sync(0xFFFFFFFF, s, 1);

        if (lane == 0) {
            out[row] = s + residual[row];
        }
    }
}

void launch_ffn_down_fused_q3_k(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream) {
    const int ROWS_PER_WARP = 4;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int num_warps = (hidden_dim + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    ffn_down_fused_q3_k_tiled_kernel<ROWS_PER_WARP><<<blocks, threads, 0, stream>>>(
        ffn_mid, static_cast<const uint8_t*>(q_w2), residual, out, K, hidden_dim);
}

// ---- Output Proj Q4_0 (M=1, decode, large N) ----
// Specialized kernel for output projection where N (vocab_size) is very large
// (e.g., 152064). Uses multiple warps per row (Split-K) to keep GPU busy.
// Each warp handles a subset of K-blocks for one row, atomicAdd to combine.

__global__ void output_proj_q4_0_kernel(const float* __restrict__ x,
                                        const uint8_t* __restrict__ q_weight,
                                        float* __restrict__ out, int K, int N, int warps_per_row) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int row = global_warp_id / warps_per_row;
    int sub_warp = global_warp_id % warps_per_row;

    if (row >= N)
        return;

    const uint8_t* row_ptr = q_weight + (size_t)row * num_blocks_row * Q4_0_BLOCK_SIZE;

    int blocks_per_sub = (num_blocks_row + warps_per_row - 1) / warps_per_row;
    int start_block = sub_warp * blocks_per_sub;
    int end_block = min(start_block + blocks_per_sub, num_blocks_row);

    float sum = 0.0f;

    for (int bi = start_block + lane; bi < end_block; bi += 32) {
        const uint8_t* block_ptr = row_ptr + bi * Q4_0_BLOCK_SIZE;

        uint16_t scale_bits;
        memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
        float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));

        const uint8_t* qs = block_ptr + sizeof(uint16_t);

        int base = bi * BLOCK_ELEMS;
#pragma unroll
        for (int j = 0; j < BLOCK_ELEMS; ++j) {
            if (base + j >= K)
                break;
            int val = q4_unpack(qs, j);
            sum += x[base + j] * (static_cast<float>(val) * scale);
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        atomicAdd(&out[row], sum);
    }
}

void launch_output_proj_q4_0(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream) {
    int num_blocks_row = (K + 31) / 32;
    int warps_per_row = (num_blocks_row + 31) / 32;
    if (warps_per_row < 1)
        warps_per_row = 1;
    if (warps_per_row > 16)
        warps_per_row = 16;

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = N * warps_per_row;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    cudaMemsetAsync(out, 0, N * sizeof(float), stream);
    output_proj_q4_0_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N, warps_per_row);
}

// ---- Output Proj Q4_K/Q6_K (M=1, decode, large N) ----
// Templated output projection using CudaQuantTraits with split-K (multiple warps per row)

template <DataType DT>
__global__ void output_proj_typed_kernel(const float* __restrict__ x,
                                          const uint8_t* __restrict__ q_weight,
                                          float* __restrict__ out,
                                          int K, int N, int warps_per_row) {
    constexpr int BE = CudaQuantTraits<DT>::block_elements;
    constexpr int BS = CudaQuantTraits<DT>::block_size;
    int num_blocks_row = (K + BE - 1) / BE;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int row = global_warp_id / warps_per_row;
    int sub_warp = global_warp_id % warps_per_row;

    if (row >= N) return;

    const uint8_t* row_ptr = q_weight + (size_t)row * num_blocks_row * BS;

    int blocks_per_sub = (num_blocks_row + warps_per_row - 1) / warps_per_row;
    int start_block = sub_warp * blocks_per_sub;
    int end_block = min(start_block + blocks_per_sub, num_blocks_row);

    float sum = 0.0f;

    for (int bi = start_block + lane; bi < end_block; bi += 32) {
        const uint8_t* block_ptr = row_ptr + bi * BS;
        int base = bi * BE;
        sum += CudaQuantTraits<DT>::dot_block(block_ptr, x, base, K);
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        atomicAdd(&out[row], sum);
    }
}

void launch_output_proj_q4_k(const float* x, const void* q_weight, float* out, int K, int N,
                              cudaStream_t stream) {
    constexpr int BE = CudaQuantTraits<DataType::Q4_K>::block_elements;
    int num_blocks_row = (K + BE - 1) / BE;

    int warps_per_row = (num_blocks_row + 31) / 32;
    if (warps_per_row < 1) warps_per_row = 1;
    if (warps_per_row > 16) warps_per_row = 16;

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = N * warps_per_row;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    cudaMemsetAsync(out, 0, N * sizeof(float), stream);
    output_proj_typed_kernel<DataType::Q4_K><<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N, warps_per_row);
}

void launch_output_proj_q5_k(const float* x, const void* q_weight, float* out, int K, int N,
                              cudaStream_t stream) {
    constexpr int BE = CudaQuantTraits<DataType::Q5_K>::block_elements;
    int num_blocks_row = (K + BE - 1) / BE;

    int warps_per_row = (num_blocks_row + 31) / 32;
    if (warps_per_row < 1) warps_per_row = 1;
    if (warps_per_row > 16) warps_per_row = 16;

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = N * warps_per_row;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    cudaMemsetAsync(out, 0, N * sizeof(float), stream);
    output_proj_typed_kernel<DataType::Q5_K><<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N, warps_per_row);
}

// ---- Output Proj Q6_K Cooperative (M=1, decode) ----
// The split-K kernel gives each lane a whole 256-element super-block, so for
// K=5120 (20 blocks/row) only 20 of 32 lanes are active and consecutive lanes
// read 210 bytes apart, which defeats coalescing entirely. Here all 32 lanes
// cooperate on one super-block: lane L takes offset L inside each 128-element
// half, making the ql/qh/x reads contiguous across the warp.
// Each warp handles one output row (vocab row).

__global__ void output_proj_q6_k_cooperative_kernel(
        const float* __restrict__ x,
        const uint8_t* __restrict__ q_weight,
        float* __restrict__ out,
        int K, int N) {
    const int QK_K = 256;
    const int Q6_K_BLOCK_SIZE = 210;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * blocks_per_row * Q6_K_BLOCK_SIZE;
    const int is_ = lane / 16;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q6_K_BLOCK_SIZE;
        const uint8_t* ql = block_ptr;
        const uint8_t* qh = ql + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(qh + 64);
        uint16_t d_bits;
        memcpy(&d_bits, sc + 16, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));

        int base = bi * QK_K;

        for (int n = 0; n < QK_K; n += 128) {
            const uint8_t* ql_cur = ql + (n >> 1);
            const uint8_t* qh_cur = qh + (n >> 2);
            const int8_t* sc_cur = sc + (n >> 4);

            uint8_t ql0 = ql_cur[lane];
            uint8_t ql1 = ql_cur[lane + 32];
            uint8_t qhv = qh_cur[lane];

            int q1 = (int)((ql0 & 0xF) | (((qhv >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql1 & 0xF) | (((qhv >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql0 >> 4) | (((qhv >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql1 >> 4) | (((qhv >> 6) & 3) << 4)) - 32;

            int idx = base + n + lane;
            if (idx + 0 < K)
                sum += x[idx + 0] * (d * (float)sc_cur[is_ + 0] * (float)q1);
            if (idx + 32 < K)
                sum += x[idx + 32] * (d * (float)sc_cur[is_ + 2] * (float)q2);
            if (idx + 64 < K)
                sum += x[idx + 64] * (d * (float)sc_cur[is_ + 4] * (float)q3);
            if (idx + 96 < K)
                sum += x[idx + 96] * (d * (float)sc_cur[is_ + 6] * (float)q4);
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        out[warp_id] = sum;
    }
}

void launch_output_proj_q6_k(const float* x, const void* q_weight, float* out, int K, int N,
                              cudaStream_t stream) {
    constexpr int BE = CudaQuantTraits<DataType::Q6_K>::block_elements;
    int num_blocks_row = (K + BE - 1) / BE;

    int warps_per_row = (num_blocks_row + 31) / 32;
    if (warps_per_row < 1) warps_per_row = 1;
    if (warps_per_row > 16) warps_per_row = 16;

    // With <=32 blocks per row the split-K kernel degenerates to one lane per
    // block (uncoalesced, most lanes idle); the cooperative kernel is far faster.
    if (warps_per_row == 1) {
        int warps_per_block = 8;
        int threads = warps_per_block * 32;
        int blocks = (N + warps_per_block - 1) / warps_per_block;
        output_proj_q6_k_cooperative_kernel<<<blocks, threads, 0, stream>>>(
            x, static_cast<const uint8_t*>(q_weight), out, K, N);
        return;
    }

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = N * warps_per_row;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    cudaMemsetAsync(out, 0, N * sizeof(float), stream);
    output_proj_typed_kernel<DataType::Q6_K><<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N, warps_per_row);
}

// ---- Output Proj Q4_K Cooperative (M=1, decode, small K) ----
// For small K (e.g., 1536 → 6 Q4_K blocks), the standard split-K kernel
// wastes 26/32 lanes per warp. This kernel uses cooperative processing:
// all 32 lanes work together on each Q4_K block, then warp-reduce the final sum.
// Each warp handles one output row (vocab row).

__global__ void output_proj_q4_k_cooperative_kernel(
        const float* __restrict__ x,
        const uint8_t* __restrict__ q_weight,
        float* __restrict__ out,
        int K, int N) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        // Load x values for this super-block
        float x_vals[8];
        {
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                int idx0 = bi * QK_K + j + lane;
                int idx1 = bi * QK_K + j + 32 + lane;
                x_vals[is * 2 + 0] = (idx0 < K) ? x[idx0] : 0.0f;
                x_vals[is * 2 + 1] = (idx1 < K) ? x[idx1] : 0.0f;
                is++;
            }
        }

        // Process weight block
        const uint8_t* block_ptr = row_ptr + bi * Q4_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));
        float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qs = block_ptr + 16;

        int si = 0;  // scale index (0,2,4,6)
        int xi = 0;  // x_vals index (0,1,2,3)
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(si, scales, &sc1, &m1);
            get_scale_min_k4(si + 1, scales, &sc2, &m2);
            float d1 = d * sc1;
            float m1_val = dmin * m1;
            float d2 = d * sc2;
            float m2_val = dmin * m2;

            int q_low = qs[lane] & 0xF;
            int q_high = qs[lane] >> 4;

            sum += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
            sum += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

            qs += 32;
            si += 2;
            xi += 1;
        }
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        out[warp_id] = sum;
    }
}

// Same as above but with fused logit softcap + suppress tokens
__global__ void output_proj_q4_k_cooperative_softcap_kernel(
        const float* __restrict__ x,
        const uint8_t* __restrict__ q_weight,
        float* __restrict__ out,
        int K, int N,
        float softcap, bool apply_softcap,
        const int* __restrict__ suppress_tokens, int num_suppress) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        float x_vals[8];
        {
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                int idx0 = bi * QK_K + j + lane;
                int idx1 = bi * QK_K + j + 32 + lane;
                x_vals[is * 2 + 0] = (idx0 < K) ? x[idx0] : 0.0f;
                x_vals[is * 2 + 1] = (idx1 < K) ? x[idx1] : 0.0f;
                is++;
            }
        }

        const uint8_t* block_ptr = row_ptr + bi * Q4_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));
        float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qs = block_ptr + 16;

        int si = 0;
        int xi = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(si, scales, &sc1, &m1);
            get_scale_min_k4(si + 1, scales, &sc2, &m2);
            float d1 = d * sc1;
            float m1_val = dmin * m1;
            float d2 = d * sc2;
            float m2_val = dmin * m2;

            int q_low = qs[lane] & 0xF;
            int q_high = qs[lane] >> 4;

            sum += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
            sum += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

            qs += 32;
            si += 2;
            xi += 1;
        }
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        if (apply_softcap) {
            sum = tanhf(sum / softcap) * softcap;
        }
        if (num_suppress > 0) {
            for (int i = 0; i < num_suppress; ++i) {
                if (warp_id == suppress_tokens[i]) {
                    sum = -INFINITY;
                    break;
                }
            }
        }
        out[warp_id] = sum;
    }
}

void launch_output_proj_q4_k_cooperative(const float* x, const void* q_weight, float* out,
                                          int K, int N, cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    output_proj_q4_k_cooperative_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N);
}

void launch_output_proj_q4_k_cooperative_softcap(const float* x, const void* q_weight, float* out,
                                                    int K, int N,
                                                    float softcap, bool apply_softcap,
                                                    const int* suppress_tokens, int num_suppress,
                                                    cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    output_proj_q4_k_cooperative_softcap_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N,
        softcap, apply_softcap, suppress_tokens, num_suppress);
}

// ---- QKV Fused Q4_0 (M=1, decode) ----
// Fuses Q, K, V projections into a single kernel:
// - Input x is loaded once into shared memory and reused by all warps
// - Each warp computes one output row of Q, K, or V
// - Eliminates 2 extra kernel launches and 2 extra x reads from global memory

__global__ void qkv_fused_q4_0_kernel(const float* __restrict__ x, const uint8_t* __restrict__ q_wq,
                                      int N_q, const uint8_t* __restrict__ q_wk, int N_k,
                                      const uint8_t* __restrict__ q_wv, int N_v,
                                      float* __restrict__ out_q, float* __restrict__ out_k,
                                      float* __restrict__ out_v, int K) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    // Shared memory for x vector
    extern __shared__ float smem_x[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Cooperative loading of x into shared memory using float4
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    int num_float4 = K / 4;
    for (int i = tid; i < num_float4; i += block_size) {
        float4 val = x_vec[i];
        smem_x[i * 4 + 0] = val.x;
        smem_x[i * 4 + 1] = val.y;
        smem_x[i * 4 + 2] = val.z;
        smem_x[i * 4 + 3] = val.w;
    }
    for (int i = num_float4 * 4 + tid; i < K; i += block_size) {
        smem_x[i] = x[i];
    }
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int total_N = N_q + N_k + N_v;
    if (warp_id >= total_N)
        return;

    // Determine which weight matrix and output this warp handles
    const uint8_t* row_ptr;
    float* out_ptr;

    if (warp_id < N_q) {
        // Q projection
        row_ptr = q_wq + (size_t)warp_id * num_blocks_row * Q4_0_BLOCK_SIZE;
        out_ptr = out_q + warp_id;
    } else if (warp_id < N_q + N_k) {
        // K projection
        int row = warp_id - N_q;
        row_ptr = q_wk + (size_t)row * num_blocks_row * Q4_0_BLOCK_SIZE;
        out_ptr = out_k + row;
    } else {
        // V projection
        int row = warp_id - N_q - N_k;
        row_ptr = q_wv + (size_t)row * num_blocks_row * Q4_0_BLOCK_SIZE;
        out_ptr = out_v + row;
    }

    float sum = 0.0f;

    for (int bi = lane; bi < num_blocks_row; bi += 32) {
        const uint8_t* block_ptr = row_ptr + bi * Q4_0_BLOCK_SIZE;

        uint16_t scale_bits;
        memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
        float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));

        const uint8_t* qs = block_ptr + sizeof(uint16_t);

        int base = bi * BLOCK_ELEMS;
#pragma unroll
        for (int j = 0; j < BLOCK_ELEMS; ++j) {
            if (base + j >= K)
                break;
            int val = q4_unpack(qs, j);
            sum += smem_x[base + j] * (static_cast<float>(val) * scale);
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        *out_ptr = sum;
    }
}

void launch_qkv_fused_q4_0(const float* x, const void* q_wq, int N_q, const void* q_wk, int N_k,
                           const void* q_wv, int N_v, float* out_q, float* out_k, float* out_v,
                           int K, cudaStream_t stream) {
    int total_N = N_q + N_k + N_v;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_N + warps_per_block - 1) / warps_per_block;
    size_t smem_bytes = K * sizeof(float);
    qkv_fused_q4_0_kernel<<<blocks, threads, smem_bytes, stream>>>(
        x, static_cast<const uint8_t*>(q_wq), N_q, static_cast<const uint8_t*>(q_wk), N_k,
        static_cast<const uint8_t*>(q_wv), N_v, out_q, out_k, out_v, K);
}

// ============================================================================
// QKV Fused Q4_K (Q8_1 pre-quantization + dp4a, M=1, decode)
// ============================================================================

__global__ void qkv_fused_q4_k_kernel(
    const block_q8_1_fused* __restrict__ x_q8,
    const uint8_t* __restrict__ q_wq, int N_q,
    const uint8_t* __restrict__ q_wk, int N_k,
    const uint8_t* __restrict__ q_wv, int N_v,
    float* __restrict__ out_q, float* __restrict__ out_k, float* __restrict__ out_v, int K)
{
    int num_blocks_row = (K + FUSED_Q4K_BE - 1) / FUSED_Q4K_BE;
    const int q8_stride = FUSED_QR4_K * FUSED_QI4_K / FUSED_QI8_1;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;
    int total_N = N_q + N_k + N_v;
    if (warp_id >= total_N) return;

    const uint8_t* row_ptr;
    float* out_ptr;
    if (warp_id < N_q) {
        row_ptr = q_wq + (size_t)warp_id * num_blocks_row * FUSED_Q4K_BS;
        out_ptr = out_q + warp_id;
    } else if (warp_id < N_q + N_k) {
        int row = warp_id - N_q;
        row_ptr = q_wk + (size_t)row * num_blocks_row * FUSED_Q4K_BS;
        out_ptr = out_k + row;
    } else {
        int row = warp_id - N_q - N_k;
        row_ptr = q_wv + (size_t)row * num_blocks_row * FUSED_Q4K_BS;
        out_ptr = out_v + row;
    }

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_fused* row_q8 = x_q8 + (size_t)bi * q8_stride;
        for (int iqs = 0; iqs < FUSED_QI4_K; iqs += 2)
            sum += vec_dot_q4_k_q8_1_fused(row_ptr, row_q8, bi, iqs);
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) *out_ptr = sum;
}

void launch_qkv_fused_q4_k(const float* x, const void* q_wq, int N_q,
                             const void* q_wk, int N_k, const void* q_wv, int N_v,
                             float* out_q, float* out_k, float* out_v, int K,
                             cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_fused);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_fused*>(q8_buf);
    int q8_threads = 256;
    int q8_blocks_grid = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_fused_kernel<<<q8_blocks_grid, q8_threads, 0, stream>>>(x, x_q8, K);

    int total_N = N_q + N_k + N_v;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_N + warps_per_block - 1) / warps_per_block;
    qkv_fused_q4_k_kernel<<<blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_wq), N_q,
        static_cast<const uint8_t*>(q_wk), N_k,
        static_cast<const uint8_t*>(q_wv), N_v,
        out_q, out_k, out_v, K);
}

// ---- Fused QKV Q5_K (M=1, decode) ----
// Shares x_vals across Q, K, V weight rows — 3x x-read savings.

__global__ void qkv_fused_q5_k_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ q_wq, int N_q,
    const uint8_t* __restrict__ q_wk, int N_k,
    const uint8_t* __restrict__ q_wv, int N_v,
    float* __restrict__ out_q, float* __restrict__ out_k, float* __restrict__ out_v, int K)
{
    const int QK_K = 256;
    const int Q5_K_BLOCK_SIZE = 176;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;
    int total_N = N_q + N_k + N_v;
    if (warp_id >= total_N) return;

    const uint8_t* row_ptr;
    float* out_ptr;
    if (warp_id < N_q) {
        row_ptr = q_wq + (size_t)warp_id * blocks_per_row * Q5_K_BLOCK_SIZE;
        out_ptr = out_q + warp_id;
    } else if (warp_id < N_q + N_k) {
        int row = warp_id - N_q;
        row_ptr = q_wk + (size_t)row * blocks_per_row * Q5_K_BLOCK_SIZE;
        out_ptr = out_k + row;
    } else {
        int row = warp_id - N_q - N_k;
        row_ptr = q_wv + (size_t)row * blocks_per_row * Q5_K_BLOCK_SIZE;
        out_ptr = out_v + row;
    }

    float sum = 0.0f;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        // Read x block once
        float x_vals[8];
        {
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                int idx0 = bi * QK_K + j + lane;
                int idx1 = bi * QK_K + j + 32 + lane;
                x_vals[is * 2 + 0] = (idx0 < K) ? x[idx0] : 0.0f;
                x_vals[is * 2 + 1] = (idx1 < K) ? x[idx1] : 0.0f;
                is++;
            }
        }

        const uint8_t* block_ptr = row_ptr + bi * Q5_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));
        float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qh = block_ptr + 16;
        const uint8_t* ql = block_ptr + 48;

        int si = 0;
        int xi = 0;
        uint16_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(si, scales, &sc1, &m1);
            get_scale_min_k4(si + 1, scales, &sc2, &m2);
            float d1 = d * sc1;
            float m1_val = dmin * m1;
            float d2 = d * sc2;
            float m2_val = dmin * m2;

            int q_low = (ql[lane] & 0xF) + ((qh[lane] & u1) ? 16 : 0);
            int q_high = (ql[lane] >> 4) + ((qh[lane] & u2) ? 16 : 0);

            sum += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
            sum += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

            ql += 32;
            si += 2;
            xi += 1;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) *out_ptr = sum;
}

void launch_qkv_fused_q5_k(const float* x, const void* q_wq, int N_q,
                             const void* q_wk, int N_k, const void* q_wv, int N_v,
                             float* out_q, float* out_k, float* out_v, int K,
                             cudaStream_t stream) {
    int total_N = N_q + N_k + N_v;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_N + warps_per_block - 1) / warps_per_block;
    qkv_fused_q5_k_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_wq), N_q,
        static_cast<const uint8_t*>(q_wk), N_k,
        static_cast<const uint8_t*>(q_wv), N_v,
        out_q, out_k, out_v, K);
}

// ---- Attn Proj Q5_K Cooperative (M=1, decode) ----
// Uses cooperative warp: all 32 lanes work on each Q5_K block together,
// sharing x_vals loaded into registers. 100% lane utilization vs 50% in
// split-K GEMV when blocks_per_row < 32 (e.g., K=4096 → 16 blocks).
__global__ void attn_proj_q5_k_cooperative_kernel(
        const float* __restrict__ x,
        const uint8_t* __restrict__ q_weight,
        float* __restrict__ out,
        int K, int N) {
    const int QK_K = 256;
    const int Q5_K_BLOCK_SIZE = 176;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N) return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * blocks_per_row * Q5_K_BLOCK_SIZE;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        float x_vals[8];
        {
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                int idx0 = bi * QK_K + j + lane;
                int idx1 = bi * QK_K + j + 32 + lane;
                x_vals[is * 2 + 0] = (idx0 < K) ? x[idx0] : 0.0f;
                x_vals[is * 2 + 1] = (idx1 < K) ? x[idx1] : 0.0f;
                is++;
            }
        }

        const uint8_t* block_ptr = row_ptr + bi * Q5_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));
        float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qh = block_ptr + 16;
        const uint8_t* ql = block_ptr + 48;

        int si = 0;
        int xi = 0;
        uint16_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(si, scales, &sc1, &m1);
            get_scale_min_k4(si + 1, scales, &sc2, &m2);
            float d1 = d * sc1;
            float m1_val = dmin * m1;
            float d2 = d * sc2;
            float m2_val = dmin * m2;

            int q_low = (ql[lane] & 0xF) + ((qh[lane] & u1) ? 16 : 0);
            int q_high = (ql[lane] >> 4) + ((qh[lane] & u2) ? 16 : 0);

            sum += x_vals[xi * 2 + 0] * (d1 * q_low - m1_val);
            sum += x_vals[xi * 2 + 1] * (d2 * q_high - m2_val);

            ql += 32;
            si += 2;
            xi += 1;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        out[warp_id] = sum;
    }
}

void launch_attn_proj_q5_k_cooperative(const float* x, const void* q_weight, float* out,
                                        int K, int N, cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    attn_proj_q5_k_cooperative_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N);
}

}  // namespace cuda
}  // namespace forge