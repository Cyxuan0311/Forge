#include "cuda_fused.h"
#include "cuda_common.h"
#include "cuda_elementwise.h"
#include "cuda_quant.h"


namespace nanoinfer {
namespace cuda {

// ---- FFN Up Fused Q4_0 (shared memory, M=1, decode) ----
// Fuses gate + up projections: out = SiLU(x @ w1^T) * (x @ w3^T)
// with shared memory to reduce x vector reads from 2x to 1x.
// x vector is loaded once into shared memory, then reused for both
// gate and up weight rows.

__global__ void ffn_up_fused_q4_0_kernel(const float* __restrict__ x,
                                           const uint8_t* __restrict__ q_w1,
                                           const uint8_t* __restrict__ q_w3,
                                           float* __restrict__ out, int K, int N) {
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

    if (warp_id >= N) return;

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
            if (base + j >= K) break;
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

void launch_ffn_up_fused_q4_0(const float* x,
                                const void* q_w1, const void* q_w3,
                                float* out, int K, int intermediate_dim,
                                cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    size_t smem_bytes = K * sizeof(float);
    ffn_up_fused_q4_0_kernel<<<blocks, threads, smem_bytes, stream>>>(
        x, static_cast<const uint8_t*>(q_w1),
        static_cast<const uint8_t*>(q_w3),
        out, K, intermediate_dim);
}

// ---- FFN Up Fused Q4_0 Batch GEMV (M > 1, small batch) ----

__global__ void ffn_up_fused_q4_0_batch_gemv_kernel(
    const float* __restrict__ x, const uint8_t* __restrict__ q_w1,
    const uint8_t* __restrict__ q_w3, float* __restrict__ out,
    int M, int K, int N) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int total_warps = M * N;
    if (warp_id >= total_warps) return;

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
            if (base + j >= K) break;
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

void launch_ffn_up_fused_q4_0_batch_gemv(const float* x,
                                            const void* q_w1, const void* q_w3,
                                            float* out, int M, int K, int N,
                                            cudaStream_t stream) {
    int total_warps = M * N;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q4_0_batch_gemv_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_w1),
        static_cast<const uint8_t*>(q_w3),
        out, M, K, N);
}

// ---- FFN Up Fused Q4_0 Batch (M > 1, prefill, dequant + cublas) ----

void launch_ffn_up_fused_q4_0_batch(const float* x,
                                      const void* q_w1, const void* q_w3,
                                      float* out, int M, int K, int intermediate_dim,
                                      cudaStream_t stream) {
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

void launch_ffn_up_fused_q4_k_batch(const float* x,
                                      const void* q_w1, const void* q_w3,
                                      float* out, int M, int K, int intermediate_dim,
                                      cudaStream_t stream) {
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

    if (warp_id >= N) return;

    const uint8_t* row_ptr = q_weight + warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;

    float sum = 0.0f;

    int super_blocks_per_thread = (blocks_per_row + 31) / 32;

    for (int sb = 0; sb < super_blocks_per_thread; ++sb) {
        int bi = sb * 32 + lane;
        if (bi >= blocks_per_row) break;

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
                if (idx0 < K) sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                if (idx1 < K) sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
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
                                          float* __restrict__ out, int K, int N,
                                          int warps_per_row) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int row = global_warp_id / warps_per_row;
    int sub_warp = global_warp_id % warps_per_row;

    if (row >= N) return;

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
                if (idx0 < K) sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                if (idx1 < K) sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
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
                                            const uint8_t* __restrict__ q_w3,
                                            float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N) return;

    const uint8_t* w1_row = q_w1 + warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;
    const uint8_t* w3_row = q_w3 + warp_id * blocks_per_row * Q4_K_BLOCK_SIZE;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    int super_blocks_per_thread = (blocks_per_row + 31) / 32;

    for (int sb = 0; sb < super_blocks_per_thread; ++sb) {
        int bi = sb * 32 + lane;
        if (bi >= blocks_per_row) break;

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
                    if (idx0 < K) gate_sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                    if (idx1 < K) gate_sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
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
                    if (idx0 < K) up_sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                    if (idx1 < K) up_sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
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

    if (row >= N) return;

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
                    if (idx0 < K) gate_sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                    if (idx1 < K) gate_sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
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
                    if (idx0 < K) up_sum += x[idx0] * (d1 * (qs[l] & 0xF) - m1_val);
                    if (idx1 < K) up_sum += x[idx1] * (d2 * (qs[l] >> 4) - m2_val);
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

    if (warp_id >= N) return;

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

            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(is, scales, &sc1, &m1);
                get_scale_min_k4(is + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;

                int q_low = qs[lane] & 0xF;
                int q_high = qs[lane] >> 4;

                gate_sum += x_vals[is * 2 + 0] * (d1 * q_low - m1_val);
                gate_sum += x_vals[is * 2 + 1] * (d2 * q_high - m2_val);

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

                int q_low = qs[lane] & 0xF;
                int q_high = qs[lane] >> 4;

                up_sum += x_vals[is * 2 + 0] * (d1 * q_low - m1_val);
                up_sum += x_vals[is * 2 + 1] * (d2 * q_high - m2_val);

                qs += 32;
                is += 2;
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

void launch_ffn_up_fused_q4_k(const float* x,
                                const void* q_w1, const void* q_w3,
                                float* out, int K, int intermediate_dim,
                                cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q4_k_v2_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_w1),
        static_cast<const uint8_t*>(q_w3),
        out, K, intermediate_dim);
}


template <int ROWS_PER_WARP>
__global__ void ffn_down_fused_q4_0_tiled_kernel(
    const float* __restrict__ ffn_mid,
    const uint8_t* __restrict__ q_w2,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int K, int N) {

    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int first_row = global_warp_id * ROWS_PER_WARP;
    if (first_row >= N) return;

    // Accumulate partial sums for ROWS_PER_WARP output rows in registers
    float sums[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) sums[r] = 0.0f;

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
            if (row >= N) break;

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
        if (row >= N) break;

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

void launch_ffn_down_fused_q4_0(const float* ffn_mid,
                                  const void* q_w2,
                                  const float* residual,
                                  float* out,
                                  int K, int hidden_dim,
                                  cudaStream_t stream) {
    const int ROWS_PER_WARP = 4;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int num_warps = (hidden_dim + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    ffn_down_fused_q4_0_tiled_kernel<ROWS_PER_WARP><<<blocks, threads, 0, stream>>>(
        ffn_mid, static_cast<const uint8_t*>(q_w2),
        residual, out, K, hidden_dim);
}

// ---- Output Proj Q4_0 (M=1, decode, large N) ----
// Specialized kernel for output projection where N (vocab_size) is very large
// (e.g., 152064). Uses multiple warps per row (Split-K) with shared memory
// for the input vector to maximize throughput.

__global__ void output_proj_q4_0_kernel(const float* __restrict__ x,
                                          const uint8_t* __restrict__ q_weight,
                                          float* __restrict__ out, int K, int N,
                                          int warps_per_row) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int row = global_warp_id / warps_per_row;
    int sub_warp = global_warp_id % warps_per_row;

    if (row >= N) return;

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
            if (base + j >= K) break;
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

void launch_output_proj_q4_0(const float* x, const void* q_weight, float* out,
                               int K, int N, cudaStream_t stream) {
    int num_blocks_row = (K + 31) / 32;
    // For output_proj with large N (e.g., 152064), use more warps per row
    // to keep GPU busy. Each warp handles a subset of blocks for one row.
    int warps_per_row = (num_blocks_row + 31) / 32;
    if (warps_per_row < 1) warps_per_row = 1;
    if (warps_per_row > 16) warps_per_row = 16;  // Allow more warps for large K

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = N * warps_per_row;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    cudaMemsetAsync(out, 0, N * sizeof(float), stream);
    output_proj_q4_0_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N, warps_per_row);
}

// ---- QKV Fused GEMV Q4_0 (M=1, decode) ----
// Fuses Q, K, V projections into a single kernel:
// - Input x is loaded once into shared memory and reused by all warps
// - Each warp computes one output row of Q, K, or V
// - Eliminates 2 extra kernel launches and 2 extra x reads from global memory

__global__ void qkv_fused_q4_0_kernel(const float* __restrict__ x,
                                        const uint8_t* __restrict__ q_wq, int N_q,
                                        const uint8_t* __restrict__ q_wk, int N_k,
                                        const uint8_t* __restrict__ q_wv, int N_v,
                                        float* __restrict__ out_q,
                                        float* __restrict__ out_k,
                                        float* __restrict__ out_v,
                                        int K) {
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
    if (warp_id >= total_N) return;

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
            if (base + j >= K) break;
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

void launch_qkv_fused_q4_0(const float* x,
                             const void* q_wq, int N_q,
                             const void* q_wk, int N_k,
                             const void* q_wv, int N_v,
                             float* out_q, float* out_k, float* out_v,
                             int K, cudaStream_t stream) {
    int total_N = N_q + N_k + N_v;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_N + warps_per_block - 1) / warps_per_block;
    size_t smem_bytes = K * sizeof(float);
    qkv_fused_q4_0_kernel<<<blocks, threads, smem_bytes, stream>>>(
        x,
        static_cast<const uint8_t*>(q_wq), N_q,
        static_cast<const uint8_t*>(q_wk), N_k,
        static_cast<const uint8_t*>(q_wv), N_v,
        out_q, out_k, out_v, K);
}

} // namespace cuda
} // namespace nanoinfer