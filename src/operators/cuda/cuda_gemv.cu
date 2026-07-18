#include "cuda_common.h"
#include "cuda_gemv.h"

namespace forge {
namespace cuda {

// ---- FP32 GEMV (M=1, decode) ----

__global__ void gemv_transB_kernel(const float* __restrict__ x, const float* __restrict__ W,
                                   float* __restrict__ out, int K, int N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    float sum = 0.0f;
    const float* row = W + warp_id * K;

    for (int k = lane; k < K; k += 32) {
        sum += x[k] * row[k];
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0)
        out[warp_id] = sum;
}

void launch_gemv_transB(const float* x, const float* W, float* out, int K, int N,
                        cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_transB_kernel<<<blocks, threads, 0, stream>>>(x, W, out, K, N);
}

__global__ void gemv_kernel(const float* x, const float* W, float* out, int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N)
        return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += x[k] * W[k * N + n];
    }
    out[n] = sum;
}

void launch_gemv(const float* x, const float* W, float* out, int K, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gemv_kernel<<<blocks, threads, 0, stream>>>(x, W, out, K, N);
}

// ---- Q4_0 GEMV (M=1, decode) - Optimized with shared memory + vectorized loads ----

// Shared-memory GEMV: x vector is loaded once into shared memory per block,
// then reused by all warps. Uses float4 vectorized loads for x and uint4
// for weight blocks to maximize memory bandwidth utilization.

template <int ROWS_PER_BLOCK>
__global__ void gemv_q4_0_transB_smem_kernel(const float* __restrict__ x,
                                             const uint8_t* __restrict__ q_weight,
                                             float* __restrict__ out, int K, int N) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    // Shared memory for x vector (padded to avoid bank conflicts)
    extern __shared__ float smem_x[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;  // ROWS_PER_BLOCK * 32

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
    // Handle remaining elements
    for (int i = num_float4 * 4 + tid; i < K; i += block_size) {
        smem_x[i] = x[i];
    }
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    // Each lane processes different Q4_0 blocks, reading x from smem
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
        out[warp_id] = sum;
    }
}

__global__ void gemv_q4_0_splitK_kernel(const float* __restrict__ x,
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

void launch_gemv_q4_0_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream) {
    int num_blocks_row = (K + 31) / 32;
    int warps_per_row = (num_blocks_row + 31) / 32;
    if (warps_per_row < 1)
        warps_per_row = 1;
    if (warps_per_row > 8)
        warps_per_row = 8;

    if (warps_per_row <= 1) {
        // Use shared-memory kernel: 8 warps per block, x loaded once per block
        const int ROWS_PER_BLOCK = 8;
        int threads = ROWS_PER_BLOCK * 32;
        int blocks = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        size_t smem_bytes = K * sizeof(float);
        gemv_q4_0_transB_smem_kernel<ROWS_PER_BLOCK><<<blocks, threads, smem_bytes, stream>>>(
            x, static_cast<const uint8_t*>(q_weight), out, K, N);
    } else {
        int warps_per_block = 8;
        int threads = warps_per_block * 32;
        int total_warps = N * warps_per_row;
        int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
        cudaMemsetAsync(out, 0, N * sizeof(float), stream);
        gemv_q4_0_splitK_kernel<<<blocks, threads, 0, stream>>>(
            x, static_cast<const uint8_t*>(q_weight), out, K, N, warps_per_row);
    }
}

// ---- Batched Q4_0 GEMV (M > 1) ----

__global__ void gemv_q4_0_transB_batch_kernel(const float* __restrict__ x,
                                              const uint8_t* __restrict__ q_weight,
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
    const uint8_t* w_row = q_weight + n * num_blocks_row * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;

    for (int bi = lane; bi < num_blocks_row; bi += 32) {
        int base = bi * BLOCK_ELEMS;

        const uint8_t* block_ptr = w_row + bi * Q4_0_BLOCK_SIZE;
        uint16_t scale_bits;
        memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
        float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
        const uint8_t* qs = block_ptr + sizeof(uint16_t);

#pragma unroll
        for (int j = 0; j < BLOCK_ELEMS; ++j) {
            if (base + j >= K)
                break;
            int val = q4_unpack(qs, j);
            sum += x_row[base + j] * (static_cast<float>(val) * scale);
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        out[m * N + n] = sum;
    }
}

void launch_gemv_q4_0_transB_batch(const float* x, const void* q_weight, float* out, int M, int K,
                                   int N, cudaStream_t stream) {
    int total_warps = M * N;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q4_0_transB_batch_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ---- Q4_0 Dual GEMV (gate + up combined) ----

__global__ void gemv_q4_0_transB_dual_kernel(const float* __restrict__ x,
                                             const uint8_t* __restrict__ q_weight1, int N1,
                                             const uint8_t* __restrict__ q_weight2, int N2,
                                             float* __restrict__ out, int K) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int total_N = N1 + N2;
    if (warp_id >= total_N)
        return;

    const uint8_t* row_ptr;
    float* out_ptr;
    if (warp_id < N1) {
        row_ptr = q_weight1 + warp_id * num_blocks_row * Q4_0_BLOCK_SIZE;
        out_ptr = out + warp_id;
    } else {
        int row = warp_id - N1;
        row_ptr = q_weight2 + row * num_blocks_row * Q4_0_BLOCK_SIZE;
        out_ptr = out + N1 + row;
    }

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row)
            break;

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
        *out_ptr = sum;
    }
}

void launch_gemv_q4_0_transB_dual(const float* x, const void* q_weight1, int N1,
                                  const void* q_weight2, int N2, float* out, int K,
                                  cudaStream_t stream) {
    int total_N = N1 + N2;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_N + warps_per_block - 1) / warps_per_block;
    gemv_q4_0_transB_dual_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight1), N1, static_cast<const uint8_t*>(q_weight2), N2,
        out, K);
}

// ---- FFN Up Fused Q4_0 (M=1, decode) - Optimized with shared memory ----

// ---- Q4_K GEMV v2 (coalesced x reads) ----
// Key optimization: all 32 threads in a warp process the SAME super-block,
// each handling a different element. This gives coalesced x reads
// (consecutive threads read consecutive x elements) vs the original kernel
// where each thread processes a different super-block with 256-float stride.

__global__ void gemv_q4_k_transB_v2_kernel(const float* __restrict__ x,
                                           const uint8_t* __restrict__ q_weight,
                                           float* __restrict__ out, int K, int N) {
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

            // Coalesced: thread lane reads x[bi*256 + j + lane]
            // Consecutive threads read consecutive x elements -> 1 memory transaction
            int idx0 = bi * QK_K + j + lane;
            int idx1 = bi * QK_K + j + 32 + lane;

            float x_val0 = (idx0 < K) ? x[idx0] : 0.0f;
            float x_val1 = (idx1 < K) ? x[idx1] : 0.0f;

            // qs[lane]: coalesced byte read (32 threads read 32 consecutive bytes)
            int q_low = qs[lane] & 0xF;
            int q_high = qs[lane] >> 4;

            sum += x_val0 * (d1 * q_low - m1_val);
            sum += x_val1 * (d2 * q_high - m2_val);

            qs += 32;
            is += 2;
        }
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0)
        out[warp_id] = sum;
}

void launch_gemv_q4_k_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q4_k_transB_v2_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N);
}

// ---- Q4_K Batch GEMV (M>1, prefill) ----
// On-the-fly dequantization with coalesced memory access.
// Each warp handles one (m, n) output element, all 32 threads process
// the same super-block for coalesced x reads.

__global__ void gemv_q4_k_transB_batch_v2_kernel(const float* __restrict__ x,
                                                 const uint8_t* __restrict__ q_weight,
                                                 float* __restrict__ out, int M, int K, int N) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int total_warps = M * N;
    if (warp_id >= total_warps)
        return;

    int m = warp_id / N;
    int n = warp_id % N;

    const float* x_row = x + (size_t)m * K;
    const uint8_t* w_row = q_weight + (size_t)n * blocks_per_row * Q4_K_BLOCK_SIZE;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = w_row + bi * Q4_K_BLOCK_SIZE;

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

            // Coalesced x reads: consecutive threads read consecutive elements
            int idx0 = bi * QK_K + j + lane;
            int idx1 = bi * QK_K + j + 32 + lane;

            float x_val0 = (idx0 < K) ? x_row[idx0] : 0.0f;
            float x_val1 = (idx1 < K) ? x_row[idx1] : 0.0f;

            int q_low = qs[lane] & 0xF;
            int q_high = qs[lane] >> 4;

            sum += x_val0 * (d1 * q_low - m1_val);
            sum += x_val1 * (d2 * q_high - m2_val);

            qs += 32;
            is += 2;
        }
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        out[m * N + n] = sum;
    }
}

void launch_gemv_q4_k_transB_batch(const float* x, const void* q_weight, float* out, int M, int K,
                                   int N, cudaStream_t stream) {
    int total_warps = M * N;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q4_k_transB_batch_v2_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}
// ---- Q4_1 GEMV (M=1, decode) ----

__global__ void gemv_q4_1_transB_kernel(const float* __restrict__ x,
                                        const uint8_t* __restrict__ q_weight,
                                        float* __restrict__ out, int K, int N) {
    const int Q4_1_BLOCK_SIZE = 20;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    const uint8_t* row_ptr = q_weight + warp_id * num_blocks_row * Q4_1_BLOCK_SIZE;

    float sum = 0.0f;

    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row)
            break;

        const uint8_t* block_ptr = row_ptr + bi * Q4_1_BLOCK_SIZE;

        uint16_t d_bits, m_bits;
        memcpy(&d_bits, block_ptr, sizeof(uint16_t));
        memcpy(&m_bits, block_ptr + 2, sizeof(uint16_t));
        float d_val = __half2float(reinterpret_cast<const __half&>(d_bits));
        float m_val = __half2float(reinterpret_cast<const __half&>(m_bits));

        const uint8_t* qs = block_ptr + 4;

        int base = bi * BLOCK_ELEMS;
#pragma unroll
        for (int j = 0; j < BLOCK_ELEMS; ++j) {
            if (base + j >= K)
                break;
            int val = q4_unpack_unsigned(qs, j);
            sum += x[base + j] * (static_cast<float>(val) * d_val + m_val);
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

void launch_gemv_q4_1_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q4_1_transB_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N);
}

// ---- Batched Q4_1 GEMV (M > 1) ----

__global__ void gemv_q4_1_transB_batch_kernel(const float* __restrict__ x,
                                              const uint8_t* __restrict__ q_weight,
                                              float* __restrict__ out, int M, int K, int N) {
    const int Q4_1_BLOCK_SIZE = 20;
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
    const uint8_t* w_row = q_weight + n * num_blocks_row * Q4_1_BLOCK_SIZE;

    float sum = 0.0f;

    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row)
            break;

        const uint8_t* block_ptr = w_row + bi * Q4_1_BLOCK_SIZE;

        uint16_t d_bits, m_bits;
        memcpy(&d_bits, block_ptr, sizeof(uint16_t));
        memcpy(&m_bits, block_ptr + 2, sizeof(uint16_t));
        float d_val = __half2float(reinterpret_cast<const __half&>(d_bits));
        float m_val = __half2float(reinterpret_cast<const __half&>(m_bits));

        const uint8_t* qs = block_ptr + 4;

        int base = bi * BLOCK_ELEMS;
#pragma unroll
        for (int j = 0; j < BLOCK_ELEMS; ++j) {
            if (base + j >= K)
                break;
            int val = q4_unpack_unsigned(qs, j);
            sum += x_row[base + j] * (static_cast<float>(val) * d_val + m_val);
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        out[m * N + n] = sum;
    }
}

void launch_gemv_q4_1_transB_batch(const float* x, const void* q_weight, float* out, int M, int K,
                                   int N, cudaStream_t stream) {
    int total_warps = M * N;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q4_1_transB_batch_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ---- Q6_K GEMV (M=1, decode) ----

__global__ void gemv_q6_k_transB_kernel(const float* __restrict__ x,
                                        const uint8_t* __restrict__ q_weight,
                                        float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q6_K_BLOCK_SIZE = 210;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_weight + warp_id * blocks_per_row * Q6_K_BLOCK_SIZE;

    float sum = 0.0f;

    int super_blocks_per_thread = (blocks_per_row + 31) / 32;

    for (int sb = 0; sb < super_blocks_per_thread; ++sb) {
        int bi = sb * 32 + lane;
        if (bi >= blocks_per_row)
            break;

        const uint8_t* block_ptr = row_ptr + bi * Q6_K_BLOCK_SIZE;
        const uint8_t* ql = block_ptr;
        const uint8_t* qh = ql + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(qh + 64);
        uint16_t d_bits;
        memcpy(&d_bits, sc + 16, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));

        const uint8_t* ql_cur = ql;
        const uint8_t* qh_cur = qh;
        const int8_t* sc_cur = sc;

        for (int n = 0; n < QK_K; n += 128) {
            int base = bi * QK_K + n;
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql_cur[l + 0] & 0xF) | (((qh_cur[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql_cur[l + 32] & 0xF) | (((qh_cur[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql_cur[l + 0] >> 4) | (((qh_cur[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql_cur[l + 32] >> 4) | (((qh_cur[l] >> 6) & 3) << 4)) - 32;

                int idx0 = base + l + 0;
                int idx1 = base + l + 32;
                int idx2 = base + l + 64;
                int idx3 = base + l + 96;

                if (idx0 < K)
                    sum +=
                        x[idx0] * d * static_cast<float>(sc_cur[is + 0]) * static_cast<float>(q1);
                if (idx1 < K)
                    sum +=
                        x[idx1] * d * static_cast<float>(sc_cur[is + 2]) * static_cast<float>(q2);
                if (idx2 < K)
                    sum +=
                        x[idx2] * d * static_cast<float>(sc_cur[is + 4]) * static_cast<float>(q3);
                if (idx3 < K)
                    sum +=
                        x[idx3] * d * static_cast<float>(sc_cur[is + 6]) * static_cast<float>(q4);
            }
            ql_cur += 64;
            qh_cur += 32;
            sc_cur += 8;
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

void launch_gemv_q6_k_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q6_k_transB_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N);
}

// ---- Batched Q6_K GEMV (M > 1) ----

__global__ void gemv_q6_k_transB_batch_kernel(const float* __restrict__ x,
                                              const uint8_t* __restrict__ q_weight,
                                              float* __restrict__ out, int M, int K, int N) {
    const int QK_K = 256;
    const int Q6_K_BLOCK_SIZE = 210;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int total_warps = M * N;
    if (warp_id >= total_warps)
        return;

    int m = warp_id / N;
    int n = warp_id % N;

    const float* x_row = x + m * K;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_weight + n * blocks_per_row * Q6_K_BLOCK_SIZE;

    float sum = 0.0f;

    int super_blocks_per_thread = (blocks_per_row + 31) / 32;

    for (int sb = 0; sb < super_blocks_per_thread; ++sb) {
        int bi = sb * 32 + lane;
        if (bi >= blocks_per_row)
            break;

        const uint8_t* block_ptr = row_ptr + bi * Q6_K_BLOCK_SIZE;
        const uint8_t* ql = block_ptr;
        const uint8_t* qh = ql + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(qh + 64);
        uint16_t d_bits;
        memcpy(&d_bits, sc + 16, 2);
        float d = __half2float(reinterpret_cast<const __half&>(d_bits));

        const uint8_t* ql_cur = ql;
        const uint8_t* qh_cur = qh;
        const int8_t* sc_cur = sc;

        for (int nn = 0; nn < QK_K; nn += 128) {
            int base = bi * QK_K + nn;
            for (int l = 0; l < 32; ++l) {
                int is_ = l / 16;
                int8_t q1 = (int8_t)((ql_cur[l + 0] & 0xF) | (((qh_cur[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql_cur[l + 32] & 0xF) | (((qh_cur[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql_cur[l + 0] >> 4) | (((qh_cur[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql_cur[l + 32] >> 4) | (((qh_cur[l] >> 6) & 3) << 4)) - 32;

                int idx0 = base + l + 0;
                int idx1 = base + l + 32;
                int idx2 = base + l + 64;
                int idx3 = base + l + 96;

                if (idx0 < K)
                    sum += x_row[idx0] * d * static_cast<float>(sc_cur[is_ + 0]) *
                           static_cast<float>(q1);
                if (idx1 < K)
                    sum += x_row[idx1] * d * static_cast<float>(sc_cur[is_ + 2]) *
                           static_cast<float>(q2);
                if (idx2 < K)
                    sum += x_row[idx2] * d * static_cast<float>(sc_cur[is_ + 4]) *
                           static_cast<float>(q3);
                if (idx3 < K)
                    sum += x_row[idx3] * d * static_cast<float>(sc_cur[is_ + 6]) *
                           static_cast<float>(q4);
            }
            ql_cur += 64;
            qh_cur += 32;
            sc_cur += 8;
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        out[m * N + n] = sum;
    }
}

void launch_gemv_q6_k_transB_batch(const float* x, const void* q_weight, float* out, int M, int K,
                                   int N, cudaStream_t stream) {
    int total_warps = M * N;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q6_k_transB_batch_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ---- Q3_K GEMV (M=1, decode) ----
// Q3_K block: 110 bytes per 256 elements
// Layout: hmask[32] + qs[64] + scales[12] + d[2]

__device__ void q3_k_unpack_scales(const uint8_t* scales_raw, int8_t* scales_out) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    memcpy(aux, scales_raw, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    memcpy(scales_out, aux, 16);
}

__global__ void gemv_q3_k_transB_kernel(const float* __restrict__ x,
                                        const uint8_t* __restrict__ q_weight,
                                        float* __restrict__ out, int K, int N) {
    const int QK_K = 256;
    const int Q3_K_BLOCK_SIZE = 110;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    if (warp_id >= N)
        return;

    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * blocks_per_row * Q3_K_BLOCK_SIZE;

    float sum = 0.0f;

    int super_blocks_per_thread = (blocks_per_row + 31) / 32;

    for (int sb = 0; sb < super_blocks_per_thread; ++sb) {
        int bi = sb * 32 + lane;
        if (bi >= blocks_per_row)
            break;

        const uint8_t* block_ptr = row_ptr + bi * Q3_K_BLOCK_SIZE;
        const uint8_t* hm = block_ptr;
        const uint8_t* qs = block_ptr + 32;
        const uint8_t* scales_raw = block_ptr + 96;
        uint16_t d_bits;
        memcpy(&d_bits, block_ptr + 108, 2);
        float d_all = __half2float(reinterpret_cast<const __half&>(d_bits));

        // Unpack scales
        int8_t scales[16];
        q3_k_unpack_scales(scales_raw, scales);

        int is = 0;
        uint8_t m = 1;
        const uint8_t* q = qs;

        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int idx = bi * QK_K + n + j * 32 + l;
                    if (idx < K) {
                        int8_t q_val =
                            static_cast<int8_t>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                        sum += x[idx] * dl * static_cast<float>(q_val);
                    }
                }

                dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int idx = bi * QK_K + n + j * 32 + 16 + l;
                    if (idx < K) {
                        int8_t q_val =
                            static_cast<int8_t>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                        sum += x[idx] * dl * static_cast<float>(q_val);
                    }
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
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

void launch_gemv_q3_k_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q3_k_transB_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, K, N);
}

// ---- Batched Q3_K GEMV (M > 1) ----

__global__ void gemv_q3_k_transB_batch_kernel(const float* __restrict__ x,
                                               const uint8_t* __restrict__ q_weight,
                                               float* __restrict__ out, int M, int K, int N) {
    const int QK_K = 256;
    const int Q3_K_BLOCK_SIZE = 110;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (blockIdx.x * blockDim.x + threadIdx.x) % 32;

    int total_warps = M * N;
    if (warp_id >= total_warps)
        return;

    int m_idx = warp_id / N;
    int n_idx = warp_id % N;

    const float* x_row = x + (size_t)m_idx * K;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_weight + (size_t)n_idx * blocks_per_row * Q3_K_BLOCK_SIZE;

    float sum = 0.0f;

    int super_blocks_per_thread = (blocks_per_row + 31) / 32;

    for (int sb = 0; sb < super_blocks_per_thread; ++sb) {
        int bi = sb * 32 + lane;
        if (bi >= blocks_per_row)
            break;

        const uint8_t* block_ptr = row_ptr + bi * Q3_K_BLOCK_SIZE;
        const uint8_t* hm = block_ptr;
        const uint8_t* qs = block_ptr + 32;
        const uint8_t* scales_raw = block_ptr + 96;
        uint16_t d_bits;
        memcpy(&d_bits, block_ptr + 108, 2);
        float d_all = __half2float(reinterpret_cast<const __half&>(d_bits));

        int8_t scales[16];
        q3_k_unpack_scales(scales_raw, scales);

        int is = 0;
        uint8_t m = 1;
        const uint8_t* q = qs;

        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int idx = bi * QK_K + n + j * 32 + l;
                    if (idx < K) {
                        int8_t q_val =
                            static_cast<int8_t>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                        sum += x_row[idx] * dl * static_cast<float>(q_val);
                    }
                }

                dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int idx = bi * QK_K + n + j * 32 + 16 + l;
                    if (idx < K) {
                        int8_t q_val =
                            static_cast<int8_t>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                        sum += x_row[idx] * dl * static_cast<float>(q_val);
                    }
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        out[m_idx * N + n_idx] = sum;
    }
}

void launch_gemv_q3_k_transB_batch(const float* x, const void* q_weight, float* out, int M, int K,
                                   int N, cudaStream_t stream) {
    int total_warps = M * N;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q3_k_transB_batch_kernel<<<blocks, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

}  // namespace cuda
}  // namespace forge
