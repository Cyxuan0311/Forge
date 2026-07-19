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

template <int ROWS_PER_BLOCK>
__global__ void gemv_q4_0_transB_smem_kernel(const float* __restrict__ x,
                                             const uint8_t* __restrict__ q_weight,
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

    if (warp_id >= N)
        return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * Q4_0_BLOCK_SIZE;

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

// All other quantized-type GEMV kernels (Q4_1, Q4_K, Q6_K, Q3_K, Q8_0)
// are now provided by the template in cuda_gemv_tmpl.cuh, instantiated in
// cuda_gemv_instances.cu.  The dispatch tables gemv_dispatch[] and
// gemv_batch_dispatch[] are also defined there.

}  // namespace cuda
}  // namespace forge
