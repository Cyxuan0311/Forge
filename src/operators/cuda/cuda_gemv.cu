#include "cuda_common.h"
#include "cuda_gemv.h"
#include "cuda_gemv_common.cuh"
#include "forge/cuda_kernels.h"

// IQ-type constant tables.
// NOTE: extern __constant__ does NOT reliably link across TUs with NVCC
// separable compilation — the GEMV kernels read all-zeros from extern
// constants defined in cuda_quant.cu.  Define local static copies here
// and upload the data from THIS translation unit.
extern __constant__ uint8_t  c_kmask_iq2xs[8];       // unused in GEMV kernels
extern __constant__ uint64_t c_iq2xxs_grid[256];     // unused in IQ2_XS/IQ3_S path
extern __constant__ uint64_t c_iq2s_grid[1024];      // unused in IQ2_XS/IQ3_S path
extern __constant__ int8_t   c_kvalues_iq4nl[16];    // IQ4_NL path (verified working)

// Local copies for IQ2_XS / IQ3_S GEMV kernels — uploaded by ensure_gemv_iq_tables()
static __constant__ uint8_t  c_ksigns_iq2xs[128];
static __constant__ uint64_t c_iq2xs_grid[512];
static __constant__ uint32_t c_iq3s_grid[512];

// Host-side ksigns data (mirrors h_ksigns_iq2xs in cuda_quant.cu)
static const uint8_t h_ksigns_iq2xs_gemv[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

// Host-side grid data (defined in cpu/matmul.cpp, declared extern in cuda_quant.cu)
namespace forge {
namespace ops {
extern const uint64_t iq2xs_grid[512];
extern const uint32_t iq3s_grid[512];
}  // namespace ops
}  // namespace forge

static bool gemv_iq_tables_uploaded = false;

static void ensure_gemv_iq_tables() {
    if (gemv_iq_tables_uploaded) return;
    cudaMemcpyToSymbol(c_ksigns_iq2xs, h_ksigns_iq2xs_gemv, sizeof(h_ksigns_iq2xs_gemv));
    cudaMemcpyToSymbol(c_iq2xs_grid, forge::ops::iq2xs_grid, sizeof(uint64_t) * 512);
    cudaMemcpyToSymbol(c_iq3s_grid, forge::ops::iq3s_grid, sizeof(uint32_t) * 512);
    gemv_iq_tables_uploaded = true;
}

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

// ---- Q3_K GEMV (M=1, decode) - Optimized with Q8_1 pre-quantization + dp4a ----
//
// Architecture (following llama.cpp's mmvq approach):
// 1. In the launch function: quantize FP32 x to Q8_1 format (one-time cost)
//    Q8_1 block: half2 ds + int8_t qs[32] = 36 bytes (same layout as llama.cpp)
// 2. In the GEMV kernel: each warp computes one row using vec_dot_q3_K_q8_1
//    - Q3_K low bits read as packed 2-bit via get_int_b2 (4 values at once)
//    - Q3_K high mask read as packed 2-bit via get_int_b2
//    - __vsubss4 for 4-way int8 subtraction (replaces per-element dequant)
//    - __dp4a for 4-way int8 dot product
//    - Q8_1 read as 32-bit aligned int via get_int_b4 (4 int8 at once)

// ---- Q3_K × Q8_1 dot product for one iqs position (ported from llama.cpp) ----
// Processes QR3_K=4 groups of 4 int8 values = 16 elements per call
static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    // Q3_K block layout: hmask[32] + qs[64] + scales[12] + d[2]
    const uint8_t* bq3_K = (const uint8_t*)vbq + kbx * 110;

    const int bq8_offset = GEMV_QR3_K * (iqs / (GEMV_QI3_K / 2));
    const int scale_offset = iqs - iqs % GEMV_QI8_1 + (iqs % GEMV_QI8_1) / (GEMV_QI8_1 / 2);

    // Read d (last 2 bytes of Q3_K block)
    uint16_t d_bits;
    memcpy(&d_bits, bq3_K + 108, 2);
    const float d = __half2float(reinterpret_cast<const __half&>(d_bits));

    // Read packed 2-bit low values from qs (offset 32 in Q3_K block)
    const int vl = get_int_b2(bq3_K + 32, iqs);

    // Read packed 2-bit high mask from hmask (offset 0), invert, shift
    const int vh = ~get_int_b2(bq3_K, iqs % (GEMV_QI3_K / 2)) >> bq8_offset;

    int    u[GEMV_QR3_K];
    float d8[GEMV_QR3_K];

#pragma unroll
    for (int i = 0; i < GEMV_QR3_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % GEMV_QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    // ---- vec_dot_q3_K_q8_1_impl_mmvq (inlined) ----
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < GEMV_QR3_K; ++i) {
        const int isc = scale_offset + 2 * i;

        // Unpack 6-bit scales (same as q3_k_unpack_scales but inline)
        const int isc_low  = isc % (GEMV_QK_K / 32);
        const int sc_shift_low  = 4 * (isc / (GEMV_QK_K / 32));
        const int sc_low   = (bq3_K[96 + isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (GEMV_QK_K / 64);
        const int sc_shift_high = 2 * (isc / (GEMV_QK_K / 64));
        const int sc_high  = ((bq3_K[96 + GEMV_QK_K / 32 + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        // Extract 4 × 2-bit values, replicate to 4 bytes
        const int vil = (vl >> (2 * i)) & 0x03030303;
        // Extract 4 × 1-bit mask values, shift to bit position 2, replicate
        const int vih = ((vh >> i) << 2) & 0x04040404;
        // 4-way int8 saturated subtraction: vi = vil - vih
        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (forge_dp4a(vi, u[i], 0) * sc);
    }

    return d * sumf;
}

// Quantize [K] FP32 values to Q8_1 format
__global__ void quantize_q8_1_kernel(
    const float* __restrict__ x, block_q8_1_gemv* __restrict__ q8, int K)
{
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = (K + 31) / 32;
    if (bi >= num_blocks) return;

    int base = bi * 32;
    int end = min(base + 32, K);

    float amax = 0.0f;
    for (int j = base; j < end; ++j) {
        float ax = fabsf(x[j]);
        if (ax > amax) amax = ax;
    }

    float d_val = (amax > 1e-10f) ? (amax / 127.0f) : (1.0f / 127.0f);
    float inv_d = 1.0f / d_val;

    block_q8_1_gemv& blk = q8[bi];
    blk.ds = __halves2half2(__float2half(d_val), __float2half(0.0f));

    for (int j = 0; j < end - base; ++j) {
        float v = x[base + j] * inv_d;
        v = fminf(fmaxf(v, -128.0f), 127.0f);
        blk.qs[j] = static_cast<int8_t>(__float2int_rn(v));
    }
}

// Q3_K × Q8_1 GEMV kernel: one warp per output row
// Each thread processes a subset of Q3_K blocks using vec_dot_q3_K_q8_1
__global__ void gemv_q3_k_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight,
    float* __restrict__ out, int K, int N)
{
    constexpr int Q3K_BE = 256;  // Q3_K block elements
    constexpr int Q3K_BS = 110;  // Q3_K block bytes
    int num_blocks_row = (K + Q3K_BE - 1) / Q3K_BE;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * Q3K_BS;

    // Q8_1 blocks: each Q3_K block (256 elems) maps to QR3_K * QI3_K / QI8_1 = 8 Q8_1 blocks
    // Total Q8_1 blocks for K elements: K / 32
    const int q8_stride_per_q3k = GEMV_QR3_K * GEMV_QI3_K / GEMV_QI8_1;  // = 8

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride_per_q3k;

        // Iterate over iqs = 0..QI3_K-1 (0..15)
        // Each call processes QR3_K * 4 = 16 elements
        // QI3_K calls = 16 * 16 = 256 = Q3K_BE elements total
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI3_K; ++iqs) {
            sum += vec_dot_q3_K_q8_1(row_ptr, row_q8, bi, iqs);
        }
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_q3_k_smem(const float* x, const void* q_weight, float* out,
                            int K, int N, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 format (external, one-time per GEMV call)
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);

    // Step 2: Launch Q3_K × Q8_1 GEMV kernel
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q3_k_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

// ============================================================================
// Q2_K GEMV (M=1, decode) - Q8_1 pre-quantization + dp4a
// ============================================================================
//
// Q2_K block layout:
//   scales[16] + qs[64] + d(f16,2B) + dmin(f16,2B) = 84 bytes, 256 elements
//   weight = d * d_scale * q2_val - dmin * min_scale
//   scales[i]: lo nibble = d_scale, hi nibble = min_scale
//
// QR2_K = 4 (4 × 2-bit values per byte, 4 Q8_1 blocks per iqs step)
// QI2_K = 16 (QK_K / (4 * QR2_K))
// Each iqs processes QR2_K * 4 = 16 elements

#define GEMV_QR2_K 4
#define GEMV_QI2_K (GEMV_QK_K / (4 * GEMV_QR2_K))  // 16

// ---- Q2_K × Q8_1 dot product for one iqs position ----
// Processes QR2_K=4 groups of 4 int8 values = 16 elements per call
static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    // Q2_K block layout: scales[16] + qs[64] + d[2] + dmin[2] = 84 bytes
    const uint8_t* bq2_K = (const uint8_t*)vbq + kbx * 84;

    const int bq8_offset = GEMV_QR2_K * (iqs / (GEMV_QI2_K / 2));  // 4 * (iqs / 8)

    // Read d and dmin (bytes 80-83 of Q2_K block)
    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits,   bq2_K + 80, 2);
    memcpy(&dmin_bits, bq2_K + 82, 2);
    const float d = __half2float(reinterpret_cast<const __half&>(d_bits));
    const float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));

    // Read packed 2-bit values from qs (offset 16): 4 bytes = 16 × 2-bit values
    const int vq = get_int_b2(bq2_K + 16, iqs);

    int u[GEMV_QR2_K];
    float d8[GEMV_QR2_K];

#pragma unroll
    for (int i = 0; i < GEMV_QR2_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % GEMV_QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    // Scale index base: h (0 or 8) + l_sub_group (0 or 1)
    const int sc_base = (iqs & 8) + ((iqs & 4) >> 2);  // {0, 1, 8, 9}

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < GEMV_QR2_K; ++i) {
        // Extract 4 × 2-bit values from the i-th shift, replicate to 4 bytes
        const int vqi = (vq >> (2 * i)) & 0x03030303;

        // Scale index for this shift: byte offset into scales[16]
        const int sc_idx = sc_base + i * 2;  // {0,2,4,6} or {1,3,5,7} or +8

        // d_scale = lo nibble, min_scale = hi nibble
        const uint8_t sc_byte = bq2_K[sc_idx];
        const int d_scale = sc_byte & 0xF;
        const int min_scale = (sc_byte >> 4) & 0xF;

        // Weighted dot: d_scale * sum(q2_val * q8_val)
        sumf_d += d8[i] * (forge_dp4a(vqi, u[i], 0) * d_scale);

        // Min offset: min_scale * sum(q8_val)
        sumf_m += d8[i] * (forge_dp4a(0x01010101, u[i], 0) * min_scale);
    }

    return d * sumf_d - dmin * sumf_m;
}

// Q2_K × Q8_1 GEMV kernel: one warp per output row
__global__ void gemv_q2_k_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight,
    float* __restrict__ out, int K, int N)
{
    constexpr int Q2K_BE = 256;
    constexpr int Q2K_BS = 84;
    int num_blocks_row = (K + Q2K_BE - 1) / Q2K_BE;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * Q2K_BS;

    // Q8_1 blocks: each Q2_K block (256 elems) maps to QR2_K * QI2_K / QI8_1 = 8 Q8_1 blocks
    const int q8_stride_per_q2k = GEMV_QR2_K * GEMV_QI2_K / GEMV_QI8_1;  // = 8

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride_per_q2k;

#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI2_K; ++iqs) {
            sum += vec_dot_q2_K_q8_1(row_ptr, row_q8, bi, iqs);
        }
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_q2_k_q8_1(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 format
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks_grid = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks_grid, q8_threads, 0, stream>>>(x, x_q8, K);

    // Step 2: Launch Q2_K × Q8_1 GEMV kernel
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q2_k_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

// ============================================================================
// Q4_K GEMV (M=1, decode) - Q8_1 pre-quantization + dp4a
// ============================================================================
//
// Q4_K block layout (same as llama.cpp):
//   half2 dm (d + dmin) + scales[12] + qs[128] = 144 bytes, 256 elements
//   weight = d * scale * qs_nibble - dmin * min
//
// Q4_K has QR4_K=2 Q8_1 blocks per iqs step, iqs in 0,2..30 (16 steps total)
// Each iqs step processes 2*8 = 16 Q8_1 int32 values = 64 elements

#define GEMV_QR4_K 2
#define GEMV_QI4_K (GEMV_QK_K / (4 * GEMV_QR4_K))  // 32

#define GEMV_QR5_K 2
#define GEMV_QI5_K (GEMV_QK_K / (4 * GEMV_QR5_K))  // 32

#define GEMV_QR6_K 2
#define GEMV_QI6_K (GEMV_QK_K / (4 * GEMV_QR6_K))  // 32

// ---- Q4_K × Q8_1 dot product for one iqs position (ported from llama.cpp) ----
// Processes QR4_K=2 groups of 2×4 int8 values = 64 elements per call
static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    // Q4_K block layout: dm[4] + scales[12] + qs[128]
    const uint8_t* bq4_K = (const uint8_t*)vbq + kbx * 144;

    int v[2];
    int u[2 * GEMV_QR4_K];
    float d8[GEMV_QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = GEMV_QR4_K * ((iqs / 2) / (GEMV_QI8_1 / 2));

    // Read 2 packed int32 values from qs (offset 16 in Q4_K block)
    // qs layout: 128 bytes, organized as 8 sub-blocks of 32 bytes
    // Each sub-block: 16 bytes for low nibble of 32 elements, 16 bytes for high nibble
    const int* q4 = (const int*)(bq4_K + 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

    // Unpack scales and mins from scales[12] (offset 4 in Q4_K block)
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

    // Read Q8_1 values
#pragma unroll
    for (int i = 0; i < GEMV_QR4_K; ++i) {
        const block_q8_1_gemv* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    // ---- vec_dot_q4_K_q8_1_impl_vmmq (inlined) ----
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < GEMV_QR4_K; ++i) {
        // Extract 4-bit nibbles, replicate to 4 bytes
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

        // dot1: weighted dot product
        const int dot1 = forge_dp4a(v1i, u[2 * i + 1], forge_dp4a(v0i, u[2 * i + 0], 0));
        // dot2: sum of Q8_1 values (for the min offset)
        const int dot2 = forge_dp4a(0x01010101, u[2 * i + 1], forge_dp4a(0x01010101, u[2 * i + 0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    // dm = half2 (d, dmin)
    const float2 dm4f = __half22float2(*(const half2*)bq4_K);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// Q4_K × Q8_1 GEMV kernel: one warp per output row
__global__ void gemv_q4_k_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight,
    float* __restrict__ out, int K, int N)
{
    constexpr int Q4K_BE = 256;
    constexpr int Q4K_BS = 144;
    int num_blocks_row = (K + Q4K_BE - 1) / Q4K_BE;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * Q4K_BS;

    // Q8_1 blocks: each Q4_K block (256 elems) maps to QR4_K * QI4_K / QI8_1 = 8 Q8_1 blocks
    const int q8_stride_per_q4k = GEMV_QR4_K * GEMV_QI4_K / GEMV_QI8_1;  // = 8

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride_per_q4k;

        // Iterate over iqs = 0,2,4..30 (16 steps)
        // Each call processes QR4_K * 2 * 4 = 16 elements
        // 16 steps * 16 elements = 256 = Q4K_BE elements total
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI4_K; iqs += 2) {
            sum += vec_dot_q4_K_q8_1(row_ptr, row_q8, bi, iqs);
        }
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_q4_k_q8_1(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 format
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks_grid = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks_grid, q8_threads, 0, stream>>>(x, x_q8, K);

    // Step 2: Launch Q4_K × Q8_1 GEMV kernel
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q4_k_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

// ============================================================================
// Q4_0 GEMV (M=1, decode) - Q8_1 pre-quantization + dp4a
// ============================================================================
//
// Replaces the old FP32-domain smem/splitK kernels with int8 dp4a.
// Strategy: quantize FP32 x → Q8_1 once, then use dp4a for int8×int8 dot product.
//
// Q4_0 block: 2 bytes fp16 scale + 16 bytes qs (4-bit packed, 2 nibbles/byte) = 18 bytes/32 elems
// Q8_1 block: 4 bytes half2 ds + 32 bytes int8 qs = 36 bytes/32 elems
//
// For each Q4_0 block:
//   qs[j] byte = low_nibble | (high_nibble << 4)
//   weight[j*2+0] = low_nibble - 8,  weight[j*2+1] = high_nibble - 8
//   dot = scale * (sum(low_nibble * q8[j]) + sum(high_nibble * q8[j+16]) - 8 * sum(q8))
//
// With dp4a:
//   dot = scale * d8 * (dp4a(packed_low, q8_packed) + dp4a(packed_high, q8_packed+16)
//                        - 8 * dp4a(0x01010101, all_q8_packed))

// ---- vec_dot_q4_0_q8_1: dot product of one Q4_0 block × one Q8_1 block (32 elements) ----
// Extracted as an independent primitive (Phase 5), aligned with vec_dot_q2_K_q8_1 etc.
static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const uint8_t* __restrict__ block_ptr,
    const block_q8_1_gemv* __restrict__ q8_blk)
{
    uint16_t scale_bits;
    memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
    float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
    float d8 = __low2float(q8_blk->ds);

    const uint8_t* qs = block_ptr + 2;
    const int* q8_int = (const int*)q8_blk->qs;

    float block_sum = 0.0f;
    int bsum = 0;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t qs4;
        memcpy(&qs4, qs + j * 4, 4);

        int q4_low = qs4 & 0x0F0F0F0F;
        int q4_high = (qs4 >> 4) & 0x0F0F0F0F;

        int u_low = q8_int[j];
        int u_high = q8_int[j + 4];

        block_sum += (float)(forge_dp4a(q4_low, u_low, 0) + forge_dp4a(q4_high, u_high, 0));
        bsum += forge_dp4a(0x01010101, u_low, 0) + forge_dp4a(0x01010101, u_high, 0);
    }

    block_sum -= 8.0f * (float)bsum;
    return scale * d8 * block_sum;
}

__global__ void gemv_q4_0_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight,
    float* __restrict__ out, int K, int N)
{
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        sum += vec_dot_q4_0_q8_1(row_ptr + (size_t)bi * Q4_0_BLOCK_SIZE, x_q8 + bi);
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_q4_0_q8_1(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 format
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);

    // Step 2: Launch Q4_0 × Q8_1 GEMV kernel
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q4_0_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

// ---- Q4_0 × Q8_1 batched GEMV (M>1, e.g. prefill / small batch) ----
// Phase 5: upgrades the batch path from FP32 scalar dequant to Q8_1 + dp4a.
// Each warp computes one (m, n) output element; x is quantized to Q8_1 once.

__global__ void gemv_q4_0_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,  // [M, num_blocks_row]
    const uint8_t* __restrict__ q_weight,       // [N, num_blocks_row * Q4_0_BLOCK_SIZE]
    float* __restrict__ out, int M, int K, int N)
{
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    int total_warps = M * N;
    if (warp_id >= total_warps) return;

    int m = warp_id / N;
    int n = warp_id % N;

    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * Q4_0_BLOCK_SIZE;

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        sum += vec_dot_q4_0_q8_1(w_row + (size_t)bi * Q4_0_BLOCK_SIZE, x_row + bi);
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_q4_0_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 format (all M rows at once, row-major)
    int num_q8_blocks = M * ((K + 31) / 32);
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);

    // Step 2: Launch batched Q4_0 × Q8_1 GEMV kernel
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q4_0_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ---- Q2_K × Q8_1 batched GEMV (M>1) ----
// Phase 5: upgrades the batch path from FP32 template to Q8_1 + dp4a.

__global__ void gemv_q2_k_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,  // [M, K/32] row-major
    const uint8_t* __restrict__ q_weight,       // [N, num_blocks_row * Q2K_BS]
    float* __restrict__ out, int M, int K, int N)
{
    constexpr int Q2K_BE = 256;
    constexpr int Q2K_BS = 84;
    int num_blocks_row = (K + Q2K_BE - 1) / Q2K_BE;
    const int q8_stride = GEMV_QR2_K * GEMV_QI2_K / GEMV_QI8_1;  // = 8

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    int total_warps = M * N;
    if (warp_id >= total_warps) return;

    int m = warp_id / N;
    int n = warp_id % N;

    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row * q8_stride;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * Q2K_BS;

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_gemv* row_q8 = x_row + (size_t)bi * q8_stride;

#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI2_K; ++iqs) {
            sum += vec_dot_q2_K_q8_1(w_row, row_q8, bi, iqs);
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_q2_k_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q2_k_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ---- Q3_K × Q8_1 batched GEMV (M>1) ----
// Phase 5: upgrades the batch path from FP32 template to Q8_1 + dp4a.

__global__ void gemv_q3_k_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,  // [M, K/32] row-major
    const uint8_t* __restrict__ q_weight,       // [N, num_blocks_row * Q3K_BS]
    float* __restrict__ out, int M, int K, int N)
{
    constexpr int Q3K_BE = 256;
    constexpr int Q3K_BS = 110;
    int num_blocks_row = (K + Q3K_BE - 1) / Q3K_BE;
    const int q8_stride = GEMV_QR3_K * GEMV_QI3_K / GEMV_QI8_1;  // = 8

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    int total_warps = M * N;
    if (warp_id >= total_warps) return;

    int m = warp_id / N;
    int n = warp_id % N;

    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row * q8_stride;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * Q3K_BS;

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_gemv* row_q8 = x_row + (size_t)bi * q8_stride;

#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI3_K; ++iqs) {
            sum += vec_dot_q3_K_q8_1(w_row, row_q8, bi, iqs);
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_q3_k_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q3_k_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ---- Q4_K × Q8_1 batched GEMV (M>1) ----
// Phase 5: upgrades the batch path from FP32 template to Q8_1 + dp4a.

__global__ void gemv_q4_k_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,  // [M, K/32] row-major
    const uint8_t* __restrict__ q_weight,       // [N, num_blocks_row * Q4K_BS]
    float* __restrict__ out, int M, int K, int N)
{
    constexpr int Q4K_BE = 256;
    constexpr int Q4K_BS = 144;
    int num_blocks_row = (K + Q4K_BE - 1) / Q4K_BE;
    const int q8_stride = GEMV_QR4_K * GEMV_QI4_K / GEMV_QI8_1;  // = 8

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    int total_warps = M * N;
    if (warp_id >= total_warps) return;

    int m = warp_id / N;
    int n = warp_id % N;

    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row * q8_stride;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * Q4K_BS;

    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_gemv* row_q8 = x_row + (size_t)bi * q8_stride;

#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI4_K; iqs += 2) {
            sum += vec_dot_q4_K_q8_1(w_row, row_q8, bi, iqs);
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_q4_k_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q4_k_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// Q4_1 GEMV (M=1 + M>1) - Q8_1 pre-quantization + dp4a (Phase 5)
// Block: d(f16,2B) + m(f16,2B) + qs[16] = 20 bytes, 32 elements
// Weight = unsigned_nibble * d + m
// ============================================================================

static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const uint8_t* __restrict__ block_ptr,
    const block_q8_1_gemv* __restrict__ q8_blk)
{
    uint16_t d_bits, m_bits;
    memcpy(&d_bits, block_ptr, sizeof(uint16_t));
    memcpy(&m_bits, block_ptr + 2, sizeof(uint16_t));
    float d = __half2float(reinterpret_cast<const __half&>(d_bits));
    float m = __half2float(reinterpret_cast<const __half&>(m_bits));
    float d8 = __low2float(q8_blk->ds);

    const uint8_t* qs = block_ptr + 4;
    const int* q8_int = (const int*)q8_blk->qs;

    float block_sum = 0.0f;
    int bsum = 0;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t qs4;
        memcpy(&qs4, qs + j * 4, 4);
        int q4_low = qs4 & 0x0F0F0F0F;
        int q4_high = (qs4 >> 4) & 0x0F0F0F0F;
        int u_low = q8_int[j];
        int u_high = q8_int[j + 4];
        block_sum += (float)(forge_dp4a(q4_low, u_low, 0) + forge_dp4a(q4_high, u_high, 0));
        bsum += forge_dp4a(0x01010101, u_low, 0) + forge_dp4a(0x01010101, u_high, 0);
    }

    return d * d8 * block_sum + m * d8 * (float)bsum;
}

__global__ void gemv_q4_1_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BS = 20, BE = 32;
    int num_blocks_row = (K + BE - 1) / BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        sum += vec_dot_q4_1_q8_1(row_ptr + (size_t)bi * BS, x_q8 + bi);
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_q4_1_q8_1(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q4_1_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_q4_1_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BS = 20, BE = 32;
    int num_blocks_row = (K + BE - 1) / BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        sum += vec_dot_q4_1_q8_1(w_row + (size_t)bi * BS, x_row + bi);
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_q4_1_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q4_1_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// Q8_0 GEMV (M=1 + M>1) - Q8_1 pre-quantization + dp4a (Phase 5)
// Block: d(f16,2B) + qs[32](int8) = 34 bytes, 32 elements
// Weight = qs[i] * d  (signed int8, no bias)
// ============================================================================

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const uint8_t* __restrict__ block_ptr,
    const block_q8_1_gemv* __restrict__ q8_blk)
{
    uint16_t d_bits;
    memcpy(&d_bits, block_ptr, sizeof(uint16_t));
    float d = __half2float(reinterpret_cast<const __half&>(d_bits));
    float d8 = __low2float(q8_blk->ds);

    const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + 2);
    const int* q8_int = (const int*)q8_blk->qs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int qs_int;
        memcpy(&qs_int, qs + j * 4, 4);
        sumi = forge_dp4a(qs_int, q8_int[j], sumi);
    }

    return d * d8 * (float)sumi;
}

__global__ void gemv_q8_0_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BS = 34, BE = 32;
    int num_blocks_row = (K + BE - 1) / BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        sum += vec_dot_q8_0_q8_1(row_ptr + (size_t)bi * BS, x_q8 + bi);
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_q8_0_q8_1(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q8_0_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_q8_0_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BS = 34, BE = 32;
    int num_blocks_row = (K + BE - 1) / BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        sum += vec_dot_q8_0_q8_1(w_row + (size_t)bi * BS, x_row + bi);
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_q8_0_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q8_0_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// Q5_K GEMV (M=1 + M>1) - Q8_1 pre-quantization + dp4a (Phase 5)
// Block: d(f16,2B) + dmin(f16,2B) + scales[12] + qh[32] + ql[128] = 176 bytes, 256 elements
// Weight = d * scale * (low_nibble + high_bit*16) - dmin * min
// ============================================================================

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    const uint8_t* bq5_K = (const uint8_t*)vbq + kbx * 176;
    const int bq8_offset = GEMV_QR5_K * ((iqs / 2) / (GEMV_QI8_1 / 2));

    // Read d and dmin (first 4 bytes)
    const float2 dm5f = __half22float2(*(const half2*)bq5_K);

    // Read packed low nibbles from ql (offset 48 in Q5_K block), same layout as Q4_K qs
    const int* ql = (const int*)(bq5_K + 48 + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    int vl[2] = {ql[0], ql[4]};

    // Read packed high bits from qh (offset 16)
    const int* qh = (const int*)(bq5_K + 16 + 4 * ((iqs / 2) % 4));
    int vh[2] = {qh[0] >> bq8_offset, qh[4] >> bq8_offset};

    // Unpack scales (same as Q4_K)
    const uint16_t* scales = (const uint16_t*)(bq5_K + 4);
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
    const uint8_t* m = sc + 2;

    int u[2 * GEMV_QR5_K];
    float d8[GEMV_QR5_K];

#pragma unroll
    for (int i = 0; i < GEMV_QR5_K; ++i) {
        const block_q8_1_gemv* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < GEMV_QR5_K; ++i) {
        const int v0i = ((vl[0] >> (4 * i)) & 0x0F0F0F0F) | (((vh[0] >> i) << 4) & 0x10101010);
        const int v1i = ((vl[1] >> (4 * i)) & 0x0F0F0F0F) | (((vh[1] >> i) << 4) & 0x10101010);

        const int dot1 = forge_dp4a(v1i, u[2 * i + 1], forge_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = forge_dp4a(0x01010101, u[2 * i + 1], forge_dp4a(0x01010101, u[2 * i + 0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__global__ void gemv_q5_k_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BE = 256, BS = 176;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = GEMV_QR5_K * GEMV_QI5_K / GEMV_QI8_1;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI5_K; iqs += 2) {
            sum += vec_dot_q5_K_q8_1(row_ptr, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_q5_k_q8_1(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q5_k_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_q5_k_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BE = 256, BS = 176;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = GEMV_QR5_K * GEMV_QI5_K / GEMV_QI8_1;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row * q8_stride;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_row + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI5_K; iqs += 2) {
            sum += vec_dot_q5_K_q8_1(w_row, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_q5_k_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q5_k_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// Q6_K GEMV (M=1 + M>1) - Q8_1 pre-quantization + dp4a (Phase 5)
// Block: ql[128] + qh[64] + sc[16] + d(f16,2B) = 210 bytes, 256 elements
// Weight = d * scale * (6bit_val - 32)
// ============================================================================

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    const uint8_t* bq6_K = (const uint8_t*)vbq + kbx * 210;

    const int bq8_offset = 2 * GEMV_QR6_K * (iqs / (GEMV_QI6_K / 2)) + (iqs % (GEMV_QI6_K / 2)) / (GEMV_QI6_K / 4);
    const int scale_offset = (GEMV_QI6_K / 4) * (iqs / (GEMV_QI6_K / 2)) + (iqs % (GEMV_QI6_K / 2)) / (GEMV_QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (GEMV_QI6_K / 2)) / (GEMV_QI6_K / 4));

    const int vl = get_int_b2(bq6_K, iqs);
    const int vh = get_int_b2(bq6_K + 128, (GEMV_QI6_K / 4) * (iqs / (GEMV_QI6_K / 2)) + iqs % (GEMV_QI6_K / 4)) >> vh_shift;

    uint16_t d_bits;
    memcpy(&d_bits, bq6_K + 208, 2);
    float d = __half2float(reinterpret_cast<const __half&>(d_bits));

    const int8_t* scales = reinterpret_cast<const int8_t*>(bq6_K + 192) + scale_offset;

    int u[GEMV_QR6_K];
    float d8[GEMV_QR6_K];

#pragma unroll
    for (int i = 0; i < GEMV_QR6_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % GEMV_QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
    }

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < GEMV_QR6_K; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4((vil | vih), 0x20202020);
        sumf += d8[i] * (forge_dp4a(vi, u[i], 0) * sc);
    }

    return d * sumf;
}

__global__ void gemv_q6_k_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BE = 256, BS = 210;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = GEMV_QR6_K * GEMV_QI6_K / GEMV_QI8_1;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI6_K; ++iqs) {
            sum += vec_dot_q6_K_q8_1(row_ptr, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_q6_k_q8_1(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q6_k_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_q6_k_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BE = 256, BS = 210;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = GEMV_QR6_K * GEMV_QI6_K / GEMV_QI8_1;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row * q8_stride;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_row + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI6_K; ++iqs) {
            sum += vec_dot_q6_K_q8_1(w_row, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_q6_k_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q6_k_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// Q5_0 GEMV - Q8_1 pre-quantization (32 elem/block, 22 bytes)
// ============================================================================

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const uint8_t* __restrict__ block_ptr,
    const block_q8_1_gemv* __restrict__ q8_blk)
{
    float d = __half2float(reinterpret_cast<const __half&>(*(const uint16_t*)block_ptr));
    float d8 = __low2float(q8_blk->ds);

    uint32_t qh;
    memcpy(&qh, block_ptr + 2, 4);
    const uint8_t* qs = block_ptr + 6;
    const int8_t* q8 = reinterpret_cast<const int8_t*>(q8_blk->qs);

    float sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
        int x0 = ((qs[i] & 0x0F) | (((qh >> i) & 1) << 4)) - 16;
        int x1 = ((qs[i] >> 4) | (((qh >> (i + 16)) & 1) << 4)) - 16;
        sum += (float)x0 * (float)q8[i] + (float)x1 * (float)q8[i + 16];
    }
    return d * d8 * sum;
}

__global__ void gemv_q5_0_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BS = 22, BE = 32;
    int num_blocks_row = (K + BE - 1) / BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        sum += vec_dot_q5_0_q8_1(row_ptr + (size_t)bi * BS, x_q8 + bi);
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_q5_0_q8_1(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q5_0_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_q5_0_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BS = 22, BE = 32;
    int num_blocks_row = (K + BE - 1) / BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        sum += vec_dot_q5_0_q8_1(w_row + (size_t)bi * BS, x_row + bi);
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_q5_0_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q5_0_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// Q5_1 GEMV - Q8_1 pre-quantization (32 elem/block, 24 bytes)
// ============================================================================

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const uint8_t* __restrict__ block_ptr,
    const block_q8_1_gemv* __restrict__ q8_blk)
{
    float d = __half2float(reinterpret_cast<const __half&>(*(const uint16_t*)block_ptr));
    float m = __half2float(reinterpret_cast<const __half&>(*(const uint16_t*)(block_ptr + 2)));
    float d8 = __low2float(q8_blk->ds);

    uint32_t qh;
    memcpy(&qh, block_ptr + 4, 4);
    const uint8_t* qs = block_ptr + 8;
    const int8_t* q8 = reinterpret_cast<const int8_t*>(q8_blk->qs);

    float sum = 0.0f, sum_q8 = 0.0f;
    for (int i = 0; i < 16; ++i) {
        int x0 = (qs[i] & 0x0F) | (((qh >> i) & 1) << 4);
        int x1 = (qs[i] >> 4) | (((qh >> (i + 16)) & 1) << 4);
        sum += (float)x0 * (float)q8[i] + (float)x1 * (float)q8[i + 16];
        sum_q8 += (float)q8[i] + (float)q8[i + 16];
    }
    return d * d8 * sum + m * d8 * sum_q8;
}

__global__ void gemv_q5_1_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BS = 24, BE = 32;
    int num_blocks_row = (K + BE - 1) / BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        sum += vec_dot_q5_1_q8_1(row_ptr + (size_t)bi * BS, x_q8 + bi);
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_q5_1_q8_1(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_q5_1_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_q5_1_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BS = 24, BE = 32;
    int num_blocks_row = (K + BE - 1) / BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        sum += vec_dot_q5_1_q8_1(w_row + (size_t)bi * BS, x_row + bi);
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_q5_1_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_q5_1_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// IQ4_NL GEMV - Q8_1 + table lookup (32 elem/block, 18 bytes)
// ============================================================================

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const uint8_t* __restrict__ block_ptr,
    const block_q8_1_gemv* __restrict__ q8_blk)
{
    float d = __half2float(reinterpret_cast<const __half&>(*(const uint16_t*)block_ptr));
    float d8 = __low2float(q8_blk->ds);

    const int* qs = (const int*)(block_ptr + 2);
    const int* q8i = (const int*)q8_blk->qs;

    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        const int2 v = get_int_from_table_16(qs[j], c_kvalues_iq4nl);
        sumi = forge_dp4a(v.x, q8i[2*j + 0], sumi);
        sumi = forge_dp4a(v.y, q8i[2*j + 1], sumi);
    }

    return d * d8 * (float)sumi;
}

__global__ void gemv_iq4_nl_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BS = 18, BE = 32;
    int num_blocks_row = (K + BE - 1) / BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        sum += vec_dot_iq4_nl_q8_1(row_ptr + (size_t)bi * BS, x_q8 + bi);
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_iq4_nl_q8_1(const float* x, const void* q_weight, float* out,
                               int K, int N, cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_iq4_nl_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_iq4_nl_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BS = 18, BE = 32;
    int num_blocks_row = (K + BE - 1) / BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        sum += vec_dot_iq4_nl_q8_1(w_row + (size_t)bi * BS, x_row + bi);
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_iq4_nl_q8_1_batch(const float* x, const void* q_weight, float* out,
                                     int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_iq4_nl_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// IQ2_XXS GEMV - Q8_1 + 2-bit table lookup (256 elem/block, 66 bytes)
// ============================================================================

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    const uint8_t* bq2 = (const uint8_t*)vbq + kbx * 66;

    const int q2 = get_int_b2(bq2 + 2, iqs);
    const uint8_t* aux8 = (const uint8_t*)&q2;
    const uint32_t aux32 = get_int_b2(bq2 + 2, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*)c_iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = forge_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = forge_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(reinterpret_cast<const __half&>(*(const uint16_t*)bq2)) *
                    __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__global__ void gemv_iq2_xxs_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BE = 256, BS = 66, QI = 16;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = 8;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < QI; ++iqs) {
            sum += vec_dot_iq2_xxs_q8_1(row_ptr, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_iq2_xxs_q8_1(const float* x, const void* q_weight, float* out,
                                int K, int N, cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_iq2_xxs_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_iq2_xxs_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BE = 256, BS = 66, QI = 16;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = 8;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row * q8_stride;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_row + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < QI; ++iqs) {
            sum += vec_dot_iq2_xxs_q8_1(w_row, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_iq2_xxs_q8_1_batch(const float* x, const void* q_weight, float* out,
                                      int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_iq2_xxs_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// IQ2_S GEMV - Q8_1 + 2-bit table lookup (256 elem/block, 82 bytes)
// ============================================================================

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    const uint8_t* bq2 = (const uint8_t*)vbq + kbx * 82;

    const int qs_packed = get_int_b2(bq2 + 2, iqs / 2);
    const uint8_t* qs = (const uint8_t*)&qs_packed;
    const int qh = bq2[2 + 64 + iqs / 2];

    const int signs_packed = get_int_b2(bq2 + 2, 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*)&signs_packed;

    const int ls0 = bq2[2 + 72 + iqs / 2] & 0x0F;
    const int ls1 = bq2[2 + 72 + iqs / 2] >> 4;

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*)(c_iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));

        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                                      ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                                      ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0);

        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = forge_dp4a(grid_l, u0, sumi0);
            sumi0 = forge_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = forge_dp4a(grid_l, u0, sumi1);
            sumi1 = forge_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;

    const float d = __half2float(reinterpret_cast<const __half&>(*(const uint16_t*)bq2)) *
                    __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__global__ void gemv_iq2_s_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BE = 256, BS = 82, QI = 16;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = 8;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < QI; ++iqs) {
            sum += vec_dot_iq2_s_q8_1(row_ptr, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_iq2_s_q8_1(const float* x, const void* q_weight, float* out,
                              int K, int N, cudaStream_t stream) {
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_iq2_s_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_iq2_s_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BE = 256, BS = 82, QI = 16;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = 8;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row * q8_stride;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_row + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < QI; ++iqs) {
            sum += vec_dot_iq2_s_q8_1(w_row, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_iq2_s_q8_1_batch(const float* x, const void* q_weight, float* out,
                                    int M, int K, int N, cudaStream_t stream) {
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_iq2_s_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// IQ2_XS GEMV - Q8_1 + 2-bit table lookup (256 elem/block, 74 bytes)
// block_iq2_xs: fp16 d (2B) + 32*uint16_t qs (64B) + 8B scales = 74B
// Each uint16_t qs: low 9 bits = iq2xs_grid index, high 7 bits = ksigns index
// ============================================================================

// Expand 4 sign bits (low nibble of sb) into 4 byte-wide masks (0xFF per set bit).
static __device__ __forceinline__ int expand_sign_mask_4(uint8_t sb, int half) {
    uint32_t s = (half == 0) ? (sb & 0x0F) : ((sb >> 4) & 0x0F);
    uint32_t bits = (s & 1) | ((s & 2) << 7) | ((s & 4) << 14) | ((s & 8) << 21);
    return __vcmpne4(bits, 0u);
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    const uint8_t* bq2 = (const uint8_t*)vbq + kbx * 74;
    const float d = __half2float(reinterpret_cast<const __half&>(*(const uint16_t*)bq2));
    const uint16_t* qs = (const uint16_t*)(bq2 + 2);     // 32 uint16_t
    const uint8_t* scales = bq2 + 66;                     // 8 bytes

    const int ib32 = iqs;  // 0..7
    const float db0 = (0.5f + (scales[ib32] & 0xF)) * 0.25f;  // l=0,1
    const float db1 = (0.5f + (scales[ib32] >> 4)) * 0.25f;   // l=2,3

    const block_q8_1_gemv& q8 = bq8_1[ib32];
    const float d_q8 = __low2float(q8.ds);

    int sum_l01 = 0, sum_l23 = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint16_t q = qs[4 * ib32 + l];
        const int* grid_ptr = (const int*)&c_iq2xs_grid[q & 511];
        const uint8_t sb = c_ksigns_iq2xs[q >> 9];
        const int smask_lo = expand_sign_mask_4(sb, 0);
        const int smask_hi = expand_sign_mask_4(sb, 4);
        const int sgrid_lo = __vsub4(grid_ptr[0] ^ smask_lo, smask_lo);
        const int sgrid_hi = __vsub4(grid_ptr[1] ^ smask_hi, smask_hi);
        const int u_lo = get_int_b4(q8.qs, l * 2 + 0);
        const int u_hi = get_int_b4(q8.qs, l * 2 + 1);
        const int dp = forge_dp4a(sgrid_lo, u_lo, 0) + forge_dp4a(sgrid_hi, u_hi, 0);
        if (l < 2) sum_l01 += dp; else sum_l23 += dp;
    }
    return d * d_q8 * ((float)sum_l01 * db0 + (float)sum_l23 * db1);
}

__global__ void gemv_iq2_xs_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BE = 256, BS = 74, QI = 8;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = 8;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < QI; ++iqs) {
            sum += vec_dot_iq2_xs_q8_1(row_ptr, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_iq2_xs_q8_1(const float* x, const void* q_weight, float* out,
                              int K, int N, cudaStream_t stream) {
    ensure_gemv_iq_tables();
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_iq2_xs_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_iq2_xs_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BE = 256, BS = 74, QI = 8;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = 8;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row * q8_stride;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_row + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < QI; ++iqs) {
            sum += vec_dot_iq2_xs_q8_1(w_row, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_iq2_xs_q8_1_batch(const float* x, const void* q_weight, float* out,
                                    int M, int K, int N, cudaStream_t stream) {
    ensure_gemv_iq_tables();
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_iq2_xs_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// IQ3_S GEMV - Q8_1 + 3-bit table lookup (256 elem/block, 110 bytes)
// block_iq3_s: fp16 d (2B) + qs[64B] + qh[8B] + signs[32B] + scales[4B] = 110B
// 8 sub-blocks (iqs=0..7), each 32 elems. pair=iqs/2, half=iqs%2.
// qs advances 8B/sub, signs 4B/sub, qh 2B/pair, scales 1B/pair.
// ============================================================================

static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    const uint8_t* bq3 = (const uint8_t*)vbq + kbx * 110;
    const float d = __half2float(reinterpret_cast<const __half&>(*(const uint16_t*)bq3));
    const int pair = iqs / 2;
    const int half = iqs & 1;
    const uint8_t* qs = bq3 + 2 + iqs * 8;             // 8 bytes for this sub
    const uint8_t qh_byte = bq3[2 + 64 + pair * 2 + half];
    const uint8_t* signs = bq3 + 2 + 64 + 8 + iqs * 4;  // 4 bytes for this sub
    const uint8_t scales_byte = bq3[2 + 64 + 8 + 32 + pair];
    const float db_f = (half == 0)
                           ? (float)(1 + 2 * (scales_byte & 0xF))
                           : (float)(1 + 2 * (scales_byte >> 4));

    const block_q8_1_gemv& q8 = bq8_1[iqs];
    const float d_q8 = __low2float(q8.ds);

    int s_all = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int idx1 = qs[2 * l + 0] | ((qh_byte << (8 - 2 * l)) & 256);
        const int idx2 = qs[2 * l + 1] | ((qh_byte << (7 - 2 * l)) & 256);
        const int grid1 = (int)c_iq3s_grid[idx1];
        const int grid2 = (int)c_iq3s_grid[idx2];
        const uint8_t sb = signs[l];
        const int smask_lo = expand_sign_mask_4(sb, 0);
        const int smask_hi = expand_sign_mask_4(sb, 4);
        const int sgrid1 = __vsub4(grid1 ^ smask_lo, smask_lo);
        const int sgrid2 = __vsub4(grid2 ^ smask_hi, smask_hi);
        const int u_lo = get_int_b4(q8.qs, l * 2 + 0);
        const int u_hi = get_int_b4(q8.qs, l * 2 + 1);
        s_all += forge_dp4a(sgrid1, u_lo, 0) + forge_dp4a(sgrid2, u_hi, 0);
    }
    return d * d_q8 * (float)s_all * db_f;
}

__global__ void gemv_iq3_s_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int K, int N)
{
    constexpr int BE = 256, BS = 110, QI = 8;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = 8;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < QI; ++iqs) {
            sum += vec_dot_iq3_s_q8_1(row_ptr, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

void launch_gemv_iq3_s_q8_1(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream) {
    ensure_gemv_iq_tables();
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int grid_blocks = (N + warps_per_block - 1) / warps_per_block;
    gemv_iq3_s_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N);
}

__global__ void gemv_iq3_s_q8_1_batch_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight, float* __restrict__ out, int M, int K, int N)
{
    constexpr int BE = 256, BS = 110, QI = 8;
    int num_blocks_row = (K + BE - 1) / BE;
    const int q8_stride = 8;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = M * N;
    if (warp_id >= total_warps) return;
    int m = warp_id / N, n = warp_id % N;
    const block_q8_1_gemv* x_row = x_q8 + (size_t)m * num_blocks_row * q8_stride;
    const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * BS;
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;
        const block_q8_1_gemv* row_q8 = x_row + (size_t)bi * q8_stride;
#pragma unroll
        for (int iqs = 0; iqs < QI; ++iqs) {
            sum += vec_dot_iq3_s_q8_1(w_row, row_q8, bi, iqs);
        }
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[m * N + n] = sum;
}

void launch_gemv_iq3_s_q8_1_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream) {
    ensure_gemv_iq_tables();
    int k_per_row = (K + 31) / 32;
    int num_q8_blocks = M * k_per_row;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);
    int q8_threads = 256, q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, M * K);
    int warps_per_block = 8, threads = warps_per_block * 32;
    int total_warps = M * N;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemv_iq3_s_q8_1_batch_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, M, K, N);
}

// ============================================================================
// FFN Up Fused: Q3_K gate + Q4_K up (M=1, decode) - Q8_1 pre-quantization + dp4a
// ============================================================================
// Computes: out[i] = SiLU(x @ gate_row_i) * (x @ up_row_i)
// Key optimization: quantize x to Q8_1 only once, shared between gate and up.
// Gate weights are Q3_K, up weights are Q4_K (Q3_K_M mixed quantization).

__global__ void ffn_up_fused_q3k_q4k_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_gate,   // Q3_K weights [N, K]
    const uint8_t* __restrict__ q_up,     // Q4_K weights [N, K]
    float* __restrict__ out, int K, int N)
{
    constexpr int Q3K_BE = 256;
    constexpr int Q3K_BS = 110;
    constexpr int Q4K_BS = 144;
    int num_blocks_row = (K + Q3K_BE - 1) / Q3K_BE;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;

    // Row pointers for gate (Q3_K) and up (Q4_K)
    const uint8_t* gate_row = q_gate + (size_t)warp_id * num_blocks_row * Q3K_BS;
    const uint8_t* up_row   = q_up   + (size_t)warp_id * num_blocks_row * Q4K_BS;

    // Q8_1 strides per super-block
    const int q8_stride_q3k = GEMV_QR3_K * GEMV_QI3_K / GEMV_QI8_1;  // = 8
    const int q8_stride_q4k = GEMV_QR4_K * GEMV_QI4_K / GEMV_QI8_1;  // = 8

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride_q3k;

        // Gate: Q3_K × Q8_1
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI3_K; ++iqs) {
            gate_sum += vec_dot_q3_K_q8_1(gate_row, row_q8, bi, iqs);
        }

        // Up: Q4_K × Q8_1 (same Q8_1 data, different stride since Q4_K block is 144 bytes)
        const block_q8_1_gemv* up_q8 = x_q8 + (size_t)bi * q8_stride_q4k;
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI4_K; iqs += 2) {
            up_sum += vec_dot_q4_K_q8_1(up_row, up_q8, bi, iqs);
        }
    }

    // Warp reduce both sums
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

void launch_ffn_up_fused_q3k_q4k(const float* x, const void* q_gate, const void* q_up,
                                   float* out, int K, int intermediate_dim, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 ONCE (shared by both gate and up)
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks_grid = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks_grid, q8_threads, 0, stream>>>(x, x_q8, K);

    // Step 2: Launch fused gate+up kernel
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int grid_blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q3k_q4k_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_gate), static_cast<const uint8_t*>(q_up),
        out, K, intermediate_dim);
}

// ============================================================================
// FFN Up Fused: Q3_K gate + Q3_K up (M=1, decode) - Q8_1 pre-quantization + dp4a
// ============================================================================
// Computes: out[i] = SiLU(x @ gate_row_i) * (x @ up_row_i)
// Key optimization: quantize x to Q8_1 only once, shared between gate and up.
// Both gate and up weights are Q3_K.

__global__ void ffn_up_fused_q3k_q3k_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_gate,   // Q3_K weights [N, K]
    const uint8_t* __restrict__ q_up,     // Q3_K weights [N, K]
    float* __restrict__ out, int K, int N)
{
    constexpr int Q3K_BE = 256;
    constexpr int Q3K_BS = 110;
    int num_blocks_row = (K + Q3K_BE - 1) / Q3K_BE;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;

    // Row pointers for gate and up (both Q3_K)
    const uint8_t* gate_row = q_gate + (size_t)warp_id * num_blocks_row * Q3K_BS;
    const uint8_t* up_row   = q_up   + (size_t)warp_id * num_blocks_row * Q3K_BS;

    const int q8_stride = GEMV_QR3_K * GEMV_QI3_K / GEMV_QI8_1;  // = 8

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride;

        // Gate: Q3_K × Q8_1
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI3_K; ++iqs) {
            gate_sum += vec_dot_q3_K_q8_1(gate_row, row_q8, bi, iqs);
        }

        // Up: Q3_K × Q8_1 (same Q8_1 data, same stride)
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI3_K; ++iqs) {
            up_sum += vec_dot_q3_K_q8_1(up_row, row_q8, bi, iqs);
        }
    }

    // Warp reduce both sums
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

void launch_ffn_up_fused_q3k_q3k(const float* x, const void* q_gate, const void* q_up,
                                   float* out, int K, int intermediate_dim, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 ONCE (shared by both gate and up)
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks_grid = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks_grid, q8_threads, 0, stream>>>(x, x_q8, K);

    // Step 2: Launch fused gate+up kernel
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int grid_blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q3k_q3k_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_gate), static_cast<const uint8_t*>(q_up),
        out, K, intermediate_dim);
}

// ============================================================================
// FFN Up Fused Q2_K (M=1, decode) - Q8_1 pre-quantization + dp4a
// ============================================================================
// Computes: out[i] = SiLU(x @ gate_row_i) * (x @ up_row_i)
// Key optimization: quantize x to Q8_1 only once, shared between gate and up.

__global__ void ffn_up_fused_q2k_q2k_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_gate,   // Q2_K weights [N, K]
    const uint8_t* __restrict__ q_up,     // Q2_K weights [N, K]
    float* __restrict__ out, int K, int N)
{
    constexpr int Q2K_BE = 256;   // elements per super-block
    constexpr int Q2K_BS = 84;    // bytes per Q2_K block
    int num_blocks_row = (K + Q2K_BE - 1) / Q2K_BE;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;

    const uint8_t* gate_row = q_gate + (size_t)warp_id * num_blocks_row * Q2K_BS;
    const uint8_t* up_row   = q_up   + (size_t)warp_id * num_blocks_row * Q2K_BS;

    const int q8_stride = GEMV_QR2_K * GEMV_QI2_K / GEMV_QI8_1;  // = 8

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride;

        // Gate: Q2_K × Q8_1
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI2_K; ++iqs) {
            gate_sum += vec_dot_q2_K_q8_1(gate_row, row_q8, bi, iqs);
        }

        // Up: Q2_K × Q8_1 (same Q8_1 data, same stride)
#pragma unroll
        for (int iqs = 0; iqs < GEMV_QI2_K; ++iqs) {
            up_sum += vec_dot_q2_K_q8_1(up_row, row_q8, bi, iqs);
        }
    }

    // Warp reduce both sums
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

void launch_ffn_up_fused_q2k_q2k(const float* x, const void* q_gate, const void* q_up,
                                   float* out, int K, int intermediate_dim, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 ONCE (shared by both gate and up)
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks_grid = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks_grid, q8_threads, 0, stream>>>(x, x_q8, K);

    // Step 2: Launch fused gate+up kernel
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int grid_blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q2k_q2k_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_gate), static_cast<const uint8_t*>(q_up),
        out, K, intermediate_dim);
}

// ============================================================================
// FFN Up Fused Q4_0 (M=1, decode) - Q8_1 pre-quantization + dp4a
// ============================================================================
// Computes: out[i] = SiLU(x @ gate_row_i) * (x @ up_row_i)
// Key optimization: quantize x to Q8_1 only once, shared between gate and up.
// Uses dp4a for int8×int8 dot product instead of FP32 scalar.

__global__ void ffn_up_fused_q4_0_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_w1,
    const uint8_t* __restrict__ q_w3,
    float* __restrict__ out, int K, int N)
{
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;

    const uint8_t* w1_row = q_w1 + (size_t)warp_id * num_blocks_row * Q4_0_BLOCK_SIZE;
    const uint8_t* w3_row = q_w3 + (size_t)warp_id * num_blocks_row * Q4_0_BLOCK_SIZE;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        const block_q8_1_gemv* q8_blk = x_q8 + bi;
        float d8 = __low2float(q8_blk->ds);
        const int* q8_int = (const int*)q8_blk->qs;

        // Process gate weights (w1)
        {
            const uint8_t* block_ptr = w1_row + (size_t)bi * Q4_0_BLOCK_SIZE;
            uint16_t scale_bits;
            memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
            float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
            const uint8_t* qs = block_ptr + 2;

            float block_sum = 0.0f;
            int bsum = 0;

#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t qs4;
                memcpy(&qs4, qs + j * 4, 4);
                int q4_low = qs4 & 0x0F0F0F0F;
                int q4_high = (qs4 >> 4) & 0x0F0F0F0F;
                int u_low = q8_int[j];
                int u_high = q8_int[j + 4];
                int dot_low  = forge_dp4a(q4_low, u_low, 0);
                int dot_high = forge_dp4a(q4_high, u_high, 0);
                bsum += forge_dp4a(0x01010101, u_low, 0);
                bsum += forge_dp4a(0x01010101, u_high, 0);
                block_sum += (float)(dot_low + dot_high);
            }
            block_sum -= 8.0f * (float)bsum;
            gate_sum += scale * d8 * block_sum;
        }

        // Process up weights (w3)
        {
            const uint8_t* block_ptr = w3_row + (size_t)bi * Q4_0_BLOCK_SIZE;
            uint16_t scale_bits;
            memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
            float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
            const uint8_t* qs = block_ptr + 2;

            float block_sum = 0.0f;
            int bsum = 0;

#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t qs4;
                memcpy(&qs4, qs + j * 4, 4);
                int q4_low = qs4 & 0x0F0F0F0F;
                int q4_high = (qs4 >> 4) & 0x0F0F0F0F;
                int u_low = q8_int[j];
                int u_high = q8_int[j + 4];
                int dot_low  = forge_dp4a(q4_low, u_low, 0);
                int dot_high = forge_dp4a(q4_high, u_high, 0);
                bsum += forge_dp4a(0x01010101, u_low, 0);
                bsum += forge_dp4a(0x01010101, u_high, 0);
                block_sum += (float)(dot_low + dot_high);
            }
            block_sum -= 8.0f * (float)bsum;
            up_sum += scale * d8 * block_sum;
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

void launch_ffn_up_fused_q4_0_q8_1(const float* x, const void* q_w1, const void* q_w3,
                                     float* out, int K, int intermediate_dim, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 format (shared between gate and up)
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, x_q8, K);

    // Step 2: Launch fused gate+up+SiLU kernel
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int grid_blocks = (intermediate_dim + warps_per_block - 1) / warps_per_block;
    ffn_up_fused_q4_0_q8_1_kernel<<<grid_blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_w1), static_cast<const uint8_t*>(q_w3),
        out, K, intermediate_dim);
}

// ============================================================================
// FFN Down Fused Q4_0 (M=1, decode) - Q8_1 pre-quantization + dp4a + residual
// ============================================================================
// Computes: out[i] = (ffn_mid @ w2_row_i) + residual[i]
// Tiled: each warp processes ROWS_PER_WARP output rows to reuse ffn_mid Q8_1.

template <int ROWS_PER_WARP>
__global__ void ffn_down_fused_q4_0_q8_1_tiled_kernel(
    const block_q8_1_gemv* __restrict__ ffn_mid_q8,
    const uint8_t* __restrict__ q_w2,
    const float* __restrict__ residual,
    float* __restrict__ out, int K, int N)
{
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    int first_row = global_warp_id * ROWS_PER_WARP;
    if (first_row >= N) return;

    float sums[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        sums[r] = 0.0f;

    int blocks_per_thread = (num_blocks_row + 31) / 32;

    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi >= num_blocks_row) break;

        // Q8_1 block shared across all output rows
        const block_q8_1_gemv* q8_blk = ffn_mid_q8 + bi;
        float d8 = __low2float(q8_blk->ds);
        const int* q8_int = (const int*)q8_blk->qs;

        // Precompute Q8_1 contributions for this block (shared across output rows)
        float dp_per_row[ROWS_PER_WARP];  // dot product accumulator per row
        int bsum_per_row[ROWS_PER_WARP];  // bias sum per row
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            dp_per_row[r] = 0.0f;
            bsum_per_row[r] = 0;
        }

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = first_row + r;
            if (row >= N) break;

            const uint8_t* block_ptr = q_w2 + (size_t)row * num_blocks_row * Q4_0_BLOCK_SIZE + (size_t)bi * Q4_0_BLOCK_SIZE;
            uint16_t scale_bits;
            memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
            float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
            const uint8_t* qs = block_ptr + 2;

            float block_sum = 0.0f;
            int bsum = 0;

#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t qs4;
                memcpy(&qs4, qs + j * 4, 4);
                int q4_low = qs4 & 0x0F0F0F0F;
                int q4_high = (qs4 >> 4) & 0x0F0F0F0F;
                int u_low = q8_int[j];
                int u_high = q8_int[j + 4];
                int dot_low  = forge_dp4a(q4_low, u_low, 0);
                int dot_high = forge_dp4a(q4_high, u_high, 0);
                bsum += forge_dp4a(0x01010101, u_low, 0);
                bsum += forge_dp4a(0x01010101, u_high, 0);
                block_sum += (float)(dot_low + dot_high);
            }
            block_sum -= 8.0f * (float)bsum;
            sums[r] += scale * d8 * block_sum;
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

void launch_ffn_down_fused_q4_0_q8_1(const float* ffn_mid, const void* q_w2,
                                       const float* residual, float* out,
                                       int K, int hidden_dim, cudaStream_t stream) {
    // Step 1: Quantize ffn_mid to Q8_1 format
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* ffn_mid_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(ffn_mid, ffn_mid_q8, K);

    // Step 2: Launch tiled fused down+residual kernel
    constexpr int ROWS_PER_WARP = 4;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int num_warps = (hidden_dim + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    ffn_down_fused_q4_0_q8_1_tiled_kernel<ROWS_PER_WARP><<<blocks, threads, 0, stream>>>(
        ffn_mid_q8, static_cast<const uint8_t*>(q_w2), residual, out, K, hidden_dim);
}

// ============================================================================
// Output Proj Q4_0 (M=1, decode, large N) - Q8_1 + dp4a + multi-warp split-K
// ============================================================================
// Specialized for lm_head where N (vocab_size) is very large (e.g., 152064).
// Uses multiple warps per row (Split-K) to keep GPU busy.
// Each warp handles a subset of K-blocks for one row, atomicAdd to combine.
// Uses dp4a for int8×int8 dot product instead of FP32 scalar.

__global__ void output_proj_q4_0_q8_1_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight,
    float* __restrict__ out, int K, int N, int warps_per_row)
{
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int BLOCK_ELEMS = 32;
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
        const uint8_t* block_ptr = row_ptr + (size_t)bi * Q4_0_BLOCK_SIZE;
        const block_q8_1_gemv* q8_blk = x_q8 + bi;

        // Read fp16 scale
        uint16_t scale_bits;
        memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
        float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));

        // Read Q8_1 scale
        float d8 = __low2float(q8_blk->ds);

        const uint8_t* qs = block_ptr + 2;
        const int* q8_int = (const int*)q8_blk->qs;

        float block_sum = 0.0f;
        int bsum = 0;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t qs4;
            memcpy(&qs4, qs + j * 4, 4);
            int q4_low = qs4 & 0x0F0F0F0F;
            int q4_high = (qs4 >> 4) & 0x0F0F0F0F;
            int u_low = q8_int[j];
            int u_high = q8_int[j + 4];
            int dot_low  = forge_dp4a(q4_low, u_low, 0);
            int dot_high = forge_dp4a(q4_high, u_high, 0);
            bsum += forge_dp4a(0x01010101, u_low, 0);
            bsum += forge_dp4a(0x01010101, u_high, 0);
            block_sum += (float)(dot_low + dot_high);
        }
        block_sum -= 8.0f * (float)bsum;
        sum += scale * d8 * block_sum;
    }

    // Warp reduce
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) {
        atomicAdd(&out[row], sum);
    }
}

void launch_output_proj_q4_0_q8_1(const float* x, const void* q_weight, float* out,
                                    int K, int N, cudaStream_t stream) {
    // Step 1: Quantize x to Q8_1 format
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q8_threads = 256;
    int q8_blocks_grid = (num_q8_blocks + q8_threads - 1) / q8_threads;
    quantize_q8_1_kernel<<<q8_blocks_grid, q8_threads, 0, stream>>>(x, x_q8, K);

    // Step 2: Launch output proj kernel with split-K
    int num_blocks_row = (K + 31) / 32;
    int warps_per_row = (num_blocks_row + 31) / 32;
    if (warps_per_row < 1) warps_per_row = 1;
    if (warps_per_row > 16) warps_per_row = 16;

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = N * warps_per_row;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    cudaMemsetAsync(out, 0, N * sizeof(float), stream);
    output_proj_q4_0_q8_1_kernel<<<blocks, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_weight), out, K, N, warps_per_row);
}

}  // namespace cuda
}  // namespace forge
