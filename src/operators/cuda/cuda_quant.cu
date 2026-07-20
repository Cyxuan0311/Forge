#include "cuda_common.h"
#include "forge/cuda_kernels.h"

namespace forge {
namespace cuda {

// ---- Q4_0 Vector Dequantization ----

__global__ void dequant_q4_0_kernel(const uint8_t* q_data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int block_idx = idx / 32;
    int within_block = idx % 32;

    const int Q4_0_BLOCK_SIZE = 18;
    const uint8_t* block_ptr = q_data + block_idx * Q4_0_BLOCK_SIZE;

    uint16_t scale_bits;
    memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
    float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));

    const uint8_t* qs = block_ptr + sizeof(uint16_t);
    int val = q4_unpack(qs, within_block);

    out[idx] = static_cast<float>(val) * scale;
}

void launch_dequant_q4_0(const void* q_data, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_q4_0_kernel<<<blocks, threads, 0, stream>>>(static_cast<const uint8_t*>(q_data), out,
                                                        n);
}

// ---- Q4_1 Vector Dequantization ----

__global__ void dequant_q4_1_kernel(const uint8_t* q_data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int block_idx = idx / 32;
    int within_block = idx % 32;

    const int Q4_1_BLOCK_SIZE = 20;
    const uint8_t* block_ptr = q_data + block_idx * Q4_1_BLOCK_SIZE;

    uint16_t d_bits, m_bits;
    memcpy(&d_bits, block_ptr, sizeof(uint16_t));
    memcpy(&m_bits, block_ptr + 2, sizeof(uint16_t));
    float d_val = __half2float(reinterpret_cast<const __half&>(d_bits));
    float m_val = __half2float(reinterpret_cast<const __half&>(m_bits));

    const uint8_t* qs = block_ptr + 4;
    int val = q4_unpack_unsigned(qs, within_block);

    out[idx] = static_cast<float>(val) * d_val + m_val;
}

void launch_dequant_q4_1(const void* q_data, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_q4_1_kernel<<<blocks, threads, 0, stream>>>(static_cast<const uint8_t*>(q_data), out,
                                                        n);
}

// ---- Q4_K Vector Dequantization ----

__device__ inline void get_scale_min_k4_cuda(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__global__ void dequant_q4_k_kernel(const uint8_t* __restrict__ q_data, float* __restrict__ out,
                                    int n) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int super_block = idx / QK_K;
    int within_block = idx % QK_K;

    const uint8_t* block_ptr = q_data + super_block * Q4_K_BLOCK_SIZE;

    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits, block_ptr, 2);
    memcpy(&dmin_bits, block_ptr + 2, 2);
    float d = __half2float(reinterpret_cast<const __half&>(d_bits));
    float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
    const uint8_t* scales = block_ptr + 4;
    const uint8_t* qs = block_ptr + 16;

    int chunk = within_block / 64;
    int within_chunk = within_block % 64;
    int is = chunk * 2;

    uint8_t sc, m;
    if (within_chunk < 32) {
        get_scale_min_k4_cuda(is, scales, &sc, &m);
        float d_val = d * sc;
        float m_val = dmin * m;
        int qs_idx = chunk * 32 + within_chunk;
        out[idx] = d_val * static_cast<float>(qs[qs_idx] & 0xF) - m_val;
    } else {
        get_scale_min_k4_cuda(is + 1, scales, &sc, &m);
        float d_val = d * sc;
        float m_val = dmin * m;
        int qs_idx = chunk * 32 + (within_chunk - 32);
        out[idx] = d_val * static_cast<float>(qs[qs_idx] >> 4) - m_val;
    }
}

void launch_dequant_q4_k(const void* q_data, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_q4_k_kernel<<<blocks, threads, 0, stream>>>(static_cast<const uint8_t*>(q_data), out,
                                                        n);
}

// ---- Q4_0 Matrix Dequantization ----

__global__ void dequant_q4_0_matrix_kernel(const uint8_t* __restrict__ q_data,
                                           float* __restrict__ out, int N, int K) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total)
        return;

    int row = idx / K;
    int col = idx % K;

    int bi = col / BLOCK_ELEMS;
    int j = col % BLOCK_ELEMS;

    const uint8_t* row_ptr = q_data + row * num_blocks_row * Q4_0_BLOCK_SIZE;
    const uint8_t* block_ptr = row_ptr + bi * Q4_0_BLOCK_SIZE;

    uint16_t scale_bits;
    memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
    float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));

    const uint8_t* qs = block_ptr + sizeof(uint16_t);
    int val = q4_unpack(qs, j);

    out[idx] = static_cast<float>(val) * scale;
}

void launch_dequant_q4_0_matrix(const void* q_data, float* out, int N, int K, cudaStream_t stream) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q4_0_matrix_kernel<<<blocks, threads, 0, stream>>>(static_cast<const uint8_t*>(q_data),
                                                               out, N, K);
}

// ---- Q4_1 Matrix Dequantization ----

__global__ void dequant_q4_1_matrix_kernel(const uint8_t* __restrict__ q_data,
                                           float* __restrict__ out, int N, int K) {
    const int Q4_1_BLOCK_SIZE = 20;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total)
        return;

    int row = idx / K;
    int col = idx % K;

    int bi = col / BLOCK_ELEMS;
    int j = col % BLOCK_ELEMS;

    const uint8_t* row_ptr = q_data + row * num_blocks_row * Q4_1_BLOCK_SIZE;
    const uint8_t* block_ptr = row_ptr + bi * Q4_1_BLOCK_SIZE;

    uint16_t d_bits, m_bits;
    memcpy(&d_bits, block_ptr, sizeof(uint16_t));
    memcpy(&m_bits, block_ptr + 2, sizeof(uint16_t));
    float d_val = __half2float(reinterpret_cast<const __half&>(d_bits));
    float m_val = __half2float(reinterpret_cast<const __half&>(m_bits));

    const uint8_t* qs = block_ptr + 4;
    int val = q4_unpack_unsigned(qs, j);

    out[idx] = static_cast<float>(val) * d_val + m_val;
}

void launch_dequant_q4_1_matrix(const void* q_data, float* out, int N, int K, cudaStream_t stream) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q4_1_matrix_kernel<<<blocks, threads, 0, stream>>>(static_cast<const uint8_t*>(q_data),
                                                               out, N, K);
}

// ---- Q4_K Matrix Dequantization ----

__global__ void dequant_q4_k_matrix_kernel(const uint8_t* __restrict__ q_data,
                                           float* __restrict__ out, int N, int K) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total)
        return;

    int row = idx / K;
    int col = idx % K;

    int bi = col / QK_K;
    int j_in_block = col % QK_K;

    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q4_K_BLOCK_SIZE;
    const uint8_t* block_ptr = row_ptr + bi * Q4_K_BLOCK_SIZE;

    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits, block_ptr, 2);
    memcpy(&dmin_bits, block_ptr + 2, 2);
    float d = __half2float(reinterpret_cast<const __half&>(d_bits));
    float dmin = __half2float(reinterpret_cast<const __half&>(dmin_bits));
    const uint8_t* scales = block_ptr + 4;
    const uint8_t* qs = block_ptr + 16;

    int group = j_in_block / 64;
    int l = j_in_block % 64;
    int is = group * 2;

    uint8_t sc, m_val;
    if (l < 32) {
        get_scale_min_k4(is, scales, &sc, &m_val);
    } else {
        get_scale_min_k4(is + 1, scales, &sc, &m_val);
    }

    float d1 = d * static_cast<float>(sc);
    float m1 = dmin * static_cast<float>(m_val);

    int qs_offset = group * 32;
    if (l < 32) {
        out[idx] = d1 * static_cast<float>(qs[qs_offset + l] & 0xF) - m1;
    } else {
        out[idx] = d1 * static_cast<float>((qs[qs_offset + l - 32] >> 4) & 0xF) - m1;
    }
}

void launch_dequant_q4_k_matrix(const void* q_data, float* out, int N, int K, cudaStream_t stream) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q4_k_matrix_kernel<<<blocks, threads, 0, stream>>>(static_cast<const uint8_t*>(q_data),
                                                               out, N, K);
}

// ---- Q6_K Matrix Dequantization ----

__global__ void dequant_q6_k_matrix_kernel(const uint8_t* __restrict__ q_data,
                                           float* __restrict__ out, int N, int K) {
    const int QK_K = 256;
    const int Q6_K_BLOCK_SIZE = 210;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total)
        return;

    int row = idx / K;
    int col = idx % K;

    int bi = col / QK_K;
    int j_in_block = col % QK_K;

    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q6_K_BLOCK_SIZE;
    const uint8_t* block_ptr = row_ptr + bi * Q6_K_BLOCK_SIZE;

    const uint8_t* ql = block_ptr;
    const uint8_t* qh = ql + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(qh + 64);
    uint16_t d_bits;
    memcpy(&d_bits, sc + 16, 2);
    float d = __half2float(reinterpret_cast<const __half&>(d_bits));

    int sub = j_in_block / 128;
    int l_full = j_in_block % 128;
    int l = l_full % 32;
    int is = l / 16;

    const uint8_t* ql_sub = ql + sub * 64;
    const uint8_t* qh_sub = qh + sub * 32;
    const int8_t* sc_sub = sc + sub * 8;

    int8_t q_val;
    float scale_val;

    if (l_full < 32) {
        q_val = (int8_t)((ql_sub[l + 0] & 0xF) | (((qh_sub[l] >> 0) & 3) << 4)) - 32;
        scale_val = static_cast<float>(sc_sub[is + 0]);
    } else if (l_full < 64) {
        q_val = (int8_t)((ql_sub[l + 32] & 0xF) | (((qh_sub[l] >> 2) & 3) << 4)) - 32;
        scale_val = static_cast<float>(sc_sub[is + 2]);
    } else if (l_full < 96) {
        q_val = (int8_t)((ql_sub[l + 0] >> 4) | (((qh_sub[l] >> 4) & 3) << 4)) - 32;
        scale_val = static_cast<float>(sc_sub[is + 4]);
    } else {
        q_val = (int8_t)((ql_sub[l + 32] >> 4) | (((qh_sub[l] >> 6) & 3) << 4)) - 32;
        scale_val = static_cast<float>(sc_sub[is + 6]);
    }

    out[idx] = d * scale_val * static_cast<float>(q_val);
}

void launch_dequant_q6_k_matrix(const void* q_data, float* out, int N, int K, cudaStream_t stream) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q6_k_matrix_kernel<<<blocks, threads, 0, stream>>>(static_cast<const uint8_t*>(q_data),
                                                               out, N, K);
}

// ---- Q4_0 Quantization ----

__global__ void quantize_q4_0_kernel(const float* data, uint8_t* q_data, int n) {
    int block_idx = blockIdx.x;
    int block_size = 32;
    int total_blocks = (n + block_size - 1) / block_size;
    if (block_idx >= total_blocks)
        return;

    int start = block_idx * block_size;
    int end = min(start + block_size, n);

    float amax = 0.0f;
    float max_val = 0.0f;
    for (int i = start; i < end; ++i) {
        if (amax < fabsf(data[i])) {
            amax = fabsf(data[i]);
            max_val = data[i];
        }
    }
    float d = max_val / -8.0f;
    if (d == 0.0f)
        d = -1.0f;
    float id = 1.0f / d;

    const int Q4_0_BLOCK_SIZE = 18;
    uint8_t* block_ptr = q_data + block_idx * Q4_0_BLOCK_SIZE;
    __half scale_half = __float2half(d);
    memcpy(block_ptr, &scale_half, sizeof(__half));
    uint8_t* qs = block_ptr + sizeof(__half);

    for (int i = 0; i < 16; ++i) {
        int idx0 = start + i;
        int idx1 = start + i + 16;

        float x0 = (idx0 < end) ? data[idx0] * id : 0.0f;
        float x1 = (idx1 < end) ? data[idx1] * id : 0.0f;

        uint8_t xi0 = min(15, max(0, (int)(x0 + 8.5f)));
        uint8_t xi1 = min(15, max(0, (int)(x1 + 8.5f)));

        qs[i] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
    }
}

void launch_quantize_q4_0(const float* data, void* q_data, int n, cudaStream_t stream) {
    int total_blocks = (n + 31) / 32;
    quantize_q4_0_kernel<<<total_blocks, 1, 0, stream>>>(data, static_cast<uint8_t*>(q_data), n);
}

// ---- Q4_0 Matrix Quantization (row-by-row) ----

__global__ void quantize_q4_0_matrix_kernel(const float* __restrict__ data,
                                            uint8_t* __restrict__ q_data, int num_rows,
                                            int row_len) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (row_len + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int row = blockIdx.x;
    if (row >= num_rows)
        return;

    const float* row_data = data + row * row_len;
    uint8_t* row_q = q_data + row * num_blocks_row * Q4_0_BLOCK_SIZE;

    for (int bi = threadIdx.x; bi < num_blocks_row; bi += blockDim.x) {
        int start = bi * BLOCK_ELEMS;
        int end = min(start + BLOCK_ELEMS, row_len);

        float amax = 0.0f;
        for (int i = start; i < end; ++i) {
            amax = fmaxf(amax, fabsf(row_data[i]));
        }
        float d = amax / -8.0f;
        if (d == 0.0f)
            d = -1.0f;
        float id = 1.0f / d;

        uint8_t* block_ptr = row_q + bi * Q4_0_BLOCK_SIZE;
        __half scale_half = __float2half(d);
        memcpy(block_ptr, &scale_half, sizeof(__half));
        uint8_t* qs = block_ptr + sizeof(__half);

        for (int i = 0; i < 16; ++i) {
            int idx0 = start + i;
            int idx1 = start + i + 16;

            float x0 = (idx0 < end) ? row_data[idx0] * id : 0.0f;
            float x1 = (idx1 < end) ? row_data[idx1] * id : 0.0f;

            uint8_t xi0 = min(15, max(0, (int)(x0 + 8.5f)));
            uint8_t xi1 = min(15, max(0, (int)(x1 + 8.5f)));

            qs[i] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        }
    }
}

void launch_quantize_q4_0_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                 cudaStream_t stream) {
    int threads = 256;
    quantize_q4_0_matrix_kernel<<<num_rows, threads, 0, stream>>>(
        data, static_cast<uint8_t*>(q_data), num_rows, row_len);
}

// ---- Q4_0 Matrix Dequantization to strided rows ----

__global__ void dequant_q4_0_matrix_to_rows_kernel(const uint8_t* __restrict__ q_data,
                                                   float* __restrict__ out, int num_rows,
                                                   int row_len, int stride, int start_row) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (row_len + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * row_len;
    if (idx >= total)
        return;

    int row = idx / row_len;
    int col = idx % row_len;

    int bi = col / BLOCK_ELEMS;
    int j = col % BLOCK_ELEMS;

    const uint8_t* row_ptr = q_data + row * num_blocks_row * Q4_0_BLOCK_SIZE;
    const uint8_t* block_ptr = row_ptr + bi * Q4_0_BLOCK_SIZE;

    uint16_t scale_bits;
    memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
    float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));

    const uint8_t* qs = block_ptr + sizeof(uint16_t);
    int val = q4_unpack(qs, j);

    out[(start_row + row) * stride + col] = static_cast<float>(val) * scale;
}

void launch_dequant_q4_0_matrix_to_rows(const void* q_data, float* out, int num_rows, int row_len,
                                        int stride, int start_row, cudaStream_t stream) {
    int total = num_rows * row_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q4_0_matrix_to_rows_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, num_rows, row_len, stride, start_row);
}

// ---- Q4_0 KV Cache dequantization ----

__global__ void dequant_q4_0_kv_kernel(const uint8_t* q_key, const uint8_t* q_value, float* key_out,
                                       float* value_out, int seq_len, int kv_dim, int filled) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * kv_dim;
    if (idx >= total)
        return;

    int block_idx = idx / 32;
    int within_block = idx % 32;

    const int Q4_0_BLOCK_SIZE = 18;

    auto dequant_val = [&](const uint8_t* q_data) -> float {
        const uint8_t* block_ptr = q_data + block_idx * Q4_0_BLOCK_SIZE;
        uint16_t scale_bits;
        memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
        float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
        const uint8_t* qs = block_ptr + sizeof(uint16_t);
        int val = q4_unpack(qs, within_block);
        return static_cast<float>(val) * scale;
    };

    key_out[filled * kv_dim + idx] = dequant_val(q_key);
    value_out[filled * kv_dim + idx] = dequant_val(q_value);
}

void launch_dequant_q4_0_kv(const void* q_key, const void* q_value, float* key_out,
                            float* value_out, int seq_len, int kv_dim, int filled,
                            cudaStream_t stream) {
    int total = seq_len * kv_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q4_0_kv_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_key), static_cast<const uint8_t*>(q_value), key_out,
        value_out, seq_len, kv_dim, filled);
}

// ---- Argmax ----

__global__ void argmax_reduce_kernel(const float* data, int32_t* block_out_idx, int n) {
    extern __shared__ char smem[];
    float* s_v = reinterpret_cast<float*>(smem);
    int32_t* s_i = reinterpret_cast<int32_t*>(smem + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    s_v[tid] = -1e30f;
    s_i[tid] = 0;

    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        if (data[i] > s_v[tid]) {
            s_v[tid] = data[i];
            s_i[tid] = i;
        }
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_v[tid + s] > s_v[tid]) {
            s_v[tid] = s_v[tid + s];
            s_i[tid] = s_i[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_out_idx[blockIdx.x] = s_i[0];
    }
}

__global__ void argmax_final_kernel(const float* data, const int32_t* block_out_idx,
                                    int32_t* final_idx, int num_blocks) {
    float best_val = -1e30f;
    int32_t best_idx = 0;
    for (int i = 0; i < num_blocks; ++i) {
        int32_t idx = block_out_idx[i];
        if (data[idx] > best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }
    *final_idx = best_idx;
}

void launch_argmax(const float* data, int32_t* out_idx, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 64)
        blocks = 64;
    size_t shared_mem = threads * sizeof(float) + threads * sizeof(int32_t);
    argmax_reduce_kernel<<<blocks, threads, shared_mem, stream>>>(data, out_idx, n);
    argmax_final_kernel<<<1, 1, 0, stream>>>(data, out_idx, out_idx, blocks);
}

void launch_cublas_sgemm(const float* A, const float* B, float* C, int M, int K, int N, bool transB,
                         cudaStream_t stream) {
#if FORGE_USE_CUBLAS
    cublasHandle_t handle = get_cublas_handle(stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasGemmEx(handle, opB, CUBLAS_OP_N, N, M, K, &alpha, B, transB ? CUDA_R_32F : CUDA_R_32F,
                 transB ? K : N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    launch_gemm_tiled(A, B, C, M, N, K, transB, stream);
#endif
}

// =========================================================================
// KV Cache quantization kernels: F16, Q8_0, Q4_K
// =========================================================================

// ---- F16 Quantize (FP32 → F16) ----

__global__ void quantize_f16_matrix_kernel(const float* __restrict__ data,
                                           uint8_t* __restrict__ q_data, int num_rows,
                                           int row_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * row_len;
    if (idx >= total)
        return;

    __half h = __float2half(data[idx]);
    reinterpret_cast<__half*>(q_data)[idx] = h;
}

void launch_quantize_f16_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                cudaStream_t stream) {
    int total = num_rows * row_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    quantize_f16_matrix_kernel<<<blocks, threads, 0, stream>>>(
        data, static_cast<uint8_t*>(q_data), num_rows, row_len);
}

// ---- F16 Dequantize (F16 → FP32) ----

__global__ void dequant_f16_matrix_kernel(const uint8_t* __restrict__ q_data,
                                          float* __restrict__ out, int num_rows, int row_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * row_len;
    if (idx >= total)
        return;

    out[idx] = __half2float(reinterpret_cast<const __half*>(q_data)[idx]);
}

void launch_dequant_f16_matrix(const void* q_data, float* out, int num_rows, int row_len,
                               cudaStream_t stream) {
    int total = num_rows * row_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_f16_matrix_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, num_rows, row_len);
}

// ---- Q8_0 Quantize (FP32 → Q8_0) ----
// Block format: fp16 d (2B) + int8 qs[32] (32B) = 34B per 32 elements

__global__ void quantize_q8_0_matrix_kernel(const float* __restrict__ data,
                                            uint8_t* __restrict__ q_data, int num_rows,
                                            int row_len) {
    const int Q8_0_BLOCK_SIZE = 34;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (row_len + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int row = blockIdx.x;
    if (row >= num_rows)
        return;

    const float* row_data = data + row * row_len;
    uint8_t* row_q = q_data + row * num_blocks_row * Q8_0_BLOCK_SIZE;

    for (int bi = threadIdx.x; bi < num_blocks_row; bi += blockDim.x) {
        int start = bi * BLOCK_ELEMS;
        int end = min(start + BLOCK_ELEMS, row_len);

        float amax = 0.0f;
        for (int i = start; i < end; ++i) {
            amax = fmaxf(amax, fabsf(row_data[i]));
        }
        float d = amax / 127.0f;
        if (d == 0.0f)
            d = 1.0f;
        float id = 1.0f / d;

        uint8_t* block_ptr = row_q + bi * Q8_0_BLOCK_SIZE;
        __half d_half = __float2half(d);
        memcpy(block_ptr, &d_half, sizeof(__half));
        int8_t* qs = reinterpret_cast<int8_t*>(block_ptr + sizeof(__half));

        for (int i = 0; i < BLOCK_ELEMS; ++i) {
            int idx = start + i;
            if (idx < end) {
                float v = row_data[idx] * id;
                v = fmaxf(-128.0f, fminf(127.0f, v));
                qs[i] = static_cast<int8_t>(roundf(v));
            } else {
                qs[i] = 0;
            }
        }
    }
}

void launch_quantize_q8_0_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                 cudaStream_t stream) {
    int threads = 256;
    quantize_q8_0_matrix_kernel<<<num_rows, threads, 0, stream>>>(
        data, static_cast<uint8_t*>(q_data), num_rows, row_len);
}

// ---- Q8_0 Dequantize (Q8_0 → FP32) ----

__global__ void dequant_q8_0_matrix_kernel(const uint8_t* __restrict__ q_data,
                                           float* __restrict__ out, int num_rows, int row_len) {
    const int Q8_0_BLOCK_SIZE = 34;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (row_len + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * row_len;
    if (idx >= total)
        return;

    int row = idx / row_len;
    int col = idx % row_len;

    int bi = col / BLOCK_ELEMS;
    int j = col % BLOCK_ELEMS;

    const uint8_t* row_ptr = q_data + row * num_blocks_row * Q8_0_BLOCK_SIZE;
    const uint8_t* block_ptr = row_ptr + bi * Q8_0_BLOCK_SIZE;

    uint16_t d_bits;
    memcpy(&d_bits, block_ptr, sizeof(uint16_t));
    float d = __half2float(reinterpret_cast<const __half&>(d_bits));

    const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + sizeof(uint16_t));
    out[idx] = static_cast<float>(qs[j]) * d;
}

void launch_dequant_q8_0_matrix(const void* q_data, float* out, int num_rows, int row_len,
                                cudaStream_t stream) {
    int total = num_rows * row_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q8_0_matrix_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, num_rows, row_len);
}

// ---- Q4_K Quantize (FP32 → Q4_K) ----
// Simplified: uniform scale per 256-element block (dmin=0, all sub-scales=1)
// Block format: fp16 d (2B) + fp16 dmin (2B) + scales[12] + qs[128] = 144B

__global__ void quantize_q4_k_matrix_kernel(const float* __restrict__ data,
                                            uint8_t* __restrict__ q_data, int num_rows,
                                            int row_len) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int num_blocks_row = (row_len + QK_K - 1) / QK_K;

    int row = blockIdx.x;
    if (row >= num_rows)
        return;

    const float* row_data = data + row * row_len;
    uint8_t* row_q = q_data + row * num_blocks_row * Q4_K_BLOCK_SIZE;

    for (int bi = threadIdx.x; bi < num_blocks_row; bi += blockDim.x) {
        int start = bi * QK_K;
        int end = min(start + QK_K, row_len);

        float amax = 0.0f;
        for (int i = start; i < end; ++i) {
            amax = fmaxf(amax, fabsf(row_data[i]));
        }
        float d = amax / 7.0f;
        if (d == 0.0f)
            d = 1.0f;
        float id = 1.0f / d;

        uint8_t* block_ptr = row_q + bi * Q4_K_BLOCK_SIZE;
        __half d_half = __float2half(d);
        __half dmin_half = __float2half(0.0f);
        memcpy(block_ptr, &d_half, sizeof(__half));
        memcpy(block_ptr + 2, &dmin_half, sizeof(__half));

        // Set all 6-bit scales to sc=1, m=0 → encoding: (m<<4)|sc = 0x01
        memset(block_ptr + 4, 0x01, 12);

        uint8_t* qs = block_ptr + 16;
        for (int i = 0; i < 128; ++i) {
            int idx0 = start + i;
            int idx1 = start + i + 128;
            float x0 = (idx0 < end) ? row_data[idx0] * id : 0.0f;
            float x1 = (idx1 < end) ? row_data[idx1] * id : 0.0f;
            uint8_t xi0 = min(15, max(0, (int)(x0 + 8.5f)));
            uint8_t xi1 = min(15, max(0, (int)(x1 + 8.5f)));
            qs[i] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        }
    }
}

void launch_quantize_q4_k_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                 cudaStream_t stream) {
    int threads = 256;
    quantize_q4_k_matrix_kernel<<<num_rows, threads, 0, stream>>>(
        data, static_cast<uint8_t*>(q_data), num_rows, row_len);
}

// =========================================================================
// IQ2_XXS CUDA Dequantization
// block_iq2_xxs: fp16 d (2B) + 32*uint16_t qs (64B) = 66B per 256 elements
// Reference: llama.cpp convert.cu dequantize_block_iq2_xxs
// =========================================================================

static const uint8_t h_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint8_t h_kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static const uint64_t h_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

// Constant memory for IQ2_XXS lookup tables
__constant__ uint8_t c_ksigns_iq2xs[128];
__constant__ uint8_t c_kmask_iq2xs[8];
__constant__ uint64_t c_iq2xxs_grid[256];

static bool iq2_xxs_tables_uploaded = false;

static void ensure_iq2_xxs_tables() {
    if (iq2_xxs_tables_uploaded) return;
    cudaMemcpyToSymbol(c_ksigns_iq2xs, h_ksigns_iq2xs, sizeof(h_ksigns_iq2xs));
    cudaMemcpyToSymbol(c_kmask_iq2xs, h_kmask_iq2xs, sizeof(h_kmask_iq2xs));
    cudaMemcpyToSymbol(c_iq2xxs_grid, h_iq2xxs_grid, sizeof(h_iq2xxs_grid));
    iq2_xxs_tables_uploaded = true;
}

// IQ2_XXS matrix dequantization kernel
// 32 threads per block, each block handles one 256-element block
__global__ void dequant_iq2_xxs_matrix_kernel(const uint8_t* __restrict__ q_data,
                                               float* __restrict__ out, int N, int K) {
    const int QK_K = 256;
    const int IQ2_XXS_BLOCK_SIZE = 66;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * K;
    if (idx >= total) return;

    int row = idx / K;
    int col = idx % K;

    int bi = col / QK_K;
    int col_in_block = col % QK_K;

    const uint8_t* row_ptr = q_data + (int64_t)row * blocks_per_row * IQ2_XXS_BLOCK_SIZE;
    const uint8_t* block_ptr = row_ptr + bi * IQ2_XXS_BLOCK_SIZE;

    float d = __half2float(reinterpret_cast<const __half&>(*reinterpret_cast<const uint16_t*>(block_ptr)));
    const uint16_t* qs = reinterpret_cast<const uint16_t*>(block_ptr + 2);

    int ib32 = col_in_block / 32;
    int l = (col_in_block % 32) / 8;
    int j = col_in_block % 8;

    const uint16_t* q2 = qs + 4 * ib32;
    uint32_t aux32[2];
    memcpy(aux32, q2, 2 * sizeof(uint32_t));
    const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);

    const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
    const float db = d * ls * 0.25f;

    const uint8_t* grid = reinterpret_cast<const uint8_t*>(&c_iq2xxs_grid[aux8[l]]);
    const uint8_t signs = c_ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];

    out[idx] = db * static_cast<float>(grid[j]) * (signs & c_kmask_iq2xs[j] ? -1.f : 1.f);
}

void launch_dequant_iq2_xxs_matrix(const void* q_data, float* out, int N, int K,
                                    cudaStream_t stream) {
    ensure_iq2_xxs_tables();
    int64_t total = (int64_t)N * K;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    dequant_iq2_xxs_matrix_kernel<<<(int)blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, N, K);
}

// =========================================================================
// IQ4_NL CUDA Dequantization
// block_iq4_nl: fp16 d (2B) + 16*qs nibble-packed (16B) = 18B per 32 elements
// Uses non-linear lookup table kvalues_iq4nl
// =========================================================================

__constant__ int8_t c_kvalues_iq4nl[16];

static bool iq4_nl_tables_uploaded = false;

static void ensure_iq4_nl_tables() {
    if (iq4_nl_tables_uploaded) return;
    static const int8_t h_kvalues_iq4nl[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };
    cudaMemcpyToSymbol(c_kvalues_iq4nl, h_kvalues_iq4nl, sizeof(h_kvalues_iq4nl));
    iq4_nl_tables_uploaded = true;
}

__global__ void dequant_iq4_nl_matrix_kernel(const uint8_t* __restrict__ q_data,
                                              float* __restrict__ out, int N, int K) {
    const int IQ4_NL_BLOCK_SIZE = 18;
    int blocks_per_row = (K + 31) / 32;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

    int row = idx / K;
    int col = idx % K;

    int bi = col / 32;
    int col_in_block = col % 32;

    const uint8_t* row_ptr = q_data + (int64_t)row * blocks_per_row * IQ4_NL_BLOCK_SIZE;
    const uint8_t* block_ptr = row_ptr + bi * IQ4_NL_BLOCK_SIZE;

    float d = __half2float(reinterpret_cast<const __half&>(*reinterpret_cast<const uint16_t*>(block_ptr)));
    const uint8_t* qs = block_ptr + 2;

    // IQ4_NL: qs[i] low nibble → element i, qs[i] high nibble → element i+16
    int nibble_idx = col_in_block < 16 ? col_in_block : col_in_block - 16;
    int8_t val;
    if (col_in_block < 16) {
        val = c_kvalues_iq4nl[qs[nibble_idx] & 0x0F];
    } else {
        val = c_kvalues_iq4nl[(qs[nibble_idx] >> 4) & 0x0F];
    }
    out[idx] = d * static_cast<float>(val);
}

void launch_dequant_iq4_nl_matrix(const void* q_data, float* out, int N, int K,
                                   cudaStream_t stream) {
    ensure_iq4_nl_tables();
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_iq4_nl_matrix_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, N, K);
}

}  // namespace cuda
}  // namespace forge
