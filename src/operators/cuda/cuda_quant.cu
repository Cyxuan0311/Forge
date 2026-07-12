#include "forge/cuda_kernels.h"
#include "cuda_common.h"

namespace forge {
namespace cuda {

// ---- Q4_0 Vector Dequantization ----

__global__ void dequant_q4_0_kernel(const uint8_t* q_data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

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
    dequant_q4_0_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, n);
}

// ---- Q4_1 Vector Dequantization ----

__global__ void dequant_q4_1_kernel(const uint8_t* q_data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

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
    dequant_q4_1_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, n);
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

__global__ void dequant_q4_k_kernel(const uint8_t* __restrict__ q_data, float* __restrict__ out, int n) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

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
    dequant_q4_k_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, n);
}

// ---- Q4_0 Matrix Dequantization ----

__global__ void dequant_q4_0_matrix_kernel(const uint8_t* __restrict__ q_data,
                                             float* __restrict__ out,
                                             int N, int K) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

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

void launch_dequant_q4_0_matrix(const void* q_data, float* out,
                                  int N, int K, cudaStream_t stream) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q4_0_matrix_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, N, K);
}

// ---- Q4_1 Matrix Dequantization ----

__global__ void dequant_q4_1_matrix_kernel(const uint8_t* __restrict__ q_data,
                                             float* __restrict__ out,
                                             int N, int K) {
    const int Q4_1_BLOCK_SIZE = 20;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (K + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

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

void launch_dequant_q4_1_matrix(const void* q_data, float* out,
                                  int N, int K, cudaStream_t stream) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q4_1_matrix_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, N, K);
}

// ---- Q4_K Matrix Dequantization ----

__global__ void dequant_q4_k_matrix_kernel(const uint8_t* __restrict__ q_data,
                                             float* __restrict__ out,
                                             int N, int K) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

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

void launch_dequant_q4_k_matrix(const void* q_data, float* out,
                                  int N, int K, cudaStream_t stream) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q4_k_matrix_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, N, K);
}

// ---- Q6_K Matrix Dequantization ----

__global__ void dequant_q6_k_matrix_kernel(const uint8_t* __restrict__ q_data,
                                             float* __restrict__ out,
                                             int N, int K) {
    const int QK_K = 256;
    const int Q6_K_BLOCK_SIZE = 210;
    int blocks_per_row = (K + QK_K - 1) / QK_K;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

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
        q_val = (int8_t)((ql_sub[l +  0] & 0xF) | (((qh_sub[l] >> 0) & 3) << 4)) - 32;
        scale_val = static_cast<float>(sc_sub[is + 0]);
    } else if (l_full < 64) {
        q_val = (int8_t)((ql_sub[l + 32] & 0xF) | (((qh_sub[l] >> 2) & 3) << 4)) - 32;
        scale_val = static_cast<float>(sc_sub[is + 2]);
    } else if (l_full < 96) {
        q_val = (int8_t)((ql_sub[l +  0] >> 4) | (((qh_sub[l] >> 4) & 3) << 4)) - 32;
        scale_val = static_cast<float>(sc_sub[is + 4]);
    } else {
        q_val = (int8_t)((ql_sub[l + 32] >> 4) | (((qh_sub[l] >> 6) & 3) << 4)) - 32;
        scale_val = static_cast<float>(sc_sub[is + 6]);
    }

    out[idx] = d * scale_val * static_cast<float>(q_val);
}

void launch_dequant_q6_k_matrix(const void* q_data, float* out,
                                  int N, int K, cudaStream_t stream) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q6_k_matrix_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, N, K);
}

// ---- Q4_0 Quantization ----

__global__ void quantize_q4_0_kernel(const float* data, uint8_t* q_data, int n) {
    int block_idx = blockIdx.x;
    int block_size = 32;
    int total_blocks = (n + block_size - 1) / block_size;
    if (block_idx >= total_blocks) return;

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
    if (d == 0.0f) d = -1.0f;
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
                                               uint8_t* __restrict__ q_data,
                                               int num_rows, int row_len) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (row_len + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int row = blockIdx.x;
    if (row >= num_rows) return;

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
        if (d == 0.0f) d = -1.0f;
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

void launch_quantize_q4_0_matrix(const float* data, void* q_data,
                                   int num_rows, int row_len,
                                   cudaStream_t stream) {
    int threads = 256;
    quantize_q4_0_matrix_kernel<<<num_rows, threads, 0, stream>>>(
        data, static_cast<uint8_t*>(q_data), num_rows, row_len);
}

// ---- Q4_0 Matrix Dequantization to strided rows ----

__global__ void dequant_q4_0_matrix_to_rows_kernel(const uint8_t* __restrict__ q_data,
                                                      float* __restrict__ out,
                                                      int num_rows, int row_len,
                                                      int stride, int start_row) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;
    int num_blocks_row = (row_len + BLOCK_ELEMS - 1) / BLOCK_ELEMS;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * row_len;
    if (idx >= total) return;

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

void launch_dequant_q4_0_matrix_to_rows(const void* q_data, float* out,
                                           int num_rows, int row_len, int stride,
                                           int start_row,
                                           cudaStream_t stream) {
    int total = num_rows * row_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q4_0_matrix_to_rows_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_data), out, num_rows, row_len, stride, start_row);
}

// ---- Q4_0 KV Cache dequantization ----

__global__ void dequant_q4_0_kv_kernel(const uint8_t* q_key, const uint8_t* q_value,
                                         float* key_out, float* value_out,
                                         int seq_len, int kv_dim, int filled) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * kv_dim;
    if (idx >= total) return;

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

void launch_dequant_q4_0_kv(const void* q_key, const void* q_value,
                              float* key_out, float* value_out,
                              int seq_len, int kv_dim, int filled,
                              cudaStream_t stream) {
    int total = seq_len * kv_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_q4_0_kv_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_key),
        static_cast<const uint8_t*>(q_value),
        key_out, value_out, seq_len, kv_dim, filled);
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

__global__ void argmax_final_kernel(const float* data, const int32_t* block_out_idx, int32_t* final_idx, int num_blocks) {
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
    if (blocks > 64) blocks = 64;
    size_t shared_mem = threads * sizeof(float) + threads * sizeof(int32_t);
    argmax_reduce_kernel<<<blocks, threads, shared_mem, stream>>>(data, out_idx, n);
    argmax_final_kernel<<<1, 1, 0, stream>>>(data, out_idx, out_idx, blocks);
}

void launch_cublas_sgemm(const float* A, const float* B, float* C,
                           int M, int K, int N, bool transB,
                           cudaStream_t stream) {
#if FORGE_USE_CUBLAS
    cublasHandle_t handle = get_cublas_handle(stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasGemmEx(handle,
                 opB, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 B, transB ? CUDA_R_32F : CUDA_R_32F, transB ? K : N,
                 A, CUDA_R_32F, K,
                 &beta,
                 C, CUDA_R_32F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    launch_gemm_tiled(A, B, C, M, N, K, transB, stream);
#endif
}

} // namespace cuda
} // namespace forge
