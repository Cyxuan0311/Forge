#include "nanoinfer/cuda_kernels.h"
#include "cuda_common.h"

namespace nanoinfer {
namespace cuda {

// ---- FP32 Embedding ----

__global__ void embedding_kernel(const float* weight, const int32_t* indices,
                                  float* out, int num_indices, int embed_dim,
                                  int vocab_size, bool transposed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / embed_dim;
    int dim_idx = idx % embed_dim;

    if (token_idx < num_indices) {
        int vocab_idx = indices[token_idx];
        if (vocab_idx < 0 || vocab_idx >= vocab_size) {
            out[idx] = 0.0f;
        } else if (transposed) {
            out[idx] = weight[dim_idx * vocab_size + vocab_idx];
        } else {
            out[idx] = weight[vocab_idx * embed_dim + dim_idx];
        }
    }
}

void launch_embedding_fp32(const float* weight, const int32_t* indices, float* out,
                            int num_indices, int embed_dim, int vocab_size,
                            bool transposed, cudaStream_t stream) {
    int total = num_indices * embed_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    embedding_kernel<<<blocks, threads, 0, stream>>>(weight, indices, out, num_indices, embed_dim, vocab_size, transposed);
}

// ---- Q4_0 Embedding ----

__global__ void embedding_q4_0_kernel(const uint8_t* q_weight, const int32_t* indices,
                                        float* out, int num_indices, int embed_dim,
                                        int vocab_size) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / embed_dim;
    int dim_idx = idx % embed_dim;

    if (token_idx >= num_indices) return;

    int vocab_idx = indices[token_idx];
    if (vocab_idx < 0 || vocab_idx >= vocab_size) {
        out[idx] = 0.0f;
        return;
    }
    int num_blocks_row = (embed_dim + BLOCK_ELEMS - 1) / BLOCK_ELEMS;
    int row_offset = vocab_idx * num_blocks_row * Q4_0_BLOCK_SIZE;

    int bi = dim_idx / BLOCK_ELEMS;
    int j = dim_idx % BLOCK_ELEMS;

    const uint8_t* block_ptr = q_weight + row_offset + bi * Q4_0_BLOCK_SIZE;

    uint16_t scale_bits;
    memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
    float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));

    const uint8_t* qs = block_ptr + sizeof(uint16_t);
    int val = q4_unpack(qs, j);

    out[idx] = static_cast<float>(val) * scale;
}

__global__ void embedding_q4_0_trans_kernel(const uint8_t* q_weight, const int32_t* indices,
                                              float* out, int num_indices, int embed_dim,
                                              int vocab_size) {
    const int Q4_0_BLOCK_SIZE = 18;
    const int BLOCK_ELEMS = 32;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / embed_dim;
    int dim_idx = idx % embed_dim;

    if (token_idx >= num_indices) return;

    int vocab_idx = indices[token_idx];
    if (vocab_idx < 0 || vocab_idx >= vocab_size) {
        out[idx] = 0.0f;
        return;
    }
    int num_blocks_col = (embed_dim + BLOCK_ELEMS - 1) / BLOCK_ELEMS;
    int bi = dim_idx / BLOCK_ELEMS;
    int j = dim_idx % BLOCK_ELEMS;

    const uint8_t* block_ptr = q_weight + bi * Q4_0_BLOCK_SIZE + vocab_idx * num_blocks_col * Q4_0_BLOCK_SIZE;

    uint16_t scale_bits;
    memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
    float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));

    const uint8_t* qs = block_ptr + sizeof(uint16_t);
    int val = q4_unpack(qs, j);

    out[idx] = static_cast<float>(val) * scale;
}

void launch_embedding_q4_0(const void* q_weight, const int32_t* indices, float* out,
                             int num_indices, int embed_dim, int vocab_size,
                             bool transposed, cudaStream_t stream) {
    int total = num_indices * embed_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (transposed) {
        embedding_q4_0_trans_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint8_t*>(q_weight), indices, out, num_indices, embed_dim, vocab_size);
    } else {
        embedding_q4_0_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint8_t*>(q_weight), indices, out, num_indices, embed_dim, vocab_size);
    }
}

// ---- Q4_1 Embedding ----

__global__ void embedding_q4_1_kernel(const uint8_t* q_weight, const int32_t* indices,
                                        float* out, int num_indices, int embed_dim,
                                        int vocab_size) {
    const int Q4_1_BLOCK_SIZE = 20;
    const int BLOCK_ELEMS = 32;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / embed_dim;
    int dim_idx = idx % embed_dim;

    if (token_idx >= num_indices) return;

    int vocab_idx = indices[token_idx];
    if (vocab_idx < 0 || vocab_idx >= vocab_size) {
        out[idx] = 0.0f;
        return;
    }
    int num_blocks_row = (embed_dim + BLOCK_ELEMS - 1) / BLOCK_ELEMS;
    int row_offset = vocab_idx * num_blocks_row * Q4_1_BLOCK_SIZE;

    int bi = dim_idx / BLOCK_ELEMS;
    int j = dim_idx % BLOCK_ELEMS;

    const uint8_t* block_ptr = q_weight + row_offset + bi * Q4_1_BLOCK_SIZE;

    uint16_t d_bits, m_bits;
    memcpy(&d_bits, block_ptr, sizeof(uint16_t));
    memcpy(&m_bits, block_ptr + 2, sizeof(uint16_t));
    float d_val = __half2float(reinterpret_cast<const __half&>(d_bits));
    float m_val = __half2float(reinterpret_cast<const __half&>(m_bits));

    const uint8_t* qs = block_ptr + 4;
    int val = q4_unpack_unsigned(qs, j);

    out[idx] = static_cast<float>(val) * d_val + m_val;
}

__global__ void embedding_q4_1_trans_kernel(const uint8_t* q_weight, const int32_t* indices,
                                              float* out, int num_indices, int embed_dim,
                                              int vocab_size) {
    const int Q4_1_BLOCK_SIZE = 20;
    const int BLOCK_ELEMS = 32;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / embed_dim;
    int dim_idx = idx % embed_dim;

    if (token_idx >= num_indices) return;

    int vocab_idx = indices[token_idx];
    if (vocab_idx < 0 || vocab_idx >= vocab_size) {
        out[idx] = 0.0f;
        return;
    }
    int num_blocks_col = (embed_dim + BLOCK_ELEMS - 1) / BLOCK_ELEMS;
    int bi = dim_idx / BLOCK_ELEMS;
    int j = dim_idx % BLOCK_ELEMS;

    const uint8_t* block_ptr = q_weight + bi * Q4_1_BLOCK_SIZE + vocab_idx * num_blocks_col * Q4_1_BLOCK_SIZE;

    uint16_t d_bits, m_bits;
    memcpy(&d_bits, block_ptr, sizeof(uint16_t));
    memcpy(&m_bits, block_ptr + 2, sizeof(uint16_t));
    float d_val = __half2float(reinterpret_cast<const __half&>(d_bits));
    float m_val = __half2float(reinterpret_cast<const __half&>(m_bits));

    const uint8_t* qs = block_ptr + 4;
    int val = q4_unpack_unsigned(qs, j);

    out[idx] = static_cast<float>(val) * d_val + m_val;
}

void launch_embedding_q4_1(const void* q_weight, const int32_t* indices, float* out,
                             int num_indices, int embed_dim, int vocab_size,
                             bool transposed, cudaStream_t stream) {
    int total = num_indices * embed_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (transposed) {
        embedding_q4_1_trans_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint8_t*>(q_weight), indices, out, num_indices, embed_dim, vocab_size);
    } else {
        embedding_q4_1_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint8_t*>(q_weight), indices, out, num_indices, embed_dim, vocab_size);
    }
}

// ---- Q4_K Embedding ----

__global__ void embedding_q4_k_kernel(const uint8_t* q_weight, const int32_t* indices,
                                        float* out, int num_indices, int embed_dim,
                                        int vocab_size) {
    const int QK_K = 256;
    const int Q4_K_BLOCK_SIZE = 144;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / embed_dim;
    int dim_idx = idx % embed_dim;

    if (token_idx >= num_indices) return;

    int vocab_idx = indices[token_idx];
    if (vocab_idx < 0 || vocab_idx >= vocab_size) {
        out[idx] = 0.0f;
        return;
    }

    int blocks_per_row = (embed_dim + QK_K - 1) / QK_K;
    int row_offset = vocab_idx * blocks_per_row * Q4_K_BLOCK_SIZE;

    int bi = dim_idx / QK_K;
    int j_in_block = dim_idx % QK_K;

    const uint8_t* block_ptr = q_weight + row_offset + bi * Q4_K_BLOCK_SIZE;

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

void launch_embedding_q4_k(const void* q_weight, const int32_t* indices, float* out,
                             int num_indices, int embed_dim, int vocab_size,
                             bool transposed, cudaStream_t stream) {
    int total = num_indices * embed_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    embedding_q4_k_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_weight), indices, out, num_indices, embed_dim, vocab_size);
}

// ---- Q6_K Embedding ----

__global__ void embedding_q6_k_kernel(const uint8_t* q_weight, const int32_t* indices,
                                        float* out, int num_indices, int embed_dim,
                                        int vocab_size) {
    const int QK_K = 256;
    const int Q6_K_BLOCK_SIZE = 210;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / embed_dim;
    int dim_idx = idx % embed_dim;

    if (token_idx >= num_indices) return;

    int vocab_idx = indices[token_idx];
    if (vocab_idx < 0 || vocab_idx >= vocab_size) {
        out[idx] = 0.0f;
        return;
    }

    int blocks_per_row = (embed_dim + QK_K - 1) / QK_K;
    int row_offset = vocab_idx * blocks_per_row * Q6_K_BLOCK_SIZE;

    int bi = dim_idx / QK_K;
    int j_in_block = dim_idx % QK_K;

    const uint8_t* block_ptr = q_weight + row_offset + bi * Q6_K_BLOCK_SIZE;
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

void launch_embedding_q6_k(const void* q_weight, const int32_t* indices, float* out,
                             int num_indices, int embed_dim, int vocab_size,
                             bool transposed, cudaStream_t stream) {
    int total = num_indices * embed_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    embedding_q6_k_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(q_weight), indices, out, num_indices, embed_dim, vocab_size);
}

} // namespace cuda
} // namespace nanoinfer
