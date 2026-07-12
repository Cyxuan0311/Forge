#pragma once

#include <cstdint>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_embedding_fp32(const float* weight, const int32_t* indices, float* out, int num_indices,
                           int embed_dim, int vocab_size, bool transposed, cudaStream_t stream = 0);

void launch_embedding_q4_0(const void* q_weight, const int32_t* indices, float* out,
                           int num_indices, int embed_dim, int vocab_size, bool transposed = false,
                           cudaStream_t stream = 0);

void launch_embedding_q4_1(const void* q_weight, const int32_t* indices, float* out,
                           int num_indices, int embed_dim, int vocab_size, bool transposed = false,
                           cudaStream_t stream = 0);

void launch_embedding_q4_k(const void* q_weight, const int32_t* indices, float* out,
                           int num_indices, int embed_dim, int vocab_size, bool transposed = false,
                           cudaStream_t stream = 0);

void launch_embedding_q6_k(const void* q_weight, const int32_t* indices, float* out,
                           int num_indices, int embed_dim, int vocab_size, bool transposed = false,
                           cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
