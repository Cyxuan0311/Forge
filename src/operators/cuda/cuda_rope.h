#pragma once

#include <cstdint>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_rope_fp32(const float* q, const float* k, float* q_out, float* k_out,
                      int num_heads, int head_dim, int seq_len, int64_t pos,
                      float theta, cudaStream_t stream = 0);

void launch_rope_gqa(const float* q, const float* k, float* q_out, float* k_out,
                     int num_q_heads, int num_kv_heads, int head_dim, int seq_len, int64_t pos,
                     float theta, cudaStream_t stream = 0);

void launch_expand_kv(const float* kv, float* out,
                      int seq_len, int num_heads, int num_kv_heads, int head_dim,
                      cudaStream_t stream = 0);

} // namespace cuda
} // namespace forge
