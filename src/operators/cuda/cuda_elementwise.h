#pragma once

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_add_bias(const float* data, const float* bias, float* out, int n,
                     cudaStream_t stream = 0);

void launch_multiply(const float* a, const float* b, float* out, int n, cudaStream_t stream = 0);

void launch_silu_multiply(const float* gate, const float* up, float* out, int n,
                          cudaStream_t stream = 0);

void launch_gelu_multiply(const float* gate, const float* up, float* out, int n,
                          cudaStream_t stream = 0);

void launch_split_q_gate(const float* q_full, float* q, float* gate,
                         int seq_len, int num_heads, int head_dim,
                         cudaStream_t stream = 0);

void launch_sigmoid_multiply(const float* gate, float* data, int n,
                             cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
