#pragma once

#include <cstdint>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_ffn_up_fused_q4_0(const float* x, const void* q_w1, const void* q_w3, float* out, int K,
                              int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_0_batch(const float* x, const void* q_w1, const void* q_w3, float* out,
                                    int M, int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_0_batch_gemv(const float* x, const void* q_w1, const void* q_w3,
                                         float* out, int M, int K, int N, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_k(const float* x, const void* q_w1, const void* q_w3, float* out, int K,
                              int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_k_batch(const float* x, const void* q_w1, const void* q_w3, float* out,
                                    int M, int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_down_fused_q4_0(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream = 0);

void launch_output_proj_q4_0(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_qkv_fused_q4_0(const float* x, const void* q_wq, int N_q, const void* q_wk, int N_k,
                           const void* q_wv, int N_v, float* out_q, float* out_k, float* out_v,
                           int K, cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
