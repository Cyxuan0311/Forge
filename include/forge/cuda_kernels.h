#pragma once

#include <cstdint>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_rms_norm(const float* x, const float* weight, float* out, int rows, int cols, float eps,
                     cudaStream_t stream = 0);

void launch_rms_norm_fp16(const void* x, const void* weight, void* out, int rows, int cols,
                          float eps, cudaStream_t stream = 0);

void launch_silu(const float* x, float* out, int n, cudaStream_t stream = 0);

void launch_silu_fp16(const void* x, void* out, int n, cudaStream_t stream = 0);

void launch_gelu(const float* x, float* out, int n, cudaStream_t stream = 0);

void launch_gelu_tanh(const float* x, float* out, int n, cudaStream_t stream = 0);

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

void launch_rope_fp32(const float* q, const float* k, float* q_out, float* k_out, int num_heads,
                      int head_dim, int seq_len, int64_t pos, float theta, cudaStream_t stream = 0);

void launch_rope_gqa(const float* q, const float* k, float* q_out, float* k_out, int num_q_heads,
                     int num_kv_heads, int head_dim, int seq_len, int64_t pos, float theta,
                     cudaStream_t stream = 0);

void launch_expand_kv(const float* kv, float* out, int seq_len, int num_heads, int num_kv_heads,
                      int head_dim, cudaStream_t stream = 0);

void launch_dequant_q4_0(const void* q_data, float* out, int n, cudaStream_t stream = 0);

void launch_dequant_q4_1(const void* q_data, float* out, int n, cudaStream_t stream = 0);

void launch_dequant_q4_k(const void* q_data, float* out, int n, cudaStream_t stream = 0);

void launch_dequant_q4_k_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_gemv_q4_k_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_gemv_q4_k_transB_batch(const float* x, const void* q_weight, float* out, int M, int K,
                                   int N, cudaStream_t stream = 0);

void launch_gemv_transB(const float* x, const float* W, float* out, int K, int N,
                        cudaStream_t stream = 0);

void launch_gemv(const float* x, const float* W, float* out, int K, int N, cudaStream_t stream = 0);

void launch_gemv_q4_0_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_gemv_q4_0_transB_batch(const float* x, const void* q_weight, float* out, int M, int K,
                                   int N, cudaStream_t stream = 0);

void launch_gemv_q4_0_transB_dual(const float* x, const void* q_weight1, int N1,
                                  const void* q_weight2, int N2, float* out, int K,
                                  cudaStream_t stream = 0);

void launch_qkv_fused_q4_0(const float* x, const void* q_wq, int N_q, const void* q_wk, int N_k,
                           const void* q_wv, int N_v, float* out_q, float* out_k, float* out_v,
                           int K, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_0(const float* x, const void* q_w1, const void* q_w3, float* out, int K,
                              int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_0_batch(const float* x, const void* q_w1, const void* q_w3, float* out,
                                    int M, int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_0_batch_gemv(const float* x, const void* q_w1, const void* q_w3,
                                         float* out, int M, int K, int N, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_k_batch(const float* x, const void* q_w1, const void* q_w3, float* out,
                                    int M, int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_k(const float* x, const void* q_w1, const void* q_w3, float* out, int K,
                              int intermediate_dim, cudaStream_t stream = 0);

void launch_gemv_q4_1_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_gemv_q4_1_transB_batch(const float* x, const void* q_weight, float* out, int M, int K,
                                   int N, cudaStream_t stream = 0);

void launch_gemv_q6_k_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_gemv_q6_k_transB_batch(const float* x, const void* q_weight, float* out, int M, int K,
                                   int N, cudaStream_t stream = 0);

void launch_gemv_q3_k_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_gemv_q3_k_transB_batch(const float* x, const void* q_weight, float* out, int M, int K,
                                   int N, cudaStream_t stream = 0);

void launch_dequant_q6_k_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_flash_attention(const float* Q, const float* K, const float* V, float* O, int q_len,
                            int kv_len, int num_heads, int head_dim, bool causal = true,
                            cudaStream_t stream = 0);

void launch_flash_attention_gqa(const float* Q, const float* K, const float* V, float* O, int q_len,
                                int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                bool causal = true, cudaStream_t stream = 0);

void launch_flash_attention_gqa_decode(const float* Q, const float* K, const float* V, float* O,
                                       int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                       cudaStream_t stream = 0);

void launch_quantize_q4_0(const float* data, void* q_data, int n, cudaStream_t stream = 0);

void launch_quantize_q4_0_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                 cudaStream_t stream = 0);

void launch_dequant_q4_0_matrix_to_rows(const void* q_data, float* out, int num_rows, int row_len,
                                        int stride, int start_row, cudaStream_t stream = 0);

void launch_dequant_q4_0_kv(const void* q_key, const void* q_value, float* key_out,
                            float* value_out, int seq_len, int kv_dim, int filled,
                            cudaStream_t stream = 0);

void launch_add_bias(const float* data, const float* bias, float* out, int n,
                     cudaStream_t stream = 0);

void launch_multiply(const float* a, const float* b, float* out, int n, cudaStream_t stream = 0);

void launch_silu_multiply(const float* gate, const float* up, float* out, int n,
                          cudaStream_t stream = 0);

void launch_argmax(const float* data, int32_t* out_idx, int n, cudaStream_t stream = 0);

void launch_dequant_q4_0_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_dequant_q4_1_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_cublas_sgemm(const float* A, const float* B, float* C, int M, int K, int N, bool transB,
                         cudaStream_t stream = 0);

void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, bool transB,
                       cudaStream_t stream = 0);

void launch_ffn_down_fused_q4_0(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream = 0);

void launch_output_proj_q4_0(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
