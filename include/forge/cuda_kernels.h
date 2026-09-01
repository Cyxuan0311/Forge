#pragma once

#include <cstddef>
#include <cstdint>

#include "forge/types.h"

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
// Forward declaration — full definition in forge/kv_cache.h.
// Used only as a parameter type in the paged KV launch functions below.
enum class KVCacheDType : int;
}  // namespace forge

namespace forge {
namespace cuda {

void launch_rms_norm(const float* x, const float* weight, float* out, int rows, int cols, float eps,
                     cudaStream_t stream = 0);

void launch_rms_norm_unweighted(const float* x, float* out, int rows, int cols, float eps,
                                cudaStream_t stream = 0);

void launch_rms_norm_fp16(const void* x, const void* weight, void* out, int rows, int cols,
                          float eps, cudaStream_t stream = 0);

void launch_silu(const float* x, float* out, int n, cudaStream_t stream = 0);

void launch_silu_fp16(const void* x, void* out, int n, cudaStream_t stream = 0);

void launch_gelu(const float* x, float* out, int n, cudaStream_t stream = 0);

void launch_gelu_tanh(const float* x, float* out, int n, cudaStream_t stream = 0);

void launch_gelu_multiply(const float* gate, const float* up, float* out, int n,
                          cudaStream_t stream = 0);

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

void launch_embedding_q5_k(const void* q_weight, const int32_t* indices, float* out,
                           int num_indices, int embed_dim, int vocab_size, bool transposed = false,
                           cudaStream_t stream = 0);

void launch_embedding_q6_k(const void* q_weight, const int32_t* indices, float* out,
                           int num_indices, int embed_dim, int vocab_size, bool transposed = false,
                           cudaStream_t stream = 0);

void launch_embedding_q2_k(const void* q_weight, const int32_t* indices, float* out,
                           int num_indices, int embed_dim, int vocab_size, bool transposed = false,
                           cudaStream_t stream = 0);

void launch_embedding_q3_k(const void* q_weight, const int32_t* indices, float* out,
                           int num_indices, int embed_dim, int vocab_size, bool transposed = false,
                           cudaStream_t stream = 0);

void launch_rope_fp32(const float* q, const float* k, float* q_out, float* k_out, int num_heads,
                      int head_dim, int seq_len, int64_t pos, float theta, cudaStream_t stream = 0);

void launch_rope_gqa(const float* q, const float* k, float* q_out, float* k_out, int num_q_heads,
                     int num_kv_heads, int head_dim, int seq_len, int64_t pos, float theta,
                     cudaStream_t stream = 0);

void launch_rope_gemma4_gqa(const float* q, const float* k, float* q_out, float* k_out,
                            int num_q_heads, int num_kv_heads, int head_dim, int seq_len,
                            int64_t pos, float theta, const float* freq_factors,
                            cudaStream_t stream = 0);

void launch_rope_gemma4_q_only(const float* q, float* q_out, int num_heads, int head_dim,
                               int seq_len, int64_t pos, float theta, const float* freq_factors,
                               cudaStream_t stream = 0);

void launch_expand_kv(const float* kv, float* out, int seq_len, int num_heads, int num_kv_heads,
                      int head_dim, cudaStream_t stream = 0);

void launch_dequant_q4_0(const void* q_data, float* out, int n, cudaStream_t stream = 0);

void launch_dequant_q4_1(const void* q_data, float* out, int n, cudaStream_t stream = 0);

void launch_dequant_q4_k(const void* q_data, float* out, int n, cudaStream_t stream = 0);

void launch_dequant_q4_k_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

// ---- FP32 GEMV ----
void launch_gemv_transB(const float* x, const float* W, float* out, int K, int N,
                        cudaStream_t stream = 0);

void launch_gemv(const float* x, const float* W, float* out, int K, int N, cudaStream_t stream = 0);

// ---- Q4_0 special GEMV (smem + splitK + dual) ----
void launch_gemv_q4_0_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_gemv_q4_0_transB_dual(const float* x, const void* q_weight1, int N1,
                                  const void* q_weight2, int N2, float* out, int K,
                                  cudaStream_t stream = 0);

// ---- Typed GEMV dispatch tables (indexed by DataType enum value) ----
using GemvFn = void (*)(const float*, const void*, float*, int, int, cudaStream_t);
using GemvBatchFn = void (*)(const float*, const void*, float*, int, int, int, cudaStream_t);

extern const GemvFn gemv_dispatch[21];
extern const GemvBatchFn gemv_batch_dispatch[21];

// ---- Phase 6: MMQ (Matrix-Matrix Quantized) ----
// For large M (>32), replaces dequantize-to-FP32 + cuBLAS with dp4a.
// Pre-quantizes FP32 activations to Q8_1_mmq, then tiled dp4a dot products.
using MmqFn = void (*)(const float*, const void*, float*, int, int, int, cudaStream_t);

extern const MmqFn mmq_dispatch[20];

void launch_mmq_q3_k(const float* x, const void* q_weight, float* out, int M, int K, int N,
                     cudaStream_t stream = 0);
void launch_mmq_q4_k(const float* x, const void* q_weight, float* out, int M, int K, int N,
                     cudaStream_t stream = 0);
void launch_mmq_q5_k(const float* x, const void* q_weight, float* out, int M, int K, int N,
                     cudaStream_t stream = 0);
void launch_mmq_q4_0(const float* x, const void* q_weight, float* out, int M, int K, int N,
                     cudaStream_t stream = 0);
void launch_mmq_q6_k(const float* x, const void* q_weight, float* out, int M, int K, int N,
                     cudaStream_t stream = 0);
void launch_mmq_q2_k(const float* x, const void* q_weight, float* out, int M, int K, int N,
                     cudaStream_t stream = 0);

// ---- Fused kernels ----
void launch_qkv_fused_q4_0(const float* x, const void* q_wq, int N_q, const void* q_wk, int N_k,
                           const void* q_wv, int N_v, float* out_q, float* out_k, float* out_v,
                           int K, cudaStream_t stream = 0);

void launch_qkv_fused_q4_k(const float* x, const void* q_wq, int N_q, const void* q_wk, int N_k,
                           const void* q_wv, int N_v, float* out_q, float* out_k, float* out_v,
                           int K, cudaStream_t stream = 0);

void launch_qkv_fused_q5_k(const float* x, const void* q_wq, int N_q, const void* q_wk, int N_k,
                           const void* q_wv, int N_v, float* out_q, float* out_k, float* out_v,
                           int K, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_0(const float* x, const void* q_w1, const void* q_w3, float* out, int K,
                              int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q5_k(const float* x, const void* q_w1, const void* q_w3, float* out, int K,
                              int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_0_q8_1(const float* x, const void* q_w1, const void* q_w3, float* out,
                                   int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_0_batch(const float* x, const void* q_w1, const void* q_w3, float* out,
                                    int M, int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_0_batch_gemv(const float* x, const void* q_w1, const void* q_w3,
                                         float* out, int M, int K, int N, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_k_batch(const float* x, const void* q_w1, const void* q_w3, float* out,
                                    int M, int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_k(const float* x, const void* q_w1, const void* q_w3, float* out, int K,
                              int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_k_q8_1(const float* x, const void* q_w1, const void* q_w3, float* out,
                                   int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_iq4_xs_q8_1(const float* x, const void* q_w1, const void* q_w3, float* out,
                                     int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_k_geglu(const float* x, const void* q_w1, const void* q_w3, float* out,
                                    int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q4_k_geglu_q8_1(const float* x, const void* q_w1, const void* q_w3,
                                         float* out, int K, int intermediate_dim,
                                         cudaStream_t stream = 0);

void launch_ffn_up_fused_q3k_q4k(const float* x, const void* q_gate, const void* q_up, float* out,
                                 int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q3k_q3k(const float* x, const void* q_gate, const void* q_up, float* out,
                                 int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_ffn_up_fused_q2k_q2k(const float* x, const void* q_gate, const void* q_up, float* out,
                                 int K, int intermediate_dim, cudaStream_t stream = 0);

void launch_dequant_q5_k_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_dequant_q6_k_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_dequant_q2_k_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_dequant_q3_k_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_dequant_q5_0_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_dequant_q5_1_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_dequant_iq2_s_matrix(const void* q_data, float* out, int N, int K,
                                 cudaStream_t stream = 0);

void launch_flash_attention(const float* Q, const float* K, const float* V, float* O, int q_len,
                            int kv_len, int num_heads, int head_dim, const float* mask = nullptr,
                            bool causal = true, cudaStream_t stream = 0);

void launch_flash_attention_gqa(const float* Q, const float* K, const float* V, float* O, int q_len,
                                int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                const float* mask = nullptr, bool causal = true,
                                cudaStream_t stream = 0);

void launch_flash_attention_gqa_decode(const float* Q, const float* K, const float* V, float* O,
                                       int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                       const float* mask_row = nullptr, cudaStream_t stream = 0);

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

void launch_scale(float* data, float s, int n, cudaStream_t stream = 0);

void launch_scale_accumulate(const float* src, float* dst, float s, int n, cudaStream_t stream = 0);

void launch_gelu_tanh_multiply(float* x, const float* y, int n_per, int n_layer, int layer_idx,
                               int seq_len, cudaStream_t stream = 0);

void launch_silu_multiply(const float* gate, const float* up, float* out, int n,
                          cudaStream_t stream = 0);

void launch_split_q_gate(const float* q_full, float* q, float* gate, int seq_len, int num_heads,
                         int head_dim, cudaStream_t stream = 0);

void launch_sigmoid_multiply(const float* gate, float* data, int n, cudaStream_t stream = 0);

void launch_argmax(const float* data, int32_t* out_idx, int n, cudaStream_t stream = 0);

// ---- Repeat Penalty ----
void launch_repeat_penalty(float* logits, const int32_t* token_history, int n_history,
                           float penalty, int vocab_size, cudaStream_t stream = 0);

// ---- GPU Gumbel-max Sampling ----
void launch_gumbel_sample(const float* logits, int32_t* out_token, float temperature, uint64_t seed,
                          int vocab_size, cudaStream_t stream = 0);

// ---- Logit Softcap ----
void launch_logit_softcap(float* logits, float cap, bool apply_softcap, const int* suppress_tokens,
                          int num_suppress, int vocab_size, cudaStream_t stream = 0);

void launch_dequant_q4_0_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_dequant_q4_0_matrix_fp16(const void* q_data, void* out, int N, int K,
                                     cudaStream_t stream = 0);

void launch_dequant_q4_k_matrix_fp16(const void* q_data, void* out, int N, int K,
                                     cudaStream_t stream = 0);

void launch_dequant_q6_k_matrix_fp16(const void* q_data, void* out, int N, int K,
                                     cudaStream_t stream = 0);

void launch_dequant_q4_1_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_cublas_sgemm(const float* A, const float* B, float* C, int M, int K, int N, bool transB,
                         cudaStream_t stream = 0);

void launch_cublas_gemm_fp16_fp32(const float* A, const void* B, float* C, int M, int K, int N,
                                  bool transB, cudaStream_t stream = 0);

void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, bool transB,
                       cudaStream_t stream = 0);

void launch_ffn_down_fused_q4_0(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream = 0);

void launch_ffn_down_fused_q4_0_q8_1(const float* ffn_mid, const void* q_w2, const float* residual,
                                     float* out, int K, int hidden_dim, cudaStream_t stream = 0);

void launch_ffn_down_fused_q4_k(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream = 0);

void launch_ffn_down_fused_q4_k_q8_1(const float* ffn_mid, const void* q_w2, const float* residual,
                                     float* out, int K, int hidden_dim, cudaStream_t stream = 0);

void launch_ffn_down_fused_q5_k(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream = 0);

void launch_ffn_down_fused_q3_k(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream = 0);

void launch_ffn_down_fused_q6_k(const float* ffn_mid, const void* q_w2, const float* residual,
                                float* out, int K, int hidden_dim, cudaStream_t stream = 0);

void launch_output_proj_q4_0(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_output_proj_q4_0_q8_1(const float* x, const void* q_weight, float* out, int K, int N,
                                  cudaStream_t stream = 0);

void launch_output_proj_q4_k(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_output_proj_q5_k(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_output_proj_q6_k(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

// ---- Output Proj Q4_K Cooperative (small K, e.g., K=1536) ----
// Uses cooperative warp processing instead of split-K for better lane utilization
void launch_output_proj_q4_k_cooperative(const float* x, const void* q_weight, float* out, int K,
                                         int N, cudaStream_t stream = 0);

// ---- Attn Proj Q5_K Cooperative (M=1, decode, 100% lane utilization) ----
void launch_attn_proj_q5_k_cooperative(const float* x, const void* q_weight, float* out, int K,
                                       int N, cudaStream_t stream = 0);

// Same but with fused logit softcap + suppress tokens (saves one kernel launch)
void launch_output_proj_q4_k_cooperative_softcap(const float* x, const void* q_weight, float* out,
                                                 int K, int N, float softcap, bool apply_softcap,
                                                 const int* suppress_tokens, int num_suppress,
                                                 cudaStream_t stream = 0);

// ---- KV Cache quantization kernels: F16, Q8_0, Q4_K ----

void launch_quantize_f16_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                cudaStream_t stream = 0);
void launch_dequant_f16_matrix(const void* q_data, float* out, int num_rows, int row_len,
                               cudaStream_t stream = 0);

void launch_quantize_q8_0_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                 cudaStream_t stream = 0);
void launch_dequant_q8_0_matrix(const void* q_data, float* out, int num_rows, int row_len,
                                cudaStream_t stream = 0);

void launch_quantize_q4_k_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                 cudaStream_t stream = 0);

// ---- FP8 KV Cache quantization kernels ----
void launch_quantize_fp8_e4m3_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                     cudaStream_t stream = 0);
void launch_quantize_fp8_e5m2_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                     cudaStream_t stream = 0);
void launch_dequant_fp8_e4m3_matrix(const void* q_data, float* out, int num_rows, int row_len,
                                    cudaStream_t stream = 0);
void launch_dequant_fp8_e5m2_matrix(const void* q_data, float* out, int num_rows, int row_len,
                                    cudaStream_t stream = 0);

// Per-(row, kv_head) scaled FP8 quantize/dequantize (maximizes E4M3/E5M2 mantissa usage).
void launch_quantize_fp8_e4m3_scaled(const float* data, void* q_data, float* scales, int num_rows,
                                     int num_kv_heads, int head_dim, cudaStream_t stream = 0);
void launch_quantize_fp8_e5m2_scaled(const float* data, void* q_data, float* scales, int num_rows,
                                     int num_kv_heads, int head_dim, cudaStream_t stream = 0);
void launch_dequant_fp8_e4m3_scaled(const void* q_data, const float* scales, float* out,
                                    int num_rows, int num_kv_heads, int head_dim,
                                    cudaStream_t stream = 0);
void launch_dequant_fp8_e5m2_scaled(const void* q_data, const float* scales, float* out,
                                    int num_rows, int num_kv_heads, int head_dim,
                                    cudaStream_t stream = 0);
// launch_dequant_q4_k_matrix already declared above

// ---- Fused Flash Attention (decode) ----
// Read quantized KV cache directly without dequantizing the entire layer to FP32.
// Q is FP32; K and V remain in their quantized block format and are dequantized on-the-fly.
//
// q_row_size: bytes per KV row (for the entire kv_dim = num_kv_heads * head_dim)

void launch_fused_flash_attention_gqa_decode_q4_0(const float* Q, const void* q_K, const void* q_V,
                                                  float* O, int kv_len, int num_heads,
                                                  int num_kv_heads, int head_dim, size_t q_row_size,
                                                  const float* mask_row = nullptr,
                                                  cudaStream_t stream = 0);

void launch_fused_flash_attention_gqa_decode_f16(const float* Q, const void* q_K, const void* q_V,
                                                 float* O, int kv_len, int num_heads,
                                                 int num_kv_heads, int head_dim, size_t q_row_size,
                                                 const float* mask_row = nullptr,
                                                 cudaStream_t stream = 0);

void launch_fused_flash_attention_gqa_decode_q8_0(const float* Q, const void* q_K, const void* q_V,
                                                  float* O, int kv_len, int num_heads,
                                                  int num_kv_heads, int head_dim, size_t q_row_size,
                                                  const float* mask_row = nullptr,
                                                  cudaStream_t stream = 0);

// FP8 fused GQA decode (online dequant). q_K/q_V point at the quantized cache.
void launch_fused_flash_attention_gqa_decode_fp8_e4m3(
    const float* Q, const void* q_K, const void* q_V, float* O, int kv_len, int num_heads,
    int num_kv_heads, int head_dim, size_t q_row_size, const float* mask_row = nullptr,
    const float* k_scales = nullptr, const float* v_scales = nullptr, cudaStream_t stream = 0);
void launch_fused_flash_attention_gqa_decode_fp8_e5m2(
    const float* Q, const void* q_K, const void* q_V, float* O, int kv_len, int num_heads,
    int num_kv_heads, int head_dim, size_t q_row_size, const float* mask_row = nullptr,
    const float* k_scales = nullptr, const float* v_scales = nullptr, cudaStream_t stream = 0);

// ---- Paged Flash Attention (Phase 4) ----
// KV cache is held in fixed-size pages; the sequence page table (page_ids)
// maps logical KV index -> physical page id. K/V rows are resolved through
// the page table at access time, never materialized into a contiguous buffer.
//
// launch_kv_scatter: writes n_tokens FP32 K/V rows starting at logical
// position `pos` into the paged cache, quantizing to `dtype` in-place.
//   k_src/v_src:       device FP32 source rows (n_tokens * kv_dim each)
//   k_page_ptrs/v_page_ptrs: device array of page base pointers (per layer)
//   page_ids:          device array, logical page idx -> physical page id
//   k_row_bytes/v_row_bytes: bytes per K/V row (kv_dim * dtype element size)
//
// launch_paged_flash_attention_gqa_decode_*: paged decode (seq_len==1)
//   Q:                 device FP32 query, [num_heads, head_dim]
//   k_page_ptrs/v_page_ptrs / page_ids: as above
//   O:                 device FP32 output, [num_heads, head_dim]
//   q_row_size:        bytes per KV row (full kv_dim = num_kv_heads*head_dim)

void launch_kv_scatter(const float* k_src, const float* v_src, void* const* k_page_ptrs,
                       void* const* v_page_ptrs, const int32_t* page_ids, int n_tokens, int64_t pos,
                       int page_size, int kv_dim, size_t k_row_bytes, size_t v_row_bytes,
                       KVCacheDType dtype, cudaStream_t stream = 0);

void launch_paged_flash_attention_gqa_decode_q4_0(
    const float* Q, void* const* k_page_ptrs, void* const* v_page_ptrs, const int32_t* page_ids,
    float* O, int kv_len, int num_heads, int num_kv_heads, int head_dim, int page_size,
    size_t q_row_size, const float* mask_row = nullptr, cudaStream_t stream = 0);

void launch_paged_flash_attention_gqa_decode_f16(const float* Q, void* const* k_page_ptrs,
                                                 void* const* v_page_ptrs, const int32_t* page_ids,
                                                 float* O, int kv_len, int num_heads,
                                                 int num_kv_heads, int head_dim, int page_size,
                                                 size_t q_row_size, const float* mask_row = nullptr,
                                                 cudaStream_t stream = 0);

void launch_paged_flash_attention_gqa_decode_q8_0(
    const float* Q, void* const* k_page_ptrs, void* const* v_page_ptrs, const int32_t* page_ids,
    float* O, int kv_len, int num_heads, int num_kv_heads, int head_dim, int page_size,
    size_t q_row_size, const float* mask_row = nullptr, cudaStream_t stream = 0);

void launch_paged_flash_attention_gqa_decode_fp8_e4m3(
    const float* Q, void* const* k_page_ptrs, void* const* v_page_ptrs, const int32_t* page_ids,
    float* O, int kv_len, int num_heads, int num_kv_heads, int head_dim, int page_size,
    size_t q_row_size, const float* mask_row = nullptr, cudaStream_t stream = 0);
void launch_paged_flash_attention_gqa_decode_fp8_e5m2(
    const float* Q, void* const* k_page_ptrs, void* const* v_page_ptrs, const int32_t* page_ids,
    float* O, int kv_len, int num_heads, int num_kv_heads, int head_dim, int page_size,
    size_t q_row_size, const float* mask_row = nullptr, cudaStream_t stream = 0);

// ---- MoE Router ----
void launch_moe_router(const float* logits, int* expert_indices, float* expert_weights,
                       float* softmax_buf, int n_expert, int n_expert_used, int seq_len,
                       cudaStream_t stream = 0);

void launch_moe_router_scale(const float* x, const float* scale, float* out, int hidden_dim,
                             float inv_sqrt, float eps, int seq_len, cudaStream_t stream = 0);

// ---- MoE Expert GEMV ----
template <DataType DT>
void launch_moe_expert_gemv(const float* x, const void* q_w_3d, float* out,
                            const int* expert_indices, const float* expert_weights, int K, int N,
                            int n_expert, int n_expert_used, int n_tokens, cudaStream_t stream = 0);

// ---- GeGLU Split ----
void launch_gelu_tanh_multiply_split(const float* gate_up, float* out, int half_dim, int n_tokens,
                                     cudaStream_t stream = 0);

// ---- Grouped IQ2_S MoE (phimoe): all-device, slot-strided ----
// gate+up: computes both projections for every (token,k) slot in one pass,
// writing unscaled per-slot results (routing weights apply only at the down
// projection). q_gate/q_up are [n_expert, N, K] axis-0 expert slabs.
void launch_moe_expert_iq2_s_gateup(const float* x, const void* q_gate, const void* q_up,
                                    float* out_gate, float* out_up, const int* expert_indices,
                                    int K, int N, int n_expert, int n_expert_used, int n_tokens,
                                    cudaStream_t stream = 0);

// down: weighted per-slot accumulate into per-token output (deterministic
// slot order, no atomics). out must be pre-zeroed.
void launch_moe_expert_iq2_s_down(const float* x, const void* q_down, float* out,
                                  const int* expert_indices, const float* expert_weights, int K,
                                  int N, int n_expert, int n_expert_used, int n_tokens,
                                  cudaStream_t stream = 0);

// ---- I-Quant Dequantization (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S, IQ4_NL) ----

void launch_dequant_iq2_xxs_matrix(const void* q_data, float* out, int N, int K,
                                   cudaStream_t stream = 0);

void launch_dequant_iq2_xs_matrix(const void* q_data, float* out, int N, int K,
                                  cudaStream_t stream = 0);

void launch_dequant_iq3_s_matrix(const void* q_data, float* out, int N, int K,
                                 cudaStream_t stream = 0);

void launch_dequant_iq4_nl_matrix(const void* q_data, float* out, int N, int K,
                                  cudaStream_t stream = 0);

// Lazily upload IQ-type lookup tables (grids + sign tables) to constant memory.
// Must be called before any GEMV/MMQ kernel that references them; safe to call
// repeatedly (no-op after the first upload).
void ensure_iq2_xs_tables();
void ensure_iq3_s_tables();

// ---- SSM kernels ----
void launch_ssm_preprocess(const float* alpha, const float* beta, const float* dt_bias,
                           const float* ssm_a, float* gate_out, float* beta_out, int seq_len,
                           int num_v_heads, cudaStream_t stream = 0);

void launch_ssm_conv1d(const float* x, const float* weight, float* conv_state, float* y,
                       int seq_len, int conv_channels, int d_conv, cudaStream_t stream = 0);

void launch_ssm_silu_split(const float* conv_out, float* q, float* k, float* v, int seq_len,
                           int key_dim, int value_dim, cudaStream_t stream = 0);

void launch_ssm_per_head_l2norm(float* data, int seq_len, int num_heads, int head_dim, float eps,
                                cudaStream_t stream = 0);

void launch_ssm_gated_delta_net(const float* q, const float* k, const float* v, const float* gate,
                                const float* beta, float* ssm_state, float* output, int seq_len,
                                int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
                                cudaStream_t stream = 0);

void launch_ssm_gated_norm(float* delta_out, const float* z, const float* norm_w, int seq_len,
                           int head_v_dim, int num_v_heads, float eps, cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
