#pragma once

#include <cstddef>
#include <cstdint>

// CUDA scalar types (__half / __nv_bfloat16) referenced by the launch
// declarations below come from the CUDA toolkit headers on CUDA builds. On
// CPU-only builds (no toolkit) a host-side software fallback with identical
// names/layout/rounding is used instead -- fixes the macOS/Windows/ppc64le CI
// "cuda_bf16.h not found" compile failure.
#if defined(__CUDACC__) || defined(USE_CUDA)
#    include <cuda_bf16.h>
#    include <cuda_fp16.h>
#else
#    include "forge/fp16_compat.h"
#endif

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

// Phase 6-A-2: real native FP16 RMSNorm (FP16 storage/arithmetic, FP32
// reduction). Gated at the operator boundary by forge/dtype_contract.h
// (op_supports_lowp / RMS_NORM).
void launch_rms_norm_fp16(const __half* x, const __half* weight, __half* out, int rows, int cols,
                          float eps, cudaStream_t stream = 0);

// Phase 6-A-2: unweighted native FP16 RMSNorm (out = x * rsqrt(mean(x^2)+eps)).
void launch_rms_norm_unweighted_fp16(const __half* x, __half* out, int rows, int cols, float eps,
                                     cudaStream_t stream = 0);

// Phase 6-B-2: real native BF16 RMSNorm (weighted + unweighted: BF16
// storage/arithmetic, FP32 reduction). Gated at the operator boundary by
// forge/dtype_contract.h (RMS_NORM already whitelisted; require_no_bf16 no
// longer applies to it -- retired op-by-op as native kernels land).
void launch_rms_norm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out,
                           int rows, int cols, float eps, cudaStream_t stream = 0);
void launch_rms_norm_unweighted_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int rows, int cols,
                                     float eps, cudaStream_t stream = 0);

// Phase 6-C-2: real native FP8 E4M3 RMSNorm (weighted + unweighted: FP8
// storage, FP32 reduction/math, FP8 output -- forge E4M3 codec via
// forge/fp8_utils.h). Gated at the operator boundary by forge/dtype_contract.h
// (RMS_NORM retired from the op_supports_fp8 ledger in 6-C-2).
void launch_rms_norm_fp8(const uint8_t* x, const uint8_t* weight, uint8_t* out, int rows, int cols,
                         float eps, cudaStream_t stream = 0);
void launch_rms_norm_unweighted_fp8(const uint8_t* x, uint8_t* out, int rows, int cols, float eps,
                                    cudaStream_t stream = 0);

void launch_silu(const float* x, float* out, int n, cudaStream_t stream = 0);

// Phase 6-A-3: real native FP16 SiLU (FP16 storage, FP32 sigmoid/exp compute,
// FP16 output). Gated at the operator boundary by forge/dtype_contract.h
// (op_supports_lowp / SILU).
void launch_silu_fp16(const __half* x, __half* out, int n, cudaStream_t stream = 0);

void launch_gelu(const float* x, float* out, int n, cudaStream_t stream = 0);

void launch_gelu_tanh(const float* x, float* out, int n, cudaStream_t stream = 0);

// Phase 6-A-4: real native FP16 activation family (FP16 storage, FP32
// transcendental compute, FP16 output) -- mirrors the 6-A-2/6-A-3 policy. Each
// op's gate is op_supports_lowp (whitelisted in forge/dtype_contract.h). The
// gated variants (silu_multiply / gelu_multiply) reuse the SILU / GELU gate.
void launch_gelu_fp16(const __half* x, __half* out, int n, cudaStream_t stream = 0);
void launch_gelu_tanh_fp16(const __half* x, __half* out, int n, cudaStream_t stream = 0);
void launch_silu_multiply_fp16(const __half* gate, const __half* up, __half* out, int n,
                               cudaStream_t stream = 0);
void launch_gelu_multiply_fp16(const __half* gate, const __half* up, __half* out, int n,
                               cudaStream_t stream = 0);

// Phase 6-B-3: real native BF16 activation family (BF16 storage, FP32
// transcendental compute, BF16 output) -- mirrors the 6-A-3/6-A-4 policy.
// SILU / GELU / GELU_TANH are retired from require_no_bf16; the gated variants
// reuse the SILU / GELU gate (no new enum).
void launch_silu_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n, cudaStream_t stream = 0);
void launch_gelu_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n, cudaStream_t stream = 0);
void launch_gelu_tanh_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n,
                           cudaStream_t stream = 0);
void launch_silu_multiply_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                               __nv_bfloat16* out, int n, cudaStream_t stream = 0);
void launch_gelu_multiply_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                               __nv_bfloat16* out, int n, cudaStream_t stream = 0);

// Phase 6-C-3: native FP8 E4M3 activation family (FP8 storage, FP32
// transcendental compute, FP8 output -- mirrors the 6-A-3/6-A-4/6-B-3 policy and
// reuses the 6-C-2-perf vec16 hardware FP8 codec). SILU / GELU / GELU_TANH are
// retired from the op_supports_fp8 ledger; the gated variants reuse the SILU /
// GELU gate (no new enum). E4M3 E4M3FN storage via the hardware cvt codec.
void launch_silu_fp8(const uint8_t* x, uint8_t* out, int n, cudaStream_t stream = 0);
void launch_gelu_fp8(const uint8_t* x, uint8_t* out, int n, cudaStream_t stream = 0);
void launch_gelu_tanh_fp8(const uint8_t* x, uint8_t* out, int n, cudaStream_t stream = 0);
void launch_silu_multiply_fp8(const uint8_t* gate, const uint8_t* up, uint8_t* out, int n,
                              cudaStream_t stream = 0);
void launch_gelu_multiply_fp8(const uint8_t* gate, const uint8_t* up, uint8_t* out, int n,
                              cudaStream_t stream = 0);

void launch_gelu_multiply(const float* gate, const float* up, float* out, int n,
                          cudaStream_t stream = 0);

// Phase 6-A-8.0: native FP16 elementwise ADD / MUL (FP16 storage, FP32 compute
// per element, FP16 output) -- the residual-add and gate*up nodes of the 6-A-8
// end-to-end FP16 loop. Gated by op_supports_lowp (ADD / MUL).
void launch_add_fp16(const __half* a, const __half* b, __half* out, int n,
                     cudaStream_t stream = 0);
void launch_multiply_fp16(const __half* a, const __half* b, __half* out, int n,
                          cudaStream_t stream = 0);

// Phase 6-B-5: native BF16 elementwise ADD / MUL (BF16 storage, FP32 compute
// per element, BF16 output). Gated by op_supports_lowp (ADD / MUL retire from
// require_no_bf16 with this step).
void launch_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int n,
                     cudaStream_t stream = 0);
void launch_multiply_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int n,
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

// Hadamard (Walsh-Hadamard) activation rotation, applied in place to `out` which
// holds `rows` independent vectors of length `order`. `order` must be a power of
// two in 32..512; anything else returns false and leaves the data untouched
// rather than applying something subtly different. Self-inverse: the same call
// both rotates and un-rotates.
bool launch_hadamard_rotate(const float* in, float* out, int rows, int order,
                            cudaStream_t stream = 0);

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

// Partial split-half RoPE (Spark-X2.5): rotate dims [0, n_rot) as pairs
// (x[d], x[d + n_rot/2]), pass through dims [n_rot, head_dim).
void launch_rope_partial_gqa(const float* q, const float* k, float* q_out, float* k_out,
                             int num_q_heads, int num_kv_heads, int head_dim, int n_rot,
                             int seq_len, int64_t pos, float theta, cudaStream_t stream = 0);

// Per-head sigmoid gate multiply: x[s, h*head_dim+d] *= sigmoid(gate[s, h])
// (Spark-X2.5 attn output gate).
void launch_headwise_sigmoid_gate_mul(float* x, const float* gate, int seq_len, int num_heads,
                                      int head_dim, cudaStream_t stream = 0);

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

// ---- Typed GEMV dispatch tables (indexed by DataType enum value) ----
using GemvFn = void (*)(const float*, const void*, float*, int, int, cudaStream_t);
using GemvBatchFn = void (*)(const float*, const void*, float*, int, int, int, cudaStream_t);

extern const GemvFn gemv_dispatch[25];
extern const GemvBatchFn gemv_batch_dispatch[25];

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

// Phase 2-C: Vectorized FP32 prefill (Q tile in shared mem + float4 K/V loads).
// Numerically identical to launch_flash_attention_gqa; prefer via the selector.
void launch_flash_attention_gqa_vectorized(const float* Q, const float* K, const float* V, float* O,
                                            int q_len, int kv_len, int num_heads, int num_kv_heads,
                                            int head_dim, const float* mask = nullptr,
                                            bool causal = true, cudaStream_t stream = 0);

void launch_flash_attention_gqa_decode(const float* Q, const float* K, const float* V, float* O,
                                       int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                       const float* mask_row = nullptr, cudaStream_t stream = 0);

// 6-A-7: native FP16 GQA decode (__half storage, FP32 QK^T/softmax/PV, __half
// output). HD 16/32/64/96/128; larger dims rejected host-side, never fallback.
void launch_flash_attention_gqa_decode_fp16(const __half* Q, const __half* K, const __half* V,
                                            __half* O, int kv_len, int num_heads,
                                            int num_kv_heads, int head_dim,
                                            const __half* mask_row_or_null = nullptr,
                                            cudaStream_t stream = 0);

// 6-B-6: native BF16-Q GQA decode (decode-first). Q is __nv_bfloat16, K/V stay
// float (authoritative FP32 KV store), FP32 math, __nv_bfloat16 output.
// HD 16/32/64/96/128; larger dims rejected host-side, never fallback.
void launch_flash_attention_gqa_decode_bf16q(const __nv_bfloat16* Q, const float* K, const float* V,
                                             __nv_bfloat16* O, int kv_len, int num_heads,
                                             int num_kv_heads, int head_dim,
                                             const float* mask_row_or_null = nullptr,
                                             cudaStream_t stream = 0);

// 6-B-6.7: all-BF16 GQA decode (BF16 KV extension; decode-only). Q/K/V all
// __nv_bfloat16, FP32 math, __nv_bfloat16 output. Same scope/HD set.
void launch_flash_attention_gqa_decode_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                            const __nv_bfloat16* V, __nv_bfloat16* O, int kv_len,
                                            int num_heads, int num_kv_heads, int head_dim,
                                            const __nv_bfloat16* mask_row_or_null = nullptr,
                                            cudaStream_t stream = 0);

// 6-B-7.0: native BF16 vectorized GQA prefill (E2E unlock). All-BF16 QKV,
// FP32 math, BF16 output. HD 16/32/64/96/128; no wmma.
void launch_flash_attention_gqa_prefill_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                             const __nv_bfloat16* V, __nv_bfloat16* O, int q_len,
                                             int kv_len, int num_heads, int num_kv_heads,
                                             int head_dim, const __nv_bfloat16* mask, bool causal,
                                             cudaStream_t stream = 0);

// 6-A-7.2: native FP16 vectorized GQA prefill (same dtype contract as decode).
// HD 16/32/64/96/128; larger dims rejected host-side, never fallback.
// Phase 2 round 5: this symbol is now the EXPLICIT vectorized baseline entry
// (bench A/B arm + fallback). The production entry below routes hd=64 causal
// no-mask prefill through the promoted wmma kernel (Br=64/Bc=16/4 warps).
void launch_flash_attention_gqa_prefill_fp16(const __half* Q, const __half* K, const __half* V,
                                             __half* O, int q_len, int kv_len, int num_heads,
                                             int num_kv_heads, int head_dim,
                                             const __half* mask_or_null = nullptr,
                                             bool causal = true, cudaStream_t stream = 0);

// Frozen 6-A-7.2 vectorized prefill, reachable explicitly. Test/bench A/B arm
// vs the promoted wmma path, and the target of the production entry's
// fallback for every scope outside (hd=64, causal, no mask).
void launch_flash_attention_gqa_prefill_fp16_vectorized(const __half* Q, const __half* K,
                                                        const __half* V, __half* O, int q_len,
                                                        int kv_len, int num_heads,
                                                        int num_kv_heads, int head_dim,
                                                        const __half* mask_or_null = nullptr,
                                                        bool causal = true,
                                                        cudaStream_t stream = 0);

// Phase 2 round 2: __half2 decode candidate (paired __half22float2 conversions
// + hoisted V loads; same FP32 math contract and summation order as the 6-A-7
// baseline; NOT bitwise vs baseline -- FMA contraction differs between the two
// compilations, measured deviation <= 1 FP16 output ULP). Bench/test-only: NOT
// wired into selector or dispatch; production stays on
// launch_flash_attention_gqa_decode_fp16 until the candidate wins a measured A/B.
void launch_flash_attention_gqa_decode_fp16_h2(const __half* Q, const __half* K, const __half* V,
                                               __half* O, int kv_len, int num_heads,
                                               int num_kv_heads, int head_dim,
                                               const __half* mask_row_or_null = nullptr,
                                               cudaStream_t stream = 0);

// Phase 2 round 3: TensorCore (wmma) prefill candidate. f16 operands, f32
// accumulator: QK^T products exact (f16*f16 exact in f32), accumulation f32;
// PV's P operand stored f16 (documented contract amendment -- the one new
// rounding vs the 6-A-7 vectorized baseline). HEAD_DIM=64 only, causal/no
// mask only. Round-3 default shape (Br=32/Bc=32/2 warps); kept for the
// round-3/4 measurement history.
void launch_flash_attention_gqa_prefill_mma_fp16(const __half* Q, const __half* K,
                                                 const __half* V, __half* O, int q_len,
                                                 int kv_len, int num_heads, int num_kv_heads,
                                                 int head_dim, bool causal,
                                                 cudaStream_t stream = 0);

// Phase 2 round 4: explicit-tile-config scan entry for the wmma prefill
// candidate (same contract amendment as above). Scan set: br in {32,64} x
// bc in {16,32,64} x warps in {2,4}, minus br=32 & warps=4 (wmma M=16: each
// warp must own >= 16 query rows). Other combos fail loudly. The round-4
// winner (64,16,4) is now wired into launch_flash_attention_gqa_prefill_fp16.
void launch_flash_attention_gqa_prefill_mma_cfg(const __half* Q, const __half* K,
                                                const __half* V, __half* O, int q_len,
                                                int kv_len, int num_heads, int num_kv_heads,
                                                int head_dim, bool causal, int br, int bc,
                                                int warps, cudaStream_t stream = 0);

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
void launch_dequant_q5_k_matrix_fp16(const void* q_data, void* out, int N, int K,
                                     cudaStream_t stream = 0);
void launch_dequant_q8_0_matrix_fp16(const void* q_data, void* out, int N, int K,
                                     cudaStream_t stream = 0);

// Q3_K joined the FP16 whole-matrix dequant family in Phase 5-D, so
// launch_cublas_gemm_quant_fp16() can serve Q3_K weights. It is deliberately
// not auto-selected: the route measured 3.2x slower than MMQ on sm_89 (see
// tensorcore_eligible() in kernel_selector.cpp). Kept because caching this
// dequant per weight tensor is what should flip the decision later.
void launch_dequant_q3_k_matrix_fp16(const void* q_data, void* out, int N, int K,
                                     cudaStream_t stream = 0);

void launch_dequant_q4_1_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);

void launch_cublas_sgemm(const float* A, const float* B, float* C, int M, int K, int N, bool transB,
                         cudaStream_t stream = 0);

void launch_cublas_gemm_fp16_fp32(const float* A, const void* B, float* C, int M, int K, int N,
                                  bool transB, cudaStream_t stream = 0);

// BF16 x BF16 -> FP32 GEMM on Tensor Cores, with an FP32 activation. The
// mixed-precision companion to launch_cublas_gemm_bf16 (all-BF16 chain): the
// weight stays BF16 in memory and the activation is narrowed into a small
// grow-only scratch, FP32 accumulate, FP32 out. Reaches this path whenever a
// natively-BF16 GGUF weight meets the FP32 activation chain, instead of falling
// to launch_cublas_sgemm (FP32 SIMT, no Tensor Cores).
void launch_cublas_gemm_bf16_fp32(const float* A, const void* B, float* C, int M, int K, int N,
                                  bool transB, cudaStream_t stream = 0);

// FP32 -> BF16 narrowing (round-to-nearest-even). Not a quantization: it only
// drops the low 16 mantissa bits. Used to match an FP32 activation to a BF16
// weight before the Tensor Core GEMM.
void launch_cast_bf16_matrix(const float* data, void* q_data, int num_rows, int row_len,
                             cudaStream_t stream = 0);

// Phase 6-A-5: native FP16 × FP16 → FP16 GEMM (FP32 accumulate on Tensor Cores).
// Storage FP16 in/out, FP32 reduction -- matches the 6-A-2..6-A-4 activation
// policy. Used by the native FP16 matmul route (MUL_MAT_TRANSB) so a weight that
// already lives as Tensor::FP16 flows into cuBLAS without an FP32 detour.
void launch_cublas_gemm_fp16(const __half* A, const __half* B, __half* C, int M, int K, int N,
                             bool transB, cudaStream_t stream = 0);

// Phase 6-B-4: native BF16 x BF16 -> BF16 GEMM (FP32 accumulate on Tensor
// Cores). Storage BF16 in/out, FP32 reduction -- matches the 6-B-2..6-B-3
// activation policy. Used by the native BF16 matmul route (MUL_MAT_TRANSB) so
// a weight that already lives as Tensor::BF16 flows into cuBLAS without an
// FP32 detour. MUL_MAT_TRANSB is retired from require_no_bf16.
void launch_cublas_gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M,
                             int K, int N, bool transB, cudaStream_t stream = 0);

// Phase 3/5 (Tensor Core): dequantize the whole quantized B [N, K] matrix to
// FP16 and run the FP16 cuBLAS GEMM (FP32 accumulate) on Tensor Cores.
// Supported for every dtype with an FP16 matrix dequant
// (Q4_0/Q4_K/Q5_K/Q6_K/Q8_0; Q5_K added in Phase 5-B). Returns false when dt
// has no FP16 dequant path or when built without FORGE_USE_CUBLAS (in which
// case nothing is launched), so callers can fall back to their SIMT path.
bool launch_cublas_gemm_quant_fp16(DataType dt, const float* A, const void* B_q, float* C, int M,
                                   int K, int N, bool transB, cudaStream_t stream);

// Budget (in MB) for the per-weight FP16 dequant cache used by
// launch_cublas_gemm_quant_fp16(). Set from the CLI (--dequant-cache-mb); if
// unset, resolved from the FORGE_DEQUANT_CACHE_MB env var (default 1024). 0
// disables the cache (callers fall back to transient scratch).
void set_dequant_cache_budget_mb(long mb);
long dequant_cache_budget_mb();
// Resolved byte budget for the per-weight FP16 dequant cache (0 disables it).
// Lazily derived from the MB budget above (CLI > FORGE_DEQUANT_CACHE_MB env >
// default 1024); an adaptive plan may re-latch it at runtime via
// set_dequant_cache_budget_mb().
size_t dequant_cache_budget_bytes();

// Drop every per-weight FP16 dequant cache entry (freeing device buffers).
// The cache keys on the raw weight pointer with no content validation, so it
// MUST be invalidated whenever weight memory can be reallocated to new
// contents: model unload/reload, and between tests that reuse addresses.
void dequant_cache_clear();

// Bytes currently resident in the dequant cache. This is the memory the memory
// planner treats as *reclaimable*: it may be counted as available when choosing
// an optional cache budget, and released on demand via dequant_cache_trim_to().
size_t dequant_cache_current_bytes();
// Free cache entries (least-recently-used first) until the cache holds at most
// budget_bytes, returning the number of bytes released. Used to hand optional
// cache memory back before a mandatory (KV / workspace) allocation.
size_t dequant_cache_trim_to(size_t budget_bytes);

// ---- VRAM-aware dispatch accounting --------------------------------------
// The TensorCore prefill path keeps a persistent FP16 dequant working set on top
// of the model + KV weights: a per-weight cache (dequant_cache_budget_bytes) plus
// a grow-only scratch holding the largest single weight's FP16 form. The engine
// informs the allocator of those magnitudes once after loading weights; the
// runtime then feeds them into the GPU memory budget
// (forge::GpuMemoryBudget) to decide the cache level and whether TC is
// affordable (see forge::plan_q3k_execution).
void set_dequant_scratch_peak_bytes(size_t bytes);
size_t dequant_scratch_peak_bytes();
// Sum of the FP16 (N*K*2) dequant of every CUDA quantized weight. The persistent
// cache can never hold more than this, so the cache's contribution to the working
// set is min(total, budget) -- not the full budget (an upper cap).
void set_dequant_total_fp16_bytes(size_t bytes);
size_t dequant_total_fp16_bytes();
// Total persistent TC dequant footprint = min(total_fp16_dequant, cache_budget) + scratch_peak.
size_t dequant_required_bytes();

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

// Mixed-dtype fused GQA decode: K and V may be stored in different formats with
// independent row sizes. k_fmt / v_fmt are 0 = q8_0, 1 = q4_0, 2 = f16.
//
// Returns false for an unsupported (head_dim, format) combination; the caller
// must then fall back to the materialising path and say so, never silently.
bool launch_fused_flash_attention_gqa_decode_mixed(const float* Q, const void* q_K, const void* q_V,
                                                   float* O, int kv_len, int num_heads,
                                                   int num_kv_heads, int head_dim,
                                                   size_t k_row_size, size_t v_row_size,
                                                   int k_fmt, int v_fmt,
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

// ---- KV-tiled flash-attention PREFILL (Phase 2-C tiling + Phase 2-B formats) ----
// One shared kernel template; the KV dtype is a compile-time policy and both
// contiguous and paged layouts are supported. Unlike the kernels above, this
// family does NOT keep a per-thread local_acc[HEAD_DIM] accumulator: KV is
// processed in tiles and the accumulator is distributed over the head dim,
// which is what makes larger head dims viable occupancy-wise.
// mask is the 2-D [q_len, kv_len] mask (or nullptr for causal-only).
// k_scales/v_scales are the per-(row, kv_head) FP8 scale buffers, or nullptr.

// Contiguous FP32 KV (row stride = num_kv_heads * head_dim floats).
void launch_fused_flash_attention_gqa_prefill_fp32(const float* Q, const float* K, const float* V,
                                                    float* O, int q_len, int kv_len, int num_heads,
                                                    int num_kv_heads, int head_dim,
                                                    const float* mask = nullptr,
                                                    bool causal = true, cudaStream_t stream = 0);

// Contiguous quantized KV (row_bytes =
// KVCache::block_nbytes(dtype, num_kv_heads * head_dim)).
#define FORGE_DECL_QUANT_PREFILL(NAME)                                                          \
    void NAME(const float* Q, const void* q_K, const void* q_V, float* O, int q_len, int kv_len, \
              int num_heads, int num_kv_heads, int head_dim, size_t q_row_size,                \
              const float* mask = nullptr, bool causal = true,                                 \
              const float* k_scales = nullptr, const float* v_scales = nullptr,                 \
              cudaStream_t stream = 0)

FORGE_DECL_QUANT_PREFILL(launch_fused_flash_attention_gqa_prefill_q4_0);
FORGE_DECL_QUANT_PREFILL(launch_fused_flash_attention_gqa_prefill_f16);
FORGE_DECL_QUANT_PREFILL(launch_fused_flash_attention_gqa_prefill_q8_0);
FORGE_DECL_QUANT_PREFILL(launch_fused_flash_attention_gqa_prefill_fp8_e4m3);
FORGE_DECL_QUANT_PREFILL(launch_fused_flash_attention_gqa_prefill_fp8_e5m2);

// Paged quantized KV: rows resolve through page_ptrs[page_ids[j / page_size]].
#define FORGE_DECL_PAGED_PREFILL(NAME)                                                        \
    void NAME(const float* Q, void* const* k_pages, void* const* v_pages,                    \
              const int32_t* page_ids, float* O, int q_len, int kv_len, int num_heads,        \
              int num_kv_heads, int head_dim, int page_size, size_t q_row_size,               \
              const float* mask = nullptr, bool causal = true,                                \
              const float* k_scales = nullptr, const float* v_scales = nullptr,               \
              cudaStream_t stream = 0)

FORGE_DECL_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_fp32);
FORGE_DECL_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_q4_0);
FORGE_DECL_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_f16);
FORGE_DECL_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_q8_0);
FORGE_DECL_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_fp8_e4m3);
FORGE_DECL_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_fp8_e5m2);

#undef FORGE_DECL_QUANT_PREFILL
#undef FORGE_DECL_PAGED_PREFILL

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
    size_t q_row_size,         const float* mask_row = nullptr, cudaStream_t stream = 0);

// ---- MoE Router ----
void launch_moe_router(const float* logits, int* expert_indices, float* expert_weights,
                       float* softmax_buf, int n_expert, int n_expert_used, int seq_len,
                       cudaStream_t stream = 0);

// bailingmoe3 语义路由 (sigmoid + bias + 分组 top-k + renorm), 全程设备端。
// bias 为 nullptr 时不加偏置; norm_w 非 0 时权重除以 max(topk_sum, 6.1e-5)。
void launch_moe_router_sigmoid(const float* logits, const float* bias,
                               int* expert_indices, float* expert_weights,
                               int n_expert, int n_expert_used, int n_group,
                               int n_group_used, int norm_w, float w_scale,
                               int seq_len, cudaStream_t stream = 0);

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

// Device-side embedding row gather for IQ3_S tables: dequantizes only the
// selected rows straight from the quantized device buffer. Eliminates the
// blocking indices-D2H + host dequant + H2D round trip of the generic IQ
// fallback (that D2H drains the whole default-stream pipeline every decode
// step). Layout must match launch_dequant_iq3_s_matrix: [N rows, K cols],
// quant row-major; transposed tables pass row=dim, col=vocab (K=vocab_size).
void launch_embedding_iq3_s(const void* q_weight, const int32_t* indices, float* out,
                            int num_indices, int embed_dim, int vocab_size,
                            bool transposed, cudaStream_t stream = 0);

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
