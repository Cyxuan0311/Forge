#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_flash_attention(const float* Q, const float* K, const float* V, float* O, int q_len,
                            int kv_len, int num_heads, int head_dim,
                            const float* mask = nullptr,
                            bool causal = true,
                            cudaStream_t stream = 0);

void launch_flash_attention_gqa(const float* Q, const float* K, const float* V, float* O, int q_len,
                                int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                const float* mask = nullptr,
                                bool causal = true,
                                cudaStream_t stream = 0);

void launch_flash_attention_gqa_vectorized(const float* Q, const float* K, const float* V, float* O,
                                            int q_len, int kv_len, int num_heads, int num_kv_heads,
                                            int head_dim, const float* mask = nullptr,
                                            bool causal = true, cudaStream_t stream = 0);

void launch_flash_attention_gqa_decode(const float* Q, const float* K, const float* V, float* O,
                                       int kv_len, int num_heads, int num_kv_heads, int head_dim,
                                       const float* mask_row = nullptr,
                                       cudaStream_t stream = 0);

// 6-A-7: native FP16 GQA decode. __half Q/K/V/O storage, FP32 QK^T / online
// softmax / PV, FP16 output. Structural mirror of the FP32 streaming decode
// above (1 block/head, Q in smem, streaming KV, float4-width loads).
// Supported head_dim: 16/32/64/96/128. Larger dims are rejected host-side
// (see flash_attn_gqa_kernel in op_kernels.cpp) — never silently reinterpreted.
void launch_flash_attention_gqa_decode_fp16(const __half* Q, const __half* K, const __half* V,
                                             __half* O, int kv_len, int num_heads,
                                             int num_kv_heads, int head_dim,
                                             const __half* mask_row_or_null = nullptr,
                                             cudaStream_t stream = 0);

// 6-B-6: native BF16-Q GQA decode (decode-first). Mixed storage contract: Q is
// __nv_bfloat16, K/V stay float (authoritative FP32 KV store), all math FP32
// (QK^T / online softmax / PV), output __nv_bfloat16. Structural mirror of
// the FP32/FP16 streaming decode above (1 block/head, Q in smem as float,
// streaming KV). Supported head_dim: 16/32/64/96/128; larger dims rejected
// host-side -- never silently reinterpreted. Mask (if any) is float.
void launch_flash_attention_gqa_decode_bf16q(const __nv_bfloat16* Q, const float* K, const float* V,
                                             __nv_bfloat16* O, int kv_len, int num_heads,
                                             int num_kv_heads, int head_dim,
                                             const float* mask_row_or_null = nullptr,
                                             cudaStream_t stream = 0);

// 6-B-6.7: all-BF16 GQA decode (BF16 KV extension; decode-only). Q/K/V all
// __nv_bfloat16, FP32 math, BF16 output. Same scope/HD set as above; mask (if
// any) is BF16. Mixed K/V dtypes keep loud rejection at the routing layers.
void launch_flash_attention_gqa_decode_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                            const __nv_bfloat16* V, __nv_bfloat16* O, int kv_len,
                                            int num_heads, int num_kv_heads, int head_dim,
                                            const __nv_bfloat16* mask_row_or_null = nullptr,
                                            cudaStream_t stream = 0);

// 6-B-7.0: native BF16 vectorized GQA prefill (E2E unlock). All-BF16 QKV,
// FP32 math, BF16 output; same causal convention as the FP16 entry.
// HD 16/32/64/96/128; larger dims rejected host-side. No wmma.
void launch_flash_attention_gqa_prefill_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                             const __nv_bfloat16* V, __nv_bfloat16* O, int q_len,
                                             int kv_len, int num_heads, int num_kv_heads,
                                             int head_dim, const __nv_bfloat16* mask, bool causal,
                                             cudaStream_t stream = 0);

// 6-A-7.2: native FP16 vectorized GQA prefill. Same dtype contract as decode;
// mirror of launch_flash_attention_gqa_vectorized (Q tile in smem, strided KV
// loop, local_acc kept). HD 16/32/64/96/128; larger dims rejected host-side.
// Phase 2 round 5: production entry -- routes hd=64 causal no-mask prefill
// through the promoted wmma kernel (Br=64/Bc=16/4 warps, round-4 scan winner)
// and falls back to the vectorized kernel below for every other scope.
void launch_flash_attention_gqa_prefill_fp16(const __half* Q, const __half* K, const __half* V,
                                             __half* O, int q_len, int kv_len, int num_heads,
                                             int num_kv_heads, int head_dim,
                                             const __half* mask_or_null = nullptr,
                                             bool causal = true, cudaStream_t stream = 0);

// Explicit vectorized baseline (frozen 6-A-7.2 kernel). Test/bench A/B arm
// vs the promoted wmma path, and the fallback target.
void launch_flash_attention_gqa_prefill_fp16_vectorized(const __half* Q, const __half* K,
                                                        const __half* V, __half* O, int q_len,
                                                        int kv_len, int num_heads,
                                                        int num_kv_heads, int head_dim,
                                                        const __half* mask_or_null = nullptr,
                                                        bool causal = true,
                                                        cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
