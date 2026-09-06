#pragma once

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

// 6-A-7.2: native FP16 vectorized GQA prefill. Same dtype contract as decode;
// mirror of launch_flash_attention_gqa_vectorized (Q tile in smem, strided KV
// loop, local_acc kept). HD 16/32/64/96/128; larger dims rejected host-side.
void launch_flash_attention_gqa_prefill_fp16(const __half* Q, const __half* K, const __half* V,
                                             __half* O, int q_len, int kv_len, int num_heads,
                                             int num_kv_heads, int head_dim,
                                             const __half* mask_or_null = nullptr,
                                             bool causal = true, cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
