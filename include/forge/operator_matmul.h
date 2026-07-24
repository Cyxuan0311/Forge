#pragma once

#include "tensor.h"

namespace forge {
namespace ops {

TensorPtr matmul(const TensorPtr& a, const TensorPtr& b, const TensorPtr& bias = nullptr);
TensorPtr matmul_transB(const TensorPtr& a, const TensorPtr& b, const TensorPtr& bias = nullptr);
TensorPtr matmul_transB_dual(const TensorPtr& a, const TensorPtr& b1, const TensorPtr& b2);
TensorPtr ffn_up_fused(const TensorPtr& input, const TensorPtr& w1, const TensorPtr& w3,
                       int intermediate_dim);

// CPU-only fused kernels for decode optimization.
// These read the input vector once instead of 2-3 times, saving memory bandwidth.

// Fused QKV projection: computes Q, K, V from a single input read.
// All three weight tensors must be Q4_0 with the same K dimension.
// Returns a tensor of shape [1, N_q + N_k + N_v] with Q, K, V concatenated.
TensorPtr matmul_transB_fused_qkv_q4_0(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv);

// Fused QKV projection for Q4_K weights: reads input once, shares Q8_K quantization
// across Q, K, V projections. Returns concatenated [1, N_q+N_k+N_v].
TensorPtr matmul_transB_fused_qkv_q4_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv);

// Fused FFN gate+up: computes SiLU(gate) * up from a single input read.
// Both weight tensors must be Q4_0 with the same K dimension and N.
TensorPtr matmul_transB_fused_ffn_up_q4_0(const TensorPtr& input, const TensorPtr& w_gate,
                                          const TensorPtr& w_up);

// Fused FFN down-projection + residual: computes input @ weight + residual.
// Saves 1 intermediate tensor write + read per layer.
TensorPtr matmul_transB_fused_ffn_down_residual_q4_0(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual);
TensorPtr matmul_transB_fused_ffn_down_residual_q4_1(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual);

// Fused attention output projection + residual: computes attn_out @ wo + residual.
// Fuses the attention output projection with the pre-attention residual add.
// Weight must be Q4_0.
TensorPtr matmul_transB_fused_attn_proj_residual_q4_0(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual);

// Q4_K fused FFN gate+up: reads input once, produces SiLU(gate)*up directly.
// Both weight tensors must be Q4_K with the same K and N.
TensorPtr matmul_transB_fused_ffn_up_q4_k(const TensorPtr& input, const TensorPtr& w_gate,
                                          const TensorPtr& w_up);

// Q5_K fused FFN gate+up: reads input once, quantizes to Q8_K one time,
// then computes gate and up dot products in parallel, producing SiLU(gate)*up.
// Both weight tensors must be Q5_K with the same K and N.
TensorPtr matmul_transB_fused_ffn_up_q5_k(const TensorPtr& input, const TensorPtr& w_gate,
                                          const TensorPtr& w_up);

// Q2_K fused FFN gate+up: reads input once, quantizes to Q8_K one time,
// then computes gate and up dot products in parallel, producing SiLU(gate)*up.
TensorPtr matmul_transB_fused_ffn_up_q2_k(const TensorPtr& input, const TensorPtr& w_gate,
                                          const TensorPtr& w_up);

// Q3_K fused FFN gate+up: reads input once, quantizes to Q8_K one time,
// then computes gate and up dot products in parallel, producing SiLU(gate)*up.
// Both weight tensors must be Q3_K with the same K and N.
TensorPtr matmul_transB_fused_ffn_up_q3_k(const TensorPtr& input, const TensorPtr& w_gate,
                                          const TensorPtr& w_up);

// Q3_K fused QK projection: reads input once, shares Q8_K quantization
// across Q and K projections. Returns concatenated [1, N_q+N_k].
// V projection is kept separate (often different quantization in practice).
TensorPtr matmul_transB_fused_qk_q3_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk);

// Q3_K fused QKV projection (all-three Q3_K): reads input once, shares Q8_K
// quantization across Q, K, V projections. Returns concatenated [1, N_q+N_k+N_v].
TensorPtr matmul_transB_fused_qkv_q3_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv);
// Mixed-precision fused QKV: Q3_K (Q,K) + Q4_K (V). Shares Q8_K quantization
// across all three. Returns concatenated [1, N_q+N_k+N_v].
TensorPtr matmul_transB_fused_qkv_q3_k_q4_k(const TensorPtr& input, const TensorPtr& wq,
                                              const TensorPtr& wk, const TensorPtr& wv);

// Q4_K fused FFN down-projection + residual: computes input @ weight + residual.
// Weight tensor must be Q4_K.
TensorPtr matmul_transB_fused_ffn_down_residual_q4_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual);

// Q6_K fused FFN down-projection + residual: computes input @ weight + residual.
// Weight tensor must be Q6_K.
TensorPtr matmul_transB_fused_ffn_down_residual_q6_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual);
TensorPtr dequantize_q4_0_weight(const TensorPtr& q_weight);
TensorPtr dequantize_q4_1_weight(const TensorPtr& q_weight);

// Dequantize any quantized weight to FP32 (returns original if already FP32)
TensorPtr dequantize_weight(const TensorPtr& weight);

// Quantize an FP32 [N, K] weight tensor to Q8_0 format
TensorPtr quantize_q8_0_weight(const TensorPtr& fp32_weight);

}  // namespace ops
}  // namespace forge
