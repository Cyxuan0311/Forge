#pragma once

// BailingMoE3 (Ling 3.0) GDN / KDA recurrent helpers — model-specific, kept
// internal to the CPU operator tree (not part of the public include/forge API).
//
// Semantics follow llama.cpp:
//   src/models/bailingmoe3.cpp          KDA branch wiring (lines ~255-299)
//   src/models/models.h:14              build_gdn_l2_norm
//   ggml/src/ggml-cpu/ops.cpp           fused ggml_gated_delta_net CPU kernel
//
// Row-major layout throughout, head-major inside every token row:
//   x[t][h * head_dim + d]  ==  llama.cpp's [head_dim, n_head, token] view at (d, h).
// Every projection (q/k/v/ssm_f_a/ssm_g_a) spans d_inner = num_heads * head_dim,
// while ssm_beta and ssm_a carry one scalar per head.

#include <cstdint>
#include <vector>

#include "forge/tensor.h"

namespace forge {
namespace ops {

// Flatten all but the leading dimension: [a, b, c, ...] -> [a, b*c*...].
TensorPtr flatten_rows(const TensorPtr& t);

// Extract expert e (of n_expert) from a [n_expert, rows, cols] weight tensor.
TensorPtr extract_expert(const TensorPtr& t, int e, int n_expert);

// Borrowed view of expert e's packed rows as a [rows, cols] tensor sharing the
// base storage — no dequant, no copy. Works for quantized dtypes (row stride =
// ceil(cols/block_el) * block_sz) and FP32. The returned tensor must not
// outlive `t`. Lets matmul_transB dispatch quantized experts straight to the
// fused dot kernels (llama.cpp runs MoE mul_mat on packed weights directly).
TensorPtr expert_quant_view(const TensorPtr& t, int e, int n_expert);

// In-place interleaved RoPE on a single vector (dim must be even).
void apply_rope_interleaved(float* vec, int dim, int64_t pos, float theta);

// In-place elementwise sigmoid. `forge::ops::sigmoid` is declared in
// operator_activation.h but has no CPU implementation, so keep this local.
void sigmoid_inplace(const TensorPtr& x);

// In-place decay gate, mirroring llama.cpp bailingmoe3.cpp:
//   f     = matmul(x, ssm_f_a) + ssm_dt(bias)             per channel
//   decay = gate_lower_bound * sigmoid(ssm_a[h] * f)      per (head, dim)
// `logits` is the [seq_len, d_inner] matmul result and is overwritten by `decay`.
// ssm_dt is a [d_inner] bias, ssm_a a [num_heads] per-head scalar.
void build_decay_gate(const TensorPtr& logits, const TensorPtr& ssm_dt_bias,
                      const TensorPtr& ssm_a, int seq_len, int num_heads, int head_dim,
                      float gate_lower_bound);

// Per-head KDA recurrence == llama.cpp's fused gated delta net.
//
// q,k,v:   [seq_len, d_inner] — q/k are L2-normalized in place per (token, head),
//          using x / sqrt(sum(x^2) + eps) (their build_gdn_l2_norm).
// decay:   [seq_len, d_inner] — log decay per (head, dim), typically < 0.
// beta:    [seq_len, num_heads] — per-head input gate.
// state:   caller-owned [num_heads * head_dim * head_dim], laid out as
//          state[h][j][i] (index j*head_dim + i) holding S[i][j], i.e. transposed
//          exactly like ggml so each update touches a contiguous row.
//
// Per head and token, with S the running state:
//   S[i][j] *= exp(decay[i])
//   delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
//   S[i][j] += k[i] * delta[j]
//   out[j]   = (sum_i S[i][j]*q[i]) / sqrt(head_dim)
// Returns [seq_len, d_inner].
TensorPtr gdn_recurrence(const TensorPtr& q, const TensorPtr& k, const TensorPtr& v,
                         const TensorPtr& decay, const TensorPtr& beta, float* state,
                         int seq_len, int num_heads, int head_dim, float eps);

// Per-head RMSNorm followed by the sigmoid output gate:
//   out = rms_norm(core_out, ssm_norm, eps) * sigmoid(out_gate_logits)
// Both tensors are [seq_len, d_inner]; ssm_norm is [head_dim] (shared per head).
TensorPtr gdn_output_gate(const TensorPtr& core_out, const TensorPtr& out_gate_logits,
                          const TensorPtr& ssm_norm, int seq_len, int num_heads,
                          int head_dim, float eps);

// Absorbed MLA attention (bailingmoe3). The caller precomputes the two history
// quantities that are fixed once a position is stored, so this op does no RoPE
// or value-contraction of its own (that work lives in the engine's incremental
// MLA cache):
//   q_abs[h]  = attn_k_b[h] @ q_nope[h]        (contracted, per head — query side)
//   v_all[j][h*v_dim+d] = attn_v_b[h*v_dim+d] @ c_nope_norm[j]  (precomputed)
//   k_pe_rot[j] = rope(c_rope[j], pos=j)                       (precomputed)
// Per (query token, head): score(j) = ((q_abs·c_nope_norm[j]) + (rope(q_rope,
// qi)·k_pe_rot[j])) * scale, causal over j<=qi, softmax, then
//   out = Σ_j w_j * v_all[j] , finally multiplied by sigmoid(attn_gate) per head.
// `q_flat`  is [seq, n_head*(qk_nope_dim + rope_dim)];
// `c_nope_norm` [total, kv_lora]; `k_pe_rot` [total, rope_dim] (already rotated);
// `kb` is the (dequantized) FP32 attn_k_b [n_head*kv_lora, qk_nope_dim];
// `v_all` [total, n_head*v_dim] (precomputed); `gate` is [seq, n_head] with the
// sigmoid already applied. Returns [seq, n_head*v_dim].
TensorPtr bailingmoe3_mla_attention(const TensorPtr& q_flat,
                                    const TensorPtr& c_nope_norm,
                                    const TensorPtr& k_pe_rot,
                                    const TensorPtr& kb, const TensorPtr& v_all,
                                    const TensorPtr& gate, int start_pos,
                                    int seq_len, int total, int n_head,
                                    int kv_lora, int qk_nope_dim, int rope_dim,
                                    int v_dim, float theta, float scale);

}  // namespace ops
}  // namespace forge
