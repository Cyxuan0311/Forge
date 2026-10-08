#pragma once

#include "forge/tensor.h"
#include <cstdint>
#include <vector>

// CUDA implementations of the bailingmoe3 GDN/MLA custom operators. These mirror
// the CPU kernels in operators/cpu/gdn.cpp exactly. The *device* state pointers
// (conv_state_dev, state_dev) are device allocations owned by the engine's
// recurrent-memory manager; the CPU path passes host pointers instead.
namespace forge {
namespace cuda {

// Depthwise causal conv (kernel d_conv) + SiLU, per channel over d_inner.
void conv1d_depthwise_silu_launch(const TensorPtr& x, const TensorPtr& w,
                                 float* conv_state_dev, int seq_len, int d_inner,
                                 int d_conv);

// Fused 3-stream depthwise causal conv + SiLU (q/k/v in one launch).
// Same per-channel math as conv1d_depthwise_silu_launch; one launch covers
// 3*d_inner channels so decode doesn't pay three small-kernel launches per
// GDN layer.
void conv3_depthwise_silu_launch(const TensorPtr& x0, const TensorPtr& w0,
                                 float* conv_state_dev0, const TensorPtr& x1,
                                 const TensorPtr& w1, float* conv_state_dev1,
                                 const TensorPtr& x2, const TensorPtr& w2,
                                 float* conv_state_dev2, int seq_len, int d_inner,
                                 int d_conv);

// Decay gate: per (token, head, dim) -> gate = lower_bound / (1 + exp(-a*(x+bias))).
// `bias` (size d_inner) and `a` (size num_heads) are device-resident pointers
// uploaded once per layer by the engine — a per-call H2D cudaMemcpy would drain
// the whole default-stream pipeline.
void build_decay_gate_launch(const TensorPtr& logits, const float* bias, const float* a,
                            int seq_len, int num_heads, int head_dim,
                            float gate_lower_bound);

// Gated delta-rule recurrence with persistent per-head S state (device pointer).
void gdn_recurrence_launch(const TensorPtr& q, const TensorPtr& k, const TensorPtr& v,
                          const TensorPtr& decay, const TensorPtr& beta,
                          float* state_dev, int seq_len, int num_heads,
                          int head_dim, float eps, TensorPtr& out);

// Per-head RMSNorm + sigmoid output gate. `nw` is a device-resident pointer
// (size head_dim), uploaded once per layer by the engine.
void gdn_output_gate_launch(const TensorPtr& core_out,
                           const TensorPtr& out_gate_logits,
                           const float* nw, int seq_len, int num_heads,
                           int head_dim, float eps, TensorPtr& out);

// In-place sigmoid on a device tensor.
void sigmoid_launch(const TensorPtr& t);

// acc_dev[d] += w * src_dev[d] for d in [0, n). Used to sum routed-expert
// outputs on the device (avoids one D2H per expert).
void moe_accumulate_launch(float* acc_dev, const float* src_dev, float w, int n);

// Absorbed MLA attention (query contraction + causal softmax + output gate) on
// device. The history quantities are precomputed by the engine and passed in:
//   `k_pe_rot` [total, rope_dim]  — c_rope already rotated at absolute pos j
//   `v_all`    [total, n_head*v_dim] — value contraction (precomputed)
// `c_nope_norm` [total, kv_lora] and `kb` (attn_k_b) are still needed for the
// query-side contraction and the nope score term. Returns [seq, n_head*v_dim].
// `yarn_s` (optional) is a device array of `rope_dim` per-element frequency
// scales and `yarn_mscale` the rotation magnitude — the YaRN correction
// (llama.cpp rope_yarn). Pass nullptr / 1.0f for plain rope.
void mla_attention_launch(const TensorPtr& q_flat, const TensorPtr& c_nope_norm,
                         const TensorPtr& k_pe_rot, const TensorPtr& kb,
                         const TensorPtr& v_all, const TensorPtr& gate,
                         int start_pos, int seq_len, int total, int n_head,
                         int kv_lora, int qk_nope_dim, int rope_dim, int v_dim,
                         float theta, float scale, TensorPtr& out,
                         const float* yarn_s = nullptr, float yarn_mscale = 1.0f);

// Rotate the new c_rope rows (absolute positions start_pos .. start_pos+total-1)
// into k_pe_rot. Mirrors the RoPE the CPU reference applies per history key.
void mla_rope_launch(const TensorPtr& c_rope, TensorPtr& k_pe_rot, int start_pos,
                     int total, int rope_dim, float theta,
                     const float* yarn_s = nullptr, float yarn_mscale = 1.0f);

// Value contraction v_all[j][h*v_dim+d] = vb[(h*v_dim+d)*kv_lora+k] · c_nope_norm[j].
// Computes only the `total` supplied rows (the engine appends them to its cache).
void mla_value_contract_launch(const TensorPtr& vb, const TensorPtr& c_nope_norm,
                               TensorPtr& v_all, int total, int n_head,
                               int kv_lora, int v_dim);

// Split a device [total, kv_lora+rope_dim] latent into c_nope [total,kv_lora]
// and c_rope [total,rope_dim] (nope first). Used to rebuild K/V on the device.
void mla_split_latent(const float* latent_dev, float* c_nope_dev, float* c_rope_dev,
                      int total, int kv_lora, int rope_dim);

}  // namespace cuda
}  // namespace forge
