#pragma once

#include "tensor.h"

namespace nanoinfer {
namespace ops {

TensorPtr scaled_dot_product_attention(const TensorPtr& q, const TensorPtr& k,
                                        const TensorPtr& v, bool causal = true);

TensorPtr scaled_dot_product_attention_2d(const TensorPtr& q, const TensorPtr& k,
                                           const TensorPtr& v, int seq_len, int num_heads,
                                           int head_dim, bool causal = true);

TensorPtr scaled_dot_product_attention_2d(const TensorPtr& q, const TensorPtr& k,
                                           const TensorPtr& v, int q_len, int kv_len,
                                           int num_heads, int head_dim, bool causal = true);

// Attention with additive mask (mask values: 0 = attend, -inf = mask out)
TensorPtr scaled_dot_product_attention_2d_masked(const TensorPtr& q, const TensorPtr& k,
                                                  const TensorPtr& v, int seq_len,
                                                  int num_heads, int head_dim,
                                                  const TensorPtr& mask);

// GQA attention: supports num_kv_heads < num_heads without explicit KV expansion.
// K/V tensors have layout [kv_len, num_kv_heads, head_dim].
// Q tensor has layout [q_len, num_heads * head_dim].
// For decode (q_len=1), uses direct head mapping instead of physical expand.
TensorPtr scaled_dot_product_attention_2d_gqa(const TensorPtr& q, const TensorPtr& k,
                                               const TensorPtr& v, int q_len, int kv_len,
                                               int num_heads, int num_kv_heads,
                                               int head_dim, bool causal = true);

} // namespace ops
} // namespace nanoinfer
