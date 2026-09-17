#pragma once

// BailingMoE3 (Ling 3.0) depthwise causal conv1d — model-specific, kept
// internal to the CPU operator tree (not part of the public include/forge API).
//
// Convention mirrors Qwen35's delta-net conv: `state` holds the previous
// (d_conv - 1) inputs per channel, laid out as [(d_conv-1) rows, dim cols].

#include <cstdint>

#include "forge/tensor.h"

namespace forge {
namespace ops {

// Flatten a depthwise conv1d weight of shape [dim, 1, k] (or [dim, k]) into a
// flat [dim, k] tensor consumed by conv1d_depthwise_silu.
TensorPtr load_conv_weight(const TensorPtr& w, int k_size, int dim);

// Depthwise causal conv1d (kernel = k_size) followed by SiLU, applied in place
// to x ([seq_len, dim]). `state` is a caller-owned buffer of size
// [(k_size - 1) * dim] holding the per-channel running window.
void conv1d_depthwise_silu(TensorPtr& x, const TensorPtr& weight, float* state,
                           int seq_len, int dim, int k_size);

}  // namespace ops
}  // namespace forge
