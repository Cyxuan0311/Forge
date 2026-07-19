#pragma once

#include <memory>

#include "forge/types.h"
#include "forge/tensor.h"

namespace forge {

class InferenceBatch;
class KVCache;

// Builds KQ attention masks for multi-sequence batched inference.
// Mask protocol: 0.0f = attend, < -1e20f = mask out, intermediate = additive bias.
//
// For single-sequence forward(), no mask is needed (causal is handled by the kernel).
// For multi-sequence forward_batch(), a mask is needed to:
//   1. Isolate sequences (prevent cross-sequence attention)
//   2. Apply causal constraints within each sequence
//   3. Handle sliding window attention (SWA)
class AttentionMaskBuilder {
public:
    // Build a KQ mask for a prefill/decode micro-batch.
    // Mask shape: [total_q_tokens, total_kv_len]
    static TensorPtr build_kq_mask(const InferenceBatch& ubatch,
                                   const KVCache& kv_cache,
                                   int layer,
                                   int num_heads,
                                   int head_dim,
                                   bool causal,
                                   DeviceType dev);

    // Build a decode mask (each sequence contributes exactly 1 query token).
    // Returns [n_seqs, total_kv_len] or nullptr if no masking needed.
    static TensorPtr build_decode_mask(const InferenceBatch& ubatch,
                                       const KVCache& kv_cache,
                                       int layer,
                                       DeviceType dev);
};

}  // namespace forge
