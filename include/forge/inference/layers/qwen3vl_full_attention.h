#pragma once

// Qwen3vlFullAttention: Qwen3-VL attention layer (per-head QK-Norm + MRoPE).
//
// Extracted from GenericEngine::forward_layer so it can be embedded as a single
// graph node. Input is the already-attn_norm'd hidden; it returns the
// attention output BEFORE the residual add (caller adds the residual).
//
// Device handling mirrors GenericEngine exactly: every executor (projection,
// QK-Norm, RoPE, attention, output projection) is handed the layer device and
// decides internally whether to run on CPU (e.g. MRoPE with n_rot < head_dim
// falls back to the CPU rotary path), so outputs are bit-identical to the
// imperative engine for FP32/F16 KV caches. The Hadamard Q/K/V rotation that
// GenericEngine applies for *quantized* KV dtypes (q4_0/q8_0 + --rope-k-quant)
// is not mirrored here: the flag lives on TransformerEngine and is not visible
// to graph builders. With quantized KV the graph path would diverge; keep the
// graph path on FP32/F16 KV until that state is threaded through.

#include "forge/inference/layer_execution_context.h"
#include "forge/inference/layers/attention_executor.h"
#include "forge/inference/layers/rope_executor.h"
#include "forge/kv_cache.h"
#include "forge/model.h"

namespace forge {

class Qwen3vlFullAttention {
public:
    // kv_memory is intentionally omitted: the graph path only supports the
    // contiguous KVCache backend (matching Qwen35FullAttention). Paged KV would
    // require routing writes through PagedKVStorage, added later if needed.
    explicit Qwen3vlFullAttention(KVCache& kv_cache)
        : kv_cache_(kv_cache), attn_(kv_cache, nullptr) {}

    // Returns nullptr on missing weights (caller decides how to degrade).
    TensorPtr attend(const TensorPtr& normed, const LayerExecutionContext& lctx);

private:
    KVCache& kv_cache_;
    // Shared, storage-aware attention backend (contiguous KV only here).
    AttentionExecutor attn_;
    // MRoPE executor (handles the single-position partial rotation for Qwen3-VL).
    RopeExecutor rope_executor_;
};

}  // namespace forge
