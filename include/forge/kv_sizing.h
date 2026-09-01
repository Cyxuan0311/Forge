#pragma once

// KV sizing — per-sequence KV footprint and auto-capacity helpers.
//
// Roadmap phase 1 / item 1.2: `max_num_seqs` used to be hard-coded (32 in both
// KVCache and PagedKVStorage), which capped concurrency no matter how much
// device memory was actually free. These helpers derive the concurrency limit
// and the per-layer page count from a device memory budget instead.
//
// The helpers are pure (except query_free_device_bytes) so they can be unit
// tested on a CPU-only build.

#include <cstddef>
#include <vector>

#include "kv_cache.h"

namespace forge {

// Everything needed to describe one KV cache for sizing purposes.
struct KVSizingParams {
    int num_layers = 0;
    std::vector<int> kv_dims;  // per-layer kv_dim (num_kv_heads * head_dim)
    int max_seq_len = 0;
    int page_size = 16;
    DeviceType device = DeviceType::CPU;
    KVCacheDType type_k = KVCacheDType::FP32;
    KVCacheDType type_v = KVCacheDType::FP32;
    std::vector<KVLayerPolicy> policies;  // empty => every layer is Full
    int swa_window = 0;                   // window size for SlidingWindow layers
};

struct KVSizingResult {
    int max_num_seqs = 0;         // derived concurrency limit (0 => undetermined)
    int max_pages_per_layer = 0;  // page pool size for Full layers
    size_t per_seq_bytes = 0;     // K+V bytes for one full-length sequence
    size_t page_bytes = 0;        // K+V bytes for one page, summed over layers
    size_t budget_bytes = 0;      // usable budget after the safety factor
    bool auto_sized = false;      // false => caller should keep its default
};

// Bytes for one token of K+V at a single layer.
size_t kv_token_bytes(KVCacheDType type_k, KVCacheDType type_v, int kv_dim);

// Effective KV length of a layer. SlidingWindow layers are bounded by the
// window instead of growing to max_seq_len (llama.cpp `iswa` behaviour).
int kv_layer_effective_len(const KVSizingParams& p, int layer);

// K+V bytes for one full-length sequence across every layer.
size_t per_seq_kv_bytes(const KVSizingParams& p);

// K+V bytes for a single page, summed over every layer.
size_t kv_page_bytes(const KVSizingParams& p);

// Free bytes reported by the device backend, minus a reserve for activations /
// logits / workspace. Returns 0 when the device is CPU or unknown, which means
// "cannot auto-size".
size_t query_free_device_bytes(DeviceType device);

// Derive max_num_seqs and the per-layer page count from a byte budget.
// `safety` (default 0.9) is the fraction of the budget the KV pool may claim.
KVSizingResult auto_size_kv(const KVSizingParams& p, size_t budget_bytes, double safety = 0.9);

// Convenience wrapper: query the device, then auto-size in one call.
KVSizingResult auto_size_kv_for_device(const KVSizingParams& p, double safety = 0.9);

}  // namespace forge
