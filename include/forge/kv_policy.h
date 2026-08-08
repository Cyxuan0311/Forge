#pragma once
// Phase 6: default KV memory policy auto-selection.
//
// Selects dtype, storage mode, and page size based on device, phase
// (prefill/decode), max_seq_len, and architecture. Behind the
// FORGE_KV_AUTO_POLICY env var (default: off) to preserve existing behavior.

#include <cstdlib>
#include <string>

#include "kv_cache.h"

namespace forge {

struct KVDefaultPolicy {
    KVCacheDType dtype = KVCacheDType::FP32;
    KVStorageMode storage = KVStorageMode::Contiguous;
    int page_size = 16;
};

// Select default policy. `device` = primary device, `is_decode` = true for
// single-token decode phase, `max_seq_len` = configured max, `n_swa` = SWA
// window size (0 = no SWA).
inline KVDefaultPolicy select_default_policy(DeviceType device, bool is_decode,
                                             int max_seq_len, int n_swa) {
    KVDefaultPolicy p;
    const char* env = std::getenv("FORGE_KV_AUTO_POLICY");
    bool enabled = env && std::string(env) == "1";
    if (!enabled) return p;  // default: FP32 contiguous, page_size=16

    if (device == DeviceType::CUDA) {
        // CUDA decode: F16 + paged; prefill: FP32 contiguous
        p.dtype = is_decode ? KVCacheDType::F16 : KVCacheDType::FP32;
        p.storage = is_decode ? KVStorageMode::Paged : KVStorageMode::Contiguous;
    } else {
        // CPU: F16 for decode, FP32 for prefill
        p.dtype = is_decode ? KVCacheDType::F16 : KVCacheDType::FP32;
        p.storage = KVStorageMode::Contiguous;
    }
    // Long context → paged regardless
    if (max_seq_len > 2048) {
        p.storage = KVStorageMode::Paged;
    }
    (void)n_swa;  // SWA affects per-layer pool sizing, not the global default
    return p;
}

}  // namespace forge
