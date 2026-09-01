#pragma once

// KVMemory: unified sequence/slot/prefix management layer.
//
// Decouples logical sequence state (id, position, shared ranges) from
// physical storage (contiguous vs paged). The storage backend is behind
// KVStorage; attention reads through KVReadView.
//
// Phase 2: wraps the existing KVCache via ContiguousKVStorage.
// Phase 3: adds PagedKVStorage as an alternative backend.

#include <cstdint>
#include <memory>
#include <vector>

#include "kv_cache.h"
#include "kv_storage.h"
#include "tensor.h"

namespace forge {

// =========================================================================
// KVReadView — what attention sees for one layer + one sequence
// =========================================================================

struct KVReadView {
    TensorPtr key;               // FP32 key tensor for attention
    TensorPtr value;             // FP32 value tensor for attention
    std::vector<int32_t> slots;  // physical slot indices (unused in contiguous mode)
    int n_tokens = 0;            // number of tokens in this view
    int kv_len = 0;              // total KV length for attention
    bool contiguous = true;      // true if slots are contiguous (no stitching needed)
};

// =========================================================================
// KVSlotInfo — describes assigned slots for one sequence
// =========================================================================

struct KVSlotInfo {
    int seq_id = -1;
    int64_t pos_begin = 0;
    std::vector<int32_t> slots;  // assigned physical slots
    bool contiguous = false;     // true if slots are a single contiguous range
};

// =========================================================================
// KVPrepareResult — outcome of KVMemory::prepare()
// =========================================================================

struct KVPrepareResult {
    std::vector<KVSlotInfo> slot_infos;
    int kv_len = 0;
    bool success = false;
};

// =========================================================================
// KVMemory — unified KV memory manager
// =========================================================================

class KVMemory {
public:
    // Contiguous mode: wraps existing KVCache.
    explicit KVMemory(KVCache& cache);

    // Paged mode: creates PagedKVStorage. KVCache is still referenced for
    // transitional access (fused CUDA paths) but sequence ops go through storage_.
    KVMemory(KVCache& cache, KVStorageMode mode);

    // ---- Storage access ----
    KVStorage& storage() { return *storage_; }
    const KVStorage& storage() const { return *storage_; }

    // Direct access to underlying KVCache (transitional; attention_executor
    // still uses KVCache directly for fused CUDA paths).
    KVCache& cache() { return cache_; }
    const KVCache& cache() const { return cache_; }

    // Storage mode query
    bool is_paged() const { return storage_->is_paged(); }

    // Initialize paged storage (called by engine during init_kv_cache).
    // No-op for contiguous mode.
    bool init_storage(int num_layers, const std::vector<int>& kv_dims, int max_seq_len,
                      DeviceType device, const KVCacheTypeConfig& kv_config, int page_size,
                      int max_num_seqs);

    // Phase 6: set per-layer policies (must be called before init_storage for
    // paged mode to size SWA pools correctly).
    void set_layer_policies(const std::vector<KVLayerPolicy>& policies, int swa_window);

    // ---- View for attention ----
    KVReadView view(int layer, int seq_id) const;

    // ---- Sequence operations ----
    bool seq_remove(int seq_id, int64_t p0, int64_t p1);
    bool seq_share(int src_seq, int dst_seq, int64_t p0, int64_t p1);
    bool seq_keep(int seq_id);
    void release(int seq_id);

    // ---- Scheduler-friendly aliases ----
    bool share_sequence(int src_seq, int dst_seq, int64_t p0, int64_t p1) {
        return seq_share(src_seq, dst_seq, p0, p1);
    }
    bool release_sequence(int seq_id);

    // Roadmap 2.2: defragment the underlying contiguous KV cache (no-op for paged).
    void defrag();

    // ---- Prefix cache ----
    // Returns the hash of the given tokens for prefix cache lookup.
    // The caller (RequestScheduler) uses this to manage its own prefix cache.
    static size_t hash_tokens(const std::vector<int32_t>& tokens);

    // ---- Statistics ----
    int max_seq_len() const;
    int num_layers() const;
    size_t nbytes() const;
    size_t active_bytes() const;
    int num_free_slots() const;

    // Prefix cache tracking (updated by scheduler, read by memory_stats)
    int prefix_hits() const { return prefix_hits_; }
    int prefix_tokens() const { return prefix_tokens_; }
    void record_prefix_hit(int tokens) {
        prefix_hits_++;
        prefix_tokens_ += tokens;
    }

private:
    KVCache& cache_;
    std::unique_ptr<KVStorage> storage_;
    int prefix_hits_ = 0;
    int prefix_tokens_ = 0;
};

}  // namespace forge
