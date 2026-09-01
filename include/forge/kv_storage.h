#pragma once

// KVStorage: abstract interface for physical K/V data storage.
//
// Separates storage layout (contiguous vs paged) from the logical
// KVMemory layer (sequence management, slot allocation, prefix cache).
//
// Phase 2: ContiguousKVStorage (wraps existing KVCache).
// Phase 3: PagedKVStorage (dtype-aware page-based storage).

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_block_swapper.h"
#include "kv_cache.h"
#include "tensor.h"

namespace forge {

// =========================================================================
// KVStorage — abstract storage backend
// =========================================================================

class KVStorage {
public:
    virtual ~KVStorage() = default;

    // ---- Storage type ----
    virtual bool is_paged() const = 0;

    // ---- Initialization ----
    // Called by KVMemory during init_kv_cache(). For contiguous mode this is a no-op
    // (KVCache is already initialized by the engine). For paged mode, allocates pages.
    virtual bool init(int num_layers, const std::vector<int>& kv_dims, int max_seq_len,
                      DeviceType device, const KVCacheTypeConfig& kv_config, int page_size,
                      int max_num_seqs) = 0;

    // ---- Write ----
    // Legacy write (contiguous mode uses KVCache::update directly; this is a no-op).
    virtual bool write_kv(int layer, int64_t pos, int seq_len, const float* k_data,
                          const float* v_data) = 0;

    // Sequence-aware write (paged mode: writes into sequence's pages).
    virtual bool write_kv_seq(int layer, int seq_id, int64_t pos, int seq_len, const float* k_data,
                              const float* v_data) = 0;

    // ---- Read for attention ----
    // Returns FP32 key/value tensors for the given layer, covering all filled cells.
    virtual TensorPtr read_key(int layer) const = 0;
    virtual TensorPtr read_value(int layer) const = 0;
    virtual int filled(int layer) const = 0;

    // Per-sequence read (paged mode): gathers only the specified sequence's
    // KV data. Returns nullptr if not applicable (contiguous mode uses
    // read_key/read_value with masking instead).
    virtual TensorPtr read_key_seq(int layer, int seq_id) const {
        (void)layer;
        (void)seq_id;
        return nullptr;
    }
    virtual TensorPtr read_value_seq(int layer, int seq_id) const {
        (void)layer;
        (void)seq_id;
        return nullptr;
    }

    // ---- Sequence operations ----
    virtual int seq_filled(int layer, int seq_id) const = 0;
    virtual bool seq_remove(int seq_id, int64_t p0, int64_t p1) = 0;
    virtual bool seq_share(int src_seq, int dst_seq, int64_t p0, int64_t p1) = 0;
    virtual bool seq_keep(int seq_id) = 0;
    virtual void release(int seq_id) = 0;

    // ---- Stats ----
    virtual int max_seq_len() const = 0;
    virtual int num_layers() const = 0;
    virtual size_t nbytes() const = 0;
    virtual size_t active_bytes() const = 0;
    virtual int num_free_slots() const = 0;
    virtual int page_size() const = 0;
    virtual int num_free_pages() const = 0;

    // ---- Roadmap 1.1: host offload (swap) ----
    // Evict/bring back a single page between primary memory and the host swap
    // pool. Eviction releases the page's primary-memory storage; the page id
    // remains reserved and its data is restored by bring_back(). Both are
    // no-ops returning false on storages without swap support (contiguous).
    virtual bool offload_to_host(int layer, int page_id) {
        (void)layer;
        (void)page_id;
        return false;
    }
    virtual bool bring_back(int layer, int page_id) {
        (void)layer;
        (void)page_id;
        return false;
    }
    // Whole-sequence helpers used by the scheduler when it preempts a request.
    virtual bool offload_seq(int seq_id) {
        (void)seq_id;
        return false;
    }
    virtual bool bring_back_seq(int seq_id) {
        (void)seq_id;
        return false;
    }
    virtual int seq_num_pages(int seq_id) const {
        (void)seq_id;
        return 0;
    }
    virtual int num_offloaded_pages() const { return 0; }
    virtual int num_brought_back_pages() const { return 0; }
    virtual size_t host_pool_bytes() const { return 0; }
    virtual int total_page_capacity() const { return 0; }
    // Pages referenced by sequences and present in primary memory (evicted
    // pages are off the device and therefore not counted).
    virtual int num_device_pages_in_use() const { return 0; }

    // ---- Underlying KVCache (nullptr for paged mode) ----
    virtual KVCache* cache() = 0;
    virtual const KVCache* cache() const = 0;

    // ---- Reset (clear all sequences, return pages to free list) ----
    virtual void reset() = 0;

    // ---- Roadmap 2.2: defragmentation (contiguous mode only) ----
    // No-op for paged storage (pages are recycled via free lists, no holes).
    virtual void defrag() {}
    virtual void defrag_if_needed() {}

    // ---- Paged CUDA accessors (Phase 4) ----
    // These default to nullptr/false for non-paged or CPU-backed storage.
    // PagedKVStorage on CUDA overrides them to expose the per-layer device
    // page pointer tables and a staging buffer for the per-sequence page table.
    virtual DeviceType device() const { return DeviceType::CPU; }
    virtual bool is_cuda() const { return false; }
    virtual void* const* d_key_page_ptrs(int layer) const { return nullptr; }
    virtual void* const* d_value_page_ptrs(int layer) const { return nullptr; }
    // Uploads the host page_ids for (layer, seq_id) to a device staging buffer
    // and returns the device pointer. Returns nullptr if not applicable.
    virtual const int32_t* upload_seq_page_table(int layer, int seq_id) { return nullptr; }
    virtual KVCacheDType kv_type_k() const { return KVCacheDType::FP32; }
    virtual KVCacheDType kv_type_v() const { return KVCacheDType::FP32; }

    // Phase 6: per-layer page pool max sizes (for SWA pool isolation verification).
    // Returns empty vector for non-paged storage; PagedKVStorage overrides.
    virtual std::vector<int> layer_pool_max_pages() const { return {}; }
};

// =========================================================================
// ContiguousKVStorage — wraps existing KVCache
// =========================================================================

class ContiguousKVStorage : public KVStorage {
public:
    explicit ContiguousKVStorage(KVCache& cache);

    bool is_paged() const override { return false; }

    bool init(int num_layers, const std::vector<int>& kv_dims, int max_seq_len, DeviceType device,
              const KVCacheTypeConfig& kv_config, int page_size, int max_num_seqs) override;

    bool write_kv(int layer, int64_t pos, int seq_len, const float* k_data,
                  const float* v_data) override;

    bool write_kv_seq(int layer, int seq_id, int64_t pos, int seq_len, const float* k_data,
                      const float* v_data) override;

    TensorPtr read_key(int layer) const override;
    TensorPtr read_value(int layer) const override;
    int filled(int layer) const override;

    int seq_filled(int layer, int seq_id) const override;
    bool seq_remove(int seq_id, int64_t p0, int64_t p1) override;
    bool seq_share(int src_seq, int dst_seq, int64_t p0, int64_t p1) override;
    bool seq_keep(int seq_id) override;
    void release(int seq_id) override;

    // Roadmap 2.2: defragmentation passthrough + automatic trigger.
    void defrag() override;
    void defrag_if_needed() override;

    int max_seq_len() const override;
    int num_layers() const override;
    size_t nbytes() const override;
    size_t active_bytes() const override;
    int num_free_slots() const override;
    int page_size() const override { return 0; }
    int num_free_pages() const override { return 0; }

    KVCache* cache() override { return &cache_; }
    const KVCache* cache() const override { return &cache_; }

    void reset() override { cache_.reset(); }

private:
    KVCache& cache_;
};

// =========================================================================
// PagedKVStorage — dtype-aware page-based storage (Phase 3)
// =========================================================================

struct KVPage {
    KVCacheStorage key;      // dtype-aware per-page K storage
    KVCacheStorage value;    // dtype-aware per-page V storage
    uint32_t filled = 0;     // how many tokens are filled in this page
    uint32_t ref_count = 0;  // number of sequences referencing this page
    // Roadmap 1.1: true while the page's data lives in the host swap pool
    // (KVBlockSwapper) instead of the device/primary memory. The page_id stays
    // reserved and the sequence page tables keep pointing at it; the data is
    // restored by bring_back() before any read.
    bool evicted = false;
};

struct SequencePageTable {
    std::vector<int32_t> page_ids;  // page IDs for this sequence (per-layer)
    int64_t logical_len = 0;        // total logical length
    int filled_in_last = 0;         // filled slots in the last page
};

// Phase 6: per-layer page pool metadata.
// Each layer has its own free list and max_pages, so SWA layers can have
// a smaller pool (window_size/page_size) than Full layers (max_seq_len/page_size).
struct LayerPoolInfo {
    std::vector<int> free_page_ids;
    std::unordered_set<int> free_page_set;
    int max_pages = 0;
    KVLayerPolicy policy = KVLayerPolicy::Full;
    int window_size = 0;  // for SlidingWindow policy
};

class PagedKVStorage : public KVStorage {
public:
    PagedKVStorage();
    ~PagedKVStorage();

    // ---- KVStorage interface ----
    bool is_paged() const override { return true; }

    bool init(int num_layers, const std::vector<int>& kv_dims, int max_seq_len, DeviceType device,
              const KVCacheTypeConfig& kv_config, int page_size, int max_num_seqs) override;

    bool write_kv(int layer, int64_t pos, int seq_len, const float* k_data,
                  const float* v_data) override;
    bool write_kv_seq(int layer, int seq_id, int64_t pos, int seq_len, const float* k_data,
                      const float* v_data) override;

    TensorPtr read_key(int layer) const override;
    TensorPtr read_value(int layer) const override;
    int filled(int layer) const override;
    TensorPtr read_key_seq(int layer, int seq_id) const override;
    TensorPtr read_value_seq(int layer, int seq_id) const override;

    int seq_filled(int layer, int seq_id) const override;
    bool seq_remove(int seq_id, int64_t p0, int64_t p1) override;
    bool seq_share(int src_seq, int dst_seq, int64_t p0, int64_t p1) override;
    bool seq_keep(int seq_id) override;
    void release(int seq_id) override;

    int max_seq_len() const override { return max_seq_len_; }
    int num_layers() const override { return num_layers_; }
    size_t nbytes() const override;
    size_t active_bytes() const override;
    int num_free_slots() const override;
    int page_size() const override { return page_size_; }
    int num_free_pages() const override;

    // ---- Roadmap 1.1: host offload (swap) ----
    bool offload_to_host(int layer, int page_id) override;
    bool bring_back(int layer, int page_id) override;
    bool offload_seq(int seq_id) override;
    bool bring_back_seq(int seq_id) override;
    int seq_num_pages(int seq_id) const override;
    int num_device_pages_in_use() const override;

    // Swap statistics and host-pool usage (for the scheduler's metrics).
    int num_offloaded_pages() const { return offloaded_pages_; }
    int num_brought_back_pages() const { return brought_back_pages_; }
    size_t host_pool_bytes() const;
    // Total page capacity across all per-layer pools.
    int total_page_capacity() const;

    // Lock-free internals (mutex_ must be held by the caller).
    bool offload_to_host_unlocked(int layer, int page_id);
    bool bring_back_unlocked(int layer, int page_id);
    void refresh_cuda_ptr_tables_unlocked(int layer);

    KVCache* cache() override { return nullptr; }
    const KVCache* cache() const override { return nullptr; }

    void reset() override;

    // ---- Paged-specific accessors ----
    int max_num_seqs() const { return max_num_seqs_; }
    bool has_seq(int seq_id) const;
    DeviceType device() const override { return device_; }
    bool is_cuda() const override { return device_ == DeviceType::CUDA; }
    const KVCacheTypeConfig& kv_config() const { return kv_config_; }
    KVCacheDType kv_type_k() const override { return kv_config_.type_k; }
    KVCacheDType kv_type_v() const override { return kv_config_.type_v; }

    // Phase 6: per-layer page pool max sizes
    std::vector<int> layer_pool_max_pages() const override;

    // Phase 6: configure per-layer policies and pool sizes.
    // SWA layers get max_pages = window_size/page_size; Full layers get
    // max_pages = max_seq_len/page_size. Must be called BEFORE init() or
    // init() will auto-derive from kv_dims (all Full).
    void set_layer_policies(const std::vector<KVLayerPolicy>& policies, int swa_window);

    // ---- Paged CUDA accessors (Phase 4) ----
    // d_key_page_ptrs(layer) returns the device array of K page base pointers
    // for the given layer (max_pages_ entries). Only valid when is_cuda().
    void* const* d_key_page_ptrs(int layer) const override;
    void* const* d_value_page_ptrs(int layer) const override;
    const int32_t* upload_seq_page_table(int layer, int seq_id) override;

private:
    int alloc_page(int layer);
    void free_page(int layer, int page_id);
    bool ensure_seq_capacity(int seq_id, int layer, int needed_slots);
    void init_seq(int seq_id);
    // Lock-free inner implementation of upload_seq_page_table.
    // Caller must hold mutex_.
    const int32_t* upload_seq_page_table_unlocked(int layer, int seq_id);
    // Phase 6: max page count for a given layer's pool.
    int layer_max_pages(int layer) const;

    int num_layers_ = 0;
    std::vector<int> kv_dims_;  // per-layer kv_dim
    int max_seq_len_ = 0;
    int page_size_ = 16;  // tokens per page
    int max_num_seqs_ = 32;
    int max_pages_ = 0;  // max across layers (for log/CUDA sizing)
    DeviceType device_ = DeviceType::CPU;
    KVCacheTypeConfig kv_config_;

    // ---- Roadmap 1.1: host swap pool and counters ----
    KVBlockSwapper swapper_;
    int offloaded_pages_ = 0;
    int brought_back_pages_ = 0;

    // Pages: layer_pages_[layer][page_id] -> KVPage
    std::vector<std::vector<KVPage>> layer_pages_;
    // Phase 6: per-layer page pools (replaces global free_page_ids_/free_page_set_).
    // layer_pools_[l] holds layer l's free list, max_pages, and policy.
    std::vector<LayerPoolInfo> layer_pools_;

    // Per-sequence page tables: seq_tables_[seq_id][layer] -> SequencePageTable
    std::unordered_map<int, std::vector<SequencePageTable>> seq_tables_;

    // ---- CUDA page pointer tables (Phase 4) ----
    // Built once in init() when device_==CUDA. Pages themselves are allocated
    // by KVCacheStorage::alloc (which places them on CUDA); these tables just
    // collect the per-page device base pointers for kernel access.
    // Host staging is [layer][page]; device array is one void** per layer.
    std::vector<std::vector<void*>> d_key_page_ptrs_host_;
    std::vector<std::vector<void*>> d_value_page_ptrs_host_;
    std::vector<void**> d_key_page_ptrs_;
    std::vector<void**> d_value_page_ptrs_;
    // Device staging buffer for the current sequence's page table. Realloc'd
    // (grow-only) by upload_seq_page_table when the sequence needs more pages
    // than the current capacity.
    int32_t* d_seq_page_ids_ = nullptr;
    int d_seq_page_ids_cap_ = 0;

    mutable std::mutex mutex_;
};

}  // namespace forge
