#pragma once

// KVBlockSwapper — host-side (optionally pinned) pool for evicted KV pages.
//
// Roadmap 1.1: when the paged GPU KV pool is under memory pressure, PagedKVStorage
// evicts whole pages to this host pool (bit-exact copies) and brings them back on
// demand. This keeps the device KV footprint bounded without ever losing state:
// an evicted page is a perfect copy of the page that was on device.
//
// The copy is raw (dtype-agnostic): a page is a fixed-size key blob followed by a
// value blob, byte-identical to what KVCacheStorage held on the device. Host
// buffers are allocated via cudaHostAlloc (pinned, DMA-able) when CUDA is built
// and available, and fall back to plain host memory otherwise (CPU backend).
//
// Capacity & LRU: by default the pool is grow-only (max_bytes()==0, historical
// behavior). When a cap is set via set_max_bytes(), offload() of a NEW page is
// rejected once the pool would exceed it. The swapper never evicts on its own —
// blobs may still be referenced by their owner (e.g. the prefix radix tree) —
// instead the owner queries lru_oldest() and drop()s entries it no longer needs.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace forge {

class KVBlockSwapper {
public:
    KVBlockSwapper() = default;
    ~KVBlockSwapper();

    KVBlockSwapper(const KVBlockSwapper&) = delete;
    KVBlockSwapper& operator=(const KVBlockSwapper&) = delete;

    // ---- Grow-only pool ops ----
    // Save the raw key/value bytes of page (layer, page_id). The slot is created
    // on first use and never shrinks. `k_bytes`/`v_bytes` are fixed per page so
    // mismatched sizes on a later call are rejected.
    //
    // `kind` says where the source/destination memory lives. It is NOT optional:
    // a paged GPU page's bytes live in device memory, and std::memcpy cannot read
    // them (it faults or silently returns garbage). PagedKVStorage knows which
    // side is device, so it says so.
    enum class CopyKind : int {
        Host,  // plain std::memcpy (CPU backend, or a pinned staging copy)
        Device,  // cudaMemcpy; the buffers are device pointers
    };
    bool offload(int layer, int page_id, const void* k_src, size_t k_bytes, const void* v_src,
                 size_t v_bytes, CopyKind kind = CopyKind::Host, bool force = false);

    // Restore the previously saved bytes into the given destinations. A
    // successful call refreshes the entry's LRU position (it counts as a use).
    bool bring_back(int layer, int page_id, void* k_dst, size_t k_bytes, void* v_dst,
                    size_t v_bytes, CopyKind kind = CopyKind::Host);

    bool has(int layer, int page_id) const;
    bool size(int layer, int page_id, size_t* k_bytes, size_t* v_bytes) const;
    // Read-only view of a pooled blob's bytes (no LRU refresh). Used when the
    // owner needs the data itself — e.g. spilling a prefix blob down to L3.
    bool peek(int layer, int page_id, const uint8_t** k, const uint8_t** v, size_t* k_bytes,
              size_t* v_bytes) const;

    // Drop a single page's host copy (used when the page is freed for good).
    void drop(int layer, int page_id);

    // Drop every saved page (frees host memory).
    void reset();

    // ---- Capacity cap & LRU ----
    // max_bytes == 0 (default) keeps the historical grow-only behavior. With a
    // cap, offload() of a NEW page fails once the pool would exceed it; the
    // owner evicts via lru_oldest() + drop().
    void set_max_bytes(size_t bytes) {
        std::lock_guard<std::mutex> l(mtx_);
        max_bytes_ = bytes;
    }
    size_t max_bytes() const {
        std::lock_guard<std::mutex> l(mtx_);
        return max_bytes_;
    }

    // Refresh an entry's LRU position (most-recently-used). False if absent.
    bool touch(int layer, int page_id);

    // Report the least-recently-used entry. False when the pool is empty.
    bool lru_oldest(int* layer, int* page_id) const;

    // Total host bytes currently held (pinned or not).
    size_t nbytes() const {
        std::lock_guard<std::mutex> l(mtx_);
        return total_bytes_;
    }
    size_t num_pages() const {
        std::lock_guard<std::mutex> l(mtx_);
        return pool_.size();
    }
    bool pinned() const { return pinned_; }

private:
    using PageKey = std::pair<int, int>;
    struct PageKeyHash {
        size_t operator()(const PageKey& k) const {
            return (static_cast<size_t>(k.first) << 32) ^ static_cast<size_t>(k.second);
        }
    };

    struct HostBlob {
        uint8_t* k = nullptr;
        size_t k_size = 0;
        uint8_t* v = nullptr;
        size_t v_size = 0;
        std::list<PageKey>::iterator lru_it{};  // node in lru_, valid while pooled
        void reset();
    };

    uint8_t* alloc_host(size_t bytes);
    void free_host(uint8_t* p);

    std::unordered_map<PageKey, HostBlob, PageKeyHash> pool_;
    std::list<PageKey> lru_;  // front = least recently used
    size_t total_bytes_ = 0;
    size_t max_bytes_ = 0;  // 0 = grow-only (no cap); historical default
    bool pinned_ = false;

    // Protects every member above. The pool is touched both by the PrefixCache's
    // background backup worker (offload / bring_back) and by the owner thread
    // (drop / lru_oldest / nbytes during eviction), so all public ops serialize
    // here. Lock order is always storage mutex_ -> mtx_ (the worker holds the
    // storage mutex_ across an offload); nothing takes mtx_ and then a coarser
    // lock, so there is no inversion.
    mutable std::mutex mtx_;
};

}  // namespace forge
