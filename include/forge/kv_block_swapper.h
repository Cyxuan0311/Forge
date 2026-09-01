#pragma once

// KVBlockSwapper — host-side (optionally pinned) pool for evicted KV pages.
//
// Roadmap 1.1: when the paged GPU KV pool is under memory pressure, PagedKVStorage
// evicts whole pages to this host pool (grow-only, bit-exact copies) and brings
// them back on demand. This keeps the device KV footprint bounded without ever
// losing state: an evicted page is a perfect copy of the page that was on device.
//
// The copy is raw (dtype-agnostic): a page is a fixed-size key blob followed by a
// value blob, byte-identical to what KVCacheStorage held on the device. Host
// buffers are allocated via cudaHostAlloc (pinned, DMA-able) when CUDA is built
// and available, and fall back to plain host memory otherwise (CPU backend).

#include <cstddef>
#include <cstdint>
#include <functional>
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
    bool offload(int layer, int page_id, const void* k_src, size_t k_bytes, const void* v_src,
                 size_t v_bytes);

    // Restore the previously saved bytes into the given destinations.
    bool bring_back(int layer, int page_id, void* k_dst, size_t k_bytes, void* v_dst,
                    size_t v_bytes) const;

    bool has(int layer, int page_id) const;
    bool size(int layer, int page_id, size_t* k_bytes, size_t* v_bytes) const;

    // Drop a single page's host copy (used when the page is freed for good).
    void drop(int layer, int page_id);

    // Drop every saved page (frees host memory).
    void reset();

    // Total host bytes currently held (pinned or not).
    size_t nbytes() const { return total_bytes_; }
    size_t num_pages() const { return pool_.size(); }
    bool pinned() const { return pinned_; }

private:
    struct HostBlob {
        uint8_t* k = nullptr;
        size_t k_size = 0;
        uint8_t* v = nullptr;
        size_t v_size = 0;
        void reset();
    };

    uint8_t* alloc_host(size_t bytes);
    void free_host(uint8_t* p);

    using PageKey = std::pair<int, int>;
    struct PageKeyHash {
        size_t operator()(const PageKey& k) const {
            return (static_cast<size_t>(k.first) << 32) ^ static_cast<size_t>(k.second);
        }
    };

    std::unordered_map<PageKey, HostBlob, PageKeyHash> pool_;
    size_t total_bytes_ = 0;
    bool pinned_ = false;
};

}  // namespace forge
