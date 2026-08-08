#pragma once

// PrefixCache: page-level prefix sharing with LRU eviction.
//
// Phase 5: upgrades the scheduler's prompt-hash cache to operate at page
// granularity. Only complete pages are cached; the trailing incomplete page
// is owned exclusively by the request.
//
// Design:
//   - Each registered prompt creates ONE persistent "cache seq_id" that holds
//     the prefix pages (transferred from the source request via seq_share +
//     seq_remove). Multiple PrefixEntry records (one per complete-page prefix
//     length) point to this cache seq_id.
//   - Lookup uses incremental FNV-1a hashes computed at each page boundary,
//     trying the longest prefix first.
//   - LRU eviction skips entries with ref_count > 0 (in use). A cache seq_id
//     is released only when all entries pointing to it are evicted.
//
// All operations go through the KVStorage abstract interface, so the cache
// works with any storage backend that implements seq_share/seq_remove/release.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kv_storage.h"

namespace forge {

// =========================================================================
// PrefixEntry — one cached prefix (a complete-page prefix of a prompt)
// =========================================================================

struct PrefixEntry {
    uint64_t hash = 0;              // incremental hash of tokens[0..token_count-1]
    int token_count = 0;            // number of tokens covered (multiple of page_size)
    int owner_seq_id = -1;          // cache seq_id holding the prefix pages
    uint32_t ref_count = 0;         // number of active requests using this prefix
    uint64_t last_used = 0;         // LRU timestamp (monotonic clock)
    bool valid = false;
};

// =========================================================================
// PrefixCache — manages prefix entries with LRU eviction
// =========================================================================

class PrefixCache {
public:
    PrefixCache() = default;

    // Try to find the longest cached prefix for the given tokens.
    // On hit: shares prefix pages with dst_seq_id via storage.seq_share,
    //         increments the entry's ref_count.
    // Returns the number of prefix tokens shared (0 = miss).
    int try_lookup(const std::vector<int32_t>& tokens, int dst_seq_id,
                   KVStorage& storage);

    // Register a prefix for future requests. Called when a request finishes
    // its prefill (and did NOT use a cached prefix). Transfers prefix page
    // ownership from src_seq_id to a new cache seq_id. Creates entries for
    // each complete-page prefix length.
    void register_prefix(const std::vector<int32_t>& tokens, int src_seq_id,
                         KVStorage& storage);

    // Release a request's prefix reference. Called when a request that used
    // a cached prefix finishes. Decrements the entry's ref_count and removes
    // the prefix pages from the request's page table (via seq_remove).
    void release_prefix(int seq_id, KVStorage& storage);

    // Evict LRU entries with ref_count == 0 until size <= max_entries.
    // Releases cache seq_ids when all their entries are evicted.
    void evict_lru(size_t max_entries, KVStorage& storage);

    // Stats
    int hits() const { return hits_; }
    int misses() const { return misses_; }
    void reset_stats() { hits_ = 0; misses_ = 0; }

    // Clear all entries (does not release pages — caller must handle).
    void clear();

    // Inspection
    size_t size() const { return entries_.size(); }
    bool has_request(int seq_id) const { return seq_to_hash_.count(seq_id) > 0; }

    // LRU capacity (default 64 entries).
    static constexpr size_t DEFAULT_MAX_ENTRIES = 64;

private:
    // Incremental FNV-1a hash: extend prev_hash with one token.
    static uint64_t hash_extend(uint64_t prev, int32_t tok) {
        uint64_t h = prev;
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(tok));
        h *= 1099511628211ULL;
        return h;
    }

    static constexpr uint64_t HASH_INIT = 14695981039346656037ULL;

    // Compute incremental hashes at each page boundary.
    // Returns hashes[k] = hash of tokens[0..k*page_size-1] for k=1..n_pages.
    // Returns empty if page_size <= 0 or tokens has fewer than 1 complete page.
    std::vector<uint64_t> compute_page_hashes(const std::vector<int32_t>& tokens,
                                               int page_size) const;

    uint64_t tick() { return ++clock_; }

    int hits_ = 0;
    int misses_ = 0;
    uint64_t clock_ = 0;

    // Keyed by hash of complete-page prefix.
    std::unordered_map<uint64_t, PrefixEntry> entries_;

    // Tracks which entry a request is using (seq_id → entry hash).
    std::unordered_map<int, uint64_t> seq_to_hash_;

    // Tracks how many entries point to each cache seq_id.
    // A cache seq_id is released only when this reaches 0.
    std::unordered_map<int, int> owner_refs_;

    // Cache seq_id allocator (offset high to avoid collision with request seq_ids).
    static constexpr int CACHE_SEQ_ID_BASE = 1000000;
    int next_cache_seq_id_ = CACHE_SEQ_ID_BASE;
};

}  // namespace forge
