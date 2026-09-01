#pragma once

// PrefixCache: radix-tree prefix sharing with LRU / ref-count eviction.
//
// Roadmap 2.1: replaces the old hash-map prefix cache (which only anchored a
// prompt's *complete-page* prefix at token 0) with a radix tree so that any
// nested common prefix can be shared between requests.
//
// Design:
//   - The tree is keyed by token sequence. Each RadixNode owns a token *segment*
//     (a run of page-aligned tokens). Descent matches segment-by-segment.
//   - Crucially, a node's `owner_seq_id` (a cache seq_id) holds the KV pages for
//     the node's *full* prefix [0, cumulative_len) — not just its own segment.
//     This lets a lookup share the matched prefix with the destination in a
//     SINGLE seq_share (the paged storage appends shared pages and sets the
//     destination logical_len from p1-p0, so one call is correct; multiple
//     segment shares would not accumulate logical_len correctly).
//   - Registering a prompt descends the tree, *splitting* nodes at the first
//     divergence so every shared sub-prefix becomes its own node. Physical KV is
//     never copied: child cache seq_ids reference the same pages as the source
//     (seq_share only bumps page ref_counts).
//   - ref_count is per node (how many live requests currently use that node's
//     prefix). Eviction drops unused *leaf* nodes bottom-up, so a node whose
//     subtree still serves requests is never orphaned.
//
// All operations go through the KVStorage abstract interface, so the cache works
// with any storage backend that implements seq_share/seq_remove/release.

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "kv_storage.h"

namespace forge {

// =========================================================================
// RadixNode — one segment of a cached prefix
// =========================================================================

struct RadixNode {
    std::vector<int32_t> seg;     // this node's token run (length multiple of page_size)
    int cumulative_len = 0;       // total prefix length covered up to the end of this node
    int owner_seq_id = -1;        // cache seq_id holding KV pages [0, cumulative_len)
    uint32_t ref_count = 0;       // # live requests currently using this node's prefix
    uint64_t last_used = 0;       // LRU timestamp (monotonic clock)
    RadixNode* parent = nullptr;  // nullptr for the root sentinel
    std::vector<std::unique_ptr<RadixNode>> children;
    bool valid = false;
};

// =========================================================================
// PrefixCache — manages the radix tree with LRU eviction
// =========================================================================

class PrefixCache {
public:
    PrefixCache() = default;

    // Try to find the longest cached prefix that is a prefix of `tokens`.
    // On hit: shares prefix KV pages [0, len) with dst_seq_id via a single
    //         storage.seq_share from the deepest matched node's owner, and
    //         increments that node's ref_count.
    // Returns the number of prefix tokens shared (0 = miss).
    int try_lookup(const std::vector<int32_t>& tokens, int dst_seq_id, KVStorage& storage);

    // Register a prefix for future requests. Called when a request finishes its
    // prefill (and did NOT use a cached prefix). Transfers prefix page ownership
    // from src_seq_id into per-node cache seq_ids, splitting the tree at any
    // divergence so shared sub-prefixes become their own nodes.
    void register_prefix(const std::vector<int32_t>& tokens, int src_seq_id, KVStorage& storage);

    // Release a request's prefix reference. Decrements the matched node's
    // ref_count and removes the prefix pages from the request's page table.
    void release_prefix(int seq_id, KVStorage& storage);

    // Evict LRU leaf nodes with ref_count == 0 until size <= max_entries.
    // Releases the node's cache seq_id when it (and its whole subtree) is unused.
    void evict_lru(size_t max_entries, KVStorage& storage);

    // Stats
    int hits() const { return hits_; }
    int misses() const { return misses_; }
    void reset_stats() {
        hits_ = 0;
        misses_ = 0;
    }

    // Clear all nodes (does not release pages — caller must handle, parity with
    // the previous hash-map implementation).
    void clear();

    // Inspection
    size_t size() const { return count_nodes(&root_); }
    bool has_request(int seq_id) const { return seq_matched_.count(seq_id) > 0; }

    // LRU capacity (default 64 entries).
    static constexpr size_t DEFAULT_MAX_ENTRIES = 64;

private:
    // Descend from `cur`, matching `tokens` segment-by-segment. Returns the
    // deepest node that the request's prefix actually matched (may be a partial
    // match within a node when the request is shorter than the node's segment),
    // and sets *matched_len to the token length covered. Returns nullptr on a
    // total miss.
    RadixNode* lookup_deepest(RadixNode* cur, const std::vector<int32_t>& tokens, int page_size,
                              int& matched_len) const;

    // Insert the registered prefix [0, total) into the tree, splitting nodes at
    // divergences. Returns the deepest node touched (the one holding [0, total)).
    RadixNode* insert_prefix(RadixNode* cur, const std::vector<int32_t>& tokens, int src_seq_id,
                             KVStorage& storage, int pos, int total);

    void collect_leaves(RadixNode* cur, std::vector<RadixNode*>& out);
    size_t count_nodes(const RadixNode* cur) const;

    uint64_t tick() { return ++clock_; }

    RadixNode root_;                                // sentinel (no tokens)
    std::unordered_map<int, int> seq_matched_;      // dst_seq -> matched token len
    std::unordered_map<int, RadixNode*> seq_node_;  // dst_seq -> matched node

    int hits_ = 0;
    int misses_ = 0;
    uint64_t clock_ = 0;

    static constexpr int CACHE_SEQ_ID_BASE = 1000000;
    int next_cache_seq_id_ = CACHE_SEQ_ID_BASE;

    friend class PrefixCacheTestFriend;  // for unit tests
};

}  // namespace forge
