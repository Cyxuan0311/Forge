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

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kv_storage.h"

namespace forge {

// =========================================================================
// Two-tier residency (GPU + host blob pool) — parity with sglang HiCache
// =========================================================================

// How KV pages get their host copy in the prefix blob pool.
//   Off          — legacy behavior: eviction destroys the node outright.
//   WriteBack    — (default) the host copy is made lazily, at eviction time:
//                  a device-resident victim is first backed up into the blob
//                  pool, then its device pages are released. The node survives
//                  as host-resident and can be restored on a later hit.
//   WriteThrough — the host copy is made eagerly at register time (every newly
//                  registered node is mirrored immediately); eviction then only
//                  releases the device pages.
enum class KVWritePolicy { Off, WriteBack, WriteThrough };

// Where a prefix hit was served from.
enum class PrefixHitKind { Miss, Device, Host };

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

    // ---- Two-tier residency ----
    // device_resident: owner_seq_id's page table currently holds pages
    //   [0, cumulative_len) (the classic invariant). Backup+release clears it;
    //   restore_node() re-establishes it (sharing from the nearest
    //   device-resident ancestor and blob-restoring the host-only segs).
    // host_resident: this node's seg pages are backed up in the storage's
    //   prefix blob pool under host_blob_keys. Seg content is immutable once
    //   registered, so a blob never needs re-copying after its first backup
    //   (re-backup only happens if the blob itself was dropped, e.g. by L3).
    // host_blob_keys[layer][i] = blob id of the node's i-th seg page for that
    //   layer (seg-relative: absolute page = seg_start_page + i, where
    //   seg_start_page = (cumulative_len - seg.size()) / page_size). A node
    //   split slices these vectors at the split page — the blob ids stay
    //   valid because they key immutable page content by absolute position.
    bool device_resident = true;
    bool host_resident = false;
    std::vector<std::vector<int32_t>> host_blob_keys;

    // ---- Phase 2: in-flight async work ----
    // A node with pending work must not be restored, backed up again, or
    // detached: the background job owns its pages (backup) or its tree state
    // (restore) until the job is harvested.
    bool backup_pending = false;
    bool restore_pending = false;

    // ---- Phase 3: L3 residency ----
    // The node's blobs were spilled from the host pool down to L3: host_blob_keys
    // are still valid (they name L3 entries) but the pool no longer holds bytes,
    // so the node costs no host memory until it is restored.
    bool blob_in_l3 = false;
};

// =========================================================================
// PrefixCache — manages the radix tree with LRU eviction
// =========================================================================

class PrefixCache {
public:
    PrefixCache() = default;
    explicit PrefixCache(KVWritePolicy policy) : policy_(policy) {}
    ~PrefixCache();

    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    void set_policy(KVWritePolicy policy) { policy_ = policy; }
    KVWritePolicy policy() const { return policy_; }

    // ---- Phase 2: asynchronous backup / restore ----
    //
    // Backup (device -> host blob copy) is the expensive half of an eviction;
    // with async_backup enabled the copy runs on a worker thread while the node
    // keeps its device pages (still readable by any live request) until the job
    // is harvested and the pages are released. Restore can be prefetched ahead
    // of time for a request that has been submitted but not scheduled yet, so
    // the host -> device rebuild overlaps with useful work.
    void set_async_backup(bool on) { async_backup_ = on; }
    bool async_backup() const { return async_backup_; }
    void set_prefetch(bool on) { prefetch_ = on; }
    bool prefetch() const { return prefetch_; }
    // Bounded wait (ms) for a prefetched restore to finish before a lookup
    // gives up and reports a miss instead of blocking the caller forever.
    void set_restore_wait_ms(int ms) { restore_wait_ms_ = ms; }
    int restore_wait_ms() const { return restore_wait_ms_; }

    // Warm the deepest host-resident node covering `tokens` (no-op when the
    // match is device-resident, prefetch is off, or a job is already queued).
    void prefetch_prefix(const std::vector<int32_t>& tokens, KVStorage& storage);

    // Block until every queued async backup/restore job has been processed and
    // committed (pending_jobs() == 0). Used by tests and by any caller that
    // needs the background worker fully drained before proceeding.
    void flush(KVStorage& storage) { harvest_async(storage, /*wait_all=*/true); }

    // Try to find the longest cached prefix that is a prefix of `tokens`.
    // On a device hit: shares prefix KV pages [0, len) with dst_seq_id via a
    //         single storage.seq_share from the deepest matched node's owner,
    //         and increments that node's ref_count.
    // On a host hit: the matched node (and any host-resident nodes between it
    //         and its nearest device-resident ancestor) is first restored into
    //         a fresh cache seq (shared pages + blob copies), which is adopted
    //         as the node's new owner; the share then proceeds as above.
    // Returns the number of prefix tokens shared (0 = miss); `kind` (optional)
    // reports where the hit was served from.
    int try_lookup(const std::vector<int32_t>& tokens, int dst_seq_id, KVStorage& storage,
                   PrefixHitKind* kind = nullptr);

    // Register a prefix for future requests. Called when a request finishes its
    // prefill (and did NOT use a cached prefix). Transfers prefix page ownership
    // from src_seq_id into per-node cache seq_ids, splitting the tree at any
    // divergence so shared sub-prefixes become their own nodes. A host-resident
    // node is restored before being split (its owner must hold pages again).
    // With WriteThrough, every newly registered node is immediately mirrored
    // into the prefix blob pool.
    void register_prefix(const std::vector<int32_t>& tokens, int src_seq_id, KVStorage& storage);

    // Release a request's prefix reference. Decrements the matched node's
    // ref_count and removes the prefix pages from the request's page table.
    void release_prefix(int seq_id, KVStorage& storage);

    // Two-tier LRU eviction:
    //   1. host tier: while the storage's prefix blob pool is over its byte
    //      budget, remove the oldest fully-host-resident subtree (dropping its
    //      blobs) — bounded host memory.
    //   2. device tier: while the number of device-resident nodes exceeds
    //      max_entries, back the LRU victim up into the blob pool (per policy)
    //      and release its device pages; the node survives as host-resident.
    //      Victims whose backup fails (pool over budget) fall back to the
    //      legacy hard eviction so device memory is always reclaimed.
    // With KVWritePolicy::Off this is exactly the legacy behavior (hard-evict
    // LRU unused leaves until the total node count <= max_entries).
    void evict_lru(size_t max_entries, KVStorage& storage);

    // Stats
    int hits() const { return hits_; }             // all served hits (device + host)
    int host_hits() const { return hits_host_; }   // subset restored from the host pool
    int misses() const { return misses_; }
    // Phase 2: background work counters (`pending_jobs` is 0 when idle).
    int pending_jobs() const { return pending_; }
    int async_backups() const { return async_backups_; }
    int async_restores() const { return async_restores_; }
    void reset_stats() {
        hits_ = 0;
        hits_host_ = 0;
        misses_ = 0;
        async_backups_ = 0;
        async_restores_ = 0;
    }

    // Clear all nodes, dropping their host blobs when a storage is given (does
    // not release device pages — caller must handle, parity with the previous
    // hash-map implementation).
    void clear(KVStorage* storage = nullptr);

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

    // ---- Two-tier residency helpers ----

    // Copy `node`'s seg pages into the prefix blob pool (in-place overwrite of
    // the existing blob ids when the node is already host-resident). Does NOT
    // touch the device pages; the caller decides when to release them.
    // Fails (returning false, node untouched) when the node is not a backup
    // candidate or the pool rejects the copy (e.g. over budget).
    bool backup_node(RadixNode* node, KVStorage& storage);
    // Same contract as backup_node, but the device -> host copy runs on the
    // worker; true means the job is queued (the node is backup_pending until
    // harvest_async commits it). Caller must hold mu_.
    bool backup_node_async(RadixNode* node, KVStorage& storage);

    // Make `node` device-resident again: adopt a fresh cache seq holding
    // [0, cumulative_len) — shared from the nearest device-resident ancestor,
    // then blob-restored for every host-resident node on the path below it.
    // Every path node is promoted to device-resident (sharing the rebuilt
    // prefix up to its own cumulative length). Returns false on failure (any
    // temporary sequence is released; the tree is unchanged).
    bool restore_node(RadixNode* node, KVStorage& storage);

    bool has_device_resident_descendant(const RadixNode* node) const;
    size_t count_device_resident(const RadixNode* cur) const;
    // Oldest node that may give up its device pages: valid, device-resident,
    // ref_count == 0, and no device-resident descendants (they would keep the
    // seg pages alive anyway, making the release pointless).
    void collect_device_evictable(const RadixNode* cur, RadixNode** best, uint64_t* oldest) const;
    // True when the subtree under `cur` is fully host-resident garbage (every
    // node host-only, unused); picks the oldest removable host node on the way.
    bool collect_host_evictable(RadixNode* cur, RadixNode** best);
    void drop_host_blobs_recursive(RadixNode* node, KVStorage& storage);
    // Mark a subtree's blobs as L3-resident (their pool copies are gone).
    void mark_blobs_in_l3(RadixNode* node);
    void detach_node(RadixNode* node);
    // Shrink the host blob pool to the storage's byte budget by removing the
    // oldest fully-host-resident subtrees.
    void evict_host_over_budget(KVStorage& storage);
    // Legacy single-tier eviction (KVWritePolicy::Off).
    void evict_lru_legacy(size_t max_entries, KVStorage& storage);

    // ---- Phase 2: background worker ----
    //
    // One worker thread serves two job kinds:
    //   Backup  — copy a victim's seg pages into the prefix blob pool (the node
    //             keeps its device pages until the job is harvested, so the
    //             copy can never race with a reader).
    //   Restore — rebuild a host-resident node (and its host-resident ancestor
    //             chain) into a fresh cache seq; runs under mu_ because it
    //             mutates the tree.
    // Jobs are enqueued by the owner thread and committed (harvested) there, so
    // all tree mutation stays on one thread unless it explicitly waits.
    struct AsyncJob {
        enum class Kind { Backup, Restore };
        Kind kind = Kind::Backup;
        RadixNode* node = nullptr;
        KVStorage* storage = nullptr;
        int seq_id = -1;      // Backup: owner seq whose pages are copied
        int p0 = 0;
        int p1 = 0;
        std::vector<std::vector<int32_t>> keys;  // Backup: blob ids (out)
        bool ok = false;
    };

    bool start_worker();
    void worker_loop();
    void enqueue(std::unique_ptr<AsyncJob> job, KVStorage& storage);
    // Commit finished jobs. With `wait_all`, block (bounded) until every queued
    // job is done — used when the caller needs the pages freed right now.
    // Must be called with mu_ held.
    void harvest_async(KVStorage& storage, bool wait_all);
    void shutdown_worker();
    // Wait (bounded by restore_wait_ms_) for a prefetched restore of `node`.
    // `lock` (on mu_) is released while waiting. Returns true once the node is
    // no longer restore_pending.
    bool wait_for_restore(std::unique_lock<std::mutex>& lock, RadixNode* node);
    void clear_pending_flags(RadixNode* cur);

    void collect_leaves(RadixNode* cur, std::vector<RadixNode*>& out);
    size_t count_nodes(const RadixNode* cur) const;

    uint64_t tick() { return ++clock_; }

    RadixNode root_;                                // sentinel (no tokens)
    std::unordered_map<int, int> seq_matched_;      // dst_seq -> matched token len
    std::unordered_map<int, RadixNode*> seq_node_;  // dst_seq -> matched node

    KVWritePolicy policy_ = KVWritePolicy::WriteBack;
    int hits_ = 0;
    int hits_host_ = 0;
    int misses_ = 0;
    uint64_t clock_ = 0;

    static constexpr int CACHE_SEQ_ID_BASE = 1000000;
    int next_cache_seq_id_ = CACHE_SEQ_ID_BASE;

    // ---- Phase 2: async state ----
    // mu_ guards the tree: the worker takes it for restore jobs, so every
    // public entry point holds it too. Lock order is always mu_ -> jobs_mu_.
    mutable std::mutex mu_;
    std::mutex jobs_mu_;
    std::condition_variable jobs_cv_;
    std::condition_variable tree_cv_;  // signalled under mu_ (restore finished)
    std::deque<std::unique_ptr<AsyncJob>> jobs_;
    std::deque<std::unique_ptr<AsyncJob>> done_;
    std::thread worker_;
    bool worker_stop_ = false;
    int pending_ = 0;

    bool async_backup_ = true;
    bool prefetch_ = false;
    int restore_wait_ms_ = 200;
    int async_backups_ = 0;
    int async_restores_ = 0;

    friend class PrefixCacheTestFriend;  // for unit tests
};

}  // namespace forge
