#pragma once

// kv_l3_backend.h — the third tier of the multi-level KV cache.
//
// L1 (device pages) and L2 (host blob pool, KVBlockSwapper) were added in
// Phase 1/2. This header defines the L3 tier: a byte-addressed store behind the
// host pool, so a blob that loses its L2 slot can still be recovered instead of
// being thrown away (parity with sglang's HiCache file backend + LRU evictor).
//
//   KVStorageBackend    — the abstract byte store (put/get/exists/touch/reserve).
//   FileKVStorageBackend— directory-backed store with an in-memory LRU index.
//   SwapperKVBackend    — adapter that exposes a KVBlockSwapper (L2 pool)
//                         through the same interface, so both pools — the
//                         request-swap pool and the prefix blob pool — are
//                         reachable through one abstraction.
//
// Env knobs (see kv_l3_backend.cpp):
//   FORGE_KV_L3_BACKEND      = file | off   (default off)
//   FORGE_KV_L3_DIR          = <path>       (default: <temp>/forge_kv_l3)
//   FORGE_KV_L3_BYTES        = <int>        (default 1 GiB; 0 = grow-only)
//   FORGE_KV_L3_SEGMENT_BYTES= <int>        (default 64 MiB; packed segment capacity)

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "forge/kv_block_swapper.h"

namespace forge {

class KVStorageBackend {
public:
    virtual ~KVStorageBackend() = default;

    // Store the key/value byte blobs under `key`. Overwrites an existing entry.
    virtual bool put(uint64_t key, const void* k, size_t k_bytes, const void* v,
                     size_t v_bytes) = 0;
    // Read the blobs back into caller-owned buffers of the given sizes.
    virtual bool get(uint64_t key, void* k_dst, size_t k_bytes, void* v_dst, size_t v_bytes) = 0;
    virtual bool exists(uint64_t key) const = 0;
    // Refresh the entry's LRU position (a use).
    virtual void touch(uint64_t key) = 0;
    // Budget check: can `bytes` be stored (possibly after evicting)?
    virtual bool reserve(size_t bytes) = 0;

    virtual size_t bytes() const = 0;
    virtual size_t entries() const = 0;

    // Counters for tests / diagnostics.
    virtual size_t puts() const { return 0; }
    virtual size_t gets() const { return 0; }
    virtual size_t evictions() const { return 0; }
};

// -------------------------------------------------------------------------
// FileKVStorageBackend — packed segment files, LRU-evicted by byte budget
// -------------------------------------------------------------------------
//
// Blobs are appended into fixed-capacity segment files instead of one file per
// blob: a churned pool spills hundreds of thousands of (layer, page) blobs, and
// per-blob files exhausted the inode budget long before the byte budget.
//
// Segment record layout: [u64 key][u64 k_bytes][u64 v_bytes][k bytes][v bytes].
// The in-memory index maps key -> (segment, offset, sizes) + LRU position.
// Evicting a blob only marks its record dead inside the segment; when a
// non-active segment is >= 50% dead it is compacted (live records appended to
// the active segment, then the file removed), keeping the on-disk file count
// bounded by bytes / segment capacity. Restart starts with an empty index and
// wipes stale segment files (L3 is a pure cache).
class FileKVStorageBackend : public KVStorageBackend {
public:
    FileKVStorageBackend(std::string dir, size_t max_bytes, size_t segment_bytes = 0);

    bool put(uint64_t key, const void* k, size_t k_bytes, const void* v, size_t v_bytes) override;
    bool get(uint64_t key, void* k_dst, size_t k_bytes, void* v_dst, size_t v_bytes) override;
    bool exists(uint64_t key) const override;
    void touch(uint64_t key) override;
    bool reserve(size_t bytes) override;

    size_t bytes() const override { return bytes_; }
    size_t entries() const override { return index_.size(); }
    size_t puts() const override { return puts_; }
    size_t gets() const override { return gets_; }
    size_t evictions() const override { return evictions_; }

    const std::string& dir() const { return dir_; }
    size_t max_bytes() const { return max_bytes_; }
    bool usable() const { return usable_; }

    // Drop everything (index + files). Used by tests and reset paths.
    void clear();

private:
    struct Entry {
        size_t bytes = 0;      // live payload bytes (k+v), budget accounting
        size_t on_disk = 0;    // record size incl. header, dead-space accounting
        uint32_t seg = 0;      // segment file holding the record
        uint64_t offset = 0;   // record offset within that segment
        uint64_t k_bytes = 0;
        uint64_t v_bytes = 0;
        std::list<uint64_t>::iterator lru_it{};
    };
    struct Segment {
        uint64_t bytes = 0;  // total appended (live + dead)
        uint64_t dead = 0;   // bytes of dead records
    };

    static constexpr size_t kRecordHeader = 3 * sizeof(uint64_t);

    std::string seg_path(uint32_t seg_id) const;
    void wipe_stale_files();
    void roll_segment_locked();  // start a fresh active segment
    void flush_active_locked();
    bool append_record_locked(uint64_t key, const void* k, size_t k_bytes, const void* v,
                              size_t v_bytes, uint32_t* seg, uint64_t* offset, size_t* on_disk);
    // Remove an entry's index slot and mark its on-disk record dead.
    void kill_entry_locked(uint64_t key, Entry& e);
    // Rewrite a dead-heavy non-active segment into the active one. Returns
    // true only when every record was replayed to EOF (the caller may then
    // delete the file); false leaves the file as backing store for entries
    // still pointing at it.
    bool compact_segment_locked(uint32_t seg_id);
    void maybe_compact_locked(uint32_t seg_id);

    mutable std::mutex mu_;
    std::string dir_;
    size_t max_bytes_;
    size_t seg_cap_;
    std::unordered_map<uint64_t, Entry> index_;
    std::list<uint64_t> lru_;  // front = least recently used
    std::unordered_map<uint32_t, Segment> segs_;
    std::ofstream active_out_;
    uint32_t active_seg_ = 0;
    uint64_t active_off_ = 0;
    uint32_t next_seg_ = 0;
    size_t bytes_ = 0;
    size_t puts_ = 0;
    size_t gets_ = 0;
    size_t evictions_ = 0;
    bool usable_ = false;
};

// -------------------------------------------------------------------------
// SwapperKVBackend — KVBlockSwapper seen as a KVStorageBackend
// -------------------------------------------------------------------------
//
// Keys are packed as (layer << 32) | page_id, matching the swapper's own
// (layer, page) identity. Sources/destinations are host buffers (CopyKind::Host)
// because a caller moving data between tiers always stages through host memory.
class SwapperKVBackend : public KVStorageBackend {
public:
    explicit SwapperKVBackend(KVBlockSwapper& pool) : pool_(pool) {}

    bool put(uint64_t key, const void* k, size_t k_bytes, const void* v, size_t v_bytes) override;
    bool get(uint64_t key, void* k_dst, size_t k_bytes, void* v_dst, size_t v_bytes) override;
    bool exists(uint64_t key) const override;
    void touch(uint64_t key) override;
    bool reserve(size_t bytes) override;

    size_t bytes() const override { return pool_.nbytes(); }
    size_t entries() const override { return pool_.num_pages(); }

    static uint64_t make_key(int layer, int32_t page_id) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
               static_cast<uint64_t>(static_cast<uint32_t>(page_id));
    }
    static void unpack_key(uint64_t key, int* layer, int32_t* page_id) {
        *layer = static_cast<int>(static_cast<uint32_t>(key >> 32));
        *page_id = static_cast<int32_t>(static_cast<uint32_t>(key & 0xffffffffu));
    }

private:
    KVBlockSwapper& pool_;
};

// Build the L3 backend from the environment. Returns nullptr when L3 is off.
std::shared_ptr<KVStorageBackend> make_l3_backend_from_env();

// Build the L3 backend with explicit --kv-l3-* flag values, falling back to
// the FORGE_KV_L3_* env vars (then the defaults) for any flag left empty/-1.
// `backend_flag`: "" | "file" | "off" ("off" wins over the env). Returns
// nullptr when L3 is off.
std::shared_ptr<KVStorageBackend> make_l3_backend_resolved(const std::string& backend_flag,
                                                           const std::string& dir_flag,
                                                           long long bytes_flag,
                                                           long long segment_flag);

}  // namespace forge
