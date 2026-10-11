#include "forge/kv_l3_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "forge/logger.h"

namespace forge {
namespace {

constexpr size_t kDefaultMaxBytes = static_cast<size_t>(1) << 30;  // 1 GiB

std::string default_l3_dir() {
    // Prefer the system temp dir: L3 is a cache, not user data.
    std::error_code ec;
    std::filesystem::path tmp = std::filesystem::temp_directory_path(ec);
    if (ec || tmp.empty())
        tmp = std::filesystem::path(".");
    return (tmp / "forge_kv_l3").string();
}

size_t env_bytes(const char* name, size_t fallback) {
    const char* env = std::getenv(name);
    if (!env || !*env)
        return fallback;
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    if (end == env || v < 0)
        return fallback;
    return static_cast<size_t>(v);
}

}  // namespace

// =========================================================================
// FileKVStorageBackend — packed segment files
// =========================================================================

namespace {
constexpr size_t kDefaultSegmentBytes = static_cast<size_t>(64) << 20;  // 64 MiB
}  // namespace

FileKVStorageBackend::FileKVStorageBackend(std::string dir, size_t max_bytes, size_t segment_bytes)
    : dir_(std::move(dir)), max_bytes_(max_bytes),
      seg_cap_(segment_bytes ? segment_bytes : kDefaultSegmentBytes) {
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    if (ec) {
        LOG_ERROR("FileKVStorageBackend: cannot create dir '" + dir_ + "': " + ec.message());
        return;
    }
    // The index is memory-only, so files from a previous run (or from the old
    // one-blob-per-file layout) are unreachable garbage — wipe them. L3 is a
    // pure cache; nothing here is user data.
    wipe_stale_files();
    usable_ = true;
}

std::string FileKVStorageBackend::seg_path(uint32_t seg_id) const {
    char name[32];
    std::snprintf(name, sizeof(name), "seg_%06u.bin", static_cast<unsigned>(seg_id));
    return (std::filesystem::path(dir_) / name).string();
}

void FileKVStorageBackend::wipe_stale_files() {
    std::error_code ec;
    for (const auto& it : std::filesystem::directory_iterator(dir_, ec)) {
        if (!it.is_regular_file(ec))
            continue;
        const std::string name = it.path().filename().string();
        // Our own artifacts: seg_%06u.bin and the legacy %016llx.bin layout.
        const bool ours = (name.size() > 4 && name.compare(name.size() - 4, 4, ".bin") == 0 &&
                           (name.rfind("seg_", 0) == 0 || name.size() == 20));
        if (ours)
            std::filesystem::remove(it.path(), ec);
    }
}

void FileKVStorageBackend::flush_active_locked() {
    if (active_out_.is_open())
        active_out_.flush();
}

void FileKVStorageBackend::roll_segment_locked() {
    flush_active_locked();
    active_out_.close();
    active_seg_ = next_seg_++;
    active_out_.open(seg_path(active_seg_), std::ios::binary | std::ios::trunc);
    if (!active_out_) {
        LOG_ERROR("FileKVStorageBackend: cannot open segment '" + seg_path(active_seg_) + "'");
        usable_ = false;
        return;
    }
    active_off_ = 0;
    segs_[active_seg_] = Segment{};
}

bool FileKVStorageBackend::append_record_locked(uint64_t key, const void* k, size_t k_bytes,
                                                const void* v, size_t v_bytes, uint32_t* seg,
                                                uint64_t* offset, size_t* on_disk) {
    if (!active_out_.is_open())
        roll_segment_locked();
    if (!usable_)
        return false;

    const size_t rec = kRecordHeader + k_bytes + v_bytes;
    // Roll when the record would not fit in the nominal capacity. A record
    // larger than the capacity itself still gets a fresh segment (advisory
    // cap; the segment simply ends up holding that one oversized record).
    if (active_off_ > 0 && active_off_ + rec > seg_cap_)
        roll_segment_locked();
    if (!usable_)
        return false;

    *seg = active_seg_;
    *offset = active_off_;
    *on_disk = rec;

    const uint64_t header[3] = {key, static_cast<uint64_t>(k_bytes),
                                static_cast<uint64_t>(v_bytes)};
    active_out_.write(reinterpret_cast<const char*>(header), sizeof(header));
    if (k_bytes)
        active_out_.write(static_cast<const char*>(k), static_cast<std::streamsize>(k_bytes));
    if (v_bytes)
        active_out_.write(static_cast<const char*>(v), static_cast<std::streamsize>(v_bytes));
    // Offset advances even on a failed write: a torn tail is never indexed, so
    // nothing will ever read it; the segment just carries the dead bytes.
    active_off_ += rec;
    segs_[active_seg_].bytes += rec;
    if (!active_out_.good()) {
        LOG_ERROR("FileKVStorageBackend: write failed in segment " + std::to_string(active_seg_));
        return false;
    }
    return true;
}

void FileKVStorageBackend::kill_entry_locked(uint64_t key, Entry& e) {
    lru_.erase(e.lru_it);
    bytes_ -= e.bytes;
    auto sit = segs_.find(e.seg);
    if (sit != segs_.end())
        sit->second.dead += e.on_disk;
    index_.erase(key);
}

bool FileKVStorageBackend::compact_segment_locked(uint32_t seg_id) {
    const std::string path = seg_path(seg_id);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        LOG_ERROR("FileKVStorageBackend: cannot open '" + path + "' for compaction");
        return false;
    }

    // Replay the segment record-by-record, re-appending live records into the
    // active segment. The victim is never the active segment, so its file is
    // immutable while we read it (all writes happen under mu_).
    std::vector<char> kbuf, vbuf;
    size_t migrated_dead = 0;
    bool completed = true;
    while (true) {
        uint64_t header[3] = {0, 0, 0};
        in.read(reinterpret_cast<char*>(header), sizeof(header));
        if (!in.good()) {
            if (!in.eof())
                completed = false;  // torn tail: stop, keep the file
            break;
        }
        const uint64_t key = header[0];
        const size_t k_bytes = static_cast<size_t>(header[1]);
        const size_t v_bytes = static_cast<size_t>(header[2]);

        // Live only if the index still points at THIS segment's record (a key
        // re-put after this record was written is already dead here).
        auto eit = index_.find(key);
        const bool live = (eit != index_.end() && eit->second.seg == seg_id);
        if (!live) {
            in.seekg(static_cast<std::streamoff>(k_bytes + v_bytes), std::ios::cur);
            if (!in.good()) {
                completed = false;
                break;
            }
            continue;
        }

        kbuf.resize(k_bytes ? k_bytes : 1);
        vbuf.resize(v_bytes ? v_bytes : 1);
        if (k_bytes)
            in.read(kbuf.data(), static_cast<std::streamsize>(k_bytes));
        if (v_bytes)
            in.read(vbuf.data(), static_cast<std::streamsize>(v_bytes));
        if (!in.good()) {
            completed = false;
            break;  // torn tail: the record was never fully written
        }

        uint32_t nseg = 0;
        uint64_t noff = 0;
        size_t nond = 0;
        if (!append_record_locked(key, kbuf.data(), k_bytes, vbuf.data(), v_bytes, &nseg, &noff,
                                  &nond)) {
            completed = false;
            break;  // IO failure: the index entry keeps pointing at this file
        }
        eit->second.seg = nseg;
        eit->second.offset = noff;
        eit->second.on_disk = nond;
        migrated_dead += kRecordHeader + k_bytes + v_bytes;
    }

    // Look the segment up again: roll_segment_locked() inside
    // append_record_locked() may have rehashed segs_ mid-loop.
    if (auto sit = segs_.find(seg_id); sit != segs_.end()) {
        if (completed)
            sit->second.dead = sit->second.bytes;  // everything on disk is dead
        else
            sit->second.dead += migrated_dead;  // migrated copies are dead here
    }
    return completed;
}

void FileKVStorageBackend::maybe_compact_locked(uint32_t seg_id) {
    if (seg_id == active_seg_)
        return;  // the active segment is still accepting writes
    auto sit = segs_.find(seg_id);
    if (sit == segs_.end())
        return;
    const Segment& s = sit->second;
    // Fully-dead or >= 50% dead segments are rewritten to reclaim their files.
    if (s.bytes == 0 || s.dead * 2 < s.bytes)
        return;

    // The file is only removed when EVERY live record was migrated; on a
    // partial compaction it stays as the backing store of the entries still
    // pointing at it, and a later pass reclaims it.
    if (compact_segment_locked(seg_id)) {
        flush_active_locked();
        std::error_code ec;
        std::filesystem::remove(seg_path(seg_id), ec);
        segs_.erase(seg_id);
    }
}

bool FileKVStorageBackend::put(uint64_t key, const void* k, size_t k_bytes, const void* v,
                               size_t v_bytes) {
    if (!usable_)
        return false;
    std::lock_guard<std::mutex> lock(mu_);

    const size_t need = k_bytes + v_bytes;
    if (max_bytes_ && need > max_bytes_)
        return false;  // can never fit, even alone

    // Spill on write (not on reserve): the caller only loses data it is about
    // to replace, so a rejected put is always recoverable by the owner.
    while (max_bytes_ && !lru_.empty() && bytes_ + need > max_bytes_) {
        const uint64_t victim = lru_.front();
        auto vit = index_.find(victim);
        const uint32_t vseg = vit->second.seg;
        kill_entry_locked(victim, vit->second);
        ++evictions_;
        maybe_compact_locked(vseg);
        if (!usable_)
            return false;
    }

    // Overwrite: the old record becomes dead space in its segment.
    if (auto it = index_.find(key); it != index_.end()) {
        const uint32_t oseg = it->second.seg;
        kill_entry_locked(key, it->second);
        maybe_compact_locked(oseg);
    }

    uint32_t seg = 0;
    uint64_t offset = 0;
    size_t on_disk = 0;
    if (!append_record_locked(key, k, k_bytes, v, v_bytes, &seg, &offset, &on_disk))
        return false;

    lru_.push_back(key);
    Entry e;
    e.bytes = need;
    e.on_disk = on_disk;
    e.seg = seg;
    e.offset = offset;
    e.k_bytes = k_bytes;
    e.v_bytes = v_bytes;
    e.lru_it = std::prev(lru_.end());
    index_.emplace(key, e);
    bytes_ += need;
    ++puts_;
    return true;
}

bool FileKVStorageBackend::get(uint64_t key, void* k_dst, size_t k_bytes, void* v_dst,
                               size_t v_bytes) {
    if (!usable_)
        return false;
    std::lock_guard<std::mutex> lock(mu_);

    auto it = index_.find(key);
    if (it == index_.end())
        return false;
    Entry& e = it->second;
    if (e.k_bytes != k_bytes || e.v_bytes != v_bytes)
        return false;  // size mismatch: the caller's geometry changed

    if (e.seg == active_seg_)
        flush_active_locked();
    std::ifstream in(seg_path(e.seg), std::ios::binary);
    if (!in)
        return false;
    in.seekg(static_cast<std::streamoff>(e.offset), std::ios::beg);
    uint64_t header[3] = {0, 0, 0};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in.good() || header[0] != key || header[1] != k_bytes || header[2] != v_bytes)
        return false;
    if (k_bytes)
        in.read(static_cast<char*>(k_dst), static_cast<std::streamsize>(k_bytes));
    if (v_bytes)
        in.read(static_cast<char*>(v_dst), static_cast<std::streamsize>(v_bytes));
    if (!in.good())
        return false;

    lru_.splice(lru_.end(), lru_, e.lru_it);
    ++gets_;
    return true;
}

bool FileKVStorageBackend::exists(uint64_t key) const {
    std::lock_guard<std::mutex> lock(mu_);
    return index_.find(key) != index_.end();
}

void FileKVStorageBackend::touch(uint64_t key) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = index_.find(key);
    if (it == index_.end())
        return;
    lru_.splice(lru_.end(), lru_, it->second.lru_it);
}

bool FileKVStorageBackend::reserve(size_t bytes) {
    if (!usable_)
        return false;
    std::lock_guard<std::mutex> lock(mu_);
    if (!max_bytes_)
        return true;
    return bytes <= max_bytes_;
}

void FileKVStorageBackend::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    flush_active_locked();
    active_out_.close();
    std::error_code ec;
    for (const auto& s : segs_)
        std::filesystem::remove(seg_path(s.first), ec);
    segs_.clear();
    index_.clear();
    lru_.clear();
    bytes_ = 0;
    active_seg_ = 0;
    active_off_ = 0;
}

// =========================================================================
// SwapperKVBackend
// =========================================================================

bool SwapperKVBackend::put(uint64_t key, const void* k, size_t k_bytes, const void* v,
                           size_t v_bytes) {
    int layer = 0;
    int32_t page_id = 0;
    unpack_key(key, &layer, &page_id);
    return pool_.offload(layer, page_id, k, k_bytes, v, v_bytes, KVBlockSwapper::CopyKind::Host);
}

bool SwapperKVBackend::get(uint64_t key, void* k_dst, size_t k_bytes, void* v_dst, size_t v_bytes) {
    int layer = 0;
    int32_t page_id = 0;
    unpack_key(key, &layer, &page_id);
    return pool_.bring_back(layer, page_id, k_dst, k_bytes, v_dst, v_bytes,
                            KVBlockSwapper::CopyKind::Host);
}

bool SwapperKVBackend::exists(uint64_t key) const {
    int layer = 0;
    int32_t page_id = 0;
    unpack_key(key, &layer, &page_id);
    return pool_.has(layer, page_id);
}

void SwapperKVBackend::touch(uint64_t key) {
    int layer = 0;
    int32_t page_id = 0;
    unpack_key(key, &layer, &page_id);
    pool_.touch(layer, page_id);
}

bool SwapperKVBackend::reserve(size_t bytes) {
    const size_t cap = pool_.max_bytes();
    if (!cap)
        return true;
    return pool_.nbytes() + bytes <= cap;
}

// =========================================================================
// Env wiring
// =========================================================================

std::shared_ptr<KVStorageBackend> make_l3_backend_resolved(const std::string& backend_flag,
                                                           const std::string& dir_flag,
                                                           long long bytes_flag,
                                                           long long segment_flag) {
    std::string backend = backend_flag;
    if (backend.empty()) {
        const char* env = std::getenv("FORGE_KV_L3_BACKEND");
        backend = (env && *env) ? env : "";
    }
    if (backend.empty() || backend == "off" || backend == "none" || backend == "0")
        return nullptr;  // L3 off
    if (backend != "file") {
        LOG_WARN("FileKVStorageBackend: unknown L3 backend='" + backend + "'; L3 disabled");
        return nullptr;
    }

    std::string dir = dir_flag;
    if (dir.empty()) {
        const char* dir_env = std::getenv("FORGE_KV_L3_DIR");
        dir = (dir_env && *dir_env) ? std::string(dir_env) : default_l3_dir();
    }
    const size_t cap =
        bytes_flag >= 0 ? static_cast<size_t>(bytes_flag) : env_bytes("FORGE_KV_L3_BYTES", kDefaultMaxBytes);
    const size_t seg_cap = segment_flag >= 0
                               ? static_cast<size_t>(segment_flag)
                               : env_bytes("FORGE_KV_L3_SEGMENT_BYTES", kDefaultSegmentBytes);

    auto backend_ptr = std::make_shared<FileKVStorageBackend>(dir, cap, seg_cap);
    if (!backend_ptr->usable())
        return nullptr;
    LOG_INFO("KV L3 (file backend): dir=" + dir + " cap=" + std::to_string(cap) +
             " bytes segment=" + std::to_string(seg_cap) + " bytes");
    return backend_ptr;
}

std::shared_ptr<KVStorageBackend> make_l3_backend_from_env() {
    return make_l3_backend_resolved("", "", -1, -1);
}

}  // namespace forge
