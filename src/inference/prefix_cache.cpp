#include "forge/prefix_cache.h"

#include <algorithm>
#include <climits>

#include "forge/logger.h"

namespace forge {

// =========================================================================
// PrefixCache — implementation
// =========================================================================

std::vector<uint64_t> PrefixCache::compute_page_hashes(
    const std::vector<int32_t>& tokens, int page_size) const {
    std::vector<uint64_t> hashes;
    if (page_size <= 0)
        return hashes;

    int n_pages = static_cast<int>(tokens.size()) / page_size;
    if (n_pages == 0)
        return hashes;

    hashes.resize(n_pages + 1);
    hashes[0] = HASH_INIT;
    uint64_t h = HASH_INIT;
    for (int i = 0; i < n_pages * page_size; ++i) {
        h = hash_extend(h, tokens[i]);
        if ((i + 1) % page_size == 0) {
            hashes[(i + 1) / page_size] = h;
        }
    }
    return hashes;
}

int PrefixCache::try_lookup(const std::vector<int32_t>& tokens, int dst_seq_id,
                             KVStorage& storage) {
    int page_size = storage.page_size();
    if (page_size <= 0) {
        misses_++;
        return 0;
    }

    auto hashes = compute_page_hashes(tokens, page_size);
    if (hashes.empty()) {
        misses_++;
        return 0;
    }

    int n_pages = static_cast<int>(hashes.size()) - 1;

    // Try longest prefix first
    for (int k = n_pages; k >= 1; --k) {
        auto it = entries_.find(hashes[k]);
        if (it == entries_.end() || !it->second.valid)
            continue;

        auto& entry = it->second;
        if (entry.owner_seq_id < 0)
            continue;

        // Share prefix pages from cache seq_id to dst_seq_id
        if (!storage.seq_share(entry.owner_seq_id, dst_seq_id, 0, entry.token_count)) {
            LOG_WARN("PrefixCache: seq_share failed for owner=" +
                     std::to_string(entry.owner_seq_id) + " dst=" +
                     std::to_string(dst_seq_id));
            continue;
        }

        entry.ref_count++;
        entry.last_used = tick();
        seq_to_hash_[dst_seq_id] = entry.hash;

        hits_++;
        LOG_DEBUG("PrefixCache HIT: dst=" + std::to_string(dst_seq_id) +
                  " prefix_len=" + std::to_string(entry.token_count) +
                  " owner=" + std::to_string(entry.owner_seq_id));
        return entry.token_count;
    }

    misses_++;
    LOG_DEBUG("PrefixCache MISS: dst=" + std::to_string(dst_seq_id) +
              " prompt_len=" + std::to_string(tokens.size()));
    return 0;
}

void PrefixCache::register_prefix(const std::vector<int32_t>& tokens, int src_seq_id,
                                   KVStorage& storage) {
    int page_size = storage.page_size();
    if (page_size <= 0)
        return;

    auto hashes = compute_page_hashes(tokens, page_size);
    if (hashes.empty())
        return;

    int n_pages = static_cast<int>(hashes.size()) - 1;

    // If the longest prefix is already registered, skip
    if (entries_.count(hashes[n_pages]) > 0) {
        LOG_DEBUG("PrefixCache: prefix already registered (n_pages=" +
                  std::to_string(n_pages) + ")");
        return;
    }

    // Allocate a cache seq_id to hold the prefix pages
    int cache_seq_id = next_cache_seq_id_++;

    // Transfer prefix pages from src to cache seq_id:
    //   seq_share  → cache seq_id gets a reference to prefix pages (ref_count++)
    //   seq_remove → src releases its reference (ref_count--)
    // Net effect: pages now held by cache_seq_id instead of src_seq_id.
    if (!storage.seq_share(src_seq_id, cache_seq_id, 0, n_pages * page_size)) {
        LOG_WARN("PrefixCache: seq_share failed during register (src=" +
                 std::to_string(src_seq_id) + ")");
        return;
    }
    storage.seq_remove(src_seq_id, 0, n_pages * page_size);

    // Register entries for each complete-page prefix length
    for (int k = 1; k <= n_pages; ++k) {
        if (entries_.count(hashes[k]) > 0)
            continue;

        PrefixEntry entry;
        entry.hash = hashes[k];
        entry.token_count = k * page_size;
        entry.owner_seq_id = cache_seq_id;
        entry.ref_count = 0;
        entry.last_used = tick();
        entry.valid = true;
        entries_[hashes[k]] = std::move(entry);
    }

    // Track how many entries reference this cache seq_id
    owner_refs_[cache_seq_id] = n_pages;

    LOG_DEBUG("PrefixCache REGISTER: src=" + std::to_string(src_seq_id) +
              " → cache_seq=" + std::to_string(cache_seq_id) +
              " n_pages=" + std::to_string(n_pages) +
              " total_entries=" + std::to_string(entries_.size()));
}

void PrefixCache::release_prefix(int seq_id, KVStorage& storage) {
    auto it = seq_to_hash_.find(seq_id);
    if (it == seq_to_hash_.end())
        return;

    uint64_t h = it->second;
    seq_to_hash_.erase(it);

    auto entry_it = entries_.find(h);
    if (entry_it == entries_.end())
        return;

    auto& entry = entry_it->second;
    // Release the request's prefix reference (decrements page ref_counts)
    if (entry.token_count > 0) {
        storage.seq_remove(seq_id, 0, entry.token_count);
    }
    if (entry.ref_count > 0) {
        entry.ref_count--;
    }

    LOG_DEBUG("PrefixCache RELEASE: seq=" + std::to_string(seq_id) +
              " prefix_len=" + std::to_string(entry.token_count) +
              " ref_count=" + std::to_string(entry.ref_count));
}

void PrefixCache::evict_lru(size_t max_entries, KVStorage& storage) {
    while (entries_.size() > max_entries) {
        // Find LRU entry with ref_count == 0
        uint64_t lru_hash = 0;
        uint64_t oldest_time = UINT64_MAX;
        bool found = false;

        for (const auto& [h, entry] : entries_) {
            if (entry.ref_count == 0 && entry.last_used < oldest_time) {
                oldest_time = entry.last_used;
                lru_hash = h;
                found = true;
            }
        }

        if (!found)
            break;  // all entries in use

        auto& entry = entries_[lru_hash];
        int owner = entry.owner_seq_id;
        entries_.erase(lru_hash);

        // Decrement owner ref count; release cache seq_id when no entries remain
        auto ref_it = owner_refs_.find(owner);
        if (ref_it != owner_refs_.end()) {
            ref_it->second--;
            if (ref_it->second <= 0) {
                storage.release(owner);
                owner_refs_.erase(ref_it);
                LOG_DEBUG("PrefixCache EVICT: released cache_seq=" +
                          std::to_string(owner));
            }
        }
    }
}

void PrefixCache::clear() {
    entries_.clear();
    seq_to_hash_.clear();
    owner_refs_.clear();
    hits_ = 0;
    misses_ = 0;
    clock_ = 0;
}

}  // namespace forge
