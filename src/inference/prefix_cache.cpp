#include "forge/prefix_cache.h"

#include <algorithm>
#include <climits>

#include "forge/logger.h"

namespace forge {

// =========================================================================
// PrefixCache — radix-tree implementation
// =========================================================================

size_t PrefixCache::count_nodes(const RadixNode* cur) const {
    size_t n = 0;
    for (const auto& c : cur->children) {
        if (!c->valid)
            continue;
        n += 1 + count_nodes(c.get());
    }
    return n;
}

void PrefixCache::collect_leaves(RadixNode* cur, std::vector<RadixNode*>& out) {
    for (auto& c : cur->children) {
        if (!c->valid)
            continue;
        if (c->children.empty())
            out.push_back(c.get());
        else
            collect_leaves(c.get(), out);
    }
}

namespace {

// Find the child of `cur` whose first page equals the request's page at `pos`.
// Returns nullptr if no child shares the first token / page (a divergence at
// page 0). Only one child can match a given page, so the first match wins.
forge::RadixNode* find_matching_child(forge::RadixNode* cur, const std::vector<int32_t>& tokens,
                                      int pos, int page_size) {
    int32_t key = tokens[pos];
    for (auto& c : cur->children) {
        if (!c->valid || c->seg.empty() || c->seg[0] != key)
            continue;
        bool page0_match = true;
        for (int t = 0; t < page_size; ++t) {
            if (tokens[pos + t] != c->seg[t]) {
                page0_match = false;
                break;
            }
        }
        if (page0_match)
            return c.get();
    }
    return nullptr;
}

}  // namespace

RadixNode* PrefixCache::lookup_deepest(RadixNode* cur, const std::vector<int32_t>& tokens,
                                       int page_size, int& matched_len) const {
    int pos = 0;
    int n = static_cast<int>(tokens.size());
    RadixNode* deepest = nullptr;
    int deepest_pages = 0;

    while (pos < n) {
        RadixNode* child = find_matching_child(cur, tokens, pos, page_size);
        if (!child)
            break;

        int seg_pages = static_cast<int>(child->seg.size()) / page_size;
        int avail_pages = (n - pos) / page_size;  // full pages we can compare
        int cmp_pages = std::min(seg_pages, avail_pages);

        int i = 0;
        for (; i < cmp_pages; ++i) {
            bool eq = true;
            for (int t = 0; t < page_size; ++t) {
                if (tokens[pos + i * page_size + t] != child->seg[i * page_size + t]) {
                    eq = false;
                    break;
                }
            }
            if (!eq)
                break;
        }

        if (i == cmp_pages) {
            // All comparable pages matched.
            if (seg_pages == avail_pages) {
                deepest = child;
                deepest_pages = seg_pages;
                break;
            } else if (seg_pages < avail_pages) {
                deepest = child;
                deepest_pages = seg_pages;
                pos += seg_pages * page_size;
                cur = child;
            } else {  // request is a prefix of child's segment
                deepest = child;
                deepest_pages = avail_pages;
                break;
            }
        } else {
            // Divergence at page i. Pages [0, i) are a fully shared, page-aligned
            // prefix; page i differs (its KV depends on the differing token) so it
            // must NOT be shared.
            if (i > 0) {
                deepest = child;
                deepest_pages = i;
            }
            break;
        }
    }

    matched_len = deepest_pages * page_size;
    return deepest;
}

int PrefixCache::try_lookup(const std::vector<int32_t>& tokens, int dst_seq_id,
                            KVStorage& storage) {
    int page_size = storage.page_size();
    if (page_size <= 0) {
        misses_++;
        return 0;
    }

    int matched_len = 0;
    RadixNode* node = lookup_deepest(&root_, tokens, page_size, matched_len);
    if (!node || matched_len < page_size) {
        misses_++;
        LOG_DEBUG("PrefixCache MISS: dst=" + std::to_string(dst_seq_id) +
                  " prompt_len=" + std::to_string(tokens.size()));
        return 0;
    }

    // Share [0, matched_len) from the deepest matched node's owner in ONE call.
    if (!storage.seq_share(node->owner_seq_id, dst_seq_id, 0, matched_len)) {
        LOG_WARN("PrefixCache: seq_share failed for owner=" + std::to_string(node->owner_seq_id) +
                 " dst=" + std::to_string(dst_seq_id));
        misses_++;
        return 0;
    }

    node->ref_count++;
    node->last_used = tick();
    seq_matched_[dst_seq_id] = matched_len;
    seq_node_[dst_seq_id] = node;

    hits_++;
    LOG_DEBUG("PrefixCache HIT: dst=" + std::to_string(dst_seq_id) + " prefix_len=" +
              std::to_string(matched_len) + " owner=" + std::to_string(node->owner_seq_id));
    return matched_len;
}

RadixNode* PrefixCache::insert_prefix(RadixNode* cur, const std::vector<int32_t>& tokens,
                                      int src_seq_id, KVStorage& storage, int pos, int total) {
    if (pos >= total)
        return cur;

    RadixNode* child = find_matching_child(cur, tokens, pos, storage.page_size());
    int page_size = storage.page_size();

    if (!child) {
        // No branch here yet — create a node covering [pos, total).
        auto node = std::make_unique<RadixNode>();
        node->seg.assign(tokens.begin() + pos, tokens.begin() + total);
        node->cumulative_len = total;
        node->owner_seq_id = next_cache_seq_id_++;
        node->ref_count = 0;
        node->last_used = tick();
        node->valid = true;
        node->parent = cur;
        if (!storage.seq_share(src_seq_id, node->owner_seq_id, 0, total)) {
            LOG_WARN("PrefixCache: seq_share failed during register (src=" +
                     std::to_string(src_seq_id) + ")");
            return cur;
        }
        RadixNode* raw = node.get();
        cur->children.push_back(std::move(node));
        return raw;
    }

    int seg_pages = static_cast<int>(child->seg.size()) / page_size;
    int avail_pages = (total - pos) / page_size;
    int cmp_pages = std::min(seg_pages, avail_pages);
    int i = 0;
    for (; i < cmp_pages; ++i) {
        bool eq = true;
        for (int t = 0; t < page_size; ++t) {
            if (tokens[pos + i * page_size + t] != child->seg[i * page_size + t]) {
                eq = false;
                break;
            }
        }
        if (!eq)
            break;
    }

    if (i == cmp_pages) {
        // All comparable pages matched fully.
        if (seg_pages == avail_pages) {
            return child;  // exact match: child already caches [0, total)
        } else if (seg_pages < avail_pages) {
            return insert_prefix(child, tokens, src_seq_id, storage, pos + seg_pages * page_size,
                                 total);
        } else {           // seg_pages > avail_pages
            return child;  // request is a prefix of child's segment
        }
    }

    // Divergence at page i (0 <= i < cmp_pages). Pages [0, i) are a fully shared,
    // page-aligned prefix; page i differs and must not be shared.
    int page_boundary = i * page_size;  // shared prefix length (page-aligned)
    if (page_boundary == 0) {
        // Divergence within the first page: no shared prefix with `child`.
        // Create a *sibling* node under `cur` for this request's prefix.
        auto node = std::make_unique<RadixNode>();
        node->seg.assign(tokens.begin() + pos, tokens.begin() + total);
        node->cumulative_len = total;
        node->owner_seq_id = next_cache_seq_id_++;
        node->ref_count = 0;
        node->last_used = tick();
        node->valid = true;
        node->parent = cur;
        if (!storage.seq_share(src_seq_id, node->owner_seq_id, 0, total)) {
            LOG_WARN("PrefixCache: seq_share failed during register (src=" +
                     std::to_string(src_seq_id) + ")");
            return cur;
        }
        RadixNode* raw = node.get();
        cur->children.push_back(std::move(node));
        return raw;
    }

    // Split `child` into prefix node P ([0, page_boundary)) + suffix
    // ([page_boundary, seg_len)). P becomes a shared node; the request's
    // remainder is added as a fresh branch under P.
    int child_cum = child->cumulative_len;

    auto pnode = std::make_unique<RadixNode>();
    pnode->seg.assign(child->seg.begin(), child->seg.begin() + page_boundary);
    pnode->cumulative_len = pos + page_boundary;
    pnode->owner_seq_id = next_cache_seq_id_++;
    pnode->ref_count = 0;
    pnode->last_used = tick();
    pnode->valid = true;
    pnode->parent = cur;
    if (!storage.seq_share(child->owner_seq_id, pnode->owner_seq_id, 0, pos + page_boundary)) {
        LOG_WARN("PrefixCache: split seq_share failed");
        return child;
    }

    std::vector<int32_t> suffix_seg(child->seg.begin() + page_boundary, child->seg.end());
    std::unique_ptr<RadixNode> c_owned;
    for (auto it = cur->children.begin(); it != cur->children.end(); ++it) {
        if (it->get() == child) {
            c_owned = std::move(*it);
            cur->children.erase(it);
            break;
        }
    }
    child->seg = std::move(suffix_seg);
    child->parent = pnode.get();
    pnode->children.push_back(std::move(c_owned));

    RadixNode* praw = pnode.get();
    cur->children.push_back(std::move(pnode));

    return insert_prefix(praw, tokens, src_seq_id, storage, pos + page_boundary, total);
}

void PrefixCache::register_prefix(const std::vector<int32_t>& tokens, int src_seq_id,
                                  KVStorage& storage) {
    int page_size = storage.page_size();
    if (page_size <= 0)
        return;

    int total = static_cast<int>(tokens.size() / page_size) * page_size;
    if (total < page_size)
        return;

    insert_prefix(&root_, tokens, src_seq_id, storage, 0, total);

    // Release the source request's prefix pages; they are now owned by the
    // per-node cache seq_ids (page references are shared, KV is not copied).
    storage.seq_remove(src_seq_id, 0, total);

    LOG_DEBUG("PrefixCache REGISTER: src=" + std::to_string(src_seq_id) +
              " total=" + std::to_string(total) + " nodes=" + std::to_string(count_nodes(&root_)));
}

void PrefixCache::release_prefix(int seq_id, KVStorage& storage) {
    auto it = seq_matched_.find(seq_id);
    if (it == seq_matched_.end())
        return;

    int len = it->second;
    auto nit = seq_node_.find(seq_id);
    if (nit != seq_node_.end() && nit->second) {
        if (nit->second->ref_count > 0)
            nit->second->ref_count--;
    }

    seq_matched_.erase(it);
    seq_node_.erase(seq_id);

    if (len > 0)
        storage.seq_remove(seq_id, 0, len);

    LOG_DEBUG("PrefixCache RELEASE: seq=" + std::to_string(seq_id) +
              " prefix_len=" + std::to_string(len));
}

void PrefixCache::evict_lru(size_t max_entries, KVStorage& storage) {
    while (count_nodes(&root_) > max_entries) {
        std::vector<RadixNode*> leaves;
        collect_leaves(&root_, leaves);

        RadixNode* victim = nullptr;
        uint64_t oldest = UINT64_MAX;
        for (auto* l : leaves) {
            if (l->ref_count == 0 && l->valid && l->last_used < oldest) {
                oldest = l->last_used;
                victim = l;
            }
        }
        if (!victim)
            break;  // every leaf is in use

        RadixNode* parent = victim->parent;
        if (parent) {
            for (auto it = parent->children.begin(); it != parent->children.end(); ++it) {
                if (it->get() == victim) {
                    parent->children.erase(it);
                    break;
                }
            }
        }
        if (victim->owner_seq_id >= 0)
            storage.release(victim->owner_seq_id);
        // victim is destroyed via its unique_ptr
    }
}

void PrefixCache::clear() {
    root_.children.clear();
    seq_matched_.clear();
    seq_node_.clear();
    hits_ = 0;
    misses_ = 0;
    clock_ = 0;
    next_cache_seq_id_ = CACHE_SEQ_ID_BASE;
}

}  // namespace forge
