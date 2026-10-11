#include "forge/prefix_cache.h"

#include <cstdio>
#include <algorithm>
#include <chrono>
#include <climits>
#include <utility>

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
    int matched = 0;  // cumulative token length matched from the root

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
                matched += seg_pages * page_size;
                deepest = child;
                break;
            } else if (seg_pages < avail_pages) {
                matched += seg_pages * page_size;
                deepest = child;  // last fully-matched interior node (kept if descent stops)
                pos += seg_pages * page_size;
                cur = child;
            } else {  // request is a prefix of child's segment
                matched += avail_pages * page_size;
                deepest = child;
                break;
            }
        } else {
            // Divergence at page i. Pages [0, i) are a fully shared, page-aligned
            // prefix; page i differs (its KV depends on the differing token) so it
            // must NOT be shared.
            if (i > 0) {
                matched += i * page_size;
                deepest = child;
            }
            break;
        }
    }

    // The deepest node's owner holds KV pages [0, node->cumulative_len), which
    // subsumes every ancestor's prefix, so the cumulative matched length is the
    // correct amount to share (NOT just the deepest node's own segment).
    matched_len = matched;
    return deepest;
}

int PrefixCache::try_lookup(const std::vector<int32_t>& tokens, int dst_seq_id, KVStorage& storage,
                            PrefixHitKind* kind) {
    if (kind)
        *kind = PrefixHitKind::Miss;

    std::unique_lock<std::mutex> lock(mu_);
    harvest_async(storage, false);

    int page_size = storage.page_size();
    if (page_size <= 0) {
        misses_++;
        return 0;
    }

    int matched_len = 0;
    RadixNode* node = lookup_deepest(&root_, tokens, page_size, matched_len);
    if (!node || matched_len < page_size) {
        misses_++;
        return 0;
    }

    if (node->restore_pending) {
        // A prefetched restore is still running: wait a bounded amount, then
        // give up and let the request prefill from scratch. Never touch a node
        // that another thread is rebuilding.
        if (!wait_for_restore(lock, node)) {
            LOG_DEBUG("PrefixCache: prefetched restore still pending; serving as miss");
            misses_++;
            return 0;
        }
    }
    if (node->device_resident && node->backup_pending) {
        // Backup in flight: the device pages are still alive and complete, so
        // the hit is served normally (it just keeps the pages a bit longer).
    }

    const bool from_host = !node->device_resident;
    if (from_host) {
        // Host-resident match: rebuild the node's device pages from the blob
        // pool (plus shared pages from the nearest device-resident ancestor),
        // then serve the hit from the rebuilt owner like a device hit. On
        // failure fall back to a full miss (the request prefills everything).
        if (!restore_node(node, storage)) {
            LOG_WARN("PrefixCache: host restore failed for prefix len " +
                     std::to_string(matched_len) + "; falling back to full prefill");
            misses_++;
            return 0;
        }
    }

    // Share [0, matched_len) from the deepest matched node's owner in ONE call.
    bool shared = storage.seq_share(node->owner_seq_id, dst_seq_id, 0, matched_len);
    if (!shared) {
        misses_++;
        return 0;
    }

    node->ref_count++;
    node->last_used = tick();
    seq_matched_[dst_seq_id] = matched_len;
    seq_node_[dst_seq_id] = node;

    // hits_ counts every served prefix (device or host); hits_host_ is the
    // subset that required a restore from the host blob pool.
    if (kind)
        *kind = from_host ? PrefixHitKind::Host : PrefixHitKind::Device;
    hits_++;
    if (from_host)
        hits_host_++;
    return matched_len;
}

// =========================================================================
// Two-tier residency: backup / restore
// =========================================================================

bool PrefixCache::has_device_resident_descendant(const RadixNode* node) const {
    for (const auto& c : node->children) {
        if (!c->valid)
            continue;
        if (c->device_resident)
            return true;
        if (has_device_resident_descendant(c.get()))
            return true;
    }
    return false;
}

size_t PrefixCache::count_device_resident(const RadixNode* cur) const {
    size_t n = 0;
    for (const auto& c : cur->children) {
        if (!c->valid)
            continue;
        if (c->device_resident)
            n += 1;
        n += count_device_resident(c.get());
    }
    return n;
}

bool PrefixCache::backup_node(RadixNode* node, KVStorage& storage) {
    if (!node->valid || !node->device_resident || node->ref_count != 0)
        return false;
    if (node->owner_seq_id < 0)
        return false;

    const int page_size = storage.page_size();
    const int seg_pages = static_cast<int>(node->seg.size()) / page_size;
    if (seg_pages <= 0)
        return false;
    // Device-resident descendants keep referencing this node's seg pages, so
    // the pages would stay alive after the release — a pointless copy. The
    // caller (device-tier eviction) rescans after each victim.
    if (has_device_resident_descendant(node))
        return false;

    const int seg_start_page = (node->cumulative_len - static_cast<int>(node->seg.size())) / page_size;

    std::vector<std::vector<int32_t>> keys;
    if (node->host_resident) {
        // Re-backup (e.g. the blob was dropped by an upper tier): overwrite the
        // existing ids in place — same node, same page geometry, same sizes.
        keys = node->host_blob_keys;
        if (static_cast<int>(keys.size()) != storage.num_layers())
            keys.assign(storage.num_layers(), std::vector<int32_t>(seg_pages, -1));
    } else {
        keys.assign(storage.num_layers(), std::vector<int32_t>(seg_pages, -1));
    }

    if (!storage.backup_prefix_pages(node->owner_seq_id, seg_start_page,
                                     seg_start_page + seg_pages, &keys)) {
        // e.g. host pool over budget: the node stays device-resident and the
        // caller falls back to a hard eviction.
        return false;
    }

    node->host_blob_keys = std::move(keys);
    node->host_resident = true;
    return true;
}

bool PrefixCache::backup_node_async(RadixNode* node, KVStorage& storage) {
    if (!node->valid || !node->device_resident || node->ref_count != 0)
        return false;
    if (node->owner_seq_id < 0)
        return false;

    const int page_size = storage.page_size();
    const int seg_pages = static_cast<int>(node->seg.size()) / page_size;
    if (seg_pages <= 0)
        return false;
    if (has_device_resident_descendant(node))
        return false;

    const int seg_start_page = (node->cumulative_len - static_cast<int>(node->seg.size())) / page_size;

    std::vector<std::vector<int32_t>> keys;
    if (node->host_resident && static_cast<int>(node->host_blob_keys.size()) == storage.num_layers()) {
        keys = node->host_blob_keys;
    } else {
        keys.assign(storage.num_layers(), std::vector<int32_t>(seg_pages, -1));
    }

    if (!start_worker())
        return false;  // caller falls back to the synchronous path

    node->backup_pending = true;
    auto job = std::make_unique<AsyncJob>();
    job->kind = AsyncJob::Kind::Backup;
    job->node = node;
    job->seq_id = node->owner_seq_id;
    job->p0 = seg_start_page;
    job->p1 = seg_start_page + seg_pages;
    job->keys = std::move(keys);
    enqueue(std::move(job), storage);
    return true;
}

bool PrefixCache::restore_node(RadixNode* node, KVStorage& storage) {
    if (node->device_resident)
        return true;  // nothing to do
    if (!node->host_resident)
        return false;  // nothing to restore from

    const int page_size = storage.page_size();

    // Nearest device-resident ancestor (or the root sentinel, which holds no
    // pages and starts the rebuilt sequence empty).
    RadixNode* anc = node->parent;
    while (anc && anc != &root_ && !anc->device_resident)
        anc = anc->parent;

    // Path nodes strictly below `anc`, node-first. Every one of them must be
    // host-resident: a device-resident node below `anc` would contradict the
    // "nearest ancestor" walk above.
    std::vector<RadixNode*> path;
    bool chain_ok = true;
    for (RadixNode* n = node; n && n != anc; n = n->parent) {
        if (n == &root_ || !n->host_resident) {
            chain_ok = false;
            break;
        }
        path.push_back(n);
    }
    if (!chain_ok || path.empty())
        return false;

    const int new_owner = next_cache_seq_id_++;
    bool ok = true;
    if (anc != &root_ && anc->cumulative_len > 0) {
        ok = storage.seq_share(anc->owner_seq_id, new_owner, 0, anc->cumulative_len);
        if (!ok)
            LOG_WARN("PrefixCache: restore seq_share from ancestor failed");
    }
    if (ok) {
        // Root-first: each seg restore appends its pages after the previous
        // range, rebuilding [anc.cumulative_len, node->cumulative_len).
        for (auto it = path.rbegin(); it != path.rend() && ok; ++it) {
            RadixNode* h = *it;
            const int seg_start_page =
                (h->cumulative_len - static_cast<int>(h->seg.size())) / page_size;
            ok = storage.restore_prefix_pages(new_owner, seg_start_page, h->host_blob_keys);
            if (!ok)
                LOG_WARN("PrefixCache: restore_prefix_pages failed for seg at page " +
                         std::to_string(seg_start_page));
        }
    }
    if (!ok) {
        storage.release(new_owner);  // release() on a never-created seq is a no-op
        return false;
    }

    // Re-establish the classic invariant along the path: every node strictly
    // between `anc` and `node` shares the rebuilt prefix up to its own
    // cumulative length (page-table sharing only — no data copies), so the
    // whole matched chain becomes device-resident again.
    for (size_t k = 1; k < path.size(); ++k) {
        RadixNode* h = path[k];
        const int h_owner = next_cache_seq_id_++;
        if (!storage.seq_share(new_owner, h_owner, 0, h->cumulative_len)) {
            LOG_WARN("PrefixCache: restore path promotion failed; continuing");
            break;
        }
        if (h->owner_seq_id >= 0)
            storage.release(h->owner_seq_id);  // host-only owner: already empty
        h->owner_seq_id = h_owner;
        h->device_resident = true;
        // Any L3 copy was fetched back into the pool by restore_prefix_pages.
        h->blob_in_l3 = false;
    }

    if (node->owner_seq_id >= 0)
        storage.release(node->owner_seq_id);  // host-only owner: already empty
    node->owner_seq_id = new_owner;
    node->device_resident = true;
    node->blob_in_l3 = false;
    return true;
}

// =========================================================================
// Phase 2: asynchronous backup / restore worker
// =========================================================================

PrefixCache::~PrefixCache() {
    shutdown_worker();
}

void PrefixCache::clear_pending_flags(RadixNode* cur) {
    for (auto& c : cur->children) {
        if (!c->valid)
            continue;
        c->backup_pending = false;
        c->restore_pending = false;
        clear_pending_flags(c.get());
    }
}

bool PrefixCache::start_worker() {
    if (worker_.joinable())
        return true;
    try {
        worker_ = std::thread([this] { worker_loop(); });
    } catch (const std::exception&) {
        return false;  // fall back to synchronous execution
    }
    return true;
}

void PrefixCache::enqueue(std::unique_ptr<AsyncJob> job, KVStorage& storage) {
    job->storage = &storage;
    {
        std::lock_guard<std::mutex> jl(jobs_mu_);
        jobs_.push_back(std::move(job));
        ++pending_;
    }
    jobs_cv_.notify_one();
}

void PrefixCache::worker_loop() {
    for (;;) {
        std::unique_ptr<AsyncJob> job;
        {
            std::unique_lock<std::mutex> jl(jobs_mu_);
            jobs_cv_.wait(jl, [this] { return !jobs_.empty() || worker_stop_; });
            if (jobs_.empty())
                return;  // stopping
            job = std::move(jobs_.front());
            jobs_.pop_front();
        }

        if (!job->storage)
            continue;

        if (job->kind == AsyncJob::Kind::Backup) {
            // The node keeps its device pages for the whole copy: nothing can
            // read a half-written blob, and the pages stay valid because the
            // owner seq is only released at harvest time.
            std::vector<std::vector<int32_t>> keys = job->keys;
            job->ok = job->storage->backup_prefix_pages(job->seq_id, job->p0, job->p1, &keys);
            job->keys = std::move(keys);
        } else {
            // Restore mutates the tree (new owner seq, residency flags), so it
            // runs under mu_ — the owner thread blocks instead of seeing a
            // half-rebuilt chain.
            std::lock_guard<std::mutex> lock(mu_);
            job->ok = restore_node(job->node, *job->storage);
            job->node->restore_pending = false;
            tree_cv_.notify_all();
        }

        {
            std::lock_guard<std::mutex> jl(jobs_mu_);
            done_.push_back(std::move(job));
        }
        jobs_cv_.notify_all();
    }
}

bool PrefixCache::wait_for_restore(std::unique_lock<std::mutex>& lock, RadixNode* node) {
    if (restore_wait_ms_ <= 0)
        return !node->restore_pending;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(restore_wait_ms_);
    while (node->restore_pending) {
        if (tree_cv_.wait_until(lock, deadline) == std::cv_status::timeout)
            return !node->restore_pending;
    }
    return true;
}

void PrefixCache::harvest_async(KVStorage& storage, bool wait_all) {
    if (wait_all) {
        // Only Backup jobs can be waited on from here: a Restore job needs mu_
        // (already held by the caller) and would deadlock. Restores are waited
        // on explicitly, at lookup time, with mu_ released.
        std::unique_lock<std::mutex> jl(jobs_mu_);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
        jobs_cv_.wait_until(jl, deadline, [this] { return pending_ <= 0; });
    }

    for (;;) {
        std::unique_ptr<AsyncJob> job;
        {
            std::lock_guard<std::mutex> jl(jobs_mu_);
            if (done_.empty())
                break;
            job = std::move(done_.front());
            done_.pop_front();
            --pending_;
        }
        if (!job->node || !job->node->valid)
            continue;

        if (job->kind == AsyncJob::Kind::Backup) {
            RadixNode* node = job->node;
            node->backup_pending = false;
            if (job->ok) {
                node->host_blob_keys = std::move(job->keys);
                node->host_resident = true;
                node->device_resident = false;
                if (node->owner_seq_id >= 0)
                    storage.release(node->owner_seq_id);
                async_backups_++;
            }
            // A failed backup leaves the node device-resident; a later
            // eviction pass retries (or hard-evicts it).
        } else {
            if (job->ok)
                async_restores_++;
        }
    }
}

void PrefixCache::shutdown_worker() {
    {
        std::lock_guard<std::mutex> jl(jobs_mu_);
        worker_stop_ = true;
        jobs_.clear();
        done_.clear();
        pending_ = 0;
    }
    jobs_cv_.notify_all();
    if (worker_.joinable())
        worker_.join();

    // Queued-but-dropped jobs never cleared their node flags; do it here so no
    // node stays permanently unevictable.
    clear_pending_flags(&root_);
}

void PrefixCache::prefetch_prefix(const std::vector<int32_t>& tokens, KVStorage& storage) {
    if (!prefetch_ || policy_ == KVWritePolicy::Off)
        return;

    std::unique_lock<std::mutex> lock(mu_);
    harvest_async(storage, false);

    const int page_size = storage.page_size();
    if (page_size <= 0)
        return;

    int matched_len = 0;
    RadixNode* node = lookup_deepest(&root_, tokens, page_size, matched_len);
    if (!node || matched_len < page_size)
        return;
    if (node->device_resident || node->restore_pending)
        return;
    // The chain is rebuilt as a unit: skip if any link is already in flight.
    for (RadixNode* n = node; n && n != &root_; n = n->parent) {
        if (n->restore_pending || n->backup_pending)
            return;
    }

    node->restore_pending = true;
    auto job = std::make_unique<AsyncJob>();
    job->kind = AsyncJob::Kind::Restore;
    job->node = node;
    if (start_worker()) {
        enqueue(std::move(job), storage);
    } else {
        node->restore_pending = false;
    }
}

// =========================================================================
// Insert / register
// =========================================================================

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
    //
    // The split shares pages from child->owner, so a host-resident child (its
    // owner was emptied by the backup+release) must be restored first. On
    // restore failure the split is skipped and the prompt is simply not
    // registered beyond `child` — a graceful degradation, never a corruption.
    if (!child->device_resident && !restore_node(child, storage)) {
        LOG_WARN("PrefixCache: host restore before split failed; skipping registration");
        return child;
    }
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

    // Blob keys are seg-relative but keyed by absolute page content, so the
    // split just slices them: P inherits the prefix pages, child keeps the
    // suffix pages. Both stay valid (seg content is immutable).
    if (child->host_resident) {
        const int pb_pages = page_boundary / page_size;
        const int layers = static_cast<int>(child->host_blob_keys.size());
        pnode->host_resident = true;
        pnode->blob_in_l3 = child->blob_in_l3;  // tier of the inherited blobs
        pnode->host_blob_keys.resize(layers);
        for (int l = 0; l < layers; ++l) {
            auto& ck = child->host_blob_keys[l];
            if (static_cast<int>(ck.size()) >= pb_pages) {
                pnode->host_blob_keys[l].assign(ck.begin(), ck.begin() + pb_pages);
                ck.erase(ck.begin(), ck.begin() + pb_pages);
            }
        }
    }

    pnode->children.push_back(std::move(c_owned));

    RadixNode* praw = pnode.get();
    cur->children.push_back(std::move(pnode));

    return insert_prefix(praw, tokens, src_seq_id, storage, pos + page_boundary, total);
}

void PrefixCache::register_prefix(const std::vector<int32_t>& tokens, int src_seq_id,
                                  KVStorage& storage) {
    std::unique_lock<std::mutex> lock(mu_);
    harvest_async(storage, false);

    int page_size = storage.page_size();
    if (page_size <= 0)
        return;

    int total = static_cast<int>(tokens.size() / page_size) * page_size;
    if (total < page_size)
        return;

    RadixNode* deepest = insert_prefix(&root_, tokens, src_seq_id, storage, 0, total);

    // Release the source request's prefix pages; they are now owned by the
    // per-node cache seq_ids (page references are shared, KV is not copied).
    storage.seq_remove(src_seq_id, 0, total);

    // WriteThrough: mirror every newly registered node into the prefix blob
    // pool right away, so a later eviction only releases the device pages.
    // Walk deepest-first; already-mirrored ancestors are skipped by
    // backup_node's host_resident state.
    if (policy_ == KVWritePolicy::WriteThrough) {
        for (RadixNode* n = deepest; n && n != &root_; n = n->parent) {
            if (n->valid && n->device_resident && !n->host_resident)
                backup_node(n, storage);
        }
    }
}

void PrefixCache::release_prefix(int seq_id, KVStorage& storage) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = seq_matched_.find(seq_id);
    if (it == seq_matched_.end())
        return;

    int len = it->second;
    auto nit = seq_node_.find(seq_id);
    if (nit != seq_node_.end() && nit->second && nit->second->ref_count > 0)
        nit->second->ref_count--;

    seq_matched_.erase(it);
    seq_node_.erase(seq_id);

    if (len > 0)
        storage.seq_remove(seq_id, 0, len);
}

// =========================================================================
// Eviction
// =========================================================================

void PrefixCache::detach_node(RadixNode* node) {
    RadixNode* parent = node->parent;
    if (!parent)
        return;
    for (auto it = parent->children.begin(); it != parent->children.end(); ++it) {
        if (it->get() == node) {
            parent->children.erase(it);
            break;
        }
    }
}

void PrefixCache::drop_host_blobs_recursive(RadixNode* node, KVStorage& storage) {
    for (auto& c : node->children)
        drop_host_blobs_recursive(c.get(), storage);
    if (node->host_resident) {
        for (int l = 0; l < static_cast<int>(node->host_blob_keys.size()); ++l) {
            for (int32_t id : node->host_blob_keys[l]) {
                if (id >= 0)
                    storage.drop_prefix_blob(l, id);
            }
        }
        node->host_resident = false;
        node->host_blob_keys.clear();
    }
}

void PrefixCache::mark_blobs_in_l3(RadixNode* node) {
    for (auto& c : node->children) {
        if (c->valid)
            mark_blobs_in_l3(c.get());
    }
    if (node->host_resident && !node->host_blob_keys.empty())
        node->blob_in_l3 = true;
}

bool PrefixCache::collect_host_evictable(RadixNode* cur, RadixNode** best) {
    // A node whose blobs live in L3 costs no host bytes, so removing it cannot
    // free anything — it is left in place (and still restorable).
    if (cur != &root_ &&
        (!cur->valid || cur->device_resident || cur->ref_count != 0 || cur->backup_pending ||
         cur->restore_pending || cur->blob_in_l3))
        return false;
    bool clean = true;
    for (auto& c : cur->children) {
        if (!collect_host_evictable(c.get(), best))
            clean = false;
    }
    // Only a host-resident node's own blobs are counted; a fully-clean subtree
    // may be destroyed wholesale, dropping every blob inside it.
    if (clean && cur->host_resident) {
        if (!*best || cur->last_used < (*best)->last_used)
            *best = cur;
    }
    return clean;
}

void PrefixCache::evict_host_over_budget(KVStorage& storage) {
    const size_t budget = storage.prefix_pool_max_bytes();
    if (budget == 0)
        return;  // grow-only host pool (no cap configured)

    int guard = 0;
    while (storage.prefix_pool_bytes() > budget && guard++ < 100000) {
        RadixNode* victim = nullptr;
        collect_host_evictable(&root_, &victim);
        if (!victim)
            break;  // nothing removable: pool may stay over budget until
                    // device-tier eviction hostifies more subtrees

        // Phase 3: prefer spilling the victim's blobs to L3 over destroying the
        // node — the prefix stays reusable, it just costs host memory no more.
        if (storage.l3_enabled() && storage.spill_prefix_blobs(victim->host_blob_keys)) {
            mark_blobs_in_l3(victim);
            continue;
        }
        drop_host_blobs_recursive(victim, storage);
        detach_node(victim);
    }
}

void PrefixCache::collect_device_evictable(const RadixNode* cur, RadixNode** best,
                                           uint64_t* oldest) const {
    for (const auto& c : cur->children) {
        if (!c->valid)
            continue;
        if (c->device_resident && c->ref_count == 0 && !c->backup_pending && !c->restore_pending &&
            !has_device_resident_descendant(c.get()) && c->last_used < *oldest) {
            *oldest = c->last_used;
            *best = c.get();
        }
        collect_device_evictable(c.get(), best, oldest);
    }
}

void PrefixCache::evict_lru_legacy(size_t max_entries, KVStorage& storage) {
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

        drop_host_blobs_recursive(victim, storage);
        if (victim->owner_seq_id >= 0)
            storage.release(victim->owner_seq_id);
        detach_node(victim);
        // victim is destroyed via its unique_ptr
    }
}

void PrefixCache::evict_lru(size_t max_entries, KVStorage& storage) {
    std::lock_guard<std::mutex> lock(mu_);

    // Commit whatever the worker finished since the last pass.
    harvest_async(storage, false);

    // Host tier first: keep the blob pool within its byte budget.
    evict_host_over_budget(storage);

    if (policy_ == KVWritePolicy::Off) {
        evict_lru_legacy(max_entries, storage);
        return;
    }

    // Device tier: shrink the device-resident population to max_entries AND keep
    // the paged page pool out of pressure. The entry cap alone is not enough:
    // on a real model each cached prefix holds many KV pages, so capping by
    // entry count can still exhaust the shared page pool and corrupt later
    // requests. We therefore also evict device-resident nodes whenever the
    // per-layer free-page headroom drops below a watermark, releasing their
    // device pages back to the allocator.
    size_t device_count = count_device_resident(&root_);
    // Page-pressure watermark: keep enough free pages for the next request so the
    // shared paged pool never exhausts. We can only reach base-class accessors
    // here (KVStorage&), so derive headroom from layer_pool_max_pages() (per-layer
    // caps) and num_free_pages() (aggregate) — uniform across layers in practice.
    std::vector<int> lmax = storage.layer_pool_max_pages();
    long long total_max = 0;
    for (int m : lmax)
        total_max += m;
    long long low_wm = total_max / 8;  // ~12.5% headroom
    if (low_wm < 64)
        low_wm = 64;
    int guard = 0;
    while (guard++ < 100000) {
        if (device_count <= max_entries &&
            static_cast<long long>(storage.num_free_pages()) >= low_wm)
            break;
        RadixNode* victim = nullptr;
        uint64_t oldest = UINT64_MAX;
        collect_device_evictable(&root_, &victim, &oldest);
        if (!victim) {
            // Everything left is pinned, or mid-backup. Commit finished jobs
            // and try once more before giving up.
            harvest_async(storage, false);
            collect_device_evictable(&root_, &victim, &oldest);
            if (!victim)
                break;
        }

        if (async_backup_) {
            // Phase 2: queue the copy and count the node as "on its way out".
            // Its device pages survive until the job is harvested, so a live
            // reader can never observe a torn page.
            if (backup_node_async(victim, storage)) {
                --device_count;
                continue;
            }
        }

        bool ok = true;
        if (!victim->host_resident)
            ok = backup_node(victim, storage);
        if (ok) {
            // Host copy secured (or already present): drop the device copy.
            // The node survives as host-resident and can be restored later.
            if (victim->owner_seq_id >= 0)
                storage.release(victim->owner_seq_id);
            victim->device_resident = false;
            --device_count;
        } else {
            // Backup failed (host pool over budget / IO error): fall back to
            // the legacy hard eviction so eviction still frees device memory.
            drop_host_blobs_recursive(victim, storage);
            if (victim->owner_seq_id >= 0)
                storage.release(victim->owner_seq_id);
            detach_node(victim);
        }
    }

    // Allocation-pressure callers (max_entries == 0) need the pages freed
    // before they return, so in-flight backups are waited on here.
    if (max_entries == 0)
        harvest_async(storage, true);
}

// =========================================================================
// Reset
// =========================================================================

void PrefixCache::clear(KVStorage* storage) {
    // Join the worker before touching the tree: a queued/ running job may hold
    // node pointers (and mu_, for restores).
    shutdown_worker();

    std::lock_guard<std::mutex> lock(mu_);
    if (storage)
        drop_host_blobs_recursive(&root_, *storage);
    root_.children.clear();
    seq_matched_.clear();
    seq_node_.clear();
    hits_ = 0;
    hits_host_ = 0;
    misses_ = 0;
    clock_ = 0;
    next_cache_seq_id_ = CACHE_SEQ_ID_BASE;
}

}  // namespace forge
