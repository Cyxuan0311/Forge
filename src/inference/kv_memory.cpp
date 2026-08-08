#include "forge/kv_memory.h"

#include <cstring>

#include "forge/logger.h"

namespace forge {

// =========================================================================
// KVMemory
// =========================================================================

KVMemory::KVMemory(KVCache& cache)
    : cache_(cache), storage_(std::make_unique<ContiguousKVStorage>(cache)) {}

KVMemory::KVMemory(KVCache& cache, KVStorageMode mode) : cache_(cache) {
    if (mode == KVStorageMode::Paged) {
        storage_ = std::make_unique<PagedKVStorage>();
    } else {
        storage_ = std::make_unique<ContiguousKVStorage>(cache);
    }
}

bool KVMemory::init_storage(int num_layers, const std::vector<int>& kv_dims, int max_seq_len,
                             DeviceType device, const KVCacheTypeConfig& kv_config,
                             int page_size, int max_num_seqs) {
    return storage_->init(num_layers, kv_dims, max_seq_len, device, kv_config,
                          page_size, max_num_seqs);
}

void KVMemory::set_layer_policies(const std::vector<KVLayerPolicy>& policies, int swa_window) {
    if (storage_->is_paged()) {
        auto* paged = dynamic_cast<PagedKVStorage*>(storage_.get());
        if (paged) paged->set_layer_policies(policies, swa_window);
    }
}

KVReadView KVMemory::view(int layer, int seq_id) const {
    KVReadView v;
    v.key = storage_->read_key(layer);
    v.value = storage_->read_value(layer);
    v.kv_len = storage_->filled(layer);
    v.n_tokens = storage_->seq_filled(layer, seq_id);
    v.contiguous = !storage_->is_paged();
    return v;
}

bool KVMemory::seq_remove(int seq_id, int64_t p0, int64_t p1) {
    return storage_->seq_remove(seq_id, p0, p1);
}

bool KVMemory::seq_share(int src_seq, int dst_seq, int64_t p0, int64_t p1) {
    return storage_->seq_share(src_seq, dst_seq, p0, p1);
}

bool KVMemory::seq_keep(int seq_id) {
    return storage_->seq_keep(seq_id);
}

void KVMemory::release(int seq_id) {
    storage_->release(seq_id);
}

bool KVMemory::release_sequence(int seq_id) {
    release(seq_id);
    return true;
}

// FNV-1a 64-bit hash (same as RequestScheduler's hash_prompt)
size_t KVMemory::hash_tokens(const std::vector<int32_t>& tokens) {
    size_t h = 14695981039346656037ULL;
    for (int32_t t : tokens) {
        h ^= static_cast<size_t>(t);
        h *= 1099511628211ULL;
    }
    return h;
}

int KVMemory::max_seq_len() const {
    return storage_->max_seq_len();
}

int KVMemory::num_layers() const {
    return storage_->num_layers();
}

size_t KVMemory::nbytes() const {
    return storage_->nbytes();
}

size_t KVMemory::active_bytes() const {
    return storage_->active_bytes();
}

int KVMemory::num_free_slots() const {
    return storage_->num_free_slots();
}

}  // namespace forge
