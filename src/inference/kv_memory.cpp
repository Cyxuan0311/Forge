#include "forge/kv_memory.h"

#include <cstring>

#include "forge/logger.h"
#include "forge/inference/tensor_device_utils.h"

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
                            DeviceType device, const KVCacheTypeConfig& kv_config, int page_size,
                            int max_num_seqs) {
    return storage_->init(num_layers, kv_dims, max_seq_len, device, kv_config, page_size,
                          max_num_seqs);
}

bool KVMemory::write_kv(int layer, int seq_id, int64_t pos, int seq_len, const TensorPtr& k,
                        const TensorPtr& v) {
    if (is_paged()) {
        // Paged path: write into the sequence's pages. CUDA-backed storage takes
        // device pointers directly (no D2H); CPU-backed storage needs the data on
        // the host, so stage it there if it currently lives on the device.
        KVStorage& st = storage();
        TensorPtr kc = k, vc = v;
        if (st.device() == DeviceType::CPU && kc->device() == DeviceType::CUDA) {
            kc = ensure_cpu(k);
            vc = ensure_cpu(v);
        }
        const float* kd = static_cast<const float*>(kc->data());
        const float* vd = static_cast<const float*>(vc->data());
        return st.write_kv_seq(layer, seq_id, pos, seq_len, kd, vd);
    }
    // Contiguous path: write to the KVCache directly (handles device placement
    // internally). Returns the new filled length, or < 0 on overflow/invalid.
    return cache().update(layer, seq_id, pos, k, v, seq_len) >= 0;
}

void KVMemory::set_layer_policies(const std::vector<KVLayerPolicy>& policies, int swa_window) {
    if (storage_->is_paged()) {
        auto* paged = dynamic_cast<PagedKVStorage*>(storage_.get());
        if (paged)
            paged->set_layer_policies(policies, swa_window);
    }
}

bool KVMemory::resize_storage() {
    if (storage_->is_paged()) {
        auto* paged = dynamic_cast<PagedKVStorage*>(storage_.get());
        if (paged)
            return paged->resize_to_budget();
    }
    return true;
}

int KVMemory::max_pool_tokens() const {
    if (storage_->is_paged()) {
        auto* paged = dynamic_cast<const PagedKVStorage*>(storage_.get());
        if (paged)
            return paged->max_pool_tokens();
    }
    return 0;
}

int KVMemory::free_pages_full() const {
    if (storage_->is_paged()) {
        auto* paged = dynamic_cast<const PagedKVStorage*>(storage_.get());
        if (paged)
            return paged->free_pages_full();
    }
    return -1;
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

void KVMemory::defrag() {
    storage_->defrag();
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
