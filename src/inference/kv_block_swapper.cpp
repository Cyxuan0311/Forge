#include "forge/kv_block_swapper.h"

#include <cstdlib>
#include <cstring>

#include "forge/logger.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

void KVBlockSwapper::HostBlob::reset() {
    // Memory is released via the swapper's free_host; this only resets state.
    k = nullptr;
    k_size = 0;
    v = nullptr;
    v_size = 0;
}

uint8_t* KVBlockSwapper::alloc_host(size_t bytes) {
#ifdef USE_CUDA
    static bool cuda_ok = [] {
        int dev = -1;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0)
            return true;
        return false;
    }();
    if (cuda_ok) {
        void* p = nullptr;
        if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess) {
            pinned_ = true;
            return static_cast<uint8_t*>(p);
        }
        // Fall through to plain host allocation if pinned alloc fails.
    }
#else
    (void)bytes;
#endif
    (void)bytes;
    return static_cast<uint8_t*>(std::malloc(bytes));
}

void KVBlockSwapper::free_host(uint8_t* p) {
    if (!p)
        return;
#ifdef USE_CUDA
    if (pinned_) {
        cudaFreeHost(p);
        return;
    }
#else
    (void)p;
#endif
    std::free(p);
}

KVBlockSwapper::~KVBlockSwapper() {
    for (auto& kv : pool_)
        kv.second.reset();
    for (auto& kv : pool_) {
        if (kv.second.k)
            free_host(kv.second.k);
        if (kv.second.v)
            free_host(kv.second.v);
    }
    pool_.clear();
    total_bytes_ = 0;
}

// Raw-byte copies that respect where the memory actually lives. A paged GPU
// page's bytes are device memory; std::memcpy on a device pointer faults (or,
// worse, reads whatever happens to be mapped there). This is the whole reason
// CopyKind exists.
namespace {

// Destination is the page's own (device) buffer: host blob -> device.
bool copy_host_to_dst(void* dst, const void* src, size_t bytes,
                      KVBlockSwapper::CopyKind dst_kind) {
#ifdef USE_CUDA
    if (dst_kind == KVBlockSwapper::CopyKind::Device)
        return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
#else
    (void)dst;
    (void)src;
    (void)bytes;
    (void)dst_kind;
#endif
    std::memcpy(dst, src, bytes);
    return true;
}

// Destination is the pinned host blob: device -> host.
bool copy_src_to_host(void* host_dst, const void* src, size_t bytes,
                      KVBlockSwapper::CopyKind src_kind) {
#ifdef USE_CUDA
    if (src_kind == KVBlockSwapper::CopyKind::Device)
        return cudaMemcpy(host_dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
#else
    (void)host_dst;
    (void)src;
    (void)bytes;
    (void)src_kind;
#endif
    std::memcpy(host_dst, src, bytes);
    return true;
}

}  // namespace

bool KVBlockSwapper::offload(int layer, int page_id, const void* k_src, size_t k_bytes,
                             const void* v_src, size_t v_bytes, CopyKind kind, bool force) {
    std::lock_guard<std::mutex> lock(mtx_);
    if ((!k_src && k_bytes) || (!v_src && v_bytes)) {
        LOG_ERROR("KVBlockSwapper::offload: null source with non-zero size");
        return false;
    }
    PageKey key{layer, page_id};
    auto it = pool_.find(key);
    if (it != pool_.end()) {
        // Slot already exists: reject size mismatch.
        if (it->second.k_size != k_bytes || it->second.v_size != v_bytes) {
            LOG_ERROR("KVBlockSwapper::offload: size mismatch for existing page (" +
                      std::to_string(layer) + "," + std::to_string(page_id) + ")");
            return false;
        }
        if (!copy_src_to_host(it->second.k, k_src, k_bytes, kind))
            return false;
        if (!copy_src_to_host(it->second.v, v_src, v_bytes, kind))
            return false;
        lru_.splice(lru_.end(), lru_, it->second.lru_it);  // refresh LRU
        return true;
    }

    const size_t blob_bytes = k_bytes + v_bytes;
    if (!force && max_bytes_ && total_bytes_ + blob_bytes > max_bytes_) {
        LOG_DEBUG("KVBlockSwapper::offload: host pool full (" + std::to_string(total_bytes_) + "/" +
                  std::to_string(max_bytes_) + " bytes), rejecting page (" + std::to_string(layer) +
                  "," + std::to_string(page_id) + ")");
        return false;
    }

    HostBlob blob;
    if (k_bytes) {
        blob.k = alloc_host(k_bytes);
        if (!blob.k) {
            LOG_ERROR("KVBlockSwapper::offload: host alloc failed (" + std::to_string(k_bytes) +
                      " bytes)");
            return false;
        }
        if (!copy_src_to_host(blob.k, k_src, k_bytes, kind)) {
            free_host(blob.k);
            return false;
        }
        blob.k_size = k_bytes;
    }
    if (v_bytes) {
        blob.v = alloc_host(v_bytes);
        if (!blob.v) {
            free_host(blob.k);
            LOG_ERROR("KVBlockSwapper::offload: host alloc failed (" + std::to_string(v_bytes) +
                      " bytes)");
            return false;
        }
        if (!copy_src_to_host(blob.v, v_src, v_bytes, kind)) {
            free_host(blob.v);
            free_host(blob.k);
            return false;
        }
        blob.v_size = v_bytes;
    }
    auto res = pool_.emplace(key, blob);
    lru_.push_back(key);
    res.first->second.lru_it = std::prev(lru_.end());
    total_bytes_ += blob_bytes;
    return true;
}

bool KVBlockSwapper::bring_back(int layer, int page_id, void* k_dst, size_t k_bytes, void* v_dst,
                                size_t v_bytes, CopyKind kind) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pool_.find(PageKey{layer, page_id});
    if (it == pool_.end())
        return false;
    const auto& blob = it->second;
    if (blob.k_size != k_bytes || blob.v_size != v_bytes)
        return false;
    if (k_bytes && (!k_dst || !blob.k))
        return false;
    if (v_bytes && (!v_dst || !blob.v))
        return false;
    if (!copy_host_to_dst(k_dst, blob.k, k_bytes, kind))
        return false;
    if (!copy_host_to_dst(v_dst, blob.v, v_bytes, kind))
        return false;
    lru_.splice(lru_.end(), lru_, it->second.lru_it);  // refresh LRU
    return true;
}

bool KVBlockSwapper::has(int layer, int page_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return pool_.find(PageKey{layer, page_id}) != pool_.end();
}

bool KVBlockSwapper::size(int layer, int page_id, size_t* k_bytes, size_t* v_bytes) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pool_.find(PageKey{layer, page_id});
    if (it == pool_.end())
        return false;
    if (k_bytes)
        *k_bytes = it->second.k_size;
    if (v_bytes)
        *v_bytes = it->second.v_size;
    return true;
}

bool KVBlockSwapper::peek(int layer, int page_id, const uint8_t** k, const uint8_t** v,
                          size_t* k_bytes, size_t* v_bytes) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pool_.find(PageKey{layer, page_id});
    if (it == pool_.end())
        return false;
    if (k)
        *k = it->second.k;
    if (v)
        *v = it->second.v;
    if (k_bytes)
        *k_bytes = it->second.k_size;
    if (v_bytes)
        *v_bytes = it->second.v_size;
    return true;
}

void KVBlockSwapper::drop(int layer, int page_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pool_.find(PageKey{layer, page_id});
    if (it == pool_.end())
        return;
    lru_.erase(it->second.lru_it);
    total_bytes_ -= (it->second.k_size + it->second.v_size);
    if (it->second.k)
        free_host(it->second.k);
    if (it->second.v)
        free_host(it->second.v);
    it->second.reset();
    pool_.erase(it);
}

void KVBlockSwapper::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& kv : pool_) {
        if (kv.second.k)
            free_host(kv.second.k);
        if (kv.second.v)
            free_host(kv.second.v);
        kv.second.reset();
    }
    pool_.clear();
    lru_.clear();
    total_bytes_ = 0;
    pinned_ = false;
}

bool KVBlockSwapper::touch(int layer, int page_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pool_.find(PageKey{layer, page_id});
    if (it == pool_.end())
        return false;
    lru_.splice(lru_.end(), lru_, it->second.lru_it);
    return true;
}

bool KVBlockSwapper::lru_oldest(int* layer, int* page_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (lru_.empty())
        return false;
    const PageKey& oldest = lru_.front();
    if (layer)
        *layer = oldest.first;
    if (page_id)
        *page_id = oldest.second;
    return true;
}

}  // namespace forge
