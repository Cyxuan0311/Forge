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

bool KVBlockSwapper::offload(int layer, int page_id, const void* k_src, size_t k_bytes,
                             const void* v_src, size_t v_bytes) {
    if ((!k_src && k_bytes) || (!v_src && v_bytes)) {
        LOG_ERROR("KVBlockSwapper::offload: null source with non-zero size");
        return false;
    }
    PageKey key{layer, page_id};
    auto it = pool_.find(key);
    if (it != pool_.end()) {
        // Slot already exists (grow-only): reject size mismatch.
        if (it->second.k_size != k_bytes || it->second.v_size != v_bytes) {
            LOG_ERROR("KVBlockSwapper::offload: size mismatch for existing page (" +
                      std::to_string(layer) + "," + std::to_string(page_id) + ")");
            return false;
        }
        if (k_bytes)
            std::memcpy(it->second.k, k_src, k_bytes);
        if (v_bytes)
            std::memcpy(it->second.v, v_src, v_bytes);
        return true;
    }

    HostBlob blob;
    if (k_bytes) {
        blob.k = alloc_host(k_bytes);
        if (!blob.k) {
            LOG_ERROR("KVBlockSwapper::offload: host alloc failed (" + std::to_string(k_bytes) +
                      " bytes)");
            return false;
        }
        std::memcpy(blob.k, k_src, k_bytes);
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
        std::memcpy(blob.v, v_src, v_bytes);
        blob.v_size = v_bytes;
    }
    pool_.emplace(key, blob);
    total_bytes_ += k_bytes + v_bytes;
    return true;
}

bool KVBlockSwapper::bring_back(int layer, int page_id, void* k_dst, size_t k_bytes, void* v_dst,
                                size_t v_bytes) const {
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
    if (k_bytes)
        std::memcpy(k_dst, blob.k, k_bytes);
    if (v_bytes)
        std::memcpy(v_dst, blob.v, v_bytes);
    return true;
}

bool KVBlockSwapper::has(int layer, int page_id) const {
    return pool_.find(PageKey{layer, page_id}) != pool_.end();
}

bool KVBlockSwapper::size(int layer, int page_id, size_t* k_bytes, size_t* v_bytes) const {
    auto it = pool_.find(PageKey{layer, page_id});
    if (it == pool_.end())
        return false;
    if (k_bytes)
        *k_bytes = it->second.k_size;
    if (v_bytes)
        *v_bytes = it->second.v_size;
    return true;
}

void KVBlockSwapper::drop(int layer, int page_id) {
    auto it = pool_.find(PageKey{layer, page_id});
    if (it == pool_.end())
        return;
    total_bytes_ -= (it->second.k_size + it->second.v_size);
    if (it->second.k)
        free_host(it->second.k);
    if (it->second.v)
        free_host(it->second.v);
    it->second.reset();
    pool_.erase(it);
}

void KVBlockSwapper::reset() {
    for (auto& kv : pool_) {
        if (kv.second.k)
            free_host(kv.second.k);
        if (kv.second.v)
            free_host(kv.second.v);
        kv.second.reset();
    }
    pool_.clear();
    total_bytes_ = 0;
    pinned_ = false;
}

}  // namespace forge
