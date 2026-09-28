#include "forge/cuda_mem_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {
namespace cuda_mem {

namespace {

// All buffers we hand out are at least this aligned (cudaMalloc guarantees
// 256-byte alignment, so cached pointers preserve it).
constexpr size_t kAlignment = 256;

// Cap on bytes retained in the cache. Decode's working set is a few MB, so
// this only kicks in if the allocation pattern grows (e.g. large prefill).
constexpr size_t kMaxCachedBytes = 256ull * 1024 * 1024;

// Blocks larger than this are never cached (returned straight to the driver).
// Decode working-set buffers are small; this keeps one-time prefill/model-load
// allocations from pinning scarce device memory.
constexpr size_t kMaxCachedBlockBytes = 64ull * 1024 * 1024;

struct Pool {
    std::mutex mutex;

    // Free buffers, bucketed by aligned size.
    std::unordered_map<size_t, std::vector<void*>> free_list;

    // Authoritative pointer -> aligned bucket size for every buffer we know
    // about (both handed-out and cached), so deallocate() is self-describing.
    std::unordered_map<void*, size_t> sizes;

    size_t cached = 0;   // bytes currently sitting in the free list
    size_t allocs = 0;   // total allocate() calls
    size_t hits = 0;     // allocate() calls served from the free list

    static size_t align_up(size_t bytes) {
        return (bytes + kAlignment - 1) & ~(kAlignment - 1);
    }
};

Pool& pool() {
    static Pool p;
    return p;
}

}  // namespace

void* allocate(size_t bytes) {
#ifdef USE_CUDA
    if (bytes == 0)
        return nullptr;
    Pool& p = pool();
    const size_t aligned = Pool::align_up(bytes);

    {
        std::lock_guard<std::mutex> lock(p.mutex);
        p.allocs++;
        auto it = p.free_list.find(aligned);
        if (it != p.free_list.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            p.cached -= aligned;
            p.hits++;
            return ptr;
        }
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, aligned);
    if (err != cudaSuccess) {
        // Driver may be low on memory: drop the cache and retry once.
        clear_cache();
        err = cudaMalloc(&ptr, aligned);
    }
    // WSL2 GPU-PV transient: the host may refuse make_resident (-ENOMEM) for
    // seconds at a time while Windows reclaims backing pages. These failures
    // clear on their own, so retry with a bounded backoff instead of failing
    // the (possibly minutes-long) model load on the first blip.
    if (err != cudaSuccess) {
        for (int attempt = 1; attempt <= 6; ++attempt) {
            const int wait_ms = 250 * (1 << (attempt - 1));  // 250..8000 ms
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
            err = cudaMalloc(&ptr, aligned);
            if (err == cudaSuccess) {
                std::fprintf(stderr,
                             "[cuda_mem_pool] cudaMalloc(%zu) recovered after retry %d "
                             "(waited %d ms)\n",
                             aligned, attempt, wait_ms);
                break;
            }
            std::fprintf(stderr,
                         "[cuda_mem_pool] cudaMalloc(%zu) retry %d/6 failed: %s\n",
                         aligned, attempt, cudaGetErrorString(err));
        }
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[cuda_mem_pool] cudaMalloc(%zu) failed: %s\n", aligned,
                     cudaGetErrorString(err));
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(p.mutex);
        p.sizes[ptr] = aligned;
    }
    return ptr;
#else
    (void)bytes;
    return nullptr;
#endif
}

void deallocate(void* ptr) {
    if (!ptr)
        return;
#ifdef USE_CUDA
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);

    auto sit = p.sizes.find(ptr);
    if (sit == p.sizes.end()) {
        // Not one of ours (external cudaMalloc): hand straight back to driver.
        cudaFree(ptr);
        return;
    }
    const size_t aligned = sit->second;

    if (aligned > kMaxCachedBlockBytes || p.cached + aligned > kMaxCachedBytes) {
        // Block too large or cache full: release straight to the driver.
        cudaFree(ptr);
        p.sizes.erase(sit);
        return;
    }

    p.free_list[aligned].push_back(ptr);
    p.cached += aligned;
#else
    (void)ptr;
#endif
}

void clear_cache() {
#ifdef USE_CUDA
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    for (auto& [aligned, blocks] : p.free_list) {
        for (void* ptr : blocks) {
            cudaFree(ptr);
            p.sizes.erase(ptr);
        }
        blocks.clear();
    }
    p.free_list.clear();
    p.cached = 0;
#else
    (void)0;
#endif
}

size_t cached_bytes() {
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    return p.cached;
}

size_t alloc_count() {
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    return p.allocs;
}

size_t hit_count() {
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    return p.hits;
}

size_t live_count() {
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    return p.sizes.size();
}

void print_stats() {
#ifdef USE_CUDA
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    double hit_pct = p.allocs > 0 ? 100.0 * p.hits / p.allocs : 0.0;
    std::fprintf(stderr, "[cuda_mem_pool] allocs=%zu hits=%zu (%.1f%%) cached=%zuKB live=%zu\n",
                 p.allocs, p.hits, hit_pct, p.cached / 1024, p.sizes.size());
#endif
}

}  // namespace cuda_mem
}  // namespace forge