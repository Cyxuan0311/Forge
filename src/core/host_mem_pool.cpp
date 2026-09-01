#include "forge/host_mem_pool.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace forge {
namespace host_mem {

namespace {

// 64 bytes covers AVX-512, NEON (16) and VSX (16) alignment requirements.
// std::aligned_alloc also requires size to be a multiple of alignment, which
// align_up() guarantees.
constexpr size_t kAlignment = 64;

// MSVC's <cstdlib> does not provide std::aligned_alloc (C11 aligned_alloc is
// unimplemented there), so fall back to _aligned_malloc/_aligned_free. Note the
// argument order differs: _aligned_malloc(size, alignment).
void* raw_aligned_alloc(size_t alignment, size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

void raw_aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// Cap on bytes retained in the cache. System RAM is cheap; keep it generous so
// the per-op working set never thrashes, but bounded so a growing allocation
// pattern (e.g. long prefill) returns memory to the OS.
constexpr size_t kMaxCachedBytes = 256ull * 1024 * 1024;

// Blocks larger than this are never cached (returned straight to the OS).
// Decode/prefill working-set buffers are at most a few MB; this prevents
// one-time load buffers (e.g. model preprocessing) from pinning RAM forever.
constexpr size_t kMaxCachedBlockBytes = 8ull * 1024 * 1024;

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

    void* ptr = raw_aligned_alloc(kAlignment, aligned);
    if (!ptr) {
        // Allocation failed (or unsupported): drop the cache and retry once.
        clear_cache();
        ptr = raw_aligned_alloc(kAlignment, aligned);
    }
    if (!ptr) {
        std::fprintf(stderr, "[host_mem_pool] aligned_alloc(%zu) failed\n", aligned);
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(p.mutex);
        p.sizes[ptr] = aligned;
    }
    return ptr;
}

void deallocate(void* ptr) {
    if (!ptr)
        return;
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);

    auto sit = p.sizes.find(ptr);
    if (sit == p.sizes.end()) {
        // Not one of ours: hand straight back to the OS.
        std::free(ptr);
        return;
    }
    const size_t aligned = sit->second;

    if (aligned > kMaxCachedBlockBytes || p.cached + aligned > kMaxCachedBytes) {
        // Block too large or cache full: release straight to the OS.
        raw_aligned_free(ptr);
        p.sizes.erase(sit);
        return;
    }

    p.free_list[aligned].push_back(ptr);
    p.cached += aligned;
}

void clear_cache() {
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    for (auto& [aligned, blocks] : p.free_list) {
        for (void* ptr : blocks) {
            raw_aligned_free(ptr);
            p.sizes.erase(ptr);
        }
        blocks.clear();
    }
    p.free_list.clear();
    p.cached = 0;
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
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    double hit_pct = p.allocs > 0 ? 100.0 * p.hits / p.allocs : 0.0;
    std::fprintf(stderr, "[host_mem_pool] allocs=%zu hits=%zu (%.1f%%) cached=%zuKB live=%zu\n",
                 p.allocs, p.hits, hit_pct, p.cached / 1024, p.sizes.size());
}

}  // namespace host_mem
}  // namespace forge