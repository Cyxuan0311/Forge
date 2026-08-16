#pragma once

#include <cstddef>

namespace forge {
namespace cuda_mem {

// Caching allocator for CUDA device buffers used by Tensors.
//
// Buffers returned by allocate() are 256-byte aligned. deallocate() returns
// them to a bounded, size-bucketed free list instead of calling cudaFree,
// eliminating per-op driver overhead (~450 malloc/free per decoded token) and
// the device-sync stalls that raw cudaMalloc can trigger on a busy stream.
//
// The allocator is self-describing: deallocate(ptr) resolves the buffer size
// internally, so callers never need to pass it. Memory is returned to the
// driver once the cached total exceeds the cap or clear_cache() is called.
// All operations are thread-safe. No-ops when built without USE_CUDA.
void* allocate(size_t bytes);
void deallocate(void* ptr);

// Free every cached block back to the driver.
void clear_cache();

// Diagnostics (peak working set is kept small by design).
size_t cached_bytes();
size_t alloc_count();   // total allocate() calls
size_t hit_count();     // reuses from the cache
size_t live_count();    // distinct buffers currently handed out or cached
void print_stats();

}  // namespace cuda_mem
}  // namespace forge