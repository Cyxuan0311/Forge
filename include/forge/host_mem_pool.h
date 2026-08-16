#pragma once

#include <cstddef>
#include <new>

namespace forge {
namespace host_mem {

// Caching allocator for host (CPU) memory used by Tensors and CPU kernels.
//
// Mirrors cuda_mem_pool: buffers returned by allocate() are 64-byte aligned
// (satisfies AVX2/AVX-512, NEON and VSX kernel requirements), and deallocate()
// returns them to a bounded, size-bucketed free list instead of calling
// std::free. This removes per-operator malloc/free churn (hundreds to
// thousands per decoded token on the CPU inference path).
//
// deallocate(ptr) resolves the buffer size internally, so callers never need
// to pass it. Pointers not allocated by this pool fall back to std::free.
// Memory is returned to the OS once the cached total exceeds the cap or
// clear_cache() is called. All operations are thread-safe.
void* allocate(size_t bytes);
void deallocate(void* ptr);

// Free every cached block back to the OS.
void clear_cache();

// Diagnostics.
size_t cached_bytes();
size_t alloc_count();   // total allocate() calls
size_t hit_count();     // reuses from the cache
size_t live_count();    // distinct buffers currently handed out or cached
void print_stats();

// std::allocator-compliant allocator backed by the host memory pool.
//
// Allows std::vector (and other STL containers) to draw their backing storage
// from the caching pool instead of malloc/free, so the hot CPU kernels
// (per-token GEMV scratch buffers, etc.) reuse cached, 64-byte aligned blocks
// instead of churning the system allocator.
template <typename T>
struct host_allocator {
    using value_type = T;

    host_allocator() = default;
    template <class U>
    constexpr host_allocator(const host_allocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > max_size())
            throw std::bad_array_new_length();
        return static_cast<T*>(forge::host_mem::allocate(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t) noexcept {
        forge::host_mem::deallocate(static_cast<void*>(p));
    }

    std::size_t max_size() const noexcept {
        return static_cast<std::size_t>(-1) / sizeof(T);
    }

    template <class U>
    struct rebind {
        using other = host_allocator<U>;
    };
};

template <class T, class U>
constexpr bool operator==(const host_allocator<T>&, const host_allocator<U>&) noexcept {
    return true;
}

template <class T, class U>
constexpr bool operator!=(const host_allocator<T>&, const host_allocator<U>&) noexcept {
    return false;
}

}  // namespace host_mem
}  // namespace forge