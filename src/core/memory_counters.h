#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace forge {

// Lightweight global counters for memory allocation and cross-device copies.
// Thread-safe via std::atomic. Used by tests and benchmarks to verify:
//   - graph-mode execution produces zero cudaMalloc/cudaFree calls (Phase 2 goal)
//   - no silent D2H/H2D fallback paths exist (Phase 4 goal)
//
// These counters are ALWAYS compiled in (zero cost when not read).
// No conditional compilation needed — the counters are just atomic integers.
struct MemoryCounters {
    // Allocation counters
    std::atomic<uint64_t> cpu_malloc_count{0};
    std::atomic<uint64_t> cpu_free_count{0};
    std::atomic<uint64_t> cuda_malloc_count{0};
    std::atomic<uint64_t> cuda_free_count{0};

    // Cross-device copy counters
    std::atomic<uint64_t> h2d_copy_count{0};   // Host → Device
    std::atomic<uint64_t> d2h_copy_count{0};   // Device → Host
    std::atomic<uint64_t> d2d_copy_count{0};   // Device → Device

    // Total bytes transferred cross-device
    std::atomic<uint64_t> h2d_bytes{0};
    std::atomic<uint64_t> d2h_bytes{0};
    std::atomic<uint64_t> d2d_bytes{0};

    static MemoryCounters& instance() {
        static MemoryCounters c;
        return c;
    }

    // Reset all counters to zero
    void reset() {
        cpu_malloc_count = 0;
        cpu_free_count = 0;
        cuda_malloc_count = 0;
        cuda_free_count = 0;
        h2d_copy_count = 0;
        d2h_copy_count = 0;
        d2d_copy_count = 0;
        h2d_bytes = 0;
        d2h_bytes = 0;
        d2d_bytes = 0;
    }

    // Snapshot all counters into a struct for Python binding
    struct Snapshot {
        uint64_t cpu_malloc;
        uint64_t cpu_free;
        uint64_t cuda_malloc;
        uint64_t cuda_free;
        uint64_t h2d_copies;
        uint64_t d2h_copies;
        uint64_t d2d_copies;
        uint64_t h2d_bytes;
        uint64_t d2h_bytes;
        uint64_t d2d_bytes;
    };

    Snapshot snapshot() const {
        return {
            cpu_malloc_count.load(),
            cpu_free_count.load(),
            cuda_malloc_count.load(),
            cuda_free_count.load(),
            h2d_copy_count.load(),
            d2h_copy_count.load(),
            d2d_copy_count.load(),
            h2d_bytes.load(),
            d2h_bytes.load(),
            d2d_bytes.load(),
        };
    }

private:
    MemoryCounters() = default;
};

}  // namespace forge