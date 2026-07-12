#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <set>
#include "types.h"

namespace nanoinfer {

class MemoryPool {
public:
    explicit MemoryPool(DeviceType device, size_t pool_size = 0);
    ~MemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr, size_t size);

    size_t used() const { return used_; }
    size_t capacity() const { return capacity_; }
    void reset();

    void* allocate_reuse(size_t size);
    void mark_reusable(void* ptr, size_t size);

    size_t total_allocated() const { return total_allocated_; }
    size_t reuse_count() const { return reuse_count_; }

private:
    size_t align_size(size_t size) const;

    DeviceType device_;
    size_t capacity_ = 0;
    size_t used_ = 0;
    size_t total_allocated_ = 0;
    size_t reuse_count_ = 0;
    std::mutex mutex_;
    std::unordered_map<void*, size_t> allocations_;

    std::unordered_map<size_t, std::vector<void*>> free_lists_;
};

} // namespace nanoinfer
