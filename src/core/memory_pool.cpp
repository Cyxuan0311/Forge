#include "forge/memory_pool.h"
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace forge {

MemoryPool::MemoryPool(DeviceType device, size_t pool_size)
    : device_(device), capacity_(pool_size) {}

MemoryPool::~MemoryPool() {
    reset();
}

size_t MemoryPool::align_size(size_t size) const {
    const size_t alignment = 256;
    return (size + alignment - 1) & ~(alignment - 1);
}

void* MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t aligned = align_size(size);

    void* ptr = nullptr;
    if (device_ == DeviceType::CPU) {
        ptr = std::malloc(aligned);
    } else {
#ifdef USE_CUDA
        cudaError_t err = cudaMalloc(&ptr, aligned);
        if (err != cudaSuccess) return nullptr;
#endif
    }

    if (ptr) {
        allocations_[ptr] = aligned;
        used_ += aligned;
        total_allocated_ += aligned;
    }
    return ptr;
}

void MemoryPool::deallocate(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        used_ -= it->second;
        allocations_.erase(it);

        if (device_ == DeviceType::CPU) {
            std::free(ptr);
        } else {
#ifdef USE_CUDA
            cudaFree(ptr);
#endif
        }
    }
}

void* MemoryPool::allocate_reuse(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t aligned = align_size(size);

    auto it = free_lists_.find(aligned);
    if (it != free_lists_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        allocations_[ptr] = aligned;
        used_ += aligned;
        reuse_count_++;
        return ptr;
    }

    void* ptr = nullptr;
    if (device_ == DeviceType::CPU) {
        ptr = std::malloc(aligned);
    } else {
#ifdef USE_CUDA
        cudaError_t err = cudaMalloc(&ptr, aligned);
        if (err != cudaSuccess) return nullptr;
#endif
    }

    if (ptr) {
        allocations_[ptr] = aligned;
        used_ += aligned;
        total_allocated_ += aligned;
    }
    return ptr;
}

void MemoryPool::mark_reusable(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t aligned = align_size(size);

    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        used_ -= it->second;
        allocations_.erase(it);
        free_lists_[aligned].push_back(ptr);
    }
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [ptr, size] : allocations_) {
        if (device_ == DeviceType::CPU) {
            std::free(ptr);
        } else {
#ifdef USE_CUDA
            cudaFree(ptr);
#endif
        }
    }
    allocations_.clear();

    for (auto& [size, ptrs] : free_lists_) {
        for (auto* ptr : ptrs) {
            if (device_ == DeviceType::CPU) {
                std::free(ptr);
            } else {
#ifdef USE_CUDA
                cudaFree(ptr);
#endif
            }
        }
    }
    free_lists_.clear();

    used_ = 0;
    total_allocated_ = 0;
    reuse_count_ = 0;
}

} // namespace forge
