#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "tensor.h"

namespace forge {

class ComputeGraph;

struct PlannedAllocation {
    int node_idx;          // Which node's output this is
    size_t size;           // Bytes needed (logical)
    size_t alloc_size;     // Bytes needed (backend-aligned, for actual allocation)
    int lifetime_begin;    // Producing node index
    int lifetime_end;      // Last consuming node index
    size_t offset;         // Assigned offset in the global buffer
    bool is_graph_output;  // true = last node, keep alive
};

class MemoryPlanner {
public:
    MemoryPlanner() = default;

    // Analyze a ComputeGraph and compute optimal allocation plan.
    // Uses lifetime-based allocation: tensors with non-overlapping lifetimes
    // share the same memory region (aligned with llama.cpp ggml_gallocr).
    void plan(const ComputeGraph& graph, bool release_intermediates);

    // Total buffer size needed across all devices
    size_t total_buffer_size() const { return total_size_; }

    // Get planned allocation for a given node (O(1) via hash map cache)
    const PlannedAllocation* get_allocation(int node_idx) const;

    // Estimated size for a node's output (before execution)
    static size_t estimate_output_size(const class GraphNode& node,
                                       const std::vector<TensorPtr>& graph_inputs);

    // Debug info
    size_t allocation_count() const { return allocations_.size(); }

    // Set backend for allocation size queries (Phase 2: backend-specific alloc size)
    void set_backend(Backend* backend) { backend_ = backend; }

    // Lifetime-based allocation: allocate offset for a tensor, considering
    // currently active allocations. Frees regions whose lifetime has ended.
    size_t allocate_with_lifetime(size_t size, int lifetime_begin, int lifetime_end,
                                  bool is_graph_output);

    // Release allocations whose lifetime_end < current_node_idx.
    // Returns freed regions to free_regions_ and merges adjacent regions.
    void free_expired(int current_node_idx);

    // Verify the allocation plan: no overlaps, no out-of-bounds.
    // Returns true if valid.
    bool verify_plan() const;

private:
    // Interval allocation: find a free region that fits the given size (best-fit)
    size_t allocate_offset(size_t size);

    // Release a region back to free_regions_ and merge adjacent regions
    void release_region(size_t offset, size_t size);

    std::vector<PlannedAllocation> allocations_;

    // O(1) lookup cache: node_idx -> index in allocations_
    std::unordered_map<int, size_t> node_to_alloc_;

    struct FreeRegion {
        size_t offset, size;
    };
    std::vector<FreeRegion> free_regions_;
    size_t total_size_ = 0;

    // Backend for allocation size queries (may be null)
    Backend* backend_ = nullptr;

    // Active allocations for lifetime tracking: {offset, {size, lifetime_end}}
    struct ActiveAlloc {
        size_t offset;
        size_t size;
        int lifetime_end;
        int node_idx;
    };
    std::vector<ActiveAlloc> active_allocs_;
};

// RAII handle for a pre-allocated graph buffer
class GraphBuffer {
public:
    GraphBuffer(std::shared_ptr<Backend> backend, size_t size);
    ~GraphBuffer();

    GraphBuffer(const GraphBuffer&) = delete;
    GraphBuffer& operator=(const GraphBuffer&) = delete;
    GraphBuffer(GraphBuffer&& other) noexcept;
    GraphBuffer& operator=(GraphBuffer&& other) noexcept;

    void* data() const { return data_; }
    size_t size() const { return size_; }
    DeviceType device() const { return backend_ ? backend_->device_type() : DeviceType::CPU; }
    bool valid() const { return data_ != nullptr; }

private:
    std::shared_ptr<Backend> backend_;
    void* data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace forge