#pragma once

#include <cstddef>
#include <vector>
#include <memory>

#include "tensor.h"
#include "backend.h"

namespace nanoinfer {

class ComputeGraph;

struct PlannedAllocation {
    int node_idx;           // Which node's output this is
    size_t size;            // Bytes needed
    int lifetime_begin;     // Producing node index
    int lifetime_end;       // Last consuming node index
    size_t offset;          // Assigned offset in the global buffer
    bool is_graph_output;   // true = last node, keep alive
};

class MemoryPlanner {
public:
    MemoryPlanner() = default;

    // Analyze a ComputeGraph and compute optimal allocation plan
    void plan(const ComputeGraph& graph, bool release_intermediates);

    // Total buffer size needed across all devices
    size_t total_buffer_size() const { return total_size_; }

    // Get planned allocation for a given node
    const PlannedAllocation* get_allocation(int node_idx) const;

    // Estimated size for a node's output (before execution)
    static size_t estimate_output_size(const class GraphNode& node,
                                        const std::vector<TensorPtr>& graph_inputs);

    // Debug info
    size_t allocation_count() const { return allocations_.size(); }

private:
    // Interval allocation: find a free region that fits the given size
    size_t allocate_offset(size_t size);

    std::vector<PlannedAllocation> allocations_;
    struct FreeRegion { size_t offset, size; };
    std::vector<FreeRegion> free_regions_;
    size_t total_size_ = 0;
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

} // namespace nanoinfer
