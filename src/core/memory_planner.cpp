#include "nanoinfer/memory_planner.h"
#include "nanoinfer/compute_graph.h"
#include "nanoinfer/types.h"
#include "nanoinfer/logger.h"
#include <algorithm>
#include <cstring>

namespace nanoinfer {

size_t MemoryPlanner::estimate_output_size(const GraphNode& node,
                                            const std::vector<TensorPtr>& graph_inputs) {
    if (node.input_indices.empty()) return 0;

    if (node.compute_fn) {
        // Legacy compute_fn closures: can't estimate; allocate dynamically
        return 0;
    }

    // For OpDispatch nodes, try to resolve first input's size
    TensorPtr first_input;
    int first_idx = node.input_indices[0];
    if (first_idx >= 0 && first_idx < static_cast<int>(graph_inputs.size()) && graph_inputs[first_idx]) {
        first_input = graph_inputs[first_idx];
    }

    if (!first_input) {
        // Input references another node's output which hasn't executed yet
        // MUL_MAT_TRANSB and RMS_NORM handle this via graph inputs (weights)
        return 0;
    }

    // Most element-wise ops produce same-shaped FP32 output
    size_t input_size = first_input->nbytes();

    switch (node.op_type) {
        case OpType::ADD:
        case OpType::MUL:
        case OpType::SILU:
        case OpType::RMS_NORM:
        case OpType::ROPE:
            return input_size;

        case OpType::MUL_MAT_TRANSB: {
            // Output = (rows, N) where N = weight shape[0], rows = input[0] rows
            if (node.input_indices.size() < 2) return 0;
            int w_idx = node.input_indices[1];
            if (w_idx >= 0 && w_idx < static_cast<int>(graph_inputs.size()) && graph_inputs[w_idx]) {
                auto weight = graph_inputs[w_idx];
                int64_t rows = first_input->shape()[0];
                int64_t cols = weight->shape()[0];
                return static_cast<size_t>(rows * cols * sizeof(float));
            }
            return 0;
        }

        case OpType::FLASH_ATTN_GQA: {
            // Need Q shape + num_heads/head_dim from params
            if (!node.op_params) return 0;
            int num_heads = node.op_params[0];
            int head_dim = node.op_params[2];
            int64_t seq_len_q = first_input->shape()[0];
            return static_cast<size_t>(seq_len_q * num_heads * head_dim * sizeof(float));
        }

        default:
            return input_size;
    }
}

void MemoryPlanner::plan(const ComputeGraph& graph, bool release_intermediates) {
    int n_nodes = graph.num_nodes();
    if (n_nodes == 0) return;

    // Step 1: Collect output sizes and build consumer lists
    struct NodeAllocInfo {
        size_t size = 0;
        int lifetime_begin = -1;
        int lifetime_end = -1;
        bool is_output = false;
        std::vector<int> consumers;
    };
    std::vector<NodeAllocInfo> node_info(n_nodes);

    // Identify graph inputs: nodes that reference inputs_ via non-negative indices
    // These inputs are external and not managed by us.

    // Step 2: Build consumer relationships
    for (int i = 0; i < n_nodes; ++i) {
        const auto& node = graph.node(i);
        for (int idx : node.input_indices) {
            if (idx < 0) {
                int producer = -idx - 1;
                if (producer >= 0 && producer < n_nodes) {
                    node_info[producer].consumers.push_back(i);
                }
            }
        }
    }

    // Step 3: Set lifetime_begin and estimate size for each node
    for (int i = 0; i < n_nodes; ++i) {
        node_info[i].lifetime_begin = i;
        node_info[i].lifetime_end = -1;  // unknown yet

        // Mark last node as output
        if (i == n_nodes - 1) {
            node_info[i].is_output = true;
        }

        // Estimate size
        node_info[i].size = estimate_output_size(graph.node(i), graph.inputs());
    }

    // Step 4: Set lifetime_end = last consumer index
    for (int i = 0; i < n_nodes; ++i) {
        auto& info = node_info[i];
        if (info.consumers.empty()) {
            // No consumers: either graph output or dead node
            info.lifetime_end = info.is_output ? n_nodes - 1 : i;
        } else {
            info.lifetime_end = *std::max_element(
                info.consumers.begin(), info.consumers.end());
        }
    }

    // Step 5: Build allocation list
    allocations_.clear();

    // Helper to check if a node is a graph input (referenced by some other node)
    std::vector<bool> is_graph_input(n_nodes, false);
    for (int i = 0; i < n_nodes; ++i) {
        const auto& node = graph.node(i);
        (void)node;
        // A node whose output is used as a graph input (referenced via non-negative index)
        // Currently we don't have this case, but check anyway:
        for (int j = 0; j < n_nodes; ++j) {
            const auto& other = graph.node(j);
            for (int idx : other.input_indices) {
                if (idx >= 0) {
                    // This node reads from graph inputs_, not from another node
                    // No special handling needed
                }
            }
        }
    }

    for (int i = 0; i < n_nodes; ++i) {
        const auto& info = node_info[i];
        if (info.size == 0) continue;
        // Skip nodes that are just compute_fn wrappers (OpType::CUSTOM with size 0)
        if (info.lifetime_begin < 0) continue;

        PlannedAllocation alloc;
        alloc.node_idx = i;
        alloc.size = info.size;
        alloc.lifetime_begin = info.lifetime_begin;
        alloc.lifetime_end = info.lifetime_end;
        alloc.is_graph_output = info.is_output;
        alloc.offset = 0;

        allocations_.push_back(alloc);
    }

    // Step 6: Sort by size descending, then lifetime_begin ascending
    // This is a simple heuristic: allocate largest tensors first to minimize fragmentation.
    std::sort(allocations_.begin(), allocations_.end(),
        [](const PlannedAllocation& a, const PlannedAllocation& b) {
            if (a.size != b.size) return a.size > b.size;
            return a.lifetime_begin < b.lifetime_begin;
        });

    // Step 7: Interval allocation
    free_regions_.clear();
    free_regions_.push_back({0, SIZE_MAX});

    total_size_ = 0;
    for (auto& alloc : allocations_) {
        size_t offset = allocate_offset(alloc.size);
        alloc.offset = offset;
        total_size_ = std::max(total_size_, offset + alloc.size);
    }

    LOG_INFO("MemoryPlanner: planned " + std::to_string(allocations_.size()) +
             " allocations, total buffer = " + std::to_string(total_size_) +
             " bytes (" + std::to_string(total_size_ / (1024*1024)) + " MB)");
}

size_t MemoryPlanner::allocate_offset(size_t size) {
    // Align to 256 bytes
    size_t alignment = 256;
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    for (auto it = free_regions_.begin(); it != free_regions_.end(); ++it) {
        size_t aligned_offset = (it->offset + alignment - 1) & ~(alignment - 1);
        size_t available = it->size - (aligned_offset - it->offset);

        if (available >= aligned_size) {
            size_t result = aligned_offset;

            // Update free region
            size_t consumed = aligned_size + (aligned_offset - it->offset);
            it->offset += consumed;
            it->size -= consumed;

            if (it->size == 0) {
                free_regions_.erase(it);
            }

            return result;
        }
    }

    // No suitable free region found — extend at end
    size_t current_end = 0;
    for (const auto& r : free_regions_) {
        current_end = std::max(current_end, r.offset + r.size);
    }
    size_t result = (current_end + alignment - 1) & ~(alignment - 1);
    // Free region tracking: reduce the last best free region or just use the end
    total_size_ = std::max(total_size_, result + aligned_size);
    return result;
}

const PlannedAllocation* MemoryPlanner::get_allocation(int node_idx) const {
    for (const auto& alloc : allocations_) {
        if (alloc.node_idx == node_idx) return &alloc;
    }
    return nullptr;
}

// GraphBuffer implementation

GraphBuffer::GraphBuffer(std::shared_ptr<Backend> backend, size_t size)
    : backend_(std::move(backend)), size_(size) {
    if (backend_ && size_ > 0) {
        data_ = backend_->allocate(size_);
        if (data_) {
            backend_->memset(data_, 0, size_);
        }
    }
}

GraphBuffer::~GraphBuffer() {
    if (data_ && backend_) {
        backend_->deallocate(data_, size_);
    }
}

GraphBuffer::GraphBuffer(GraphBuffer&& other) noexcept
    : backend_(std::move(other.backend_)),
      data_(other.data_),
      size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

GraphBuffer& GraphBuffer::operator=(GraphBuffer&& other) noexcept {
    if (this != &other) {
        if (data_ && backend_) {
            backend_->deallocate(data_, size_);
        }
        backend_ = std::move(other.backend_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

} // namespace nanoinfer
