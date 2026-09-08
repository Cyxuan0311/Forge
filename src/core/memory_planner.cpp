#include "forge/memory_planner.h"

#include <algorithm>
#include <cstring>

#include "forge/quant_traits.h"

#include "forge/compute_graph.h"
#include "forge/logger.h"
#include "forge/types.h"

namespace forge {

// ---- Estimate output size for a node (before execution) ----
size_t MemoryPlanner::estimate_output_size(const GraphNode& node,
                                           const std::vector<TensorPtr>& graph_inputs) {
    if (node.input_indices.empty())
        return 0;

    if (node.compute_fn) {
        // Legacy compute_fn closures: can't estimate; allocate dynamically
        return 0;
    }

    // For OpDispatch nodes, try to resolve first input's size
    TensorPtr first_input;
    int first_idx = node.input_indices[0];
    if (first_idx >= 0 && first_idx < static_cast<int>(graph_inputs.size()) &&
        graph_inputs[first_idx]) {
        first_input = graph_inputs[first_idx];
    }

    if (!first_input) {
        // Input references another node's output which hasn't executed yet
        return 0;
    }

    size_t input_size = first_input->nbytes();

    switch (node.op_type) {
    case OpType::ADD:
    case OpType::MUL:
    case OpType::SILU:
    case OpType::RMS_NORM:
    case OpType::ROPE:
        return input_size;

    case OpType::MUL_MAT_TRANSB: {
        if (node.input_indices.size() < 2)
            return 0;
        int w_idx = node.input_indices[1];
        if (w_idx >= 0 && w_idx < static_cast<int>(graph_inputs.size()) && graph_inputs[w_idx]) {
            auto weight = graph_inputs[w_idx];
            int64_t rows = first_input->shape()[0];
            int64_t cols = weight->shape()[0];
            // 6-A-6.2: size by the (first) input's element width, not hardcoded FP32 — an FP16
            // chain must allocate an FP16-sized buffer (no FP32 promotion in the planner).
            size_t es = first_input->numel() ? first_input->nbytes() / first_input->numel()
                                             : sizeof(float);
            return static_cast<size_t>(rows * cols * es);
        }
        return 0;
    }

    case OpType::FLASH_ATTN_GQA: {
        int num_heads = node.op_params[0];
        int head_dim = node.op_params[2];
        int64_t seq_len_q = first_input->shape()[0];
        // 6-A-7: size by the (first) input's element width, not hardcoded FP32 —
        // same 6-A-6.2口径 as MUL_MAT_TRANSB above. An FP16 attention chain must
        // allocate an FP16-sized output (no FP32 promotion in the planner).
        size_t es = first_input->numel() ? first_input->nbytes() / first_input->numel()
                                         : sizeof(float);
        return static_cast<size_t>(seq_len_q * num_heads * head_dim * es);
    }

    default:
        return input_size;
    }
}

// ---- Lifetime-based allocation: the core of Phase 2 ----
// Allocates offset for a tensor, considering currently active allocations.
// Graph outputs are never freed (kept alive until graph reset).
size_t MemoryPlanner::allocate_with_lifetime(size_t size, int lifetime_begin, int lifetime_end,
                                             bool is_graph_output) {
    // Align to 256 bytes
    constexpr size_t alignment = 256;
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    // Best-fit search: find the smallest free region that fits
    size_t best_offset = SIZE_MAX;
    size_t best_region_idx = SIZE_MAX;
    size_t best_waste = SIZE_MAX;

    for (size_t i = 0; i < free_regions_.size(); ++i) {
        const auto& region = free_regions_[i];
        size_t aligned_offset = (region.offset + alignment - 1) & ~(alignment - 1);
        size_t available = region.size - (aligned_offset - region.offset);

        if (available >= aligned_size) {
            size_t waste = available - aligned_size;
            if (waste < best_waste) {
                best_waste = waste;
                best_offset = aligned_offset;
                best_region_idx = i;
            }
        }
    }

    if (best_region_idx != SIZE_MAX) {
        // Found a suitable free region
        auto& region = free_regions_[best_region_idx];
        size_t consumed = aligned_size + (best_offset - region.offset);
        region.offset += consumed;
        region.size -= consumed;

        if (region.size == 0) {
            free_regions_.erase(free_regions_.begin() +
                                static_cast<std::ptrdiff_t>(best_region_idx));
        }

        total_size_ = std::max(total_size_, best_offset + aligned_size);

        // Track as active allocation (unless it's a graph output — those stay forever)
        if (!is_graph_output) {
            active_allocs_.push_back({best_offset, aligned_size, lifetime_end, lifetime_begin});
        }

        return best_offset;
    }

    // No suitable free region — extend at end
    size_t current_end = total_size_;
    size_t result = (current_end + alignment - 1) & ~(alignment - 1);
    total_size_ = result + aligned_size;

    if (!is_graph_output) {
        active_allocs_.push_back({result, aligned_size, lifetime_end, lifetime_begin});
    }

    return result;
}

// ---- Free expired allocations: release regions whose lifetime has ended ----
void MemoryPlanner::free_expired(int current_node_idx) {
    std::vector<ActiveAlloc> still_active;
    still_active.reserve(active_allocs_.size());

    for (auto& alloc : active_allocs_) {
        if (alloc.lifetime_end < current_node_idx) {
            // Lifetime ended — release the region
            release_region(alloc.offset, alloc.size);
        } else {
            still_active.push_back(alloc);
        }
    }

    active_allocs_ = std::move(still_active);
}

// ---- Release a region back to free_regions_ and merge adjacent regions ----
void MemoryPlanner::release_region(size_t offset, size_t size) {
    if (size == 0)
        return;

    // Find insertion point to keep free_regions_ sorted by offset
    auto it = std::lower_bound(
        free_regions_.begin(), free_regions_.end(), offset,
        [](const FreeRegion& r, size_t off) { return r.offset < off; });

    // Check if we can merge with the previous region
    bool merged_prev = false;
    if (it != free_regions_.begin()) {
        auto prev = std::prev(it);
        if (prev->offset + prev->size == offset) {
            prev->size += size;
            merged_prev = true;
            // Now check if we can also merge with the next region
            if (it != free_regions_.end() && prev->offset + prev->size == it->offset) {
                prev->size += it->size;
                free_regions_.erase(it);
            }
            return;
        }
    }

    // Check if we can merge with the next region
    if (it != free_regions_.end() && offset + size == it->offset) {
        it->offset = offset;
        it->size += size;
        return;
    }

    // Insert new free region
    free_regions_.insert(it, {offset, size});
}

// ---- Verify the allocation plan: no LIVE overlaps, no out-of-bounds ----
// 6-A-8: lifetime-aware. Two allocations may legally share the same memory
// region when their lifetimes do not intersect -- that reuse is the whole
// point of lifetime-based planning. The previous static interval check flagged
// any offset reuse as "overlap" (false positive on every graph big enough to
// reuse a freed block). Overlap is an error only when both buffers are
// simultaneously live: max(lifetime_begin) <= min(lifetime_end).
bool MemoryPlanner::verify_plan() const {
    for (size_t i = 0; i < allocations_.size(); ++i) {
        const auto& a = allocations_[i];
        if (a.alloc_size == 0)
            continue;
        for (size_t j = i + 1; j < allocations_.size(); ++j) {
            const auto& b = allocations_[j];
            if (b.alloc_size == 0)
                continue;
            bool live_together = std::max(a.lifetime_begin, b.lifetime_begin) <=
                                 std::min(a.lifetime_end, b.lifetime_end);
            if (!live_together)
                continue;
            if (a.offset < b.offset + b.alloc_size && b.offset < a.offset + a.alloc_size) {
                LOG_ERROR("MemoryPlanner: LIVE overlap between node " +
                          std::to_string(a.node_idx) + " [" +
                          std::to_string(a.lifetime_begin) + "," +
                          std::to_string(a.lifetime_end) + "] @+" +
                          std::to_string(a.offset) + "(" + std::to_string(a.alloc_size) +
                          "B) and node " + std::to_string(b.node_idx) + " [" +
                          std::to_string(b.lifetime_begin) + "," +
                          std::to_string(b.lifetime_end) + "] @+" +
                          std::to_string(b.offset) + "(" + std::to_string(b.alloc_size) + "B)");
                return false;
            }
        }
    }

    // Check no allocation exceeds total_size_
    for (const auto& alloc : allocations_) {
        if (alloc.offset + alloc.alloc_size > total_size_) {
            LOG_ERROR("MemoryPlanner: allocation " + std::to_string(alloc.node_idx) +
                      " exceeds total buffer size");
            return false;
        }
    }

    return true;
}

// ---- Main planning function: lifetime-based allocation ----
void MemoryPlanner::plan(const ComputeGraph& graph, bool release_intermediates) {
    int n_nodes = graph.num_nodes();
    if (n_nodes == 0)
        return;

    // Clear state
    allocations_.clear();
    node_to_alloc_.clear();
    free_regions_.clear();
    active_allocs_.clear();
    total_size_ = 0;

    // Step 1: Build consumer relationships and compute lifetimes
    struct NodeAllocInfo {
        size_t size = 0;           // logical size
        size_t alloc_size = 0;     // backend-aligned size
        int lifetime_begin = -1;
        int lifetime_end = -1;
        bool is_output = false;
        std::vector<int> consumers;
    };
    std::vector<NodeAllocInfo> node_info(n_nodes);

    // Build consumer lists
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

    // Phase 6-A-6.2: propagate output shapes/dtypes across the graph (topological order) so
    // every node — not just the first — can be sized/typed by plan() above.
    std::vector<std::vector<int64_t>> node_shapes(n_nodes);
    std::vector<DataType> node_dtypes(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        const auto& node = graph.node(i);
        if (node.input_indices.empty())
            continue;
        auto resolve = [&](int ref, const std::vector<int64_t>*& shape, DataType& dt) {
            shape = nullptr;
            dt = DataType::FP32;
            if (ref >= 0 && ref < static_cast<int>(graph.inputs().size()) && graph.inputs()[ref]) {
                shape = &graph.inputs()[ref]->shape();
                dt = graph.inputs()[ref]->dtype();
                return;
            }
            int prod = -ref - 1;
            if (prod >= 0 && prod < n_nodes && !node_shapes[prod].empty()) {
                shape = &node_shapes[prod];
                dt = node_dtypes[prod];
            }
        };
        const std::vector<int64_t>* s0 = nullptr;
        DataType dt0 = DataType::FP32;
        resolve(node.input_indices[0], s0, dt0);
        if (!s0)
            continue;
        std::vector<int64_t> out_shape = *s0;
        if (node.op_type == OpType::MUL_MAT_TRANSB && node.input_indices.size() >= 2) {
            const std::vector<int64_t>* ws = nullptr;
            DataType wdt = DataType::FP32;
            resolve(node.input_indices[1], ws, wdt);
            if (ws && ws->size() >= 1)
                out_shape = {(*s0)[0], (*ws)[0]};
        }
        node_shapes[i] = std::move(out_shape);
        node_dtypes[i] = dt0;
    }

    // Set lifetime_begin, estimate size, mark graph outputs
    for (int i = 0; i < n_nodes; ++i) {
        node_info[i].lifetime_begin = i;
        node_info[i].lifetime_end = -1;

        // Mark last node as output (graph output is never freed)
        if (i == n_nodes - 1) {
            node_info[i].is_output = true;
        }

        // Phase 6-A-6.2: propagate shapes/dtypes from graph inputs so downstream nodes
        // (whose first input is another node's output, not a graph input) get a correctly
        // sized AND correctly typed allocation. Without this, only the first node is sized
        // and the rest of a chain is never given a dst-injection buffer (and an FP16 chain
        // would otherwise be under-sized as FP32). `node_shapes`/`node_dtypes` are built in
        // topological (index) order, which the graph builder guarantees.
        if (!node_shapes[i].empty()) {
            size_t n = 1;
            for (auto d : node_shapes[i]) n *= static_cast<size_t>(d);
            node_info[i].size = n * dtype_size(node_dtypes[i]);
        } else {
            node_info[i].size = estimate_output_size(graph.node(i), graph.inputs());
        }

        // Compute backend-specific allocation size
        if (node_info[i].size > 0) {
            if (backend_) {
                node_info[i].alloc_size = backend_->get_alloc_size(node_info[i].size);
            } else {
                // Default: align to 256 bytes
                constexpr size_t alignment = 256;
                node_info[i].alloc_size =
                    (node_info[i].size + alignment - 1) & ~(alignment - 1);
            }
        }
    }

    // Set lifetime_end = last consumer index
    for (int i = 0; i < n_nodes; ++i) {
        auto& info = node_info[i];
        if (info.consumers.empty()) {
            info.lifetime_end = info.is_output ? n_nodes - 1 : i;
        } else {
            info.lifetime_end = *std::max_element(info.consumers.begin(), info.consumers.end());
        }
    }

    // Step 2: Process nodes in execution order (topological = index order)
    // For each node: free expired allocations, then allocate for this node.
    for (int i = 0; i < n_nodes; ++i) {
        const auto& info = node_info[i];
        if (info.size == 0 || info.alloc_size == 0)
            continue;

        // Free allocations whose lifetime ended before this node
        free_expired(i);

        // Allocate for this node
        size_t offset = allocate_with_lifetime(info.alloc_size, info.lifetime_begin,
                                                info.lifetime_end, info.is_output);

        PlannedAllocation alloc;
        alloc.node_idx = i;
        alloc.size = info.size;
        alloc.alloc_size = info.alloc_size;
        alloc.lifetime_begin = info.lifetime_begin;
        alloc.lifetime_end = info.lifetime_end;
        alloc.is_graph_output = info.is_output;
        alloc.offset = offset;

        size_t alloc_idx = allocations_.size();
        allocations_.push_back(alloc);
        node_to_alloc_[i] = alloc_idx;
    }

    // Step 3: Verify the plan
    if (!verify_plan()) {
        LOG_ERROR("MemoryPlanner: allocation plan verification FAILED — overlaps or OOB detected");
    }

    LOG_INFO("MemoryPlanner: planned " + std::to_string(allocations_.size()) +
             " allocations, total buffer = " + std::to_string(total_size_) + " bytes (" +
             std::to_string(total_size_ / (1024 * 1024)) + " MB)");
}

// ---- O(1) lookup via hash map cache ----
const PlannedAllocation* MemoryPlanner::get_allocation(int node_idx) const {
    auto it = node_to_alloc_.find(node_idx);
    if (it == node_to_alloc_.end())
        return nullptr;
    return &allocations_[it->second];
}

// ---- Legacy interval allocation (used by allocate_with_lifetime internally) ----
size_t MemoryPlanner::allocate_offset(size_t size) {
    constexpr size_t alignment = 256;
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    for (auto it = free_regions_.begin(); it != free_regions_.end(); ++it) {
        size_t aligned_offset = (it->offset + alignment - 1) & ~(alignment - 1);
        size_t available = it->size - (aligned_offset - it->offset);

        if (available >= aligned_size) {
            size_t result = aligned_offset;
            size_t consumed = aligned_size + (aligned_offset - it->offset);
            it->offset += consumed;
            it->size -= consumed;

            if (it->size == 0) {
                free_regions_.erase(it);
            }

            total_size_ = std::max(total_size_, result + aligned_size);
            return result;
        }
    }

    // No suitable free region — extend at end
    size_t result = (total_size_ + alignment - 1) & ~(alignment - 1);
    total_size_ = result + aligned_size;
    return result;
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
    : backend_(std::move(other.backend_)), data_(other.data_), size_(other.size_) {
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

}  // namespace forge