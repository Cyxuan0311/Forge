#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend.h"
#include "backend_scheduler.h"
#include "memory_planner.h"
#include "op_dispatch.h"
#include "op_enum.h"
#include "tensor.h"

namespace forge {

struct GraphNode {
    std::string name;
    OpType op_type = OpType::NONE;
    std::vector<int> input_indices;          // >=0 = graph input, <0 = -node_idx-1
    std::vector<TensorPtr> resolved_inputs;  // Resolved at execution time
    TensorPtr output;

    // Legacy: optional closure for ops that haven't migrated to OpDispatch yet
    // When set, takes priority over OpDispatch for execution
    std::function<TensorPtr(const std::vector<TensorPtr>&)> compute_fn;

    // Fixed-size op params (e.g., eps for rms_norm, scale for scale)
    int32_t op_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};

    DeviceType device = DeviceType::CPU;
};

class ComputeGraph {
public:
    ComputeGraph() = default;

    // Add an input tensor, returns its index
    int add_input(const TensorPtr& tensor);

    // Add a node with an OpType (for OpDispatch-based execution)
    int add_node(const std::string& name, OpType op_type, const std::vector<int>& input_indices,
                 const int32_t* op_params = nullptr, DeviceType dev = DeviceType::CPU);

    // Legacy: add a node with a compute_fn closure
    int add_node(const std::string& name, const std::string& op_type_str,
                 const std::vector<int>& input_indices,
                 std::function<TensorPtr(const std::vector<TensorPtr>&)> compute_fn,
                 DeviceType dev = DeviceType::CPU);

    // Execute the full graph sequentially
    TensorPtr execute();

    // Execute a single node by index
    TensorPtr execute_node(int node_idx);

    // Pre-allocate memory for all intermediate tensors using MemoryPlanner.
    // Must be called before execute(). Returns true on success.
    bool allocate_graph();

    // Free the pre-allocated graph buffer.
    void release_graph();

    void reset();

    int num_nodes() const { return static_cast<int>(nodes_.size()); }
    const GraphNode& node(int idx) const { return nodes_[idx]; }
    TensorPtr get_output(int node_idx) const;

    void set_workspace_backend(std::shared_ptr<Backend> backend);
    Backend* workspace_backend();

    size_t peak_memory() const { return peak_memory_; }

    // Device-aware execution: automatically insert data transfers
    void set_auto_transfer(bool enable) { auto_transfer_ = enable; }
    bool auto_transfer() const { return auto_transfer_; }

    // Release intermediate tensors as soon as they are no longer needed
    void set_release_intermediates(bool enable) { release_intermediates_ = enable; }
    bool release_intermediates() const { return release_intermediates_; }

    const std::vector<TensorPtr>& inputs() const { return inputs_; }

    // Operator fusion: merge consecutive fusible ops
    int optimize_fusion();

    // Apply a scheduling plan: override each node's device based on the plan
    void apply_schedule(const SchedulingPlan& plan);

    // Get the current scheduling plan (if applied)
    const SchedulingPlan* schedule() const { return schedule_ ? &*schedule_ : nullptr; }

    TensorPtr last_output() const;

private:
    TensorPtr ensure_device(const TensorPtr& tensor, DeviceType target_dev);
    std::unordered_set<int> find_consumers(int node_idx) const;

    std::vector<TensorPtr> inputs_;
    std::vector<GraphNode> nodes_;
    std::shared_ptr<Backend> workspace_backend_;
    size_t peak_memory_ = 0;
    bool auto_transfer_ = false;
    bool release_intermediates_ = false;

    // Memory planner and pre-allocated buffer
    std::unique_ptr<MemoryPlanner> planner_;
    std::unique_ptr<GraphBuffer> graph_buffer_;
    bool graph_allocated_ = false;

    // Scheduling plan (optional)
    std::unique_ptr<SchedulingPlan> schedule_;
};

class GraphBuilder {
public:
    GraphBuilder() = default;

    GraphBuilder& input(const TensorPtr& tensor);

    GraphBuilder& op(const std::string& name, const std::string& op_type_str,
                     const std::vector<int>& deps,
                     std::function<TensorPtr(const std::vector<TensorPtr>&)> fn);

    std::unique_ptr<ComputeGraph> build();

private:
    std::vector<TensorPtr> inputs_;
    struct PendingNode {
        std::string name;
        std::string op_type_str;
        std::vector<int> input_indices;
        std::function<TensorPtr(const std::vector<TensorPtr>&)> compute_fn;
    };
    std::vector<PendingNode> pending_nodes_;
};

}  // namespace forge
