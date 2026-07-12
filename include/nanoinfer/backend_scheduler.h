#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstddef>
#include "backend.h"
#include "op_enum.h"

namespace nanoinfer {

class ComputeGraph;
struct GraphNode;

// Per-node scheduling decision
struct NodeAssignment {
    int node_idx;
    int device_index;               // Index into SchedulingPlan::devices
    size_t estimated_bytes = 0;
};

// Result of scheduling: maps each graph node to a device
struct SchedulingPlan {
    std::vector<DeviceInfo> devices;
    std::vector<NodeAssignment> assignments;     // size == num_nodes, order matches node indices
    std::vector<int> node_to_device;            // convenience: node_idx → device_index
    size_t total_gpu_bytes = 0;
    size_t total_cpu_bytes = 0;
    bool valid = false;
};

// Device-aware scheduler that assigns ops to optimal backends.
// Uses a greedy strategy: place compute-intensive ops on GPU,
// fall back to CPU when GPU memory is exhausted,
// and minimize cross-device data transfers through node affinity.
class BackendScheduler {
public:
    BackendScheduler();

    // Set available GPU memory budget per device (0 = use all available)
    void set_gpu_memory_budget(size_t bytes_per_device);

    // Set per-op device hints (e.g., always run matmul on CUDA)
    void set_op_device_hint(OpType op, DeviceType preferred_dev);

    // Build a schedule for the given graph
    SchedulingPlan schedule(const ComputeGraph& graph);

    // Static estimate of a node's output size (in bytes)
    static size_t estimate_node_size(const GraphNode& node);

private:
    // Collect available devices from BackendManager
    std::vector<DeviceInfo> enumerate_devices();

    // Greedy assignment logic

    // Simple greedy assignment
    std::vector<int> greedy_assign(const ComputeGraph& graph,
                                   const std::vector<DeviceInfo>& devices,
                                   std::vector<size_t>& node_sizes);

    // Find the best device for a node given its predecessors' devices
    int pick_device(int node_idx,
                    const std::vector<int>& current_assignments,
                    const std::vector<size_t>& node_sizes,
                    const std::vector<DeviceInfo>& devices,
                    std::vector<size_t>& device_used_bytes,
                    const std::vector<std::vector<int>>& pred_indices) const;

    size_t gpu_budget_per_device_ = 0;
    std::unordered_map<OpType, DeviceType> op_hints_;
    bool hints_enabled_ = false;
};

} // namespace nanoinfer
