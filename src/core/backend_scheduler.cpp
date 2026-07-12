#include "nanoinfer/backend_scheduler.h"
#include "nanoinfer/compute_graph.h"
#include "nanoinfer/logger.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace nanoinfer {

BackendScheduler::BackendScheduler() = default;

void BackendScheduler::set_gpu_memory_budget(size_t bytes_per_device) {
    gpu_budget_per_device_ = bytes_per_device;
}

void BackendScheduler::set_op_device_hint(OpType op, DeviceType preferred_dev) {
    op_hints_[op] = preferred_dev;
    hints_enabled_ = true;
}

std::vector<DeviceInfo> BackendScheduler::enumerate_devices() {
    return BackendManager::instance().available_devices();
}

size_t BackendScheduler::estimate_node_size(const GraphNode& node) {
    // Use resolved input size as heuristic for most ops
    if (!node.resolved_inputs.empty() && node.resolved_inputs[0]) {
        return node.resolved_inputs[0]->nbytes();
    }
    if (!node.input_indices.empty()) {
        // If we can't resolve yet, return a conservative estimate
        return 0;
    }
    return 0;
}

std::vector<int> BackendScheduler::greedy_assign(
        const ComputeGraph& graph,
        const std::vector<DeviceInfo>& devices,
        std::vector<size_t>& node_sizes) {
    int n_nodes = graph.num_nodes();
    int n_devices = static_cast<int>(devices.size());
    std::vector<int> assignments(n_nodes, 0); // Default: device 0 (CPU)

    if (n_devices == 0) return assignments;
    if (n_devices == 1) {
        // Only one device available
        std::fill(assignments.begin(), assignments.end(), 0);
        return assignments;
    }

    // Build predecessors list for each node
    std::vector<std::vector<int>> pred_indices(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        const auto& node = graph.node(i);
        for (int idx : node.input_indices) {
            if (idx < 0) {
                int pred = -idx - 1;
                if (pred >= 0 && pred < i) {
                    pred_indices[i].push_back(pred);
                }
            }
        }
    }

    // Track used bytes per device
    std::vector<size_t> device_used_bytes(n_devices, 0);
    for (int i = 0; i < n_devices; ++i) {
        // Reserve some memory for weights and cache
        device_used_bytes[i] = 0;
    }

    // Build node order: sort by size descending (large nodes first)
    std::vector<int> order(n_nodes);
    for (int i = 0; i < n_nodes; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return node_sizes[a] > node_sizes[b]; });

    // Assign each node to best device
    for (int node_idx : order) {
        if (node_sizes[node_idx] == 0) {
            // Small/unknown-sized nodes: keep on CPU by default
            assignments[node_idx] = 0;
            continue;
        }
        assignments[node_idx] = pick_device(
            node_idx, assignments, node_sizes,
            devices, device_used_bytes, pred_indices);
    }

    return assignments;
}

int BackendScheduler::pick_device(
        int node_idx,
        const std::vector<int>& current_assignments,
        const std::vector<size_t>& node_sizes,
        const std::vector<DeviceInfo>& devices,
        std::vector<size_t>& device_used_bytes,
        const std::vector<std::vector<int>>& pred_indices) const {
    int n_devices = static_cast<int>(devices.size());
    size_t node_size = node_sizes[node_idx];

    // Check op hint
    int hint_device = -1;
    if (hints_enabled_) {
        // We don't have the op_type here directly from GraphNode in the current API...
        // Skip for now
    }

    // Count predecessors per device for affinity scoring
    std::vector<int> pred_count(n_devices, 0);
    for (int pred : pred_indices[node_idx]) {
        if (pred >= 0 && pred < static_cast<int>(current_assignments.size())) {
            int dev = current_assignments[pred];
            if (dev >= 0 && dev < n_devices) {
                pred_count[dev]++;
            }
        }
    }

    // Score each device
    int best_device = 0; // Default: CPU
    int best_score = -1;

    for (int d = 0; d < n_devices; ++d) {
        if (devices[d].type == DeviceType::CPU) {
            // CPU always has capacity (unlimited virtual memory)
            int score = pred_count[d] * 10;
            if (score > best_score) {
                best_score = score;
                best_device = d;
            }
            continue;
        }

        // GPU device: check memory budget
        size_t budget = gpu_budget_per_device_;
        if (budget == 0) {
            budget = devices[d].memory_free;
        }

        size_t used = device_used_bytes[d];
        size_t available = (used < budget) ? (budget - used) : 0;

        if (node_size <= available || node_size == 0) {
            // Can fit on this GPU
            int score = pred_count[d] * 10 + 5; // GPU bonus
            if (score > best_score) {
                best_score = score;
                best_device = d;
            }
        }
    }

    // Update used bytes for GPU devices
    if (best_device < n_devices && devices[best_device].type != DeviceType::CPU) {
        device_used_bytes[best_device] += node_size;
    }

    return best_device;
}

SchedulingPlan BackendScheduler::schedule(const ComputeGraph& graph) {
    SchedulingPlan plan;

    // Step 1: Enumerate available devices
    plan.devices = enumerate_devices();
    if (plan.devices.empty()) {
        LOG_ERROR("BackendScheduler: no devices available");
        return plan;
    }

    int n_nodes = graph.num_nodes();
    if (n_nodes == 0) {
        plan.valid = true;
        return plan;
    }

    // Step 2: Estimate per-node sizes
    std::vector<size_t> node_sizes(n_nodes, 0);
    for (int i = 0; i < n_nodes; ++i) {
        node_sizes[i] = estimate_node_size(graph.node(i));
    }

    // Step 3: Greedy assignment
    auto node_to_device = greedy_assign(graph, plan.devices, node_sizes);

    // Step 4: Build assignment list
    plan.node_to_device = node_to_device;
    for (int i = 0; i < n_nodes; ++i) {
        NodeAssignment assign;
        assign.node_idx = i;
        assign.device_index = node_to_device[i];
        assign.estimated_bytes = node_sizes[i];
        plan.assignments.push_back(assign);

        if (assign.device_index < static_cast<int>(plan.devices.size())) {
            if (plan.devices[assign.device_index].type == DeviceType::CUDA) {
                plan.total_gpu_bytes += node_sizes[i];
            } else {
                plan.total_cpu_bytes += node_sizes[i];
            }
        }
    }

    plan.valid = true;

    // Log summary
    LOG_INFO("BackendScheduler: " + std::to_string(n_nodes) + " nodes scheduled across " +
             std::to_string(plan.devices.size()) + " devices");
    for (size_t d = 0; d < plan.devices.size(); ++d) {
        const auto& dev = plan.devices[d];
        int count = 0;
        for (int a : node_to_device) {
            if (a == static_cast<int>(d)) count++;
        }
        LOG_INFO("  " + dev.name + ": " + std::to_string(count) + " nodes");
    }

    return plan;
}

} // namespace nanoinfer
