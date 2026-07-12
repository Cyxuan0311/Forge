#include "forge/compute_graph.h"
#include "forge/memory_planner.h"
#include "forge/backend.h"
#include "forge/logger.h"
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace forge {

int ComputeGraph::add_input(const TensorPtr& tensor) {
    int idx = static_cast<int>(inputs_.size());
    inputs_.push_back(tensor);
    return idx;
}

int ComputeGraph::add_node(const std::string& name,
                            OpType op_type,
                            const std::vector<int>& input_indices,
                            const int32_t* op_params,
                            DeviceType dev) {
    int node_idx = static_cast<int>(nodes_.size());

    GraphNode node;
    node.name = name;
    node.op_type = op_type;
    node.input_indices = input_indices;
    node.device = dev;
    if (op_params) {
        std::copy(op_params, op_params + OP_PARAMS_MAX_SIZE / sizeof(int32_t),
                  node.op_params);
    }

    nodes_.push_back(std::move(node));
    return node_idx;
}

int ComputeGraph::add_node(const std::string& name,
                            const std::string& op_type_str,
                            const std::vector<int>& input_indices,
                            std::function<TensorPtr(const std::vector<TensorPtr>&)> compute_fn,
                            DeviceType dev) {
    int node_idx = static_cast<int>(nodes_.size());

    GraphNode node;
    node.name = name;
    node.op_type = OpType::CUSTOM;
    node.input_indices = input_indices;
    node.compute_fn = std::move(compute_fn);
    node.device = dev;

    nodes_.push_back(std::move(node));
    return node_idx;
}

TensorPtr ComputeGraph::ensure_device(const TensorPtr& tensor, DeviceType target_dev) {
    if (!tensor || tensor->device() == target_dev) return tensor;
    auto transferred = std::make_shared<Tensor>(tensor->dtype(), tensor->shape(), target_dev);
    transferred->copy_from(*tensor);
    return transferred;
}

std::unordered_set<int> ComputeGraph::find_consumers(int node_idx) const {
    std::unordered_set<int> consumers;
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        for (int idx : nodes_[i].input_indices) {
            if (idx < 0 && (-idx - 1) == node_idx) {
                consumers.insert(i);
            }
        }
    }
    return consumers;
}

bool ComputeGraph::allocate_graph() {
    if (graph_allocated_) return true;

    planner_ = std::make_unique<MemoryPlanner>();
    planner_->plan(*this, release_intermediates_);

    size_t total_size = planner_->total_buffer_size();
    if (total_size == 0) {
        graph_allocated_ = true;
        return true;
    }

    auto* backend = workspace_backend();
    if (!backend) {
        backend = BackendManager::instance().get_backend(DeviceType::CPU).get();
    }

    graph_buffer_ = std::make_unique<GraphBuffer>(
        BackendManager::instance().get_backend(backend->device_type()),
        total_size);

    if (!graph_buffer_->valid()) {
        LOG_ERROR("ComputeGraph: failed to allocate graph buffer of " +
                  std::to_string(total_size) + " bytes");
        return false;
    }

    graph_allocated_ = true;
    return true;
}

void ComputeGraph::release_graph() {
    graph_buffer_.reset();
    planner_.reset();
    graph_allocated_ = false;
}

TensorPtr ComputeGraph::execute() {
    std::vector<int> ref_count(nodes_.size(), 0);
    for (const auto& node : nodes_) {
        for (int idx : node.input_indices) {
            if (idx < 0) {
                int node_idx = -idx - 1;
                if (node_idx >= 0 && node_idx < static_cast<int>(nodes_.size())) {
                    ref_count[node_idx]++;
                }
            }
        }
    }

    // Auto-allocate graph buffer if planner is configured
    if (!graph_allocated_) {
        allocate_graph();
    }

    // Apply scheduling plan on every execute (handles reset + re-execute)
    if (schedule_ && schedule_->valid) {
        int n_plan = static_cast<int>(nodes_.size());
        for (int i = 0; i < n_plan && i < static_cast<int>(schedule_->node_to_device.size()); ++i) {
            int dev_idx = schedule_->node_to_device[i];
            if (dev_idx >= 0 && dev_idx < static_cast<int>(schedule_->devices.size())) {
                nodes_[i].device = schedule_->devices[dev_idx].type;
            }
        }
    }

    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        auto& node = nodes_[i];

        if (node.op_type == OpType::NONE && !node.compute_fn) continue;
        if (node.op_type == OpType::CUSTOM && !node.compute_fn) continue;

        node.resolved_inputs.clear();

        for (int idx : node.input_indices) {
            TensorPtr input_tensor;
            if (idx >= 0 && idx < static_cast<int>(inputs_.size())) {
                input_tensor = inputs_[idx];
            } else {
                int node_idx = -idx - 1;
                if (node_idx >= 0 && node_idx < static_cast<int>(nodes_.size())) {
                    input_tensor = nodes_[node_idx].output;
                }
            }

            if (auto_transfer_ && input_tensor) {
                input_tensor = ensure_device(input_tensor, node.device);
            }

            node.resolved_inputs.push_back(input_tensor);
        }

        // Execute
        if (node.compute_fn) {
            node.output = node.compute_fn(node.resolved_inputs);
        } else if (node.op_type != OpType::NONE) {
            node.output = OpDispatch::instance().execute(
                node.op_type, node.device, node.resolved_inputs, node.op_params);
        }

        // Copy to pre-allocated buffer and free temp memory
        if (node.output && graph_buffer_ && graph_buffer_->valid() && planner_) {
            const PlannedAllocation* alloc = planner_->get_allocation(i);
            if (alloc && alloc->size > 0 &&
                alloc->offset + alloc->size <= graph_buffer_->size()) {
                void* planned_ptr = static_cast<char*>(graph_buffer_->data()) + alloc->offset;
                size_t copy_size = std::min(node.output->nbytes(), alloc->size);

                if (copy_size > 0 && planned_ptr != node.output->data()) {
                    if (node.output->device() == DeviceType::CPU) {
                        std::memcpy(planned_ptr, node.output->data(), copy_size);
                    }
#ifdef USE_CUDA
                    else if (node.output->device() == DeviceType::CUDA) {
                        cudaMemcpyAsync(planned_ptr, node.output->data(),
                                        copy_size, cudaMemcpyDeviceToDevice);
                    }
#endif
                    void* old_data = node.output->replace_data(planned_ptr, copy_size);
                    if (old_data) {
                        if (node.output->device() == DeviceType::CPU) {
                            std::free(old_data);
                        }
#ifdef USE_CUDA
                        else {
                            cudaFree(old_data);
                        }
#endif
                    }
                }
            }
        }

        // Track peak memory
        if (node.output) {
            peak_memory_ += node.output->nbytes();
        }

        // Release intermediates
        if (release_intermediates_) {
            for (int idx : node.input_indices) {
                if (idx < 0) {
                    int dep_idx = -idx - 1;
                    ref_count[dep_idx]--;
                    if (ref_count[dep_idx] == 0 && dep_idx < i) {
                        nodes_[dep_idx].output.reset();
                        nodes_[dep_idx].resolved_inputs.clear();
                    }
                }
            }
        }
    }

    return last_output();
}

TensorPtr ComputeGraph::execute_node(int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size())) return nullptr;

    auto& node = nodes_[node_idx];
    node.resolved_inputs.clear();

    for (int idx : node.input_indices) {
        if (idx >= 0 && idx < static_cast<int>(inputs_.size())) {
            auto input_tensor = inputs_[idx];
            if (auto_transfer_ && input_tensor) {
                input_tensor = ensure_device(input_tensor, node.device);
            }
            node.resolved_inputs.push_back(input_tensor);
        } else {
            int dep_idx = -idx - 1;
            if (dep_idx >= 0 && dep_idx < static_cast<int>(nodes_.size())) {
                auto input_tensor = nodes_[dep_idx].output;
                if (auto_transfer_ && input_tensor) {
                    input_tensor = ensure_device(input_tensor, node.device);
                }
                node.resolved_inputs.push_back(input_tensor);
            }
        }
    }

    if (node.compute_fn) {
        node.output = node.compute_fn(node.resolved_inputs);
    } else if (node.op_type != OpType::NONE) {
        node.output = OpDispatch::instance().execute(
            node.op_type, node.device, node.resolved_inputs, node.op_params);
    }

    return node.output;
}

void ComputeGraph::reset() {
    for (auto& node : nodes_) {
        node.output.reset();
        node.resolved_inputs.clear();
    }
}

TensorPtr ComputeGraph::get_output(int node_idx) const {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size())) return nullptr;
    return nodes_[node_idx].output;
}

TensorPtr ComputeGraph::last_output() const {
    if (nodes_.empty()) return nullptr;
    for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
        if (nodes_[i].output) return nodes_[i].output;
    }
    return nullptr;
}

void ComputeGraph::set_workspace_backend(std::shared_ptr<Backend> backend) {
    workspace_backend_ = std::move(backend);
}

Backend* ComputeGraph::workspace_backend() {
    return workspace_backend_ ? workspace_backend_.get() : nullptr;
}

void ComputeGraph::apply_schedule(const SchedulingPlan& plan) {
    if (!plan.valid) return;
    schedule_ = std::make_unique<SchedulingPlan>(plan);

    // Override each node's device based on the plan
    int n_nodes = static_cast<int>(nodes_.size());
    int n_plan = static_cast<int>(plan.node_to_device.size());
    for (int i = 0; i < n_nodes && i < n_plan; ++i) {
        int dev_idx = plan.node_to_device[i];
        if (dev_idx >= 0 && dev_idx < static_cast<int>(plan.devices.size())) {
            nodes_[i].device = plan.devices[dev_idx].type;
        }
    }
}

int ComputeGraph::optimize_fusion() {
    int fused_count = 0;
    for (int i = 0; i < static_cast<int>(nodes_.size()) - 1; ++i) {
        auto& curr = nodes_[i];
        auto& next = nodes_[i + 1];

        // Pattern: rms_norm followed by matmul_transB
        bool curr_is_norm = (curr.op_type == OpType::RMS_NORM) ||
                            (curr.op_type == OpType::CUSTOM &&
                             curr.compute_fn != nullptr);
        bool next_is_matmul = (next.op_type == OpType::MUL_MAT_TRANSB) ||
                               (next.op_type == OpType::CUSTOM &&
                                next.compute_fn != nullptr);

        // For string-based ops, check the name heuristic
        if (curr.op_type == OpType::CUSTOM && curr.compute_fn) {
            curr_is_norm = (curr.name.find("rms_norm") != std::string::npos ||
                            curr.name.find("norm") != std::string::npos);
        }
        if (next.op_type == OpType::CUSTOM && next.compute_fn) {
            next_is_matmul = (next.name.find("matmul") != std::string::npos ||
                              next.name.find("proj") != std::string::npos);
        }

        if (curr_is_norm && next_is_matmul) {
            // Check that next's only input is curr's output
            if (next.input_indices.size() >= 1 && next.input_indices[0] == (-i - 1)) {
                // Fuse: create a combined compute function
                auto norm_fn = curr.compute_fn;
                auto matmul_fn = next.compute_fn;

                if (norm_fn && matmul_fn) {
                    next.compute_fn = [norm_fn, matmul_fn](const std::vector<TensorPtr>& inputs) -> TensorPtr {
                        auto normed = norm_fn(inputs);
                        return matmul_fn({normed});
                    };
                }

                // Update next's inputs to point to curr's inputs
                next.input_indices = curr.input_indices;
                next.name = curr.name + "+" + next.name;
                next.op_type = OpType::CUSTOM;

                // Mark curr as no-op
                curr.compute_fn = nullptr;
                curr.op_type = OpType::NONE;

                fused_count++;
            }
        }
    }

    if (fused_count > 0) {
        LOG_INFO("ComputeGraph: fused " + std::to_string(fused_count) + " op groups");
    }

    return fused_count;
}

GraphBuilder& GraphBuilder::input(const TensorPtr& tensor) {
    inputs_.push_back(tensor);
    return *this;
}

GraphBuilder& GraphBuilder::op(const std::string& name,
                                const std::string& op_type_str,
                                const std::vector<int>& deps,
                                std::function<TensorPtr(const std::vector<TensorPtr>&)> fn) {
    PendingNode node;
    node.name = name;
    node.op_type_str = op_type_str;
    node.input_indices = deps;
    node.compute_fn = std::move(fn);
    pending_nodes_.push_back(std::move(node));
    return *this;
}

std::unique_ptr<ComputeGraph> GraphBuilder::build() {
    auto graph = std::make_unique<ComputeGraph>();

    for (auto& input : inputs_) {
        graph->add_input(input);
    }

    for (auto& node : pending_nodes_) {
        graph->add_node(node.name, node.op_type_str, node.input_indices,
                        std::move(node.compute_fn));
    }

    return graph;
}

} // namespace forge
