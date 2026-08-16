#include "forge/compute_graph.h"

#include <algorithm>
#include <cstring>

#include "forge/backend.h"
#include "forge/cuda_mem_pool.h"
#include "forge/host_mem_pool.h"
#include "forge/logger.h"
#include "forge/memory_planner.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

// Helper: get backend name for logging
namespace {
const char* device_name(DeviceType dev) {
    switch (dev) {
    case DeviceType::CPU: return "CPU";
    case DeviceType::CUDA: return "CUDA";
    default: return "?";
    }
}
}  // namespace

int ComputeGraph::add_input(const TensorPtr& tensor) {
    int idx = static_cast<int>(inputs_.size());
    inputs_.push_back(tensor);
    return idx;
}

int ComputeGraph::add_node(const std::string& name, OpType op_type,
                           const std::vector<int>& input_indices, const int32_t* op_params,
                           DeviceType dev) {
    int node_idx = static_cast<int>(nodes_.size());

    GraphNode node;
    node.name = name;
    node.op_type = op_type;
    node.input_indices = input_indices;
    node.device = dev;
    if (op_params) {
        std::copy(op_params, op_params + OP_PARAMS_MAX_SIZE / sizeof(int32_t), node.op_params);
    }

    nodes_.push_back(std::move(node));
    return node_idx;
}

int ComputeGraph::add_node(const std::string& name, const std::string& op_type_str,
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

void ComputeGraph::patch_node_param_i64(int node_idx, int int32_offset, int64_t value) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size()))
        return;
    std::memcpy(nodes_[node_idx].op_params + int32_offset, &value, sizeof(int64_t));
}

TensorPtr ComputeGraph::ensure_device(const TensorPtr& tensor, DeviceType target_dev) {
    if (!tensor || tensor->device() == target_dev)
        return tensor;
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
    if (graph_allocated_)
        return true;

    auto* backend = workspace_backend();
    if (!backend) {
        backend = BackendManager::instance().get_backend(DeviceType::CPU).get();
    }

    planner_ = std::make_unique<MemoryPlanner>();
    planner_->set_backend(backend);
    planner_->plan(*this, release_intermediates_);

    // Phase 3: per-backend buffer compaction.
    // MemoryPlanner produces interleaved offsets across all devices.
    // We remap offsets within each device to eliminate gaps from other-device nodes.
    int n = num_nodes();
    node_buffer_map_.clear();
    node_buffer_map_.resize(n);

    // Collect allocations per device, sorted by original offset (preserves allocation order)
    std::unordered_map<DeviceType, std::vector<std::pair<int, size_t>>> per_dev_allocs;
    for (int i = 0; i < n; ++i) {
        auto* alloc = planner_->get_allocation(i);
        if (!alloc || alloc->alloc_size == 0) continue;
        per_dev_allocs[nodes_[i].device].push_back({i, alloc->offset});
    }

    for (auto& [dev, allocs] : per_dev_allocs) {
        // Sort by original offset (stable order within this device)
        std::sort(allocs.begin(), allocs.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Compact: reassign offsets sequentially, removing gaps
        size_t current_offset = 0;
        for (const auto& [node_idx, orig_offset] : allocs) {
            auto* alloc = planner_->get_allocation(node_idx);
            if (!alloc) continue;
            current_offset = (current_offset + 255) & ~255ULL;  // 256-byte alignment
            node_buffer_map_[node_idx] = {nullptr, current_offset};
            current_offset += alloc->alloc_size;
        }

        size_t buf_size = current_offset;
        if (buf_size == 0) continue;

        auto buf = std::make_unique<GraphBuffer>(
            BackendManager::instance().get_backend(dev), buf_size);
        if (!buf->valid()) {
            LOG_ERROR("ComputeGraph: failed to allocate " + std::to_string(buf_size) +
                      " bytes for " + device_name(dev) + " backend");
            return false;
        }

        // Fill in buffer pointers for all nodes on this device
        for (const auto& [node_idx, orig_offset] : allocs) {
            if (node_buffer_map_[node_idx].offset == SIZE_MAX) continue;
            node_buffer_map_[node_idx].buffer = buf.get();
        }

        graph_buffers_[dev] = std::move(buf);
    }

    if (!graph_buffers_.empty()) {
        LOG_INFO("ComputeGraph: per-backend buffers allocated across " +
                 std::to_string(graph_buffers_.size()) + " backends");
    }

    graph_allocated_ = true;
    return true;
}

void ComputeGraph::release_graph() {
    graph_buffers_.clear();
    node_buffer_map_.clear();
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

        if (node.op_type == OpType::NONE && !node.compute_fn)
            continue;
        if (node.op_type == OpType::CUSTOM && !node.compute_fn)
            continue;

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
            // Phase 3: prefer dst injection when a dst kernel is registered
            bool has_dst = OpDispatch::instance().has_dst_kernel(node.op_type, node.device);
            const PlannedAllocation* alloc =
                (has_dst && planner_) ? planner_->get_allocation(i) : nullptr;

            // Phase 3: per-backend buffer lookup
            NodeBufferMapping* buf_map = nullptr;
            if (i < static_cast<int>(node_buffer_map_.size()) && node_buffer_map_[i].buffer) {
                buf_map = &node_buffer_map_[i];
            }

            if (has_dst && alloc && alloc->size > 0 && buf_map) {
                // Construct dst view into pre-allocated per-backend buffer
                void* planned_ptr =
                    static_cast<char*>(buf_map->buffer->data()) + buf_map->offset;
                DeviceType buf_dev = buf_map->buffer->device();
                auto dst = std::make_shared<Tensor>(Tensor::from_buffer(
                    planned_ptr, DataType::FP32, node.resolved_inputs[0]->shape(), buf_dev, false));
                OpDispatch::instance().execute_dst(node.op_type, node.device, node.resolved_inputs,
                                                   dst, node.op_params);
                node.output = dst;
            } else {
                // Legacy path: execute (allocates internally), then copy to graph buffer
                node.output = OpDispatch::instance().execute(node.op_type, node.device,
                                                             node.resolved_inputs, node.op_params);

                // Copy to pre-allocated per-backend buffer and free temp memory
                if (node.output && buf_map) {
                    const PlannedAllocation* legacy_alloc = planner_->get_allocation(i);
                    if (legacy_alloc && legacy_alloc->size > 0) {
                        void* planned_ptr =
                            static_cast<char*>(buf_map->buffer->data()) + buf_map->offset;
                        size_t copy_size = std::min(node.output->nbytes(), legacy_alloc->size);

                        if (copy_size > 0 && planned_ptr != node.output->data()) {
                            if (node.output->device() == DeviceType::CPU) {
                                std::memcpy(planned_ptr, node.output->data(), copy_size);
                            }
#ifdef USE_CUDA
                            else if (node.output->device() == DeviceType::CUDA) {
                                cudaMemcpyAsync(planned_ptr, node.output->data(), copy_size,
                                                cudaMemcpyDeviceToDevice);
                            }
#endif
                            void* old_data = node.output->replace_data(planned_ptr, copy_size);
                            if (old_data) {
                                if (node.output->device() == DeviceType::CPU) {
                                    host_mem::deallocate(old_data);
                                }
#ifdef USE_CUDA
                                else {
                                    cuda_mem::deallocate(old_data);
                                }
#endif
                            }
                        }
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
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size()))
        return nullptr;

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
        node.output = OpDispatch::instance().execute(node.op_type, node.device,
                                                     node.resolved_inputs, node.op_params);
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
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size()))
        return nullptr;
    return nodes_[node_idx].output;
}

TensorPtr ComputeGraph::last_output() const {
    if (nodes_.empty())
        return nullptr;
    for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
        if (nodes_[i].output)
            return nodes_[i].output;
    }
    return nullptr;
}

void ComputeGraph::set_workspace_backend(std::shared_ptr<Backend> backend) {
    workspace_backend_ = std::move(backend);
}

Backend* ComputeGraph::workspace_backend() {
    return workspace_backend_ ? workspace_backend_.get() : nullptr;
}

int ComputeGraph::insert_copy_nodes() {
    int n = static_cast<int>(nodes_.size());
    if (n == 0) return 0;

    // Collect all cross-device edges: (producer_idx, consumer_idx, input_slot)
    struct CrossDevEdge {
        int producer;
        int consumer;
        int input_slot;
    };
    std::vector<CrossDevEdge> edges;

    for (int i = 0; i < n; ++i) {
        for (size_t j = 0; j < nodes_[i].input_indices.size(); ++j) {
            int ref = nodes_[i].input_indices[j];
            if (ref >= 0) continue;  // graph input
            int prod = -ref - 1;
            if (prod < 0 || prod >= n) continue;
            if (nodes_[prod].device == nodes_[i].device) continue;

            edges.push_back({prod, i, static_cast<int>(j)});
        }
    }

    if (edges.empty()) return 0;

    // Process in reverse consumer order so indices stay valid
    std::sort(edges.begin(), edges.end(),
              [](const CrossDevEdge& a, const CrossDevEdge& b) {
                  return a.consumer > b.consumer;
              });

    int inserted = 0;
    for (const auto& e : edges) {
        DeviceType dst_dev = nodes_[e.consumer].device;
        int copy_idx = add_node(
            "copy_device",
            "copy_device",
            {-(e.producer + 1)},
            [dst_dev](const std::vector<TensorPtr>& inputs) -> TensorPtr {
                auto src = inputs[0];
                auto dst = std::make_shared<Tensor>(src->dtype(), src->shape(), dst_dev);
                dst->copy_from(*src);
                return dst;
            },
            dst_dev  // COPY node runs on the destination (consumer's) device
        );

        // Reroute the consumer to read from the copy node instead
        nodes_[e.consumer].input_indices[e.input_slot] = -(copy_idx + 1);
        inserted++;
    }

    if (inserted > 0) {
        LOG_INFO("ComputeGraph: inserted " + std::to_string(inserted) +
                 " cross-device copy nodes (auto_transfer disabled)");
        auto_transfer_ = false;  // copies are now explicit graph nodes
    }

    return inserted;
}

void ComputeGraph::apply_schedule(const SchedulingPlan& plan) {
    if (!plan.valid)
        return;
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
                            (curr.op_type == OpType::CUSTOM && curr.compute_fn != nullptr);
        bool next_is_matmul = (next.op_type == OpType::MUL_MAT_TRANSB) ||
                              (next.op_type == OpType::CUSTOM && next.compute_fn != nullptr);

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
                    next.compute_fn =
                        [norm_fn, matmul_fn](const std::vector<TensorPtr>& inputs) -> TensorPtr {
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

GraphBuilder& GraphBuilder::op(const std::string& name, const std::string& op_type_str,
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

}  // namespace forge
