#include "forge/cuda_graph_runner.h"

#ifdef USE_CUDA

#include <sstream>

#include "forge/compute_graph.h"
#include "forge/logger.h"
#include "forge/op_enum.h"

namespace forge {

// ---- Op compatibility table ----
// Ops safe for CUDA Graph capture: no synchronization, no allocation,
// no host-side control flow that would break capture.
static bool is_graph_safe_op(OpType op) {
    switch (op) {
    // Shape/data movement — safe (no sync, no alloc when dst-injected)
    case OpType::VIEW:
    case OpType::PERMUTE:
    case OpType::RESHAPE:
    case OpType::TRANSPOSE:
    case OpType::GET_ROWS:
    case OpType::CPY:
    case OpType::CONT:
    case OpType::REPEAT:
    case OpType::CONCAT:
    case OpType::PAD:
    // Unary element-wise — safe
    case OpType::SILU:
    case OpType::GELU:
    case OpType::GELU_TANH:
    case OpType::RELU:
    case OpType::NEG:
    case OpType::SQR:
    case OpType::SQRT:
    case OpType::EXP:
    case OpType::LOG:
    // Binary element-wise — safe
    case OpType::ADD:
    case OpType::SUB:
    case OpType::MUL:
    case OpType::DIV:
    case OpType::SCALE:
    // Normalization — safe (no sync)
    case OpType::RMS_NORM:
    case OpType::LAYER_NORM:
    case OpType::GROUP_NORM:
    // Matrix multiply — safe (kernel launch only)
    case OpType::MUL_MAT:
    case OpType::MUL_MAT_TRANSB:
    case OpType::MUL_MAT_ID:
    case OpType::OUT_PROD:
    // Attention — safe (kernel launch only)
    case OpType::SOFT_MAX:
    case OpType::SOFT_MAX_MASKED:
    case OpType::FLASH_ATTN_EXT:
    case OpType::FLASH_ATTN_GQA:
    // Position encoding — safe
    case OpType::ROPE:
    // Embedding — safe
    case OpType::EMBEDDING:
    // GLU — safe
    case OpType::GLU:
    // KV cache ops — safe when using fixed-address dst injection
    case OpType::CPY_K:
    case OpType::CPY_V:
    case OpType::GET_K:
    case OpType::GET_V:
        return true;

    // Reduction — may involve synchronization; conservatively unsafe
    case OpType::SUM:
    case OpType::MEAN:
    case OpType::ARGMAX:
        return false;

    // NONE / CUSTOM — unsafe by default (may have closures with sync/alloc)
    case OpType::NONE:
    case OpType::CUSTOM:
    case OpType::COUNT:
        return false;
    }
    return false;
}

// ---- CudaGraphRunner ----

CudaGraphRunner::CudaGraphRunner() = default;

CudaGraphRunner::~CudaGraphRunner() {
    destroy_graph();
}

void CudaGraphRunner::destroy_graph() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    cached_key_ = CudaGraphKey{};
    node_snapshots_.clear();
    needs_capture_ = true;
}

void CudaGraphRunner::invalidate() {
    destroy_graph();
}

bool CudaGraphRunner::is_compatible(const ComputeGraph& graph) {
    for (int i = 0; i < graph.num_nodes(); ++i) {
        const auto& node = graph.node(i);

        // Skip NONE nodes (fused/no-op)
        if (node.op_type == OpType::NONE)
            continue;

        // Must be on CUDA
        if (node.device != DeviceType::CUDA)
            return false;

        // Closure-based nodes are unsafe: they may allocate, sync, or
        // use host-side control flow.
        if (node.compute_fn)
            return false;

        // Check op compatibility
        if (!is_graph_safe_op(node.op_type))
            return false;
    }
    return true;
}

CudaGraphKey CudaGraphRunner::compute_key(const ComputeGraph& graph) const {
    CudaGraphKey key;
    std::ostringstream ops, shapes;

    for (int i = 0; i < graph.num_nodes(); ++i) {
        const auto& node = graph.node(i);

        // Op sequence
        ops << static_cast<uint32_t>(node.op_type) << ",";

        // Shape signature: for each input and output, record shape dimensions
        for (const auto& input : node.resolved_inputs) {
            if (input) {
                for (auto d : input->shape())
                    shapes << d << "x";
                shapes << ";";
            }
        }
        if (node.output) {
            for (auto d : node.output->shape())
                shapes << d << "x";
            shapes << "|";
        }
        shapes << ",";
    }

    key.op_sequence = ops.str();
    key.shape_signature = shapes.str();
    return key;
}

void CudaGraphRunner::snapshot_nodes(const ComputeGraph& graph) {
    node_snapshots_.clear();
    node_snapshots_.resize(graph.num_nodes());

    for (int i = 0; i < graph.num_nodes(); ++i) {
        const auto& node = graph.node(i);
        auto& snap = node_snapshots_[i];

        // Snapshot input pointers
        snap.input_ptrs.clear();
        for (const auto& input : node.resolved_inputs) {
            snap.input_ptrs.push_back(input ? input->data() : nullptr);
        }

        // Snapshot output pointer and shape
        snap.output_ptr = node.output ? node.output->data() : nullptr;
        snap.shape_sig.clear();
        if (node.output) {
            for (auto d : node.output->shape())
                snap.shape_sig += std::to_string(d) + "x";
        }
    }
}

bool CudaGraphRunner::try_replay_or_capture(ComputeGraph& graph,
                                            std::function<void()> execute_fn,
                                            cudaStream_t stream) {
    if (!enabled_) {
        return false;
    }

    // --- Fast path: replay existing captured graph ---
    if (has_graph() && !needs_capture_) {
        // Check if key still matches (same topology)
        CudaGraphKey cur_key = compute_key(graph);
        if (cur_key == cached_key_) {
            // Replay the captured graph
            cudaError_t err = cudaGraphLaunch(graph_exec_, stream);
            if (err == cudaSuccess) {
                replay_count_++;
                return true;
            }

            // Replay failed — destroy and re-capture on next call
            LOG_WARN("CudaGraphRunner: cudaGraphLaunch failed: " +
                     std::string(cudaGetErrorString(err)) + ", destroying graph");
            fallback_count_++;
            destroy_graph();
            return false;
        }

        // Key changed — need re-capture
        LOG_INFO("CudaGraphRunner: key mismatch, re-capturing");
        destroy_graph();
        // Fall through to capture
    }

    // --- Check compatibility before capture ---
    if (!is_compatible(graph)) {
        // Not compatible — don't attempt capture, let caller use normal execution
        return false;
    }

    // --- Capture path ---
    // We use cudaStreamCaptureModeGlobal which captures all work on the stream.
    // All CUDA kernels launched during execute_fn() will be captured into the graph.

    // Step 1: Begin stream capture
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        LOG_WARN("CudaGraphRunner: cudaStreamBeginCapture failed: " +
                 std::string(cudaGetErrorString(err)));
        fallback_count_++;
        return false;
    }

    // Step 2: Execute the graph normally — all kernel launches are captured
    execute_fn();

    // Step 3: End stream capture
    cudaGraph_t captured_graph = nullptr;
    err = cudaStreamEndCapture(stream, &captured_graph);
    if (err != cudaSuccess) {
        LOG_WARN("CudaGraphRunner: cudaStreamEndCapture failed: " +
                 std::string(cudaGetErrorString(err)));
        fallback_count_++;
        return false;
    }

    if (!captured_graph) {
        LOG_WARN("CudaGraphRunner: captured graph is null (no work captured)");
        fallback_count_++;
        return false;
    }

    // Step 4: Destroy old graph if any
    destroy_graph();
    graph_ = captured_graph;

    // Step 5: Instantiate (CUDA 12+ API: 3 args — pGraphExec, graph, flags)
    err = cudaGraphInstantiate(&graph_exec_, graph_, 0);
    if (err != cudaSuccess) {
        LOG_WARN("CudaGraphRunner: cudaGraphInstantiate failed: " +
                 std::string(cudaGetErrorString(err)));
        destroy_graph();
        fallback_count_++;
        return false;
    }

    // Step 6: Snapshot key and node state
    cached_key_ = compute_key(graph);
    snapshot_nodes(graph);
    needs_capture_ = false;

    capture_count_++;
    LOG_INFO("CudaGraphRunner: captured and instantiated graph with " +
             std::to_string(graph.num_nodes()) + " nodes (capture #" +
             std::to_string(capture_count_) + ")");

    return true;
}

}  // namespace forge

#endif  // USE_CUDA
