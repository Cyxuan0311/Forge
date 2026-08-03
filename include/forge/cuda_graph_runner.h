#pragma once

// CudaGraphRunner: CUDA Graph capture, instantiate, update, and replay.
//
// Design (per operator_develop.md Phase 10):
//   - Decode phase uses CUDA Graph to eliminate kernel launch overhead.
//   - Prerequisites: zero cudaMalloc in graph path, stable workspace addresses,
//     stable input buffers, stable stream, trackable dst pointer/shape/stride,
//     kernel capability determined.
//   - Graph compatibility check: no sync ops, no allocation, no CPU ops in graph.
//   - Graph key: node sequence + shape signature.
//   - Node pointer/shape/stride tracking; on change, re-capture.
//   - Fallback to normal execution on incompatibility.
//   - Default off; enable explicitly or after benchmark validation.
//
// Usage flow:
//   1. Caller calls try_replay_or_capture(graph, execute_fn, stream).
//   2. If a captured graph exists and key matches: cudaGraphLaunch (fast path).
//   3. If no captured graph but compatible: first do a normal execute,
//      then cudaStreamBeginCapture -> execute_fn -> cudaStreamEndCapture -> instantiate.
//   4. If incompatible or capture fails: fall back to normal execution.
//   5. On subsequent calls, the captured graph is replayed until key changes.

#ifdef USE_CUDA

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace forge {

class ComputeGraph;

// Per-node tracking for CUDA Graph update detection.
struct CudaGraphNodeSnapshot {
    std::vector<void*> input_ptrs;  // Input tensor data pointers (device)
    void* output_ptr = nullptr;     // Output tensor data pointer (device)
    std::string shape_sig;          // Compact shape representation
};

// Graph key: determines whether the captured graph can be replayed.
struct CudaGraphKey {
    std::string op_sequence;      // Concatenated op types
    std::string shape_signature;  // Concatenated shape signatures

    bool operator==(const CudaGraphKey& o) const {
        return op_sequence == o.op_sequence && shape_signature == o.shape_signature;
    }
    bool operator!=(const CudaGraphKey& o) const { return !(*this == o); }
    bool empty() const { return op_sequence.empty(); }
};

class CudaGraphRunner {
public:
    CudaGraphRunner();
    ~CudaGraphRunner();

    CudaGraphRunner(const CudaGraphRunner&) = delete;
    CudaGraphRunner& operator=(const CudaGraphRunner&) = delete;

    // Check if CUDA Graph can be used with the given ComputeGraph.
    // Returns true if all nodes are CUDA, no sync ops, no allocation in path.
    static bool is_compatible(const ComputeGraph& graph);

    // Main entry point: try replay, or capture if no graph exists.
    // Returns true if CUDA Graph was used (replay or newly captured).
    // Returns false if fallback to normal execution is needed.
    //
    // execute_fn: callable that runs ComputeGraph::execute() on the stream.
    //   The runner wraps this with capture logic.
    // stream: CUDA stream to use (typically 0 for default stream).
    bool try_replay_or_capture(ComputeGraph& graph,
                               std::function<void()> execute_fn,
                               cudaStream_t stream);

    // Invalidate the captured graph. Next call will re-capture.
    void invalidate();

    // Whether a captured graph is available for replay.
    bool has_graph() const { return graph_exec_ != nullptr; }

    // Whether CUDA Graph mode is enabled.
    bool enabled() const { return enabled_; }
    void set_enabled(bool v) { enabled_ = v; }

    // Stats
    int replay_count() const { return replay_count_; }
    int capture_count() const { return capture_count_; }
    int fallback_count() const { return fallback_count_; }

private:
    // Compute the graph key from the current ComputeGraph state.
    CudaGraphKey compute_key(const ComputeGraph& graph) const;

    // Snapshot all node input/output pointers and shapes.
    void snapshot_nodes(const ComputeGraph& graph);

    // Destroy the instantiated graph exec and captured graph.
    void destroy_graph();

    bool enabled_ = false;  // Default off
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    CudaGraphKey cached_key_;
    std::vector<CudaGraphNodeSnapshot> node_snapshots_;
    bool needs_capture_ = true;  // True = next call should capture

    // Stats
    int replay_count_ = 0;
    int capture_count_ = 0;
    int fallback_count_ = 0;
};

}  // namespace forge

#endif  // USE_CUDA
