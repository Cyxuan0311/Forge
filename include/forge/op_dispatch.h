#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "op_enum.h"
#include "tensor.h"
#include "types.h"

namespace forge {

// Max size of fixed op params (matching ggml's GGML_MAX_OP_PARAMS)
static constexpr int OP_PARAMS_MAX_SIZE = 64;

// Result of a capability query. When `ok` is false, `reason` explains why the backend
// cannot run the op — the caller must NOT silently fall back to another device.
struct CapabilityResult {
    bool ok = true;
    std::string reason;

    explicit operator bool() const { return ok; }

    static CapabilityResult yes() { return CapabilityResult{true, {}}; }
    static CapabilityResult no(std::string why) { return CapabilityResult{false, std::move(why)}; }
};

// ---- Phase 3: dst injection signature ----
// New kernel signature: (inputs, dst, params) -> void
// The kernel writes directly into dst->data(), no internal allocation.
// This eliminates the graph-mode memcpy (compute_graph.cpp:186-214).
using OpKernelFnDst =
    std::function<void(const std::vector<TensorPtr>& inputs, TensorPtr dst, const int32_t* params)>;

// Legacy kernel signature: (inputs, params) -> output tensor
// Kept for backward compatibility. Ops not yet migrated to dst injection use this.
using OpKernelFn =
    std::function<TensorPtr(const std::vector<TensorPtr>& inputs, const int32_t* params)>;

// Per-op, per-device kernel registration
class OpDispatch {
public:
    static OpDispatch& instance();

    // Register legacy kernel (allocates output internally)
    void register_kernel(OpType op, DeviceType dev, OpKernelFn kernel);

    // Register dst-injection kernel (writes into caller-provided dst)
    void register_kernel_dst(OpType op, DeviceType dev, OpKernelFnDst kernel);

    bool has_kernel(OpType op, DeviceType dev) const;
    bool has_dst_kernel(OpType op, DeviceType dev) const;

    // Phase 4: capability query. Decides whether the backend described by `caps` can run
    // `op` with the given inputs/dst. On failure the reason is returned so the caller can
    // report it instead of silently switching device.
    CapabilityResult supports(OpType op, const std::vector<TensorPtr>& inputs,
                              const TensorPtr& dst, const BackendCapabilities& caps) const;

    // Convenience overload: builds caps from the registered backend for `dev`.
    CapabilityResult supports(OpType op, DeviceType dev, const std::vector<TensorPtr>& inputs,
                              const TensorPtr& dst) const;

    // Legacy execute: internally allocates dst, calls dst kernel if available
    TensorPtr execute(OpType op, DeviceType dev, const std::vector<TensorPtr>& inputs,
                      const int32_t* params = nullptr) const;

    // Phase 3: execute with dst injection
    // If a dst kernel is registered, calls it directly (no allocation, no memcpy).
    // If only legacy kernel is registered, calls legacy kernel and copies result into dst.
    void execute_dst(OpType op, DeviceType dev, const std::vector<TensorPtr>& inputs,
                     TensorPtr dst, const int32_t* params = nullptr) const;

    // Check if an op can be done in-place (output can reuse one input's memory)
    bool can_inplace(OpType op) const;

private:
    OpDispatch() = default;

    using Key = std::pair<OpType, DeviceType>;
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return (static_cast<size_t>(k.first) << 16) ^ static_cast<size_t>(k.second);
        }
    };

    std::unordered_map<Key, OpKernelFn, KeyHash> kernels_;
    std::unordered_map<Key, OpKernelFnDst, KeyHash> dst_kernels_;
};

// RAII auto-registration helper
struct OpKernelAutoRegister {
    OpKernelAutoRegister(OpType op, DeviceType dev, OpKernelFn kernel) {
        OpDispatch::instance().register_kernel(op, dev, std::move(kernel));
    }
};

// RAII auto-registration helper for dst-injection kernels
struct OpKernelAutoRegisterDst {
    OpKernelAutoRegisterDst(OpType op, DeviceType dev, OpKernelFnDst kernel) {
        OpDispatch::instance().register_kernel_dst(op, dev, std::move(kernel));
    }
};

// Convenience: register a kernel for both CPU and CUDA with the same function
// (handles dispatching internally)
#define FORGE_REGISTER_OP_KERNEL(op, dev, fn)                                             \
    static ::forge::OpKernelAutoRegister _op_kernel_reg_##op##_##dev(::forge::OpType::op, \
                                                                     ::forge::DeviceType::dev, fn)

// Register a dst-injection kernel
#define FORGE_REGISTER_OP_KERNEL_DST(op, dev, fn)                                        \
    static ::forge::OpKernelAutoRegisterDst _op_kernel_dst_reg_##op##_##dev(             \
        ::forge::OpType::op, ::forge::DeviceType::dev, fn)

// Explicit registration of built-in op kernels.
// Required because op_kernels.cpp lives in a static library where
// static-initialization-based auto-registration can be dropped by the linker.
void register_builtin_op_kernels();

}  // namespace forge