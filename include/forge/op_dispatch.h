#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "op_enum.h"
#include "tensor.h"
#include "types.h"

namespace forge {

// Max size of fixed op params (matching ggml's GGML_MAX_OP_PARAMS)
static constexpr int OP_PARAMS_MAX_SIZE = 64;

// Kernel signature: (inputs, params) -> output tensor
using OpKernelFn =
    std::function<TensorPtr(const std::vector<TensorPtr>& inputs, const int32_t* params)>;

// Per-op, per-device kernel registration
class OpDispatch {
public:
    static OpDispatch& instance();

    void register_kernel(OpType op, DeviceType dev, OpKernelFn kernel);

    bool has_kernel(OpType op, DeviceType dev) const;

    TensorPtr execute(OpType op, DeviceType dev, const std::vector<TensorPtr>& inputs,
                      const int32_t* params = nullptr) const;

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
};

// RAII auto-registration helper
struct OpKernelAutoRegister {
    OpKernelAutoRegister(OpType op, DeviceType dev, OpKernelFn kernel) {
        OpDispatch::instance().register_kernel(op, dev, std::move(kernel));
    }
};

// Convenience: register a kernel for both CPU and CUDA with the same function
// (handles dispatching internally)
#define FORGE_REGISTER_OP_KERNEL(op, dev, fn)                                             \
    static ::forge::OpKernelAutoRegister _op_kernel_reg_##op##_##dev(::forge::OpType::op, \
                                                                     ::forge::DeviceType::dev, fn)

}  // namespace forge
