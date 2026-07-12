#include "forge/op_dispatch.h"

namespace forge {

OpDispatch& OpDispatch::instance() {
    static OpDispatch dispatch;
    return dispatch;
}

void OpDispatch::register_kernel(OpType op, DeviceType dev, OpKernelFn kernel) {
    Key key(op, dev);
    kernels_[key] = std::move(kernel);
}

bool OpDispatch::has_kernel(OpType op, DeviceType dev) const {
    Key key(op, dev);
    return kernels_.find(key) != kernels_.end();
}

TensorPtr OpDispatch::execute(OpType op, DeviceType dev, const std::vector<TensorPtr>& inputs,
                              const int32_t* params) const {
    Key key(op, dev);
    auto it = kernels_.find(key);
    if (it == kernels_.end()) {
        // Fallback: try the other device type (CPU <-> CUDA)
        DeviceType fallback_dev = (dev == DeviceType::CPU) ? DeviceType::CUDA : DeviceType::CPU;
        Key fallback_key(op, fallback_dev);
        auto fallback_it = kernels_.find(fallback_key);
        if (fallback_it == kernels_.end()) {
            return nullptr;
        }
        return fallback_it->second(inputs, params);
    }
    return it->second(inputs, params);
}

bool OpDispatch::can_inplace(OpType op) const {
    // Operations where output can safely overwrite the first input
    switch (op) {
    case OpType::SILU:
    case OpType::GELU:
    case OpType::GELU_TANH:
    case OpType::RELU:
    case OpType::NEG:
    case OpType::SQR:
    case OpType::SQRT:
    case OpType::EXP:
    case OpType::LOG:
    case OpType::SCALE:
    case OpType::RMS_NORM:
    case OpType::LAYER_NORM:
    case OpType::ROPE:
        return true;
    default:
        return false;
    }
}

}  // namespace forge
