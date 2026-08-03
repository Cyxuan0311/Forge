#include "forge/op_dispatch.h"

#include <cstdint>
#include <cstring>

#include "forge/logger.h"
#include "forge/quant_traits.h"

namespace forge {

namespace {
const char* device_type_name(DeviceType dev) {
    switch (dev) {
    case DeviceType::CPU:
        return "CPU";
    case DeviceType::CUDA:
        return "CUDA";
    default:
        return "UNKNOWN";
    }
}
}  // namespace

OpDispatch& OpDispatch::instance() {
    static OpDispatch dispatch;
    return dispatch;
}

void OpDispatch::register_kernel(OpType op, DeviceType dev, OpKernelFn kernel) {
    Key key(op, dev);
    kernels_[key] = std::move(kernel);
}

void OpDispatch::register_kernel_dst(OpType op, DeviceType dev, OpKernelFnDst kernel) {
    Key key(op, dev);
    dst_kernels_[key] = std::move(kernel);
}

bool OpDispatch::has_kernel(OpType op, DeviceType dev) const {
    Key key(op, dev);
    return kernels_.find(key) != kernels_.end() || dst_kernels_.find(key) != dst_kernels_.end();
}

bool OpDispatch::has_dst_kernel(OpType op, DeviceType dev) const {
    Key key(op, dev);
    return dst_kernels_.find(key) != dst_kernels_.end();
}

CapabilityResult OpDispatch::supports(OpType op, const std::vector<TensorPtr>& inputs,
                                      const TensorPtr& dst,
                                      const BackendCapabilities& caps) const {
    const std::string op_name = op_type_name(op);

    if (!has_kernel(op, caps.device)) {
        return CapabilityResult::no("no kernel registered for op " + op_name + " on device " +
                                    device_type_name(caps.device));
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& in = inputs[i];
        if (!in)
            continue;
        if (in->device() != caps.device) {
            return CapabilityResult::no(op_name + ": input " + std::to_string(i) + " lives on " +
                                        device_type_name(in->device()) + " but backend is " +
                                        device_type_name(caps.device) +
                                        " (cross-device fallback disabled)");
        }
        if (is_quantized_type(in->dtype()) && !caps.has(BackendCapability::Quantized)) {
            return CapabilityResult::no(op_name + ": input " + std::to_string(i) + " dtype " +
                                        dtype_name(in->dtype()) +
                                        " requires quantized support, backend " +
                                        device_type_name(caps.device) + " lacks it");
        }
        if (in->dtype() == DataType::FP16 && !caps.has(BackendCapability::FP16)) {
            return CapabilityResult::no(op_name + ": input " + std::to_string(i) +
                                        " is FP16 but backend lacks FP16 support");
        }
        if (caps.alignment > 1 && in->data()) {
            auto addr = reinterpret_cast<uintptr_t>(in->data());
            if (addr % caps.alignment != 0) {
                return CapabilityResult::no(op_name + ": input " + std::to_string(i) +
                                            " data is not " + std::to_string(caps.alignment) +
                                            "-byte aligned");
            }
        }
    }

    if (dst) {
        if (dst->device() != caps.device) {
            return CapabilityResult::no(op_name + ": dst lives on " +
                                        device_type_name(dst->device()) + " but backend is " +
                                        device_type_name(caps.device) +
                                        " (cross-device fallback disabled)");
        }
        if (!inputs.empty() && inputs[0] && dst->numel() < inputs[0]->numel() &&
            can_inplace(op)) {
            return CapabilityResult::no(op_name + ": dst has " + std::to_string(dst->numel()) +
                                        " elements, smaller than input 0 (" +
                                        std::to_string(inputs[0]->numel()) + ")");
        }
    }

    return CapabilityResult::yes();
}

CapabilityResult OpDispatch::supports(OpType op, DeviceType dev,
                                      const std::vector<TensorPtr>& inputs,
                                      const TensorPtr& dst) const {
    auto backend = BackendManager::instance().get_backend(dev);
    BackendCapabilities caps;
    if (backend) {
        caps = backend->full_capabilities();
    } else {
        caps.device = dev;
    }
    return supports(op, inputs, dst, caps);
}

TensorPtr OpDispatch::execute(OpType op, DeviceType dev, const std::vector<TensorPtr>& inputs,
                              const int32_t* params) const {
    Key key(op, dev);

    // Prefer dst kernel: allocate output internally, then call dst kernel
    auto dst_it = dst_kernels_.find(key);
    if (dst_it != dst_kernels_.end()) {
        // Infer output shape from first input (works for elementwise/unary ops)
        // For ops with different output shapes, the dst kernel must handle shape inference
        // via params or input inspection.
        TensorPtr dst;
        if (!inputs.empty() && inputs[0]) {
            dst = std::make_shared<Tensor>(DataType::FP32, inputs[0]->shape(), dev);
        } else {
            // Cannot infer shape without inputs — fall through to legacy
            auto legacy_it = kernels_.find(key);
            if (legacy_it != kernels_.end()) {
                return legacy_it->second(inputs, params);
            }
            return nullptr;
        }
        dst_it->second(inputs, dst, params);
        return dst;
    }

    // Legacy kernel
    auto it = kernels_.find(key);
    if (it == kernels_.end()) {
        // Phase 4 hard rule: no cross-device fallback. Report the capability failure.
        LOG_ERROR("OpDispatch::execute: no kernel for op " + op_type_name(op) + " on device " +
                  device_type_name(dev) + " (cross-device fallback disabled)");
        return nullptr;
    }
    return it->second(inputs, params);
}

void OpDispatch::execute_dst(OpType op, DeviceType dev, const std::vector<TensorPtr>& inputs,
                             TensorPtr dst, const int32_t* params) const {
    Key key(op, dev);

    // Prefer dst kernel — direct write, no allocation, no memcpy
    auto dst_it = dst_kernels_.find(key);
    if (dst_it != dst_kernels_.end()) {
        dst_it->second(inputs, dst, params);
        return;
    }

    // Fall back to legacy kernel: execute, then copy result into dst
    auto it = kernels_.find(key);
    if (it != kernels_.end()) {
        TensorPtr result = it->second(inputs, params);
        if (result && dst) {
            dst->copy_from(*result);
        }
        return;
    }

    // Phase 4 hard rule: no cross-device fallback.
    LOG_ERROR("OpDispatch::execute_dst: no kernel for op " + op_type_name(op) + " on device " +
              device_type_name(dev) + " (cross-device fallback disabled)");
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