#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/op_dispatch.h"
#include "forge/operator_activation.h"
#include "forge/operator_elementwise.h"
#include "forge/perf_profiler.h"

#include "cpu/simd.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {
namespace ops {

TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
    auto out = std::make_shared<Tensor>(DataType::FP32, a->shape(), a->device());
    int n = static_cast<int>(a->numel());

    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_add_bias(static_cast<const float*>(a->data()),
                              static_cast<const float*>(b->data()),
                              static_cast<float*>(out->data()), n);
#endif
    } else {
        PERF_SCOPE("add/cpu");
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* o_data = static_cast<float*>(out->data());
        forge::cpu::add_f32_vec(a_data, b_data, o_data, n);
    }
    return out;
}

TensorPtr multiply(const TensorPtr& a, const TensorPtr& b) {
    auto out = std::make_shared<Tensor>(DataType::FP32, a->shape(), a->device());
    int n = static_cast<int>(a->numel());

    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_multiply(static_cast<const float*>(a->data()),
                              static_cast<const float*>(b->data()),
                              static_cast<float*>(out->data()), n);
#endif
    } else {
        PERF_SCOPE("multiply/cpu");
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* o_data = static_cast<float*>(out->data());
        forge::cpu::mul_f32_vec(a_data, b_data, o_data, n);
    }
    return out;
}

TensorPtr silu_multiply(const TensorPtr& gate, const TensorPtr& up) {
    auto out = std::make_shared<Tensor>(DataType::FP32, gate->shape(), gate->device());
    int n = static_cast<int>(gate->numel());

    if (gate->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_silu_multiply(static_cast<const float*>(gate->data()),
                                   static_cast<const float*>(up->data()),
                                   static_cast<float*>(out->data()), n);
#endif
    } else {
        PERF_SCOPE("silu_multiply/cpu");
        const float* g_data = static_cast<const float*>(gate->data());
        const float* u_data = static_cast<const float*>(up->data());
        float* o_data = static_cast<float*>(out->data());
        forge::cpu::silu_mul_f32_vec(g_data, u_data, o_data, n);
    }
    return out;
}

TensorPtr gelu_multiply(const TensorPtr& gate, const TensorPtr& up) {
    auto out = std::make_shared<Tensor>(DataType::FP32, gate->shape(), gate->device());
    int n = static_cast<int>(gate->numel());

    if (gate->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        auto gelu_gate = gelu(gate);
        cuda::launch_multiply(static_cast<const float*>(gelu_gate->data()),
                              static_cast<const float*>(up->data()),
                              static_cast<float*>(out->data()), n);
#endif
    } else {
        PERF_SCOPE("gelu_multiply/cpu");
        const float* g_data = static_cast<const float*>(gate->data());
        const float* u_data = static_cast<const float*>(up->data());
        float* o_data = static_cast<float*>(out->data());
        forge::cpu::gelu_mul_f32_vec(g_data, u_data, o_data, n);
    }
    return out;
}

TensorPtr softmax(const TensorPtr& x, float temperature) {
    // Phase 4 hard rule: no cross-device fallback. There is no CUDA softmax kernel, so a
    // CUDA-resident input is a capability failure instead of silent D2H/H2D staging.
    if (x->device() != DeviceType::CPU) {
        std::string msg =
            "softmax capability failure: input lives on a non-CPU device but there is no "
            "CUDA softmax kernel (cross-device fallback to CPU is disabled)";
        LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }

    auto out = std::make_shared<Tensor>(DataType::FP32, x->shape(), x->device());

    if (x->ndim() != 2)
        throw std::runtime_error("softmax expects 2D input");
    int rows = static_cast<int>(x->shape()[0]);
    int cols = static_cast<int>(x->shape()[1]);

    const float* x_data = static_cast<const float*>(x->data());
    float* o_data = static_cast<float*>(out->data());

    for (int r = 0; r < rows; ++r) {
        float max_val = -1e30f;
        for (int c = 0; c < cols; ++c) {
            max_val = std::max(max_val, x_data[r * cols + c] / temperature);
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            o_data[r * cols + c] = std::exp(x_data[r * cols + c] / temperature - max_val);
            sum += o_data[r * cols + c];
        }
        for (int c = 0; c < cols; ++c) {
            o_data[r * cols + c] /= sum;
        }
    }

    return out;
}

}  // namespace ops

namespace {
static void register_elementwise_ops() {
    auto& dispatch = OpDispatch::instance();

    // ADD: inputs[0] = a, inputs[1] = b
    dispatch.register_kernel(OpType::ADD, DeviceType::CPU,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*) -> TensorPtr {
                                 return ops::add(inputs[0], inputs[1]);
                             });

    dispatch.register_kernel(OpType::ADD, DeviceType::CUDA,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*) -> TensorPtr {
                                 return ops::add(inputs[0], inputs[1]);
                             });

    // MUL: inputs[0] = a, inputs[1] = b
    dispatch.register_kernel(OpType::MUL, DeviceType::CPU,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*) -> TensorPtr {
                                 return ops::multiply(inputs[0], inputs[1]);
                             });

    dispatch.register_kernel(OpType::MUL, DeviceType::CUDA,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*) -> TensorPtr {
                                 return ops::multiply(inputs[0], inputs[1]);
                             });

    // SOFT_MAX: inputs[0] = x, op_params[0] = float temperature (bit-cast)
    dispatch.register_kernel(
        OpType::SOFT_MAX, DeviceType::CPU,
        [](const std::vector<TensorPtr>& inputs, const int32_t* params) -> TensorPtr {
            float temp = 1.0f;
            if (params)
                std::memcpy(&temp, params, sizeof(temp));
            return ops::softmax(inputs[0], temp);
        });
}
static const bool _auto_register_elementwise_ops = (register_elementwise_ops(), true);
}  // namespace

}  // namespace forge
