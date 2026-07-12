#include "nanoinfer/operator_activation.h"
#include "nanoinfer/cuda_kernels.h"
#include "nanoinfer/perf_profiler.h"
#include "nanoinfer/op_dispatch.h"
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nanoinfer {
namespace ops {

TensorPtr silu(const TensorPtr& x) {
    auto out = std::make_shared<Tensor>(DataType::FP32, x->shape(), x->device());
    int n = static_cast<int>(x->numel());

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_silu(
            static_cast<const float*>(x->data()),
            static_cast<float*>(out->data()), n
        );
#endif
    } else {
        PERF_SCOPE("silu/cpu");
        const float* x_data = static_cast<const float*>(x->data());
        float* o_data = static_cast<float*>(out->data());
        for (int i = 0; i < n; ++i) {
            float v = x_data[i];
            o_data[i] = v / (1.0f + std::exp(-v));
        }
    }
    return out;
}

TensorPtr gelu(const TensorPtr& x) {
    auto out = std::make_shared<Tensor>(DataType::FP32, x->shape(), x->device());
    int n = static_cast<int>(x->numel());

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_gelu(
            static_cast<const float*>(x->data()),
            static_cast<float*>(out->data()), n
        );
#endif
    } else {
        PERF_SCOPE("gelu/cpu");
        const float* x_data = static_cast<const float*>(x->data());
        float* o_data = static_cast<float*>(out->data());
        for (int i = 0; i < n; ++i) {
            float v = x_data[i];
            o_data[i] = 0.5f * v * (1.0f + std::erf(v * 0.7071067811865475f));
        }
    }
    return out;
}

TensorPtr gelu_tanh(const TensorPtr& x) {
    auto out = std::make_shared<Tensor>(DataType::FP32, x->shape(), x->device());
    int n = static_cast<int>(x->numel());

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_gelu_tanh(
            static_cast<const float*>(x->data()),
            static_cast<float*>(out->data()), n
        );
#endif
    } else {
        PERF_SCOPE("gelu_tanh/cpu");
        const float* x_data = static_cast<const float*>(x->data());
        float* o_data = static_cast<float*>(out->data());
        for (int i = 0; i < n; ++i) {
            float v = x_data[i];
            // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float cdf = 0.5f * (1.0f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
            o_data[i] = v * cdf;
        }
    }
    return out;
}

} // namespace ops

namespace {
__attribute__((constructor)) void register_activation_ops() {
    auto& dispatch = OpDispatch::instance();

    dispatch.register_kernel(OpType::SILU, DeviceType::CPU,
        [](const std::vector<TensorPtr>& inputs, const int32_t*) -> TensorPtr {
            return ops::silu(inputs[0]);
        });

    dispatch.register_kernel(OpType::SILU, DeviceType::CUDA,
        [](const std::vector<TensorPtr>& inputs, const int32_t*) -> TensorPtr {
            return ops::silu(inputs[0]);
        });

    dispatch.register_kernel(OpType::GELU, DeviceType::CPU,
        [](const std::vector<TensorPtr>& inputs, const int32_t*) -> TensorPtr {
            return ops::gelu(inputs[0]);
        });

    dispatch.register_kernel(OpType::GELU, DeviceType::CUDA,
        [](const std::vector<TensorPtr>& inputs, const int32_t*) -> TensorPtr {
            return ops::gelu(inputs[0]);
        });

    dispatch.register_kernel(OpType::GELU_TANH, DeviceType::CPU,
        [](const std::vector<TensorPtr>& inputs, const int32_t*) -> TensorPtr {
            return ops::gelu_tanh(inputs[0]);
        });

    dispatch.register_kernel(OpType::GELU_TANH, DeviceType::CUDA,
        [](const std::vector<TensorPtr>& inputs, const int32_t*) -> TensorPtr {
            return ops::gelu_tanh(inputs[0]);
        });
}
}

} // namespace nanoinfer
