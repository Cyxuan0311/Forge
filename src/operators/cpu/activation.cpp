#include <cmath>

#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/op_dispatch.h"
#include "forge/operator_activation.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {
namespace ops {

TensorPtr silu(const TensorPtr& x) {
    // Output dtype follows the input: a native FP16 path produces FP16, the
    // FP32 path produces FP32. (CPU keeps FP32-only; native FP16 is CUDA,
    // mirroring the 6-A-2 RMSNorm routing.)
    DataType dt = x->dtype();
    auto out = std::make_shared<Tensor>(dt, x->shape(), x->device());
    int n = static_cast<int>(x->numel());

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        // Contract gate (passes for SILU since 6-A-3 whitelisted it; BF16 retired
        // from require_no_bf16 in 6-B-3 with a real __nv_bfloat16 kernel).
        require_dtype_supported("ops::silu/x", OpType::SILU, dt);
        if (dt == DataType::BF16) {
            cuda::launch_silu_bf16(static_cast<const __nv_bfloat16*>(x->data()),
                                   static_cast<__nv_bfloat16*>(out->data()), n);
        } else if (dt == DataType::FP16) {
            cuda::launch_silu_fp16(static_cast<const __half*>(x->data()),
                                   static_cast<__half*>(out->data()), n);
        } else if (dt == DataType::FP32) {
            cuda::launch_silu(static_cast<const float*>(x->data()),
                              static_cast<float*>(out->data()), n);
        } else if (dt == DataType::FP8_E4M3) {
            // 6-C-3: native FP8 SiLU (FP8 storage, FP32 sigmoid, FP8 output).
            cuda::launch_silu_fp8(static_cast<const uint8_t*>(x->data()),
                                  static_cast<uint8_t*>(out->data()), n);
        } else {
            throw std::runtime_error(
                std::string("ops::silu (CUDA): dtype '") + dtype_name(dt) +
                "' has no native kernel. Native paths exist for FP16 (6-A-3), BF16 "
                "(6-B-3) and FP8_E4M3 (6-C-3) only.");
        }
#endif
    } else {
        if (dt != DataType::FP32) {
            throw std::runtime_error(
                std::string("ops::silu (CPU): FP32 only; got dtype '") + dtype_name(dt) +
                "'. Use the CUDA device for native FP16 SiLU (Phase 6-A-3).");
        }
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
    // Output dtype follows the input (mirrors 6-A-3 SiLU routing). Native FP16 is
    // CUDA-only; CPU keeps FP32-only.
    DataType dt = x->dtype();
    auto out = std::make_shared<Tensor>(dt, x->shape(), x->device());
    int n = static_cast<int>(x->numel());

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        require_dtype_supported("ops::gelu/x", OpType::GELU, dt);
        if (dt == DataType::BF16) {
            cuda::launch_gelu_bf16(static_cast<const __nv_bfloat16*>(x->data()),
                                   static_cast<__nv_bfloat16*>(out->data()), n);
        } else if (dt == DataType::FP16) {
            cuda::launch_gelu_fp16(static_cast<const __half*>(x->data()),
                                   static_cast<__half*>(out->data()), n);
        } else if (dt == DataType::FP32) {
            cuda::launch_gelu(static_cast<const float*>(x->data()),
                              static_cast<float*>(out->data()), n);
        } else if (dt == DataType::FP8_E4M3) {
            // 6-C-3: native FP8 GELU (FP8 storage, FP32 erf, FP8 output).
            cuda::launch_gelu_fp8(static_cast<const uint8_t*>(x->data()),
                                  static_cast<uint8_t*>(out->data()), n);
        } else {
            throw std::runtime_error(
                std::string("ops::gelu (CUDA): dtype '") + dtype_name(dt) +
                "' has no native kernel. Native paths exist for FP16 (6-A-4), BF16 "
                "(6-B-3) and FP8_E4M3 (6-C-3) only.");
        }
#endif
    } else {
        if (dt != DataType::FP32) {
            throw std::runtime_error(
                std::string("ops::gelu (CPU): FP32 only; got dtype '") + dtype_name(dt) +
                "'. Use the CUDA device for native FP16 GELU (Phase 6-A-4).");
        }
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
    // Output dtype follows the input. Native FP16 is CUDA-only; CPU keeps FP32.
    DataType dt = x->dtype();
    auto out = std::make_shared<Tensor>(dt, x->shape(), x->device());
    int n = static_cast<int>(x->numel());

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        require_dtype_supported("ops::gelu_tanh/x", OpType::GELU_TANH, dt);
        if (dt == DataType::BF16) {
            cuda::launch_gelu_tanh_bf16(static_cast<const __nv_bfloat16*>(x->data()),
                                        static_cast<__nv_bfloat16*>(out->data()), n);
        } else if (dt == DataType::FP16) {
            cuda::launch_gelu_tanh_fp16(static_cast<const __half*>(x->data()),
                                        static_cast<__half*>(out->data()), n);
        } else if (dt == DataType::FP32) {
            cuda::launch_gelu_tanh(static_cast<const float*>(x->data()),
                                   static_cast<float*>(out->data()), n);
        } else if (dt == DataType::FP8_E4M3) {
            // 6-C-3: native FP8 GELU-tanh (FP8 storage, FP32 tanh, FP8 output).
            cuda::launch_gelu_tanh_fp8(static_cast<const uint8_t*>(x->data()),
                                      static_cast<uint8_t*>(out->data()), n);
        } else {
            throw std::runtime_error(
                std::string("ops::gelu_tanh (CUDA): dtype '") + dtype_name(dt) +
                "' has no native kernel. Native paths exist for FP16 (6-A-4), BF16 "
                "(6-B-3) and FP8_E4M3 (6-C-3) only.");
        }
#endif
    } else {
        if (dt != DataType::FP32) {
            throw std::runtime_error(
                std::string("ops::gelu_tanh (CPU): FP32 only; got dtype '") + dtype_name(dt) +
                "'. Use the CUDA device for native FP16 GELU-tanh (Phase 6-A-4).");
        }
        PERF_SCOPE("gelu_tanh/cpu");
        const float* x_data = static_cast<const float*>(x->data());
        float* o_data = static_cast<float*>(out->data());
        for (int i = 0; i < n; ++i) {
            float v = x_data[i];
            // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float cdf =
                0.5f * (1.0f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
            o_data[i] = v * cdf;
        }
    }
    return out;
}

}  // namespace ops

namespace {
static void register_activation_ops() {
    auto& dispatch = OpDispatch::instance();

    // 6-A-6.4: legacy kernels keep legacy-stream behavior; stream ignored.
    dispatch.register_kernel(OpType::SILU, DeviceType::CPU,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t) -> TensorPtr {
                                 return ops::silu(inputs[0]);
                             });

    dispatch.register_kernel(OpType::SILU, DeviceType::CUDA,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t) -> TensorPtr {
                                 return ops::silu(inputs[0]);
                             });

    dispatch.register_kernel(OpType::GELU, DeviceType::CPU,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t) -> TensorPtr {
                                 return ops::gelu(inputs[0]);
                             });

    dispatch.register_kernel(OpType::GELU, DeviceType::CUDA,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t) -> TensorPtr {
                                 return ops::gelu(inputs[0]);
                             });

    dispatch.register_kernel(OpType::GELU_TANH, DeviceType::CPU,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t) -> TensorPtr {
                                 return ops::gelu_tanh(inputs[0]);
                             });

    dispatch.register_kernel(OpType::GELU_TANH, DeviceType::CUDA,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t) -> TensorPtr {
                                 return ops::gelu_tanh(inputs[0]);
                             });
}
static const bool _auto_register_activation_ops = (register_activation_ops(), true);
}  // namespace

}  // namespace forge
