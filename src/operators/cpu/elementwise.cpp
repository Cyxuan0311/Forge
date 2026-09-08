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
#include "forge/dtype_contract.h"

#include "cpu/simd.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {
namespace ops {

TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
    // 6-B-5: ADD is retired from require_no_bf16 -- BF16 takes the native
    // kernel below (BF16 storage, FP32 per-element compute, BF16 out).
    // FP16 eager still fails loudly (native FP16 lives in dst-injection only,
    // per the 6-B-1 FP32-only eager discipline); anything else is rejected.
    require_dtype_supported("ops::add/a", OpType::ADD, a->dtype());
    require_dtype_supported("ops::add/b", OpType::ADD, b->dtype());
    if (a->dtype() == DataType::BF16 || b->dtype() == DataType::BF16) {
        if (a->dtype() != DataType::BF16 || b->dtype() != DataType::BF16) {
            throw std::runtime_error(
                std::string("ops::add: BF16 add needs all-BF16 inputs, got '") +
                dtype_name(a->dtype()) + "' / '" + dtype_name(b->dtype()) +
                "' (mixed dtypes fail loudly).");
        }
        auto out = std::make_shared<Tensor>(DataType::BF16, a->shape(), a->device());
        int n = static_cast<int>(a->numel());
        if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            cuda::launch_add_bf16(static_cast<const __nv_bfloat16*>(a->data()),
                                  static_cast<const __nv_bfloat16*>(b->data()),
                                  static_cast<__nv_bfloat16*>(out->data()), n);
#endif
        } else {
            throw std::runtime_error(
                "ops::add (CPU): BF16 has no CPU kernel; use the CUDA device "
                "(Phase 6-B-5).");
        }
        return out;
    }
    if (a->dtype() != DataType::FP32 || b->dtype() != DataType::FP32) {
        throw std::runtime_error(
            std::string("ops::add: dtype '") + dtype_name(a->dtype()) +
            "' has no native kernel. Eager add is FP32-only except BF16 (6-B-5).");
    }
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
    // 6-B-5: MUL is retired from require_no_bf16 -- BF16 takes the native
    // kernel below (BF16 storage, FP32 per-element compute, BF16 out).
    // FP16 eager still fails loudly (dst-injection only, 6-B-1 discipline).
    require_dtype_supported("ops::multiply/a", OpType::MUL, a->dtype());
    require_dtype_supported("ops::multiply/b", OpType::MUL, b->dtype());
    if (a->dtype() == DataType::BF16 || b->dtype() == DataType::BF16) {
        if (a->dtype() != DataType::BF16 || b->dtype() != DataType::BF16) {
            throw std::runtime_error(
                std::string("ops::multiply: BF16 multiply needs all-BF16 inputs, got '") +
                dtype_name(a->dtype()) + "' / '" + dtype_name(b->dtype()) +
                "' (mixed dtypes fail loudly).");
        }
        auto out = std::make_shared<Tensor>(DataType::BF16, a->shape(), a->device());
        int n = static_cast<int>(a->numel());
        if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            cuda::launch_multiply_bf16(static_cast<const __nv_bfloat16*>(a->data()),
                                       static_cast<const __nv_bfloat16*>(b->data()),
                                       static_cast<__nv_bfloat16*>(out->data()), n);
#endif
        } else {
            throw std::runtime_error(
                "ops::multiply (CPU): BF16 has no CPU kernel; use the CUDA device "
                "(Phase 6-B-5).");
        }
        return out;
    }
    if (a->dtype() != DataType::FP32 || b->dtype() != DataType::FP32) {
        throw std::runtime_error(
            std::string("ops::multiply: dtype '") + dtype_name(a->dtype()) +
            "' has no native kernel. Eager multiply is FP32-only except BF16 (6-B-5).");
    }
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
    // Gated activation: SiLU(gate) * up. Output dtype follows gate (== up dtype).
    // SILU is whitelisted in the dtype contract (6-A-3); BF16 retired from
    // require_no_bf16 in 6-B-3 with a real __nv_bfloat16 kernel.
    DataType dt = gate->dtype();
    auto out = std::make_shared<Tensor>(dt, gate->shape(), gate->device());
    int n = static_cast<int>(gate->numel());

    if (gate->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        require_dtype_supported("ops::silu_multiply/gate", OpType::SILU, dt);
        if (dt == DataType::BF16) {
            if (up->dtype() != DataType::BF16) {
                throw std::runtime_error(
                    std::string("ops::silu_multiply (CUDA): BF16 gate needs a BF16 up, got '") +
                    dtype_name(up->dtype()) + "' (mixed dtypes fail loudly).");
            }
            cuda::launch_silu_multiply_bf16(static_cast<const __nv_bfloat16*>(gate->data()),
                                            static_cast<const __nv_bfloat16*>(up->data()),
                                            static_cast<__nv_bfloat16*>(out->data()), n);
        } else if (dt == DataType::FP16) {
            cuda::launch_silu_multiply_fp16(static_cast<const __half*>(gate->data()),
                                            static_cast<const __half*>(up->data()),
                                            static_cast<__half*>(out->data()), n);
        } else if (dt == DataType::FP32) {
            cuda::launch_silu_multiply(static_cast<const float*>(gate->data()),
                                       static_cast<const float*>(up->data()),
                                       static_cast<float*>(out->data()), n);
        } else if (dt == DataType::FP8_E4M3) {
            // 6-C-3: native FP8 SiLU-gate (FP8 storage, FP32 silu, FP8 output).
            if (up->dtype() != DataType::FP8_E4M3) {
                throw std::runtime_error(
                    std::string("ops::silu_multiply (CUDA): FP8 gate needs an FP8 up, got '") +
                    dtype_name(up->dtype()) + "' (mixed dtypes fail loudly).");
            }
            cuda::launch_silu_multiply_fp8(static_cast<const uint8_t*>(gate->data()),
                                           static_cast<const uint8_t*>(up->data()),
                                           static_cast<uint8_t*>(out->data()), n);
        } else {
            throw std::runtime_error(
                std::string("ops::silu_multiply (CUDA): dtype '") + dtype_name(dt) +
                "' has no native kernel. Native paths exist for FP16 (6-A-4), BF16 "
                "(6-B-3) and FP8_E4M3 (6-C-3) only.");
        }
#endif
    } else {
        if (dt != DataType::FP32) {
            throw std::runtime_error(
                std::string("ops::silu_multiply (CPU): FP32 only; got dtype '") + dtype_name(dt) +
                "'. Use the CUDA device for native FP16 SiLU-gate (Phase 6-A-4).");
        }
        PERF_SCOPE("silu_multiply/cpu");
        const float* g_data = static_cast<const float*>(gate->data());
        const float* u_data = static_cast<const float*>(up->data());
        float* o_data = static_cast<float*>(out->data());
        forge::cpu::silu_mul_f32_vec(g_data, u_data, o_data, n);
    }
    return out;
}

TensorPtr gelu_multiply(const TensorPtr& gate, const TensorPtr& up) {
    // Gated activation: GELU(gate) * up (GeGLU). Output dtype follows gate.
    // GELU is whitelisted in the dtype contract (6-A-4); BF16 retired from
    // require_no_bf16 in 6-B-3 with a real __nv_bfloat16 kernel.
    DataType dt = gate->dtype();
    auto out = std::make_shared<Tensor>(dt, gate->shape(), gate->device());
    int n = static_cast<int>(gate->numel());

    if (gate->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        require_dtype_supported("ops::gelu_multiply/gate", OpType::GELU, dt);
        if (dt == DataType::BF16) {
            if (up->dtype() != DataType::BF16) {
                throw std::runtime_error(
                    std::string("ops::gelu_multiply (CUDA): BF16 gate needs a BF16 up, got '") +
                    dtype_name(up->dtype()) + "' (mixed dtypes fail loudly).");
            }
            cuda::launch_gelu_multiply_bf16(static_cast<const __nv_bfloat16*>(gate->data()),
                                            static_cast<const __nv_bfloat16*>(up->data()),
                                            static_cast<__nv_bfloat16*>(out->data()), n);
        } else if (dt == DataType::FP16) {
            cuda::launch_gelu_multiply_fp16(static_cast<const __half*>(gate->data()),
                                            static_cast<const __half*>(up->data()),
                                            static_cast<__half*>(out->data()), n);
        } else if (dt == DataType::FP32) {
            // FP32 path keeps the original two-kernel form (gelu then multiply).
            auto gelu_gate = gelu(gate);
            cuda::launch_multiply(static_cast<const float*>(gelu_gate->data()),
                                  static_cast<const float*>(up->data()),
                                  static_cast<float*>(out->data()), n);
        } else if (dt == DataType::FP8_E4M3) {
            // 6-C-3: native FP8 GeGLU-gate (FP8 storage, FP32 gelu, FP8 output).
            if (up->dtype() != DataType::FP8_E4M3) {
                throw std::runtime_error(
                    std::string("ops::gelu_multiply (CUDA): FP8 gate needs an FP8 up, got '") +
                    dtype_name(up->dtype()) + "' (mixed dtypes fail loudly).");
            }
            cuda::launch_gelu_multiply_fp8(static_cast<const uint8_t*>(gate->data()),
                                           static_cast<const uint8_t*>(up->data()),
                                           static_cast<uint8_t*>(out->data()), n);
        } else {
            throw std::runtime_error(
                std::string("ops::gelu_multiply (CUDA): dtype '") + dtype_name(dt) +
                "' has no native kernel. Native paths exist for FP16 (6-A-4), BF16 "
                "(6-B-3) and FP8_E4M3 (6-C-3) only.");
        }
#endif
    } else {
        if (dt != DataType::FP32) {
            throw std::runtime_error(
                std::string("ops::gelu_multiply (CPU): FP32 only; got dtype '") + dtype_name(dt) +
                "'. Use the CUDA device for native FP16 GeGLU (Phase 6-A-4).");
        }
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
    // 6-A-6.4: legacy kernels keep legacy-stream behavior; stream ignored.
    dispatch.register_kernel(OpType::ADD, DeviceType::CPU,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t) -> TensorPtr {
                                 return ops::add(inputs[0], inputs[1]);
                             });

    dispatch.register_kernel(OpType::ADD, DeviceType::CUDA,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t) -> TensorPtr {
                                 return ops::add(inputs[0], inputs[1]);
                             });

    // MUL: inputs[0] = a, inputs[1] = b
    dispatch.register_kernel(OpType::MUL, DeviceType::CPU,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t) -> TensorPtr {
                                 return ops::multiply(inputs[0], inputs[1]);
                             });

    dispatch.register_kernel(OpType::MUL, DeviceType::CUDA,
                             [](const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t) -> TensorPtr {
                                 return ops::multiply(inputs[0], inputs[1]);
                             });

    // SOFT_MAX: inputs[0] = x, op_params[0] = float temperature (bit-cast)
    dispatch.register_kernel(
        OpType::SOFT_MAX, DeviceType::CPU,
        [](const std::vector<TensorPtr>& inputs, const int32_t* params,
           cudaStream_t) -> TensorPtr {
            float temp = 1.0f;
            if (params)
                std::memcpy(&temp, params, sizeof(temp));
            return ops::softmax(inputs[0], temp);
        });
}
static const bool _auto_register_elementwise_ops = (register_elementwise_ops(), true);
}  // namespace

}  // namespace forge
