#include <cmath>
#include <cstring>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/op_dispatch.h"
#include "forge/operator_norm.h"
#include "forge/perf_profiler.h"

#include "cpu/simd.h"
#include "cpu/common/thread_pool.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {
namespace ops {

TensorPtr rms_norm(const TensorPtr& x, const TensorPtr& weight, float eps) {
    if (x->ndim() != 2)
        throw std::runtime_error("rms_norm expects 2D input");
    int rows = static_cast<int>(x->shape()[0]);
    int cols = static_cast<int>(x->shape()[1]);

    DataType dt = x->dtype();
    DataType wdt = weight ? weight->dtype() : DataType::FP32;
    // Output follows the input dtype (FP16 in -> FP16 out; FP32 in -> FP32 out).
    // The native FP16 path stores/computes in FP16 with an FP32 reduction (6-A-2).
    auto out = std::make_shared<Tensor>(dt, x->shape(), x->device());

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        // Phase 6-A-1 contract gate. RMS_NORM is in op_supports_lowp, so
        // FP16/BF16 pass here; the routing below picks a real kernel or fails
        // loudly (mixed dtypes have no kernel; BF16 retired from require_no_bf16
        // in 6-B-2 with a real __nv_bfloat16 kernel).
        require_dtype_supported("ops::rms_norm/x", OpType::RMS_NORM, dt);
        if (weight) {
            require_dtype_supported("ops::rms_norm/weight", OpType::RMS_NORM, wdt);
        }

        const bool fp16_path = (dt == DataType::FP16) && (!weight || wdt == DataType::FP16);
        const bool bf16_path = (dt == DataType::BF16) && (!weight || wdt == DataType::BF16);
        const bool fp32_path = (dt == DataType::FP32) && (!weight || wdt == DataType::FP32);
        // 6-C-2: native FP8 path requires BOTH x and weight in FP8_E4M3 -- mixed
        // FP8/FP32 has no kernel and must land in the loud-throw below, never in
        // a silent re-cast (weight would be read as uint8_t).
        const bool fp8_path = (dt == DataType::FP8_E4M3) &&
                              (!weight || wdt == DataType::FP8_E4M3);

        if (fp8_path) {
            // 6-C-2: native FP8 RMSNorm -- FP8 storage with an FP32 reduction
            // (forge E4M3 codec; norm_compute stays FP32 while activation is FP8).
            const uint8_t* xp = static_cast<const uint8_t*>(x->data());
            uint8_t* op = static_cast<uint8_t*>(out->data());
            if (weight) {
                cuda::launch_rms_norm_fp8(
                    xp, static_cast<const uint8_t*>(weight->data()), op, rows, cols, eps);
            } else {
                // Dedicated unweighted kernel (same host-pointer discipline as
                // the FP16/BF16 paths: no host buffer may reach the device).
                cuda::launch_rms_norm_unweighted_fp8(xp, op, rows, cols, eps);
            }
        } else if (bf16_path) {
            // 6-B-2: native BF16 RMSNorm -- BF16 storage/arithmetic with an FP32
            // reduction (norm_compute stays FP32 while activation is BF16).
            const __nv_bfloat16* xp = static_cast<const __nv_bfloat16*>(x->data());
            __nv_bfloat16* op = static_cast<__nv_bfloat16*>(out->data());
            if (weight) {
                cuda::launch_rms_norm_bf16(
                    xp, static_cast<const __nv_bfloat16*>(weight->data()), op, rows, cols, eps);
            } else {
                // Dedicated unweighted kernel (same host-pointer discipline as
                // the FP16 path: no host buffer may reach the device).
                cuda::launch_rms_norm_unweighted_bf16(xp, op, rows, cols, eps);
            }
        } else if (fp16_path) {
            const __half* xp = static_cast<const __half*>(x->data());
            __half* op = static_cast<__half*>(out->data());
            if (weight) {
                cuda::launch_rms_norm_fp16(xp, static_cast<const __half*>(weight->data()), op,
                                           rows, cols, eps);
            } else {
                // out = x * rsqrt(mean(x^2) + eps); dedicated FP16 kernel (no host
                // buffer must reach the device -- passing a host pointer would read
                // garbage, the bug that 6-A-2's correctness test caught in the
                // weighted-only path's original ones-vector).
                cuda::launch_rms_norm_unweighted_fp16(xp, op, rows, cols, eps);
            }
        } else if (fp32_path) {
            const float* xp = static_cast<const float*>(x->data());
            float* op = static_cast<float*>(out->data());
            if (weight) {
                cuda::launch_rms_norm(xp, static_cast<const float*>(weight->data()), op, rows,
                                      cols, eps);
            } else {
                // Unweighted: use the dedicated kernel (the original code passed a
                // host `ones` vector to launch_rms_norm, which is a latent device
                // read of host memory).
                cuda::launch_rms_norm_unweighted(xp, op, rows, cols, eps);
            }
        } else {
            throw std::runtime_error(
                std::string("ops::rms_norm: dtype '") + dtype_name(dt) + "' (weight '" +
                dtype_name(wdt) + "') has no native kernel for op '" +
                op_type_name(OpType::RMS_NORM) +
                "'. Native paths exist for FP16 (6-A-2), BF16 (6-B-2) and FP8 E4M3 "
                "(6-C-2, both inputs FP8 only); mixed dtypes fail loudly.");
        }
#endif
    } else {
        PERF_SCOPE("rms_norm/cpu");
        // CPU path is FP32-only: there is no native FP16 CPU kernel in 6-A-2, and
        // falling through to the float path would silently misread FP16 bytes.
        if (dt != DataType::FP32 || (weight && wdt != DataType::FP32)) {
            throw std::runtime_error(
                std::string("ops::rms_norm (CPU): FP32 only; got dtype '") + dtype_name(dt) +
                "'. Use the CUDA device for native FP16 RMSNorm (Phase 6-A-2).");
        }
        // Hybrid CPU+GPU offload (G1): activations stay on the host while layer
        // weights may live on the device. ensure_cpu keeps the CPU-only RMSNorm
        // kernel correct regardless of where the weight is allocated. The CUDA
        // RMSNorm branch above only triggers when x is already on the device, so
        // this never affects the pure-CUDA path.
        const auto xc = ensure_cpu(x);
        const auto wc = weight ? ensure_cpu(weight) : nullptr;
        const float* x_data = static_cast<const float*>(xc->data());
        const float* w_data = wc ? static_cast<const float*>(wc->data()) : nullptr;
        float* o_data = static_cast<float*>(out->data());

        forge::cpu::parallel_for_chunks(rows, 1, [&](int r0, int r1) {
            for (int r = r0; r < r1; ++r) {
                const float* x_row = x_data + r * cols;
                float* o_row = o_data + r * cols;
                forge::cpu::rms_norm_row_f32(x_row, w_data, o_row, cols, eps);
            }
        });
    }
    return out;
}

TensorPtr layer_norm(const TensorPtr& x, const TensorPtr& weight, const TensorPtr& bias,
                     float eps) {
    if (x->ndim() != 2)
        throw std::runtime_error("layer_norm expects 2D input");
    int rows = static_cast<int>(x->shape()[0]);
    int cols = static_cast<int>(x->shape()[1]);

    // Device path: FP32 activations with FP32 (or absent) weight/bias. The CPU
    // reference below copies x D2H, normalizes on the host and copies back --
    // fine for a single layer, ruinous for a vision tower that norms twice per
    // block (a 2304x1024 tower moved ~900 MB per image through the host).
    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        require_dtype_supported("ops::layer_norm/x", OpType::LAYER_NORM, x->dtype());
        if (weight)
            require_dtype_supported("ops::layer_norm/weight", OpType::LAYER_NORM, weight->dtype());
        if (bias)
            require_dtype_supported("ops::layer_norm/bias", OpType::LAYER_NORM, bias->dtype());

        const bool fp32_path = x->dtype() == DataType::FP32 &&
                               (!weight || weight->dtype() == DataType::FP32) &&
                               (!bias || bias->dtype() == DataType::FP32);
        if (fp32_path) {
            // The kernel dereferences weight/bias on the device, so a host-side
            // gamma/beta has to be staged (they are 1-D and tiny).
            TensorPtr w_dev, b_dev;
            const float* w_ptr = nullptr;
            const float* b_ptr = nullptr;
            if (weight) {
                w_dev = (weight->device() == DeviceType::CUDA)
                            ? weight
                            : std::make_shared<Tensor>(DataType::FP32, weight->shape(),
                                                       DeviceType::CUDA);
                if (w_dev != weight)
                    w_dev->copy_from(*weight);
                w_ptr = static_cast<const float*>(w_dev->data());
            }
            if (bias) {
                b_dev = (bias->device() == DeviceType::CUDA)
                            ? bias
                            : std::make_shared<Tensor>(DataType::FP32, bias->shape(),
                                                       DeviceType::CUDA);
                if (b_dev != bias)
                    b_dev->copy_from(*bias);
                b_ptr = static_cast<const float*>(b_dev->data());
            }
            auto out = std::make_shared<Tensor>(DataType::FP32, x->shape(), DeviceType::CUDA);
            PERF_SCOPE("layer_norm");
            cuda::launch_layer_norm(static_cast<const float*>(x->data()), w_ptr, b_ptr,
                                    static_cast<float*>(out->data()), rows, cols, eps);
            return out;
        }
        // FP16/BF16/FP8 activations, or a weight dtype with no kernel: fall
        // through to the reference implementation below, which dequantizes on
        // the host.
#endif
    }

    // Dequantize weight/bias to FP32 CPU first
    auto w_cpu = weight;
    auto b_cpu = bias;
    if (weight && weight->device() == DeviceType::CUDA) {
        w_cpu = std::make_shared<Tensor>(DataType::FP32, weight->shape(), DeviceType::CPU);
        w_cpu->copy_from(*weight);
    }
    if (bias && bias->device() == DeviceType::CUDA) {
        b_cpu = std::make_shared<Tensor>(DataType::FP32, bias->shape(), DeviceType::CPU);
        b_cpu->copy_from(*bias);
    }

    // Dequantize FP16 weights to FP32
    std::vector<float> w_fp32(cols, 1.0f), b_fp32(cols, 0.0f);
    if (w_cpu) {
        if (w_cpu->dtype() == DataType::FP32) {
            std::memcpy(w_fp32.data(), w_cpu->data(), cols * sizeof(float));
        } else if (w_cpu->dtype() == DataType::FP16) {
            const uint16_t* fp16 = static_cast<const uint16_t*>(w_cpu->data());
            for (int i = 0; i < cols; ++i) {
                uint16_t h = fp16[i];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1f;
                uint32_t mant = h & 0x3ff;
                float f;
                if (exp == 0)
                    f = mant == 0 ? 0.0f : std::ldexp(static_cast<float>(mant), -24);
                else if (exp == 31)
                    f = std::numeric_limits<float>::infinity();
                else
                    f = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
                w_fp32[i] = sign ? -f : f;
            }
        }
    }
    if (b_cpu) {
        if (b_cpu->dtype() == DataType::FP32) {
            std::memcpy(b_fp32.data(), b_cpu->data(), cols * sizeof(float));
        } else if (b_cpu->dtype() == DataType::FP16) {
            const uint16_t* fp16 = static_cast<const uint16_t*>(b_cpu->data());
            for (int i = 0; i < cols; ++i) {
                uint16_t h = fp16[i];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1f;
                uint32_t mant = h & 0x3ff;
                float f;
                if (exp == 0)
                    f = mant == 0 ? 0.0f : std::ldexp(static_cast<float>(mant), -24);
                else if (exp == 31)
                    f = std::numeric_limits<float>::infinity();
                else
                    f = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
                b_fp32[i] = sign ? -f : f;
            }
        }
    }

    // Move input to CPU for LayerNorm (no CUDA kernel yet)
    auto x_cpu = x;
    if (x->device() == DeviceType::CUDA) {
        x_cpu = std::make_shared<Tensor>(DataType::FP32, x->shape(), DeviceType::CPU);
        x_cpu->copy_from(*x);
    }

    auto out = std::make_shared<Tensor>(DataType::FP32, x_cpu->shape(), DeviceType::CPU);
    const float* x_data = static_cast<const float*>(x_cpu->data());
    float* o_data = static_cast<float*>(out->data());

    forge::cpu::parallel_for_chunks(rows, 1, [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c)
                mean += x_data[r * cols + c];
            mean /= cols;

            float var = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float d = x_data[r * cols + c] - mean;
                var += d * d;
            }
            var /= cols;

            float inv_std = 1.0f / std::sqrt(var + eps);
            for (int c = 0; c < cols; ++c) {
                o_data[r * cols + c] = (x_data[r * cols + c] - mean) * inv_std * w_fp32[c] + b_fp32[c];
            }
        }
    });

    // Move back to original device if needed
    if (x->device() == DeviceType::CUDA) {
        auto out_gpu = std::make_shared<Tensor>(DataType::FP32, out->shape(), DeviceType::CUDA);
        out_gpu->copy_from(*out);
        return out_gpu;
    }
    return out;
}

}  // namespace ops

// OpDispatch registration
namespace {
static void register_norm_ops() {
    auto& dispatch = OpDispatch::instance();

    // rms_norm: inputs[0]=x, inputs[1]=weight (optional, fallback to 1.0)
    //           op_params[0] = float eps (bit-cast)
    dispatch.register_kernel(
        OpType::RMS_NORM, DeviceType::CPU,
        [](const std::vector<TensorPtr>& inputs, const int32_t* params,
           cudaStream_t /*stream*/) -> TensorPtr {
            float eps = 1e-5f;
            if (params) {
                int32_t eps_bits = params[0];
                std::memcpy(&eps, &eps_bits, sizeof(eps));
            }
            static const bool _auto_register_norm_ops = (register_norm_ops(), true);
            TensorPtr weight;
            if (inputs.size() > 1 && inputs[1]) {
                weight = inputs[1];
            }
            return ops::rms_norm(inputs[0], weight, eps);
        });

    dispatch.register_kernel(
        OpType::RMS_NORM, DeviceType::CUDA,
        [](const std::vector<TensorPtr>& inputs, const int32_t* params,
           cudaStream_t /*stream*/) -> TensorPtr {
            float eps = 1e-5f;
            if (params) {
                int32_t eps_bits = params[0];
                std::memcpy(&eps, &eps_bits, sizeof(eps));
            }
            TensorPtr weight;
            if (inputs.size() > 1 && inputs[1]) {
                weight = inputs[1];
            }
            return ops::rms_norm(inputs[0], weight, eps);
        });
}
static const bool _auto_register_norm_ops = (register_norm_ops(), true);
}  // namespace

}  // namespace forge
