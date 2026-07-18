#include <cmath>
#include <cstring>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/op_dispatch.h"
#include "forge/operator_norm.h"
#include "forge/perf_profiler.h"

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

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

    auto out = std::make_shared<Tensor>(DataType::FP32, x->shape(), x->device());

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        if (weight) {
            cuda::launch_rms_norm(static_cast<const float*>(x->data()),
                                  static_cast<const float*>(weight->data()),
                                  static_cast<float*>(out->data()), rows, cols, eps);
        } else {
            // rms_norm without weight: output = x * rms(x)
            std::vector<float> ones(cols, 1.0f);
            cuda::launch_rms_norm(static_cast<const float*>(x->data()),
                                  ones.data(),
                                  static_cast<float*>(out->data()), rows, cols, eps);
        }
#endif
    } else {
        PERF_SCOPE("rms_norm/cpu");
        const float* x_data = static_cast<const float*>(x->data());
        const float* w_data = weight ? static_cast<const float*>(weight->data()) : nullptr;
        float* o_data = static_cast<float*>(out->data());

#pragma omp parallel for schedule(static) if (rows > 1)
        for (int r = 0; r < rows; ++r) {
            const float* x_row = x_data + r * cols;
            float* o_row = o_data + r * cols;
#ifdef USE_AVX2
            __m256 sum_sq_v = _mm256_setzero_ps();
            int c = 0;
            for (; c + 8 <= cols; c += 8) {
                __m256 xv = _mm256_loadu_ps(x_row + c);
                sum_sq_v = _mm256_fmadd_ps(xv, xv, sum_sq_v);
            }
            // Horizontal sum
            __m128 hi128 = _mm256_extractf128_ps(sum_sq_v, 1);
            __m128 lo128 = _mm256_castps256_ps128(sum_sq_v);
            __m128 sum128 = _mm_add_ps(lo128, hi128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum_sq = _mm_cvtss_f32(sum128);
            for (; c < cols; ++c) {
                float v = x_row[c];
                sum_sq += v * v;
            }
            float rms = 1.0f / std::sqrt(sum_sq / cols + eps);
            __m256 rms_v = _mm256_set1_ps(rms);
            c = 0;
            if (w_data) {
                for (; c + 8 <= cols; c += 8) {
                    __m256 xv = _mm256_loadu_ps(x_row + c);
                    __m256 wv = _mm256_loadu_ps(w_data + c);
                    __m256 ov = _mm256_mul_ps(_mm256_mul_ps(xv, rms_v), wv);
                    _mm256_storeu_ps(o_row + c, ov);
                }
                for (; c < cols; ++c) {
                    o_row[c] = x_row[c] * rms * w_data[c];
                }
            } else {
                for (; c + 8 <= cols; c += 8) {
                    __m256 xv = _mm256_loadu_ps(x_row + c);
                    _mm256_storeu_ps(o_row + c, _mm256_mul_ps(xv, rms_v));
                }
                for (; c < cols; ++c) {
                    o_row[c] = x_row[c] * rms;
                }
            }
#else
            float sum_sq = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float v = x_row[c];
                sum_sq += v * v;
            }
            float rms = 1.0f / std::sqrt(sum_sq / cols + eps);
            if (w_data) {
                for (int c = 0; c < cols; ++c) {
                    o_row[c] = x_row[c] * rms * w_data[c];
                }
            } else {
                for (int c = 0; c < cols; ++c) {
                    o_row[c] = x_row[c] * rms;
                }
            }
#endif
        }
    }
    return out;
}

TensorPtr layer_norm(const TensorPtr& x, const TensorPtr& weight, const TensorPtr& bias,
                     float eps) {
    if (x->ndim() != 2)
        throw std::runtime_error("layer_norm expects 2D input");
    int rows = static_cast<int>(x->shape()[0]);
    int cols = static_cast<int>(x->shape()[1]);

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

#pragma omp parallel for schedule(static) if (rows > 1)
    for (int r = 0; r < rows; ++r) {
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
        [](const std::vector<TensorPtr>& inputs, const int32_t* params) -> TensorPtr {
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
        [](const std::vector<TensorPtr>& inputs, const int32_t* params) -> TensorPtr {
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
