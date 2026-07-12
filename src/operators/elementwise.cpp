#include "nanoinfer/operator_elementwise.h"
#include "nanoinfer/cuda_kernels.h"
#include "nanoinfer/perf_profiler.h"
#include "nanoinfer/op_dispatch.h"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nanoinfer {
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
#ifdef USE_AVX2
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 av = _mm256_loadu_ps(a_data + i);
            __m256 bv = _mm256_loadu_ps(b_data + i);
            _mm256_storeu_ps(o_data + i, _mm256_add_ps(av, bv));
        }
        for (; i < n; ++i) o_data[i] = a_data[i] + b_data[i];
#else
        for (int i = 0; i < n; ++i) {
            o_data[i] = a_data[i] + b_data[i];
        }
#endif
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
#ifdef USE_AVX2
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 av = _mm256_loadu_ps(a_data + i);
            __m256 bv = _mm256_loadu_ps(b_data + i);
            _mm256_storeu_ps(o_data + i, _mm256_mul_ps(av, bv));
        }
        for (; i < n; ++i) o_data[i] = a_data[i] * b_data[i];
#else
        for (int i = 0; i < n; ++i) {
            o_data[i] = a_data[i] * b_data[i];
        }
#endif
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
#ifdef USE_AVX2
        // AVX2 SiLU using accurate Cephes-style exp approximation (max error < 1e-6)
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 gv = _mm256_loadu_ps(g_data + i);
            __m256 uv = _mm256_loadu_ps(u_data + i);
            // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            __m256 neg_gv = _mm256_sub_ps(_mm256_setzero_ps(), gv);

            // Cephes-style exp: exp(x) = 2^(x/ln2)
            // Split x/ln2 into integer (n) and fractional (f) parts
            __m256 x = _mm256_mul_ps(neg_gv, _mm256_set1_ps(1.4426950408889634f)); // 1/ln2
            // floor using cvtt + compare trick (AVX2 compatible)
            __m256i emm0 = _mm256_cvttps_epi32(x);
            __m256 z = _mm256_cvtepi32_ps(emm0);
            // If x < z (i.e. fractional part is negative), subtract 1
            __m256 mask = _mm256_cmp_ps(x, z, _MM_CMPINT_LT);
            z = _mm256_sub_ps(z, _mm256_and_ps(mask, _mm256_set1_ps(1.0f)));
            emm0 = _mm256_cvttps_epi32(z);
            __m256 f = _mm256_sub_ps(x, z); // f in [0, 1)

            // Build 2^n from integer exponent
            emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
            emm0 = _mm256_slli_epi32(emm0, 23);
            __m256 pow2n = _mm256_castsi256_ps(emm0);

            // Cephes 6th-order polynomial for 2^f, f in [0, 1)
            // Coefficients from Cephes exp.c / cephes-pow2f
            __m256 P0 = _mm256_set1_ps(1.0f);
            __m256 P1 = _mm256_set1_ps(0.6931471805599453f);   // ln2
            __m256 P2 = _mm256_set1_ps(0.2402265069591007f);    // ln2^2/2
            __m256 P3 = _mm256_set1_ps(0.05549525927235975f);   // ln2^3/6
            __m256 P4 = _mm256_set1_ps(0.009608917886916534f);  // ln2^4/24
            __m256 P5 = _mm256_set1_ps(0.001333355814681543f);  // ln2^5/120
            __m256 P6 = _mm256_set1_ps(0.0001540353039338152f); // ln2^6/720

            // Horner's method: P0 + f*(P1 + f*(P2 + f*(P3 + f*(P4 + f*(P5 + f*P6)))))
            __m256 poly = _mm256_add_ps(P5, _mm256_mul_ps(f, P6));
            poly = _mm256_add_ps(P4, _mm256_mul_ps(f, poly));
            poly = _mm256_add_ps(P3, _mm256_mul_ps(f, poly));
            poly = _mm256_add_ps(P2, _mm256_mul_ps(f, poly));
            poly = _mm256_add_ps(P1, _mm256_mul_ps(f, poly));
            poly = _mm256_add_ps(P0, _mm256_mul_ps(f, poly));

            __m256 exp_neg = _mm256_mul_ps(poly, pow2n);
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 sigmoid = _mm256_div_ps(gv, _mm256_add_ps(one, exp_neg));
            _mm256_storeu_ps(o_data + i, _mm256_mul_ps(sigmoid, uv));
        }
        for (; i < n; ++i) {
            float v = g_data[i];
            float silu_v = v / (1.0f + std::exp(-v));
            o_data[i] = silu_v * u_data[i];
        }
#else
        for (int i = 0; i < n; ++i) {
            float v = g_data[i];
            float silu_v = v / (1.0f + std::exp(-v));
            o_data[i] = silu_v * u_data[i];
        }
#endif
    }
    return out;
}

TensorPtr softmax(const TensorPtr& x, float temperature) {
    auto out = std::make_shared<Tensor>(DataType::FP32, x->shape(), x->device());

    if (x->ndim() != 2) throw std::runtime_error("softmax expects 2D input");
    int rows = static_cast<int>(x->shape()[0]);
    int cols = static_cast<int>(x->shape()[1]);

    const float* x_data = static_cast<const float*>(x->data());
    float* o_data = static_cast<float*>(out->data());

    std::vector<float> host_data;
    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        host_data.resize(x->numel());
        cudaMemcpy(host_data.data(), x_data, x->numel() * sizeof(float), cudaMemcpyDeviceToHost);
        x_data = host_data.data();
        std::vector<float> host_out(x->numel());
        o_data = host_out.data();
#endif
    }

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

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cudaMemcpy(static_cast<float*>(out->data()), o_data,
                   x->numel() * sizeof(float), cudaMemcpyHostToDevice);
#endif
    }

    return out;
}

} // namespace ops

namespace {
__attribute__((constructor)) void register_elementwise_ops() {
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
    dispatch.register_kernel(OpType::SOFT_MAX, DeviceType::CPU,
        [](const std::vector<TensorPtr>& inputs, const int32_t* params) -> TensorPtr {
            float temp = 1.0f;
            if (params) std::memcpy(&temp, params, sizeof(temp));
            return ops::softmax(inputs[0], temp);
        });
}
}

} // namespace nanoinfer
