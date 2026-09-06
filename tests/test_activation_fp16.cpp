// =========================================================================
// test_activation_fp16.cpp -- Phase 6-A-4: native FP16 activation family
// correctness + micro-benchmark (requires CUDA; skips gracefully w/o GPU).
//
// Covers the real main-chain activations: GELU, GELU-tanh (unary) and the gated
// variants silu_multiply (SiLU-gated, LLaMA/Qwen/DeepSeek FFN) and gelu_multiply
// (GeGLU, Gemma4 FFN). Compute model (same as 6-A-2/6-A-3): FP16 storage +
// FP32 transcendental compute (erf / tanh / sigmoid) + FP16 output, so the only
// residual vs an FP32 reference built from the *same decoded FP16 inputs* is the
// FP16 output store rounding.
//
// Correctness: run the real CUDA op (ops::gelu / gelu_tanh / silu_multiply /
// gelu_multiply, which now route FP16 to the native __half kernels) and compare
// against an FP32 CPU reference. Structural defects -> O(1) garbage -> fail.
//
// Micro-bench: FP16 vs FP32 kernel timing across two regimes:
//     small n -> launch-bound (FP16 not generally faster)
//     large n -> bandwidth/memory-efficiency bound (FP16 ~2x from half traffic)
// =========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/operator_activation.h"
#include "forge/operator_elementwise.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

// ---- fp16 codec (matches the kernel storage convention) ----------------
uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0)
        return static_cast<uint16_t>(sign);
    if (exp >= 31)
        return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    float s = sign ? -1.0f : 1.0f;
    if (exp == 0)
        return (mant == 0) ? 0.0f : std::ldexp(static_cast<float>(mant), -24) * s;
    if (exp == 31)
        return std::numeric_limits<float>::infinity() * s;
    return std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25) * s;
}

std::vector<int64_t> shape1(int64_t n) {
    return std::vector<int64_t>{n};
}

float urand(std::mt19937& g, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g);
}

TensorPtr make_dev_fp16(const std::vector<uint16_t>& h, int64_t n) {
    auto host = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 2);
    auto dev = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    dev->copy_from(*host);
    return dev;
}

// FP32 reference activations (erf/tanh/sigmoid in float -- matches the kernels).
inline float gelu_ref(float v) {
    return 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
}
inline float gelu_tanh_ref(float v) {
    return v * 0.5f * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
inline float silu_ref(float v) {
    return v / (1.0f + expf(-v));
}

int failures = 0;

// Unary: op(fn) over FP16 input -> FP16 output vs ref_fn(decoded input).
void check_unary(const char* name, const std::function<TensorPtr(const TensorPtr&)>& fn,
                 const std::function<float(float)>& ref_fn, std::mt19937& rng) {
    const std::vector<int> ns = {8192, 512 * 4096};
    for (int n : ns) {
        std::vector<uint16_t> xb(n);
        for (auto& b : xb)
            b = f32_to_f16(urand(rng, -4.0f, 4.0f));
        auto out = fn(make_dev_fp16(xb, n));
        if (!out || out->dtype() != DataType::FP16) {
            std::printf("FAIL %s: FP16 output dtype expected\n", name);
            ++failures;
            return;
        }
        auto host = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
        host->copy_from(*out);
        const uint16_t* ob = static_cast<const uint16_t*>(host->data());
        float max_abs = 0, max_rel = 0;
        for (int i = 0; i < n; ++i) {
            float got = f16_to_f32(ob[i]);
            float ref = ref_fn(f16_to_f32(xb[i]));
            float diff = std::fabs(got - ref);
            if (diff > max_abs)
                max_abs = diff;
            if (std::fabs(ref) > 0.1f) {
                float re = diff / std::fabs(ref);
                if (re > max_rel)
                    max_rel = re;
            }
        }
        std::printf("  %-12s n=%8d : max_abs=%.3e max_rel=%.3e\n", name, n, max_abs, max_rel);
        if (max_abs > 0.02f) {
            std::printf("    FAIL %s max_abs too large\n", name);
            ++failures;
        }
        if (max_rel > 0.05f) {
            std::printf("    FAIL %s max_rel too large\n", name);
            ++failures;
        }
    }
}

// Binary gated: silu_multiply / gelu_multiply over FP16 gate,up -> FP16 out.
void check_binary(const char* name,
                  const std::function<TensorPtr(const TensorPtr&, const TensorPtr&)>& fn,
                  const std::function<float(float, float)>& ref_fn, std::mt19937& rng) {
    const std::vector<int> ns = {8192, 512 * 4096};
    for (int n : ns) {
        std::vector<uint16_t> gb(n), ub(n);
        for (auto& b : gb)
            b = f32_to_f16(urand(rng, -4.0f, 4.0f));
        for (auto& b : ub)
            b = f32_to_f16(urand(rng, -4.0f, 4.0f));
        auto out = fn(make_dev_fp16(gb, n), make_dev_fp16(ub, n));
        if (!out || out->dtype() != DataType::FP16) {
            std::printf("FAIL %s: FP16 output dtype expected\n", name);
            ++failures;
            return;
        }
        auto host = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
        host->copy_from(*out);
        const uint16_t* ob = static_cast<const uint16_t*>(host->data());
        float max_abs = 0, max_rel = 0;
        for (int i = 0; i < n; ++i) {
            float got = f16_to_f32(ob[i]);
            float ref = ref_fn(f16_to_f32(gb[i]), f16_to_f32(ub[i]));
            float diff = std::fabs(got - ref);
            if (diff > max_abs)
                max_abs = diff;
            if (std::fabs(ref) > 0.1f) {
                float re = diff / std::fabs(ref);
                if (re > max_rel)
                    max_rel = re;
            }
        }
        std::printf("  %-12s n=%8d : max_abs=%.3e max_rel=%.3e\n", name, n, max_abs, max_rel);
        if (max_abs > 0.02f) {
            std::printf("    FAIL %s max_abs too large\n", name);
            ++failures;
        }
        if (max_rel > 0.05f) {
            std::printf("    FAIL %s max_rel too large\n", name);
            ++failures;
        }
    }
}

// micro-bench: one unary op, FP16 vs FP32, two regimes.
void bench_unary(const char* name, const std::function<void(const __half*, __half*, int)>& fp16,
                 const std::function<void(const float*, float*, int)>& fp32, int n, int iters) {
    auto x16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto o16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto x32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    std::vector<uint16_t> c16(n, f32_to_f16(0.5f));
    std::vector<float> c32(n, 0.5f);
    auto h16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
    auto h32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CPU);
    std::memcpy(h16->data(), c16.data(), n * 2);
    std::memcpy(h32->data(), c32.data(), n * 4);
    x16->copy_from(*h16);
    x32->copy_from(*h32);
    for (int i = 0; i < 5; ++i)
        fp16(static_cast<const __half*>(x16->data()), static_cast<__half*>(o16->data()), n);
    for (int i = 0; i < 5; ++i)
        fp32(static_cast<const float*>(x32->data()), static_cast<float*>(o32->data()), n);
    cudaDeviceSynchronize();

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);
    float ms16 = 0, ms32 = 0;
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        fp16(static_cast<const __half*>(x16->data()), static_cast<__half*>(o16->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms16, es, ee);
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        fp32(static_cast<const float*>(x32->data()), static_cast<float*>(o32->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms32, es, ee);
    cudaEventDestroy(es);
    cudaEventDestroy(ee);

    const double b16 = static_cast<double>(n) * 2.0 * 2.0;  // read+write, 2B
    const double b32 = static_cast<double>(n) * 2.0 * 4.0;  // read+write, 4B
    const char* reg = (n <= 8192) ? "launch-bound" : "bandwidth-bound";
    std::printf("bench %s n=%d iters=%d [%s]: FP16 %.4f ms/iter %.1f GB/s | FP32 %.4f ms/iter %.1f GB/s\n",
                name, n, iters, reg, ms16 / iters, b16 / (ms16 / 1000.0 / iters) / 1e9, ms32 / iters,
                b32 / (ms32 / 1000.0 / iters) / 1e9);
}

// micro-bench: one gated op (reads gate+up, writes 1), FP16 vs FP32, two regimes.
void bench_gated(const char* name,
                 const std::function<void(const __half*, const __half*, __half*, int)>& fp16,
                 const std::function<void(const float*, const float*, float*, int)>& fp32, int n,
                 int iters) {
    auto g16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto u16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto o16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto g32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    auto u32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    std::vector<uint16_t> c16(n, f32_to_f16(0.5f));
    std::vector<float> c32(n, 0.5f);
    auto h16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
    auto h32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CPU);
    std::memcpy(h16->data(), c16.data(), n * 2);
    std::memcpy(h32->data(), c32.data(), n * 4);
    g16->copy_from(*h16);
    u16->copy_from(*h16);
    g32->copy_from(*h32);
    u32->copy_from(*h32);
    for (int i = 0; i < 5; ++i)
        fp16(static_cast<const __half*>(g16->data()), static_cast<const __half*>(u16->data()),
             static_cast<__half*>(o16->data()), n);
    for (int i = 0; i < 5; ++i)
        fp32(static_cast<const float*>(g32->data()), static_cast<const float*>(u32->data()),
             static_cast<float*>(o32->data()), n);
    cudaDeviceSynchronize();

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);
    float ms16 = 0, ms32 = 0;
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        fp16(static_cast<const __half*>(g16->data()), static_cast<const __half*>(u16->data()),
             static_cast<__half*>(o16->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms16, es, ee);
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        fp32(static_cast<const float*>(g32->data()), static_cast<const float*>(u32->data()),
             static_cast<float*>(o32->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms32, es, ee);
    cudaEventDestroy(es);
    cudaEventDestroy(ee);

    const double b16 = static_cast<double>(n) * 3.0 * 2.0;  // gate+up+out, 2B
    const double b32 = static_cast<double>(n) * 3.0 * 4.0;  // gate+up+out, 4B
    const char* reg = (n <= 8192) ? "launch-bound" : "bandwidth-bound";
    std::printf("bench %s n=%d iters=%d [%s]: FP16 %.4f ms/iter %.1f GB/s | FP32 %.4f ms/iter %.1f GB/s\n",
                name, n, iters, reg, ms16 / iters, b16 / (ms16 / 1000.0 / iters) / 1e9, ms32 / iters,
                b32 / (ms32 / 1000.0 / iters) / 1e9);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_activation_fp16)\n");
        return 0;
    }

    std::printf("[6-A-4] FP16 activation family correctness vs FP32 reference\n");
    std::mt19937 rng(0x6A4F);

    check_unary("gelu", [](const TensorPtr& x) { return forge::ops::gelu(x); }, gelu_ref, rng);
    check_unary("gelu_tanh", [](const TensorPtr& x) { return forge::ops::gelu_tanh(x); }, gelu_tanh_ref,
                rng);
    check_binary("silu_multiply",
                 [](const TensorPtr& g, const TensorPtr& u) { return forge::ops::silu_multiply(g, u); },
                 [](float g, float u) { return silu_ref(g) * u; }, rng);
    check_binary("gelu_multiply",
                 [](const TensorPtr& g, const TensorPtr& u) { return forge::ops::gelu_multiply(g, u); },
                 [](float g, float u) { return gelu_ref(g) * u; }, rng);

    if (failures == 0) {
        std::printf("[6-A-4] micro-bench (FP16 vs FP32, two regimes)\n");
        bench_unary("gelu",
                    [](const __half* x, __half* o, int n) { forge::cuda::launch_gelu_fp16(x, o, n); },
                    [](const float* x, float* o, int n) { forge::cuda::launch_gelu(x, o, n); }, 4096,
                    2000);
        bench_unary("gelu",
                    [](const __half* x, __half* o, int n) { forge::cuda::launch_gelu_fp16(x, o, n); },
                    [](const float* x, float* o, int n) { forge::cuda::launch_gelu(x, o, n); },
                    512 * 8192, 50);
        bench_gated("silu_multiply",
                    [](const __half* g, const __half* u, __half* o, int n) {
                        forge::cuda::launch_silu_multiply_fp16(g, u, o, n);
                    },
                    [](const float* g, const float* u, float* o, int n) {
                        forge::cuda::launch_silu_multiply(g, u, o, n);
                    },
                    4096, 2000);
        bench_gated("silu_multiply",
                    [](const __half* g, const __half* u, __half* o, int n) {
                        forge::cuda::launch_silu_multiply_fp16(g, u, o, n);
                    },
                    [](const float* g, const float* u, float* o, int n) {
                        forge::cuda::launch_silu_multiply(g, u, o, n);
                    },
                    512 * 8192, 50);
        std::printf("PASS activation-fp16 (Phase 6-A-4)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
