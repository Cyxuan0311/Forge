// =========================================================================
// test_activation_fp8.cpp -- Phase 6-C-3: native FP8 E4M3 activation family
// correctness + micro-benchmark (requires CUDA; skips gracefully w/o GPU).
//
// Covers silu / gelu / gelu_tanh (unary) and the gated variants silu_multiply
// (SiLU-gated, LLaMA/Qwen/DeepSeek FFN) and gelu_multiply (GeGLU, Gemma4 FFN).
// Compute model (same as 6-A-3/6-A-4/6-B-3, storage widened to FP8 E4M3): FP8
// storage + FP32 transcendental compute (erf / tanh / sigmoid) + FP8 output, so
// the only residual vs an FP32 reference built from the *same decoded FP8
// inputs* is the 3-bit-mantissa E4M3 output store rounding.
//
// Correctness: run the real CUDA op (ops::silu / gelu / gelu_tanh /
// silu_multiply / gelu_multiply, which now route FP8 E4M3 to the native
// uint8_t kernels) and compare against an FP32 CPU reference built from the
// same decoded inputs. Structural defects -> O(1) garbage -> fail (max_abs
// gates sized to E4M3 store rounding; cosine is the primary structural gate).
//
// Micro-bench: FP8 vs FP32 kernel timing across two regimes (+ FP16 byte-parity
// sanity arm):
//     small n -> launch-bound (FP8 not generally faster)
//     large n -> bandwidth/memory-efficiency bound (FP8 1B store vs FP32 4B)
// No speedup is preset; measurement decides.
//
// Retirement proof: this test calls all five ops with FP8 E4M3 and expects
// success -- it fails if any routing site still rejects FP8 for SILU / GELU /
// GELU_TANH (see test_dtype_contract fp8 ledger bookkeeping).
// =========================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "forge/compute_graph.h"
#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/fp8_utils.h"
#include "forge/operator_activation.h"
#include "forge/operator_elementwise.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

std::vector<int64_t> shape1(int64_t n) {
    return std::vector<int64_t>{n};
}

float urand(std::mt19937& g, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g);
}

// FP8 E4M3 codec (matches the kernel's hardware cvt convention for |x| <= 240).
TensorPtr make_dev_fp8(const std::vector<uint8_t>& h, int n) {
    auto host = std::make_shared<Tensor>(DataType::FP8_E4M3, shape1(n), DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size());
    auto dev = std::make_shared<Tensor>(DataType::FP8_E4M3, shape1(n), DeviceType::CUDA);
    dev->copy_from(*host);
    return dev;
}

std::vector<uint8_t> fp8_host_bytes(const TensorPtr& dev) {
    auto host = std::make_shared<Tensor>(DataType::FP8_E4M3, dev->shape(), DeviceType::CPU);
    host->copy_from(*dev);
    const uint8_t* p = static_cast<const uint8_t*>(host->data());
    return std::vector<uint8_t>(p, p + static_cast<size_t>(dev->numel()));
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

struct Metrics {
    float max_abs = 0.0f;
    double cosine = 1.0;
};

Metrics compare(const std::vector<float>& got, const std::vector<float>& ref) {
    Metrics m;
    double dot = 0.0, ng = 0.0, nr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::fabs(got[i] - ref[i]);
        m.max_abs = std::max(m.max_abs, diff);
        dot += static_cast<double>(got[i]) * ref[i];
        ng += static_cast<double>(got[i]) * got[i];
        nr += static_cast<double>(ref[i]) * ref[i];
    }
    m.cosine = dot / (std::sqrt(ng) * std::sqrt(nr) + 1e-30);
    return m;
}

// Unary: op(fn) over FP8 input -> FP8 output vs ref_fn(decoded input). The
// reference consumes the DECODED E4M3 input, so input quantization error stays
// out of the metric (only the output store rounding remains).
void check_unary(const char* name, const std::function<TensorPtr(const TensorPtr&)>& fn,
                 const std::function<float(float)>& ref_fn, float max_abs_gate,
                 std::mt19937& rng) {
    const std::vector<int> ns = {8192, 512 * 4096};
    for (int n : ns) {
        std::vector<uint8_t> xb(n);
        std::vector<float> xf(n);
        for (int i = 0; i < n; ++i) {
            float r = urand(rng, -4.0f, 4.0f);
            xb[i] = forge::fp32_to_fp8_e4m3(r);
            xf[i] = forge::fp8_e4m3_to_fp32(xb[i]);  // exact FP8-represented input
        }
        auto out = fn(make_dev_fp8(xb, n));
        if (!out || out->dtype() != DataType::FP8_E4M3) {
            std::printf("FAIL %s: FP8_E4M3 output dtype expected\n", name);
            ++failures;
            return;
        }
        std::vector<uint8_t> ob = fp8_host_bytes(out);
        std::vector<float> got(n), ref(n);
        for (int i = 0; i < n; ++i) {
            got[i] = forge::fp8_e4m3_to_fp32(ob[i]);
            ref[i] = ref_fn(xf[i]);
        }
        Metrics m = compare(got, ref);
        std::printf("  %-12s n=%8d : max_abs=%.3e cosine=%.6f\n", name, n, m.max_abs, m.cosine);
        // cosine is the primary structural gate: a dropped exp / wrong axis /
        // sign flip collapses direction. max_abs catches silu-as-linear etc.;
        // both are sized to E4M3 store rounding (3-bit mantissa), so a legit
        // value sits far below while an O(1) defect sits far above.
        if (m.cosine < 0.998) {
            std::printf("    FAIL %s cosine too low (structural defect?)\n", name);
            ++failures;
        }
        if (m.max_abs > max_abs_gate) {
            std::printf("    FAIL %s max_abs too large (gate=%.3e)\n", name, max_abs_gate);
            ++failures;
        }
    }
}

// Binary gated: silu_multiply / gelu_multiply over FP8 gate,up -> FP8 out.
void check_binary(const char* name,
                  const std::function<TensorPtr(const TensorPtr&, const TensorPtr&)>& fn,
                  const std::function<float(float, float)>& ref_fn, float max_abs_gate,
                  std::mt19937& rng) {
    const std::vector<int> ns = {8192, 512 * 4096};
    for (int n : ns) {
        std::vector<uint8_t> gb(n), ub(n);
        std::vector<float> gf(n), uf(n);
        for (int i = 0; i < n; ++i) {
            float rg = urand(rng, -4.0f, 4.0f);
            float ru = urand(rng, -4.0f, 4.0f);
            gb[i] = forge::fp32_to_fp8_e4m3(rg);
            ub[i] = forge::fp32_to_fp8_e4m3(ru);
            gf[i] = forge::fp8_e4m3_to_fp32(gb[i]);
            uf[i] = forge::fp8_e4m3_to_fp32(ub[i]);
        }
        auto out = fn(make_dev_fp8(gb, n), make_dev_fp8(ub, n));
        if (!out || out->dtype() != DataType::FP8_E4M3) {
            std::printf("FAIL %s: FP8_E4M3 output dtype expected\n", name);
            ++failures;
            return;
        }
        std::vector<uint8_t> ob = fp8_host_bytes(out);
        std::vector<float> got(n), ref(n);
        for (int i = 0; i < n; ++i) {
            got[i] = forge::fp8_e4m3_to_fp32(ob[i]);
            ref[i] = ref_fn(gf[i], uf[i]);
        }
        Metrics m = compare(got, ref);
        std::printf("  %-12s n=%8d : max_abs=%.3e cosine=%.6f\n", name, n, m.max_abs, m.cosine);
        if (m.cosine < 0.998) {
            std::printf("    FAIL %s cosine too low (structural defect?)\n", name);
            ++failures;
        }
        if (m.max_abs > max_abs_gate) {
            std::printf("    FAIL %s max_abs too large (gate=%.3e)\n", name, max_abs_gate);
            ++failures;
        }
    }
}

// Mixed dtypes (FP8 gate x FP32 up) must fail loudly, never mis-cast.
void check_mixed_loud() {
    auto g = make_dev_fp8(std::vector<uint8_t>(64, forge::fp32_to_fp8_e4m3(0.5f)), 64);
    auto u = std::make_shared<Tensor>(DataType::FP32, shape1(64), DeviceType::CUDA);
    bool threw = false;
    try {
        (void)forge::ops::silu_multiply(g, u);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        std::printf("FAIL: mixed FP8 gate + FP32 up did not throw\n");
        ++failures;
    } else {
        std::printf("  mixed FP8 gate + FP32 up: loud reject OK\n");
    }
}

// Single-node graph smoke: FP8 SILU through dst injection, dtype follows,
// bitwise vs eager.
void run_graph_smoke(std::mt19937& rng) {
    const int n = 4096;
    std::vector<uint8_t> xb(n);
    for (int i = 0; i < n; ++i) {
        float r = urand(rng, -4.0f, 4.0f);
        xb[i] = forge::fp32_to_fp8_e4m3(r);
    }
    auto x = make_dev_fp8(xb, n);
    auto ref = forge::ops::silu(x);

    forge::ComputeGraph g;
    int xi = g.add_input(x);
    g.add_node("silu", forge::OpType::SILU, {xi}, nullptr, DeviceType::CUDA);
    TensorPtr out;
    try {
        out = g.execute();
    } catch (const std::exception& e) {
        std::printf("FAIL: graph FP8 SILU threw: %s\n", e.what());
        ++failures;
        return;
    }
    if (!out || out->dtype() != DataType::FP8_E4M3) {
        std::printf("FAIL: graph FP8 SILU wrong dtype\n");
        ++failures;
        return;
    }
    std::vector<uint8_t> ob = fp8_host_bytes(out);
    std::vector<uint8_t> rb = fp8_host_bytes(ref);
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i)
        max_abs = std::max(max_abs, std::fabs(forge::fp8_e4m3_to_fp32(ob[i]) -
                                              forge::fp8_e4m3_to_fp32(rb[i])));
    std::printf("  graph smoke: dtype=%s max_abs_vs_eager=%.3e\n",
                forge::dtype_name(out->dtype()).c_str(), max_abs);
    if (!(max_abs == 0.0f)) {
        std::printf("FAIL: graph FP8 SILU differs from eager (bitwise expected)\n");
        ++failures;
    }
}

// micro-bench: one unary op, FP8 vs FP32 (+ FP16 byte-parity sanity), two regimes.
void bench_unary(const char* name,
                 const std::function<void(const uint8_t*, uint8_t*, int)>& fp8,
                 const std::function<void(const __half*, __half*, int)>& fp16,
                 const std::function<void(const float*, float*, int)>& fp32, int n, int iters) {
    auto x8 = std::make_shared<Tensor>(DataType::FP8_E4M3, shape1(n), DeviceType::CUDA);
    auto o8 = std::make_shared<Tensor>(DataType::FP8_E4M3, shape1(n), DeviceType::CUDA);
    auto x16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto o16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto x32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    std::vector<uint8_t> c8(n, forge::fp32_to_fp8_e4m3(0.5f));
    std::vector<uint16_t> c16(n, 0x3800u);  // 0.5f exactly in FP16
    std::vector<float> c32(n, 0.5f);
    auto h8 = std::make_shared<Tensor>(DataType::FP8_E4M3, shape1(n), DeviceType::CPU);
    auto h16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
    auto h32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CPU);
    std::memcpy(h8->data(), c8.data(), n);
    std::memcpy(h16->data(), c16.data(), n * 2);
    std::memcpy(h32->data(), c32.data(), n * 4);
    x8->copy_from(*h8);
    x16->copy_from(*h16);
    x32->copy_from(*h32);

    for (int i = 0; i < 5; ++i) {
        fp8(static_cast<const uint8_t*>(x8->data()), static_cast<uint8_t*>(o8->data()), n);
        fp16(static_cast<const __half*>(x16->data()), static_cast<__half*>(o16->data()), n);
        fp32(static_cast<const float*>(x32->data()), static_cast<float*>(o32->data()), n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);
    float ms8 = 0, ms16 = 0, ms32 = 0;
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        fp8(static_cast<const uint8_t*>(x8->data()), static_cast<uint8_t*>(o8->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms8, es, ee);
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

    const double b8 = static_cast<double>(n) * 2.0 * 1.0;   // read+write, 1B
    const double b32 = static_cast<double>(n) * 2.0 * 4.0;  // read+write, 4B
    const char* reg = (n <= 8192) ? "launch-bound" : "bandwidth-bound";
    std::printf("bench %s n=%d iters=%d [%s]: FP8 %.4f ms/iter %.1f GB/s | FP16 %.4f ms/iter %.1f GB/s | FP32 %.4f ms/iter %.1f GB/s\n",
                name, n, iters, reg, ms8 / iters, b8 / (ms8 / 1000.0 / iters) / 1e9,
                ms16 / iters, b8 * 2.0 / (ms16 / 1000.0 / iters) / 1e9, ms32 / iters,
                b32 / (ms32 / 1000.0 / iters) / 1e9);
}

// micro-bench: one gated op (reads gate+up, writes 1), three arms, two regimes.
void bench_gated(const char* name,
                 const std::function<void(const uint8_t*, const uint8_t*, uint8_t*, int)>& fp8,
                 const std::function<void(const __half*, const __half*, __half*, int)>& fp16,
                 const std::function<void(const float*, const float*, float*, int)>& fp32, int n,
                 int iters) {
    auto g8 = std::make_shared<Tensor>(DataType::FP8_E4M3, shape1(n), DeviceType::CUDA);
    auto u8 = std::make_shared<Tensor>(DataType::FP8_E4M3, shape1(n), DeviceType::CUDA);
    auto o8 = std::make_shared<Tensor>(DataType::FP8_E4M3, shape1(n), DeviceType::CUDA);
    auto g16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto u16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto o16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto g32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    auto u32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    std::vector<uint8_t> c8(n, forge::fp32_to_fp8_e4m3(0.5f));
    std::vector<uint16_t> c16(n, 0x3800u);
    std::vector<float> c32(n, 0.5f);
    auto h8 = std::make_shared<Tensor>(DataType::FP8_E4M3, shape1(n), DeviceType::CPU);
    auto h16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
    auto h32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CPU);
    std::memcpy(h8->data(), c8.data(), n);
    std::memcpy(h16->data(), c16.data(), n * 2);
    std::memcpy(h32->data(), c32.data(), n * 4);
    g8->copy_from(*h8);
    u8->copy_from(*h8);
    g16->copy_from(*h16);
    u16->copy_from(*h16);
    g32->copy_from(*h32);
    u32->copy_from(*h32);
    for (int i = 0; i < 5; ++i) {
        fp8(static_cast<const uint8_t*>(g8->data()), static_cast<const uint8_t*>(u8->data()),
            static_cast<uint8_t*>(o8->data()), n);
        fp16(static_cast<const __half*>(g16->data()), static_cast<const __half*>(u16->data()),
             static_cast<__half*>(o16->data()), n);
        fp32(static_cast<const float*>(g32->data()), static_cast<const float*>(u32->data()),
             static_cast<float*>(o32->data()), n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);
    float ms8 = 0, ms16 = 0, ms32 = 0;
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        fp8(static_cast<const uint8_t*>(g8->data()), static_cast<const uint8_t*>(u8->data()),
            static_cast<uint8_t*>(o8->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms8, es, ee);
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

    const double b8 = static_cast<double>(n) * 3.0 * 1.0;   // gate+up+out, 1B
    const double b32 = static_cast<double>(n) * 3.0 * 4.0;  // gate+up+out, 4B
    const char* reg = (n <= 8192) ? "launch-bound" : "bandwidth-bound";
    std::printf("bench %s n=%d iters=%d [%s]: FP8 %.4f ms/iter %.1f GB/s | FP16 %.4f ms/iter %.1f GB/s | FP32 %.4f ms/iter %.1f GB/s\n",
                name, n, iters, reg, ms8 / iters, b8 / (ms8 / 1000.0 / iters) / 1e9,
                ms16 / iters, b8 * 2.0 / (ms16 / 1000.0 / iters) / 1e9, ms32 / iters,
                b32 / (ms32 / 1000.0 / iters) / 1e9);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_activation_fp8)\n");
        return 0;
    }

    // Retirement proof: 6-C-3 retired SILU / GELU / GELU_TANH from the
    // op_supports_fp8 ledger (gated variants reuse the SILU / GELU gate).
    if (!forge::op_supports_fp8(forge::OpType::SILU) ||
        !forge::op_supports_fp8(forge::OpType::GELU) ||
        !forge::op_supports_fp8(forge::OpType::GELU_TANH)) {
        std::printf("FAIL: SILU/GELU/GELU_TANH must be retired in op_supports_fp8 (6-C-3)\n");
        return 1;
    }

    std::printf("[6-C-3] FP8 E4M3 activation family correctness vs FP32 reference\n");
    std::mt19937 rng(0xC3AC71);

    // max_abs gates sized to E4M3 store rounding: unary output |out| <= ~4 sits in
    // the [2,4] binade (step 0.25, half-step 0.125); gated output |out| <= ~16 sits
    // in the [8,16] binade (step 1.0, half-step 0.5). Both leave comfortable margin
    // above the legitimate store rounding while still catching O(1) defects.
    check_unary("silu", [](const TensorPtr& x) { return forge::ops::silu(x); }, silu_ref, 0.15f,
                 rng);
    check_unary("gelu", [](const TensorPtr& x) { return forge::ops::gelu(x); }, gelu_ref, 0.15f,
                 rng);
    check_unary("gelu_tanh", [](const TensorPtr& x) { return forge::ops::gelu_tanh(x); },
                gelu_tanh_ref, 0.15f, rng);
    check_binary("silu_multiply",
                 [](const TensorPtr& g, const TensorPtr& u) { return forge::ops::silu_multiply(g, u); },
                 [](float g, float u) { return silu_ref(g) * u; }, 0.60f, rng);
    check_binary("gelu_multiply",
                 [](const TensorPtr& g, const TensorPtr& u) { return forge::ops::gelu_multiply(g, u); },
                 [](float g, float u) { return gelu_ref(g) * u; }, 0.60f, rng);
    check_mixed_loud();
    run_graph_smoke(rng);

    if (failures == 0) {
        std::printf("[6-C-3] micro-bench (FP8 vs FP32 + FP16 sanity, two regimes)\n");
        bench_unary("silu",
                    [](const uint8_t* x, uint8_t* o, int n) { forge::cuda::launch_silu_fp8(x, o, n); },
                    [](const __half* x, __half* o, int n) { forge::cuda::launch_silu_fp16(x, o, n); },
                    [](const float* x, float* o, int n) { forge::cuda::launch_silu(x, o, n); }, 4096,
                    2000);
        bench_unary("silu",
                    [](const uint8_t* x, uint8_t* o, int n) { forge::cuda::launch_silu_fp8(x, o, n); },
                    [](const __half* x, __half* o, int n) { forge::cuda::launch_silu_fp16(x, o, n); },
                    [](const float* x, float* o, int n) { forge::cuda::launch_silu(x, o, n); },
                    512 * 8192, 50);
        bench_gated("silu_multiply",
                    [](const uint8_t* g, const uint8_t* u, uint8_t* o, int n) {
                        forge::cuda::launch_silu_multiply_fp8(g, u, o, n);
                    },
                    [](const __half* g, const __half* u, __half* o, int n) {
                        forge::cuda::launch_silu_multiply_fp16(g, u, o, n);
                    },
                    [](const float* g, const float* u, float* o, int n) {
                        forge::cuda::launch_silu_multiply(g, u, o, n);
                    },
                    4096, 2000);
        bench_gated("silu_multiply",
                    [](const uint8_t* g, const uint8_t* u, uint8_t* o, int n) {
                        forge::cuda::launch_silu_multiply_fp8(g, u, o, n);
                    },
                    [](const __half* g, const __half* u, __half* o, int n) {
                        forge::cuda::launch_silu_multiply_fp16(g, u, o, n);
                    },
                    [](const float* g, const float* u, float* o, int n) {
                        forge::cuda::launch_silu_multiply(g, u, o, n);
                    },
                    512 * 8192, 50);
        std::printf("PASS activation-fp8 (Phase 6-C-3)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
