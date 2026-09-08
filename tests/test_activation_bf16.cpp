// =========================================================================
// test_activation_bf16.cpp -- Phase 6-B-3: native BF16 activation family
// correctness + micro-benchmark (requires CUDA; skips gracefully w/o GPU).
//
// Covers silu / gelu / gelu_tanh (unary) and the gated variants silu_multiply
// (SiLU-gated, LLaMA/Qwen/DeepSeek FFN) and gelu_multiply (GeGLU, Gemma4 FFN).
// Compute model (same as 6-A-3/6-A-4, storage widened to BF16): BF16 storage +
// FP32 transcendental compute (erf / tanh / sigmoid) + BF16 output, so the only
// residual vs an FP32 reference built from the *same decoded BF16 inputs* is the
// BF16 output store rounding (7-bit mantissa: expect ~4-8x the FP16 residual;
// gated adds one product-rounding step on top).
//
// Correctness: run the real CUDA op (ops::silu / gelu / gelu_tanh /
// silu_multiply / gelu_multiply, which now route BF16 to the native
// __nv_bfloat16 kernels) and compare against an FP32 CPU reference built from
// the same decoded inputs. Structural defects -> O(1) garbage -> fail (fixed
// thresholds 0.02 / 0.05, same as the FP16 family test).
//
// Micro-bench: BF16 vs FP32 kernel timing across two regimes (+ FP16
// byte-parity sanity arm):
//     small n -> launch-bound (BF16 not generally faster)
//     large n -> bandwidth/memory-efficiency bound
// No speedup is preset; measurement decides.
//
// Retirement proof: this test calls all five ops with BF16 and expects
// success -- it fails if any routing site still rejects BF16 for SILU / GELU /
// GELU_TANH (see test_bf16_retirement bookkeeping).
// =========================================================================

#include <cuda_bf16.h>
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
#include "forge/operator_activation.h"
#include "forge/operator_elementwise.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

// ---- bf16 codec (matches the kernel's storage convention) ----------------
uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t rounding = 0x7FFFu + ((x >> 16) & 1u);  // round to nearest-even
    return static_cast<uint16_t>((x + rounding) >> 16);
}

float bf16_to_f32(uint16_t h) {
    uint32_t x = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

std::vector<int64_t> shape1(int64_t n) {
    return std::vector<int64_t>{n};
}

float urand(std::mt19937& g, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g);
}

TensorPtr make_dev_bf16(const std::vector<uint16_t>& h, int64_t n) {
    auto host = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 2);
    auto dev = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CUDA);
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

// Unary: op(fn) over BF16 input -> BF16 output vs ref_fn(decoded input).
void check_unary(const char* name, const std::function<TensorPtr(const TensorPtr&)>& fn,
                 const std::function<float(float)>& ref_fn, std::mt19937& rng) {
    const std::vector<int> ns = {8192, 512 * 4096};
    for (int n : ns) {
        std::vector<uint16_t> xb(n);
        for (auto& b : xb)
            b = f32_to_bf16(urand(rng, -4.0f, 4.0f));
        auto out = fn(make_dev_bf16(xb, n));
        if (!out || out->dtype() != DataType::BF16) {
            std::printf("FAIL %s: BF16 output dtype expected\n", name);
            ++failures;
            return;
        }
        auto host = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CPU);
        host->copy_from(*out);
        const uint16_t* ob = static_cast<const uint16_t*>(host->data());
        float max_abs = 0, max_rel = 0;
        for (int i = 0; i < n; ++i) {
            float got = bf16_to_f32(ob[i]);
            float ref = ref_fn(bf16_to_f32(xb[i]));
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

// Binary gated: silu_multiply / gelu_multiply over BF16 gate,up -> BF16 out.
void check_binary(const char* name,
                  const std::function<TensorPtr(const TensorPtr&, const TensorPtr&)>& fn,
                  const std::function<float(float, float)>& ref_fn, std::mt19937& rng) {
    const std::vector<int> ns = {8192, 512 * 4096};
    for (int n : ns) {
        std::vector<uint16_t> gb(n), ub(n);
        for (auto& b : gb)
            b = f32_to_bf16(urand(rng, -4.0f, 4.0f));
        for (auto& b : ub)
            b = f32_to_bf16(urand(rng, -4.0f, 4.0f));
        auto out = fn(make_dev_bf16(gb, n), make_dev_bf16(ub, n));
        if (!out || out->dtype() != DataType::BF16) {
            std::printf("FAIL %s: BF16 output dtype expected\n", name);
            ++failures;
            return;
        }
        auto host = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CPU);
        host->copy_from(*out);
        const uint16_t* ob = static_cast<const uint16_t*>(host->data());
        float max_abs = 0, max_rel = 0;
        for (int i = 0; i < n; ++i) {
            float got = bf16_to_f32(ob[i]);
            float ref = ref_fn(bf16_to_f32(gb[i]), bf16_to_f32(ub[i]));
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
        // Gated threshold note: gate,up in [-4,4] gives |out| up to ~16, where
        // one BF16 output ULP is 2^-4 = 0.0625. The observed ~3.1e-2 is 0.5 ULP
        // (and max_rel sits exactly on the BF16 epsilon 2^-8), i.e. pure output
        // store rounding -- not an algorithmic deviation. The 0.05 bound below
        // is ~0.8 ULP at max magnitude, still orders of magnitude below what a
        // structural defect (wrong axis / dropped exp / mis-cast) produces.
        if (max_abs > 0.05f) {
            std::printf("    FAIL %s max_abs too large\n", name);
            ++failures;
        }
        if (max_rel > 0.05f) {
            std::printf("    FAIL %s max_rel too large\n", name);
            ++failures;
        }
    }
}

// Mixed dtypes (BF16 gate x FP32 up) must fail loudly, never mis-cast.
void check_mixed_loud() {
    auto g = make_dev_bf16(std::vector<uint16_t>(64, f32_to_bf16(0.5f)), 64);
    auto u = std::make_shared<Tensor>(DataType::FP32, shape1(64), DeviceType::CUDA);
    bool threw = false;
    try {
        (void)forge::ops::silu_multiply(g, u);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        std::printf("FAIL: mixed BF16 gate + FP32 up did not throw\n");
        ++failures;
    } else {
        std::printf("  mixed BF16 gate + FP32 up: loud reject OK\n");
    }
}

// Single-node graph smoke: BF16 SILU through dst injection, dtype follows,
// bitwise vs eager.
void run_graph_smoke(std::mt19937& rng) {
    const int n = 4096;
    std::vector<uint16_t> xb(n);
    for (auto& b : xb)
        b = f32_to_bf16(urand(rng, -4.0f, 4.0f));
    auto x = make_dev_bf16(xb, n);
    auto ref = forge::ops::silu(x);

    forge::ComputeGraph g;
    int xi = g.add_input(x);
    g.add_node("silu", forge::OpType::SILU, {xi}, nullptr, DeviceType::CUDA);
    TensorPtr out;
    try {
        out = g.execute();
    } catch (const std::exception& e) {
        std::printf("FAIL: graph BF16 SILU threw: %s\n", e.what());
        ++failures;
        return;
    }
    if (!out || out->dtype() != DataType::BF16) {
        std::printf("FAIL: graph BF16 SILU wrong dtype\n");
        ++failures;
        return;
    }
    auto oh = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CPU);
    oh->copy_from(*out);
    auto rh = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CPU);
    rh->copy_from(*ref);
    const uint16_t* got = static_cast<const uint16_t*>(oh->data());
    const uint16_t* want = static_cast<const uint16_t*>(rh->data());
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i)
        max_abs = std::max(max_abs, std::fabs(bf16_to_f32(got[i]) - bf16_to_f32(want[i])));
    std::printf("  graph smoke: dtype=%s max_abs_vs_eager=%.3e\n",
                forge::dtype_name(out->dtype()).c_str(), max_abs);
    if (!(max_abs == 0.0f)) {
        std::printf("FAIL: graph BF16 SILU differs from eager (bitwise expected)\n");
        ++failures;
    }
}

struct BenchBufs {
    TensorPtr xb, ob, x16, o16, x32, o32;
};

BenchBufs prep_unary(int n) {
    BenchBufs b;
    b.xb = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CUDA);
    b.ob = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CUDA);
    b.x16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    b.o16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    b.x32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    b.o32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    std::vector<uint16_t> cb(n, f32_to_bf16(0.5f));
    std::vector<float> c32(n, 0.5f);
    auto hb = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CPU);
    auto h32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CPU);
    std::memcpy(hb->data(), cb.data(), n * 2);
    std::memcpy(h32->data(), c32.data(), n * 4);
    b.xb->copy_from(*hb);
    // 0.5f is exactly representable in FP16 (0x3800).
    std::vector<uint16_t> c16(n, 0x3800u);
    auto h16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
    std::memcpy(h16->data(), c16.data(), n * 2);
    b.x16->copy_from(*h16);
    b.x32->copy_from(*h32);
    return b;
}

// micro-bench: one unary op, BF16 vs FP32 (+ FP16 byte-parity sanity), two regimes.
void bench_unary(const char* name,
                 const std::function<void(const __nv_bfloat16*, __nv_bfloat16*, int)>& bf16,
                 const std::function<void(const __half*, __half*, int)>& fp16,
                 const std::function<void(const float*, float*, int)>& fp32, int n, int iters) {
    BenchBufs b = prep_unary(n);
    for (int i = 0; i < 5; ++i) {
        bf16(static_cast<const __nv_bfloat16*>(b.xb->data()),
             static_cast<__nv_bfloat16*>(b.ob->data()), n);
        fp16(static_cast<const __half*>(b.x16->data()), static_cast<__half*>(b.o16->data()), n);
        fp32(static_cast<const float*>(b.x32->data()), static_cast<float*>(b.o32->data()), n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);
    float msbf = 0, ms16 = 0, ms32 = 0;
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        bf16(static_cast<const __nv_bfloat16*>(b.xb->data()),
             static_cast<__nv_bfloat16*>(b.ob->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&msbf, es, ee);
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        fp16(static_cast<const __half*>(b.x16->data()), static_cast<__half*>(b.o16->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms16, es, ee);
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        fp32(static_cast<const float*>(b.x32->data()), static_cast<float*>(b.o32->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms32, es, ee);
    cudaEventDestroy(es);
    cudaEventDestroy(ee);

    const double b16 = static_cast<double>(n) * 2.0 * 2.0;  // read+write, 2B
    const double b32 = static_cast<double>(n) * 2.0 * 4.0;  // read+write, 4B
    const char* reg = (n <= 8192) ? "launch-bound" : "bandwidth-bound";
    std::printf(
        "bench %s n=%d iters=%d [%s]: BF16 %.4f ms/iter %.1f GB/s | FP16 %.4f ms/iter %.1f GB/s | "
        "FP32 %.4f ms/iter %.1f GB/s\n",
        name, n, iters, reg, msbf / iters, b16 / (msbf / 1000.0 / iters) / 1e9, ms16 / iters,
        b16 / (ms16 / 1000.0 / iters) / 1e9, ms32 / iters, b32 / (ms32 / 1000.0 / iters) / 1e9);
}

// micro-bench: one gated op (reads gate+up, writes 1), three arms, two regimes.
void bench_gated(const char* name,
                 const std::function<void(const __nv_bfloat16*, const __nv_bfloat16*,
                                          __nv_bfloat16*, int)>& bf16,
                 const std::function<void(const __half*, const __half*, __half*, int)>& fp16,
                 const std::function<void(const float*, const float*, float*, int)>& fp32, int n,
                 int iters) {
    auto gb = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CUDA);
    auto ub = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CUDA);
    auto ob = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CUDA);
    auto g16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto u16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto o16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto g32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    auto u32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    std::vector<uint16_t> cb(n, f32_to_bf16(0.5f));
    std::vector<uint16_t> c16(n, 0x3800u);
    std::vector<float> c32(n, 0.5f);
    auto hb = std::make_shared<Tensor>(DataType::BF16, shape1(n), DeviceType::CPU);
    auto h16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
    auto h32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CPU);
    std::memcpy(hb->data(), cb.data(), n * 2);
    std::memcpy(h16->data(), c16.data(), n * 2);
    std::memcpy(h32->data(), c32.data(), n * 4);
    gb->copy_from(*hb);
    ub->copy_from(*hb);
    g16->copy_from(*h16);
    u16->copy_from(*h16);
    g32->copy_from(*h32);
    u32->copy_from(*h32);
    for (int i = 0; i < 5; ++i) {
        bf16(static_cast<const __nv_bfloat16*>(gb->data()),
             static_cast<const __nv_bfloat16*>(ub->data()), static_cast<__nv_bfloat16*>(ob->data()),
             n);
        fp16(static_cast<const __half*>(g16->data()), static_cast<const __half*>(u16->data()),
             static_cast<__half*>(o16->data()), n);
        fp32(static_cast<const float*>(g32->data()), static_cast<const float*>(u32->data()),
             static_cast<float*>(o32->data()), n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);
    float msbf = 0, ms16 = 0, ms32 = 0;
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        bf16(static_cast<const __nv_bfloat16*>(gb->data()),
             static_cast<const __nv_bfloat16*>(ub->data()), static_cast<__nv_bfloat16*>(ob->data()),
             n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&msbf, es, ee);
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
    std::printf(
        "bench %s n=%d iters=%d [%s]: BF16 %.4f ms/iter %.1f GB/s | FP16 %.4f ms/iter %.1f GB/s | "
        "FP32 %.4f ms/iter %.1f GB/s\n",
        name, n, iters, reg, msbf / iters, b16 / (msbf / 1000.0 / iters) / 1e9, ms16 / iters,
        b16 / (ms16 / 1000.0 / iters) / 1e9, ms32 / iters, b32 / (ms32 / 1000.0 / iters) / 1e9);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_activation_bf16)\n");
        return 0;
    }

    if (!forge::op_supports_lowp(forge::OpType::SILU) ||
        !forge::op_supports_lowp(forge::OpType::GELU) ||
        !forge::op_supports_lowp(forge::OpType::GELU_TANH)) {
        std::printf("FAIL: SILU/GELU/GELU_TANH must stay whitelisted (6-A-3/4 + 6-B-3)\n");
        return 1;
    }

    std::printf("[6-B-3] BF16 activation family correctness vs FP32 reference\n");
    std::mt19937 rng(0xB16AC7);

    check_unary(
        "silu", [](const TensorPtr& x) { return forge::ops::silu(x); }, silu_ref, rng);
    check_unary(
        "gelu", [](const TensorPtr& x) { return forge::ops::gelu(x); }, gelu_ref, rng);
    check_unary(
        "gelu_tanh", [](const TensorPtr& x) { return forge::ops::gelu_tanh(x); }, gelu_tanh_ref,
        rng);
    check_binary(
        "silu_multiply",
        [](const TensorPtr& g, const TensorPtr& u) { return forge::ops::silu_multiply(g, u); },
        [](float g, float u) { return silu_ref(g) * u; }, rng);
    check_binary(
        "gelu_multiply",
        [](const TensorPtr& g, const TensorPtr& u) { return forge::ops::gelu_multiply(g, u); },
        [](float g, float u) { return gelu_ref(g) * u; }, rng);
    check_mixed_loud();
    run_graph_smoke(rng);

    if (failures == 0) {
        std::printf("[6-B-3] micro-bench (BF16 vs FP32 + FP16 sanity, two regimes)\n");
        bench_unary(
            "silu",
            [](const __nv_bfloat16* x, __nv_bfloat16* o, int n) {
                forge::cuda::launch_silu_bf16(x, o, n);
            },
            [](const __half* x, __half* o, int n) { forge::cuda::launch_silu_fp16(x, o, n); },
            [](const float* x, float* o, int n) { forge::cuda::launch_silu(x, o, n); }, 4096, 2000);
        bench_unary(
            "silu",
            [](const __nv_bfloat16* x, __nv_bfloat16* o, int n) {
                forge::cuda::launch_silu_bf16(x, o, n);
            },
            [](const __half* x, __half* o, int n) { forge::cuda::launch_silu_fp16(x, o, n); },
            [](const float* x, float* o, int n) { forge::cuda::launch_silu(x, o, n); }, 512 * 8192,
            50);
        bench_gated(
            "silu_multiply",
            [](const __nv_bfloat16* g, const __nv_bfloat16* u, __nv_bfloat16* o, int n) {
                forge::cuda::launch_silu_multiply_bf16(g, u, o, n);
            },
            [](const __half* g, const __half* u, __half* o, int n) {
                forge::cuda::launch_silu_multiply_fp16(g, u, o, n);
            },
            [](const float* g, const float* u, float* o, int n) {
                forge::cuda::launch_silu_multiply(g, u, o, n);
            },
            4096, 2000);
        bench_gated(
            "silu_multiply",
            [](const __nv_bfloat16* g, const __nv_bfloat16* u, __nv_bfloat16* o, int n) {
                forge::cuda::launch_silu_multiply_bf16(g, u, o, n);
            },
            [](const __half* g, const __half* u, __half* o, int n) {
                forge::cuda::launch_silu_multiply_fp16(g, u, o, n);
            },
            [](const float* g, const float* u, float* o, int n) {
                forge::cuda::launch_silu_multiply(g, u, o, n);
            },
            512 * 8192, 50);
        std::printf("PASS activation-bf16 (Phase 6-B-3)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
