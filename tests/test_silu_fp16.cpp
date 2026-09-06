// =========================================================================
// test_silu_fp16.cpp -- Phase 6-A-3: native FP16 SiLU correctness +
// micro-benchmark (requires CUDA; skips gracefully when no GPU is present).
//
// Compute model under test (from the 6-A-3 plan):
//     FP16 x -> FP32 widen -> sigmoid(x)=1/(1+exp(-x)) -> x*sigmoid -> FP16 out
// i.e. a native FP16 *storage/arithmetic path* with FP32 transcendental compute,
// NOT "all intermediates in half". Matches the 6-A-2 RMSNorm policy
// (low-precision storage, required FP32 compute) so numerics are preserved.
//
// Correctness: run the real CUDA kernel (via forge::ops::silu, which now routes
// FP16 to launch_silu_fp16) and compare against an FP32 CPU reference built from
// the *same decoded FP16 inputs* -- so the only residual error is the FP16 output
// store, which is exactly what we want to bound. Structural defects produce O(1)
// garbage and fail the tolerance.
//
// Micro-bench: kernel-only launch timing of FP16 vs the existing FP32 kernel,
// reported as ms/iter and achieved bandwidth, across two workload regimes:
//     small  n  -> launch-bound (both near-identical; FP16 not generally faster)
//     large  n  -> bandwidth/memory-efficiency bound (FP16 ~2x from half traffic)
// =========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/operator_activation.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

// ---- fp16 codec (matches the kernel's storage convention) ----------------
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

// FP32 reference SiLU operating on already-decoded FP16 float values.
void silu_ref(const std::vector<float>& x, std::vector<float>& out) {
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}

void fill_const(const TensorPtr& t, float v) {
    auto h = std::make_shared<Tensor>(t->dtype(), t->shape(), DeviceType::CPU);
    const size_t n = static_cast<size_t>(t->numel());
    if (t->dtype() == DataType::FP16) {
        std::vector<uint16_t> buf(n, f32_to_f16(v));
        std::memcpy(h->data(), buf.data(), n * 2);
    } else {
        std::vector<float> buf(n, v);
        std::memcpy(h->data(), buf.data(), n * 4);
    }
    t->copy_from(*h);
}

// ---- correctness: native FP16 SiLU vs FP32 reference --------------------
int run_correctness() {
    int failures = 0;
    std::mt19937 rng(0x5EED);

    // Sanity: 6-A-3 whitelisted SILU in the contract (ties to 6-A-1).
    if (!forge::op_supports_lowp(forge::OpType::SILU)) {
        std::printf("FAIL: op_supports_lowp(SILU) expected true in 6-A-3\n");
        return 1;
    }

    // Several element counts to exercise different block/launch shapes.
    const std::vector<int> ns = {4096, 8192, 512 * 4096, 512 * 8192};

    for (int n : ns) {
        std::vector<uint16_t> xb(n);
        for (auto& b : xb)
            b = f32_to_f16(urand(rng, -3.0f, 3.0f));

        auto x_host = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
        std::memcpy(x_host->data(), xb.data(), xb.size() * 2);
        auto x_dev = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
        x_dev->copy_from(*x_host);

        auto out_dev = forge::ops::silu(x_dev);
        if (!out_dev || out_dev->dtype() != DataType::FP16) {
            std::printf("FAIL: FP16 SiLU returned wrong dtype\n");
            return 1;
        }
        auto out_host = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CPU);
        out_host->copy_from(*out_dev);

        // Reference from the same decoded FP16 inputs.
        std::vector<float> xf(n), ref(n);
        for (int i = 0; i < n; ++i)
            xf[i] = f16_to_f32(xb[i]);
        silu_ref(xf, ref);

        const uint16_t* out_bits = static_cast<const uint16_t*>(out_host->data());
        float max_abs = 0.0f, max_rel = 0.0f;
        for (int i = 0; i < n; ++i) {
            float got = f16_to_f32(out_bits[i]);
            float diff = std::fabs(got - ref[i]);
            if (diff > max_abs)
                max_abs = diff;
            if (std::fabs(ref[i]) > 0.1f) {
                float re = diff / std::fabs(ref[i]);
                if (re > max_rel)
                    max_rel = re;
            }
        }
        std::printf("  n=%8d : max_abs_err=%.3e max_rel_err=%.3e\n", n, max_abs, max_rel);
        if (max_abs > 0.02f) {
            std::printf("    FAIL max_abs_err too large (fp16 store rounding only)\n");
            ++failures;
        }
        if (max_rel > 0.05f) {
            std::printf("    FAIL max_rel_err too large\n");
            ++failures;
        }
    }
    return failures;
}

// ---- micro-bench: FP16 kernel vs FP32 kernel (kernel-only timing) -------
void run_bench(int n, int iters) {
    auto x16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto o16 = std::make_shared<Tensor>(DataType::FP16, shape1(n), DeviceType::CUDA);
    auto x32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, shape1(n), DeviceType::CUDA);
    fill_const(x16, 0.5f);
    fill_const(x32, 0.5f);

    for (int i = 0; i < 5; ++i)
        forge::cuda::launch_silu_fp16(static_cast<const __half*>(x16->data()),
                                      static_cast<__half*>(o16->data()), n);
    for (int i = 0; i < 5; ++i)
        forge::cuda::launch_silu(static_cast<const float*>(x32->data()),
                                 static_cast<float*>(o32->data()), n);
    cudaDeviceSynchronize();

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);
    float ms16 = 0.0f, ms32 = 0.0f;

    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        forge::cuda::launch_silu_fp16(static_cast<const __half*>(x16->data()),
                                      static_cast<__half*>(o16->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms16, es, ee);

    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        forge::cuda::launch_silu(static_cast<const float*>(x32->data()),
                                 static_cast<float*>(o32->data()), n);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms32, es, ee);

    cudaEventDestroy(es);
    cudaEventDestroy(ee);

    // SiLU is elementwise: bytes = 2 * n * element_size (read x + write out).
    const double bytes16 = static_cast<double>(n) * 2.0 * 2.0;  // 2 bytes * (in+out)
    const double bytes32 = static_cast<double>(n) * 2.0 * 4.0;  // 4 bytes * (in+out)
    const char* regime = (n <= 8192) ? "launch-bound" : "bandwidth-bound";
    std::printf("bench n=%d iters=%d [%s]:\n", n, iters, regime);
    std::printf("  FP16: %.4f ms/iter  %.1f GB/s\n", ms16 / iters,
                bytes16 / (ms16 / 1000.0 / iters) / 1e9);
    std::printf("  FP32: %.4f ms/iter  %.1f GB/s\n", ms32 / iters,
                bytes32 / (ms32 / 1000.0 / iters) / 1e9);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_silu_fp16)\n");
        return 0;
    }

    std::printf("[6-A-3] FP16 SiLU correctness vs FP32 reference\n");
    int failures = run_correctness();
    if (failures == 0) {
        std::printf("[6-A-3] micro-bench (FP16 vs FP32, two regimes)\n");
        run_bench(4096, 2000);       // small  -> launch-bound
        run_bench(8192, 1000);       // small  -> launch-bound
        run_bench(512 * 4096, 50);   // large  -> bandwidth-bound
        run_bench(512 * 8192, 50);   // large  -> bandwidth-bound
        std::printf("PASS silu-fp16 (Phase 6-A-3)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
