// =========================================================================
// test_matmul_fp16_chain.cpp -- Phase 6-A-5.3: real native FP16 compute chain
//
//     FP16 input
//        -> Native FP16 RMSNorm      (6-A-2)
//        -> Native FP16 Matmul       (6-A-5)
//        -> Native FP16 activation    (6-A-4: silu_multiply)
//        -> FP16 output
//
// This is the first end-to-end FP16 storage -> FP16 compute -> Tensor Core ->
// FP16 output chain Forge can run. The test asserts the **dtype invariant** at
// every stage (no intermediate tensor is silently promoted to FP32) and checks
// correctness against a verified FP32 reference chain built from the same decoded
// FP16 inputs.
//
// Requires CUDA; skips gracefully without a GPU.
// =========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cstring>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/operator_activation.h"
#include "forge/operator_elementwise.h"
#include "forge/operator_matmul.h"
#include "forge/operator_norm.h"
#include "forge/operator_matmul.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;
using namespace forge::cuda;  // launch_cublas_gemm_fp16 (used via ops route, kept for parity)

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
std::vector<int64_t> shape2(int64_t r, int64_t c) {
    return std::vector<int64_t>{r, c};
}

TensorPtr make_dev_fp16(const std::vector<uint16_t>& h, int64_t r, int64_t c) {
    auto host = std::make_shared<Tensor>(DataType::FP16, shape2(r, c), DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 2);
    auto dev = std::make_shared<Tensor>(DataType::FP16, shape2(r, c), DeviceType::CUDA);
    dev->copy_from(*host);
    return dev;
}
TensorPtr make_host_fp32(const std::vector<float>& h, int64_t r, int64_t c) {
    auto t = std::make_shared<Tensor>(DataType::FP32, shape2(r, c), DeviceType::CPU);
    std::memcpy(t->data(), h.data(), h.size() * 4);
    return t;
}

int failures = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                    \
        }                                                                 \
    } while (0)

void rand_fp16(int64_t n, float lo, float hi, std::mt19937& g, std::vector<uint16_t>& codes,
               std::vector<float>& decoded) {
    std::uniform_real_distribution<float> d(lo, hi);
    codes.resize(n);
    decoded.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        float v = d(g);
        codes[i] = f32_to_f16(v);
        decoded[i] = f16_to_f32(codes[i]);
    }
}

// Run the chain in FP16 on CUDA, assert every stage stays FP16, compare output
// to the verified FP32 chain on the same decoded inputs.
void check_chain(int M, int K, int N, std::mt19937& g) {
    std::vector<uint16_t> x_code, b_code, up_code, w_code;
    std::vector<float> x_dec, b_dec, up_dec, w_dec;
    rand_fp16((int64_t)M * K, -2.0f, 2.0f, g, x_code, x_dec);
    rand_fp16((int64_t)N * K, -1.0f, 1.0f, g, b_code, b_dec);
    rand_fp16((int64_t)M * N, -1.0f, 1.0f, g, up_code, up_dec);
    // RMSNorm weight [1,K] in [0.9,1.1] (the realistic, weighted path).
    rand_fp16((int64_t)K, 0.9f, 1.1f, g, w_code, w_dec);

    auto x_dev = make_dev_fp16(x_code, M, K);
    auto b_dev = make_dev_fp16(b_code, N, K);
    auto up_dev = make_dev_fp16(up_code, M, N);
    auto w_dev = make_dev_fp16(w_code, 1, K);  // broadcast over rows

    // --- native FP16 chain ---
    TensorPtr h = forge::ops::rms_norm(x_dev, w_dev, 1e-5f);
    TensorPtr p = forge::ops::matmul_transB(h, b_dev);
    TensorPtr o = forge::ops::silu_multiply(p, up_dev);

    // dtype invariant: no stage is silently promoted to FP32.
    CHECK(x_dev->dtype() == DataType::FP16);
    CHECK(b_dev->dtype() == DataType::FP16);
    CHECK(w_dev->dtype() == DataType::FP16);
    CHECK(h->dtype() == DataType::FP16);
    CHECK(p->dtype() == DataType::FP16);
    CHECK(o->dtype() == DataType::FP16);
    if (h->dtype() != DataType::FP16 || p->dtype() != DataType::FP16 || o->dtype() != DataType::FP16) {
        std::printf("  FAIL chain M=%d: a stage was promoted away from FP16\n", M);
        return;
    }

    auto o_host = std::make_shared<Tensor>(DataType::FP16, shape2(M, N), DeviceType::CPU);
    o_host->copy_from(*o);
    const uint16_t* out = static_cast<const uint16_t*>(o_host->data());
    std::vector<uint16_t> got(out, out + (size_t)M * N);

    // --- verified FP32 reference chain (CPU) ---
    // NOTE: the CPU silu reference (forge::cpu::silu_mul_f32_vec, AVX2) had a
    // pre-existing large-activation bug (Cephes exp missing the FP32 clamp) that
    // is fixed alongside 6-A-5; with it fixed, ops::silu_multiply is a trustworthy
    // reference. RMSNorm / Matmul on CPU are exact and already validated.
    auto x_ref = make_host_fp32(x_dec, M, K);
    auto b_ref = make_host_fp32(b_dec, N, K);
    auto up_ref = make_host_fp32(up_dec, M, N);
    auto w_ref = make_host_fp32(w_dec, 1, K);
    TensorPtr h32 = forge::ops::rms_norm(x_ref, w_ref, 1e-5f);
    TensorPtr p32 = forge::ops::matmul_transB(h32, b_ref);
    TensorPtr o32 = forge::ops::silu_multiply(p32, up_ref);
    const float* ref = static_cast<const float*>(o32->data());
    std::vector<float> ref_vec(ref, ref + (size_t)M * N);

    float max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < ref_vec.size(); ++i) {
        float v = f16_to_f32(got[i]);
        float diff = std::fabs(v - ref_vec[i]);
        if (diff > max_abs)
            max_abs = diff;
        if (std::fabs(ref_vec[i]) > 0.1f) {
            float re = diff / std::fabs(ref_vec[i]);
            if (re > max_rel)
                max_rel = re;
        }
    }
    std::printf("  chain M=%d K=%d N=%d : max_abs=%.3e max_rel=%.3e (all stages FP16)\n", M, K, N,
                max_abs, max_rel);
    // Tolerances reflect native FP16 STORAGE end-to-end: the activation (RMSNorm
    // output ~1.9) is stored in FP16, so its ~1e-3 rounding propagates through the
    // K=4096 matmul accumulation and shows up as a few-% relative error on small
    // output elements (matmul output magnitude ~147, FP16 ULP ~0.125). A real
    // regression (e.g. a GPU op falling back to FP32 or producing garbage) blows
    // these bounds by 3+ orders of magnitude, so they are tight enough to catch it.
    CHECK(max_abs < 0.3f);
    CHECK(max_rel < 0.2f);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_matmul_fp16_chain)\n");
        return 0;
    }

    std::printf("[6-A-5.3] Real native FP16 chain: RMSNorm -> Matmul -> SiLU (dtype invariant + correctness)\n");
    std::mt19937 rng(0x6A53);

    // M=1 (decode GEMV), M=8 and M=32 (prefill rows) exercise both branches.
    check_chain(1, 4096, 11008, rng);
    check_chain(8, 4096, 11008, rng);
    check_chain(32, 4096, 11008, rng);

    if (failures == 0) {
        std::printf("PASS matmul-fp16-chain (Phase 6-A-5)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
