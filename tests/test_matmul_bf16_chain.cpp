// =========================================================================
// test_matmul_bf16_chain.cpp -- Phase 6-B-4.3: real native BF16 compute chain
//
//     BF16 input
//        -> Native BF16 RMSNorm      (6-B-2)
//        -> Native BF16 Matmul       (6-B-4)
//        -> Native BF16 activation   (6-B-3: silu_multiply)
//        -> BF16 output
//
// Mirrors test_matmul_fp16_chain.cpp (6-A-5.3). Asserts the **dtype invariant**
// at every stage (no intermediate tensor is silently promoted to FP32) and
// checks correctness against a verified FP32 reference chain built from the
// same decoded BF16 inputs.
//
// Tolerance note: BF16 storage is 8x coarser than FP16, and the RMSNorm-store
// rounding propagates through the K=4096 FP32 accumulation, so the chain
// thresholds are looser than 6-A-5.3's (0.3 / 0.2) by design. They still sit
// orders of magnitude below what a real regression (fallback to FP32,
// garbage kernel output) produces; see the calibration comment in check_chain.
//
// Requires CUDA; skips gracefully without a GPU.
// =========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/operator_elementwise.h"
#include "forge/operator_matmul.h"
#include "forge/operator_norm.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

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
std::vector<int64_t> shape2(int64_t r, int64_t c) {
    return std::vector<int64_t>{r, c};
}

TensorPtr make_dev_bf16(const std::vector<uint16_t>& h, int64_t r, int64_t c) {
    auto host = std::make_shared<Tensor>(DataType::BF16, shape2(r, c), DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 2);
    auto dev = std::make_shared<Tensor>(DataType::BF16, shape2(r, c), DeviceType::CUDA);
    dev->copy_from(*host);
    return dev;
}
TensorPtr make_host_fp32(const std::vector<float>& h, int64_t r, int64_t c) {
    auto t = std::make_shared<Tensor>(DataType::FP32, shape2(r, c), DeviceType::CPU);
    std::memcpy(t->data(), h.data(), h.size() * 4);
    return t;
}

int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

void rand_bf16(int64_t n, float lo, float hi, std::mt19937& g, std::vector<uint16_t>& codes,
               std::vector<float>& decoded) {
    std::uniform_real_distribution<float> d(lo, hi);
    codes.resize(n);
    decoded.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        float v = d(g);
        codes[i] = f32_to_bf16(v);
        decoded[i] = bf16_to_f32(codes[i]);
    }
}

// Run the chain in BF16 on CUDA, assert every stage stays BF16, compare output
// to the verified FP32 chain on the same decoded inputs.
void check_chain(int M, int K, int N, std::mt19937& g) {
    std::vector<uint16_t> x_code, b_code, up_code, w_code;
    std::vector<float> x_dec, b_dec, up_dec, w_dec;
    rand_bf16((int64_t)M * K, -2.0f, 2.0f, g, x_code, x_dec);
    rand_bf16((int64_t)N * K, -1.0f, 1.0f, g, b_code, b_dec);
    rand_bf16((int64_t)M * N, -1.0f, 1.0f, g, up_code, up_dec);
    // RMSNorm weight [1,K] in [0.9,1.1] (the realistic, weighted path).
    rand_bf16((int64_t)K, 0.9f, 1.1f, g, w_code, w_dec);

    auto x_dev = make_dev_bf16(x_code, M, K);
    auto b_dev = make_dev_bf16(b_code, N, K);
    auto up_dev = make_dev_bf16(up_code, M, N);
    auto w_dev = make_dev_bf16(w_code, 1, K);  // broadcast over rows

    // --- native BF16 chain ---
    TensorPtr h = forge::ops::rms_norm(x_dev, w_dev, 1e-5f);
    TensorPtr p = forge::ops::matmul_transB(h, b_dev);
    TensorPtr o = forge::ops::silu_multiply(p, up_dev);

    // dtype invariant: no stage is silently promoted to FP32.
    CHECK(x_dev->dtype() == DataType::BF16);
    CHECK(b_dev->dtype() == DataType::BF16);
    CHECK(w_dev->dtype() == DataType::BF16);
    CHECK(h->dtype() == DataType::BF16);
    CHECK(p->dtype() == DataType::BF16);
    CHECK(o->dtype() == DataType::BF16);
    if (h->dtype() != DataType::BF16 || p->dtype() != DataType::BF16 ||
        o->dtype() != DataType::BF16) {
        std::printf("  FAIL chain M=%d: a stage was promoted away from BF16\n", M);
        return;
    }

    auto o_host = std::make_shared<Tensor>(DataType::BF16, shape2(M, N), DeviceType::CPU);
    o_host->copy_from(*o);
    const uint16_t* out = static_cast<const uint16_t*>(o_host->data());
    std::vector<uint16_t> got(out, out + (size_t)M * N);

    // --- verified FP32 reference chain (CPU) ---
    auto x_ref = make_host_fp32(x_dec, M, K);
    auto b_ref = make_host_fp32(b_dec, N, K);
    auto up_ref = make_host_fp32(up_dec, M, N);
    auto w_ref = make_host_fp32(w_dec, 1, K);
    TensorPtr h32 = forge::ops::rms_norm(x_ref, w_ref, 1e-5f);
    TensorPtr p32 = forge::ops::matmul_transB(h32, b_ref);
    TensorPtr o32 = forge::ops::silu_multiply(p32, up_ref);
    const float* ref = static_cast<const float*>(o32->data());
    std::vector<float> ref_vec(ref, ref + (size_t)M * N);

    // Metric design (diagnosed 2026-09-07): cosine is the primary chain metric
    // (observed 0.999996 on all M -- the chain is numerically exact up to BF16
    // storage). max_rel over |ref|>0.1 is dominated by near-zero outputs where
    // the propagated RMSNorm-store rounding (~0.1 abs) divides by |ref|~0.1, so
    // it is reported for information only. The binding relative metric uses a
    // |ref|>=1.0 floor (6-A-8 e2e precedent: guard must exclude the denominator
    // artifact, not the signal).
    float max_abs = 0, max_rel_info = 0, max_rel = 0;
    double dot = 0, n1 = 0, n2 = 0;
    for (size_t i = 0; i < ref_vec.size(); ++i) {
        float v = bf16_to_f32(got[i]);
        float diff = std::fabs(v - ref_vec[i]);
        if (diff > max_abs)
            max_abs = diff;
        if (std::fabs(ref_vec[i]) > 0.1f) {
            float re = diff / std::fabs(ref_vec[i]);
            if (re > max_rel_info)
                max_rel_info = re;
        }
        if (std::fabs(ref_vec[i]) >= 1.0f) {
            float re = diff / std::fabs(ref_vec[i]);
            if (re > max_rel)
                max_rel = re;
        }
        dot += (double)v * ref_vec[i];
        n1 += (double)v * v;
        n2 += (double)ref_vec[i] * ref_vec[i];
    }
    float cosine = (n1 > 0 && n2 > 0) ? (float)(dot / std::sqrt(n1 * n2)) : 0.0f;
    std::printf(
        "  chain M=%d K=%d N=%d : max_abs=%.3e max_rel(|ref|>=1)=%.3e max_rel_info=%.3e "
        "cosine=%.7f (all stages BF16)\n",
        M, K, N, max_abs, max_rel, max_rel_info, cosine);
    CHECK(cosine >= 0.999f);
    // Calibration (RTX 4050, 2026-09-07): cosine 0.999996 all M; max_abs
    // 0.40/0.54/0.76 (propagated RMSNorm-store rounding through K=4096, ~2-3x
    // the FP16 chain bound 0.3); max_rel(|ref|>=1) 0.098/0.134/0.160 -- the
    // RMSNorm-store rounding (4e-3 rel per element) random-walks through 4096
    // accumulations into ~0.1-0.5 abs error, which reads as ~10-16% on O(1)
    // outputs. Same phenomenon as 6-A-5.3's "few-%" at 1/8 scale: honest
    // propagation, not kernel error (cosine proves it). Thresholds at ~2x
    // observed -- a real regression (fallback/garbage) still blows past by
    // orders of magnitude and collapses cosine below 0.99.
    CHECK(max_abs < 2.0f);
    CHECK(max_rel < 0.3f);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_matmul_bf16_chain)\n");
        return 0;
    }

    std::printf(
        "[6-B-4.3] Real native BF16 chain: RMSNorm -> Matmul -> SiLU (dtype invariant + "
        "correctness)\n");
    std::mt19937 rng(0x6B44);

    // M=1 (decode GEMV), M=8 and M=32 (prefill rows) exercise both branches.
    check_chain(1, 4096, 11008, rng);
    check_chain(8, 4096, 11008, rng);
    check_chain(32, 4096, 11008, rng);

    if (failures == 0) {
        std::printf("PASS matmul-bf16-chain (Phase 6-B-4)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
