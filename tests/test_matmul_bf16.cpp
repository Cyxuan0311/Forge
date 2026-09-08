// =========================================================================
// test_matmul_bf16.cpp -- Phase 6-B-4: Native BF16 Matmul
// correctness + micro-benchmark (requires CUDA; skips gracefully w/o GPU).
//
// 6-B-4.1 (direct launch): BF16 A + BF16 B -> cuBLAS (CUDA_R_16BF,
//                         CUBLAS_COMPUTE_32F) -> BF16 C, FP32 accumulate.
// 6-B-4.2 (route):        ops::matmul_transB(BF16, BF16) -> BF16 out, routed to
//                         the native kernel; FP32 / quantized paths untouched.
// 6-B-4.4 (M sweep):      BF16 vs FP32 cuBLAS across M=1..512 (no preset
//                         speedup; M=1 re-measured, not assumed from FP16).
//
// "No implicit FP32" guarantee is checked structurally: the native route returns
// a BF16 tensor (out dtype == BF16) and feeds __nv_bfloat16 buffers straight
// into cuBLAS with no FP32 staging tensor. Correctness is checked against the
// verified FP32 matmul on the *same decoded BF16 inputs* -- never against raw
// FP32 model inputs (that would mix input-quantization error with kernel error).
//
// Tolerance note: BF16 ULP is 2^-7 (8x FP16), so single-matmul thresholds are
// set from the measured distribution (max_rel anchored on the 2^-8 epsilon,
// max_abs scaled to the output magnitude), separating single-storage error
// from accumulation propagation. Structural defects still blow these by
// orders of magnitude.
//
// Retirement proof: 4.2 calls ops::matmul_transB with BF16 and expects success
// -- it fails if any routing site still rejects BF16 for MUL_MAT_TRANSB.
// non-transB MUL_MAT stays rejecting (no row-major-B BF16 path this round).
// =========================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/operator_matmul.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;
using namespace forge::cuda;  // launch_cublas_gemm_bf16 / launch_cublas_sgemm

// ---- bf16 codec (matches the kernel storage convention) ----------------
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

// Host bf16 buffer -> CUDA BF16 tensor.
TensorPtr make_dev_bf16(const std::vector<uint16_t>& h, int64_t r, int64_t c) {
    auto host = std::make_shared<Tensor>(DataType::BF16, shape2(r, c), DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 2);
    auto dev = std::make_shared<Tensor>(DataType::BF16, shape2(r, c), DeviceType::CUDA);
    dev->copy_from(*host);
    return dev;
}
// Host fp32 buffer -> CPU FP32 tensor (used for the verified reference).
TensorPtr make_host_fp32(const std::vector<float>& h, int64_t r, int64_t c) {
    auto t = std::make_shared<Tensor>(DataType::FP32, shape2(r, c), DeviceType::CPU);
    std::memcpy(t->data(), h.data(), h.size() * 4);
    return t;
}

// BF16 uniform random in [-1,1], returns the bf16 codes and the fp32-decoded
// values (so the reference reflects exactly the bf16 input rounding).
void rand_bf16_matrix(int64_t r, int64_t c, std::mt19937& g, std::vector<uint16_t>& codes,
                      std::vector<float>& decoded) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    int64_t n = r * c;
    codes.resize(n);
    decoded.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        float v = d(g);
        codes[i] = f32_to_bf16(v);
        decoded[i] = bf16_to_f32(codes[i]);
    }
}

int failures = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

// Compare a native BF16 result (decoded) to the FP32 reference; the residual
// is the BF16 output store rounding (native accumulates in FP32).
// Calibration (RTX 4050, 2026-09-07, K=N=4096, inputs in [-1,1]):
// max_rel 3.8-4.2e-3 (anchored on the BF16 epsilon 2^-8), max_abs 0.24-0.25
// (outputs O(10-100); 1 BF16 ULP at 64-128 is 0.5-1.0, so this is sub-ULP
// output-store rounding). Set at ~4-5x observed; structural defects (wrong
// kernel, mis-cast, dropped accumulation) blow past by 10x+.
void compare_to_fp32_ref(const char* name, const std::vector<uint16_t>& got_bf16,
                         const std::vector<float>& ref_fp32) {
    float max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < ref_fp32.size(); ++i) {
        float got = bf16_to_f32(got_bf16[i]);
        float diff = std::fabs(got - ref_fp32[i]);
        if (diff > max_abs)
            max_abs = diff;
        if (std::fabs(ref_fp32[i]) > 0.1f) {
            float re = diff / std::fabs(ref_fp32[i]);
            if (re > max_rel)
                max_rel = re;
        }
    }
    std::printf("  %-22s : max_abs=%.3e max_rel=%.3e\n", name, max_abs, max_rel);
    CHECK(max_abs < 1.0f);
    CHECK(max_rel < 0.02f);
}

// ---- 6-B-4.1: direct native BF16 GEMM launch --------------------------
void check_native_launch(int M, int K, int N, std::mt19937& g) {
    std::vector<uint16_t> a_code, b_code;
    std::vector<float> a_dec, b_dec;
    rand_bf16_matrix(M, K, g, a_code, a_dec);
    rand_bf16_matrix(N, K, g, b_code, b_dec);

    auto a_dev = make_dev_bf16(a_code, M, K);
    auto b_dev = make_dev_bf16(b_code, N, K);

    auto c_dev = std::make_shared<Tensor>(DataType::BF16, shape2(M, N), DeviceType::CUDA);
    launch_cublas_gemm_bf16(static_cast<const __nv_bfloat16*>(a_dev->data()),
                            static_cast<const __nv_bfloat16*>(b_dev->data()),
                            static_cast<__nv_bfloat16*>(c_dev->data()), M, K, N,
                            /*transB=*/true, 0);

    auto c_host = std::make_shared<Tensor>(DataType::BF16, shape2(M, N), DeviceType::CPU);
    c_host->copy_from(*c_dev);
    const uint16_t* c_out = static_cast<const uint16_t*>(c_host->data());
    std::vector<uint16_t> got(c_out, c_out + (size_t)M * N);

    // Verified FP32 reference on the same decoded BF16 inputs (CPU sgemm).
    auto a_ref = make_host_fp32(a_dec, M, K);
    auto b_ref = make_host_fp32(b_dec, N, K);
    auto c_ref = forge::ops::matmul_transB(a_ref, b_ref);
    const float* ref = static_cast<const float*>(c_ref->data());
    std::vector<float> ref_vec(ref, ref + (size_t)M * N);

    char tag[64];
    std::snprintf(tag, sizeof(tag), "native_launch M=%d", M);
    compare_to_fp32_ref(tag, got, ref_vec);
}

// ---- 6-B-4.2: ops::matmul_transB route (BF16 in -> BF16 out) ----------
void check_route(int M, int K, int N, std::mt19937& g) {
    std::vector<uint16_t> a_code, b_code;
    std::vector<float> a_dec, b_dec;
    rand_bf16_matrix(M, K, g, a_code, a_dec);
    rand_bf16_matrix(N, K, g, b_code, b_dec);

    auto a_dev = make_dev_bf16(a_code, M, K);
    auto b_dev = make_dev_bf16(b_code, N, K);

    // The whole point: input + weight are BF16, and the op must stay BF16.
    CHECK(a_dev->dtype() == DataType::BF16);
    CHECK(b_dev->dtype() == DataType::BF16);

    auto out = forge::ops::matmul_transB(a_dev, b_dev);

    // "No implicit FP32" structural guarantee: the result is a real BF16 tensor.
    CHECK(out->dtype() == DataType::BF16);
    if (out->dtype() != DataType::BF16) {
        std::printf("  FAIL route M=%d: output dtype is not BF16\n", M);
        return;
    }

    auto out_host = std::make_shared<Tensor>(DataType::BF16, shape2(M, N), DeviceType::CPU);
    out_host->copy_from(*out);
    const uint16_t* c_out = static_cast<const uint16_t*>(out_host->data());
    std::vector<uint16_t> got(c_out, c_out + (size_t)M * N);

    auto a_ref = make_host_fp32(a_dec, M, K);
    auto b_ref = make_host_fp32(b_dec, N, K);
    auto c_ref = forge::ops::matmul_transB(a_ref, b_ref);
    const float* ref = static_cast<const float*>(c_ref->data());
    std::vector<float> ref_vec(ref, ref + (size_t)M * N);

    char tag[64];
    std::snprintf(tag, sizeof(tag), "route M=%d", M);
    compare_to_fp32_ref(tag, got, ref_vec);
}

// Mixed BF16 x FP32 must fail loudly, never mis-cast.
void check_mixed_loud(std::mt19937& g) {
    std::vector<uint16_t> a_code;
    std::vector<float> a_dec;
    rand_bf16_matrix(1, 64, g, a_code, a_dec);
    auto a_dev = make_dev_bf16(a_code, 1, 64);
    auto b_dev = std::make_shared<Tensor>(DataType::FP32, shape2(64, 64), DeviceType::CUDA);
    bool threw = false;
    try {
        (void)forge::ops::matmul_transB(a_dev, b_dev);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    if (threw) {
        std::printf("  mixed BF16 A + FP32 B: loud reject OK\n");
    }
}

// ---- 6-B-4.4: native BF16 cuBLAS vs FP32 cuBLAS, M sweep --------------
// Protocol (per the 6-B-3 bench lessons): alternate the two arms within each
// round and take best-of-3 per arm -- a single sequential pass is not evidence
// (clock/thermal state). No speedup is preset; M=1 is re-measured.
void bench_matmul(int M, int K, int N, int iters) {
    auto ab = std::make_shared<Tensor>(DataType::BF16, shape2(M, K), DeviceType::CUDA);
    auto bb = std::make_shared<Tensor>(DataType::BF16, shape2(N, K), DeviceType::CUDA);
    auto cb = std::make_shared<Tensor>(DataType::BF16, shape2(M, N), DeviceType::CUDA);
    auto a32 = std::make_shared<Tensor>(DataType::FP32, shape2(M, K), DeviceType::CUDA);
    auto b32 = std::make_shared<Tensor>(DataType::FP32, shape2(N, K), DeviceType::CUDA);
    auto c32 = std::make_shared<Tensor>(DataType::FP32, shape2(M, N), DeviceType::CUDA);

    std::vector<uint16_t> fillb((size_t)M * K, f32_to_bf16(0.5f));
    std::vector<uint16_t> fillbb((size_t)N * K, f32_to_bf16(0.5f));
    std::vector<float> fill32((size_t)M * K, 0.5f);
    std::vector<float> fillb32((size_t)N * K, 0.5f);
    auto hb = std::make_shared<Tensor>(DataType::BF16, shape2(M, K), DeviceType::CPU);
    auto hbb = std::make_shared<Tensor>(DataType::BF16, shape2(N, K), DeviceType::CPU);
    auto ha32 = std::make_shared<Tensor>(DataType::FP32, shape2(M, K), DeviceType::CPU);
    auto hb32 = std::make_shared<Tensor>(DataType::FP32, shape2(N, K), DeviceType::CPU);
    std::memcpy(hb->data(), fillb.data(), fillb.size() * 2);
    std::memcpy(hbb->data(), fillbb.data(), fillbb.size() * 2);
    std::memcpy(ha32->data(), fill32.data(), fill32.size() * 4);
    std::memcpy(hb32->data(), fillb32.data(), fillb32.size() * 4);
    ab->copy_from(*hb);
    bb->copy_from(*hbb);
    a32->copy_from(*ha32);
    b32->copy_from(*hb32);

    auto time_bf16 = [&]() {
        cudaEvent_t es, ee;
        cudaEventCreate(&es);
        cudaEventCreate(&ee);
        cudaEventRecord(es);
        for (int i = 0; i < iters; ++i)
            launch_cublas_gemm_bf16(static_cast<const __nv_bfloat16*>(ab->data()),
                                    static_cast<const __nv_bfloat16*>(bb->data()),
                                    static_cast<__nv_bfloat16*>(cb->data()), M, K, N, true);
        cudaEventRecord(ee);
        cudaEventSynchronize(ee);
        float ms = 0;
        cudaEventElapsedTime(&ms, es, ee);
        cudaEventDestroy(es);
        cudaEventDestroy(ee);
        return ms / iters;
    };
    auto time_fp32 = [&]() {
        cudaEvent_t es, ee;
        cudaEventCreate(&es);
        cudaEventCreate(&ee);
        cudaEventRecord(es);
        for (int i = 0; i < iters; ++i)
            launch_cublas_sgemm(static_cast<const float*>(a32->data()),
                                static_cast<const float*>(b32->data()),
                                static_cast<float*>(c32->data()), M, K, N, true);
        cudaEventRecord(ee);
        cudaEventSynchronize(ee);
        float ms = 0;
        cudaEventElapsedTime(&ms, es, ee);
        cudaEventDestroy(es);
        cudaEventDestroy(ee);
        return ms / iters;
    };

    for (int i = 0; i < 3; ++i) {
        time_bf16();
        time_fp32();
    }
    cudaDeviceSynchronize();

    // Best-of-3, strictly alternating arms.
    float best_bf = 1e30f, best_32 = 1e30f;
    for (int r = 0; r < 3; ++r) {
        best_bf = std::min(best_bf, time_bf16());
        best_32 = std::min(best_32, time_fp32());
    }

    // Traffic: A,B read + C written. BF16 => 3 * 2B/elt; FP32 => 3 * 4B/elt.
    const double bytes_bf = static_cast<double>(M) * (K + K + N) * 2.0;
    const double bytes32 = static_cast<double>(M) * (K + K + N) * 4.0;
    std::printf("bench M=%4d K=%d N=%d iters=%d : BF16 %.5f ms/iter %.1f GB/s | FP32 %.5f ms/iter %.1f GB/s | %.2fx\n",
                M, K, N, iters, best_bf, bytes_bf / (best_bf / 1000.0) / 1e9, best_32,
                bytes32 / (best_32 / 1000.0) / 1e9, best_32 / best_bf);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_matmul_bf16)\n");
        return 0;
    }

    if (!forge::op_supports_lowp(forge::OpType::MUL_MAT_TRANSB)) {
        std::printf("FAIL: op_supports_lowp(MUL_MAT_TRANSB) expected true (6-A-5 + 6-B-4)\n");
        return 1;
    }

    std::printf("[6-B-4] Native BF16 Matmul correctness (vs FP32 reference)\n");
    std::mt19937 rng(0x6B44);

    std::printf("  6-B-4.1 native launch (BF16 A x BF16 B -> BF16 C, FP32 accumulate):\n");
    check_native_launch(1, 4096, 4096, rng);
    check_native_launch(8, 4096, 4096, rng);
    check_native_launch(64, 4096, 4096, rng);

    std::printf("  6-B-4.2 ops::matmul_transB route (BF16 in -> BF16 out):\n");
    check_route(1, 4096, 4096, rng);
    check_route(1, 256, 512, rng);
    check_route(8, 4096, 4096, rng);
    check_route(64, 4096, 4096, rng);
    check_mixed_loud(rng);

    if (failures == 0) {
        std::printf("[6-B-4.4] micro-bench (BF16 vs FP32 cuBLAS, M sweep, best-of-3 alternating; K=N=4096)\n");
        bench_matmul(1, 4096, 4096, 200);
        bench_matmul(8, 4096, 4096, 100);
        bench_matmul(32, 4096, 4096, 50);
        bench_matmul(128, 4096, 4096, 20);
        bench_matmul(512, 4096, 4096, 10);
        std::printf("PASS matmul-bf16 (Phase 6-B-4)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
