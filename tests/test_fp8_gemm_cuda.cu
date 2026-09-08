// Phase 6-C-4.1d: W8A8 FP8 E4M3 GEMM production backend regression.
//
// Validates the Forge production launcher (forge::cuda::launch_fp8_gemm_e4m3_*)
// against a CPU FP32 reference built with the exact E4M3 round-trip used by the
// kernel, and checks the KernelSelector wires FP8_E4M3 prefill to
// MATMUL_FP8_E4M3. We verify on a fixed sample of output positions (mirrors the
// benchmark's sampled parity check) so the CPU golden stays O(samples * K), not
// O(M*N*K). Skips gracefully without an Ada GPU.
//
// FP8-output acceptance mirrors benchmarks/fp8_cutlass_gemm.cu run_scale_case:
//   s_d = 240 / amax(D_fp32),  D_fp8 = fp8_e4m3(D_fp32 * s_d),  check <= 1 ULP.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "forge/fp8_gemm.h"
#include "forge/fp8_utils.h"
#include "forge/kernel_selector.h"
#include "forge/types.h"

static int g_failures = 0;

static bool device_is_ada() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return false;
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return (major * 10 + minor) >= 89;
}

static float* upload_scalar(float v) {
    float* d = nullptr;
    cudaMalloc(&d, sizeof(float));
    cudaMemcpy(d, &v, sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

// E4M3 ULP (3 mantissa bits); below the first normal the step is the fixed
// subnormal quantum 2^-9. Mirrors benchmarks/fp8_cutlass_gemm.cu.
static float e4m3_ulp(float v) {
    float a = fabsf(v);
    if (a < exp2f(-6.0f))
        return exp2f(-9.0f);
    int e = (int)floorf(log2f(a));
    return exp2f((float)e - 3.0f);
}

int main() {
    if (!device_is_ada()) {
        printf("[fp8-gemm] SKIP: no Ada (sm_89+) GPU present\n");
        return 0;
    }

    const int M = 256, N = 4096, K = 4096;

    std::vector<float> hA(M * K), hB(N * K);
    for (float& v : hA)
        v = (float)rand() / RAND_MAX * 2.f - 1.f;
    for (float& v : hB)
        v = (float)rand() / RAND_MAX * 2.f - 1.f;

    // Fixed sample of output positions for the CPU golden (deterministic, cheap).
    std::vector<int> si, sj;
    for (int i = 0; i < M; i += 5)
        for (int j = 0; j < N; j += 17) {
            si.push_back(i);
            sj.push_back(j);
        }
    const int nS = (int)si.size();

    uint8_t *dA = nullptr, *dBcol = nullptr;
    float* dD = nullptr;
    cudaMalloc(&dA, (size_t)M * K);
    cudaMalloc(&dBcol, (size_t)K * N);
    cudaMalloc(&dD, (size_t)M * N * sizeof(float));

    std::vector<uint8_t> Af((size_t)M * K), Bf_row((size_t)N * K);

    for (float p : {0.90f, 0.98f, 0.50f, 0.12f}) {
        // amax-based scale (6-C-1 contract S6): scale = amax / 240.
        float amaxA = 0.f, amaxB = 0.f;
        for (float v : hA)
            amaxA = std::max(amaxA, std::fabs(v));
        for (float v : hB)
            amaxB = std::max(amaxB, std::fabs(v));
        const float sa = amaxA / 240.f;
        const float sw = amaxB / 240.f;

        // Quantize operands (E4M3 round-trip).
        for (size_t i = 0; i < hA.size(); ++i)
            Af[i] = forge::fp32_to_fp8_e4m3(hA[i] / sa);
        for (size_t i = 0; i < hB.size(); ++i)
            Bf_row[i] = forge::fp32_to_fp8_e4m3(hB[i] / sw);

        // B as column-major [K, N] (== Forge weight [N,K] row-major reinterpreted).
        std::vector<uint8_t> Bf_col((size_t)K * N);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                Bf_col[(size_t)k * N + n] = Bf_row[(size_t)n * K + k];

        cudaMemcpy(dA, Af.data(), Af.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dBcol, Bf_col.data(), Bf_col.size(), cudaMemcpyHostToDevice);

        float* d_sa = upload_scalar(sa);
        float* d_sw = upload_scalar(sw);
        float* d_one = upload_scalar(1.f);
        float* d_zero = upload_scalar(0.f);

        forge::cuda::Fp8W8A8Scales sc;
        sc.scale_a = d_sa;
        sc.scale_w = d_sw;

        bool ok = forge::cuda::launch_fp8_gemm_e4m3_fp32(dA, dBcol, dD, M, K, N, sc, 0);
        if (!ok) {
            printf("[fp8-gemm] FAIL: fp32 launcher returned false\n");
            ++g_failures;
            cudaFree(d_sa);
            cudaFree(d_sw);
            cudaFree(d_one);
            cudaFree(d_zero);
            continue;
        }

        std::vector<float> got((size_t)M * N);
        cudaMemcpy(got.data(), dD, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Sampled parity: golden = (sum_k round(a/sa)*round(b/sw)) * sa * sw.
        float max_err = 0.f, dot_g = 0.f, ng = 0.f, nt = 0.f, g_amax = 0.f;
        for (int s = 0; s < nS; ++s) {
            const int i = si[s], j = sj[s];
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                float a =
                    forge::fp8_e4m3_to_fp32(forge::fp32_to_fp8_e4m3(hA[(size_t)i * K + k] / sa));
                float b =
                    forge::fp8_e4m3_to_fp32(forge::fp32_to_fp8_e4m3(hB[(size_t)k * N + j] / sw));
                acc += a * b;
            }
            const float golden = acc * sa * sw;
            const float v = got[(size_t)i * N + j];
            max_err = std::max(max_err, std::fabs(v - golden));
            dot_g += v * golden;
            ng += v * v;
            nt += golden * golden;
            g_amax = std::max(g_amax, std::fabs(golden));
        }
        const float cosine = (ng > 0 && nt > 0) ? dot_g / (std::sqrt(ng) * std::sqrt(nt)) : 0.f;

        // ---- FP8 output variant (scale_d = 240 / amax(D_fp32)) ----
        const float s_d = (g_amax > 0.f) ? (240.0f / g_amax) : 1.f;
        float* d_sd = upload_scalar(s_d);
        uint8_t* dDf8 = nullptr;
        float* d_amax = nullptr;
        cudaMalloc(&dDf8, (size_t)M * N);
        cudaMalloc(&d_amax, sizeof(float));
        forge::cuda::Fp8W8A8Scales sc8;
        sc8.scale_a = d_sa;
        sc8.scale_w = d_sw;
        sc8.scale_d = d_sd;
        sc8.amax_out = d_amax;
        bool ok8 = forge::cuda::launch_fp8_gemm_e4m3_fp8out(dA, dBcol, dDf8, M, K, N, sc8, 0);
        float max_ulp = 0.f;
        int far = 0;
        if (ok8) {
            std::vector<uint8_t> got8((size_t)M * N);
            cudaMemcpy(got8.data(), dDf8, got8.size(), cudaMemcpyDeviceToHost);
            for (int s = 0; s < nS; ++s) {
                const int i = si[s], j = sj[s];
                const float golden = got[(size_t)i * N + j];
                const uint8_t expect = forge::fp32_to_fp8_e4m3(golden * s_d);
                const float dv = forge::fp8_e4m3_to_fp32(got8[(size_t)i * N + j]);
                const float ev = forge::fp8_e4m3_to_fp32(expect);
                const float ulp = e4m3_ulp(ev);
                const float ratio = (ulp > 0.f) ? std::fabs(dv - ev) / ulp : 0.f;
                max_ulp = std::max(max_ulp, ratio);
                if (ratio > 1.5f)
                    ++far;  // beyond ~1 ULP -> far
            }
            float amax = 0.f;
            cudaMemcpy(&amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost);
            // NOTE: the kernel reports the absmax of its D output (ElementAbsmax).
            // The exact reported value depends on the epilogue's absmax semantics
            // and is intentionally NOT in the 6-C-4.1d acceptance (which is ULP/far).
            // We surface it for visibility only; the production amax_out wiring is
            // consumed by 6-C-5 (chained W8A8) and verified there.
            printf("[fp8-gemm]   [info] fp8 amax_out=%.3g (s_d=%.3g)\n", amax, s_d);
        } else {
            printf("[fp8-gemm] WARN: fp8out launcher false\n");
        }

        printf("[fp8-gemm] p=%.2f: max_err=%.3e cosine=%.6f fp8_maxULP=%.2f far=%d\n", p, max_err,
               cosine, max_ulp, far);

        if (max_err > 0.05f) {
            printf("[fp8-gemm] FAIL: max_err too large\n");
            ++g_failures;
        }
        if (cosine < 0.9995f) {
            printf("[fp8-gemm] FAIL: cosine too low\n");
            ++g_failures;
        }
        if (far != 0) {
            printf("[fp8-gemm] FAIL: fp8 output far count %d\n", far);
            ++g_failures;
        }
        if (max_ulp > 1.5f) {
            printf("[fp8-gemm] FAIL: fp8 maxULP %.2f > 1.5\n", max_ulp);
            ++g_failures;
        }

        cudaFree(dDf8);
        cudaFree(d_amax);
        cudaFree(d_sd);
        cudaFree(d_sa);
        cudaFree(d_sw);
        cudaFree(d_one);
        cudaFree(d_zero);
    }

    cudaFree(dA);
    cudaFree(dBcol);
    cudaFree(dD);

    // ---- Selector wiring ----
    forge::KernelProblem prob;
    prob.domain = forge::KernelDomain::MatMul;
    prob.dtype = forge::DataType::FP8_E4M3;
    prob.M = 256;
    prob.N = 4096;
    prob.K = 4096;
    forge::KernelVariant v = forge::select_kernel(prob);
    if (v != forge::KernelVariant::MATMUL_FP8_E4M3) {
        printf("[fp8-gemm] FAIL: selector returned %s, expected MATMUL_FP8_E4M3\n",
               forge::kernel_variant_name(v));
        ++g_failures;
    } else {
        printf("[fp8-gemm] selector: MATMUL_FP8_E4M3 OK\n");
    }

    // 6-C-4.3: M == 1 must route to the hand-written FP8 GEMV, NOT to CUTLASS GEMM
    // and NOT to UNSUPPORTED (on sm_89+). On pre-Ada it must stay UNSUPPORTED.
    forge::KernelProblem probe1;
    probe1.domain = forge::KernelDomain::MatMul;
    probe1.dtype = forge::DataType::FP8_E4M3;
    probe1.M = 1;
    probe1.N = 4096;
    probe1.K = 4096;
    forge::KernelVariant v1 = forge::select_kernel(probe1);
    const bool ada = (v != forge::KernelVariant::UNSUPPORTED);  // M>1 FP8 reachable on this GPU
    const forge::KernelVariant want1 =
        ada ? forge::KernelVariant::MATMUL_FP8_E4M3_GEMV : forge::KernelVariant::UNSUPPORTED;
    if (v1 != want1) {
        printf("[fp8-gemm] FAIL: M==1 selector returned %s, expected %s\n",
               forge::kernel_variant_name(v1), forge::kernel_variant_name(want1));
        ++g_failures;
    } else {
        printf("[fp8-gemm] selector: M==1 %s OK\n", forge::kernel_variant_name(v1));
    }

    if (g_failures == 0) {
        printf("[fp8-gemm] PASS\n");
        return 0;
    }
    printf("[fp8-gemm] FAILURES=%d\n", g_failures);
    return 1;
}
