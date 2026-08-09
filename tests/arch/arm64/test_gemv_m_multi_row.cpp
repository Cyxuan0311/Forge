/**
 * Standalone QEMU test for NEON GEMV M-loop fixes.
 *
 * Verifies that gemv_fp32_transB_neon, gemv_q4_0_transB_neon, and
 * gemv_q8_0_transB_neon compute all M rows correctly.
 *
 * Cross-compile:
 *   aarch64-linux-gnu-g++ -std=c++17 -O2 -march=armv8-a+simd -static \
 *     -I ../../../src/operators/cpu/arch/arm64 \
 *     test_gemv_m_multi_row.cpp -o test_gemv_m_multi_row
 * Run:
 *   qemu-aarch64 ./test_gemv_m_multi_row
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// --- Include NEON GEMV headers (vec_dot.h must come before gemv.h) ---
#define USE_NEON
#include "vec_dot.h"
#include "gemv.h"

using namespace forge::cpu;

// ======================================================================
// Reference implementations (M-aware, scalar fallback)
// ======================================================================

static void gemv_fp32_ref(const float* a, const float* b, float* out, int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[(size_t)m * K + k] * b[(size_t)n * K + k];
            }
            out[(size_t)m * N + n] = sum;
        }
    }
}

static void gemv_q4_0_ref(const float* a, const uint8_t* w, float* out, int M, int K, int N) {
    const int QK = 32, BLK = 18;
    int nb = K / QK;
    std::vector<block_q8_0_act> q8(nb);
    for (int m = 0; m < M; ++m) {
        quantize_row_q8_0_act(a + (size_t)m * K, q8.data(), K);
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const uint8_t* wr = w + (size_t)n * nb * BLK;
            for (int bi = 0; bi < nb; ++bi) {
                uint16_t ws; std::memcpy(&ws, wr + (size_t)bi * BLK, 2);
                float sw = arm_dot::fp16_to_fp32(ws);
                float s = sw * q8[bi].d;
                const uint8_t* qs = wr + (size_t)bi * BLK + 2;
                int32_t d = 0;
                for (int j = 0; j < QK/2; ++j) {
                    d += (int32_t)((qs[j] & 0xF) - 8) * (int32_t)q8[bi].qs[j*2];
                    d += (int32_t)((qs[j] >> 4) - 8)   * (int32_t)q8[bi].qs[j*2+1];
                }
                sum += s * (float)d;
            }
            out[(size_t)m * N + n] = sum;
        }
    }
}

static void gemv_q8_0_ref(const float* a, const uint8_t* w, float* out, int M, int K, int N) {
    const int QK = 32, BLK = 34;
    int nb = K / QK;
    std::vector<block_q8_0_act> q8(nb);
    for (int m = 0; m < M; ++m) {
        quantize_row_q8_0_act(a + (size_t)m * K, q8.data(), K);
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const uint8_t* wr = w + (size_t)n * nb * BLK;
            for (int bi = 0; bi < nb; ++bi) {
                uint16_t ws; std::memcpy(&ws, wr + (size_t)bi * BLK, 2);
                float sw = arm_dot::fp16_to_fp32(ws);
                float s = sw * q8[bi].d;
                const int8_t* qs = (const int8_t*)(wr + (size_t)bi * BLK + 2);
                int32_t d = 0;
                for (int j = 0; j < QK; ++j) d += (int32_t)qs[j] * (int32_t)q8[bi].qs[j];
                sum += s * (float)d;
            }
            out[(size_t)m * N + n] = sum;
        }
    }
}

// ======================================================================
// Test driver
// ======================================================================

static int errors = 0;

static void check_fp32(const char* label, int M, int K, int N) {
    size_t as = (size_t)M * K, bs = (size_t)N * K, os = (size_t)M * N;
    float *a = new float[as], *b = new float[bs];
    float *neon = new float[os], *ref = new float[os];

    for (size_t i = 0; i < as; ++i) a[i] = (float)((int)(i % 37) - 18) * 0.1f;
    for (size_t i = 0; i < bs; ++i) b[i] = (float)((int)(i % 53) - 26) * 0.1f;

    memset(neon, 0, os * sizeof(float));
    gemv_fp32_transB_neon(a, b, neon, M, K, N);
    gemv_fp32_ref(a, b, ref, M, K, N);

    bool ok = true;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float d = fabsf(neon[(size_t)m * N + n] - ref[(size_t)m * N + n]);
            if (d > 1e-4f) {
                printf("  FAIL [%s] m=%d n=%d NEON=%f REF=%f\n", label, m, n,
                       neon[(size_t)m * N + n], ref[(size_t)m * N + n]);
                ok = false; ++errors;
            }
        }
    }
    if (ok) {
        printf("  OK   %-35s M=%d K=%d N=%d\n", label, M, K, N);
        for (int m = 0; m < M; ++m)
            printf("    row[%d]: NEON=%-12.6f REF=%-12.6f\n", m, neon[(size_t)m * N], ref[(size_t)m * N]);
    }

    delete[] a; delete[] b; delete[] neon; delete[] ref;
}

template<void (*NEON)(const float*, const uint8_t*, float*, int, int, int)>
static void check_quant(const char* label, int M, int K, int N,
                         void (*REF)(const float*, const uint8_t*, float*, int, int, int)) {
    const int QK = 32;
    int BLK = (label[0] == 'Q' && label[1] == '4') ? 18 : 34;
    int nb = K / QK;

    size_t as = (size_t)M * K, ws = (size_t)N * nb * BLK, os = (size_t)M * N;
    float   *a = new float[as];
    uint8_t *w = new uint8_t[ws];
    float *neon = new float[os], *ref = new float[os];

    for (size_t i = 0; i < as; ++i) a[i] = (float)((int)(i % 37) - 18) * 0.1f;

    // Build quantized weights
    for (int nn = 0; nn < N; ++nn) {
        for (int bi = 0; bi < nb; ++bi) {
            size_t off = (size_t)nn * nb * BLK + (size_t)bi * BLK;
            float vals[32], amax = 0.0f;
            for (int j = 0; j < QK; ++j) {
                vals[j] = (float)(((int)((nn * nb + bi) * 32 + j) % 53) - 26) * 0.1f;
                amax = fmaxf(amax, fabsf(vals[j]));
            }
            float d = amax / ((BLK == 18) ? 7.0f : 127.0f);
            if (d == 0.0f) d = 1.0f;

            // Store fp16 scale
            uint16_t f16 = 0;
            {
                uint32_t bits; memcpy(&bits, &d, 4);
                uint32_t sgn = (bits >> 31) & 1;
                int exp = (int)((bits >> 23) & 0xFF) - 127;
                if (exp < -14) exp = -14; if (exp > 15) exp = 15;
                f16 = (uint16_t)((sgn << 15) | ((exp + 15) << 10));
            }
            memcpy(w + off, &f16, 2);

            if (BLK == 18) { // Q4_0
                for (int j = 0; j < QK/2; ++j) {
                    int v0 = (int)roundf(vals[j*2]/d), v1 = (int)roundf(vals[j*2+1]/d);
                    if (v0 < -8) v0 = -8; if (v0 > 7) v0 = 7;
                    if (v1 < -8) v1 = -8; if (v1 > 7) v1 = 7;
                    w[off+2+j] = (uint8_t)((v0 & 0xF) | ((v1 & 0xF) << 4));
                }
            } else { // Q8_0
                for (int j = 0; j < QK; ++j) {
                    int v = (int)roundf(vals[j]/d);
                    if (v < -127) v = -127; if (v > 127) v = 127;
                    w[off+2+j] = (uint8_t)(int8_t)v;
                }
            }
        }
    }

    memset(neon, 0, os * sizeof(float));
    NEON(a, w, neon, M, K, N);
    REF(a, w, ref, M, K, N);

    bool ok = true;
    float tol = (BLK == 18) ? 2.5f : 2.0f;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float d = fabsf(neon[(size_t)m * N + n] - ref[(size_t)m * N + n]);
            if (d > tol) {
                printf("  FAIL [%s] m=%d n=%d NEON=%f REF=%f diff=%f\n", label, m, n,
                       neon[(size_t)m * N + n], ref[(size_t)m * N + n], d);
                ok = false; ++errors;
            }
        }
    }
    if (ok) {
        printf("  OK   %-35s M=%d K=%d N=%d\n", label, M, K, N);
        for (int m = 0; m < M; ++m)
            printf("    row[%d]: NEON=%-12.6f REF=%-12.6f\n", m, neon[(size_t)m * N], ref[(size_t)m * N]);
    }

    delete[] a; delete[] w; delete[] neon; delete[] ref;
}

int main() {
    printf("\n=== NEON GEMV M-loop Verification (QEMU aarch64) ===\n\n");

    // ---- FP32 (FIXED) ----
    printf("[gemv_fp32_transB_neon]  (M-loop FIXED)\n");
    check_fp32("M=1 decode", 1, 32, 8);
    check_fp32("M=3 prefill", 3, 32, 8);
    check_fp32("M=5 prefill", 5, 64, 16);
    check_fp32("M=7 prefill", 7, 128, 32);

    // ---- Q4_0 (check for bug) ----
    printf("\n[gemv_q4_0_transB_neon] (check M>1)\n");
    check_quant<gemv_q4_0_transB_neon>("Q4_0 M=1", 1, 32, 4, gemv_q4_0_ref);
    check_quant<gemv_q4_0_transB_neon>("Q4_0 M=3", 3, 32, 4, gemv_q4_0_ref);

    // ---- Q8_0 (check for bug) ----
    printf("\n[gemv_q8_0_transB_neon] (check M>1)\n");
    check_quant<gemv_q8_0_transB_neon>("Q8_0 M=1", 1, 32, 4, gemv_q8_0_ref);
    check_quant<gemv_q8_0_transB_neon>("Q8_0 M=3", 3, 32, 4, gemv_q8_0_ref);

    printf("\n========================================\n");
    if (errors == 0) {
        printf("  ALL PASSED — VERIFIED\n");
    } else {
        printf("  %d ERROR(S) DETECTED\n", errors);
    }
    printf("========================================\n\n");
    return errors ? 1 : 0;
}
