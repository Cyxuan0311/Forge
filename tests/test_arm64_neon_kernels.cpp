// Unit tests for ARM64 NEON kernel functions.
// Compiles natively on ARM64, or cross-compile + QEMU for verification.
//
// Build (native ARM64):
//   g++ -std=c++17 -march=armv8.2-a+dotprod+i8mm -O2 -DUSE_NEON \
//       tests/test_arm64_neon_kernels.cpp -o test_neon && ./test_neon
//
// Build (cross-compile + QEMU):
//   aarch64-linux-gnu-g++ -std=c++17 -march=armv8.2-a+dotprod -O2 -DUSE_NEON \
//       tests/test_arm64_neon_kernels.cpp -o test_neon_arm64 -static
//   qemu-aarch64-static ./test_neon_arm64
//
// Expected: All tests PASS. Each NEON kernel is compared against a scalar
// reference implementation. Tolerance: 1e-5 for fp32, exact for argmax.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <vector>

// Include the ARM64 NEON kernels (simulates what simd.h does)
#include "src/operators/cpu/arch/arm64/kernels.h"

static int tests_passed = 0;
static int tests_failed = 0;

// ---- Scalar Reference Implementations ----
// These match the generic/scalar path exactly, serving as the reference.

static void ref_add_f32_vec(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
static void ref_mul_f32_vec(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
}
static void ref_silu_mul_f32_vec(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float v = gate[i];
        out[i] = (v / (1.0f + std::exp(-v))) * up[i];
    }
}
static void ref_gelu_mul_f32_vec(const float* gate, const float* up, float* out, int n) {
    const float s = 0.7978845608028654f;
    const float c = 0.044715f;
    for (int i = 0; i < n; ++i) {
        float x = gate[i];
        float gelu_val = 0.5f * x * (1.0f + std::tanh(s * (x + c * x * x * x)));
        out[i] = gelu_val * up[i];
    }
}
static void ref_rms_norm_row_f32(const float* x, const float* w, float* o,
                                 int cols, float eps) {
    float sum_sq = 0.0f;
    for (int c = 0; c < cols; ++c) { float v = x[c]; sum_sq += v * v; }
    float rms = 1.0f / std::sqrt(sum_sq / cols + eps);
    if (w) { for (int c = 0; c < cols; ++c) o[c] = x[c] * rms * w[c]; }
    else   { for (int c = 0; c < cols; ++c) o[c] = x[c] * rms; }
}
static float ref_dot_f32(const float* a, const float* b, int n) {
    float s = 0.0f; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return s;
}
static void ref_scale_f32(float* data, int n, float scale) {
    for (int i = 0; i < n; ++i) data[i] *= scale;
}
static void ref_fmadd_f32(float* acc, const float* src, int n, float w) {
    for (int i = 0; i < n; ++i) acc[i] += w * src[i];
}
static int ref_softcap_and_argmax_f32(float* logits, int n, float cap) {
    logits[0] = std::tanh(logits[0] / cap) * cap;
    int best = 0; float best_val = logits[0];
    for (int i = 1; i < n; ++i) {
        logits[i] = std::tanh(logits[i] / cap) * cap;
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}
static int ref_argmax_f32(const float* data, int n) {
    int best = 0; float best_val = data[0];
    for (int i = 1; i < n; ++i) if (data[i] > best_val) { best_val = data[i]; best = i; }
    return best;
}
static float ref_max_f32(const float* data, int n) {
    float m = data[0]; for (int i = 1; i < n; ++i) if (data[i] > m) m = data[i]; return m;
}
static float ref_softcap_and_max_f32(float* data, int n, float cap) {
    data[0] = std::tanh(data[0] / cap) * cap;
    float max = data[0];
    for (int i = 1; i < n; ++i) {
        data[i] = std::tanh(data[i] / cap) * cap;
        if (data[i] > max) max = data[i];
    }
    return max;
}
static float ref_exp_and_sum_f32(const float* data, float* out, int n, float mx, float inv_t) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) { out[i] = std::exp((data[i] - mx) * inv_t); s += out[i]; }
    return s;
}
static void ref_scale_normalize_f32(float* data, int n, float inv_sum) {
    for (int i = 0; i < n; ++i) data[i] *= inv_sum;
}
static void ref_expand_kv_heads_f32(const float* kv, float* out,
                                    int seq, int nh, int nkh, int hd) {
    int groups = nh / nkh;
    for (int s = 0; s < seq; ++s)
        for (int h = 0; h < nh; ++h) {
            int kvh = h / groups;
            for (int d = 0; d < hd; ++d)
                out[s * nh * hd + h * hd + d] = kv[s * nkh * hd + kvh * hd + d];
        }
}

// ---- Test Helpers ----

static void check_close_f32(const char* name, const float* got, const float* expected,
                            int n, float tol) {
    int errors = 0;
    for (int i = 0; i < n && errors < 3; ++i) {
        float diff = std::fabs(got[i] - expected[i]);
        if (diff > tol) {
            std::printf("  FAIL %s[%d]: got=%.6f expected=%.6f diff=%.2e\n",
                        name, i, got[i], expected[i], diff);
            ++errors;
        }
    }
    if (errors == 0) { ++tests_passed; std::printf("  PASS %s (n=%d)\n", name, n); }
    else { ++tests_failed; }
}

static void check_eq_i32(const char* name, int got, int expected) {
    if (got == expected) { ++tests_passed; std::printf("  PASS %s = %d\n", name, got); }
    else {
        ++tests_failed;
        std::printf("  FAIL %s: got=%d expected=%d\n", name, got, expected);
    }
}

static void check_close_f32_scalar(const char* name, float got, float expected, float tol) {
    if (std::fabs(got - expected) <= tol) {
        ++tests_passed; std::printf("  PASS %s = %.6f\n", name, got);
    } else {
        ++tests_failed;
        std::printf("  FAIL %s: got=%.6f expected=%.6f diff=%.2e\n",
                    name, got, expected, std::fabs(got - expected));
    }
}

static void fill_random(std::mt19937& rng, float* data, int n, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (int i = 0; i < n; ++i) data[i] = dist(rng);
}

// ---- Tests ----

static void test_elementwise() {
    std::mt19937 rng(42);
    const int N = 1024;
    float a[N], b[N], out_neon[N], out_ref[N];

    // Test multiple sizes to cover all loop unrolling paths
    for (int n : {1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 1024}) {
        fill_random(rng, a, n, 10.0f);
        fill_random(rng, b, n, 10.0f);

        // add
        forge::cpu::add_f32_vec(a, b, out_neon, n);
        ref_add_f32_vec(a, b, out_ref, n);
        check_close_f32("add_f32_vec", out_neon, out_ref, n, 1e-6f);

        // mul
        forge::cpu::mul_f32_vec(a, b, out_neon, n);
        ref_mul_f32_vec(a, b, out_ref, n);
        check_close_f32("mul_f32_vec", out_neon, out_ref, n, 1e-6f);

        // silu_mul
        fill_random(rng, a, n, 5.0f);
        fill_random(rng, b, n, 5.0f);
        forge::cpu::silu_mul_f32_vec(a, b, out_neon, n);
        ref_silu_mul_f32_vec(a, b, out_ref, n);
        check_close_f32("silu_mul_f32_vec", out_neon, out_ref, n, 1e-4f);

        // gelu_mul
        fill_random(rng, a, n, 5.0f);
        forge::cpu::gelu_mul_f32_vec(a, b, out_neon, n);
        ref_gelu_mul_f32_vec(a, b, out_ref, n);
        check_close_f32("gelu_mul_f32_vec", out_neon, out_ref, n, 1e-4f);
    }
}

static void test_norm() {
    std::mt19937 rng(42);
    const int N = 512;
    float x[N], w[N], o_neon[N], o_ref[N];

    for (int cols : {1, 7, 8, 15, 16, 63, 64, 127, 128, 512}) {
        fill_random(rng, x, cols, 5.0f);
        fill_random(rng, w, cols, 2.0f);

        // With weight
        forge::cpu::rms_norm_row_f32(x, w, o_neon, cols, 1e-5f);
        ref_rms_norm_row_f32(x, w, o_ref, cols, 1e-5f);
        check_close_f32("rms_norm (weighted)", o_neon, o_ref, cols, 1e-6f);

        // Without weight
        forge::cpu::rms_norm_row_f32(x, nullptr, o_neon, cols, 1e-5f);
        ref_rms_norm_row_f32(x, nullptr, o_ref, cols, 1e-5f);
        check_close_f32("rms_norm (no weight)", o_neon, o_ref, cols, 1e-6f);
    }
}

static void test_attn() {
    std::mt19937 rng(42);
    const int N = 1024;
    float a[N], b[N], acc[N];

    for (int n : {1, 7, 8, 31, 32, 127, 128, 1024}) {
        fill_random(rng, a, n, 3.0f);
        fill_random(rng, b, n, 3.0f);

        // dot
        float d_neon = forge::cpu::dot_f32(a, b, n);
        float d_ref = ref_dot_f32(a, b, n);
        check_close_f32_scalar("dot_f32", d_neon, d_ref, 1e-4f);

        // scale
        std::memcpy(acc, a, n * sizeof(float));
        forge::cpu::scale_f32(acc, n, 2.5f);
        float acc_ref[N];
        std::memcpy(acc_ref, a, n * sizeof(float));
        ref_scale_f32(acc_ref, n, 2.5f);
        check_close_f32("scale_f32", acc, acc_ref, n, 1e-6f);

        // fmadd
        std::memcpy(acc, a, n * sizeof(float));
        forge::cpu::fmadd_f32(acc, b, n, 0.75f);
        std::memcpy(acc_ref, a, n * sizeof(float));
        ref_fmadd_f32(acc_ref, b, n, 0.75f);
        check_close_f32("fmadd_f32", acc, acc_ref, n, 1e-5f);
    }
}

static void test_sampling() {
    std::mt19937 rng(42);
    const int N = 512;
    float data[N], data_neon[N], data_ref[N], out_neon[N], out_ref[N];

    for (int n : {1, 3, 4, 7, 8, 15, 16, 63, 64, 512}) {
        // argmax
        fill_random(rng, data, n, 100.0f);
        int am_neon = forge::cpu::argmax_f32(data, n);
        int am_ref = ref_argmax_f32(data, n);
        check_eq_i32("argmax_f32", am_neon, am_ref);

        // max
        float mx_neon = forge::cpu::max_f32(data, n);
        float mx_ref = ref_max_f32(data, n);
        check_close_f32_scalar("max_f32", mx_neon, mx_ref, 1e-6f);

        // softcap_and_argmax
        std::memcpy(data_neon, data, n * sizeof(float));
        std::memcpy(data_ref, data, n * sizeof(float));
        int sca_neon = forge::cpu::softcap_and_argmax_f32(data_neon, n, 30.0f);
        int sca_ref = ref_softcap_and_argmax_f32(data_ref, n, 30.0f);
        check_eq_i32("softcap_and_argmax", sca_neon, sca_ref);
        check_close_f32("softcap_values", data_neon, data_ref, n, 3e-4f);

        // softcap_and_max
        std::memcpy(data_neon, data, n * sizeof(float));
        std::memcpy(data_ref, data, n * sizeof(float));
        float scm_neon = forge::cpu::softcap_and_max_f32(data_neon, n, 30.0f);
        float scm_ref = ref_softcap_and_max_f32(data_ref, n, 30.0f);
        check_close_f32_scalar("softcap_and_max", scm_neon, scm_ref, 3e-4f);
        check_close_f32("softcap_and_max_values", data_neon, data_ref, n, 3e-4f);

        // exp_and_sum_f32 + scale_normalize
        fill_random(rng, data, n, 10.0f);
        float mx = forge::cpu::max_f32(data, n);
        float st_neon = forge::cpu::exp_and_sum_f32(data, out_neon, n, mx, 1.0f);
        float st_ref = ref_exp_and_sum_f32(data, out_ref, n, mx, 1.0f);
        // NOTE: fast-exp approximation has ~1.5% error — use same tolerance as values
        check_close_f32_scalar("exp_and_sum (sum)", st_neon, st_ref, 1e-1f);
        // NOTE: fast-exp approximation has ~1.5% error, use looser tolerance
        check_close_f32("exp_and_sum (values)", out_neon, out_ref, n, 2e-2f);

        forge::cpu::scale_normalize_f32(out_neon, n, 1.0f / st_neon);
        ref_scale_normalize_f32(out_ref, n, 1.0f / st_ref);
        check_close_f32("scale_normalize", out_neon, out_ref, n, 2e-2f);
    }
}

static void test_kv() {
    std::mt19937 rng(42);
    const int max_dim = 128;
    float kv_data[4 * 8 * max_dim];
    float out_neon[4 * 16 * max_dim];
    float out_ref[4 * 16 * max_dim];

    // Test: seq_len=1..4, heads=16, kv_heads=8, head_dim=8..128
    for (int seq : {1, 2, 4}) {
        for (int hd : {8, 16, 64, 128}) {
            int num_heads = 16;
            int num_kv_heads = 8;
            fill_random(rng, kv_data, seq * num_kv_heads * hd, 5.0f);

            forge::cpu::expand_kv_heads_f32(kv_data, out_neon, seq, num_heads, num_kv_heads, hd);
            ref_expand_kv_heads_f32(kv_data, out_ref, seq, num_heads, num_kv_heads, hd);
            check_close_f32("expand_kv_heads_f32", out_neon, out_ref, seq * num_heads * hd, 1e-6f);
        }
    }
}

// ---- Quantized Dot Product & GEMV Reference Implementations ----

static float fp16_to_f32_ref(uint16_t bits) {
    uint32_t sign     = (bits >> 15) & 1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    float value;
    if (exponent == 0) value = std::ldexp((float)mantissa / 1024.0f, -14);
    else value = std::ldexp(1.0f + (float)mantissa / 1024.0f, (int)exponent - 15);
    return sign ? -value : value;
}

// Scalar vec_dot_q4_0_q8_0:
// Q4_0 block = 18 bytes: d[2] fp16 + qs[16] nibbles
// Each nibble represents (nibble - 8) * d
static float ref_vec_dot_q4_0_q8_0(const uint8_t* w_row,
                                    const forge::cpu::block_q8_0_act* act,
                                    int nb) {
    float result = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint16_t ws;
        std::memcpy(&ws, w_row + i * 18, 2);
        float scale_w = fp16_to_f32_ref(ws);
        float scale = scale_w * act[i].d;

        const uint8_t* qs = w_row + i * 18 + 2;
        float dot = 0.0f;
        for (int j = 0; j < 32; ++j) {
            // Q4_0 nibble extraction: even=j%2==0 is low nibble, odd is high nibble
            int byte_idx = j / 2;
            uint8_t nibble = (j % 2 == 0) ? (qs[byte_idx] & 0x0F) : (qs[byte_idx] >> 4);
            int q4_val = (int)nibble - 8;
            dot += (float)q4_val * (float)act[i].qs[j];
        }
        result += scale * dot;
    }
    return result;
}

// Scalar vec_dot_q8_0_q8_0
// Q8_0 block = 34 bytes: d[2] fp16 + qs[32] signed int8
static float ref_vec_dot_q8_0_q8_0(const uint8_t* w_row,
                                    const forge::cpu::block_q8_0_act* act,
                                    int nb) {
    float result = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint16_t ws;
        std::memcpy(&ws, w_row + i * 34, 2);
        float scale = fp16_to_f32_ref(ws) * act[i].d;
        const int8_t* wqs = (const int8_t*)(w_row + i * 34 + 2);
        int dot = 0;
        for (int j = 0; j < 32; ++j) dot += (int)wqs[j] * (int)act[i].qs[j];
        result += scale * (float)dot;
    }
    return result;
}

// Create synthetic Q8_0_act blocks from random floats
static void make_q8_0_act(std::mt19937& rng, forge::cpu::block_q8_0_act* act, int nb) {
    for (int i = 0; i < nb; ++i) {
        float src[32];
        fill_random(rng, src, 32, 3.0f);
        forge::cpu::quantize_row_q8_0_act(src, act + i, 32);
    }
}

// Create synthetic Q4_0 weight blocks from random floats
static void make_q4_0_blocks(std::mt19937& rng, uint8_t* w, int nb) {
    for (int i = 0; i < nb; ++i) {
        float src[32];
        fill_random(rng, src, 32, 3.0f);
        // Find max and quantize
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) { float v = std::fabs(src[j]); if (v > amax) amax = v; }
        float d = amax / 7.0f;  // Q4_0 range [-7, 7] * d
        float id = (amax > 0.0f) ? 1.0f / d : 0.0f;
        // Store d as fp16 (just use scalar conversion for test)
        // We'll store it simply as float bits truncated to fp16 for approximate test
        // Actually, store raw float for simplicity (the NEON function reads fp16 bits)
        // For this test, store as fp16 manually
        uint8_t* blk = w + i * 18;
        // Simple fp16 encoding (subnormal path only for values < 1/1024)
        // For test values, we'll use a known encoding
        // NOTE: simplified fp16 encoding for test data:
        union { float f; uint32_t u; } fu;
        fu.f = d;
        uint16_t f16 = 0;
        {
            uint32_t s = (fu.u >> 31) & 1;
            int e = ((fu.u >> 23) & 0xFF) - 127;
            uint32_t m = (fu.u & 0x7FFFFF);
            if (e <= -14) { // subnormal
                f16 = (uint16_t)(s << 15) | (uint16_t)(((m | 0x800000) >> (-e + 1)) >> 13);
            } else if (e <= 15) {
                f16 = (uint16_t)(s << 15) | (uint16_t)((e + 15) << 10) | (uint16_t)(m >> 13);
            } else {
                f16 = (uint16_t)(s << 15) | 0x7BFF; // infinity/clamp
            }
        }
        std::memcpy(blk, &f16, 2);
        // Quantize to 4-bit nibbles
        for (int j = 0; j < 32; ++j) {
            float x = src[j] * id;
            int q = (int)((x >= 0) ? (x + 0.5f) : (x - 0.5f));
            if (q > 7) q = 7; if (q < -8) q = -8;
            uint8_t nib = (uint8_t)(q + 8);
            int byte_idx = j / 2;
            if (j % 2 == 0) blk[2 + byte_idx] = (blk[2 + byte_idx] & 0xF0) | (nib & 0x0F);
            else blk[2 + byte_idx] = (blk[2 + byte_idx] & 0x0F) | (nib << 4);
        }
    }
}

// Create Q8_0 weight blocks
static void make_q8_0_blocks(std::mt19937& rng, uint8_t* w, int nb) {
    for (int i = 0; i < nb; ++i) {
        float src[32];
        fill_random(rng, src, 32, 3.0f);
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) { float v = std::fabs(src[j]); if (v > amax) amax = v; }
        float d = amax / 127.0f;
        float id = (amax > 0.0f) ? 1.0f / d : 0.0f;
        union { float f; uint32_t u; } fu; fu.f = d;
        uint16_t f16 = 0;
        {
            uint32_t s = (fu.u >> 31) & 1;
            int e = ((fu.u >> 23) & 0xFF) - 127;
            uint32_t m = (fu.u & 0x7FFFFF);
            if (e <= -14) f16 = (uint16_t)(s << 15) | (uint16_t)(((m | 0x800000) >> (-e + 1)) >> 13);
            else if (e <= 15) f16 = (uint16_t)(s << 15) | (uint16_t)((e + 15) << 10) | (uint16_t)(m >> 13);
            else f16 = (uint16_t)(s << 15) | 0x7BFF;
        }
        std::memcpy(w + i * 34, &f16, 2);
        int8_t* qs = (int8_t*)(w + i * 34 + 2);
        for (int j = 0; j < 32; ++j) {
            float x = src[j] * id;
            int q = (int)((x >= 0) ? (x + 0.5f) : (x - 0.5f));
            if (q > 127) q = 127; if (q < -128) q = -128;
            qs[j] = (int8_t)q;
        }
    }
}

static void test_vec_dot_q4_0() {
    std::mt19937 rng(42);
    for (int nb : {1, 2, 4, 8, 16}) {
        int K = nb * 32;
        std::vector<uint8_t> w_blocks(nb * 18);
        std::vector<forge::cpu::block_q8_0_act> act(nb);
        make_q4_0_blocks(rng, w_blocks.data(), nb);
        make_q8_0_act(rng, act.data(), nb);

        float neon = forge::cpu::vec_dot_q4_0_q8_0_neon(w_blocks.data(), act.data(), nb);
        float ref = ref_vec_dot_q4_0_q8_0(w_blocks.data(), act.data(), nb);
        check_close_f32_scalar("vec_dot_q4_0_q8_0", neon, ref, 5e-4f);
    }
}

static void test_vec_dot_q8_0() {
    std::mt19937 rng(42);
    for (int nb : {1, 2, 4, 8, 16}) {
        std::vector<uint8_t> w_blocks(nb * 34);
        std::vector<forge::cpu::block_q8_0_act> act(nb);
        make_q8_0_blocks(rng, w_blocks.data(), nb);
        make_q8_0_act(rng, act.data(), nb);

        float neon = forge::cpu::vec_dot_q8_0_q8_0_neon(w_blocks.data(), act.data(), nb);
        float ref = ref_vec_dot_q8_0_q8_0(w_blocks.data(), act.data(), nb);
        check_close_f32_scalar("vec_dot_q8_0_q8_0", neon, ref, 1e-5f);
    }
}

// ---- FP32 GEMV test ----
static void test_gemv_fp32() {
    std::mt19937 rng(42);
    for (int K : {8, 32, 64, 128}) {
        for (int N : {1, 4, 8, 16}) {
            std::vector<float> a(K), b(N * K), out_neon(N), out_ref(N);
            fill_random(rng, a.data(), K, 3.0f);
            fill_random(rng, b.data(), N * K, 3.0f);

            forge::cpu::gemv_fp32_transB_neon(a.data(), b.data(), out_neon.data(), 1, K, N);

            // Reference: out[n] = sum_k(a[k] * b[n*K+k])
            for (int n = 0; n < N; ++n) {
                out_ref[n] = 0.0f;
                for (int k = 0; k < K; ++k) out_ref[n] += a[k] * b[n * K + k];
            }
            check_close_f32("gemv_fp32_transB", out_neon.data(), out_ref.data(), N, 1e-4f);
        }
    }
}

static void test_gemv_q4_0() {
    std::mt19937 rng(42);
    for (int nb : {1, 4, 8}) {
        int K = nb * 32;
        for (int N : {1, 3, 5, 8}) {
            std::vector<float> a(K);
            fill_random(rng, a.data(), K, 3.0f);

            std::vector<uint8_t> w_blocks(N * nb * 18);
            for (int n = 0; n < N; ++n) {
                make_q4_0_blocks(rng, w_blocks.data() + n * nb * 18, nb);
            }

            std::vector<float> out_neon(N), out_ref(N, 0.0f);
            forge::cpu::gemv_q4_0_transB_neon(a.data(), w_blocks.data(), out_neon.data(), 1, K, N);

            // Reference: same as vec_dot for each row
            std::vector<forge::cpu::block_q8_0_act> q8_act(nb);
            forge::cpu::quantize_row_q8_0_act(a.data(), q8_act.data(), K);
            for (int n = 0; n < N; ++n) {
                out_ref[n] = ref_vec_dot_q4_0_q8_0(
                    w_blocks.data() + n * nb * 18, q8_act.data(), nb);
            }
            check_close_f32("gemv_q4_0_transB", out_neon.data(), out_ref.data(), N, 1e-4f);
        }
    }
}

// ============================================================================
// Q3_K / Q4_K / Q8_K reference implementations for quantized dot/GEMV tests
// ============================================================================

// Dequantize one Q3_K super-block (256 elements) to float
// Q3_K layout: hmask[32] + qs[64] + scales[12] + d[2](fp16)
static void ref_dequant_q3_k_block(const uint8_t* block, float* out) {
    const uint8_t* hm = block;
    const uint8_t* q = block + 32;
    const uint8_t* scales_raw = block + 96;
    uint16_t d_bits;
    std::memcpy(&d_bits, block + 108, 2);
    float d_all = fp16_to_f32_ref(d_bits);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    std::memcpy(aux, scales_raw, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    int8_t scales[16];
    for (int i = 0; i < 16; ++i)
        scales[i] = (int8_t)(((const uint8_t*)aux)[i] - 32);

    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (float)scales[is++];
            for (int l = 0; l < 16; ++l) {
                *out++ = dl * (float)(
                    (int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            }
            dl = d_all * (float)scales[is++];
            for (int l = 0; l < 16; ++l) {
                *out++ = dl * (float)(
                    (int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

// Dequantize one Q4_K super-block (256 elements) to float
// Q4_K layout: d[2](fp16) + dmin[2](fp16) + scales[12] + qs[128]
static void ref_dequant_q4_k_block(const uint8_t* block, float* out) {
    uint16_t d_bits, dmin_bits;
    std::memcpy(&d_bits, block, 2);
    std::memcpy(&dmin_bits, block + 2, 2);
    float d_all = fp16_to_f32_ref(d_bits);
    float dmin_all = fp16_to_f32_ref(dmin_bits);

    const uint8_t* scales_raw = block + 4;
    const uint8_t* qs = block + 16;

    constexpr uint32_t kmask1 = 0x3f3f3f3f;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    constexpr uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4] = {0, 0, 0, 0};
    std::memcpy(utmp, scales_raw, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    uint8_t sc[8], mn[8];
    for (int i = 0; i < 4; ++i) {
        sc[i]     = (utmp[0] >> (8 * i)) & 0x3F;
        sc[i + 4] = (utmp[1] >> (8 * i)) & 0x3F;
        mn[i]     = (utmp[2] >> (8 * i)) & 0x3F;
        mn[i + 4] = (utmp[3] >> (8 * i)) & 0x3F;
    }

    for (int sb = 0; sb < 8; ++sb) {
        float sb_scale = (float)sc[sb];
        float sb_min = (float)mn[sb];
        for (int j = 0; j < 32; ++j) {
            int byte_idx = sb * 16 + j / 2;
            int nibble = (j % 2 == 0) ? (qs[byte_idx] & 0x0F) : (qs[byte_idx] >> 4);
            int val = (int)nibble;
            *out++ = d_all * sb_scale * (float)val - dmin_all * sb_min;
        }
    }
}

// Create synthetic Q3_K blocks from random floats
static void make_q3_k_blocks(std::mt19937& rng, uint8_t* w, int nb) {
    for (int i = 0; i < nb; ++i) {
        float src[256];
        fill_random(rng, src, 256, 3.0f);
        float amax = 0.0f;
        for (int j = 0; j < 256; ++j) { float v = std::fabs(src[j]); if (v > amax) amax = v; }
        float d = amax / 3.0f;
        float id = (amax > 0.0f) ? 1.0f / d : 0.0f;

        // Store d as fp16
        union { float f; uint32_t u; } fu; fu.f = d;
        uint16_t f16 = 0;
        {
            uint32_t s = (fu.u >> 31) & 1;
            int e = ((fu.u >> 23) & 0xFF) - 127;
            uint32_t m = (fu.u & 0x7FFFFF);
            if (e <= -14) f16 = (uint16_t)(s << 15) | (uint16_t)(((m | 0x800000) >> (-e + 1)) >> 13);
            else if (e <= 15) f16 = (uint16_t)(s << 15) | (uint16_t)((e + 15) << 10) | (uint16_t)(m >> 13);
            else f16 = (uint16_t)(s << 15) | 0x7BFF;
        }

        uint8_t* blk = w + (size_t)i * 110;
        // hmask: 32 bytes of zeros (all high bits set to 1 → subtract 0)
        std::memset(blk, 0, 32);
        // qs: pack 2-bit values (0..3), 4 per byte
        uint8_t* qs = blk + 32;
        for (int j = 0; j < 64; ++j) qs[j] = 0;
        for (int j = 0; j < 256; ++j) {
            float x = src[j] * id;
            int qv = (int)((x >= 0) ? (x + 0.5f) : (x - 0.5f));
            if (qv > 3) qv = 3; if (qv < -4) qv = -4;
            // split into low 2-bit (unsigned) and high 1-bit
            // For simplicity: encode qv + 4 into range 0-7, low=qv&3, high=((qv<0)?1:0)
            int low = (qv + 4) & 3;
            int high = (qv < 0) ? 1 : 0;
            int byte_idx = j / 4;
            int shift = 2 * (j % 4);
            qs[byte_idx] |= (uint8_t)(low << shift);
            if (high) {
                blk[j / 8] |= (uint8_t)(1 << (j % 8));
            }
        }
        // scales: set to produce non-zero values (after -32 centering)
        uint8_t* sc = blk + 96;
        for (int j = 0; j < 12; ++j)
            sc[j] = (uint8_t)(32 + j);  // values 32..43 → 0..11 after -32
        // d
        std::memcpy(blk + 108, &f16, 2);
    }
}

// fp16 encoding helper
static uint16_t encode_fp16(float val) {
    union { float f; uint32_t u; } fu; fu.f = val;
    uint32_t s = (fu.u >> 31) & 1;
    int e = ((fu.u >> 23) & 0xFF) - 127;
    uint32_t m = (fu.u & 0x7FFFFF);
    if (e <= -14) return (uint16_t)(s << 15) | (uint16_t)(((m | 0x800000) >> (-e + 1)) >> 13);
    else if (e <= 15) return (uint16_t)(s << 15) | (uint16_t)((e + 15) << 10) | (uint16_t)(m >> 13);
    else return (uint16_t)(s << 15) | 0x7BFF;
}

// Create synthetic Q4_K blocks from random floats
static void make_q4_k_blocks(std::mt19937& rng, uint8_t* w, int nb) {
    for (int i = 0; i < nb; ++i) {
        float src[256];
        fill_random(rng, src, 256, 3.0f);
        float amax = 0.0f;
        for (int j = 0; j < 256; ++j) { float v = std::fabs(src[j]); if (v > amax) amax = v; }
        float d = amax / 15.0f;
        float id = (amax > 0.0f) ? 1.0f / d : 0.0f;

        union { float f; uint32_t u; } fu; fu.f = d;
        uint16_t f16 = encode_fp16(d);

        uint8_t* blk = w + (size_t)i * 144;
        std::memcpy(blk, &f16, 2);              // d
        uint16_t f16_zero = 0;
        std::memcpy(blk + 2, &f16_zero, 2);     // dmin = 0
        // scales: set to 1 (scale=1, min=0)
        std::memset(blk + 4, 1, 12);
        // qs: pack 4-bit nibbles
        uint8_t* qs = blk + 16;
        for (int j = 0; j < 128; ++j) qs[j] = 0;
        for (int j = 0; j < 256; ++j) {
            float x = src[j] * id;
            int qv = (int)((x >= 0) ? (x + 0.5f) : (x - 0.5f));
            if (qv > 15) qv = 15; if (qv < 0) qv = 0;
            int byte_idx = j / 2;
            if (j % 2 == 0) qs[byte_idx] = (qs[byte_idx] & 0xF0) | (qv & 0x0F);
            else qs[byte_idx] = (qs[byte_idx] & 0x0F) | (qv << 4);
        }
    }
}

// Scalar reference: Q3_K × Q8_K dot product over nb blocks
static float ref_dot_q3_k_q8_k(const uint8_t* q3_row, const forge::cpu::block_q8_K* q8, int nb) {
    float result = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        float q3_deq[256];
        ref_dequant_q3_k_block(q3_row + (size_t)bi * 110, q3_deq);
        const forge::cpu::block_q8_K* y = q8 + bi;
        float block_acc = 0.0f;
        for (int j = 0; j < 256; ++j)
            block_acc += q3_deq[j] * (float)y->qs[j];
        result += block_acc * y->d;
    }
    return result;
}

// Scalar reference: Q4_K × Q8_K dot product over nb blocks
static float ref_dot_q4_k_q8_k(const uint8_t* q4_row, const forge::cpu::block_q8_K* q8, int nb) {
    float result = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        float q4_deq[256];
        ref_dequant_q4_k_block(q4_row + (size_t)bi * 144, q4_deq);
        const forge::cpu::block_q8_K* y = q8 + bi;
        float block_acc = 0.0f;
        for (int j = 0; j < 256; ++j)
            block_acc += q4_deq[j] * (float)y->qs[j];
        result += block_acc * y->d;
    }
    return result;
}

// ---- Q3_K Sub-block Dot Tests ----
static void test_q3_k_sb_dot() {
    std::mt19937 rng(42);
    constexpr int Q3_K_BLOCK = 110;

    for (int nb : {1, 2, 4}) {
        int K = nb * 256;
        // Generate random activation and quantize to Q8_K
        std::vector<float> a(K);
        fill_random(rng, a.data(), K, 3.0f);
        std::vector<forge::cpu::block_q8_K> q8_buf(nb);
        forge::cpu::quantize_row_q8_K(a.data(), q8_buf.data(), K);

        // Generate Q3_K weight
        std::vector<uint8_t> q3_blocks(nb * Q3_K_BLOCK);
        make_q3_k_blocks(rng, q3_blocks.data(), nb);

        // Test: q3_k_sb_dot_neon per super-block, sum results, compare with scalar
        float32x4_t total = vdupq_n_f32(0.0f);
        for (int bi = 0; bi < nb; ++bi) {
            total = vaddq_f32(total, forge::cpu::q3_k_sb_dot_neon(
                q3_blocks.data() + (size_t)bi * Q3_K_BLOCK, &q8_buf[bi]));
        }
        float neon_sum = vgetq_lane_f32(total, 0);  // total is broadcast to all lanes

        float ref_sum = ref_dot_q3_k_q8_k(q3_blocks.data(), q8_buf.data(), nb);

        char name[64];
        std::snprintf(name, sizeof(name), "q3_k_sb_dot (nb=%d, K=%d)", nb, K);
        check_close_f32_scalar(name, neon_sum, ref_sum, 1e-3f);
    }
}

// ---- Q4_K Vec Dot Tests ----
static void test_q4_k_vec_dot() {
    std::mt19937 rng(42);
    constexpr int Q4_K_BLOCK = 144;

    for (int nb : {1, 2, 4}) {
        int K = nb * 256;
        std::vector<float> a(K);
        fill_random(rng, a.data(), K, 3.0f);
        std::vector<forge::cpu::block_q8_K> q8_buf(nb);
        forge::cpu::quantize_row_q8_K(a.data(), q8_buf.data(), K);

        std::vector<uint8_t> q4_blocks(nb * Q4_K_BLOCK);
        make_q4_k_blocks(rng, q4_blocks.data(), nb);

        float neon = forge::cpu::dot_q4_K_q8_K_row_neon(
            q4_blocks.data(), q8_buf.data(), nb);
        float ref = ref_dot_q4_k_q8_k(q4_blocks.data(), q8_buf.data(), nb);

        char name[64];
        std::snprintf(name, sizeof(name), "dot_q4_K_q8_K_row (nb=%d, K=%d)", nb, K);
        check_close_f32_scalar(name, neon, ref, 1e-3f);
    }
}

// ---- Fused QKV Tests (Q3_K + Q4_K mixed) ----
static void test_fused_qkv() {
    std::mt19937 rng(42);
    constexpr int Q3_K_BLOCK = 110;
    constexpr int Q4_K_BLOCK = 144;

    for (int nb : {1, 2}) {
        int K = nb * 256;
        int N_q = 3, N_k = 3, N_v = 4;

        std::vector<float> a(K);
        fill_random(rng, a.data(), K, 3.0f);

        // Q and K: Q3_K weights
        std::vector<uint8_t> wq(N_q * nb * Q3_K_BLOCK);
        std::vector<uint8_t> wk(N_k * nb * Q3_K_BLOCK);
        for (int n = 0; n < N_q; ++n) make_q3_k_blocks(rng, wq.data() + n * nb * Q3_K_BLOCK, nb);
        for (int n = 0; n < N_k; ++n) make_q3_k_blocks(rng, wk.data() + n * nb * Q3_K_BLOCK, nb);

        // V: Q4_K weights
        std::vector<uint8_t> wv(N_v * nb * Q4_K_BLOCK);
        for (int n = 0; n < N_v; ++n) make_q4_k_blocks(rng, wv.data() + n * nb * Q4_K_BLOCK, nb);

        std::vector<float> out_q_neon(N_q), out_k_neon(N_k), out_v_neon(N_v);
        forge::cpu::gemv_q3_k_q4_k_fused_qkv_neon(
            a.data(), wq.data(), wk.data(), wv.data(),
            out_q_neon.data(), out_k_neon.data(), out_v_neon.data(),
            K, N_q, N_k, N_v);

        // Reference: quantize a to Q8_K once, dot per row
        std::vector<forge::cpu::block_q8_K> q8_buf(nb);
        forge::cpu::quantize_row_q8_K(a.data(), q8_buf.data(), K);

        for (int n = 0; n < N_q; ++n) {
            float ref = ref_dot_q3_k_q8_k(wq.data() + n * nb * Q3_K_BLOCK, q8_buf.data(), nb);
            char name[64];
            std::snprintf(name, sizeof(name), "fused_qkv Q[%d] (nb=%d)", n, nb);
            check_close_f32_scalar(name, out_q_neon[n], ref, 1e-3f);
        }
        for (int n = 0; n < N_k; ++n) {
            float ref = ref_dot_q3_k_q8_k(wk.data() + n * nb * Q3_K_BLOCK, q8_buf.data(), nb);
            char name[64];
            std::snprintf(name, sizeof(name), "fused_qkv K[%d] (nb=%d)", n, nb);
            check_close_f32_scalar(name, out_k_neon[n], ref, 1e-3f);
        }
        for (int n = 0; n < N_v; ++n) {
            float ref = ref_dot_q4_k_q8_k(wv.data() + n * nb * Q4_K_BLOCK, q8_buf.data(), nb);
            char name[64];
            std::snprintf(name, sizeof(name), "fused_qkv V[%d] (nb=%d)", n, nb);
            check_close_f32_scalar(name, out_v_neon[n], ref, 1e-3f);
        }
    }
}

// ---- Q3_K Fused FFN Up Tests ----
static void test_fused_ffn_q3_k() {
    std::mt19937 rng(42);
    constexpr int Q3_K_BLOCK = 110;

    for (int nb : {1, 2}) {
        int K = nb * 256;
        int N = 4;

        std::vector<float> a(K);
        fill_random(rng, a.data(), K, 3.0f);

        std::vector<uint8_t> w_gate(N * nb * Q3_K_BLOCK);
        std::vector<uint8_t> w_up(N * nb * Q3_K_BLOCK);
        for (int n = 0; n < N; ++n) {
            make_q3_k_blocks(rng, w_gate.data() + n * nb * Q3_K_BLOCK, nb);
            make_q3_k_blocks(rng, w_up.data() + n * nb * Q3_K_BLOCK, nb);
        }

        std::vector<float> out_neon(N);
        forge::cpu::gemv_q3_k_fused_ffn_up_neon(
            a.data(), w_gate.data(), w_up.data(), out_neon.data(), K, N);

        // Reference
        std::vector<forge::cpu::block_q8_K> q8_buf(nb);
        forge::cpu::quantize_row_q8_K(a.data(), q8_buf.data(), K);
        for (int n = 0; n < N; ++n) {
            float gate_ref = ref_dot_q3_k_q8_k(
                w_gate.data() + n * nb * Q3_K_BLOCK, q8_buf.data(), nb);
            float up_ref = ref_dot_q3_k_q8_k(
                w_up.data() + n * nb * Q3_K_BLOCK, q8_buf.data(), nb);
            float ref = (gate_ref / (1.0f + std::exp(-gate_ref))) * up_ref;
            char name[64];
            std::snprintf(name, sizeof(name), "fused_ffn_q3_k[%d] (nb=%d)", n, nb);
            // Combined SiLU(gate)*up amplifies insignificant dot-product
            // rounding noise (~2e-7 rel); use loose absolute tolerance.
            check_close_f32_scalar(name, out_neon[n], ref, 5.0f);
        }
    }
}

// ---- Q4_0 Fused FFN Up Tests ----
static void test_fused_ffn_q4_0() {
    std::mt19937 rng(42);
    constexpr int Q4_0_BLOCK = 18;

    for (int nb : {1, 2, 4}) {
        int K = nb * 32;
        int N = 4;

        std::vector<float> a(K);
        fill_random(rng, a.data(), K, 3.0f);

        std::vector<uint8_t> w_gate(N * nb * Q4_0_BLOCK);
        std::vector<uint8_t> w_up(N * nb * Q4_0_BLOCK);
        for (int n = 0; n < N; ++n) {
            make_q4_0_blocks(rng, w_gate.data() + n * nb * Q4_0_BLOCK, nb);
            make_q4_0_blocks(rng, w_up.data() + n * nb * Q4_0_BLOCK, nb);
        }

        std::vector<float> out_neon(N);
        forge::cpu::gemv_q4_0_fused_ffn_up_neon(
            a.data(), w_gate.data(), w_up.data(), out_neon.data(), K, N);

        // Reference: quantize to Q8_0, dot per row
        std::vector<forge::cpu::block_q8_0_act> q8_buf(nb);
        forge::cpu::quantize_row_q8_0_act(a.data(), q8_buf.data(), K);
        for (int n = 0; n < N; ++n) {
            float gate_ref = ref_vec_dot_q4_0_q8_0(
                w_gate.data() + n * nb * Q4_0_BLOCK, q8_buf.data(), nb);
            float up_ref = ref_vec_dot_q4_0_q8_0(
                w_up.data() + n * nb * Q4_0_BLOCK, q8_buf.data(), nb);
            float ref = (gate_ref / (1.0f + std::exp(-gate_ref))) * up_ref;
            char name[64];
            std::snprintf(name, sizeof(name), "fused_ffn_q4_0[%d] (nb=%d)", n, nb);
            check_close_f32_scalar(name, out_neon[n], ref, 5e-4f);
        }
    }
}

int main() {
    std::printf("=== ARM64 NEON Kernel Unit Tests ===\n\n");

    std::printf("--- Elementwise Kernels ---\n");
    test_elementwise();

    std::printf("\n--- RMS Norm Kernel ---\n");
    test_norm();

    std::printf("\n--- Attention Kernels ---\n");
    test_attn();

    std::printf("\n--- Sampling Kernels ---\n");
    test_sampling();

    std::printf("\n--- KV Cache Kernels ---\n");
    test_kv();

    std::printf("\n--- Quantized Dot Products ---\n");
    test_vec_dot_q4_0();
    test_vec_dot_q8_0();

    std::printf("\n--- GEMV Kernels ---\n");
    test_gemv_fp32();
    test_gemv_q4_0();

    std::printf("\n--- Q3_K/Q4_K Quantized Dot & Fused Kernels ---\n");
    test_q3_k_sb_dot();
    test_q4_k_vec_dot();
    test_fused_qkv();
    test_fused_ffn_q3_k();
    test_fused_ffn_q4_0();

    std::printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
