// Unit tests for PowerPC64 VSX kernel functions.
// Compiles on PPC64LE, or cross-compile + QEMU for verification on x86.
//
// Build (native PPC64LE):
//   g++ -std=c++17 -mcpu=power8 -O2 -DUSE_VSX \
//       tests/test_ppc64_vsx_kernels.cpp -o test_vsx && ./test_vsx
//
// Build (cross-compile + QEMU):
//   powerpc64le-linux-gnu-g++ -std=c++17 -mcpu=power8 -O2 -DUSE_VSX \
//       tests/test_ppc64_vsx_kernels.cpp -o test_vsx_ppc64 -static
//   qemu-ppc64le-static ./test_vsx_ppc64
//
// Expected: All tests PASS. Each VSX kernel is compared against a scalar
// reference implementation.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <vector>

// Include the PPC64 VSX kernels (simulates what simd.h does)
#include "src/operators/cpu/arch/ppc64/kernels.h"

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
    float a[N], b[N], out_vsx[N], out_ref[N];

    // Test multiple sizes to cover all loop unrolling paths
    for (int n : {1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 1024}) {
        fill_random(rng, a, n, 10.0f);
        fill_random(rng, b, n, 10.0f);

        // add
        forge::cpu::add_f32_vec(a, b, out_vsx, n);
        ref_add_f32_vec(a, b, out_ref, n);
        check_close_f32("add_f32_vec", out_vsx, out_ref, n, 1e-6f);

        // mul
        forge::cpu::mul_f32_vec(a, b, out_vsx, n);
        ref_mul_f32_vec(a, b, out_ref, n);
        check_close_f32("mul_f32_vec", out_vsx, out_ref, n, 1e-6f);

        // silu_mul
        fill_random(rng, a, n, 5.0f);
        fill_random(rng, b, n, 5.0f);
        forge::cpu::silu_mul_f32_vec(a, b, out_vsx, n);
        ref_silu_mul_f32_vec(a, b, out_ref, n);
        check_close_f32("silu_mul_f32_vec", out_vsx, out_ref, n, 1e-4f);

        // gelu_mul
        fill_random(rng, a, n, 5.0f);
        forge::cpu::gelu_mul_f32_vec(a, b, out_vsx, n);
        ref_gelu_mul_f32_vec(a, b, out_ref, n);
        check_close_f32("gelu_mul_f32_vec", out_vsx, out_ref, n, 1e-4f);
    }
}

static void test_norm() {
    std::mt19937 rng(42);
    const int N = 512;
    float x[N], w[N], o_vsx[N], o_ref[N];

    for (int cols : {1, 7, 8, 15, 16, 63, 64, 127, 128, 512}) {
        fill_random(rng, x, cols, 5.0f);
        fill_random(rng, w, cols, 2.0f);

        // With weight
        forge::cpu::rms_norm_row_f32(x, w, o_vsx, cols, 1e-5f);
        ref_rms_norm_row_f32(x, w, o_ref, cols, 1e-5f);
        check_close_f32("rms_norm (weighted)", o_vsx, o_ref, cols, 1e-6f);

        // Without weight
        forge::cpu::rms_norm_row_f32(x, nullptr, o_vsx, cols, 1e-5f);
        ref_rms_norm_row_f32(x, nullptr, o_ref, cols, 1e-5f);
        check_close_f32("rms_norm (no weight)", o_vsx, o_ref, cols, 1e-6f);
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
        float d_vsx = forge::cpu::dot_f32(a, b, n);
        float d_ref = ref_dot_f32(a, b, n);
        check_close_f32_scalar("dot_f32", d_vsx, d_ref, 1e-4f);

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
    float data[N], data_vsx[N], data_ref[N], out_vsx[N], out_ref[N];

    for (int n : {1, 3, 4, 7, 8, 15, 16, 63, 64, 512}) {
        // argmax
        fill_random(rng, data, n, 100.0f);
        int am_vsx = forge::cpu::argmax_f32(data, n);
        int am_ref = ref_argmax_f32(data, n);
        check_eq_i32("argmax_f32", am_vsx, am_ref);

        // max
        float mx_vsx = forge::cpu::max_f32(data, n);
        float mx_ref = ref_max_f32(data, n);
        check_close_f32_scalar("max_f32", mx_vsx, mx_ref, 1e-6f);

        // softcap_and_argmax
        std::memcpy(data_vsx, data, n * sizeof(float));
        std::memcpy(data_ref, data, n * sizeof(float));
        int sca_vsx = forge::cpu::softcap_and_argmax_f32(data_vsx, n, 30.0f);
        int sca_ref = ref_softcap_and_argmax_f32(data_ref, n, 30.0f);
        check_eq_i32("softcap_and_argmax", sca_vsx, sca_ref);
        check_close_f32("softcap_values", data_vsx, data_ref, n, 3e-4f);

        // softcap_and_max
        std::memcpy(data_vsx, data, n * sizeof(float));
        std::memcpy(data_ref, data, n * sizeof(float));
        float scm_vsx = forge::cpu::softcap_and_max_f32(data_vsx, n, 30.0f);
        float scm_ref = ref_softcap_and_max_f32(data_ref, n, 30.0f);
        check_close_f32_scalar("softcap_and_max", scm_vsx, scm_ref, 3e-4f);
        check_close_f32("softcap_and_max_values", data_vsx, data_ref, n, 3e-4f);

        // exp_and_sum_f32 + scale_normalize
        fill_random(rng, data, n, 10.0f);
        float mx = forge::cpu::max_f32(data, n);
        float st_vsx = forge::cpu::exp_and_sum_f32(data, out_vsx, n, mx, 1.0f);
        float st_ref = ref_exp_and_sum_f32(data, out_ref, n, mx, 1.0f);
        // NOTE: fast-exp approximation has ~1.5% error — use same tolerance as values
        check_close_f32_scalar("exp_and_sum (sum)", st_vsx, st_ref, 1e-1f);
        // NOTE: fast-exp approximation has ~1.5% error, use looser tolerance
        check_close_f32("exp_and_sum (values)", out_vsx, out_ref, n, 2e-2f);

        forge::cpu::scale_normalize_f32(out_vsx, n, 1.0f / st_vsx);
        ref_scale_normalize_f32(out_ref, n, 1.0f / st_ref);
        check_close_f32("scale_normalize", out_vsx, out_ref, n, 2e-2f);
    }
}

static void test_kv() {
    std::mt19937 rng(42);
    const int max_dim = 128;
    float kv_data[4 * 8 * max_dim];
    float out_vsx[4 * 16 * max_dim];
    float out_ref[4 * 16 * max_dim];

    // Test: seq_len=1..4, heads=16, kv_heads=8, head_dim=8..128
    for (int seq : {1, 2, 4}) {
        for (int hd : {8, 16, 64, 128}) {
            int num_heads = 16;
            int num_kv_heads = 8;
            fill_random(rng, kv_data, seq * num_kv_heads * hd, 5.0f);

            forge::cpu::expand_kv_heads_f32(kv_data, out_vsx, seq, num_heads, num_kv_heads, hd);
            ref_expand_kv_heads_f32(kv_data, out_ref, seq, num_heads, num_kv_heads, hd);
            check_close_f32("expand_kv_heads_f32", out_vsx, out_ref, seq * num_heads * hd, 1e-6f);
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

// Scalar vec_dot_q4_1_q8_0
// Q4_1 block = 20 bytes: d[2] fp16 + m[2] fp16 + qs[16] nibbles
// Q4_1 value = nibble * d + m, so dot = d_scale*sum(nibble*q8) + m_scale*sum(q8)
static float ref_vec_dot_q4_1_q8_0(const uint8_t* w_row,
                                    const forge::cpu::block_q8_0_act* act,
                                    int nb) {
    float acc_d = 0.0f, acc_m = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint16_t d_bits, m_bits;
        std::memcpy(&d_bits, w_row + i * 20, 2);
        std::memcpy(&m_bits, w_row + i * 20 + 2, 2);
        float d_val = fp16_to_f32_ref(d_bits);
        float m_val = fp16_to_f32_ref(m_bits);
        float d_scale = d_val * act[i].d;
        float m_scale = m_val * act[i].d;

        const uint8_t* qs = w_row + i * 20 + 4;
        float dot = 0.0f;
        for (int j = 0; j < 32; ++j) {
            int byte_idx = j / 2;
            int nibble = (j % 2 == 0) ? (qs[byte_idx] & 0x0F) : (qs[byte_idx] >> 4);
            dot += (float)nibble * (float)act[i].qs[j];
        }
        float sum_act = 0.0f;
        for (int j = 0; j < 32; ++j) sum_act += (float)act[i].qs[j];
        acc_d += d_scale * dot;
        acc_m += m_scale * sum_act;
    }
    return acc_d + acc_m;
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
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) { float v = std::fabs(src[j]); if (v > amax) amax = v; }
        float d = amax / 7.0f;
        float id = (amax > 0.0f) ? 1.0f / d : 0.0f;

        uint8_t* blk = w + i * 18;
        uint16_t f16 = encode_fp16(d);
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
        uint16_t f16 = encode_fp16(d);
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

// Create synthetic Q4_1 weight blocks from random floats
static void make_q4_1_blocks(std::mt19937& rng, uint8_t* w, int nb) {
    for (int i = 0; i < nb; ++i) {
        float src[32];
        fill_random(rng, src, 32, 3.0f);
        float amax = 0.0f, amin = 0.0f;
        for (int j = 0; j < 32; ++j) {
            if (src[j] > amax) amax = src[j];
            if (src[j] < amin) amin = src[j];
        }
        float d = (amax - amin) / 15.0f;
        float m = amin;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;

        uint16_t f16_d = encode_fp16(d);
        uint16_t f16_m = encode_fp16(m);
        std::memcpy(w + i * 20, &f16_d, 2);
        std::memcpy(w + i * 20 + 2, &f16_m, 2);

        uint8_t* qs = w + i * 20 + 4;
        for (int j = 0; j < 16; ++j) qs[j] = 0;
        for (int j = 0; j < 32; ++j) {
            float x = (src[j] - m) * id;
            int qv = (int)((x >= 0) ? (x + 0.5f) : (x - 0.5f));
            if (qv > 15) qv = 15; if (qv < 0) qv = 0;
            int byte_idx = j / 2;
            if (j % 2 == 0) qs[byte_idx] = (qs[byte_idx] & 0xF0) | (qv & 0x0F);
            else qs[byte_idx] = (qs[byte_idx] & 0x0F) | (qv << 4);
        }
    }
}

// ---- Quantized Dot Product Tests ----

static void test_vec_dot_q4_0() {
    std::mt19937 rng(42);
    for (int nb : {1, 2, 4, 8, 16}) {
        int K = nb * 32;
        std::vector<uint8_t> w_blocks(nb * 18);
        std::vector<forge::cpu::block_q8_0_act> act(nb);
        make_q4_0_blocks(rng, w_blocks.data(), nb);
        make_q8_0_act(rng, act.data(), nb);

        float vsx = forge::cpu::vec_dot_q4_0_q8_0_vsx(w_blocks.data(), act.data(), nb);
        float ref = ref_vec_dot_q4_0_q8_0(w_blocks.data(), act.data(), nb);
        check_close_f32_scalar("vec_dot_q4_0_q8_0", vsx, ref, 5e-4f);
    }
}

static void test_vec_dot_q8_0() {
    std::mt19937 rng(42);
    for (int nb : {1, 2, 4, 8, 16}) {
        std::vector<uint8_t> w_blocks(nb * 34);
        std::vector<forge::cpu::block_q8_0_act> act(nb);
        make_q8_0_blocks(rng, w_blocks.data(), nb);
        make_q8_0_act(rng, act.data(), nb);

        float vsx = forge::cpu::vec_dot_q8_0_q8_0_vsx(w_blocks.data(), act.data(), nb);
        float ref = ref_vec_dot_q8_0_q8_0(w_blocks.data(), act.data(), nb);
        check_close_f32_scalar("vec_dot_q8_0_q8_0", vsx, ref, 1e-5f);
    }
}

static void test_vec_dot_q4_1() {
    std::mt19937 rng(42);
    for (int nb : {1, 2, 4, 8, 16}) {
        std::vector<uint8_t> w_blocks(nb * 20);
        std::vector<forge::cpu::block_q8_0_act> act(nb);
        make_q4_1_blocks(rng, w_blocks.data(), nb);
        make_q8_0_act(rng, act.data(), nb);

        float vsx = forge::cpu::vec_dot_q4_1_q8_0_vsx(w_blocks.data(), act.data(), nb);
        float ref = ref_vec_dot_q4_1_q8_0(w_blocks.data(), act.data(), nb);
        check_close_f32_scalar("vec_dot_q4_1_q8_0", vsx, ref, 5e-4f);
    }
}

// ---- GEMV Tests ----

static void test_gemv_fp32() {
    std::mt19937 rng(42);
    for (int K : {8, 32, 64, 128}) {
        for (int N : {1, 4, 8, 16}) {
            std::vector<float> a(K), b(N * K), out_vsx(N), out_ref(N);
            fill_random(rng, a.data(), K, 3.0f);
            fill_random(rng, b.data(), N * K, 3.0f);

            forge::cpu::gemv_fp32_transB_vsx(a.data(), b.data(), out_vsx.data(), 1, K, N);

            // Reference: out[n] = sum_k(a[k] * b[n*K+k])
            for (int n = 0; n < N; ++n) {
                out_ref[n] = 0.0f;
                for (int k = 0; k < K; ++k) out_ref[n] += a[k] * b[n * K + k];
            }
            check_close_f32("gemv_fp32_transB", out_vsx.data(), out_ref.data(), N, 1e-4f);
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

            std::vector<float> out_vsx(N), out_ref(N, 0.0f);
            forge::cpu::gemv_q4_0_transB_vsx(a.data(), w_blocks.data(), out_vsx.data(), 1, K, N);

            // Reference: same as vec_dot for each row
            std::vector<forge::cpu::block_q8_0_act> q8_act(nb);
            forge::cpu::quantize_row_q8_0_act(a.data(), q8_act.data(), K);
            for (int n = 0; n < N; ++n) {
                out_ref[n] = ref_vec_dot_q4_0_q8_0(
                    w_blocks.data() + n * nb * 18, q8_act.data(), nb);
            }
            check_close_f32("gemv_q4_0_transB", out_vsx.data(), out_ref.data(), N, 1e-4f);
        }
    }
}

static void test_gemv_q8_0() {
    std::mt19937 rng(42);
    for (int nb : {1, 4, 8}) {
        int K = nb * 32;
        for (int N : {1, 3, 5, 8}) {
            std::vector<float> a(K);
            fill_random(rng, a.data(), K, 3.0f);

            std::vector<uint8_t> w_blocks(N * nb * 34);
            for (int n = 0; n < N; ++n) {
                make_q8_0_blocks(rng, w_blocks.data() + n * nb * 34, nb);
            }

            std::vector<float> out_vsx(N), out_ref(N, 0.0f);
            forge::cpu::gemv_q8_0_transB_vsx(a.data(), w_blocks.data(), out_vsx.data(), 1, K, N);

            // Reference: same as vec_dot for each row
            std::vector<forge::cpu::block_q8_0_act> q8_act(nb);
            forge::cpu::quantize_row_q8_0_act(a.data(), q8_act.data(), K);
            for (int n = 0; n < N; ++n) {
                out_ref[n] = ref_vec_dot_q8_0_q8_0(
                    w_blocks.data() + n * nb * 34, q8_act.data(), nb);
            }
            check_close_f32("gemv_q8_0_transB", out_vsx.data(), out_ref.data(), N, 1e-4f);
        }
    }
}

int main() {
    std::printf("=== PPC64 VSX Kernel Unit Tests ===\n\n");

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
    test_vec_dot_q4_1();

    std::printf("\n--- GEMV Kernels ---\n");
    test_gemv_fp32();
    test_gemv_q4_0();
    test_gemv_q8_0();

    std::printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
