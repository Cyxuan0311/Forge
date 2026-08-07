// Test generic (scalar) kernel accuracy — verifies that the FORGE_ARCH_GENERIC
// fallback path provides correct scalar implementations for all kernel functions.
//
// Compile (on any host — forces generic path):
//   g++ -std=c++17 -O2 -DFORGE_ARCH_GENERIC \
//       -I/path/to/forge -I/path/to/forge/include \
//       -I/path/to/forge/src/operators/cpu \
//       -I/path/to/forge/src/operators/cpu/arch/generic \
//       tests/test_generic_scalar_kernels.cpp -lm -o test_generic
//
// Run: ./test_generic

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

#include "src/operators/cpu/arch/generic/kernels.h"

using namespace forge::cpu;

// ---- helpers ----

static std::mt19937 g_rng(42);

template <typename T>
static bool check_close_f32(const T* a, const T* b, int n, float eps = 1e-5f) {
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs((float)a[i] - (float)b[i]);
        if (diff > eps) {
            std::cerr << "Mismatch at " << i << ": " << a[i] << " vs " << b[i]
                      << " (diff=" << diff << ")\n";
            return false;
        }
    }
    return true;
}

static bool check_close_scalar(float a, float b, float eps = 1e-5f) {
    float diff = std::fabs(a - b);
    if (diff > eps) {
        std::cerr << "Mismatch: " << a << " vs " << b << " (diff=" << diff << ")\n";
        return false;
    }
    return true;
}

static bool check_eq_i32(int a, int b) {
    if (a != b) {
        std::cerr << "Mismatch: " << a << " vs " << b << "\n";
        return false;
    }
    return true;
}

static void fill_random(std::mt19937& rng, float* data, int n, float scale = 3.0f) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (int i = 0; i < n; ++i) data[i] = dist(rng);
}

// ---- reference implementations ----

static void ref_add_f32(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
static void ref_mul_f32(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
}
static void ref_silu_mul_f32(const float* g, const float* u, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float v = g[i];
        out[i] = (v / (1.0f + std::exp(-v))) * u[i];
    }
}
static void ref_gelu_mul_f32(const float* g, const float* u, float* out, int n) {
    const float c = 0.7978845608028654f;
    const float k = 0.044715f;
    for (int i = 0; i < n; ++i) {
        float x = g[i];
        out[i] = 0.5f * x * (1.0f + std::tanh(c * (x + k * x * x * x))) * u[i];
    }
}
static void ref_rms_norm_row_f32(const float* x, const float* w, float* o, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float r = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) o[i] = x[i] * r * (w ? w[i] : 1.0f);
}
static float ref_dot_f32(const float* a, const float* b, int n) {
    float s = 0; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return s;
}
static void ref_scale_f32(float* d, int n, float s) {
    for (int i = 0; i < n; ++i) d[i] *= s;
}
static void ref_fmadd_f32(float* acc, const float* src, int n, float w) {
    for (int i = 0; i < n; ++i) acc[i] += w * src[i];
}
static int ref_argmax_f32(const float* d, int n) {
    int b = 0; float bv = d[0];
    for (int i = 1; i < n; ++i) { if (d[i] > bv) { bv = d[i]; b = i; } }
    return b;
}
static float ref_max_f32(const float* d, int n) {
    float m = d[0]; for (int i = 1; i < n; ++i) { if (d[i] > m) m = d[i]; } return m;
}
static int ref_softcap_and_argmax_f32(float* d, int n, float cap) {
    int b = 0; float bv = d[0];
    for (int i = 0; i < n; ++i) { d[i] = std::tanh(d[i] / cap) * cap; if (d[i] > bv) { bv = d[i]; b = i; } }
    return b;
}
static float ref_softcap_and_max_f32(float* d, int n, float cap) {
    float m = d[0];
    for (int i = 0; i < n; ++i) { d[i] = std::tanh(d[i] / cap) * cap; if (d[i] > m) m = d[i]; }
    return m;
}
static float ref_exp_and_sum_f32(const float* d, float* o, int n, float mv, float it) {
    float s = 0; for (int i = 0; i < n; ++i) { o[i] = std::exp((d[i] - mv) * it); s += o[i]; } return s;
}
static void ref_scale_normalize_f32(float* d, int n, float is) {
    for (int i = 0; i < n; ++i) d[i] *= is;
}
static void ref_expand_kv_heads_f32(const float* kv, float* out,
                                    int sl, int nh, int nkv, int hd) {
    int ng = nh / nkv;
    for (int s = 0; s < sl; ++s)
        for (int h = 0; h < nh; ++h)
            for (int d = 0; d < hd; ++d)
                out[s * nh * hd + h * hd + d] = kv[s * nkv * hd + (h/ng) * hd + d];
}

// ---- tests ----

static int g_passed = 0, g_failed = 0;
#define TEST(name) do { \
    std::cout << "  " << #name << "... "; \
    if (test_##name()) { g_passed++; std::cout << "PASS\n"; } \
    else { g_failed++; std::cout << "FAIL\n"; } \
} while(0)

// --------

static bool test_elementwise() {
    for (int n : {16, 32, 64, 128, 257, 512}) {
        std::vector<float> a(n), b(n), out1(n), out2(n);
        fill_random(g_rng, a.data(), n); fill_random(g_rng, b.data(), n);
        
        add_f32_vec(a.data(), b.data(), out1.data(), n);
        ref_add_f32(a.data(), b.data(), out2.data(), n);
        if (!check_close_f32(out1.data(), out2.data(), n)) return false;

        mul_f32_vec(a.data(), b.data(), out1.data(), n);
        ref_mul_f32(a.data(), b.data(), out2.data(), n);
        if (!check_close_f32(out1.data(), out2.data(), n)) return false;

        silu_mul_f32_vec(a.data(), b.data(), out1.data(), n);
        ref_silu_mul_f32(a.data(), b.data(), out2.data(), n);
        if (!check_close_f32(out1.data(), out2.data(), n, 1e-4f)) return false;

        gelu_mul_f32_vec(a.data(), b.data(), out1.data(), n);
        ref_gelu_mul_f32(a.data(), b.data(), out2.data(), n);
        if (!check_close_f32(out1.data(), out2.data(), n, 1e-4f)) return false;
    }
    return true;
}

static bool test_norm() {
    for (int n : {32, 128, 256, 512}) {
        std::vector<float> x(n), w(n), o1(n), o2(n);
        fill_random(g_rng, x.data(), n); fill_random(g_rng, w.data(), n);

        rms_norm_row_f32(x.data(), nullptr, o1.data(), n, 1e-5f);
        ref_rms_norm_row_f32(x.data(), nullptr, o2.data(), n, 1e-5f);
        if (!check_close_f32(o1.data(), o2.data(), n, 1e-4f)) return false;

        rms_norm_row_f32(x.data(), w.data(), o1.data(), n, 1e-5f);
        ref_rms_norm_row_f32(x.data(), w.data(), o2.data(), n, 1e-5f);
        if (!check_close_f32(o1.data(), o2.data(), n, 1e-4f)) return false;
    }
    return true;
}

static bool test_attn() {
    for (int n : {32, 128, 256}) {
        std::vector<float> a(n), b(n);
        fill_random(g_rng, a.data(), n); fill_random(g_rng, b.data(), n);

        float d1 = dot_f32(a.data(), b.data(), n);
        float d2 = ref_dot_f32(a.data(), b.data(), n);
        if (!check_close_scalar(d1, d2, 1e-4f * n)) return false;

        memcpy(b.data(), a.data(), n * sizeof(float));
        scale_f32(a.data(), n, 2.5f);
        ref_scale_f32(b.data(), n, 2.5f);
        if (!check_close_f32(a.data(), b.data(), n)) return false;

        fill_random(g_rng, a.data(), n); fill_random(g_rng, b.data(), n);
        std::vector<float> acc1(n), acc2(n);
        memcpy(acc1.data(), a.data(), n * sizeof(float));
        memcpy(acc2.data(), a.data(), n * sizeof(float));
        fmadd_f32(acc1.data(), b.data(), n, 0.5f);
        ref_fmadd_f32(acc2.data(), b.data(), n, 0.5f);
        if (!check_close_f32(acc1.data(), acc2.data(), n)) return false;
    }
    return true;
}

static bool test_sampling() {
    for (int n : {64, 256, 1024}) {
        std::vector<float> d1(n), d2(n);
        fill_random(g_rng, d1.data(), n); 
        memcpy(d2.data(), d1.data(), n * sizeof(float));

        int a1 = argmax_f32(d1.data(), n);
        int a2 = ref_argmax_f32(d2.data(), n);
        if (!check_eq_i32(a1, a2)) return false;

        float m1 = max_f32(d1.data(), n);
        float m2 = ref_max_f32(d2.data(), n);
        if (!check_close_scalar(m1, m2)) return false;

        // softcap+argmax (mutates data)
        memcpy(d2.data(), d1.data(), n * sizeof(float));
        int sa1 = softcap_and_argmax_f32(d1.data(), n, 25.0f);
        int sa2 = ref_softcap_and_argmax_f32(d2.data(), n, 25.0f);
        if (!check_eq_i32(sa1, sa2)) return false;
        if (!check_close_f32(d1.data(), d2.data(), n, 1e-4f)) return false;

        // softcap+max
        fill_random(g_rng, d1.data(), n); memcpy(d2.data(), d1.data(), n * sizeof(float));
        float sm1 = softcap_and_max_f32(d1.data(), n, 25.0f);
        float sm2 = ref_softcap_and_max_f32(d2.data(), n, 25.0f);
        if (!check_close_scalar(sm1, sm2, 1e-4f)) return false;
        if (!check_close_f32(d1.data(), d2.data(), n, 1e-4f)) return false;

        // exp_and_sum
        fill_random(g_rng, d1.data(), n); memcpy(d2.data(), d1.data(), n * sizeof(float));
        std::vector<float> o1(n), o2(n);
        float mv = max_f32(d1.data(), n);
        float s1 = exp_and_sum_f32(d1.data(), o1.data(), n, mv, 1.0f);
        float s2 = ref_exp_and_sum_f32(d2.data(), o2.data(), n, mv, 1.0f);
        if (!check_close_scalar(s1, s2, 1e-1f)) return false; // fast-exp tolerance
        if (!check_close_f32(o1.data(), o2.data(), n, 2e-2f)) return false;

        // scale_normalize
        fill_random(g_rng, d1.data(), n); memcpy(d2.data(), d1.data(), n * sizeof(float));
        scale_normalize_f32(d1.data(), n, 0.1f);
        ref_scale_normalize_f32(d2.data(), n, 0.1f);
        if (!check_close_f32(d1.data(), d2.data(), n)) return false;
    }
    return true;
}

static bool test_kv() {
    for (int sl : {1, 4})
    for (int nh : {8, 16})
    for (int nkv : {2, 4}) {
        int hd = 64;
        std::vector<float> kv(sl * nkv * hd), o1(sl * nh * hd), o2(sl * nh * hd);
        fill_random(g_rng, kv.data(), (int)kv.size());

        expand_kv_heads_f32(kv.data(), o1.data(), sl, nh, nkv, hd);
        ref_expand_kv_heads_f32(kv.data(), o2.data(), sl, nh, nkv, hd);
        if (!check_close_f32(o1.data(), o2.data(), (int)o1.size())) return false;
    }
    return true;
}

// ---- main ----

int main() {
    std::cout << "=== Generic (Scalar) Kernel Unit Tests ===\n\n";

    TEST(elementwise);
    TEST(norm);
    TEST(attn);
    TEST(sampling);
    TEST(kv);

    std::cout << "\n=== Results: " << g_passed << " passed, " << g_failed << " failed ===\n";
    return g_failed > 0 ? 1 : 0;
}
