#pragma once
// x86 AVX2 attention kernels (FORGE_ARCH_X86 + USE_AVX2).
// hsum_f32 / dot_f32 — extracted from attention.cpp.
// scale_f32 / fmadd_f32 — building blocks for online-softmax accumulate.

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_AVX2

// ---- hsum_f32: horizontal sum of 8 floats ----
static inline float hsum_f32(__m256 v) {
    __m128 hi128 = _mm256_extractf128_ps(v, 1);
    __m128 lo128 = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// ---- dot_f32: dot product of two float vectors ----
static inline float dot_f32(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= n; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    float sum = hsum_f32(acc);
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

// ---- scale_f32: out[i] *= scale for i in [0, n) ----
static inline void scale_f32(float* data, int n, float scale) {
    __m256 r = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_mul_ps(a, r));
    }
    for (; i < n; ++i)
        data[i] *= scale;
}

// ---- fmadd_f32: acc[i] += src[i] * weight for i in [0, n) ----
static inline void fmadd_f32(float* acc, const float* src, int n, float weight) {
    __m256 w_vec = _mm256_set1_ps(weight);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(acc + i);
        __m256 vr = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(w_vec, vr, a));
    }
    for (; i < n; ++i)
        acc[i] += weight * src[i];
}

// ---- mul_vec_f32: dst[i] *= v[i] for i in [0, n) ----
static inline void mul_vec_f32(float* dst, const float* v, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(dst + i);
        __m256 b = _mm256_loadu_ps(v + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(a, b));
    }
    for (; i < n; ++i) dst[i] *= v[i];
}

// ---- simd_gemm: C[M x N] += A[M x K] * B[K x N] (row-major, N contiguous) ----
// Accumulating GEMM used by tiled prefill attention. Adapted from llama.cpp
// ggml-cpu/simd-gemm.h (AVX2, RM=6 x RN=2 micro-kernel). No internal memset and
// no internal OpenMP, so it is safe to call from a per-head parallel region.
static inline void simd_gemm_ukernel(float* C, const float* A, const float* B, int K, int N) {
    constexpr int KN = 8;
    __m256 acc[6][2];
    for (int i = 0; i < 6; ++i)
        for (int r = 0; r < 2; ++r)
            acc[i][r] = _mm256_loadu_ps(C + i * N + r * KN);
    for (int kk = 0; kk < K; ++kk) {
        __m256 Bv0 = _mm256_loadu_ps(B + kk * N);
        __m256 Bv1 = _mm256_loadu_ps(B + kk * N + KN);
        for (int i = 0; i < 6; ++i) {
            __m256 p = _mm256_set1_ps(A[i * K + kk]);
            // NOTE: operand order matters -- FMA adds the accumulator as the
            // 3rd argument (matches llama.cpp GGML_F32x8_FMA(a,b,c)=b*c+a):
            //   acc += p * Bv  ==  _mm256_fmadd_ps(Bv, p, acc)
            acc[i][0] = _mm256_fmadd_ps(Bv0, p, acc[i][0]);
            acc[i][1] = _mm256_fmadd_ps(Bv1, p, acc[i][1]);
        }
    }
    for (int i = 0; i < 6; ++i) {
        _mm256_storeu_ps(C + i * N, acc[i][0]);
        _mm256_storeu_ps(C + i * N + KN, acc[i][1]);
    }
}

static inline void simd_gemm_ukernel_1(float* C, const float* A, const float* B, int K, int N) {
    constexpr int KN = 8;
    __m256 acc[6];
    for (int i = 0; i < 6; ++i) acc[i] = _mm256_loadu_ps(C + i * N);
    for (int kk = 0; kk < K; ++kk) {
        __m256 Bv = _mm256_loadu_ps(B + kk * N);
        for (int i = 0; i < 6; ++i) {
            __m256 p = _mm256_set1_ps(A[i * K + kk]);
            acc[i] = _mm256_fmadd_ps(Bv, p, acc[i]);
        }
    }
    for (int i = 0; i < 6; ++i) _mm256_storeu_ps(C + i * N, acc[i]);
}

static inline void simd_gemm(float* C, const float* A, const float* B, int M, int K, int N) {
    constexpr int KN = 8;
    int ii = 0;
    for (; ii + 6 <= M; ii += 6) {
        int jj = 0;
        for (; jj + 2 * KN <= N; jj += 2 * KN)
            simd_gemm_ukernel(C + jj, A, B + jj, K, N);
        for (; jj + KN <= N; jj += KN)
            simd_gemm_ukernel_1(C + jj, A, B + jj, K, N);
        for (; jj < N; ++jj)
            for (int i = 0; i < 6; ++i) {
                float* c = C + i * N + jj;
                const float* a = A + i * K;
                float v = *c;
                for (int kk = 0; kk < K; ++kk) v += a[kk] * B[kk * N + jj];
                *c = v;
            }
        A += 6 * K;
        C += 6 * N;
    }
    for (; ii < M; ++ii) {
        float* c_row = C;
        const float* a_row = A;
        for (int kk = 0; kk < K; ++kk) {
            float av = a_row[kk];
            const float* b_row = B + kk * N;
            for (int jj = 0; jj < N; ++jj) c_row[jj] += av * b_row[jj];
        }
        A += K;
        C += N;
    }
}

#endif  // USE_AVX2

#ifndef USE_AVX2
// Scalar fallback (only compiled if this header is ever included without AVX2).
static inline float dot_f32(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
static inline void scale_f32(float* data, int n, float scale) {
    for (int i = 0; i < n; ++i) data[i] *= scale;
}
static inline void fmadd_f32(float* acc, const float* src, int n, float weight) {
    for (int i = 0; i < n; ++i) acc[i] += weight * src[i];
}
static inline void mul_vec_f32(float* dst, const float* v, int n) {
    for (int i = 0; i < n; ++i) dst[i] *= v[i];
}
static inline void simd_gemm(float* C, const float* A, const float* B, int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        const float* a_row = A + (size_t)m * K;
        float* c_row = C + (size_t)m * N;
        for (int kk = 0; kk < K; ++kk) {
            float av = a_row[kk];
            const float* b_row = B + (size_t)kk * N;
            for (int n = 0; n < N; ++n) c_row[n] += av * b_row[n];
        }
    }
}
#endif  // !USE_AVX2

}  // namespace cpu
}  // namespace forge