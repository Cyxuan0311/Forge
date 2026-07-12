#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_AVX2
#include <immintrin.h>
#endif

namespace nanoinfer {
namespace cpu {

// Q8_K block structure (256 elements)
struct block_q8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

// Q6_K block structure (256 elements, 210 bytes)
struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
};

// Quantize one FP32 row to Q8_K format
static void quantize_row_q8_K(const float* src, block_q8_K* dst, int k) {
    constexpr int QK_K = 256;
    const int nb = (k + QK_K - 1) / QK_K;
    for (int bi = 0; bi < nb; ++bi) {
        int base = bi * QK_K;
        int n_el = std::min(QK_K, k - base);
        float amax = 0.0f;
        for (int j = 0; j < n_el; ++j) {
            float v = std::abs(src[base + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        dst[bi].d = d;
        int sum[16] = {0};
        for (int j = 0; j < n_el; ++j) {
            int q = (int)(src[base + j] * id + (src[base + j] >= 0 ? 0.5f : -0.5f));
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            dst[bi].qs[j] = (int8_t)q;
            sum[j / 16] += q;
        }
        for (int j = n_el; j < QK_K; ++j)
            dst[bi].qs[j] = 0;
        for (int j = 0; j < 16; ++j)
            dst[bi].bsums[j] = (int16_t)sum[j];
    }
}

// Round-robin parallel_for
template<typename Fn>
static void parallel_for_steal(int total, int chunk_size, Fn&& fn) {
    if (total <= 0) return;
    int n_chunks = (total + chunk_size - 1) / chunk_size;
#ifdef _OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n_threads = omp_get_num_threads();
        for (int c = tid; c < n_chunks; c += n_threads) {
            int start = c * chunk_size;
            int end = start + chunk_size;
            if (end > total) end = total;
            fn(start, end);
        }
    }
#else
    fn(0, total);
#endif
}

#ifdef USE_AVX2
// Decode Q4_K 6-bit scales: 2 bytes per 32 elements -> 6-bit scale (sc) and min (mn)
static inline void decode_q4_k_scales(const uint8_t* scales, uint8_t* sc, uint8_t* mn) {
    for (int i = 0; i < 4; ++i) {
        sc[i] = scales[i] & 63;
        mn[i] = scales[i + 4] & 63;
    }
    sc[4] = (scales[8] & 0xF) | ((scales[4] >> 6) << 4);
    mn[4] = (scales[8] >> 4) | ((scales[0] >> 6) << 4);
}

// Helper: generate 128-bit shuffle mask for Q6_K / Q3_K scale extraction
// Duplicates scale[i*2] into lower 8 bytes, scale[i*2+1] into upper 8 bytes
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
#endif // USE_AVX2

} // namespace cpu
} // namespace nanoinfer
