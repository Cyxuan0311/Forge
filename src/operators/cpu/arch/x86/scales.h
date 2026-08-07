#pragma once
// AVX2 scale-decode / shuffle-mask helpers for K-quant dot kernels.
// Split out of the old cpu/quant_helpers.h — these are x86-SIMD specific and
// belong to the arch/x86 variant (see cpu_arch_split_plan.md).

#include <cstdint>
#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

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
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,
        2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,
        5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,
        8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
        13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}

// Helper: generate 256-bit shuffle mask for Q3_K scale extraction
// Used by dot_q3_K_q8_K_avx2 to broadcast 16-bit scales across 32-byte vectors
static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
         2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
         4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
         6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
         8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

// Helper: generate 256-bit shuffle mask for Q4_K / Q5_K scale broadcast
// Each 16-bit scale is duplicated across 16 bytes of the 256-bit vector
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
        0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
        0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
        2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  4,  5,
        4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
        4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
        6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,  8,  9,
        8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
        8,  9,  8,  9,  8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
    static_assert(sizeof(k_shuffle) == 256, "k_shuffle must have exactly 256 entries");
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge