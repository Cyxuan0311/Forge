#pragma once
// x86 AVX2 quantized kernel implementations (FORGE_ARCH_X86).
// Moved from matmul.cpp during the arch-split refactor (see cpu_arch_split_plan.md).
// IQ2_S / IQ2_XS / IQ3_S / IQ4_NL fused dot-product and GEMV kernels.

#ifdef USE_AVX2

#include <immintrin.h>
#include <cmath>
#include <vector>

#include "../../common/quant_helpers.h"
#include "../../common/quant_tables.h"
#include "scales.h"
#include "vec.h"
#include "forge/types.h"

namespace forge {
namespace ops {

// Helper for llama.cpp-compatible scale broadcast (used by IQ2_XS/IQ2_S).
static inline __m256i make_m128i_si256(__m128i lo, __m128i hi) {
    return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

// Expand 4 sign bytes (32 sign bits) into a 32-byte sign vector for
// _mm256_sign_epi8: byte j = 0x01 if bit (j%8) of sign byte (j/8) is clear,
// 0xFF if set.  Used by the vectorized IQ2_S / IQ3_S dot kernels.
static inline __m256i expand_iq_signs_32(const uint8_t* sign_bytes) {
    static const int8_t ctl_lo[16] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
    static const int8_t ctl_hi[16] = {2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3};
    static const uint8_t bit_sel[32] = {
        0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80, 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,
        0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80, 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,
    };

    uint32_t sb;
    memcpy(&sb, sign_bytes, 4);
    const __m128i s4 = _mm_cvtsi32_si128((int)sb);
    const __m128i r_lo = _mm_shuffle_epi8(s4, _mm_loadu_si128((const __m128i*)ctl_lo));
    const __m128i r_hi = _mm_shuffle_epi8(s4, _mm_loadu_si128((const __m128i*)ctl_hi));
    const __m256i sdup = make_m128i_si256(r_lo, r_hi);

    const __m256i sel = _mm256_loadu_si256((const __m256i*)bit_sel);
    const __m256i m = _mm256_and_si256(sdup, sel);
    const __m256i s = _mm256_cmpeq_epi8(m, sel);      // 0xFF/0x00
    return _mm256_or_si256(s, _mm256_set1_epi8(1));   // 0xFF/0x01 for sign_epi8
}

// ============================================================================
// IQ2_S x Q8_K fused dot product (AVX2)
// IQ2_S block: 82 bytes = d[2] + qs[32B] + signs[32B] + qh[8B] + sc[8B]
// ============================================================================

static inline float dot_iq2_s_q8_K_avx2(const uint8_t* iq2s_row,
                                        const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ2_S_BLOCK_SIZE = 82;

    __m256 accumf = _mm256_setzero_ps();

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq2s_row + (size_t)bi * IQ2_S_BLOCK_SIZE;
        const float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr)) * q8[bi].d;
        const uint8_t* qs = block_ptr + 2;      // 32 value bytes (4 per ib32)
        const uint8_t* signs = block_ptr + 34;  // 32 sign bytes  (4 per ib32)
        const uint8_t* qh = block_ptr + 66;     // 8 high-bit bytes (1 per ib32)
        const uint8_t* sc = block_ptr + 74;     // 8 scale bytes   (1 per ib32)
        const int8_t* q8d = q8[bi].qs;

        __m256i sumi = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const uint8_t qhb = qh[ib32];
            const int idx0 = qs[0] | ((qhb << 8) & 0x300);
            const int idx1 = qs[1] | ((qhb << 6) & 0x300);
            const int idx2 = qs[2] | ((qhb << 4) & 0x300);
            const int idx3 = qs[3] | ((qhb << 2) & 0x300);
            qs += 4;

            const __m256i g = _mm256_set_epi64x(iq2s_grid[idx3], iq2s_grid[idx2],
                                                iq2s_grid[idx1], iq2s_grid[idx0]);
            const __m256i sm = expand_iq_signs_32(signs);
            signs += 4;

            const __m256i q8v = _mm256_loadu_si256((const __m256i*)(q8d + ib32 * 32));
            const __m256i q8s = _mm256_sign_epi8(q8v, sm);

            const __m256i dot16 = _mm256_maddubs_epi16(g, q8s);

            // Scales: (0.5+n)*0.25 == (2*n+1)*0.125; fold the (2*n+1) part into
            // the integer dot and apply 0.125*d at block end.
            const int s0 = 2 * (sc[ib32] & 0xf) + 1;
            const int s1 = 2 * (sc[ib32] >> 4) + 1;
            const __m256i scv = _mm256_set_m128i(_mm_set1_epi16((short)s1),
                                                 _mm_set1_epi16((short)s0));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot16, scv));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d * 0.125f), _mm256_cvtepi32_ps(sumi), accumf);
    }

    return cpu::hsum_avx2(accumf);
}

static void gemv_iq2_s_q8k_transB_avx2(const float* a, const uint8_t* w, float* out,
                                       int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ2_S_BLOCK_SIZE = 82;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_S_BLOCK_SIZE;
            out[n] = dot_iq2_s_q8_K_avx2(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_S_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[m * N + n] = dot_iq2_s_q8_K_avx2(row, q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ2_XS x Q8_K fused dot product (AVX2)
// IQ2_XS block: 74 bytes = d[2] + qs[32 uint16_t = 64B] + scales[8B]
// ============================================================================

static inline float dot_iq2_xs_q8_K_avx2(const uint8_t* iq2xs_row,
                                         const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;

    // Sign expansion masks (ported from llama.cpp ggml-cpu/arch/x86/quants.c)
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };
    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };

    const __m256i mone = _mm256_set1_epi8(1);
    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i*)bit_selector_mask_bytes);
    const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_1);
    const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_2);
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i*)k_bit_helper);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    __m256 accumf = _mm256_setzero_ps();

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq2xs_row + (size_t)bi * 74;
        const float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr)) * q8[bi].d;
        const uint16_t* q2 = reinterpret_cast<const uint16_t*>(block_ptr + 2);
        const int8_t* q8d = q8[bi].qs;

        // Decode 8 scales -> 16 int8 scales (2*scale+1), matching llama.cpp
        uint64_t aux64;
        memcpy(&aux64, block_ptr + 66, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 4) {
            const __m256i q2_data = _mm256_loadu_si256((const __m256i*)q2);
            q2 += 16;

            // 16 grid indices (uint16, values 0..511) packed in 32-bit lanes.
            // Widen to int32 (the gather uses full 32-bit indices), then gather
            // the 8-byte grid entries directly with 4 x _mm256_i32gather_epi64
            // instead of spilling to the stack and doing 16 scalar loads.
            const __m256i g16 = _mm256_and_si256(q2_data, m511);
            const __m256i lo32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g16)); // indices 0-7
            const __m256i hi32 = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(g16, 1)); // 8-15
            const __m256i q2_1 = _mm256_i32gather_epi64((const long long*)iq2xs_grid,
                                                        _mm256_castsi256_si128(lo32), 8);
            const __m256i q2_2 = _mm256_i32gather_epi64((const long long*)iq2xs_grid,
                                                        _mm256_extractf128_si256(lo32, 1), 8);
            const __m256i q2_3 = _mm256_i32gather_epi64((const long long*)iq2xs_grid,
                                                        _mm256_castsi256_si128(hi32), 8);
            const __m256i q2_4 = _mm256_i32gather_epi64((const long long*)iq2xs_grid,
                                                        _mm256_extractf128_si256(hi32, 1), 8);

            // Sign bits: low 3 bits select ksigns_iq2xs entry, high bit via xors
            const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            const __m256i partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
            const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
            const __m256i q8_4 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;

            const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            const __m256i full_signs_1 = make_m128i_si256(full_signs_l, full_signs_l);
            const __m256i full_signs_2 = make_m128i_si256(full_signs_h, full_signs_h);

            __m256i signs;
            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const __m256i dot3 = _mm256_maddubs_epi16(q2_3, q8s_3);
            const __m256i dot4 = _mm256_maddubs_epi16(q2_4, q8s_4);

            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, cpu::get_scale_shuffle(ib32 + 0)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, cpu::get_scale_shuffle(ib32 + 1)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, cpu::get_scale_shuffle(ib32 + 2)));
            const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, cpu::get_scale_shuffle(ib32 + 3)));

            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    return 0.125f * cpu::hsum_avx2(accumf);
}

static void gemv_iq2_xs_q8k_transB_avx2(const float* a, const uint8_t* w, float* out,
                                        int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ2_XS_BLOCK_SIZE = 74;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_XS_BLOCK_SIZE;
            out[n] = dot_iq2_xs_q8_K_avx2(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_XS_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[m * N + n] = dot_iq2_xs_q8_K_avx2(row, q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ3_S x Q8_K fused dot product (AVX2)
// IQ3_S block: 110 bytes = d[2] + qs[64B] + qh[8B] + signs[32B] + scales[4B]
// ============================================================================

// Build the 4 grid rows (each 8 bytes = [g1(4), g2(4)]) for one 32-element
// IQ3_S sub-block, using 8 qs bytes + one qh byte (9-bit indices).
static inline __m256i iq3s_grid_vec(const uint8_t* qs8, uint8_t qhb) {
    const uint64_t row0 = ((uint64_t)iq3s_grid[qs8[1] | ((qhb << 7) & 256)] << 32) |
                           (uint64_t)iq3s_grid[qs8[0] | ((qhb << 8) & 256)];
    const uint64_t row1 = ((uint64_t)iq3s_grid[qs8[3] | ((qhb << 5) & 256)] << 32) |
                           (uint64_t)iq3s_grid[qs8[2] | ((qhb << 6) & 256)];
    const uint64_t row2 = ((uint64_t)iq3s_grid[qs8[5] | ((qhb << 3) & 256)] << 32) |
                           (uint64_t)iq3s_grid[qs8[4] | ((qhb << 4) & 256)];
    const uint64_t row3 = ((uint64_t)iq3s_grid[qs8[7] | ((qhb << 1) & 256)] << 32) |
                           (uint64_t)iq3s_grid[qs8[6] | ((qhb << 2) & 256)];
    return _mm256_set_epi64x(row3, row2, row1, row0);
}

static inline float dot_iq3_s_q8_K_avx2(const uint8_t* iq3s_row,
                                        const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ3_S_BLOCK_SIZE = 110;

    __m256 accumf = _mm256_setzero_ps();

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq3s_row + (size_t)bi * IQ3_S_BLOCK_SIZE;
        const float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr)) * q8[bi].d;
        const uint8_t* qs = block_ptr + 2;
        const uint8_t* qh = block_ptr + 2 + 64;
        const uint8_t* signs = block_ptr + 2 + 64 + 8;
        const uint8_t* sc = block_ptr + 2 + 64 + 8 + 32;
        const int8_t* q8d = q8[bi].qs;

        __m256i sumi = _mm256_setzero_si256();

        for (int k = 0; k < QK_K / 64; ++k) {
            const int sA = 2 * (sc[k] & 0xf) + 1;
            const int sB = 2 * (sc[k] >> 4) + 1;

            const __m256i gA = iq3s_grid_vec(qs + 16 * k, qh[2 * k]);
            const __m256i smA = expand_iq_signs_32(signs + 8 * k);
            const __m256i q8a = _mm256_loadu_si256((const __m256i*)(q8d + 64 * k));
            const __m256i q8sa = _mm256_sign_epi8(q8a, smA);
            const __m256i dotA = _mm256_maddubs_epi16(gA, q8sa);
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dotA, _mm256_set1_epi16((short)sA)));

            const __m256i gB = iq3s_grid_vec(qs + 16 * k + 8, qh[2 * k + 1]);
            const __m256i smB = expand_iq_signs_32(signs + 8 * k + 4);
            const __m256i q8b = _mm256_loadu_si256((const __m256i*)(q8d + 64 * k + 32));
            const __m256i q8sb = _mm256_sign_epi8(q8b, smB);
            const __m256i dotB = _mm256_maddubs_epi16(gB, q8sb);
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dotB, _mm256_set1_epi16((short)sB)));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accumf);
    }

    return cpu::hsum_avx2(accumf);
}

static void gemv_iq3_s_q8k_transB_avx2(const float* a, const uint8_t* w, float* out,
                                       int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ3_S_BLOCK_SIZE = 110;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ3_S_BLOCK_SIZE;
            out[n] = dot_iq3_s_q8_K_avx2(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ3_S_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[m * N + n] = dot_iq3_s_q8_K_avx2(row, q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ4_NL x Q8_K fused dot product (AVX2)
// IQ4_NL block: 18 bytes = d[2] + qs[16B] per 32 elements
// ============================================================================

static inline float dot_iq4_nl_q8_K_avx2(const uint8_t* iq4nl_row,
                                         const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ4_NL_BLOCK_SIZE = 18;
    constexpr int IQ4_NL_BLOCK_EL = 32;

    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i kvalues = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        const uint8_t* block_ptr = iq4nl_row + (size_t)bi * (QK_K / IQ4_NL_BLOCK_EL) * IQ4_NL_BLOCK_SIZE;
        float block_acc = 0.0f;

        for (int sb = 0; sb < QK_K / IQ4_NL_BLOCK_EL; ++sb) {
            float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr));
            const uint8_t* qs = block_ptr + 2;

            __m128i qs_vec = _mm_loadu_si128((const __m128i*)qs);
            __m128i low_nibbles = _mm_and_si128(qs_vec, m4);
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(qs_vec, 4), m4);

            __m128i vals_low = _mm_shuffle_epi8(kvalues, low_nibbles);
            __m128i vals_high = _mm_shuffle_epi8(kvalues, high_nibbles);

            __m128i q8_low = _mm_loadu_si128((const __m128i*)(q8d + sb * 32));
            __m128i q8_high = _mm_loadu_si128((const __m128i*)(q8d + sb * 32 + 16));

            __m256i vals_lo16 = _mm256_cvtepi8_epi16(vals_low);
            __m256i q8_lo16 = _mm256_cvtepi8_epi16(q8_low);
            __m256i p_lo = _mm256_madd_epi16(vals_lo16, q8_lo16);

            __m256i vals_hi16 = _mm256_cvtepi8_epi16(vals_high);
            __m256i q8_hi16 = _mm256_cvtepi8_epi16(q8_high);
            __m256i p_hi = _mm256_madd_epi16(vals_hi16, q8_hi16);

            __m256i p32 = _mm256_add_epi32(p_lo, p_hi);
            __m128i lo128 = _mm256_extracti128_si256(p32, 0);
            __m128i hi128 = _mm256_extracti128_si256(p32, 1);
            __m128i h = _mm_hadd_epi32(lo128, hi128);
            h = _mm_hadd_epi32(h, h);
            h = _mm_hadd_epi32(h, h);
            const int sum = _mm_cvtsi128_si32(h);

            block_acc += d * (float)sum;
            block_ptr += IQ4_NL_BLOCK_SIZE;
        }

        acc += d_q8 * block_acc;
    }

    return acc;
}

// IQ4_XS x Q8_K dot: 256-element block (136 bytes), 8 sub-blocks of 32.
// Sub-block scale (6-bit): ls = (scales_l[g] nibble) | (scales_h 2 bits), dl = d*(ls-32).
// Matches llama.cpp ggml_vec_dot_iq4_xs_q8_K_generic bit layout.
static inline int dot32_iq4xs_avx2(const uint8_t* qs, const int8_t* q8d, const __m128i& kvalues,
                                   const __m128i& m4) {
    __m128i qs_vec = _mm_loadu_si128((const __m128i*)qs);
    __m128i low_nibbles = _mm_and_si128(qs_vec, m4);
    __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(qs_vec, 4), m4);

    __m128i vals_low = _mm_shuffle_epi8(kvalues, low_nibbles);
    __m128i vals_high = _mm_shuffle_epi8(kvalues, high_nibbles);

    __m128i q8_low = _mm_loadu_si128((const __m128i*)q8d);
    __m128i q8_high = _mm_loadu_si128((const __m128i*)(q8d + 16));

    __m256i p32 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_cvtepi8_epi16(vals_low),
                                                     _mm256_cvtepi8_epi16(q8_low)),
                                   _mm256_madd_epi16(_mm256_cvtepi8_epi16(vals_high),
                                                     _mm256_cvtepi8_epi16(q8_high)));
    __m128i lo128 = _mm256_extracti128_si256(p32, 0);
    __m128i hi128 = _mm256_extracti128_si256(p32, 1);
    __m128i h = _mm_hadd_epi32(lo128, hi128);
    h = _mm_hadd_epi32(h, h);
    h = _mm_hadd_epi32(h, h);
    return _mm_cvtsi128_si32(h);
}

static inline float dot_iq4_xs_q8_K_avx2(const uint8_t* iq4xs_row,
                                         const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ4_XS_BLOCK_SIZE = 136;

    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i kvalues = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq4xs_row + (size_t)bi * IQ4_XS_BLOCK_SIZE;
        const cpu::block_q8_K* y = q8 + bi;
        const float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr));
        const float d_q8 = y->d;
        uint16_t h = *reinterpret_cast<const uint16_t*>(block_ptr + 2);
        const uint8_t* scales_l = block_ptr + 4;
        const uint8_t* qs = block_ptr + 8;
        const int8_t* q8d = y->qs;

        float block_acc = 0.0f;
        for (int g = 0; g < QK_K / 64; ++g) {
            const uint8_t ls1 = static_cast<uint8_t>(scales_l[g] & 0xf) |
                                static_cast<uint8_t>((h << 4) & 0x30);
            const uint8_t ls2 = static_cast<uint8_t>(scales_l[g] >> 4) |
                                static_cast<uint8_t>((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d * static_cast<float>(ls1 - 32);
            const float d2 = d * static_cast<float>(ls2 - 32);
            block_acc += d1 * dot32_iq4xs_avx2(qs + g * 32, q8d + g * 64, kvalues, m4);
            block_acc += d2 * dot32_iq4xs_avx2(qs + g * 32 + 16, q8d + g * 64 + 32, kvalues, m4);
        }

        acc += d_q8 * block_acc;
    }

    return acc;
}

static void gemv_iq4_xs_q8k_transB_avx2(const float* a, const uint8_t* w, float* out,
                                        int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ4_XS_BLOCK_SIZE = 136;
    const int nb = (K + QK_K - 1) / QK_K;
    const size_t row_bytes = (size_t)nb * IQ4_XS_BLOCK_SIZE;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * row_bytes;
            out[n] = dot_iq4_xs_q8_K_avx2(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * row_bytes;
            for (int m = 0; m < M; ++m)
                out[m * N + n] = dot_iq4_xs_q8_K_avx2(row, q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

}  // namespace ops
}  // namespace forge

namespace forge {
namespace cpu {

static void gemv_iq4_xs_ffn_down_residual_avx2(const float* a, const uint8_t* w,
                                               const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ4_XS_BLOCK_SIZE = 136;
    const int nb = (K + QK_K - 1) / QK_K;

    scratch_vec<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = w + (size_t)n * nb * IQ4_XS_BLOCK_SIZE;
        out[n] = ops::dot_iq4_xs_q8_K_avx2(row, q8_buf.data(), nb) + residual[n];
    }
}

static void gemv_iq4_xs_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate,
                                          const uint8_t* w_up, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ4_XS_BLOCK_SIZE = 136;
    const int nb = (K + QK_K - 1) / QK_K;

    scratch_vec<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * IQ4_XS_BLOCK_SIZE;
        const uint8_t* up_row = w_up + (size_t)n * nb * IQ4_XS_BLOCK_SIZE;

        float gate_val = silu(ops::dot_iq4_xs_q8_K_avx2(gate_row, q8_buf.data(), nb));
        float up_val = ops::dot_iq4_xs_q8_K_avx2(up_row, q8_buf.data(), nb);
        out[n] = gate_val * up_val;
    }
}

}  // namespace cpu
}  // namespace forge

namespace forge {
namespace ops {

static void gemv_iq4_nl_q8k_transB_avx2(const float* a, const uint8_t* w, float* out,
                                        int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ4_NL_BLOCK_SIZE = 18;
    constexpr int IQ4_NL_BLOCK_EL = 32;
    const int nb = (K + QK_K - 1) / QK_K;
    const size_t row_bytes = (size_t)(K + IQ4_NL_BLOCK_EL - 1) / IQ4_NL_BLOCK_EL * IQ4_NL_BLOCK_SIZE;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * row_bytes;
            out[n] = dot_iq4_nl_q8_K_avx2(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * row_bytes;
            for (int m = 0; m < M; ++m)
                out[m * N + n] = dot_iq4_nl_q8_K_avx2(row, q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// DotQ8KFn dispatcher for batched MoE matmul
// ============================================================================

using DotQ8KFn = float (*)(const uint8_t*, const cpu::block_q8_K*, int);

static inline DotQ8KFn get_dot_q8k_fn(DataType dt) {
    switch (dt) {
    case DataType::IQ2_XS: return dot_iq2_xs_q8_K_avx2;
    case DataType::IQ3_S:  return dot_iq3_s_q8_K_avx2;
    case DataType::IQ4_NL: return dot_iq4_nl_q8_K_avx2;
    case DataType::IQ4_XS: return dot_iq4_xs_q8_K_avx2;
    case DataType::IQ2_S:  return dot_iq2_s_q8_K_avx2;
    default: return nullptr;
    }
}

// ============================================================================
// Q4_K x Q8_K fused dot product (AVX2) — used by mixed-precision QKV fusion
// ============================================================================

static inline float dot_q4_K_q8_K_row_avx2(const uint8_t* q4_row, const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    for (int i = 0; i < nb; ++i) {
        const cpu::block_q4_K* x = reinterpret_cast<const cpu::block_q4_K*>(q4_row) + i;
        const cpu::block_q8_K* y = q8 + i;

        _mm_prefetch((const char*)((const cpu::block_q4_K*)q4_row + i + 1), _MM_HINT_T0);
        _mm_prefetch((const char*)(q8 + i + 1), _MM_HINT_T0);

        const float d = y->d * cpu::fp16_to_float_scalar(x->d);
        const float dmin = -y->d * cpu::fp16_to_float_scalar(x->dmin);

        uint32_t utmp[4];
        memcpy(utmp, x->scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                           _mm256_extracti128_si256(q8sums, 1));
        const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i prod = _mm_madd_epi16(mins128, q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);

        const uint8_t* q4 = x->qs;
        const int8_t* q8d = y->qs;
        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, cpu::get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, cpu::get_scale_shuffle_k4(2 * j + 1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
            q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8d);
            q8d += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8d);
            q8d += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    return cpu::hsum_avx2(acc) + _mm_cvtss_f32(acc_m);
}

static void gemv_q3_k_q4_k_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                           const uint8_t* wv, float* out_q, float* out_k,
                                           float* out_v, int K, int N_q, int N_k, int N_v) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<cpu::block_q8_K> q8_buf(nb);
    cpu::quantize_row_q8_K(a, q8_buf.data(), K);

    // Q and K use Q3_K Q8_K dot product
    auto dot_q3_rows = [&](const uint8_t* w, float* out, int N) {
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q3_row = w + (size_t)n * nb * Q3_K_BLOCK_BYTES;
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < nb; ++i) {
                _mm_prefetch((const char*)(q3_row + (i + 1) * Q3_K_BLOCK_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*)(q8_buf.data() + i + 1), _MM_HINT_T0);
                acc = _mm256_add_ps(
                    acc, cpu::q3_k_sb_dot_avx2(q3_row + (size_t)i * Q3_K_BLOCK_BYTES, &q8_buf[i]));
            }
            out[n] = cpu::hsum_avx2(acc);
        }
    };

    dot_q3_rows(wq, out_q, N_q);
    dot_q3_rows(wk, out_k, N_k);

    // V uses Q4_K Q8_K dot product (inline, shares same q8_buf)
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N_v; ++n) {
        const uint8_t* q4_row = wv + (size_t)n * nb * Q4_K_BLOCK_BYTES;
        out_v[n] = dot_q4_K_q8_K_row_avx2(q4_row, q8_buf.data(), nb);
    }
}

// ============================================================================
// Q5_K x Q8_K fused dot product (AVX2)
// ============================================================================

static inline float dot_q5_K_q8_K_row_avx2(const uint8_t* q5_row, const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();
    float summs = 0.f;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    for (int i = 0; i < nb; ++i) {
        const cpu::block_q5_K* x = reinterpret_cast<const cpu::block_q5_K*>(q5_row) + i;
        const cpu::block_q8_K* y = q8 + i;

        _mm_prefetch((const char*)((const cpu::block_q5_K*)q5_row + i + 1), _MM_HINT_T0);
        _mm_prefetch((const char*)(q8 + i + 1), _MM_HINT_T0);

        const uint8_t* q5 = x->ql;
        const int8_t*  q8d = y->qs;

        const float d = y->d * cpu::fp16_to_float_scalar(x->d);
        const float dmin = -y->d * cpu::fp16_to_float_scalar(x->dmin);

        memcpy(utmp, x->scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                           _mm256_extracti128_si256(q8sums, 1));
        const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i prod = _mm_madd_epi16(mins128, q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * (float)_mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x->qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, cpu::get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, cpu::get_scale_shuffle_k4(2 * j + 1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5);
            q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return cpu::hsum_avx2(acc) + summs;
}

static void gemv_q5_k_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                    int N) {
    constexpr int QK_K = 256;
    constexpr int Q5_K_BLOCK_BYTES = 176;
    constexpr int nrc = 2;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            _mm_prefetch((const char*)(w + (size_t)(n + 4) * nb * Q5_K_BLOCK_BYTES), _MM_HINT_T1);
            const uint8_t* q5_row = w + (size_t)n * nb * Q5_K_BLOCK_BYTES;
            out[n] = dot_q5_K_q8_K_row_avx2(q5_row, q8_buf.data(), nb);
        }
    } else {
        for (int m_start = 0; m_start < M; m_start += nrc) {
            int m_cur = (m_start + nrc <= M) ? nrc : (M - m_start);
            std::vector<cpu::block_q8_K> q8_tile(m_cur * nb);
            for (int m = 0; m < m_cur; ++m)
                cpu::quantize_row_q8_K(a + (m_start + m) * K, q8_tile.data() + m * nb, K);

#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n) {
                const uint8_t* q5_row = w + (size_t)n * nb * Q5_K_BLOCK_BYTES;
                for (int m = 0; m < m_cur; ++m)
                    out[(m_start + m) * N + n] = dot_q5_K_q8_K_row_avx2(q5_row, q8_tile.data() + m * nb, nb);
            }
        }
    }
}

static void gemv_q5_k_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate,
                                         const uint8_t* w_up, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q5_K_BLOCK_BYTES = 176;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<cpu::block_q8_K> q8_buf(nb);
    cpu::quantize_row_q8_K(a, q8_buf.data(), K);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * Q5_K_BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * nb * Q5_K_BLOCK_BYTES;

        float gate_val = silu(dot_q5_K_q8_K_row_avx2(gate_row, q8_buf.data(), nb));
        float up_val = dot_q5_K_q8_K_row_avx2(up_row, q8_buf.data(), nb);
        out[n] = gate_val * up_val;
    }
}

static void gemv_q2_k_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate,
                                         const uint8_t* w_up, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q2_K_BLOCK_BYTES = 84;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<cpu::block_q8_K> q8_buf(nb);
    cpu::quantize_row_q8_K(a, q8_buf.data(), K);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * Q2_K_BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * nb * Q2_K_BLOCK_BYTES;

        float gate_val = silu(cpu::dot_q2_K_q8_K_avx2(gate_row, q8_buf.data(), nb, nullptr));
        float up_val = cpu::dot_q2_K_q8_K_avx2(up_row, q8_buf.data(), nb, nullptr);
        out[n] = gate_val * up_val;
    }
}

}  // namespace ops
}  // namespace forge

#endif  // USE_AVX2