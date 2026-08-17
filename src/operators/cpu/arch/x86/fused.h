#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "gemv.h"
#include "cpu_gemv.h"
#include "vec.h"
#include "../../common/quant_helpers.h"
#ifdef _OPENMP
#    include <omp.h>
#endif
#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

// ---- Fused QKV projection for Q4_0 decode ----
// Reads input vector once, computes Q + K + V outputs simultaneously.

#ifdef USE_AVX2
static void gemv_q4_0_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                     const uint8_t* wv, float* out_q, float* out_k, float* out_v,
                                     int K, int N_q, int N_k, int N_v) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m128i eight = _mm_set1_epi8(8);

    auto process_row = [&](const float* a_row, const uint8_t* w_row, float* out, int N_out) {
#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N_out; ++n) {
            const uint8_t* row = w_row + (size_t)n * blocks_per_row * BLOCK_BYTES;
            __m256 acc = _mm256_setzero_ps();

            for (int bi = 0; bi < blocks_per_row; ++bi) {
                int base = bi * BLOCK_SIZE;
                if (K - base < BLOCK_SIZE) {
                    const uint8_t* block = row + bi * BLOCK_BYTES;
                    uint16_t sb;
                    memcpy(&sb, block, 2);
                    uint32_t sign = (sb >> 15) & 1;
                    uint32_t exponent = (sb >> 10) & 0x1F;
                    uint32_t mantissa = sb & 0x3FF;
                    float scale_f;
                    if (exponent == 0) {
                        scale_f = mantissa == 0
                                      ? 0.0f
                                      : (sign ? -1 : 1) *
                                            std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
                    } else {
                        scale_f = (sign ? -1 : 1) *
                                  std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                                             static_cast<int>(exponent) - 15);
                    }
                    const uint8_t* qs = block + 2;
                    for (int j = 0; j < 16 && base + j < K; ++j) {
                        float qv = static_cast<float>((qs[j] & 0x0F) - 8) * scale_f;
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + j]), _mm256_set1_ps(qv),
                                              acc);
                    }
                    for (int j = 0; j < 16 && base + 16 + j < K; ++j) {
                        float qv = static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale_f;
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + 16 + j]),
                                              _mm256_set1_ps(qv), acc);
                    }
                    continue;
                }

                const uint8_t* block = row + bi * BLOCK_BYTES;
                uint16_t scale_bits;
                memcpy(&scale_bits, block, 2);
                __m256 scale = fp16_to_fp32_broadcast_avx2(scale_bits);
                const uint8_t* qs = block + 2;

                __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
                __m128i q_lo = _mm_and_si128(q8, lo_mask);
                __m128i q_lo_signed = _mm_sub_epi8(q_lo, eight);
                __m128i q_hi = _mm_and_si128(_mm_srli_epi16(q8, 4), lo_mask);
                __m128i q_hi_signed = _mm_sub_epi8(q_hi, eight);

                __m256 partial =
                    _mm256_mul_ps(_mm256_loadu_ps(a_row + base),
                                  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_signed)));
                partial = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row + base + 8),
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo_signed, 8))),
                    partial);
                partial =
                    _mm256_fmadd_ps(_mm256_loadu_ps(a_row + base + 16),
                                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_signed)), partial);
                partial = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row + base + 24),
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi_signed, 8))),
                    partial);

                acc = _mm256_fmadd_ps(scale, partial, acc);
            }
            out[n] = hsum_avx2(acc);
        }
    };

    process_row(a, wq, out_q, N_q);
    process_row(a, wk, out_k, N_k);
    process_row(a, wv, out_v, N_v);
}

// ---- Fused QKV projection for Q4_0 decode, repacked layout ----
// Same as gemv_q4_0_fused_qkv_avx2 but consumes the repacked weight layout
// (repack_q4_0_weights). The activation is quantized once and shared by all
// three matrices; each matrix is decoded with the proven repacked 4-row
// tile-group body. Three sequential OpenMP regions keep the per-matrix N
// independent (N_q != N_k != N_v) while reusing the same thread team.
static void gemv_q4_0_fused_qkv_repacked_avx2(const float* a, const uint8_t* wq_repacked,
                                              const uint8_t* wk_repacked, const uint8_t* wv_repacked,
                                              float* out_q, float* out_k, float* out_v,
                                              int K, int N_q, int N_k, int N_v) {
    constexpr int RM = 4;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t nb = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    scratch_vec<block_q8_0_act> q8_act(nb);
    quantize_row_q8_0_act(a, q8_act.data(), K);

    auto decode = [&](const uint8_t* w_repacked, float* out, int N) {
        const int64_t num_groups = (N + RM - 1) / RM;
#    pragma omp parallel for schedule(static)
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t n0 = g * RM;
            int64_t rows = std::min<int64_t>(n0 + RM, N) - n0;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            const uint8_t* group_base = w_repacked + (size_t)g * nb * RM * BLOCK_BYTES;

            for (int64_t l = 0; l < nb; ++l) {
                if (l + 4 < nb) {
                    _mm_prefetch((const char*)(group_base + (l + 4) * RM * BLOCK_BYTES), _MM_HINT_T0);
                }

                const uint8_t* block_base = group_base + l * RM * BLOCK_BYTES;

                __m256i wvec0 = BlockLoader<block_q4_0_tag>::load(block_base + 0 * BLOCK_BYTES);
                __m256i wvec1 = (rows > 1) ? BlockLoader<block_q4_0_tag>::load(block_base + 1 * BLOCK_BYTES) : _mm256_setzero_si256();
                __m256i wvec2 = (rows > 2) ? BlockLoader<block_q4_0_tag>::load(block_base + 2 * BLOCK_BYTES) : _mm256_setzero_si256();
                __m256i wvec3 = (rows > 3) ? BlockLoader<block_q4_0_tag>::load(block_base + 3 * BLOCK_BYTES) : _mm256_setzero_si256();

                __m256i avec = BlockLoader<block_q8_0_act>::load(&q8_act[l]);

                uint64_t packed_scales = 0;
                uint16_t d0, d1, d2, d3;
                memcpy(&d0, block_base + 0 * BLOCK_BYTES, 2);
                memcpy(&d1, block_base + 1 * BLOCK_BYTES, 2);
                memcpy(&d2, block_base + 2 * BLOCK_BYTES, 2);
                memcpy(&d3, block_base + 3 * BLOCK_BYTES, 2);
                packed_scales = (uint64_t)d0 | ((uint64_t)d1 << 16) | ((uint64_t)d2 << 32) | ((uint64_t)d3 << 48);

                __m128 sw_f16 = _mm_cvtph_ps(_mm_set_epi64x(0, packed_scales));
                float act_scale = q8_act[l].d;
                __m128 sw_all = _mm_mul_ps(sw_f16, _mm_set1_ps(act_scale));

                __m256 sc0 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x00));
                sc0 = _mm256_permute2f128_ps(sc0, sc0, 0x00);
                __m256 sc1 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x55));
                sc1 = _mm256_permute2f128_ps(sc1, sc1, 0x00);
                __m256 sc2 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xAA));
                sc2 = _mm256_permute2f128_ps(sc2, sc2, 0x00);
                __m256 sc3 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xFF));
                sc3 = _mm256_permute2f128_ps(sc3, sc3, 0x00);

                __m256i sa0 = _mm256_sign_epi8(wvec0, wvec0);
                __m256i sa1 = (rows > 1) ? _mm256_sign_epi8(wvec1, wvec1) : _mm256_setzero_si256();
                __m256i sa2 = (rows > 2) ? _mm256_sign_epi8(wvec2, wvec2) : _mm256_setzero_si256();
                __m256i sa3 = (rows > 3) ? _mm256_sign_epi8(wvec3, wvec3) : _mm256_setzero_si256();

                __m256i sb0 = _mm256_sign_epi8(avec, wvec0);
                __m256i sb1 = (rows > 1) ? _mm256_sign_epi8(avec, wvec1) : _mm256_setzero_si256();
                __m256i sb2 = (rows > 2) ? _mm256_sign_epi8(avec, wvec2) : _mm256_setzero_si256();
                __m256i sb3 = (rows > 3) ? _mm256_sign_epi8(avec, wvec3) : _mm256_setzero_si256();

                acc0 = _mm256_fmadd_ps(sc0, updot_avx2(sa0, sb0), acc0);
                acc1 = _mm256_fmadd_ps(sc1, updot_avx2(sa1, sb1), acc1);
                acc2 = _mm256_fmadd_ps(sc2, updot_avx2(sa2, sb2), acc2);
                acc3 = _mm256_fmadd_ps(sc3, updot_avx2(sa3, sb3), acc3);
            }

            out[n0 + 0] = vdot::hsum_ps256(acc0);
            if (rows > 1) out[n0 + 1] = vdot::hsum_ps256(acc1);
            if (rows > 2) out[n0 + 2] = vdot::hsum_ps256(acc2);
            if (rows > 3) out[n0 + 3] = vdot::hsum_ps256(acc3);
        }
    };

    decode(wq_repacked, out_q, N_q);
    decode(wk_repacked, out_k, N_k);
    decode(wv_repacked, out_v, N_v);
}

// Same as gemv_q4_0_fused_qkv_repacked_avx2 but consumes the llama-style
// block_q4_0x8 layout (repack_q4_0_weights_8x8). Requires N_q, N_k, N_v all
// divisible by 8 (checked at dispatch time via get_repacked_q4_0_8x8).
static void gemv_q4_0_fused_qkv_8x8_avx2(const float* a, const uint8_t* wq_8x8,
                                         const uint8_t* wk_8x8, const uint8_t* wv_8x8,
                                         float* out_q, float* out_k, float* out_v,
                                         int64_t K, int64_t N_q, int64_t N_k, int64_t N_v) {
    constexpr int RM = 8;
    constexpr int BLOCK_SIZE = 32;
    const int64_t nb = K / BLOCK_SIZE;

    scratch_vec<block_q8_0_act> q8_act(nb);
    quantize_row_q8_0_act(a, q8_act.data(), (int)K);

    const __m128i lut128 = _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8,
                                        7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i lut = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(lut128), _mm256_castsi128_si256(lut128), 0);
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i finalperm = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    auto decode = [&](const uint8_t* w_8x8, float* out, int64_t N) {
        const int64_t num_groups = N / RM;
#    pragma omp parallel for schedule(static)
        for (int64_t g = 0; g < num_groups; ++g) {
            __m256 acc_row = forge::cpu::gemv_q4_0_8x8_dot_group(
                w_8x8 + g * nb * sizeof(block_q4_0x8), q8_act.data(), nb, lut, m4b);
            _mm256_storeu_ps(out + g * RM, _mm256_permutevar8x32_ps(acc_row, finalperm));
        }
    };

    decode(wq_8x8, out_q, N_q);
    decode(wk_8x8, out_k, N_k);
    decode(wv_8x8, out_v, N_v);
}
#endif  // USE_AVX2

// ---- Fused FFN gate+up projection for Q4_0 decode ----
// Reads input vector once, computes both gate and up projections simultaneously.
// Then applies SiLU(gate) * up in-place.

#ifdef USE_AVX2
static void gemv_q4_0_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate, const uint8_t* w_up,
                                        float* out, int K, int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Quantize activation once — shared by both gate and up
    scratch_vec<block_q8_0_act> q8_act(blocks_per_row);
    quantize_row_q8_0_act(a, q8_act.data(), K);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

    constexpr int RM = 4;
    constexpr int TILE_NR = 8;
#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n += TILE_NR) {
        int rows = (n + TILE_NR <= N) ? TILE_NR : (N - n);

        for (int t = 0; t < rows; t += RM) {
            int tile_rows = (t + RM <= rows) ? RM : (rows - t);
            const uint8_t* g_ptrs[4];
            const uint8_t* u_ptrs[4];
            for (int ri = 0; ri < tile_rows; ri++) {
                g_ptrs[ri] = w_gate + (size_t)(n + t + ri) * blocks_per_row * BLOCK_BYTES;
                u_ptrs[ri] = w_up + (size_t)(n + t + ri) * blocks_per_row * BLOCK_BYTES;
            }
            for (int ri = tile_rows; ri < 4; ri++) {
                g_ptrs[ri] = g_ptrs[0];
                u_ptrs[ri] = u_ptrs[0];
            }

            float gate_tile[4];
            float up_tile[4];
            gemm_q4_0_tile_4x1_f16c(g_ptrs, q8_act.data(), gate_tile, blocks_per_row, tile_rows);
            gemm_q4_0_tile_4x1_f16c(u_ptrs, q8_act.data(), up_tile, blocks_per_row, tile_rows);
            for (int ri = 0; ri < tile_rows; ri++)
                out[n + t + ri] = silu(gate_tile[ri]) * up_tile[ri];
        }
    }
}

// ---- Fused FFN gate+up projection for Q4_0 decode, repacked layout ----
// Same as gemv_q4_0_fused_ffn_up_avx2 but consumes the repacked weight layout
// (repack_q4_0_weights) where the 4 rows of a tile group have their same-block
// bytes contiguous. Gate and up are processed with two sequential repacked
// decode loops sharing the quantized activation, keeping register pressure
// identical to the proven gemm_q4_0_decode_repacked_f16c_avx2 kernel.

static void gemv_q4_0_fused_ffn_up_repacked_avx2(
    const float* a, const uint8_t* w_gate, const uint8_t* w_up,
    float* out, int K, int N) {
    constexpr int RM = 4;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t nb = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t num_groups = (N + RM - 1) / RM;

    // Quantize activation once — shared by both gate and up
    scratch_vec<block_q8_0_act> q8_act(nb);
    quantize_row_q8_0_act(a, q8_act.data(), K);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

#    pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t n0 = g * RM;
        int64_t rows = std::min<int64_t>(n0 + RM, N) - n0;

        const uint8_t* gbase = w_gate + (size_t)g * nb * RM * BLOCK_BYTES;
        const uint8_t* ubase = w_up + (size_t)g * nb * RM * BLOCK_BYTES;

        __m256 ag0 = _mm256_setzero_ps();
        __m256 ag1 = _mm256_setzero_ps();
        __m256 ag2 = _mm256_setzero_ps();
        __m256 ag3 = _mm256_setzero_ps();

        for (int64_t l = 0; l < nb; ++l) {
            if (l + 4 < nb)
                _mm_prefetch((const char*)(gbase + (l + 4) * RM * BLOCK_BYTES), _MM_HINT_T0);

            const uint8_t* gb = gbase + l * RM * BLOCK_BYTES;
            __m256i wg0 = BlockLoader<block_q4_0_tag>::load(gb + 0 * BLOCK_BYTES);
            __m256i wg1 = (rows > 1) ? BlockLoader<block_q4_0_tag>::load(gb + 1 * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wg2 = (rows > 2) ? BlockLoader<block_q4_0_tag>::load(gb + 2 * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wg3 = (rows > 3) ? BlockLoader<block_q4_0_tag>::load(gb + 3 * BLOCK_BYTES) : _mm256_setzero_si256();

            __m256i avec = BlockLoader<block_q8_0_act>::load(&q8_act[l]);

            uint64_t packed_scales = 0;
            uint16_t d0, d1, d2, d3;
            memcpy(&d0, gb + 0 * BLOCK_BYTES, 2);
            memcpy(&d1, gb + 1 * BLOCK_BYTES, 2);
            memcpy(&d2, gb + 2 * BLOCK_BYTES, 2);
            memcpy(&d3, gb + 3 * BLOCK_BYTES, 2);
            packed_scales = (uint64_t)d0 | ((uint64_t)d1 << 16) | ((uint64_t)d2 << 32) | ((uint64_t)d3 << 48);
            __m128 sw_f16 = _mm_cvtph_ps(_mm_set_epi64x(0, packed_scales));
            float act_scale = q8_act[l].d;
            __m128 sw_all = _mm_mul_ps(sw_f16, _mm_set1_ps(act_scale));

            __m256 sc0 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x00));
            sc0 = _mm256_permute2f128_ps(sc0, sc0, 0x00);
            __m256 sc1 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x55));
            sc1 = _mm256_permute2f128_ps(sc1, sc1, 0x00);
            __m256 sc2 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xAA));
            sc2 = _mm256_permute2f128_ps(sc2, sc2, 0x00);
            __m256 sc3 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xFF));
            sc3 = _mm256_permute2f128_ps(sc3, sc3, 0x00);

            __m256i sag0 = _mm256_sign_epi8(wg0, wg0);
            __m256i sag1 = (rows > 1) ? _mm256_sign_epi8(wg1, wg1) : _mm256_setzero_si256();
            __m256i sag2 = (rows > 2) ? _mm256_sign_epi8(wg2, wg2) : _mm256_setzero_si256();
            __m256i sag3 = (rows > 3) ? _mm256_sign_epi8(wg3, wg3) : _mm256_setzero_si256();

            __m256i sbg0 = _mm256_sign_epi8(avec, wg0);
            __m256i sbg1 = (rows > 1) ? _mm256_sign_epi8(avec, wg1) : _mm256_setzero_si256();
            __m256i sbg2 = (rows > 2) ? _mm256_sign_epi8(avec, wg2) : _mm256_setzero_si256();
            __m256i sbg3 = (rows > 3) ? _mm256_sign_epi8(avec, wg3) : _mm256_setzero_si256();

            ag0 = _mm256_fmadd_ps(sc0, updot_avx2(sag0, sbg0), ag0);
            ag1 = _mm256_fmadd_ps(sc1, updot_avx2(sag1, sbg1), ag1);
            ag2 = _mm256_fmadd_ps(sc2, updot_avx2(sag2, sbg2), ag2);
            ag3 = _mm256_fmadd_ps(sc3, updot_avx2(sag3, sbg3), ag3);
        }

        __m256 au0 = _mm256_setzero_ps();
        __m256 au1 = _mm256_setzero_ps();
        __m256 au2 = _mm256_setzero_ps();
        __m256 au3 = _mm256_setzero_ps();

        for (int64_t l = 0; l < nb; ++l) {
            if (l + 4 < nb)
                _mm_prefetch((const char*)(ubase + (l + 4) * RM * BLOCK_BYTES), _MM_HINT_T0);

            const uint8_t* ub = ubase + l * RM * BLOCK_BYTES;
            __m256i wu0 = BlockLoader<block_q4_0_tag>::load(ub + 0 * BLOCK_BYTES);
            __m256i wu1 = (rows > 1) ? BlockLoader<block_q4_0_tag>::load(ub + 1 * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wu2 = (rows > 2) ? BlockLoader<block_q4_0_tag>::load(ub + 2 * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wu3 = (rows > 3) ? BlockLoader<block_q4_0_tag>::load(ub + 3 * BLOCK_BYTES) : _mm256_setzero_si256();

            __m256i avec = BlockLoader<block_q8_0_act>::load(&q8_act[l]);

            uint64_t packed_scales = 0;
            uint16_t d0, d1, d2, d3;
            memcpy(&d0, ub + 0 * BLOCK_BYTES, 2);
            memcpy(&d1, ub + 1 * BLOCK_BYTES, 2);
            memcpy(&d2, ub + 2 * BLOCK_BYTES, 2);
            memcpy(&d3, ub + 3 * BLOCK_BYTES, 2);
            packed_scales = (uint64_t)d0 | ((uint64_t)d1 << 16) | ((uint64_t)d2 << 32) | ((uint64_t)d3 << 48);
            __m128 sw_f16 = _mm_cvtph_ps(_mm_set_epi64x(0, packed_scales));
            float act_scale = q8_act[l].d;
            __m128 sw_all = _mm_mul_ps(sw_f16, _mm_set1_ps(act_scale));

            __m256 sc0 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x00));
            sc0 = _mm256_permute2f128_ps(sc0, sc0, 0x00);
            __m256 sc1 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x55));
            sc1 = _mm256_permute2f128_ps(sc1, sc1, 0x00);
            __m256 sc2 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xAA));
            sc2 = _mm256_permute2f128_ps(sc2, sc2, 0x00);
            __m256 sc3 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xFF));
            sc3 = _mm256_permute2f128_ps(sc3, sc3, 0x00);

            __m256i sau0 = _mm256_sign_epi8(wu0, wu0);
            __m256i sau1 = (rows > 1) ? _mm256_sign_epi8(wu1, wu1) : _mm256_setzero_si256();
            __m256i sau2 = (rows > 2) ? _mm256_sign_epi8(wu2, wu2) : _mm256_setzero_si256();
            __m256i sau3 = (rows > 3) ? _mm256_sign_epi8(wu3, wu3) : _mm256_setzero_si256();

            __m256i sbu0 = _mm256_sign_epi8(avec, wu0);
            __m256i sbu1 = (rows > 1) ? _mm256_sign_epi8(avec, wu1) : _mm256_setzero_si256();
            __m256i sbu2 = (rows > 2) ? _mm256_sign_epi8(avec, wu2) : _mm256_setzero_si256();
            __m256i sbu3 = (rows > 3) ? _mm256_sign_epi8(avec, wu3) : _mm256_setzero_si256();

            au0 = _mm256_fmadd_ps(sc0, updot_avx2(sau0, sbu0), au0);
            au1 = _mm256_fmadd_ps(sc1, updot_avx2(sau1, sbu1), au1);
            au2 = _mm256_fmadd_ps(sc2, updot_avx2(sau2, sbu2), au2);
            au3 = _mm256_fmadd_ps(sc3, updot_avx2(sau3, sbu3), au3);
        }

        float g0 = vdot::hsum_ps256(ag0);
        float u0 = vdot::hsum_ps256(au0);
        out[n0 + 0] = silu(g0) * u0;
        if (rows > 1) {
            float g1 = vdot::hsum_ps256(ag1);
            float u1 = vdot::hsum_ps256(au1);
            out[n0 + 1] = silu(g1) * u1;
        }
        if (rows > 2) {
            float g2 = vdot::hsum_ps256(ag2);
            float u2 = vdot::hsum_ps256(au2);
            out[n0 + 2] = silu(g2) * u2;
        }
        if (rows > 3) {
            float g3 = vdot::hsum_ps256(ag3);
            float u3 = vdot::hsum_ps256(au3);
            out[n0 + 3] = silu(g3) * u3;
        }
    }
}
#endif  // USE_AVX2

// ---- Fused FFN gate+up projection for Q4_K decode ----
// Reads input vector once, computes both gate and up projections simultaneously.
// Then applies SiLU(gate) * up in-place.

#ifdef USE_AVX2
static void gemv_q4_K_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate, const uint8_t* w_up,
                                        float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    scratch_vec<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * Q4_K_BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * nb * Q4_K_BLOCK_BYTES;

        float gate_val = silu(dot_q4_K_q8_K_avx2(gate_row, q8_buf.data(), nb));
        float up_val = dot_q4_K_q8_K_avx2(up_row, q8_buf.data(), nb);
        out[n] = gate_val * up_val;
    }
}
#endif  // USE_AVX2

// ---- Fused QKV projection for Q4_K decode ----
// Reads input vector once, computes Q + K + V outputs simultaneously
// with a single activation quantization.

#ifdef USE_AVX2
static void gemv_q4_K_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                     const uint8_t* wv, float* out_q, float* out_k, float* out_v,
                                     int K, int N_q, int N_k, int N_v) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    scratch_vec<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

    auto process_row = [&](const uint8_t* w_row, float* out, int N_out) {
#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N_out; ++n) {
            const uint8_t* q4_row = w_row + (size_t)n * nb * Q4_K_BLOCK_BYTES;
            out[n] = dot_q4_K_q8_K_avx2(q4_row, q8_buf.data(), nb);
        }
    };

    process_row(wq, out_q, N_q);
    process_row(wk, out_k, N_k);
    process_row(wv, out_v, N_v);
}
#endif  // USE_AVX2

// ---- Fused QKV + z + alpha + beta projection for qwen35 linear attention ----
// All four projections share the same input vector. Computed in a single OpenMP
// region so the Q8_K activation quantization and region setup are amortized
// once instead of four separate gemm calls.
// qkv may be Q4_K or Q6_K; z/alpha/beta are always Q4_K.

#ifdef USE_AVX2
static void gemv_fused_qkv_z_ab_avx2(
    const float* a, const uint8_t* w_qkv, bool qkv_q6, const uint8_t* w_z,
    const uint8_t* w_alpha, const uint8_t* w_beta, float* out_qkv, float* out_z,
    float* out_alpha, float* out_beta, int K, int N_qkv, int N_z, int N_alpha, int N_beta) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    constexpr int Q6_K_BLOCK_BYTES = 210;
    const int nb = (K + QK_K - 1) / QK_K;
    const size_t qkv_stride = (size_t)nb * (qkv_q6 ? Q6_K_BLOCK_BYTES : Q4_K_BLOCK_BYTES);
    const size_t q4_stride = (size_t)nb * Q4_K_BLOCK_BYTES;

    scratch_vec<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

    const int n_qkv_z = N_qkv + N_z;
    const int n_qkv_z_ab = n_qkv_z + N_alpha;
    const int total = n_qkv_z_ab + N_beta;

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < total; ++n) {
        float v;
        if (n < N_qkv) {
            const uint8_t* row = w_qkv + (size_t)n * qkv_stride;
            v = qkv_q6 ? dot_q6_K_q8_K_avx2(row, q8_buf.data(), nb)
                       : dot_q4_K_q8_K_avx2(row, q8_buf.data(), nb);
            out_qkv[n] = v;
        } else if (n < n_qkv_z) {
            const uint8_t* row = w_z + (size_t)(n - N_qkv) * q4_stride;
            out_z[n - N_qkv] = dot_q4_K_q8_K_avx2(row, q8_buf.data(), nb);
        } else if (n < n_qkv_z_ab) {
            const uint8_t* row = w_alpha + (size_t)(n - n_qkv_z) * q4_stride;
            out_alpha[n - n_qkv_z] = dot_q4_K_q8_K_avx2(row, q8_buf.data(), nb);
        } else {
            const uint8_t* row = w_beta + (size_t)(n - n_qkv_z_ab) * q4_stride;
            out_beta[n - n_qkv_z_ab] = dot_q4_K_q8_K_avx2(row, q8_buf.data(), nb);
        }
    }
}
#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge
