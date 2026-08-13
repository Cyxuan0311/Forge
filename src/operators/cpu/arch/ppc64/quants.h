#pragma once
// PPC64 VSX I-quant (IQ2_S / IQ2_XS / IQ3_S / IQ4_NL) fused dot-product and
// GEMV kernels.
//
// These are VSX translations of the ARM64 NEON kernels in arch/arm64/quants.h
// (same block layouts, same grid/LUT tables from quant_tables.h). VSX has no
// dotprod instruction, so every 16-element dot uses the vec_mule/vec_mulo +
// vec_sum4s fallback from vec_dot.h.
//
// All I-quant grid values are small non-negative integers (0..7), so their
// bit patterns are identical when reinterpreted as signed int8; the signed×
// signed dot helper therefore computes the same result as the unsigned×signed
// maddubs used on NEON.

#ifdef USE_VSX
#include <altivec.h>
#endif
#include <cstring>
#include <cmath>
#include <vector>

#include "../../common/quant_tables.h"
#include "../../common/quant_helpers.h"
#include "../../common/scalar.h"
#include "forge/types.h"
#include "vec_dot.h"

namespace forge {
namespace cpu {

#ifdef USE_VSX

// ============================================================================
// Internal helpers
// ============================================================================

// Dot product of 16 grid bytes (non-negative) with 16 signed q8 bytes.
// Grid bytes are small (0..7) so the signed×signed helper gives the same
// arithmetic result as the NEON unsigned×signed maddubs.
static inline int dot_16_grid_q8(__vector unsigned char grid,
                                 __vector signed char q8) {
    return vsx_dot::dot_16_signed_signed(
        (__vector signed char)grid, q8);
}

// ============================================================================
// IQ4_NL x Q8_K fused dot product (VSX)
// IQ4_NL block: 18 bytes per 32 elements = d[2] fp16 + qs[16] nibbles
// 8 sub-blocks per 256-element super-block
// ============================================================================

static inline float dot_iq4_nl_q8_K_vsx(const uint8_t* iq4nl_row,
                                        const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ4_NL_BLOCK_SIZE = 18;
    constexpr int IQ4_NL_BLOCK_EL = 32;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        const uint8_t* block_ptr = iq4nl_row +
            (size_t)bi * (QK_K / IQ4_NL_BLOCK_EL) * IQ4_NL_BLOCK_SIZE;
        float block_acc = 0.0f;

        for (int sb = 0; sb < QK_K / IQ4_NL_BLOCK_EL; ++sb) {
            float d = vsx_dot::fp16_to_fp32(
                *reinterpret_cast<const uint16_t*>(block_ptr));
            const uint8_t* qs = block_ptr + 2;

            // Build the 32 LUT values: nibble -> kvalues_iq4nl[nibble]
            alignas(16) int8_t vals[32];
            for (int i = 0; i < 16; ++i) {
                vals[i]       = forge::ops::kvalues_iq4nl[qs[i] & 0x0F];
                vals[16 + i]  = forge::ops::kvalues_iq4nl[(qs[i] >> 4) & 0x0F];
            }

            __vector signed char v_lo = vec_xl(0, vals);
            __vector signed char v_hi = vec_xl(0, vals + 16);
            __vector signed char q8_lo = (__vector signed char)vec_xl(0, q8d + sb * 32);
            __vector signed char q8_hi = (__vector signed char)vec_xl(0, q8d + sb * 32 + 16);

            int sum = vsx_dot::dot_16_signed_signed(v_lo, q8_lo)
                    + vsx_dot::dot_16_signed_signed(v_hi, q8_hi);

            block_acc += d * (float)sum;
            block_ptr += IQ4_NL_BLOCK_SIZE;
        }

        acc += d_q8 * block_acc;
    }

    return acc;
}

// ============================================================================
// IQ4_NL GEMV: out = a * w^T  (M×K × N×K → M×N)
// ============================================================================

static void gemv_iq4_nl_q8k_transB_vsx(const float* a, const uint8_t* w, float* out,
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
            out[n] = dot_iq4_nl_q8_K_vsx(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * row_bytes;
            for (int m = 0; m < M; ++m)
                out[(size_t)m * N + n] = dot_iq4_nl_q8_K_vsx(row,
                    q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ2_XS x Q8_K fused dot product (VSX)
// IQ2_XS block: 74 bytes = d[2] fp16 + qs[64] uint16 + scales[8]
// 8 ib32 groups, each with 4 lanes × 8 elements
// ============================================================================

static inline float dot_iq2_xs_q8_K_vsx(const uint8_t* iq2xs_row,
                                        const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ2_XS_BLOCK_SIZE = 74;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq2xs_row + (size_t)bi * IQ2_XS_BLOCK_SIZE;
        float d = vsx_dot::fp16_to_fp32(
            *reinterpret_cast<const uint16_t*>(block_ptr));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(block_ptr + 2);
        const uint8_t* scales = block_ptr + 2 + 64;

        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        float block_acc = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db0_f = (0.5f + (scales[ib32] & 0xf)) * 0.25f;
            const float db1_f = (0.5f + (scales[ib32] >> 4)) * 0.25f;

            // Build grid_vals[32] and sign_mask[32] for this ib32 group
            alignas(16) uint8_t grid_vals[32];
            alignas(16) uint8_t sign_mask[32];
            for (int l = 0; l < 4; ++l) {
                const uint16_t qs_u16 = qs[4 * ib32 + l];
                memcpy(grid_vals + l * 8,
                       &forge::ops::iq2xs_grid[qs_u16 & 511], 8);
                const uint8_t signs_byte = forge::ops::ksigns_iq2xs[qs_u16 >> 9];
                for (int j = 0; j < 8; ++j)
                    sign_mask[l * 8 + j] =
                        (signs_byte & forge::ops::kmask_iq2xs[j]) ? 0xFF : 0x00;
            }

            __vector unsigned char grid_lo = vec_xl(0, grid_vals);
            __vector unsigned char grid_hi = vec_xl(0, grid_vals + 16);
            __vector unsigned char sign_lo = vec_xl(0, sign_mask);
            __vector unsigned char sign_hi = vec_xl(0, sign_mask + 16);

            __vector signed char q8_lo = (__vector signed char)vec_xl(0, q8d + ib32 * 32);
            __vector signed char q8_hi = (__vector signed char)vec_xl(0, q8d + ib32 * 32 + 16);

            // Apply sign: negate q8 where sign byte = 0xFF (vec_sel selects
            // the second argument when the mask bit is 1).
            __vector signed char q8_signed_lo =
                vec_sel(q8_lo, -q8_lo, sign_lo);
            __vector signed char q8_signed_hi =
                vec_sel(q8_hi, -q8_hi, sign_hi);

            int s_l01 = dot_16_grid_q8(grid_lo, q8_signed_lo);
            int s_l23 = dot_16_grid_q8(grid_hi, q8_signed_hi);

            block_acc += (float)s_l01 * db0_f + (float)s_l23 * db1_f;
        }

        acc += d * d_q8 * block_acc;
    }

    return acc;
}

// ============================================================================
// IQ2_XS GEMV
// ============================================================================

static void gemv_iq2_xs_q8k_transB_vsx(const float* a, const uint8_t* w, float* out,
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
            out[n] = dot_iq2_xs_q8_K_vsx(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_XS_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[(size_t)m * N + n] = dot_iq2_xs_q8_K_vsx(row,
                    q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ3_S x Q8_K fused dot product (VSX)
// IQ3_S block: 110 bytes = d[2] + qs[64] + qh[8] + signs[32] + scales[4]
// ============================================================================

static inline float dot_iq3_s_q8_K_vsx(const uint8_t* iq3s_row,
                                       const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ3_S_BLOCK_SIZE = 110;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq3s_row + (size_t)bi * IQ3_S_BLOCK_SIZE;
        float d = vsx_dot::fp16_to_fp32(
            *reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs    = block_ptr + 2;
        const uint8_t* qh    = block_ptr + 2 + 64;
        const uint8_t* signs = block_ptr + 2 + 64 + 8;
        const uint8_t* scales = block_ptr + 2 + 64 + 8 + 32;

        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        float block_acc = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1_f = (float)(1 + 2 * (scales[ib32 / 2] & 0xf));
            const float db2_f = (float)(1 + 2 * (scales[ib32 / 2] >> 4));

            for (int half = 0; half < 2; ++half) {
                const float db_f = (half == 0) ? db1_f : db2_f;

                // Build grid_vals[32] and sign_mask[32] for this half
                alignas(16) uint8_t grid_vals[32];
                alignas(16) uint8_t sign_mask_arr[32];
                for (int l = 0; l < 4; ++l) {
                    const int idx1 = qs[2 * l + 0] |
                        ((qh[half] << (8 - 2 * l)) & 256);
                    const int idx2 = qs[2 * l + 1] |
                        ((qh[half] << (7 - 2 * l)) & 256);
                    // iq3s_grid entries are 4 bytes each
                    memcpy(grid_vals + l * 8 + 0,
                           &forge::ops::iq3s_grid[idx1], 4);
                    memcpy(grid_vals + l * 8 + 4,
                           &forge::ops::iq3s_grid[idx2], 4);
                    const uint8_t s = signs[l];
                    for (int j = 0; j < 8; ++j)
                        sign_mask_arr[l * 8 + j] =
                            (s & forge::ops::kmask_iq2xs[j]) ? 0xFF : 0x00;
                }

                __vector unsigned char grid_lo = vec_xl(0, grid_vals);
                __vector unsigned char grid_hi = vec_xl(0, grid_vals + 16);
                __vector unsigned char sign_lo = vec_xl(0, sign_mask_arr);
                __vector unsigned char sign_hi = vec_xl(0, sign_mask_arr + 16);

                __vector signed char q8_lo = (__vector signed char)vec_xl(0, q8d + ib32 * 32 + half * 32);
                __vector signed char q8_hi = (__vector signed char)vec_xl(0, q8d + ib32 * 32 + half * 32 + 16);

                __vector signed char q8_signed_lo =
                    vec_sel(q8_lo, -q8_lo, sign_lo);
                __vector signed char q8_signed_hi =
                    vec_sel(q8_hi, -q8_hi, sign_hi);

                int s_all = dot_16_grid_q8(grid_lo, q8_signed_lo)
                          + dot_16_grid_q8(grid_hi, q8_signed_hi);

                block_acc += (float)s_all * db_f;

                qs += 8;
                signs += 4;
            }
            qh += 2;
        }

        acc += d * d_q8 * block_acc;
    }

    return acc;
}

// ============================================================================
// IQ3_S GEMV
// ============================================================================

static void gemv_iq3_s_q8k_transB_vsx(const float* a, const uint8_t* w, float* out,
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
            out[n] = dot_iq3_s_q8_K_vsx(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ3_S_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[(size_t)m * N + n] = dot_iq3_s_q8_K_vsx(row,
                    q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ2_S x Q8_K fused dot product (VSX)
// IQ2_S block: 82 bytes = d[2] + qs[32] + signs[32] + qh[8] + sc[8]
// ============================================================================

static inline float dot_iq2_s_q8_K_vsx(const uint8_t* iq2s_row,
                                       const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ2_S_BLOCK_SIZE = 82;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq2s_row + (size_t)bi * IQ2_S_BLOCK_SIZE;
        float d = vsx_dot::fp16_to_fp32(
            *reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs    = block_ptr + 2;
        const uint8_t* signs = qs + 32;
        const uint8_t* qh    = block_ptr + 66;
        const uint8_t* sc    = block_ptr + 74;

        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        float block_acc = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db0_f = (0.5f + (sc[ib32] & 0xf)) * 0.25f;
            const float db1_f = (0.5f + (sc[ib32] >> 4)) * 0.25f;

            // Build grid_vals[32] and sign_mask[32]
            alignas(16) uint8_t grid_vals[32];
            alignas(16) uint8_t sign_mask_arr[32];
            for (int l = 0; l < 4; ++l) {
                const int grid_idx = qs[l] |
                    ((qh[ib32] << (8 - 2 * l)) & 0x300);
                // iq2s_grid entries are 8 bytes each
                memcpy(grid_vals + l * 8,
                       &forge::ops::iq2s_grid[grid_idx], 8);
                const uint8_t s = signs[l];
                for (int j = 0; j < 8; ++j)
                    sign_mask_arr[l * 8 + j] =
                        (s & forge::ops::kmask_iq2xs[j]) ? 0xFF : 0x00;
            }

            __vector unsigned char grid_lo = vec_xl(0, grid_vals);
            __vector unsigned char grid_hi = vec_xl(0, grid_vals + 16);
            __vector unsigned char sign_lo = vec_xl(0, sign_mask_arr);
            __vector unsigned char sign_hi = vec_xl(0, sign_mask_arr + 16);

            __vector signed char q8_lo = (__vector signed char)vec_xl(0, q8d + ib32 * 32);
            __vector signed char q8_hi = (__vector signed char)vec_xl(0, q8d + ib32 * 32 + 16);

            __vector signed char q8_signed_lo =
                vec_sel(q8_lo, -q8_lo, sign_lo);
            __vector signed char q8_signed_hi =
                vec_sel(q8_hi, -q8_hi, sign_hi);

            int s_l01 = dot_16_grid_q8(grid_lo, q8_signed_lo);
            int s_l23 = dot_16_grid_q8(grid_hi, q8_signed_hi);

            block_acc += (float)s_l01 * db0_f + (float)s_l23 * db1_f;

            qs += 4;
            signs += 4;
        }

        acc += d * d_q8 * block_acc;
    }

    return acc;
}

// ============================================================================
// IQ2_S GEMV
// ============================================================================

static void gemv_iq2_s_q8k_transB_vsx(const float* a, const uint8_t* w, float* out,
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
            out[n] = dot_iq2_s_q8_K_vsx(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_S_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[(size_t)m * N + n] = dot_iq2_s_q8_K_vsx(row,
                    q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// DotQ8KFn dispatcher for batched MoE matmul
// ============================================================================

using DotQ8KFn = float (*)(const uint8_t*, const cpu::block_q8_K*, int);

static inline DotQ8KFn get_dot_q8k_fn(DataType dt) {
    switch (dt) {
    case DataType::IQ2_XS: return dot_iq2_xs_q8_K_vsx;
    case DataType::IQ3_S:  return dot_iq3_s_q8_K_vsx;
    case DataType::IQ4_NL: return dot_iq4_nl_q8_K_vsx;
    case DataType::IQ2_S:  return dot_iq2_s_q8_K_vsx;
    default: return nullptr;
    }
}

// K-quant fused wrappers reuse the format-correct VSX GEMM implementations.
template <typename GemmFn>
static inline void fused_ffn_up_gemm_vsx(const float* a, const uint8_t* gate, const uint8_t* up, float* out, int K, int N, GemmFn gemm) {
    std::vector<float> g(N), u(N);
    gemm(a, gate, g.data(), 1, K, N);
    gemm(a, up, u.data(), 1, K, N);
    for (int n = 0; n < N; ++n) out[n] = (g[n] / (1.0f + std::exp(-g[n]))) * u[n];
}
template <typename GemmFn>
static inline void fused_qkv_gemm_vsx(const float* a, const uint8_t* q, const uint8_t* k, const uint8_t* v, float* oq, float* ok, float* ov, int K, int Nq, int Nk, int Nv, GemmFn gemm) {
    gemm(a, q, oq, 1, K, Nq); gemm(a, k, ok, 1, K, Nk); gemm(a, v, ov, 1, K, Nv);
}
template <typename GemmFn>
static inline void fused_residual_gemm_vsx(const float* a, const uint8_t* w, const float* r, float* o, int K, int N, GemmFn gemm) {
    std::vector<float> dot(N); gemm(a, w, dot.data(), 1, K, N);
    for (int n = 0; n < N; ++n) o[n] = r[n] + dot[n];
}
#define FORGE_VSX_FUSED_K(NAME, GEMM) \
static inline void gemv_##NAME##_fused_ffn_up_vsx(const float* a, const uint8_t* g, const uint8_t* u, float* o, int K, int N) { fused_ffn_up_gemm_vsx(a,g,u,o,K,N,GEMM); } \
static inline void gemv_##NAME##_fused_qkv_vsx(const float* a, const uint8_t* q, const uint8_t* k, const uint8_t* v, float* oq, float* ok, float* ov, int K, int Nq, int Nk, int Nv) { fused_qkv_gemm_vsx(a,q,k,v,oq,ok,ov,K,Nq,Nk,Nv,GEMM); } \
static inline void gemv_##NAME##_ffn_down_residual_vsx(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_gemm_vsx(a,w,r,o,K,N,GEMM); } \
static inline void gemv_##NAME##_attn_proj_residual_vsx(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_gemm_vsx(a,w,r,o,K,N,GEMM); }
FORGE_VSX_FUSED_K(q2_K, gemm_q2_K_vsx)
FORGE_VSX_FUSED_K(q3_K, gemm_q3_K_vsx)
FORGE_VSX_FUSED_K(q4_K, gemm_q4_K_vsx)
FORGE_VSX_FUSED_K(q5_K, gemm_q5_K_vsx)
FORGE_VSX_FUSED_K(q6_K, gemm_q6_K_vsx)
#undef FORGE_VSX_FUSED_K

#endif // USE_VSX

}  // namespace cpu
}  // namespace forge
