// Scalar dequant row functions (architecture-independent).
// Split out of matmul.cpp during the arch-split refactor (see cpu_arch_split_plan.md).
// Referenced by forge::data_type_traits[].dequant_row via quant_traits.cpp.
#include <cmath>
#include <cstdint>
#include <cstring>

#include "dequant.h"
#include "quant_helpers.h"
#include "quant_tables.h"
#include "scalar.h"

namespace forge {
namespace ops {

static constexpr int QK_K = 256;

static inline float fp16_to_fp32(uint16_t bits) {
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0)
            return 0.0f;
        float v = std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
        return sign ? -v : v;
    }
    float v = std::ldexp((1.0f + static_cast<float>(mantissa) / 1024.0f),
                         static_cast<int>(exponent) - 15);
    return sign ? -v : v;
}

void dequantize_q4_0_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q4_0_BLOCK_SIZE = 18;
    int blocks_per_row = (K + 31) / 32;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q4_0_BLOCK_SIZE;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q4_0_BLOCK_SIZE;
        float scale = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs = block_ptr + 2;
        int base = bi * 32;
        for (int j = 0; j < 16 && base + j < K; ++j)
            out[base + j] = static_cast<float>((qs[j] & 0x0F) - 8) * scale;
        for (int j = 0; j < 16 && base + 16 + j < K; ++j)
            out[base + 16 + j] = static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale;
    }
}

void dequantize_q4_1_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q4_1_BLOCK_SIZE = 20;
    int blocks_per_row = (K + 31) / 32;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q4_1_BLOCK_SIZE;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q4_1_BLOCK_SIZE;
        float d_val = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        float m_val = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr + 2));
        const uint8_t* qs = block_ptr + 4;
        int base = bi * 32;
        for (int j = 0; j < 16 && base + j < K; ++j)
            out[base + j] = static_cast<float>(qs[j] & 0x0F) * d_val + m_val;
        for (int j = 0; j < 16 && base + 16 + j < K; ++j)
            out[base + 16 + j] = static_cast<float>((qs[j] >> 4) & 0x0F) * d_val + m_val;
    }
}

void dequantize_q8_0_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q8_0_BLOCK_SIZE = 34;
    int blocks_per_row = (K + 31) / 32;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q8_0_BLOCK_SIZE;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q8_0_BLOCK_SIZE;
        float scale = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + 2);
        int base = bi * 32;
        for (int j = 0; j < 32 && base + j < K; ++j)
            out[base + j] = static_cast<float>(qs[j]) * scale;
    }
}


// Q5_0 dequantize: 22 bytes per block of 32 elements
// Block layout: d[2] + qh[4] + qs[16]
// Ported from llama.cpp dequantize_row_q5_0
void dequantize_q5_0_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q5_0_BLOCK_SIZE = 22;
    int blocks_per_row = (K + 31) / 32;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q5_0_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q5_0_BLOCK_SIZE;
        float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        uint32_t qh;
        memcpy(&qh, block_ptr + 2, 4);
        const uint8_t* qs = block_ptr + 6;

        int base = bi * 32;
        for (int j = 0; j < 16; ++j) {
            uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            uint8_t xh_1 = ((qh >> (j + 12))       ) & 0x10;

            int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16;
            int32_t x1 = ((qs[j] >>   4) | xh_1) - 16;

            if (base + j < K)
                out[base + j] = static_cast<float>(x0) * d;
            if (base + 16 + j < K)
                out[base + 16 + j] = static_cast<float>(x1) * d;
        }
    }
}

// Q5_1 dequantize: 24 bytes per block of 32 elements
// Block layout: d[2] + m[2] + qh[4] + qs[16]
// Ported from llama.cpp dequantize_row_q5_1
void dequantize_q5_1_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q5_1_BLOCK_SIZE = 24;
    int blocks_per_row = (K + 31) / 32;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q5_1_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q5_1_BLOCK_SIZE;
        float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        float m = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr + 2));
        uint32_t qh;
        memcpy(&qh, block_ptr + 4, 4);
        const uint8_t* qs = block_ptr + 8;

        int base = bi * 32;
        for (int j = 0; j < 16; ++j) {
            uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            uint8_t xh_1 = ((qh >> (j + 12))       ) & 0x10;

            int x0 = (qs[j] & 0x0F) | xh_0;
            int x1 = (qs[j] >>   4) | xh_1;

            if (base + j < K)
                out[base + j] = static_cast<float>(x0) * d + m;
            if (base + 16 + j < K)
                out[base + 16 + j] = static_cast<float>(x1) * d + m;
        }
    }
}

// Q2_K dequantize: 84 bytes per block of 256 elements
// Block layout: scales[16] + qs[64] + d[2] + dmin[2]
// 16 sub-blocks of 16 elements each, scales are 4-bit packed (lo=scale, hi=min)
// qs are 2-bit values (4 per byte)
// Ported from llama.cpp dequantize_row_q2_K
void dequantize_q2_k_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q2_K_BLOCK_SIZE = 84;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q2_K_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q2_K_BLOCK_SIZE;
        const uint8_t* scales = block_ptr;         // bytes 0..15
        const uint8_t* q = block_ptr + 16;         // bytes 16..79
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr + 80, 2);        // bytes 80..81
        memcpy(&dmin_bits, block_ptr + 82, 2);     // bytes 82..83
        float d = fp16_to_fp32(d_bits);
        float min = fp16_to_fp32(dmin_bits);

        float* y = out + bi * QK_K;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    y[l] = dl * static_cast<float>(static_cast<int8_t>((q[l] >> shift) & 3)) - ml;
                }

                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    y[16 + l] = dl * static_cast<float>(static_cast<int8_t>((q[l + 16] >> shift) & 3)) - ml;
                }

                shift += 2;
                y += 32;
            }
            q += 32;
        }
    }
}

static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void dequantize_q6_k_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q6_K_BLOCK_SIZE = 210;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q6_K_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q6_K_BLOCK_SIZE;
        const uint8_t* ql = block_ptr;
        const uint8_t* qh = ql + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(qh + 64);
        uint16_t d_bits;
        memcpy(&d_bits, sc + 16, 2);
        float d = fp16_to_fp32(d_bits);

        float* y = out + bi * QK_K;
        const uint8_t* ql_cur = ql;
        const uint8_t* qh_cur = qh;
        const int8_t* sc_cur = sc;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql_cur[l + 0] & 0xF) | (((qh_cur[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql_cur[l + 32] & 0xF) | (((qh_cur[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql_cur[l + 0] >> 4) | (((qh_cur[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql_cur[l + 32] >> 4) | (((qh_cur[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * static_cast<float>(sc_cur[is + 0]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(sc_cur[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(sc_cur[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(sc_cur[is + 6]) * static_cast<float>(q4);
            }
            y += 128;
            ql_cur += 64;
            qh_cur += 32;
            sc_cur += 8;
        }
    }
}

// Q3_K dequantize: 110 bytes per block of 256 elements
// Block layout: hmask[32] + qs[64] + scales[12] + d[2]
// Ported from llama.cpp dequantize_row_q3_K (pointer-increment style)
void dequantize_q3_k_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q3_K_BLOCK_SIZE = 110;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q3_K_BLOCK_SIZE;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q3_K_BLOCK_SIZE;
        const uint8_t* hm = block_ptr;              // hmask: bytes 0..31
        const uint8_t* q = block_ptr + 32;          // qs: bytes 32..95
        const uint8_t* scales_raw = block_ptr + 96;  // scales: bytes 96..107
        uint16_t d_bits;
        memcpy(&d_bits, block_ptr + 108, 2);         // d: bytes 108..109
        float d_all = fp16_to_fp32(d_bits);

        // Unpack 12 bytes of scales into 16 signed 6-bit scale values
        uint32_t aux[4];
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        memcpy(aux, scales_raw, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        float* y = out + bi * QK_K;
        int is = 0;
        uint8_t m = 1;

        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * static_cast<float>(
                        static_cast<int8_t>((q[l] >> shift) & 3) -
                        ((hm[l] & m) ? 0 : 4));
                }

                dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * static_cast<float>(
                        static_cast<int8_t>((q[l + 16] >> shift) & 3) -
                        ((hm[l + 16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

void dequantize_q4_k_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q4_K_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q4_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = fp16_to_fp32(d_bits);
        float dmin = fp16_to_fp32(dmin_bits);
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qs = block_ptr + 16;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * sc1;
            float m1_val = dmin * m1;
            float d2 = d * sc2;
            float m2_val = dmin * m2;
            int base = bi * QK_K + j;
            for (int l = 0; l < 32; ++l) {
                if (base + l < K)
                    out[base + l] = d1 * static_cast<float>(qs[l] & 0xF) - m1_val;
            }
            for (int l = 0; l < 32; ++l) {
                if (base + 32 + l < K)
                    out[base + 32 + l] = d2 * static_cast<float>(qs[l] >> 4) - m2_val;
            }
            qs += 32;
            is += 2;
        }
    }
}

// Q5_K dequantize: 176 bytes per block of 256 elements
// Block layout: d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
// Ported from llama.cpp dequantize_row_q5_K (pointer-increment style)
void dequantize_q5_k_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q5_K_BLOCK_SIZE = 176;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q5_K_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q5_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = fp16_to_fp32(d_bits);
        float min = fp16_to_fp32(dmin_bits);
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qh = block_ptr + 16;
        const uint8_t* ql = block_ptr + 48;

        float* y = out + bi * QK_K;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc; float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc; float m2 = min * m;
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * static_cast<float>((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * static_cast<float>((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            }
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

// IQ2_S dequantize: 82 bytes per block of 256 elements
// Block layout: d[2] + qs[64] + qh[8] + scales[8]
// qs[0..31] = quantized values (8 groups of 4 bytes), qs[32..63] = sign bits
// Ported from llama.cpp dequantize_row_iq2_s (pointer-increment style)
void dequantize_iq2_s_row(const uint8_t* q_data, float* out, int K, int row) {
    const int IQ2_S_BLOCK_SIZE = 82;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * IQ2_S_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * IQ2_S_BLOCK_SIZE;
        float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs = block_ptr + 2;      // bytes 2..65: qs[0..31]=values, qs[32..63]=signs
        const uint8_t* qh = block_ptr + 2 + 64; // bytes 66..73: high bits
        const uint8_t* sc = block_ptr + 2 + 72; // bytes 74..81: scales

        float* y = out + bi * QK_K;
        // signs points to the second half of qs array (after the 32 value bytes)
        const uint8_t* signs = qs + QK_K / 8; // = qs + 32

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            float db[2];
            db[0] = d * (0.5f + (sc[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (sc[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                float dl = db[l / 2];
                int grid_idx = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(&iq2s_grid[grid_idx]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * static_cast<float>(grid[j]) *
                           (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

// ============================================================================
// Fused IQ2_S × Q8_K GEMV kernel (AVX2)
// Combines IQ2_S grid-lookup dequantization with int8×uint8 multiply-add,
// avoiding the FP32 intermediate buffer used by the dequant+gemv fallback.
// Strategy matches dot_q4_K_q8_K_avx2: quantize activation to Q8_K once,
// then for each weight row compute the dot product in the integer domain.
// ============================================================================


// --- IQ2_XXS / IQ4_NL dequant (kept after AVX2 IQ kernels section) ---

void dequantize_iq2_xxs_row(const uint8_t* q_data, float* out, int K, int row) {
    const int IQ2_XXS_BLOCK_SIZE = 66;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * IQ2_XXS_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * IQ2_XXS_BLOCK_SIZE;
        float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(block_ptr + 2); // 32 uint16_t

        float* y = out + bi * QK_K;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            // Each ib32 group uses 4 uint16_t from qs
            const uint16_t* q2 = qs + 4 * ib32;
            uint32_t aux32[2];
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            const float db = d * ls * 0.25f;

            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(&iq2xxs_grid[aux8[l]]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * static_cast<float>(grid[j]) *
                           (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

void dequantize_iq4_nl_row(const uint8_t* q_data, float* out, int K, int row) {
    const int IQ4_NL_BLOCK_SIZE = 18;
    int blocks_per_row = (K + 31) / 32;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * IQ4_NL_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * IQ4_NL_BLOCK_SIZE;
        float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs = block_ptr + 2;

        float* y = out + bi * 32;
        for (int i = 0; i < 16; ++i) {
            y[i] = d * static_cast<float>(kvalues_iq4nl[qs[i] & 0x0F]);
            y[i + 16] = d * static_cast<float>(kvalues_iq4nl[(qs[i] >> 4) & 0x0F]);
        }
    }
}

// --- IQ2_XS / IQ3_S dequant ---

void dequantize_iq2_xs_row(const uint8_t* q_data, float* out, int K, int row) {
    const int IQ2_XS_BLOCK_SIZE = 74;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * IQ2_XS_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * IQ2_XS_BLOCK_SIZE;
        float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(block_ptr + 2); // 32 uint16_t
        const uint8_t* scales = block_ptr + 2 + 64;                           // 8 bytes

        float* y = out + bi * QK_K;
        float db[2];
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(&iq2xs_grid[qs[4 * ib32 + l] & 511]);
                const uint8_t signs = ksigns_iq2xs[qs[4 * ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l / 2] * static_cast<float>(grid[j]) *
                           (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// ============================================================================
// IQ3_S dequantize: 110 bytes per block of 256 elements
// Block layout: d[2] + qs[64B] + qh[8B] + signs[32B] + scales[4B]
// Ported from llama.cpp dequantize_row_iq3_s (pointer-increment style)
// ============================================================================
void dequantize_iq3_s_row(const uint8_t* q_data, float* out, int K, int row) {
    const int IQ3_S_BLOCK_SIZE = 110;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * IQ3_S_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * IQ3_S_BLOCK_SIZE;
        float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs = block_ptr + 2;                   // 64 bytes
        const uint8_t* qh = block_ptr + 2 + 64;              // 8 bytes
        const uint8_t* signs = block_ptr + 2 + 64 + 8;       // 32 bytes
        const uint8_t* scales = block_ptr + 2 + 64 + 8 + 32; // 4 bytes

        float* y = out + bi * QK_K;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3s_grid[qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3s_grid[qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * static_cast<float>(grid1[j]) * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db1 * static_cast<float>(grid2[j]) * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3s_grid[qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3s_grid[qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * static_cast<float>(grid1[j]) * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db2 * static_cast<float>(grid2[j]) * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}


}  // namespace ops
}  // namespace forge
