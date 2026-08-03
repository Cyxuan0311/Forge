#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#    include <cpuid.h>
#endif

#include "cpu_gemv.h"
#include "gemm_microkernel.h"
#include "forge/logger.h"
#include "forge/operator_elementwise.h"
#include "forge/operator_matmul.h"
#include "forge/perf_profiler.h"

namespace {

// Runtime CPU feature detection for x86.
// WSL2 may not expose AVX-512 even when the host CPU supports it,
// so we must check at runtime rather than relying on compile-time macros.

#if defined(__x86_64__) || defined(_M_X64)

bool cpu_has_avx512_vnni() {
    uint32_t eax, ebx, ecx, edx;
    // CPUID leaf 7, sub-leaf 0
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0)
        return false;
    // AVX-512F:  EBX[16]
    if (!(ebx & (1u << 16)))
        return false;
    // AVX-512BW: EBX[30]
    if (!(ebx & (1u << 30)))
        return false;
    // AVX-512VNNI: ECX[11]
    if (!(ecx & (1u << 11)))
        return false;
    return true;
}

bool cached_has_avx512_vnni() {
    static bool result = cpu_has_avx512_vnni();
    return result;
}

#else

bool cached_has_avx512_vnni() { return false; }

#endif

// ---- Q4_0 Weight Repack Cache ----
// Repacked weights are keyed by the original Tensor data pointer.
// This ensures each weight matrix is repacked at most once.
struct RepackEntry {
    uint8_t* data = nullptr;
    size_t size = 0;
    int64_t K = 0;
    int64_t N = 0;
};

static std::unordered_map<const void*, RepackEntry>& repack_cache() {
    static std::unordered_map<const void*, RepackEntry> cache;
    return cache;
}

static std::mutex& repack_mutex() {
    static std::mutex mtx;
    return mtx;
}

// Get or create repacked Q4_0 weights for decode.
// Returns nullptr if repack is not applicable (e.g., N < 4).
const uint8_t* get_repacked_q4_0(const void* orig_data, int64_t K, int64_t N) {
    if (N < 4) return nullptr;  // too small for RM=4 tile

    std::lock_guard<std::mutex> lock(repack_mutex());
    auto& cache = repack_cache();
    auto it = cache.find(orig_data);
    if (it != cache.end()) {
        // Validate dimensions match
        if (it->second.K == K && it->second.N == N) {
            return it->second.data;
        }
        // Dimensions changed (shouldn't happen for same tensor, but handle it)
        delete[] it->second.data;
        cache.erase(it);
    }

    // Repack — forge::cpu::repack_q4_0_weights is in gemm_microkernel.h
    auto result = forge::cpu::repack_q4_0_weights(
        static_cast<const uint8_t*>(orig_data), K, N);
    if (!result.first) return nullptr;

    RepackEntry entry;
    entry.data = result.first;
    entry.size = result.second;
    entry.K = K;
    entry.N = N;
    cache[orig_data] = entry;
    return entry.data;
}

}  // namespace

#ifdef USE_CUDA
#    include <cuda_runtime.h>

#    include "../cuda/matmul_cuda.h"
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

#if FORGE_USE_OPENBLAS
#    include <cblas.h>
#endif

namespace forge {
namespace ops {

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

static constexpr int QK_K = 256;

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

// IQ2_S dequantization lookup tables (ported from ggml-common.h)
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

extern const uint64_t iq2s_grid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

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
// IQ2_XXS dequantization lookup tables (ported from ggml-common.h)
// block_iq2_xxs: fp16 d (2B) + 32*uint16_t qs (64B) = 66B per 256 elements
// qs layout: 8 groups of 4 uint16_t. Each group decodes 32 elements (4 sub-blocks of 8).
//   aux8[0..3] = grid indices into iq2xxs_grid
//   bits from aux32[1] provide scale (top 4 bits) and signs (7*4 bits)
// ============================================================================

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

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

// ============================================================================
// IQ4_NL dequantization
// block_iq4_nl: fp16 d (2B) + 16*qs nibble-packed (16B) = 18B per 32 elements
// qs[i] low  nibble → element i      (0..15)
// qs[i] high nibble → element i + 16 (16..31)
// Uses non-linear lookup table kvalues_iq4nl instead of linear -8..7
// ============================================================================

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

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

TensorPtr dequantize_q4_0_weight(const TensorPtr& q_weight) {
    int N = static_cast<int>(q_weight->shape()[0]);
    int K = static_cast<int>(q_weight->shape()[1]);
    auto fp32_weight =
        std::make_shared<Tensor>(DataType::FP32, q_weight->shape(), q_weight->device());
    const uint8_t* q_data = static_cast<const uint8_t*>(q_weight->data());
    float* out = static_cast<float*>(fp32_weight->data());
    std::vector<float> row_buf(K);
    for (int n = 0; n < N; ++n) {
        dequantize_q4_0_row(q_data, row_buf.data(), K, n);
        std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
    }
    return fp32_weight;
}

TensorPtr dequantize_q4_1_weight(const TensorPtr& q_weight) {
    int N = static_cast<int>(q_weight->shape()[0]);
    int K = static_cast<int>(q_weight->shape()[1]);
    auto fp32_weight =
        std::make_shared<Tensor>(DataType::FP32, q_weight->shape(), q_weight->device());
    const uint8_t* q_data = static_cast<const uint8_t*>(q_weight->data());
    float* out = static_cast<float*>(fp32_weight->data());
    std::vector<float> row_buf(K);
    for (int n = 0; n < N; ++n) {
        dequantize_q4_1_row(q_data, row_buf.data(), K, n);
        std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
    }
    return fp32_weight;
}

// Encode a float value to IEEE 754 half-precision (fp16) as uint16_t
static uint16_t fp32_to_fp16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (x >> 13) & 0x3FFU;

    if (exponent <= 0) {
        // Zero or subnormal — round to zero for simplicity
        return static_cast<uint16_t>(sign);
    } else if (exponent >= 0x1F) {
        // Overflow to infinity
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

// Quantize an FP32 [N, K] weight tensor to Q8_0 format.
// Q8_0 block: 2 bytes fp16 scale + 32 bytes int8 values = 34 bytes per 32 elements.
TensorPtr quantize_q8_0_weight(const TensorPtr& fp32_weight) {
    if (!fp32_weight)
        return nullptr;

    const auto& shape = fp32_weight->shape();
    int N = static_cast<int>(shape[0]);
    int K = static_cast<int>(shape[1]);

    constexpr int BLOCK_EL = 32;
    constexpr int BLOCK_BYTES = 34;
    int blocks_per_row = (K + BLOCK_EL - 1) / BLOCK_EL;
    size_t row_bytes = static_cast<size_t>(blocks_per_row) * BLOCK_BYTES;

    auto q_weight = std::make_shared<Tensor>(DataType::Q8_0, shape, fp32_weight->device());
    const float* src = static_cast<const float*>(fp32_weight->data());
    uint8_t* dst = static_cast<uint8_t*>(q_weight->data());

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const float* row = src + n * K;
        uint8_t* q_row = dst + static_cast<size_t>(n) * row_bytes;

        for (int bi = 0; bi < blocks_per_row; ++bi) {
            int base = bi * BLOCK_EL;
            int remaining = K - base;
            int nel = remaining < BLOCK_EL ? remaining : BLOCK_EL;

            // Find max absolute value in this block
            float amax = 0.0f;
            for (int j = 0; j < nel; ++j) {
                float av = std::fabs(row[base + j]);
                if (av > amax)
                    amax = av;
            }

            // Compute scale: amax / 127
            float scale = amax / 127.0f;
            float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            // Encode scale as fp16
            uint16_t scale_fp16 = fp32_to_fp16_bits(scale);
            memcpy(q_row + bi * BLOCK_BYTES, &scale_fp16, 2);

            // Quantize values to int8
            int8_t* qs = reinterpret_cast<int8_t*>(q_row + bi * BLOCK_BYTES + 2);
            for (int j = 0; j < nel; ++j) {
                float v = row[base + j] * inv_scale;
                int qv = static_cast<int>(std::roundf(v));
                qs[j] = static_cast<int8_t>(std::max(-128, std::min(127, qv)));
            }
            // Zero-fill remaining bytes in partial block
            for (int j = nel; j < BLOCK_EL; ++j) {
                qs[j] = 0;
            }
        }
    }
    return q_weight;
}

// Quantize an FP32 [N, K] weight tensor to Q4_0 format.
// Q4_0 block: 2 bytes fp16 scale + 16 bytes nibbles = 18 bytes per 32 elements.
// Each nibble stores value+8 in [0..15], actual value = nibble - 8.
TensorPtr quantize_q4_0_weight(const TensorPtr& fp32_weight) {
    if (!fp32_weight)
        return nullptr;

    const auto& shape = fp32_weight->shape();
    int N = static_cast<int>(shape[0]);
    int K = static_cast<int>(shape[1]);

    constexpr int BLOCK_EL = 32;
    constexpr int BLOCK_BYTES = 18;
    int blocks_per_row = (K + BLOCK_EL - 1) / BLOCK_EL;
    size_t row_bytes = static_cast<size_t>(blocks_per_row) * BLOCK_BYTES;

    auto q_weight = std::make_shared<Tensor>(DataType::Q4_0, shape, fp32_weight->device());
    const float* src = static_cast<const float*>(fp32_weight->data());
    uint8_t* dst = static_cast<uint8_t*>(q_weight->data());

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const float* row = src + n * K;
        uint8_t* q_row = dst + static_cast<size_t>(n) * row_bytes;

        for (int bi = 0; bi < blocks_per_row; ++bi) {
            int base = bi * BLOCK_EL;
            int remaining = K - base;
            int nel = remaining < BLOCK_EL ? remaining : BLOCK_EL;

            // Find max absolute value in this block
            float amax = 0.0f;
            for (int j = 0; j < nel; ++j) {
                float av = std::fabs(row[base + j]);
                if (av > amax)
                    amax = av;
            }

            // Q4_0 scale: amax / 7 (nibble offset range [-8, 7], positive max offset is 7).
            // This matches ggml convention: value = (nibble - 8) * scale
            // where nibble ∈ [0,15], so (nibble-8) ∈ [-8,7].
            float scale = amax / 7.0f;
            float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            // Encode scale as fp16
            uint16_t scale_fp16 = fp32_to_fp16_bits(scale);
            memcpy(q_row + bi * BLOCK_BYTES, &scale_fp16, 2);

            // Quantize values to nibbles: nibble = round(v * inv_scale) + 8, clamped to [0, 15]
            uint8_t* qs = q_row + bi * BLOCK_BYTES + 2;
            for (int j = 0; j < BLOCK_EL; j += 2) {
                uint8_t lo = 0, hi = 0;
                if (j < nel) {
                    float v0 = row[base + j] * inv_scale;
                    int q0 = static_cast<int>(std::roundf(v0)) + 8;
                    lo = static_cast<uint8_t>(std::max(0, std::min(15, q0)));
                }
                if (j + 1 < nel) {
                    float v1 = row[base + j + 1] * inv_scale;
                    int q1 = static_cast<int>(std::roundf(v1)) + 8;
                    hi = static_cast<uint8_t>(std::max(0, std::min(15, q1)));
                }
                qs[j / 2] = lo | (hi << 4);
            }
        }
    }
    return q_weight;
}

// Re-quantize Q8_0 weight to Q4_0 (dequantize → requantize, ~1-2s for 7B output weight)
TensorPtr requantize_q8_0_to_q4_0(const TensorPtr& q8_weight) {
    if (!q8_weight || q8_weight->dtype() != DataType::Q8_0)
        return nullptr;
    auto fp32 = dequantize_weight(q8_weight);
    return quantize_q4_0_weight(fp32);
}

TensorPtr dequantize_weight(const TensorPtr& weight) {
    if (!weight || !is_quantized_type(weight->dtype()))
        return weight;
    auto dequant_fn = get_dequant_row_fn(weight->dtype());
    if (!dequant_fn)
        return weight;

    int N = static_cast<int>(weight->shape()[0]);
    int K = static_cast<int>(weight->shape()[1]);
    auto fp32_weight = std::make_shared<Tensor>(DataType::FP32, weight->shape(), weight->device());
    const uint8_t* q_data = static_cast<const uint8_t*>(weight->data());
    float* out = static_cast<float*>(fp32_weight->data());

#pragma omp parallel
    {
        std::vector<float> row_buf(K);
#pragma omp for schedule(dynamic, 64)
        for (int n = 0; n < N; ++n) {
            dequant_fn(q_data, row_buf.data(), K, n);
            std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
        }
    }
    return fp32_weight;
}

static void apply_bias(TensorPtr& out, const TensorPtr& bias, int M, int N) {
    if (!bias)
        return;

    if (out->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda_apply_bias(out, bias, M, N);
#endif
        return;
    }

    const float* bias_data = static_cast<const float*>(bias->data());
    float* o_data = static_cast<float*>(out->data());

    if (bias->ndim() == 1) {
        int bias_size = static_cast<int>(bias->shape()[0]);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < bias_size && n < N; ++n) {
                o_data[m * N + n] += bias_data[n];
            }
        }
    } else {
        int total = static_cast<int>(out->numel());
        for (int i = 0; i < total; ++i) {
            o_data[i] += bias_data[i];
        }
    }
}

TensorPtr matmul(const TensorPtr& a, const TensorPtr& b, const TensorPtr& bias) {
    if (a->ndim() != 2 || b->ndim() != 2)
        throw std::runtime_error("matmul expects 2D tensors");

    TensorPtr b_fp32 = b;
    if (is_quantized_type(b->dtype())) {
        int N = static_cast<int>(b->shape()[0]);
        int K = static_cast<int>(b->shape()[1]);
        if (a->device() == DeviceType::CUDA) {
            // Phase 4: dequantize on device — no D2H/H2D staging.
#ifdef USE_CUDA
            b_fp32 = cuda_dequantize_matrix(b, N, K);
#else
            throw std::runtime_error("matmul: CUDA tensor without CUDA support");
#endif
        } else {
            auto dequant_fn = get_dequant_row_fn(b->dtype());
            if (!dequant_fn)
                throw std::runtime_error("Unsupported quantized type in matmul");
            if (b->device() != DeviceType::CPU)
                throw std::runtime_error(
                    "matmul: CPU path requires a host-resident quantized weight "
                    "(cross-device staging is disabled)");
            b_fp32 = std::make_shared<Tensor>(DataType::FP32, b->shape(), DeviceType::CPU);
            const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
            float* out = static_cast<float*>(b_fp32->data());
            std::vector<float> row_buf(K);
            for (int n = 0; n < N; ++n) {
                dequant_fn(q_data, row_buf.data(), K, n);
                std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
            }
        }
    }

    int M = static_cast<int>(a->shape()[0]);
    int K = static_cast<int>(a->shape()[1]);
    int K2 = static_cast<int>(b_fp32->shape()[0]);
    int N = static_cast<int>(b_fp32->shape()[1]);
    if (K != K2)
        throw std::runtime_error("matmul dimension mismatch");

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, a->device());

    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda_matmul(a, b_fp32, out, M, K, N);
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b_fp32->data());
        float* o_data = static_cast<float*>(out->data());
#if FORGE_USE_OPENBLAS
        PERF_SCOPE("matmul/fp32_gemm_openblas");
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a_data, K, b_data, N,
                    0.0f, o_data, N);
#elif defined(USE_AVX2)
        PERF_SCOPE("matmul/fp32_gemm_avx2");
        cpu::gemm_fp32_avx2(a_data, b_data, o_data, M, K, N);
#else
        PERF_SCOPE("matmul/fp32_gemm_scalar");
        std::memset(o_data, 0, M * N * sizeof(float));
#    pragma omp parallel for schedule(dynamic) if (M * N > 64)
        for (int m = 0; m < M; ++m) {
            const float* a_row = a_data + m * K;
            float* o_row = o_data + m * N;
            for (int k = 0; k < K; ++k) {
                float a_val = a_row[k];
                const float* b_row = b_data + k * N;
                for (int n = 0; n < N; ++n) {
                    o_row[n] += a_val * b_row[n];
                }
            }
        }
#endif
    }

    apply_bias(out, bias, M, N);
    return out;
}

// Q5_K fused GEMV forward declaration (implemented after mixed-precision QKV)
#ifdef USE_AVX2
static void gemv_q5_k_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                    int N);
#endif

TensorPtr matmul_transB(const TensorPtr& a, const TensorPtr& b, const TensorPtr& bias) {
    if (a->ndim() != 2 || b->ndim() != 2)
        throw std::runtime_error("matmul_transB expects 2D tensors");

    int M = static_cast<int>(a->shape()[0]);
    int K = static_cast<int>(a->shape()[1]);
    int N = static_cast<int>(b->shape()[0]);
    int K2 = static_cast<int>(b->shape()[1]);
    if (K != K2)
        throw std::runtime_error("matmul_transB dimension mismatch: K=" + std::to_string(K) +
                                 " K2=" + std::to_string(K2) + " b_shape=[" +
                                 std::to_string(b->shape()[0]) + "," +
                                 std::to_string(b->shape()[1]) + "]");

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, a->device());

    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda_matmul_transB(a, b, out, M, K, N);
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        float* o_data = static_cast<float*>(out->data());

        if (is_quantized_type(b->dtype())) {
#ifdef USE_AVX2
            if (b->dtype() == DataType::Q4_0) {
                if (M == 1) {
#ifdef USE_AVX512_VNNI
                    if (cached_has_avx512_vnni()) {
                        PERF_SCOPE("matmul_transB/q4_0_gemm_decode_vnni");
                        cpu::gemm_q4_0_decode_vnni(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                    K, N);
                    } else
#endif
                    {
                        // Try repacked weights for better cache locality
                        const uint8_t* repacked = get_repacked_q4_0(b->data(), K, N);
                        if (repacked) {
                            PERF_SCOPE("matmul_transB/q4_0_gemm_decode_repacked");
                            cpu::gemm_q4_0_decode_repacked_f16c_avx2(a_data, repacked, o_data, K, N);
                        } else {
                            PERF_SCOPE("matmul_transB/q4_0_gemm_decode");
                            cpu::gemm_q4_0_decode_f16c_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                              K, N);
                        }
                    }
                } else {
#ifdef USE_AVX512_VNNI
                    if (cached_has_avx512_vnni()) {
                        PERF_SCOPE("matmul_transB/q4_0_gemm_batch_vnni");
                        cpu::gemm_q4_0_batch_vnni(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                   M, K, N);
                    } else
#endif
                    {
                        PERF_SCOPE("matmul_transB/q4_0_gemm_batch");
                        cpu::gemm_q4_0_batch_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                    M, K, N);
                    }
                }
            } else if (b->dtype() == DataType::Q8_0) {
                PERF_SCOPE("matmul_transB/q8_0_maddubs_gemv");
                cpu::gemv_q8_0_maddubs_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                     M, K, N);
            } else if (b->dtype() == DataType::Q4_1) {
                PERF_SCOPE("matmul_transB/q4_1_maddubs_gemv");
                cpu::gemv_q4_1_maddubs_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                     M, K, N);
            } else if (b->dtype() == DataType::Q5_0) {
                PERF_SCOPE("matmul_transB/q5_0_maddubs_gemv");
                cpu::gemv_q5_0_maddubs_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                     M, K, N);
            } else if (b->dtype() == DataType::Q5_1) {
                PERF_SCOPE("matmul_transB/q5_1_maddubs_gemv");
                cpu::gemv_q5_1_maddubs_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                     M, K, N);
            } else if (b->dtype() == DataType::Q4_K) {
                PERF_SCOPE("matmul_transB/q4_k_gemm");
                cpu::gemm_q4_K_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q6_K) {
                PERF_SCOPE("matmul_transB/q6_k_gemm");
                cpu::gemm_q6_K_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q2_K) {
                PERF_SCOPE("matmul_transB/q2_k_gemm");
                cpu::gemm_q2_K_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q3_K) {
                PERF_SCOPE("matmul_transB/q3_k_gemm");
                cpu::gemm_q3_K_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q5_K) {
                PERF_SCOPE("matmul_transB/q5_k_gemm");
                cpu::gemm_q5_K_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else
#endif
            {
                PERF_SCOPE("matmul_transB/dequant+gemv");
                auto dequant_fn = get_dequant_row_fn(b->dtype());
                if (!dequant_fn)
                    throw std::runtime_error("Unsupported quantized type in matmul_transB: " + dtype_name(b->dtype()));
                const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
                int block_el = dtype_block_elements(b->dtype());
                int block_bytes = dtype_block_size(b->dtype());
                int blocks_per_row = (K + block_el - 1) / block_el;
                size_t row_bytes = (size_t)blocks_per_row * block_bytes;
                size_t expected_total = (size_t)N * row_bytes;
                if (expected_total > b->nbytes()) {
                    fprintf(stderr, "[ERROR] matmul_transB: expected_total(%zu) > b->nbytes(%zu)! Buffer overflow risk!\n",
                            expected_total, b->nbytes());
                    fflush(stderr);
                }
#ifdef USE_AVX2
// For Q4_K/Q6_K: dequantize scalar + AVX2 dot product
#    pragma omp parallel
                {
                    std::vector<float> row_buf(K);
#    pragma omp for schedule(dynamic)
                    for (int n = 0; n < N; ++n) {
                        dequant_fn(q_data, row_buf.data(), K, n);
                        for (int m = 0; m < M; ++m) {
                            o_data[m * N + n] =
                                cpu::dot_product_avx2(a_data + m * K, row_buf.data(), K);
                        }
                    }
                }
#else
#    if FORGE_USE_OPENBLAS
                PERF_SCOPE("matmul_transB/dequant+gemm_openblas");
                {
                    std::vector<float> b_fp32(N * K);
                    for (int n = 0; n < N; ++n) {
                        dequant_fn(q_data, &b_fp32[n * K], K, n);
                    }
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, a_data, K,
                                b_fp32.data(), K, 0.0f, o_data, N);
                }
#    else
#        pragma omp parallel
                {
                    std::vector<float> row_buf(K);
#        pragma omp for schedule(dynamic)
                    for (int n = 0; n < N; ++n) {
                        dequant_fn(q_data, row_buf.data(), K, n);
                        for (int m = 0; m < M; ++m) {
                            const float* a_row = a_data + m * K;
                            float sum = 0.0f;
                            for (int k = 0; k < K; ++k) {
                                sum += a_row[k] * row_buf[k];
                            }
                            o_data[m * N + n] = sum;
                        }
                    }
                }
#    endif
#endif
            }
        } else {
#if FORGE_USE_OPENBLAS
            PERF_SCOPE("matmul_transB/fp32_gemm_openblas");
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, a_data, K,
                        static_cast<const float*>(b->data()), K, 0.0f, o_data, N);
#elif defined(USE_AVX2)
            PERF_SCOPE("matmul_transB/fp32_gemv_avx2");
            cpu::gemv_fp32_transB_avx2(a_data, static_cast<const float*>(b->data()), o_data, M, K,
                                       N);
#else
            PERF_SCOPE("matmul_transB/fp32_gemv_scalar");
            const float* b_data = static_cast<const float*>(b->data());
#    pragma omp parallel for schedule(dynamic) if (M * N > 64)
            for (int m = 0; m < M; ++m) {
                const float* a_row = a_data + m * K;
                float* o_row = o_data + m * N;
                for (int n = 0; n < N; ++n) {
                    const float* b_row = b_data + n * K;
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += a_row[k] * b_row[k];
                    }
                    o_row[n] = sum;
                }
            }
#endif
        }
    }

    apply_bias(out, bias, M, N);
    return out;
}

TensorPtr matmul_transB_dual(const TensorPtr& a, const TensorPtr& b1, const TensorPtr& b2) {
    if (a->ndim() != 2 || b1->ndim() != 2 || b2->ndim() != 2)
        throw std::runtime_error("matmul_transB_dual expects 2D tensors");

    int M = static_cast<int>(a->shape()[0]);
    int K = static_cast<int>(a->shape()[1]);
    int N1 = static_cast<int>(b1->shape()[0]);
    int N2 = static_cast<int>(b2->shape()[0]);
    int N = N1 + N2;

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, a->device());

    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda_matmul_transB_dual(a, b1, b2, out, M, K, N1, N2);
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        float* o_data = static_cast<float*>(out->data());

        auto compute_part = [&](const TensorPtr& b, int offset, int n_cols) {
            if (is_quantized_type(b->dtype())) {
                auto dequant_fn = get_dequant_row_fn(b->dtype());
                const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
                std::vector<float> row_buf(K);
                for (int n = 0; n < n_cols; ++n) {
                    dequant_fn(q_data, row_buf.data(), K, n);
                    for (int m = 0; m < M; ++m) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            sum += a_data[m * K + k] * row_buf[k];
                        }
                        o_data[m * N + offset + n] = sum;
                    }
                }
            } else {
                const float* b_data = static_cast<const float*>(b->data());
                for (int n = 0; n < n_cols; ++n) {
                    for (int m = 0; m < M; ++m) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            sum += a_data[m * K + k] * b_data[n * K + k];
                        }
                        o_data[m * N + offset + n] = sum;
                    }
                }
            }
        };

        compute_part(b1, 0, N1);
        compute_part(b2, N1, N2);
    }

    return out;
}

TensorPtr ffn_up_fused(const TensorPtr& input, const TensorPtr& w1, const TensorPtr& w3,
                       int intermediate_dim) {
    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, intermediate_dim},
                                        input->device());

    if (input->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        out = cuda_ffn_up_fused(input, w1, w3, out, M, K, intermediate_dim);
#endif
    } else {
        auto gate = ops::matmul_transB(input, w1);
        auto up = ops::matmul_transB(input, w3);
        out = ops::silu_multiply(gate, up);
    }

    return out;
}

TensorPtr matmul_transB_fused_qkv_q4_0(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv) {
    // All three weights must be Q4_0, same K dimension
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_0 expects 2D tensors");
    if (wq->dtype() != DataType::Q4_0 || wk->dtype() != DataType::Q4_0 ||
        wv->dtype() != DataType::Q4_0)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#else
    // Fallback: separate matmul_transB calls
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    // Return Q, K, V as separate tensors packaged in a concatenated format
    // The caller (llama_engine) will extract them
    // For simplicity, we use a vector-like structure:
    // Return Q in a custom way - actually let's return them separately.
    // Since TensorPtr can only return one tensor, we pack Q, K, V consecutively.
    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }

    // Store individual outputs as metadata for easy extraction
    // We'll use a convention: the returned tensor has shape [M, N_q + N_k + N_v]
    // The caller splits it using slice operations.
    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q4_0(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_0 expects 2D tensors");
    if (weight->dtype() != DataType::Q4_0)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_ffn_down_residual_avx2(a_data + m * K,
                                              static_cast<const uint8_t*>(weight->data()),
                                              r_data + m * N, o_data + m * N, K, N);
    }
#else
    // Fallback: separate matmul + add
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q4_1(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_1 expects 2D tensors");
    if (weight->dtype() != DataType::Q4_1)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q4_1 requires Q4_1 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_1 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q4_1");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_1_ffn_down_residual_avx2(a_data + m * K,
                                               static_cast<const uint8_t*>(weight->data()),
                                               r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

// Fused attention output projection + residual for Q4_0 decode.
// Computes: out = attn_out @ wo + hidden_residual  (single pass)
TensorPtr matmul_transB_fused_attn_proj_residual_q4_0(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_0 expects 2D tensors");
    if (weight->dtype() != DataType::Q4_0)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_attn_proj_residual_q4_k(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_k expects 2D tensors");
    if (weight->dtype() != DataType::Q4_K)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_k requires Q4_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_k_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_attn_proj_residual_q5_k(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q5_k expects 2D tensors");
    if (weight->dtype() != DataType::Q5_K)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q5_k requires Q5_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q5_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q5_k_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_attn_proj_residual_q6_k(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q6_k expects 2D tensors");
    if (weight->dtype() != DataType::Q6_K)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q6_k requires Q6_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q6_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q6_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q6_k_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_attn_proj_residual_q2_k(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q2_k expects 2D tensors");
    if (weight->dtype() != DataType::Q2_K)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q2_k requires Q2_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q2_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q2_k_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_attn_proj_residual_q3_k(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q3_k expects 2D tensors");
    if (weight->dtype() != DataType::Q3_K)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q3_k requires Q3_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q3_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_qkv_q4_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_k expects 2D tensors");
    if (wq->dtype() != DataType::Q4_K || wk->dtype() != DataType::Q4_K ||
        wv->dtype() != DataType::Q4_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_k requires Q4_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_K_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q4_k(const TensorPtr& input, const TensorPtr& w_gate,
                                          const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_k expects 2D tensors");
    if (w_gate->dtype() != DataType::Q4_K || w_up->dtype() != DataType::Q4_K)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_k requires Q4_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_k_fused_ffn_up_avx2(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q3_k(const TensorPtr& input, const TensorPtr& w_gate,
                                         const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q3_k expects 2D tensors");
    if (w_gate->dtype() != DataType::Q3_K || w_up->dtype() != DataType::Q3_K)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q3_k requires Q3_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q3_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_fused_ffn_up_avx2(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

TensorPtr matmul_transB_fused_qk_q3_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qk_q3_k expects 2D tensors");
    if (wq->dtype() != DataType::Q3_K || wk->dtype() != DataType::Q3_K)
        throw std::runtime_error("matmul_transB_fused_qk_q3_k requires Q3_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qk_q3_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qk_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_fused_qk_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k, K, N_q, N_k);
    }
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
#endif

    int total_N = N_q + N_k;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB_fused_qkv_q3_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k expects 2D tensors");
    if (wq->dtype() != DataType::Q3_K || wk->dtype() != DataType::Q3_K ||
        wv->dtype() != DataType::Q3_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k requires Q3_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }
    return out;
}

// ---- Mixed-precision Q3_K (Q,K) + Q4_K (V) fused QKV projection ----
// Quantizes activation to Q8_K once, shares across Q3_K Q/K and Q4_K V rows.
// Saves 1 Q8_K quantization + 1 matmul dispatch per layer per decode token.
#ifdef USE_AVX2
static float dot_q4_K_q8_K_row_avx2(const uint8_t* q4_row, const cpu::block_q8_K* q8, int nb) {
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
#    pragma omp parallel for schedule(static)
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
#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N_v; ++n) {
        const uint8_t* q4_row = wv + (size_t)n * nb * Q4_K_BLOCK_BYTES;
        out_v[n] = dot_q4_K_q8_K_row_avx2(q4_row, q8_buf.data(), nb);
    }
}
#endif

TensorPtr matmul_transB_fused_qkv_q3_k_q4_k(const TensorPtr& input, const TensorPtr& wq,
                                             const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k_q4_k expects 2D tensors");
    if (wq->dtype() != DataType::Q3_K || wk->dtype() != DataType::Q3_K ||
        wv->dtype() != DataType::Q4_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k_q4_k requires Q3_K (Q,K) + Q4_K (V)");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q3_k_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        gemv_q3_k_q4_k_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }
    return out;
}

// ---- Q5_K fused GEMV (M=1 decode) using Q8_K dot product ----
// Quantizes activation to Q8_K once, then uses inline Q5_K×Q8_K dot product.
// Saves full dequantization + FP32 dot product overhead.
#ifdef USE_AVX2
static float dot_q5_K_q8_K_row_avx2(const uint8_t* q5_row, const cpu::block_q8_K* q8, int nb) {
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

        // Load all 32 bytes of qh at once, use hmask+bit shifting (same as llama.cpp)
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

#    pragma omp parallel for schedule(static)
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

#    pragma omp parallel for schedule(static)
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

#    pragma omp parallel for schedule(static)
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

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * Q2_K_BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * nb * Q2_K_BLOCK_BYTES;

        float gate_val = silu(cpu::dot_q2_K_q8_K_avx2(gate_row, q8_buf.data(), nb, nullptr));
        float up_val = cpu::dot_q2_K_q8_K_avx2(up_row, q8_buf.data(), nb, nullptr);
        out[n] = gate_val * up_val;
    }
}
#endif

TensorPtr matmul_transB_fused_ffn_down_residual_q6_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q6_k expects 2D tensors");
    if (weight->dtype() != DataType::Q6_K)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q6_k requires Q6_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q6_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q6_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q6_k_ffn_down_residual_avx2(a_data + m * K,
                                              static_cast<const uint8_t*>(weight->data()),
                                              r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q4_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_k expects 2D tensors");
    if (weight->dtype() != DataType::Q4_K)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q4_k requires Q4_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_k_ffn_down_residual_avx2(a_data + m * K,
                                               static_cast<const uint8_t*>(weight->data()),
                                               r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q5_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q5_k expects 2D tensors");
    if (weight->dtype() != DataType::Q5_K)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q5_k requires Q5_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q5_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q5_k_ffn_down_residual_avx2(a_data + m * K,
                                               static_cast<const uint8_t*>(weight->data()),
                                               r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q2_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q2_k expects 2D tensors");
    if (weight->dtype() != DataType::Q2_K)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q2_k requires Q2_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q2_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q2_k_ffn_down_residual_avx2(a_data + m * K,
                                               static_cast<const uint8_t*>(weight->data()),
                                               r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q3_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q3_k expects 2D tensors");
    if (weight->dtype() != DataType::Q3_K)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q3_k requires Q3_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q3_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_ffn_down_residual_avx2(a_data + m * K,
                                               static_cast<const uint8_t*>(weight->data()),
                                               r_data + m * N, o_data + m * N, K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q4_0(const TensorPtr& input, const TensorPtr& w_gate,
                                          const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_0 expects 2D tensors");
    if (w_gate->dtype() != DataType::Q4_0 || w_up->dtype() != DataType::Q4_0)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_fused_ffn_up_avx2(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q5_k(const TensorPtr& input, const TensorPtr& w_gate,
                                         const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q5_k expects 2D tensors");
    if (w_gate->dtype() != DataType::Q5_K || w_up->dtype() != DataType::Q5_K)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q5_k requires Q5_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q5_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        gemv_q5_k_fused_ffn_up_avx2(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q2_k(const TensorPtr& input, const TensorPtr& w_gate,
                                           const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q2_k expects 2D tensors");
    if (w_gate->dtype() != DataType::Q2_K || w_up->dtype() != DataType::Q2_K)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q2_k requires Q2_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q2_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        gemv_q2_k_fused_ffn_up_avx2(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

TensorPtr matmul_transB_fused_qkv_q5_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q5_k expects 2D tensors");
    if (wq->dtype() != DataType::Q5_K || wk->dtype() != DataType::Q5_K ||
        wv->dtype() != DataType::Q5_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q5_k requires Q5_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q5_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q5_K_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB_fused_qkv_q2_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q2_k expects 2D tensors");
    if (wq->dtype() != DataType::Q2_K || wk->dtype() != DataType::Q2_K ||
        wv->dtype() != DataType::Q2_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q2_k requires Q2_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q2_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q2_K_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }
    return out;
}

}  // namespace ops
}  // namespace forge
