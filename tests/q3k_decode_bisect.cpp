// Mechanical bisect for the Q3_K MMQ residual deviation.
//
// Implements the GPU kernel's decode semantics LITERALLY on the host --
// load_tiles_q3_K's smem mapping plus mul_mat_q_kernel's element pairing --
// and compares element-by-element against forge::ops::dequantize_q3_k_row
// (the golden used by the reference suite). Prints exactly which element
// indices disagree and how, turning the convention question into data.
//
//   g++ -std=c++20 -O2 -I. -Iinclude -Isrc \
//       tests/q3k_decode_bisect.cpp src/operators/cpu/common/dequant.cpp \
//       src/operators/cpu/common/quant_tables.cpp \
//       -o /tmp/q3k_bisect && /tmp/q3k_bisect

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/operators/cpu/common/dequant.h"

namespace {

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    if (exp == 0) {
        const float den = static_cast<float>(man) * (1.0f / (1 << 24));
        return sign ? -den : den;
    }
    uint32_t bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

// get_int_b2: bytes [4*i, 4*i+4) little-endian (llama.cpp/Forge semantics).
int32_t get_int_b2(const uint8_t* x, int i) {
    return static_cast<int32_t>(x[4 * i + 0]) | (static_cast<int32_t>(x[4 * i + 1]) << 8) |
           (static_cast<int32_t>(x[4 * i + 2]) << 16) |
           (static_cast<int32_t>(x[4 * i + 3]) << 24);
}

int8_t sat_sub4(int32_t v) {  // __vsubss4(v, 4) per byte lane, inputs in [-128,127]
    int32_t r = v - 4;
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    return static_cast<int8_t>(r);
}

}  // namespace

int main() {
    std::mt19937 rng(1234567);
    constexpr int K = 256;

    // Random-but-valid Q3_K block: hmask[32] qs[64] scales[12] d(fp16).
    std::vector<uint8_t> blk(110);
    for (auto& b : blk) b = static_cast<uint8_t>(rng() & 0xFF);

    // ---- Golden: Forge CPU dequant ---------------------------------------
    std::vector<float> gold(K);
    forge::ops::dequantize_q3_k_row(blk.data(), gold.data(), K, /*row=*/0);

    // ---- Kernel-semantics emulation --------------------------------------
    // Loader (per kqsx in [0,16), l in [0,4)):
    //   slot k = (kqsx/8)*32 + l*8 + kqsx%8
    //   ql0 = int32 of qs bytes [4*kqsx, +4); qh0 = int32 of hmask bytes
    //         [4*(kqsx%8), +4) >> 4*(kqsx/8)
    //   per byte t: q2 = (ql0 >> 2l byte t)&3 ; hb = (qh0 >> l byte t)&1
    //   value byte = sat_sub4(q2 | hb<<2)   (== __vsubss4(ql|qh<<2, 4))
    // Compute pairing: global element e = 4*slot + t (pass windows tile
    // contiguously: slots [0,32) -> e [0,128), slots [32,64) -> e [128,256)).
    // Scale byte index = slot/4 (impl applies scales[i0/(QI8_1/2)] over
    // 4-int groups; pointer base advances k0/4 per pass).
    //
    // Scale words (from loader): for ksc in [0,4):
    //   sc_low  = (get_int_b2(scales, ksc % 2) >> 4*(ksc/2)) & 0x0F0F0F0F
    //   sc_high = ((get_int_b2(scales, 2) >> 2*ksc) << 4) & 0x30303030
    //   word[ksc] = sat_sub4(sc_low | sc_high)   per byte (-32 bias)
    uint8_t hm[32], qs[64], sc_raw[12];
    std::memcpy(hm, blk.data(), 32);
    std::memcpy(qs, blk.data() + 32, 64);
    std::memcpy(sc_raw, blk.data() + 96, 12);
    const float d = f16_to_f32(static_cast<uint16_t>(blk[108] | (blk[109] << 8)));

    int8_t sc_byte[16];
    for (int ksc = 0; ksc < 4; ++ksc) {
        const int ksc_low = ksc % 2;
        const int shift_low = 4 * (ksc / 2);
        const int32_t sc_low = (get_int_b2(sc_raw, ksc_low) >> shift_low) & 0x0F0F0F0F;
        const int ksc_high = 2;
        const int shift_high = 2 * ksc;
        const int32_t sc_high = ((get_int_b2(sc_raw, ksc_high) >> shift_high) << 4) & 0x30303030;
        const int32_t word = sc_low | sc_high;
        for (int t = 0; t < 4; ++t) {
            const int32_t byte = (word >> (8 * t)) & 0xFF;
            // Single -32 (matches __vsubss4(word, 32) on raw 6-bit values).
            // NOTE: an earlier revision composed sat_sub4(byte - 32),
            // double-subtracting 4 and fabricating a phantom "-4 scale bias".
            sc_byte[ksc * 4 + t] = static_cast<int8_t>(byte - 32);
        }
    }

    float kern[256];
    int8_t kern_q[256];  // raw biased values before scale/d, for diagnostics
    for (int kqsx = 0; kqsx < 16; ++kqsx) {
        const int32_t ql0 = get_int_b2(qs, kqsx);
        const int32_t qh0 =
            get_int_b2(hm, kqsx % 8) >> (4 * (kqsx / 8));
        for (int l = 0; l < 4; ++l) {
            const int slot = (kqsx / 8) * 32 + l * 8 + kqsx % 8;
            for (int t = 0; t < 4; ++t) {
                const int32_t ql_byte = (ql0 >> (8 * t)) & 0xFF;
                const int32_t qh_byte = (qh0 >> (8 * t)) & 0xFF;
                const int32_t q2 = (ql_byte >> (2 * l)) & 3;
                const int32_t hb = (qh_byte >> l) & 1;
                const int8_t v = sat_sub4(q2 | (hb << 2));
                const int e = 4 * slot + t;
                kern_q[e] = v;
                kern[e] = d * static_cast<float>(sc_byte[slot / 4]) * static_cast<float>(v);
            }
        }
    }

    // ---- Compare ---------------------------------------------------------
    // First: does the kernel's scale unpack agree with the CPU aux-shuffle?
    {
        std::printf("raw scales bytes:");
        for (int i = 0; i < 12; ++i) std::printf(" %02x", sc_raw[i]);
        std::printf("\n");
        uint32_t aux[4];
        std::memcpy(aux, sc_raw, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
        aux[0] = (aux[0] & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
        aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
        int8_t cpu_sc[16];
        std::memcpy(cpu_sc, aux, 16);
        std::printf("scale bytes  idx:");
        for (int i = 0; i < 16; ++i) std::printf("%4d", i);
        std::printf("\n  cpu (-32):");
        for (int i = 0; i < 16; ++i) std::printf("%4d", static_cast<int>(cpu_sc[i]) - 32);
        std::printf("\n  kern      :");
        for (int i = 0; i < 16; ++i) std::printf("%4d", static_cast<int>(sc_byte[i]));
        // Third decoder: llama.cpp get_scale_min_k4-style scalar reference,
        // independent of the aux-shuffle (validates WHICH convention is true).
        // For Q3_K the canonical per-sub-block scale/min come from the SAME
        // 6-bit unpack used by dequantize_row_q3_K's aux path; we reproduce
        // the aux result bit-exactly above, so instead cross-check the
        // low-nibble sources: print nibble decomposition of bytes 0..11.
        std::printf("\n  lo-nibs b0-7 :");
        for (int i = 0; i < 8; ++i) std::printf(" %x", sc_raw[i] & 0xF);
        std::printf("\n  hi-nibs b0-7 :");
        for (int i = 0; i < 8; ++i) std::printf(" %x", sc_raw[i] >> 4);
        std::printf("\n  2bit fields b8-11 (LE int, per ksc shift):\n");
        const int32_t w8_11 = get_int_b2(sc_raw, 2);
        for (int k = 0; k < 4; ++k)
            std::printf("    shift=%d -> bytes:", 2 * k),
                std::printf(" %02x", (w8_11 >> (2 * k)) & 0xFF),
                std::printf(" %02x", (w8_11 >> (2 * k + 8)) & 0xFF),
                std::printf(" %02x", (w8_11 >> (2 * k + 16)) & 0xFF),
                std::printf(" %02x\n", (w8_11 >> (2 * k + 24)) & 0xFF);
        std::printf("\n");
    }

    int mismatches = 0;
    for (int e = 0; e < K; ++e) {
        if (gold[e] != kern[e]) {
            ++mismatches;
            if (mismatches <= 16) {
                std::printf("MISMATCH e=%3d (slot=%2d t=%d grp=%d): gold=%.1f kern=%.1f "
                            "kern_code=%d kern_sc=%d\n",
                            e, e / 4, e % 4, e / 16, gold[e], kern[e],
                            static_cast<int>(kern_q[e]),
                            static_cast<int>(sc_byte[e / 16]));
            }
        }
    }
    std::printf("---- %d/%d elements mismatch\n", mismatches, K);

    // Also print both decoders for the first 16 elements for eyeballing.
    std::printf("idx : %-12s %-12s\n", "cpu", "kernel");
    for (int e = 0; e < 16; ++e)
        std::printf("%3d : %11.4f %11.4f\n", e, gold[e], kern[e]);
    return mismatches ? 1 : 0;
}
