// Regression lock for the IQ-family dequant bugs found while debugging
// Ling-3.0-tiny IQ1_M garbage output:
//
//   [IQ1_M] dequantize_iq1_m_row indexed qh[2*sib+..] WHILE also advancing the
//           qh pointer by 2 bytes per sub-block -- a double advance that read
//           qh bytes 0,1 / 4,5 / 10,11 / 18,19 / ... instead of 0,1 / 2,3 /
//           4,5 / ..., running past the 16-byte qh field into the scales field
//           and the next block. llama.cpp reference (ggml-quants.c:2690-2721)
//           reads qh[0], qh[1] and advances by 2 bytes.
//
//   [IQ2_XXS] dequantize_iq2_xxs_row computed the group scale as
//           d * (2*s + 1) * 0.25f; llama.cpp (ggml-quants.c:2501) is
//           d * (0.5f + s) * 0.25f -- exactly half. Every attn_output value
//           was 2x too large.
//
// Both reference implementations below are independent re-writes following
// llama.cpp's scalar loops (fixed pointers, no pointer/index double-step), so
// any re-introduction of the double-advance or scale-formula drift fails.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "quant_tables.h"
#include "quant_tables_iq13.h"
#include "dequant.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "../src/operators/cuda/cuda_gemv.h"
#endif

namespace forge { namespace ops {
extern void dequantize_iq1_m_row(const uint8_t*, float*, int, int);
extern void dequantize_iq2_xxs_row(const uint8_t*, float*, int, int);
extern void dequantize_iq4_nl_row(const uint8_t*, float*, int, int);
}}  // namespace forge::ops

using forge::ops::IQ1S_DELTA;
using forge::ops::iq1s_grid;
using forge::ops::iq2xxs_grid;
using forge::ops::kmask_iq2xs;
using forge::ops::ksigns_iq2xs;
using forge::ops::kvalues_iq4nl;

namespace {

// Same conversion as the static helper in dequant.cpp (llama.cpp GGML_FP16_TO_FP32).
inline float fp16_to_fp32(uint16_t bits) {
    const uint32_t sign = (bits >> 15) & 1;
    const uint32_t exponent = (bits >> 10) & 0x1F;
    const uint32_t mantissa = bits & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0) return 0.0f;
        const float v = std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
        return sign ? -v : v;
    }
    const float v = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                               static_cast<int>(exponent) - 15);
    return sign ? -v : v;
}

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

// Deterministic pseudo-random byte stream (xorshift) so every run is identical.
uint32_t g_rng = 0x12345678u;
uint8_t next_byte() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return static_cast<uint8_t>(g_rng >> 24);
}

// ---------------------------------------------------------------------------
// Reference: llama.cpp dequantize_row_iq1_m (ggml-quants.c:2675-2722).
// Block (56 B): qs[32] qh[16] sc[8]; fp16 scale packed into the top nibbles of
// sc[0..3]; 8 sub-blocks each read qh[0], qh[1] then advance qh by 2 bytes.
// ---------------------------------------------------------------------------
void ref_iq1_m(const std::vector<uint8_t>& data, std::vector<float>& out) {
    const int QK_K = 256;
    const size_t nblock = data.size() / 56;
    out.assign(nblock * QK_K, 0.0f);
    float* y = out.data();
    for (size_t i = 0; i < nblock; ++i) {
        const uint8_t* blk = data.data() + i * 56;
        const uint8_t* qs = blk;
        const uint8_t* qh = blk + 32;
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk + 48);
        const uint16_t scale_u16 = static_cast<uint16_t>(
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) |
            (sc[3] & 0xf000));
        const float d = fp16_to_fp32(scale_u16);
        for (int ib = 0; ib < 8; ++ib) {
            const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
            const uint8_t b1 = qh[0];
            const uint8_t b2 = qh[1];
            const int idx[4] = {
                qs[0] | ((b1 << 8) & 0x700),
                qs[1] | ((b1 << 4) & 0x700),
                qs[2] | ((b2 << 8) & 0x700),
                qs[3] | ((b2 << 4) & 0x700),
            };
            const float delta[4] = {
                (b1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA,
                (b1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA,
                (b2 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA,
                (b2 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA,
            };
            for (int l = 0; l < 4; ++l) {
                const float dl = (l < 2) ? dl1 : dl2;
                const int8_t* g = reinterpret_cast<const int8_t*>(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) y[j] = dl * (g[j] + delta[l]);
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

// ---------------------------------------------------------------------------
// Reference: llama.cpp dequantize_row_iq2_xxs (ggml-quants.c:2488-2512).
// Block (66 B): fp16 d + 32 x uint16 qs; 8 groups of 32; group scale
// d * (0.5f + (aux32[1] >> 28)) * 0.25f.
// ---------------------------------------------------------------------------
void ref_iq2_xxs(const std::vector<uint8_t>& data, std::vector<float>& out) {
    const int QK_K = 256;
    const size_t nblock = data.size() / 66;
    out.assign(nblock * QK_K, 0.0f);
    float* y = out.data();
    for (size_t i = 0; i < nblock; ++i) {
        const uint8_t* blk = data.data() + i * 66;
        const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(blk));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32[2];
            std::memcpy(aux32, qs + 4 * ib32, 8);  // qs is uint16*: 8-byte stride
            const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid =
                    reinterpret_cast<const uint8_t*>(iq2xxs_grid + aux8[l]);
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

// ---------------------------------------------------------------------------
// Reference: llama.cpp dequantize_row_iq4_nl (ggml-quants.c:2727-2741).
// Block (18 B): fp16 d + 16 nibble bytes over a 32-element group.
// ---------------------------------------------------------------------------
void ref_iq4_nl(const std::vector<uint8_t>& data, std::vector<float>& out) {
    const size_t nblock = data.size() / 18;
    out.assign(nblock * 32, 0.0f);
    for (size_t i = 0; i < nblock; ++i) {
        const uint8_t* blk = data.data() + i * 18;
        const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(blk));
        const uint8_t* qs = blk + 2;
        float* y = out.data() + i * 32;
        for (int j = 0; j < 16; ++j) {
            y[j] = d * static_cast<float>(kvalues_iq4nl[qs[j] & 0x0F]);
            y[j + 16] = d * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
        }
    }
}

void run_case(const char* name, size_t block_size, size_t nblocks,
              void (*forge_row)(const uint8_t*, float*, int, int),
              void (*ref_fn)(const std::vector<uint8_t>&, std::vector<float>&)) {
    // Two rows worth of blocks; feed both rows through the row entry point so
    // the per-row byte offset math is exercised too.
    const size_t row_elems = nblocks * (block_size == 18 ? 32 : 256);
    std::vector<uint8_t> data(2 * nblocks * block_size);
    for (auto& b : data) b = next_byte();

    std::vector<float> want;
    ref_fn(data, want);

    std::vector<float> got(row_elems);
    forge_row(data.data(), got.data(), static_cast<int>(row_elems), 0);
    std::vector<float> got_row2(row_elems);
    forge_row(data.data(), got_row2.data(), static_cast<int>(row_elems), 1);

    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i]) ok = false;
    for (size_t i = 0; i < got_row2.size(); ++i)
        if (got_row2[i] != want[row_elems + i]) ok = false;
    check(ok, name);
}

#ifdef USE_CUDA
void run_iq1_m_cuda() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP IQ1_M CUDA: no device");
        return;
    }
    auto cuda_ok = [](cudaError_t err) {
        if (err == cudaSuccess) return true;
        check(false, cudaGetErrorString(err));
        return false;
    };
    for (int blocks : {1, 3}) {
        const int K = blocks * 256, N = 3, M = 2;
        std::vector<uint8_t> data(N * blocks * 56);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = static_cast<uint8_t>((i * 73 + 19) % 256);
        for (int b = 0; b < N * blocks; ++b) {
            for (int j = 0; j < 4; ++j) {
                auto& byte = data[b * 56 + 49 + 2 * j];
                byte = (byte & 15) | (((0x3c00 >> (4 * j)) & 15) << 4);
            }
        }
        std::vector<float> weights;
        ref_iq1_m(data, weights);
        std::vector<float> x(M * K), expected(M * N, 0.f), got(M * N);
        for (int i = 0; i < M * K; ++i)
            x[i] = i % 32 == 0 ? 127.f : static_cast<float>((i * 17) % 251 - 125);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                for (int k = 0; k < K; ++k)
                    expected[m * N + n] += x[m * K + k] * weights[n * K + k];
        float *dx = nullptr, *dy = nullptr;
        uint8_t* dw = nullptr;
        bool ok = cuda_ok(cudaMalloc(&dx, x.size() * sizeof(float))) &&
                  cuda_ok(cudaMalloc(&dy, got.size() * sizeof(float))) &&
                  cuda_ok(cudaMalloc(&dw, data.size()));
        if (ok)
            ok = cuda_ok(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice)) &&
                 cuda_ok(cudaMemcpy(dw, data.data(), data.size(), cudaMemcpyHostToDevice));
        for (int batch : {1, M}) {
            if (!ok) break;
            if (batch == 1)
                forge::cuda::launch_gemv_iq1_m_q8_1(dx, dw, dy, K, N);
            else
                forge::cuda::launch_gemv_iq1_m_q8_1_batch(dx, dw, dy, M, K, N);
            ok = cuda_ok(cudaGetLastError()) && cuda_ok(cudaDeviceSynchronize()) &&
                 cuda_ok(cudaMemcpy(got.data(), dy, batch * N * sizeof(float), cudaMemcpyDeviceToHost));
            if (!ok) break;
            bool equal = true;
            for (int i = 0; i < batch * N; ++i) {
                if (!std::isfinite(got[i]) || std::fabs(got[i] - expected[i]) > 0.01f) {
                    std::printf("IQ1_M blocks=%d M=%d i=%d expected=%g actual=%g\n",
                                blocks, batch, i, expected[i], got[i]);
                    equal = false;
                }
            }
            check(equal, "IQ1_M CUDA vs scalar reference");
        }
        if (dx) cuda_ok(cudaFree(dx));
        if (dy) cuda_ok(cudaFree(dy));
        if (dw) cuda_ok(cudaFree(dw));
    }
}
#endif

}  // namespace

int main() {
    std::printf("[1] IQ dequant row functions vs llama.cpp scalar reference\n");
    // IQ1_M: 56 B / 256 elems. The old double-advance bug read past the qh
    // field, so mismatch shows from sub-block 1 of the first block onward.
    run_case("iq1_m row dequant (qh advance + scale)", 56, 4,
             forge::ops::dequantize_iq1_m_row, ref_iq1_m);
    // IQ2_XXS: 66 B / 256 elems. The old formula scaled every value 2x.
    run_case("iq2_xxs row dequant (group scale formula)", 66, 4,
             forge::ops::dequantize_iq2_xxs_row, ref_iq2_xxs);
    // IQ4_NL: 18 B / 32 elems (block != QK_K; guards the (K+31)/32 path).
    run_case("iq4_nl row dequant (32-elem block)", 18, 4,
             forge::ops::dequantize_iq4_nl_row, ref_iq4_nl);
#ifdef USE_CUDA
    std::printf("[2] IQ1_M CUDA GEMV vs scalar reference\n");
    run_iq1_m_cuda();
#endif

    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
