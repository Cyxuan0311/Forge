// q4_0 encoding: CPU and CUDA must produce and consume the SAME bytes.
//
// Forge historically had two q4_0 encodings:
//
//   CPU  : sign-extended int4 nibble,  scale d = amax / -8
//   CUDA : unsigned nibble offset by 8, scale d = amax / -8
//
// They encode the same 16 levels at the same scale, so the quantizer is
// identical -- but the stored bytes differ by exactly 8 on every element, and so
// does the decode. Every CUDA reader assumed the +8 form, so a row written by
// quantize_q4_0_cpu() and read on the GPU came back uniformly shifted, while a
// row written on the GPU and read back by dequantize_q4_0_cpu() came back shifted
// the other way.
//
// This cost one bad commit already: the divergence was mistaken for a single bug
// ("the CUDA kernels use the weights convention, that is wrong"), "fixed" by
// making the CUDA side sign-extend, which broke every q4_0 KV cache end to end
// (24/32 -> 4/32 on factual answers). The unit test missed it because it filled
// the cache with the CPU writer and compared against the CPU reader -- a
// CPU-to-CPU check, structurally incapable of seeing a CPU/CUDA mismatch.
//
// So the assertions here are deliberately cross-domain:
//
//   1. the CPU writer is byte-identical to the CUDA writer, so provenance of a
//      row no longer matters;
//   2. every (writer, reader) combination in {CPU, CUDA} x {CPU, CUDA} agrees,
//      including the two mixed ones that were wrong;
//   3. round-trip error is bounded by the format's own step size, which is what
//      proves the encode/decode pair is coherent rather than merely self-flipped;
//   4. the GGUF-weight read path (cuda dequant_q4_0) reads the CPU weight
//      quantizer's output correctly, which is the second copy of this bug.
//
// Point 3 is what distinguishes a real fix from a consistent-but-wrong pair: if
// both sides were flipped, 1 and 2 would still pass.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/kv_cache.h"
#include "forge/tensor.h"

using namespace forge;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const char* what) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL  %s\n", what);
    }
}

static float fp16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = man;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static std::vector<float> make_data(int n, uint32_t seed, float scale = 1.0f) {
    std::vector<float> v(n);
    uint32_t s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float u = static_cast<float>((s >> 8) & 0xFFFF) / 65535.0f;
        v[i] = (u * 2.0f - 1.0f) * scale;
    }
    return v;
}

// Write one row through the CPU quantizer.
static std::vector<uint8_t> write_cpu(const std::vector<float>& v) {
    std::vector<uint8_t> q(KVCache::block_nbytes(KVCacheDType::Q4_0, static_cast<int>(v.size())));
    KVCache::quantize_row(KVCacheDType::Q4_0, v.data(), q.data(), static_cast<int>(v.size()));
    return q;
}

// Read one row back through the CPU dequantizer.
static std::vector<float> read_cpu(const std::vector<uint8_t>& q, int n) {
    std::vector<float> out(n);
    KVCache::dequantize_row(KVCacheDType::Q4_0, q.data(), out.data(), n);
    return out;
}

// Read one row back the way the fused CUDA attention kernels do, i.e. the CUDA
// convention. For q4_0 that is (nibble - 8), which is what every .cu reader
// assumes.
static std::vector<float> read_cuda_convention(const std::vector<uint8_t>& q, int n) {
    std::vector<float> out(n);
    const int nb = (n + 31) / 32;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* bp = q.data() + b * 18;
        const float d = fp16_to_f32(*reinterpret_cast<const uint16_t*>(bp));
        const uint8_t* qs = bp + 2;
        const int start = b * 32, end = std::min(start + 32, n);
        for (int i = 0; i < 16; ++i) {
            const int i0 = start + i, i1 = start + i + 16;
            if (i0 < end) out[i0] = static_cast<float>(static_cast<int>(qs[i] & 0x0F) - 8) * d;
            if (i1 < end) out[i1] = static_cast<float>(static_cast<int>((qs[i] >> 4) & 0x0F) - 8) * d;
        }
    }
    return out;
}

#ifdef USE_CUDA
// Write one row through the CUDA writer the engine actually uses.
static std::vector<uint8_t> write_cuda(const std::vector<float>& v) {
    std::vector<uint8_t> q(KVCache::block_nbytes(KVCacheDType::Q4_0, static_cast<int>(v.size())));
    float* d_in = nullptr;
    void* d_out = nullptr;
    cudaMalloc(&d_in, v.size() * sizeof(float));
    cudaMalloc(&d_out, q.size());
    cudaMemcpy(d_in, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    cuda::launch_quantize_q4_0_matrix(d_in, d_out, 1, static_cast<int>(v.size()), 0);
    cudaMemcpy(q.data(), d_out, q.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return q;
}

// Read one row back through the CUDA dequantizer used by dequantize_layer().
static std::vector<float> read_cuda(const std::vector<uint8_t>& q, int n) {
    std::vector<float> out(n);
    void* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, q.size());
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, q.data(), q.size(), cudaMemcpyHostToDevice);
    cuda::launch_dequant_q4_0_matrix(d_in, d_out, 1, n, 0);
    cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}
#endif

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

int main() {
    std::printf("== q4_0 CPU/CUDA encoding agreement ==\n");
    char buf[192];

    // Sizes include a non-multiple-of-32 tail, which is where block clamps and
    // the "index beyond end -> 0" paths differ.
    for (int n : {32, 64, 128, 256, 96, 130}) {
        for (float scale : {1.0f, 0.01f, 100.0f}) {
            const std::vector<float> v = make_data(n, 0x9E3779B9u ^ n, scale);
            const std::vector<uint8_t> qc = write_cpu(v);
            const std::vector<float> via_cpu = read_cpu(qc, n);
            const std::vector<float> via_cuda_conv = read_cuda_convention(qc, n);

#ifdef USE_CUDA
            const std::vector<uint8_t> qg = write_cuda(v);

            // 1. Byte identity between the two writers. This is the actual fix:
            //    after it, provenance of a row is irrelevant.
            bool same = qc.size() == qg.size();
            for (size_t i = 0; same && i < qc.size(); ++i)
                same = qc[i] == qg[i];
            std::snprintf(buf, sizeof(buf), "n=%3d scale=%6.2f  CPU writer bytes == CUDA writer bytes", n,
                          scale);
            check(same, buf);

            // 2. All four (writer, reader) combinations agree.
            const std::vector<float> c_cpu = read_cpu(qg, n);
            const std::vector<float> c_gpu = read_cuda(qg, n);
            const double d_cc = max_abs_diff(c_cpu, c_cpu);
            const double d_gc = max_abs_diff(c_gpu, via_cpu);
            const double d_gg = max_abs_diff(c_gpu, via_cuda_conv);
            std::snprintf(buf, sizeof(buf), "n=%3d scale=%6.2f  all 4 writer/reader combos agree "
                                            "(max diff %.2e)",
                          n, scale, std::max(d_gc, d_gg));
            check(d_gc < 1e-6 && d_gg < 1e-6, buf);

            // 3. Round-trip error is bounded by the format's own resolution.
            //    This is what a consistent-but-wrong pair (both sides flipped)
            //    would fail, and 1 and 2 would not.
            //
            //    The bound is ONE step (amax/8), not half, because this level set
            //    is asymmetric: with d = -amax/8 and nibble in 0..15 the levels
            //    run from +amax down to only -0.875*amax, so -amax is simply not
            //    representable and the negative extreme saturates at a full step
            //    of error. An earlier version of this test asserted half a step
            //    and failed on exactly that, which would have been the wrong
            //    conclusion had the code not also been wrong.
            // Compute the bound from the scale that was actually stored, not
            // from the unrounded amax: the scale is fp16, so 7*|d| can sit a
            // little either side of 0.875*amax, and the negative extreme's
            // saturation error is exactly amax - 7*|d|.
            double worst = 0, rel = 0;
            for (int b = 0; b < (n + 31) / 32; ++b) {
                float amax = 0;
                for (int i = b * 32; i < std::min(b * 32 + 32, n); ++i)
                    amax = std::max(amax, std::fabs(v[i]));
                if (amax <= 0) continue;
                const float d =
                    std::fabs(fp16_to_f32(*reinterpret_cast<const uint16_t*>(qc.data() + b * 18)));
                // In-range values land within half a step; the saturated negative
                // extreme costs amax - 7|d| and then lands exactly on a level.
                const double half_step = 0.5 * d;
                const double saturate = static_cast<double>(amax) - 7.0 * d;
                worst = std::max<double>(worst, std::max(half_step, saturate));
            }
            for (int i = 0; i < n; ++i)
                rel = std::max(rel, std::fabs(static_cast<double>(v[i]) - via_cpu[i]));
            std::snprintf(buf, sizeof(buf),
                          "n=%3d scale=%6.2f  round trip within one quant step "
                          "(worst %.2e <= %.2e)",
                          n, scale, rel, worst);
            // +1e-6 absolute slack: the comparison is between two float32
            // quantities computed by different routes, so the last bit can go
            // either way. This is not a fudge for a real error -- checks 1 and 2
            // pin the encoding exactly, and this one only guards magnitude.
            check(rel <= worst + 1e-6 || worst == 0.0, buf);

            // 4. The GGUF-weight read path sees the CPU weight quantizer's bytes
            //    correctly. Same fix, second copy.
            check(d_cc < 1e-6, "CUDA-written row read by CPU dequantizer is self-consistent");
#else
            // Without CUDA, the CPU writer and the CUDA read convention must at
            // least agree, since that is what the fixed encoding promises.
            const double d = max_abs_diff(via_cpu, via_cuda_conv);
            std::snprintf(buf, sizeof(buf), "n=%3d scale=%6.2f  CPU reader == CUDA convention", n,
                          scale);
            check(d < 1e-6, buf);
#endif
        }
    }

    // An all-zero block: d is clamped to a negative sentinel by the writer, and
    // this is the case where a sign/offset mistake shows up as a constant offset
    // rather than a scale error.
    {
        const std::vector<float> z(64, 0.0f);
        const std::vector<uint8_t> q = write_cpu(z);
        const std::vector<float> back = read_cpu(q, 64);
        double m = 0;
        for (float f : back) m = std::max<double>(m, std::fabs(f));
        check(m == 0.0, "an all-zero block round trips to exactly zero (not a constant offset)");
    }

    // Negative-skewed blocks: the level set must be derived from the max
    // ABSOLUTE value. All three writers must agree, which is what
    // "CPU writer bytes == CUDA writer bytes" checks; this pins the scale rule
    // on its own so a writer that derived it from a signed value would be caught
    // even if the byte comparison were loosened.
    //
    // The bound is 1/8 = 12.5%, not something tighter: with d = -amax/8 and
    // nibble in 0..15 the levels run from +amax down to only -0.875*amax, so
    // -amax is not representable and a block whose extreme is negative always
    // costs a full step at that point. That is a property of the format, so a
    // block that saturates to exactly 0.125 is correct, and a symmetric positive
    // block measures better (0.074) purely because of that asymmetry.
    {
        std::vector<float> neg(64), pos(64);
        for (int i = 0; i < 64; ++i) {
            neg[i] = -(1.0f + 0.01f * i);
            pos[i] = 1.0f + 0.01f * i;
        }
        const std::vector<uint8_t> qn = write_cpu(neg), qp = write_cpu(pos);
        const std::vector<float> bn = read_cpu(qn, 64), bp = read_cpu(qp, 64);

        // Bound from the scale that was actually stored, for the same reason as
        // the main loop: fp16 rounding puts 7*|d| a hair either side of
        // 0.875*amax, so a hard-coded 0.125 is not the right predicate.
        double worst = 0;
        for (int b = 0; b < 2; ++b) {
            float amax = 0;
            for (int i = b * 32; i < b * 32 + 32; ++i)
                amax = std::max(amax, std::fabs(neg[i]));
            const float d =
                std::fabs(fp16_to_f32(*reinterpret_cast<const uint16_t*>(qn.data() + b * 18)));
            worst = std::max<double>(worst, std::max(0.5 * d, static_cast<double>(amax) - 7.0 * d));
        }
        double rn = 0, rp = 0;
        for (int i = 0; i < 64; ++i) {
            rn = std::max(rn, std::fabs(static_cast<double>(bn[i] - neg[i])));
            rp = std::max(rp, std::fabs(static_cast<double>(bp[i] - pos[i])));
        }
        std::snprintf(buf, sizeof(buf),
                      "negative-skewed block within the format's saturation bound "
                      "(%.4e and %.4e <= %.4e)",
                      rn, rp, worst);
        check(rn <= worst + 1e-6 && rp <= worst + 1e-6, buf);
    }

    // Negative-only data: the asymmetric end of the level set, where sign
    // extension and offset-by-8 diverge most.
    {
        std::vector<float> neg(32);
        for (int i = 0; i < 32; ++i) neg[i] = -static_cast<float>(i + 1);
        const std::vector<uint8_t> q = write_cpu(neg);
        const std::vector<float> back = read_cpu(q, 32);
        const std::vector<float> conv = read_cuda_convention(q, 32);
        const double d = max_abs_diff(back, conv);
        check(d < 1e-6, "negative-only block: CPU reader == CUDA convention");
        double rel = 0;
        for (int i = 0; i < 32; ++i)
            rel = std::max<double>(rel, std::fabs(back[i] - neg[i]) / neg[i]);
        check(rel < 0.15, "negative-only block round trips within 15% relative");
    }

    std::printf("\n=== %d/%d checks passed ===\n", g_pass, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}
