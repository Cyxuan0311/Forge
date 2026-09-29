// Mixed-dtype fused attention: correctness against a CPU FP32 reference.
//
// The fused mixed kernel reads K and V from the cache at two different
// precisions with two different row strides. Two things can go wrong that a
// "the output still reads like prose" check will not catch:
//
//   - a wrong per-head byte offset, because the two sides have different block
//     geometries (q8_0 is 32 elems / 34 B, q4_0 is 32 elems / 18 B, f16 is
//     8 elems / 16 B), so the head slice sits at a different offset on each;
//   - dequantising with the wrong convention. q4_0 in particular splits a
//     16-nibble block into elements [0,16) from the low nibble and [16,32) from
//     the high nibble, and transposing that silently produces plausible garbage.
//
// So: fill a real KV cache through the same quantize_row() the engine uses,
// run the kernel, and compare against attention computed on the CPU from the
// dequantized rows. Every (K,V) pair the selector can hand to the mixed path
// is checked, and the symmetric pairs are checked too so the new kernel is
// proven equal to the monomorphic ones it replaces.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "forge/kv_cache.h"
#include "forge/kv_cache_dtype.h"

using namespace forge;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        ++g_checks;                                                   \
        if (!(cond)) {                                                \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); \
            ++g_fail;                                                 \
        } else {                                                      \
            std::printf("ok   %s\n", msg);                            \
        }                                                             \
    } while (0)

#ifdef USE_CUDA
#include "forge/cuda_kernels.h"

// CPU reference: dequantize both rows, then plain scaled-dot-product attention
// for one query head. Mirrors what the kernel computes, independently written.
static std::vector<float> reference_attend(const std::vector<float>& q,
                                            const std::vector<std::vector<float>>& k_rows,
                                            const std::vector<std::vector<float>>& v_rows,
                                            int num_kv_heads, int head_dim) {
    const int kv_len = static_cast<int>(k_rows.size());
    const int num_heads = static_cast<int>(q.size()) / head_dim;
    const int groups = num_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> out(static_cast<size_t>(num_heads) * head_dim, 0.0f);

    for (int h = 0; h < num_heads; ++h) {
        const int kvh = h / groups;
        std::vector<float> logits(kv_len);
        float mx = -1e30f;
        for (int j = 0; j < kv_len; ++j) {
            const float* krow = k_rows[j].data() + static_cast<size_t>(kvh) * head_dim;
            float d = 0.0f;
            for (int i = 0; i < head_dim; ++i) d += q[h * head_dim + i] * krow[i];
            logits[j] = d * scale;
            mx = std::max(mx, logits[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < kv_len; ++j) {
            logits[j] = std::exp(logits[j] - mx);
            sum += logits[j];
        }
        for (int j = 0; j < kv_len; ++j) logits[j] /= sum;
        for (int j = 0; j < kv_len; ++j) {
            const float* vrow = v_rows[j].data() + static_cast<size_t>(kvh) * head_dim;
            const float p = logits[j];
            for (int i = 0; i < head_dim; ++i) out[h * head_dim + i] += p * vrow[i];
        }
    }
    return out;
}

// Deterministic pseudo-random in [-1,1] so failures reproduce.
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        return (static_cast<float>((s >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
    }
};

// Write a row exactly as the CUDA KV cache write path does.
static void quantize_row_engine(KVCacheDType dt, const float* src, uint8_t* dst, int n) {
#ifdef USE_CUDA
    switch (dt) {
    case KVCacheDType::F16:
        cuda::launch_quantize_f16_matrix(src, dst, 1, n, 0);
        return;
    case KVCacheDType::Q8_0:
        cuda::launch_quantize_q8_0_matrix(src, dst, 1, n, 0);
        return;
    case KVCacheDType::Q4_0:
        cuda::launch_quantize_q4_0_matrix(src, dst, 1, n, 0);
        return;
    default:
        break;
    }
#endif
    KVCache::quantize_row(dt, src, dst, n);
}

// Minimal IEEE fp16 -> fp32. Avoids fp16_compat.h, which defines its own __half
// and collides with CUDA's once cuda_kernels.h is included in the same TU.
static float fp16_to_float_cpu(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;  // +/-0
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
        bits = sign | 0x7F800000u | (man << 13);  // inf / nan
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Read a row back the way the fused kernel does, i.e. the CUDA-side convention.
// For q4_0 this is unsigned-offset-by-8, which differs from
// dequantize_q4_0_cpu(); see the note in run_pair().
static void dequantize_as_engine_reads(KVCacheDType dt, const uint8_t* src, float* out, int n) {
    if (dt == KVCacheDType::Q4_0) {
        const int nb = (n + 31) / 32;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* bp = src + b * 18;
            float d;
            d = fp16_to_float_cpu(*reinterpret_cast<const uint16_t*>(bp));
            const uint8_t* qs = bp + 2;
            const int start = b * 32, end = std::min(start + 32, n);
            for (int i = 0; i < 16; ++i) {
                const int i0 = start + i, i1 = start + i + 16;
                if (i0 < end) out[i0] = static_cast<float>((qs[i] & 0x0F) - 8) * d;
                if (i1 < end) out[i1] = static_cast<float>((qs[i] >> 4) - 8) * d;
            }
        }
        return;
    }
    KVCache::dequantize_row(dt, src, out, n);
}

static bool run_pair(KVCacheDType tk, KVCacheDType tv, int head_dim, int num_kv_heads, int num_heads,
                     int kv_len, bool* launched, float* max_err) {
    const int kv_dim = num_kv_heads * head_dim;
    const int blocks = num_kv_heads * kv_len;  // rows

    // Build the float data, quantize through the real path, and keep the
    // dequantized copy for the reference.
    Rng rng(0x5eed1234u ^ static_cast<uint32_t>(tk) ^ (static_cast<uint32_t>(tv) << 8));
    std::vector<float> kf(static_cast<size_t>(blocks) * kv_dim);
    std::vector<float> vf(static_cast<size_t>(blocks) * kv_dim);
    for (auto& x : kf) x = rng.next();
    for (auto& x : vf) x = rng.next();

    // Size the quantized buffers from block_nbytes, NOT from the float source.
    // q8_0 is *larger* than fp32 for the tail of each 32-element group: 256
    // elements become 272 bytes, so a buffer sized off the float data
    // under-allocates by 6 bytes per row and corrupts the heap.
    const size_t k_stride = KVCache::block_nbytes(tk, kv_dim);
    const size_t v_stride = KVCache::block_nbytes(tv, kv_dim);
    std::vector<uint8_t> kq(blocks * k_stride), vq(blocks * v_stride);
    // Quantize with the WRITER THE ENGINE ACTUALLY USES.
    //
    // KVCache::quantize_row() dispatches to the CPU quantizer, but on a CUDA
    // layer the cache is written by cuda::launch_quantize_*_matrix. Those two
    // disagree for q4_0: the CPU path stores a sign-extended int4, the CUDA path
    // stores an unsigned nibble offset by 8. The first version of this test used
    // only the CPU pair, so it checked CPU-writer against CPU-reader and passed
    // a kernel that was wrong for every row the engine ever wrote -- which is
    // how a "pre-existing bug" fix ended up breaking q4_0 KV outright (answer
    // accuracy fell from 10/16 to 4/16 end to end).
    //
    // So: write with the device writer, and read the reference back through the
    // same decode the kernel performs. That makes the test sensitive to the
    // writer/reader agreement that actually matters, instead of to the
    // self-consistency of a path the engine does not take.
    for (int r = 0; r < blocks; ++r) {
        quantize_row_engine(tk, kf.data() + static_cast<size_t>(r) * kv_dim,
                            kq.data() + static_cast<size_t>(r) * k_stride, kv_dim);
        quantize_row_engine(tv, vf.data() + static_cast<size_t>(r) * kv_dim,
                            vq.data() + static_cast<size_t>(r) * v_stride, kv_dim);
    }
    // The reference must see what the KERNEL sees, i.e. the dequantized values.
    std::vector<std::vector<float>> k_rows(kv_len), v_rows(kv_len);
    for (int j = 0; j < kv_len; ++j) {
        k_rows[j].resize(kv_dim);
        v_rows[j].resize(kv_dim);
        dequantize_as_engine_reads(tk, kq.data() + static_cast<size_t>(j) * k_stride,
                                   k_rows[j].data(), kv_dim);
        dequantize_as_engine_reads(tv, vq.data() + static_cast<size_t>(j) * v_stride,
                                   v_rows[j].data(), kv_dim);
    }

    std::vector<float> q(static_cast<size_t>(num_heads) * head_dim);
    for (auto& x : q) x = rng.next();

    const size_t k_row_size = k_stride;
    const size_t v_row_size = v_stride;

    // Kernel-side: the cache is a flat [blocks, row] buffer, which is exactly
    // what d_q_key_cache() points at in the engine.
    float *d_k = nullptr, *d_v = nullptr, *d_q = nullptr, *d_o = nullptr;
    cudaMalloc(&d_k, kq.size());
    cudaMalloc(&d_v, vq.size());
    cudaMalloc(&d_q, q.size() * sizeof(float));
    cudaMalloc(&d_o, static_cast<size_t>(num_heads) * head_dim * sizeof(float));
    cudaMemcpy(d_k, kq.data(), kq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, vq.data(), vq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, static_cast<size_t>(num_heads) * head_dim * sizeof(float));

    const auto fmt = [](KVCacheDType d) {
        switch (d) {
        case KVCacheDType::Q8_0: return 0;
        case KVCacheDType::Q4_0: return 1;
        case KVCacheDType::F16:  return 2;
        default: return -1;
        }
    };
    const bool ok = cuda::launch_fused_flash_attention_gqa_decode_mixed(
        d_q, d_k, d_v, d_o, kv_len, num_heads, num_kv_heads, head_dim, k_row_size, v_row_size,
        fmt(tk), fmt(tv), nullptr, 0);
    *launched = ok;
    if (!ok) {
        cudaFree(d_k); cudaFree(d_v); cudaFree(d_q); cudaFree(d_o);
        return false;
    }
    cudaDeviceSynchronize();

    std::vector<float> got(static_cast<size_t>(num_heads) * head_dim);
    cudaMemcpy(got.data(), d_o, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_k); cudaFree(d_v); cudaFree(d_q); cudaFree(d_o);

    const std::vector<float> ref = reference_attend(q, k_rows, v_rows, num_kv_heads, head_dim);
    float worst = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::fabs(got[i] - ref[i]));
    *max_err = worst;
    return true;
}

// Same check, but through the long-standing monomorphic launcher, so the fix
// that landed in those kernels is covered too.
static bool run_monomorphic(KVCacheDType dt, int head_dim, int num_kv_heads, int num_heads,
                            int kv_len, bool* launched, float* max_err) {
    const int kv_dim = num_kv_heads * head_dim;
    const int blocks = num_kv_heads * kv_len;
    Rng rng(0xabcd0000u ^ static_cast<uint32_t>(dt));
    std::vector<float> kf(static_cast<size_t>(blocks) * kv_dim);
    std::vector<float> vf(static_cast<size_t>(blocks) * kv_dim);
    for (auto& x : kf) x = rng.next();
    for (auto& x : vf) x = rng.next();
    const size_t stride = KVCache::block_nbytes(dt, kv_dim);
    std::vector<uint8_t> kq(blocks * stride), vq(blocks * stride);
    std::vector<std::vector<float>> k_rows(kv_len), v_rows(kv_len);
    for (int r = 0; r < blocks; ++r) {
        quantize_row_engine(dt, kf.data() + static_cast<size_t>(r) * kv_dim,
                              kq.data() + static_cast<size_t>(r) * stride, kv_dim);
        quantize_row_engine(dt, vf.data() + static_cast<size_t>(r) * kv_dim,
                              vq.data() + static_cast<size_t>(r) * stride, kv_dim);
    }
    for (int j = 0; j < kv_len; ++j) {
        k_rows[j].resize(kv_dim);
        v_rows[j].resize(kv_dim);
        dequantize_as_engine_reads(dt, kq.data() + static_cast<size_t>(j) * stride, k_rows[j].data(),
                                kv_dim);
        dequantize_as_engine_reads(dt, vq.data() + static_cast<size_t>(j) * stride, v_rows[j].data(),
                                kv_dim);
    }
    std::vector<float> q(static_cast<size_t>(num_heads) * head_dim);
    for (auto& x : q) x = rng.next();

    float *d_k = nullptr, *d_v = nullptr, *d_q = nullptr, *d_o = nullptr;
    cudaMalloc(&d_k, kq.size());
    cudaMalloc(&d_v, vq.size());
    cudaMalloc(&d_q, q.size() * sizeof(float));
    cudaMalloc(&d_o, static_cast<size_t>(num_heads) * head_dim * sizeof(float));
    cudaMemcpy(d_k, kq.data(), kq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, vq.data(), vq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, static_cast<size_t>(num_heads) * head_dim * sizeof(float));

    const auto L = [&](auto fn) {
        switch (dt) {
        case KVCacheDType::Q8_0: fn(cuda::launch_fused_flash_attention_gqa_decode_q8_0); break;
        case KVCacheDType::Q4_0: fn(cuda::launch_fused_flash_attention_gqa_decode_q4_0); break;
        case KVCacheDType::F16:  fn(cuda::launch_fused_flash_attention_gqa_decode_f16); break;
        default: break;
        }
    };
    *launched = true;
    L([&](auto fn) {
        fn(static_cast<const float*>(d_q), d_k, d_v, static_cast<float*>(d_o), kv_len, num_heads,
           num_kv_heads, head_dim, stride, static_cast<const float*>(nullptr), 0);
    });
    cudaDeviceSynchronize();
    std::vector<float> got(static_cast<size_t>(num_heads) * head_dim);
    cudaMemcpy(got.data(), d_o, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_k); cudaFree(d_v); cudaFree(d_q); cudaFree(d_o);
    const std::vector<float> ref = reference_attend(q, k_rows, v_rows, num_kv_heads, head_dim);
    float worst = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::fabs(got[i] - ref[i]));
    *max_err = worst;
    return true;
}

static void test_pair(const char* name, KVCacheDType tk, KVCacheDType tv, int head_dim) {
    char msg[160];
    // Sweep the GQA geometry as well as the dtype pair. num_kv_heads == num_heads
    // means kv_groups == 1, i.e. every head maps to its own KV head, and that is
    // a different k_head/v_head offset path than the grouped case. The first
    // version of this test only ever used num_kv_heads = 2, so the MHA case was
    // never checked -- and it is the geometry MiniCPM5-2B actually uses, where
    // the kernel produced garbage that this test could not see.
    for (auto geom : {std::make_pair(2, 16), std::make_pair(16, 16), std::make_pair(4, 16)}) {
        const int nkv = geom.first, nh = geom.second;
        bool launched = false;
        float err = 0.0f;
        const bool ran = run_pair(tk, tv, head_dim, nkv, nh, 37, &launched, &err);
        std::snprintf(msg, sizeof(msg), "%s head_dim=%d heads=%d kv_heads=%d runs", name, head_dim,
                      nh, nkv);
        CHECK(ran && launched, msg);
        if (!ran || !launched) continue;
        const bool ok = err < 2e-2f;
        std::snprintf(msg, sizeof(msg), "%s head_dim=%d heads=%d kv_heads=%d matches CPU (%.2e)",
                      name, head_dim, nh, nkv, err);
        CHECK(ok, msg);
    }
}

static void test_pair_old(const char* name, KVCacheDType tk, KVCacheDType tv, int head_dim) {
    bool launched = false;
    float err = 0.0f;
    const bool ran = run_pair(tk, tv, head_dim, 2, 16, 37, &launched, &err);
    char msg[128];
    std::snprintf(msg, sizeof(msg), "%s head_dim=%d runs", name, head_dim);
    CHECK(ran && launched, msg);
    if (!ran || !launched) return;
    // Tolerance is set by the coarse format: q4_0 has 16 levels, so a single
    // dequantised element can be off by up to scale/2 and that propagates
    // through the weighted sum. 2e-2 catches a wrong stride or a transposed
    // nibble order by orders of magnitude while accepting format error.
    const bool ok = err < 2e-2f;
    std::snprintf(msg, sizeof(msg), "%s head_dim=%d matches CPU reference (max err %.2e)", name,
                  head_dim, err);
    CHECK(ok, msg);
}
#endif  // USE_CUDA

int main() {
    std::printf("=== forge-mixed-attn-test ===\n");
#ifdef USE_CUDA
    // The pairs the selector can route here. Symmetric ones are included so the
    // new kernel is shown equal to the monomorphic ones it stands in for.
    // All 9 combinations of {q8_0, q4_0, f16}, generated rather than listed.
    //
    // The hand-written list had 8 entries and omitted q4_0/f16, so that pair had
    // no coverage and -ctk q4_0 -ctv f16 crashed the process with a thrown
    // "launcher refused the requested combination". Enumerating the cross
    // product means a new format cannot be added to the launcher's dispatch
    // table without this test noticing.
    const KVCacheDType types[] = {KVCacheDType::Q8_0, KVCacheDType::Q4_0, KVCacheDType::F16};
    const char* names[] = {"q8_0", "q4_0", "f16"};
    struct { const char* n; KVCacheDType k, v; } pairs[9];
    int np = 0;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            static std::string buf[9];
            buf[np] = std::string(names[a]) + "/" + names[b];
            pairs[np].n = buf[np].c_str();
            pairs[np].k = types[a];
            pairs[np].v = types[b];
            ++np;
        }
    }
    CHECK(np == 9, "all 9 dtype pairs enumerated");

    for (const auto& p : pairs) {
        for (int hd : {64, 96, 128}) test_pair(p.n, p.k, p.v, hd);
    }

    // The pre-existing monomorphic kernels, against the same reference. They are
    // the paths --kv-cache-dtype q4_0 has always taken, and they carried a q4_0
    // dequantisation that disagreed with dequantize_q4_0_cpu() -- the weights
    // convention (n & 0xF) - 8 instead of int4 sign extension. Output stayed
    // fluent, so only a numeric comparison against the dequantized rows catches
    // it. Keep these checks so it cannot come back.
    std::printf("\n--- pre-existing monomorphic kernels vs the same reference ---\n");
    struct Mono { const char* n; KVCacheDType d; };
    const Mono monos[] = {
        {"q8_0/q8_0", KVCacheDType::Q8_0},
        {"q4_0/q4_0", KVCacheDType::Q4_0},
        {"f16/f16", KVCacheDType::F16},
    };
    for (const auto& m : monos) {
        for (int hd : {64, 96, 128}) {
            bool launched = false;
            float err = 0.0f;
            if (!run_monomorphic(m.d, hd, 2, 16, 37, &launched, &err)) continue;
            char msg[128];
            std::snprintf(msg, sizeof(msg), "monomorphic %s head_dim=%d matches reference (%.2e)",
                          m.n, hd, err);
            CHECK(err < 2e-2f, msg);
        }
    }
    // A head_dim the launcher does not cover must be refused, not guessed at.
    {
        bool launched = false;
        float err = 0.0f;
        const bool ran = run_pair(KVCacheDType::Q8_0, KVCacheDType::Q4_0, 256, 2, 16, 8, &launched,
                                  &err);
        CHECK(!ran || !launched, "head_dim 256 is refused, not silently wrong");
    }
#else
    std::printf("NOTE: built without CUDA; nothing to check\n");
#endif
    std::printf("\n=== %d/%d checks passed ===\n", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
