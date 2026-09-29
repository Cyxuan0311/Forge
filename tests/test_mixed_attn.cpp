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
    for (int r = 0; r < blocks; ++r) {
        KVCache::quantize_row(tk, kf.data() + static_cast<size_t>(r) * kv_dim,
                              kq.data() + static_cast<size_t>(r) * k_stride,
                              kv_dim);
        KVCache::quantize_row(tv, vf.data() + static_cast<size_t>(r) * kv_dim,
                              vq.data() + static_cast<size_t>(r) * v_stride,
                              kv_dim);
    }
    // The reference must see what the KERNEL sees, i.e. the dequantized values.
    std::vector<std::vector<float>> k_rows(kv_len), v_rows(kv_len);
    for (int j = 0; j < kv_len; ++j) {
        k_rows[j].resize(kv_dim);
        v_rows[j].resize(kv_dim);
        KVCache::dequantize_row(
            tk, kq.data() + static_cast<size_t>(j) * k_stride,
            k_rows[j].data(), kv_dim);
        KVCache::dequantize_row(
            tv, vq.data() + static_cast<size_t>(j) * v_stride,
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
        KVCache::quantize_row(dt, kf.data() + static_cast<size_t>(r) * kv_dim,
                              kq.data() + static_cast<size_t>(r) * stride, kv_dim);
        KVCache::quantize_row(dt, vf.data() + static_cast<size_t>(r) * kv_dim,
                              vq.data() + static_cast<size_t>(r) * stride, kv_dim);
    }
    for (int j = 0; j < kv_len; ++j) {
        k_rows[j].resize(kv_dim);
        v_rows[j].resize(kv_dim);
        KVCache::dequantize_row(dt, kq.data() + static_cast<size_t>(j) * stride, k_rows[j].data(),
                                kv_dim);
        KVCache::dequantize_row(dt, vq.data() + static_cast<size_t>(j) * stride, v_rows[j].data(),
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
    struct { const char* n; KVCacheDType k, v; } pairs[] = {
        {"q8_0/q8_0", KVCacheDType::Q8_0, KVCacheDType::Q8_0},
        {"q4_0/q4_0", KVCacheDType::Q4_0, KVCacheDType::Q4_0},
        {"f16/f16",   KVCacheDType::F16,  KVCacheDType::F16},
        {"q8_0/q4_0", KVCacheDType::Q8_0, KVCacheDType::Q4_0},
        {"q4_0/q8_0", KVCacheDType::Q4_0, KVCacheDType::Q8_0},
        {"q8_0/f16",  KVCacheDType::Q8_0, KVCacheDType::F16},
        {"f16/q8_0",  KVCacheDType::F16,  KVCacheDType::Q8_0},
        {"f16/q4_0",  KVCacheDType::F16,  KVCacheDType::Q4_0},
    };
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
