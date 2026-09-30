// Paged KV: the device write path and the device read path must agree.
//
// Symptom this isolates: `--paged-kv -ctk q4_0 -ctv q4_0` answers questions
// wrongly ("Seoul" for a question about China's capital) while paged f16, paged
// q8_0 and the *contiguous* q4_0 path are all correct. So the defect is
// specific to the paged q4_0 pair, and the two things that differ there are the
// device writer (quantize_row_q4_0) and the paged decode kernel.
//
// Rather than keep reading plumbing, this test drives the real device code and
// splits the two candidates:
//
//   1. launch_kv_scatter writes rows into a paged pool; the bytes are read back
//      and dequantized on the CPU with the same convention the kernel uses. If
//      that round trip is wrong, the WRITER is at fault.
//   2. Given rows written by the writer, the paged decode kernel is compared
//      against a CPU reference. If (1) passed and this fails, the READER is at
//      fault.
//   3. The same two steps for q8_0 and f16 as controls, so a shared-bug and a
//      q4_0-specific bug are distinguishable.
//
// Note the CPU reference deliberately uses the CPU writer/reader, not the
// device ones. That is the point: the paged path must be able to read what the
// CPU path writes, since row_bytes and block layout are shared with it. (The
// first q4_0 regression in this area was a CPU/CUDA convention split that a
// CPU-only test was structurally unable to see.)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include "forge/cuda_kernels.h"
#include "forge/kv_cache.h"
#include "forge/kv_cache_dtype.h"
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

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    float next() {  // [-1, 1)
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    }
};

// CPU reference attention for one query head, over dequantized rows.
// k_rows[j] / v_rows[j] are whole rows of kv_dim elements (all KV heads).
static std::vector<float> ref_attend(const std::vector<float>& q,
                                     const std::vector<std::vector<float>>& k_rows,
                                     const std::vector<std::vector<float>>& v_rows,
                                     int num_kv_heads, int head_dim) {
    const int kv_len = static_cast<int>(k_rows.size());
    const int num_heads = static_cast<int>(q.size()) / head_dim;
    const int groups = num_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> out(static_cast<size_t>(num_heads) * head_dim, 0.0f);
    for (int h = 0; h < num_heads; ++h) {
        const int kv_h = h / groups;
        std::vector<float> logits(kv_len);
        float mx = -1e30f;
        for (int j = 0; j < kv_len; ++j) {
            const std::vector<float>& kr = k_rows[j];
            double dot = 0;
            for (int d = 0; d < head_dim; ++d)
                dot += static_cast<double>(q[h * head_dim + d]) * kr[static_cast<size_t>(kv_h) * head_dim + d];
            logits[j] = static_cast<float>(dot * scale);
            mx = std::max(mx, logits[j]);
        }
        double z = 0;
        for (int j = 0; j < kv_len; ++j) {
            logits[j] = std::exp(logits[j] - mx);
            z += logits[j];
        }
        for (int j = 0; j < kv_len; ++j) {
            const std::vector<float>& vr = v_rows[j];
            const float w = static_cast<float>(logits[j] / z);
            for (int d = 0; d < head_dim; ++d)
                out[static_cast<size_t>(h) * head_dim + d] +=
                    w * vr[static_cast<size_t>(kv_h) * head_dim + d];
        }
    }
    return out;
}

// Causal prefill reference: query position i attends to k_rows[0..i].
static std::vector<float> ref_attend_causal(const std::vector<float>& q,
                                             const std::vector<std::vector<float>>& k_rows,
                                             const std::vector<std::vector<float>>& v_rows,
                                             int num_kv_heads, int head_dim) {
    const int kv_len = static_cast<int>(k_rows.size());
    const int num_heads = static_cast<int>(q.size()) / head_dim / kv_len;
    const int groups = num_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> out(q.size());
    for (int h = 0; h < num_heads; ++h) {
        const int kv_h = h / groups;
        for (int i = 0; i < kv_len; ++i) {
            std::vector<float> logits(i + 1);
            float mx = -1e30f;
            for (int j = 0; j <= i; ++j) {
                double dot = 0;
                for (int d = 0; d < head_dim; ++d)
                    dot += static_cast<double>(q[(static_cast<size_t>(i) * num_heads + h) * head_dim + d]) *
                           k_rows[j][static_cast<size_t>(kv_h) * head_dim + d];
                logits[j] = static_cast<float>(dot * scale);
                mx = std::max(mx, logits[j]);
            }
            double z = 0;
            for (int j = 0; j <= i; ++j) {
                logits[j] = std::exp(logits[j] - mx);
                z += logits[j];
            }
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0;
                for (int j = 0; j <= i; ++j) {
                    const float w = static_cast<float>(logits[j] / z);
                    acc += w * v_rows[j][static_cast<size_t>(kv_h) * head_dim + d];
                }
                out[(static_cast<size_t>(i) * num_heads + h) * head_dim + d] = acc;
            }
        }
    }
    return out;
}

// Build a paged pool on device: `n_pages` pages of page_size rows each, plus a
// trivial identity page table.
struct PagedPool {
    int page_size = 0, n_pages = 0;
    size_t row_bytes = 0;
    std::vector<void*> k_pages, v_pages;
    // Stored as void** (not void* const*) so cudaMalloc/cudaFree can take its
    // address; passing it to the kernels' void* const* parameter is fine.
    void** d_k_ptrs = nullptr;
    void** d_v_ptrs = nullptr;
    int32_t* d_page_ids = nullptr;

    ~PagedPool() {
        for (auto* p : k_pages) cudaFree(p);
        for (auto* p : v_pages) cudaFree(p);
        if (d_k_ptrs) cudaFree(d_k_ptrs);
        if (d_v_ptrs) cudaFree(d_v_ptrs);
        if (d_page_ids) cudaFree(d_page_ids);
    }
};

static PagedPool make_pool(int page_size, int n_pages, size_t row_bytes) {
    PagedPool pool;
    pool.page_size = page_size;
    pool.n_pages = n_pages;
    pool.row_bytes = row_bytes;
    const size_t page_bytes = static_cast<size_t>(page_size) * row_bytes;
    for (int i = 0; i < n_pages; ++i) {
        void* k = nullptr;
        void* v = nullptr;
        cudaMalloc(&k, page_bytes);
        cudaMalloc(&v, page_bytes);
        cudaMemset(k, 0, page_bytes);
        cudaMemset(v, 0, page_bytes);
        pool.k_pages.push_back(k);
        pool.v_pages.push_back(v);
    }
    std::vector<void*> kp = pool.k_pages, vp = pool.v_pages;
    cudaMalloc(&pool.d_k_ptrs, pool.n_pages * sizeof(void*));
    cudaMalloc(&pool.d_v_ptrs, pool.n_pages * sizeof(void*));
    cudaMemcpy(pool.d_k_ptrs, kp.data(), pool.n_pages * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(pool.d_v_ptrs, vp.data(), pool.n_pages * sizeof(void*), cudaMemcpyHostToDevice);
    std::vector<int32_t> ids(pool.n_pages);
    for (int i = 0; i < pool.n_pages; ++i) ids[i] = i;
    cudaMalloc(&pool.d_page_ids, pool.n_pages * sizeof(int32_t));
    cudaMemcpy(pool.d_page_ids, ids.data(), pool.n_pages * sizeof(int32_t), cudaMemcpyHostToDevice);
    return pool;
}

static void run_dtype(KVCacheDType dt, const char* name, int head_dim, int num_kv_heads,
                      int num_heads, int kv_len, int page_size, KVCacheDType dt_v = KVCacheDType::FP32,
                      bool paged = false) {
    char buf[200];
    const int kv_dim = num_kv_heads * head_dim;
    // row_bytes is per side: K and V may have different block geometries.
    KVCacheDType tv = dt_v == KVCacheDType::FP32 ? dt : dt_v;
    const size_t k_row_bytes = KVCache::block_nbytes(dt, kv_dim);
    const size_t v_row_bytes = KVCache::block_nbytes(tv, kv_dim);
    const size_t row_bytes = k_row_bytes;
    const bool mixed = tv != dt;
    const int n_pages = (kv_len + page_size - 1) / page_size;

    Rng rng(0xC0FFEEu ^ static_cast<uint32_t>(dt));
    std::vector<float> kf(static_cast<size_t>(kv_len) * kv_dim), vf(kf.size()), q(
        static_cast<size_t>(num_heads) * head_dim);
    for (auto& x : kf) x = rng.next();
    for (auto& x : vf) x = rng.next();
    for (auto& x : q) x = rng.next();

    // ---- 1. the device writer ----
    PagedPool pool = make_pool(page_size, n_pages, std::max(k_row_bytes, v_row_bytes));
    float *d_k = nullptr, *d_v = nullptr;
    cudaMalloc(&d_k, kf.size() * sizeof(float));
    cudaMalloc(&d_v, vf.size() * sizeof(float));
    cudaMemcpy(d_k, kf.data(), kf.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, vf.data(), vf.size() * sizeof(float), cudaMemcpyHostToDevice);
    cuda::launch_kv_scatter(d_k, d_v, pool.d_k_ptrs, pool.d_v_ptrs, pool.d_page_ids, kv_len, 0,
                            page_size, kv_dim, k_row_bytes, v_row_bytes, dt, tv, 0);
    cudaDeviceSynchronize();

    // Read the raw bytes back and dequantize on the CPU with the CPU reader.
    // If this is wrong the WRITER is at fault.
    std::vector<std::vector<float>> k_rows(kv_len), v_rows(kv_len);
    for (int kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
        for (int j = 0; j < kv_len; ++j) {
            std::vector<uint8_t> row(k_row_bytes);
            const int page = j / page_size, off = j % page_size;
            const uint8_t* base = static_cast<const uint8_t*>(pool.k_pages[static_cast<size_t>(page)]);
            cudaMemcpy(row.data(), base + static_cast<size_t>(off) * k_row_bytes, k_row_bytes,
                       cudaMemcpyDeviceToHost);
            k_rows[j].resize(kv_dim);
            KVCache::dequantize_row(dt, row.data(), k_rows[j].data(), kv_dim);
            std::vector<uint8_t> vrow(v_row_bytes);
            cudaMemcpy(vrow.data(),
                       static_cast<const uint8_t*>(pool.v_pages[static_cast<size_t>(page)]) +
                           static_cast<size_t>(off) * v_row_bytes,
                       v_row_bytes, cudaMemcpyDeviceToHost);
            v_rows[j].resize(kv_dim);
            KVCache::dequantize_row(tv, vrow.data(), v_rows[j].data(), kv_dim);
        }
    }

    // Compare the device-written rows against the CPU quantizer's own output.
    // Same input, two writers: the bytes must be identical.
    {
        std::vector<uint8_t> cpu_row(row_bytes);
        int mismatches = 0;
        for (int kv_h = 0; kv_h < num_kv_heads && mismatches < 3; ++kv_h) {
            for (int j = 0; j < kv_len && mismatches < 3; ++j) {
                const float* src = kf.data() + static_cast<size_t>(j) * kv_dim + kv_h * head_dim;
                // The device writer quantizes the whole row; slice out this head
                // by quantizing a head-sized row with the same block layout.
                std::vector<uint8_t> hrow(KVCache::block_nbytes(dt, head_dim));
                KVCache::quantize_row(dt, src, hrow.data(), head_dim);
                const size_t hbytes = KVCache::block_nbytes(dt, head_dim);
                const int page = j / page_size, off = j % page_size;
                std::vector<uint8_t> dev_row(hbytes);
                cudaMemcpy(dev_row.data(),
                           static_cast<const uint8_t*>(pool.k_pages[static_cast<size_t>(page)]) +
                               static_cast<size_t>(off) * row_bytes +
                               static_cast<size_t>(kv_h) * hbytes,
                           hbytes, cudaMemcpyDeviceToHost);
                if (dev_row != hrow) ++mismatches;
            }
        }
        std::snprintf(buf, sizeof(buf),
                      "%s writer: device bytes == CPU bytes for a head-sized row", name);
        check(mismatches == 0, buf);
    }

    // ---- 2. the paged decode kernel ----
    float* d_q = nullptr;
    float* d_o = nullptr;
    cudaMalloc(&d_q, q.size() * sizeof(float));
    cudaMalloc(&d_o, q.size() * sizeof(float));
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, q.size() * sizeof(float));

    bool launched = true;
    if (mixed) {
        const auto fmt = [](KVCacheDType d) {
            return d == KVCacheDType::Q8_0 ? 0 : (d == KVCacheDType::Q4_0 ? 1 : 2);
        };
        const bool ok = cuda::launch_paged_flash_attention_gqa_decode_mixed(
            d_q, pool.d_k_ptrs, pool.d_v_ptrs, pool.d_page_ids, d_o, kv_len, num_heads,
            num_kv_heads, head_dim, page_size, k_row_bytes, v_row_bytes, fmt(dt), fmt(tv),
            nullptr, 0);
        cudaDeviceSynchronize();
        std::snprintf(buf, sizeof(buf), "%s/%s paged MIXED decode launched", name,
                      kv_cache_dtype_name(tv));
        check(ok && cudaGetLastError() == cudaSuccess, buf);
        if (!ok) return;
        std::vector<float> gotm(q.size());
        cudaMemcpy(gotm.data(), d_o, gotm.size() * sizeof(float), cudaMemcpyDeviceToHost);
        const std::vector<float> refm = ref_attend(q, k_rows, v_rows, num_kv_heads, head_dim);
        double me = 0, ms = 0;
        for (size_t i = 0; i < gotm.size(); ++i) {
            me = std::max<double>(me, std::fabs(gotm[i] - refm[i]));
            ms = std::max<double>(ms, std::fabs(refm[i]));
        }
        std::snprintf(buf, sizeof(buf),
                      "%s/%s paged MIXED decode matches CPU reference (max err %.2e, |out| %.3f)",
                      name, kv_cache_dtype_name(tv), me, ms);
        check(me < 5e-2 * std::max(1e-3, ms), buf);
    } else {
    switch (dt) {
    case KVCacheDType::Q4_0:
        cuda::launch_paged_flash_attention_gqa_decode_q4_0(
            d_q, pool.d_k_ptrs, pool.d_v_ptrs, pool.d_page_ids, d_o, kv_len, num_heads,
            num_kv_heads, head_dim, page_size, row_bytes, nullptr, 0);
        break;
    case KVCacheDType::Q8_0:
        cuda::launch_paged_flash_attention_gqa_decode_q8_0(
            d_q, pool.d_k_ptrs, pool.d_v_ptrs, pool.d_page_ids, d_o, kv_len, num_heads,
            num_kv_heads, head_dim, page_size, row_bytes, nullptr, 0);
        break;
    case KVCacheDType::F16:
        cuda::launch_paged_flash_attention_gqa_decode_f16(
            d_q, pool.d_k_ptrs, pool.d_v_ptrs, pool.d_page_ids, d_o, kv_len, num_heads,
            num_kv_heads, head_dim, page_size, row_bytes, nullptr, 0);
        break;
    default:
        launched = false;
        break;
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    std::snprintf(buf, sizeof(buf), "%s paged decode launched cleanly (%s)", name,
                  cudaGetErrorString(err));
    check(launched && err == cudaSuccess, buf);
    if (!launched || err != cudaSuccess) return;

    std::vector<float> got(q.size());
    cudaMemcpy(got.data(), d_o, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
    const std::vector<float> ref = ref_attend(q, k_rows, v_rows, num_kv_heads, head_dim);
    double max_err = 0, scale = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::max<double>(max_err, std::fabs(got[i] - ref[i]));
        scale = std::max<double>(scale, std::fabs(ref[i]));
    }
    std::snprintf(buf, sizeof(buf), "%s paged decode matches CPU reference (max err %.2e, |out| %.3f)",
                  name, max_err, scale);
    // q4_0 has 16 levels, so a dequantized element can be off by scale/2 and it
    // propagates through the weighted sum. Tighter than the mixed-attn tolerance
    // would be, because here the writer and reader are the same pair.
    const double tol = (dt == KVCacheDType::Q4_0) ? 5e-2 : 2e-2;
    check(max_err < tol * std::max(1e-3, scale), buf);
    }  // end else: symmetric decode. The mixed case above falls through to prefill.

    // ---- 3. the paged *prefill* kernel ----
    // The decode check above cannot see a prefill bug: decode is only ever given
    // rows the prefill already produced, so a prefill that misreads the cache
    // still yields a self-consistent decode over corrupted rows, and the very
    // first generated token is already wrong. This is the coverage that was
    // missing when --paged-kv -ctk q4_0 answered "Seoul" five times out of five
    // while paged f16 and paged q8_0 were right.
    {
        const int q_len = kv_len;
        std::vector<float> qp(static_cast<size_t>(q_len) * num_heads * head_dim);
        Rng rng2(0xBEEF01u ^ static_cast<uint32_t>(dt));
        for (auto& x : qp) x = rng2.next();
        float* d_qp = nullptr;
        float* d_op = nullptr;
        cudaMalloc(&d_qp, qp.size() * sizeof(float));
        cudaMalloc(&d_op, qp.size() * sizeof(float));
        cudaMemcpy(d_qp, qp.data(), qp.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_op, 0, qp.size() * sizeof(float));

        bool ok = true;
        if (mixed) {
            const auto run = [&](auto fn) {
                fn(d_qp, pool.d_k_ptrs, pool.d_v_ptrs, pool.d_page_ids, d_op, q_len, kv_len,
                   num_heads, num_kv_heads, head_dim, page_size, k_row_bytes, v_row_bytes, nullptr,
                   true, nullptr, nullptr, 0);
            };
            if (dt == KVCacheDType::Q8_0 && tv == KVCacheDType::Q4_0)
                run(cuda::launch_paged_flash_attention_gqa_prefill_q8_0_q4_0);
            else if (dt == KVCacheDType::Q4_0 && tv == KVCacheDType::Q8_0)
                run(cuda::launch_paged_flash_attention_gqa_prefill_q4_0_q8_0);
            else if (dt == KVCacheDType::Q8_0 && tv == KVCacheDType::F16)
                run(cuda::launch_paged_flash_attention_gqa_prefill_q8_0_f16);
            else if (dt == KVCacheDType::F16 && tv == KVCacheDType::Q8_0)
                run(cuda::launch_paged_flash_attention_gqa_prefill_f16_q8_0);
            else if (dt == KVCacheDType::Q4_0 && tv == KVCacheDType::F16)
                run(cuda::launch_paged_flash_attention_gqa_prefill_q4_0_f16);
            else if (dt == KVCacheDType::F16 && tv == KVCacheDType::Q4_0)
                run(cuda::launch_paged_flash_attention_gqa_prefill_f16_q4_0);
            else
                ok = false;
            cudaDeviceSynchronize();
            std::snprintf(buf, sizeof(buf), "%s/%s paged MIXED prefill launched cleanly", name,
                          kv_cache_dtype_name(tv));
            check(ok && cudaGetLastError() == cudaSuccess, buf);
            if (!ok) {
                cudaFree(d_qp);
                cudaFree(d_op);
                return;
            }
            std::vector<float> gotm(qp.size());
            cudaMemcpy(gotm.data(), d_op, gotm.size() * sizeof(float), cudaMemcpyDeviceToHost);
            const std::vector<float> refm = ref_attend_causal(qp, k_rows, v_rows, num_kv_heads,
                                                              head_dim);
            double me = 0, ms = 0;
            for (size_t i = 0; i < gotm.size(); ++i) {
                me = std::max<double>(me, std::fabs(gotm[i] - refm[i]));
                ms = std::max<double>(ms, std::fabs(refm[i]));
            }
            std::snprintf(buf, sizeof(buf),
                          "%s/%s paged MIXED prefill matches CPU reference (max err %.2e, |out| %.3f)",
                          name, kv_cache_dtype_name(tv), me, ms);
            check(me < 5e-2 * std::max(1e-3, ms), buf);
            cudaFree(d_qp);
            cudaFree(d_op);
            return;
        }
        switch (dt) {
        case KVCacheDType::Q4_0:
            cuda::launch_paged_flash_attention_gqa_prefill_q4_0(
                d_qp, pool.d_k_ptrs, pool.d_v_ptrs, pool.d_page_ids, d_op, q_len, kv_len,
                num_heads, num_kv_heads, head_dim, page_size, row_bytes, nullptr, true, nullptr,
                nullptr, 0);
            break;
        case KVCacheDType::Q8_0:
            cuda::launch_paged_flash_attention_gqa_prefill_q8_0(
                d_qp, pool.d_k_ptrs, pool.d_v_ptrs, pool.d_page_ids, d_op, q_len, kv_len,
                num_heads, num_kv_heads, head_dim, page_size, row_bytes, nullptr, true, nullptr,
                nullptr, 0);
            break;
        case KVCacheDType::F16:
            cuda::launch_paged_flash_attention_gqa_prefill_f16(
                d_qp, pool.d_k_ptrs, pool.d_v_ptrs, pool.d_page_ids, d_op, q_len, kv_len,
                num_heads, num_kv_heads, head_dim, page_size, row_bytes, nullptr, true, nullptr,
                nullptr, 0);
            break;
        default:
            ok = false;
            break;
        }
        cudaDeviceSynchronize();
        cudaError_t perr = cudaGetLastError();
        std::snprintf(buf, sizeof(buf), "%s paged prefill launched cleanly (%s)", name,
                      cudaGetErrorString(perr));
        check(ok && perr == cudaSuccess, buf);
        if (ok && perr == cudaSuccess) {
            std::vector<float> gotp(qp.size());
            cudaMemcpy(gotp.data(), d_op, gotp.size() * sizeof(float), cudaMemcpyDeviceToHost);
            const std::vector<float> refp = ref_attend_causal(qp, k_rows, v_rows, num_kv_heads,
                                                              head_dim);
            double pe = 0, ps = 0;
            for (size_t i = 0; i < gotp.size(); ++i) {
                pe = std::max<double>(pe, std::fabs(gotp[i] - refp[i]));
                ps = std::max<double>(ps, std::fabs(refp[i]));
            }
            std::snprintf(buf, sizeof(buf),
                          "%s paged prefill matches CPU reference (max err %.2e, |out| %.3f)",
                          name, pe, ps);
            const double ptol = (dt == KVCacheDType::Q4_0) ? 5e-2 : 2e-2;
            check(pe < ptol * std::max(1e-3, ps), buf);
        }
        cudaFree(d_qp);
        cudaFree(d_op);
    }

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_q);
    cudaFree(d_o);
}

int main() {
    std::printf("== paged KV: device write/read agreement ==\n");
#ifdef USE_CUDA
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev == 0) {
        std::printf("no CUDA device; skipping\n");
        return 0;
    }
    // MiniCPM5-2B geometry, plus a GQA one and a non-multiple-of-page length.
    for (auto geom : {std::make_tuple(128, 2, 16), std::make_tuple(128, 16, 16),
                      std::make_tuple(64, 2, 16)}) {
        const int hd = std::get<0>(geom), nkv = std::get<1>(geom), nh = std::get<2>(geom);
        for (int page_size : {16}) {
            for (int kv_len : {33, 64}) {
                run_dtype(KVCacheDType::Q4_0, "q4_0", hd, nkv, nh, kv_len, page_size);
                run_dtype(KVCacheDType::Q8_0, "q8_0", hd, nkv, nh, kv_len, page_size);
                run_dtype(KVCacheDType::F16, "f16  ", hd, nkv, nh, kv_len, page_size);
                // Asymmetric K/V: the whole point of the paged mixed kernel. K and
                // V have different block geometries, so both the row strides and
                // the per-head byte offsets differ -- one row_bytes cannot serve
                // both sides, which is why this combination had no path at all.
                run_dtype(KVCacheDType::Q8_0, "q8_0", hd, nkv, nh, kv_len, page_size,
                          KVCacheDType::Q4_0);
                run_dtype(KVCacheDType::Q4_0, "q4_0", hd, nkv, nh, kv_len, page_size,
                          KVCacheDType::Q8_0);
                run_dtype(KVCacheDType::Q8_0, "q8_0", hd, nkv, nh, kv_len, page_size,
                          KVCacheDType::F16);
                run_dtype(KVCacheDType::F16, "f16  ", hd, nkv, nh, kv_len, page_size,
                          KVCacheDType::Q4_0);
            }
        }
    }
#else
    std::printf("built without CUDA; skipping\n");
#endif
    std::printf("\n=== %d/%d checks passed ===\n", g_pass, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}
