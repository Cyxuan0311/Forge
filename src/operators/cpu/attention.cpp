#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/operator_attention.h"
#include "forge/perf_profiler.h"

#include "cpu/simd.h"
#include "cpu/common/thread_pool.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {
namespace ops {

// Tiled prefill attention for a single query head.
// Adapted from llama.cpp ggml_compute_forward_flash_attn_ext_tiled (ops.cpp:8706):
//   - Q packed into BLOCK_Q rows, K packed transposed (K_f32[d*tile_kv + tk]),
//     then QK computed as a GEMM block (C += A*B, accumulating micro-kernel);
//   - per-KV-tile online softmax rescale; output accumulated as P * V via a
//     second GEMM. This amortizes SIMD over a block and keeps each KV tile in
//     cache across all Q rows in the tile (vs. the old per-(h,i) scalar dot,
//     which reloaded K once per query row).
// Layouts (row-major):
//   q: [q_len, num_heads, head_dim],  k/v: [kv_len, num_kv_heads, head_dim]
//   out: [q_len, num_heads, head_dim]  (flattened; q_stride == out stride)
static void flash_attn_prefill_head(
    const float* q_data, const float* k_data, const float* v_data,
    float* out_data, int q_len, int kv_len, int num_heads, int num_kv_heads,
    int head_dim, int h, const float* mask_data, bool causal, float scale) {
    const int kv_group_size = num_heads / num_kv_heads;
    const int kv_h = h / kv_group_size;
    const int q_stride  = num_heads * head_dim;
    const int kv_stride = num_kv_heads * head_dim;

    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;

    std::vector<float> Q_q(BLOCK_Q * head_dim);
    std::vector<float> KQ(BLOCK_Q * BLOCK_KV);
    std::vector<float> VKQ32(BLOCK_Q * head_dim);
    std::vector<float> V32(BLOCK_KV * head_dim);
    std::vector<float> K_f32(head_dim * BLOCK_KV);

    const float* q_head = q_data + (size_t)h * head_dim;
    const float* k_head = k_data + (size_t)kv_h * head_dim;
    const float* v_head = v_data + (size_t)kv_h * head_dim;
    float* out_head = out_data + (size_t)h * head_dim;

    float M[BLOCK_Q];
    float S[BLOCK_Q];

    for (int i = 0; i < q_len; i += BLOCK_Q) {
        const int tile_q = BLOCK_Q < (q_len - i) ? BLOCK_Q : (q_len - i);

        // Load Q tile (F32), zero-pad unused rows (harmless; never written out)
        for (int tq = 0; tq < tile_q; ++tq) {
            const float* qr = q_head + (size_t)(i + tq) * q_stride;
            std::memcpy(Q_q.data() + (size_t)tq * head_dim, qr, (size_t)head_dim * sizeof(float));
        }
        for (int tq = 0; tq < tile_q; ++tq) { M[tq] = -1e30f; S[tq] = 0.0f; }
        std::memset(VKQ32.data(), 0, (size_t)tile_q * head_dim * sizeof(float));

        for (int ic = 0; ic < kv_len; ic += BLOCK_KV) {
            const int tile_kv = BLOCK_KV < (kv_len - ic) ? BLOCK_KV : (kv_len - ic);

            // Pack K tile transposed: K_f32[d*tile_kv + tk] = k[kv_h][ic+tk][d]
            // Stride is tile_kv (NOT BLOCK_KV): simd_gemm reads B with stride N
            // == tile_kv, and KQ rows below are indexed with the same stride.
            for (int tk = 0; tk < tile_kv; ++tk) {
                const float* kr = k_head + (size_t)(ic + tk) * kv_stride;
                for (int d = 0; d < head_dim; ++d) K_f32[(size_t)d * tile_kv + tk] = kr[d];
            }
            std::memset(KQ.data(), 0, (size_t)BLOCK_Q * BLOCK_KV * sizeof(float));
            forge::cpu::simd_gemm(KQ.data(), Q_q.data(), K_f32.data(), tile_q, head_dim, tile_kv);

            // scale + apply mask / causal
            for (int tq = 0; tq < tile_q; ++tq) {
                const int q_abs = kv_len - q_len + (i + tq);
                float* row = KQ.data() + (size_t)tq * tile_kv;
                for (int tk = 0; tk < tile_kv; ++tk) {
                    float s = row[tk] * scale;
                    if (mask_data) {
                        float m = mask_data[(size_t)(i + tq) * kv_len + (ic + tk)];
                        if (m < -1e20f) { row[tk] = -1e30f; continue; }
                        s += m;
                    } else if (causal) {
                        if ((ic + tk) > q_abs) { row[tk] = -1e30f; continue; }
                    }
                    row[tk] = s;
                }
            }

            // Online softmax over this KV tile
            bool any_valid = false;
            for (int tq = 0; tq < tile_q; ++tq) {
                float* row = KQ.data() + (size_t)tq * tile_kv;
                float tile_max = -1e30f;
                for (int tk = 0; tk < tile_kv; ++tk) tile_max = std::max(tile_max, row[tk]);
                if (tile_max == -1e30f) continue;  // fully masked row
                any_valid = true;

                const float Mold = M[tq];
                const float Mnew = std::max(Mold, tile_max);
                if (Mnew > Mold) {
                    float ms = std::exp(Mold - Mnew);
                    forge::cpu::scale_f32(VKQ32.data() + (size_t)tq * head_dim, head_dim, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                float row_sum = 0.0f;
                for (int tk = 0; tk < tile_kv; ++tk) {
                    float e = std::exp(row[tk] - Mnew);
                    row[tk] = e;
                    row_sum += e;
                }
                S[tq] += row_sum;
            }
            if (!any_valid) continue;

            // Pack V tile: V32[tk*head_dim + d] = v[kv_h][ic+tk][d]
            for (int tk = 0; tk < tile_kv; ++tk) {
                const float* vr = v_head + (size_t)(ic + tk) * kv_stride;
                std::memcpy(V32.data() + (size_t)tk * head_dim, vr, (size_t)head_dim * sizeof(float));
            }
            // VKQ32 += P * V  (P is stored back in KQ)
            forge::cpu::simd_gemm(VKQ32.data(), KQ.data(), V32.data(), tile_q, tile_kv, head_dim);
        }

        // Normalize and store
        for (int tq = 0; tq < tile_q; ++tq) {
            float inv = 1.0f / (S[tq] + 1e-30f);
            float* orow = out_head + (size_t)(i + tq) * q_stride;
            for (int d = 0; d < head_dim; ++d) orow[d] = VKQ32[(size_t)tq * head_dim + d] * inv;
        }
    }
}

// Prefill partial for one head over the KV sub-range [kv_lo, kv_hi).
// Writes, for each query row i:
//   partial_chunk[i*(head_dim+2) + 0] = m_i   (block max)
//   partial_chunk[i*(head_dim+2) + 1] = l_i   (sum exp(score - m_i))
//   partial_chunk[i*(head_dim+2) + 2..] = acc_i[d]  (sum exp(score-m_i)*v[d])
// When causal is set, query row i only attends to KV positions
//   j <= q_abs = kv_len - q_len + (i + tq)
// (positions j > q_abs are skipped, mirroring flash_attn_prefill_head's causal
// mask). When causal is false, every query row sees the full [kv_lo, kv_hi)
// sub-range (non-causal / cross-attention). flash_attn_prefill_head_splitkv
// splits KV across threads, each running this over its sub-range, then merges
// partials per (head, query row). Layout/online softmax mirrors
// flash_attn_prefill_head; only the KV range and the final store differ
// (partials instead of normalized output).
static void flash_attn_prefill_head_partial(
    const float* q_data, const float* k_data, const float* v_data, int q_len, int kv_len,
    int num_heads, int num_kv_heads, int head_dim, int h, int kv_lo, int kv_hi,
    float* partial_chunk, const float* mask_data, bool causal, float scale) {
    const int kv_group_size = num_heads / num_kv_heads;
    const int kv_h = h / kv_group_size;
    const int q_stride  = num_heads * head_dim;
    const int kv_stride = num_kv_heads * head_dim;
    const int row_elems = head_dim + 2;

    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;

    std::vector<float> Q_q(BLOCK_Q * head_dim);
    std::vector<float> KQ(BLOCK_Q * BLOCK_KV);
    std::vector<float> VKQ32(BLOCK_Q * head_dim);
    std::vector<float> V32(BLOCK_KV * head_dim);
    std::vector<float> K_f32(head_dim * BLOCK_KV);

    const float* q_head = q_data + (size_t)h * head_dim;
    const float* k_head = k_data + (size_t)kv_h * head_dim;
    const float* v_head = v_data + (size_t)kv_h * head_dim;

    float M[BLOCK_Q];
    float S[BLOCK_Q];

    for (int i = 0; i < q_len; i += BLOCK_Q) {
        const int tile_q = BLOCK_Q < (q_len - i) ? BLOCK_Q : (q_len - i);

        for (int tq = 0; tq < tile_q; ++tq) {
            const float* qr = q_head + (size_t)(i + tq) * q_stride;
            std::memcpy(Q_q.data() + (size_t)tq * head_dim, qr, (size_t)head_dim * sizeof(float));
        }
        for (int tq = 0; tq < tile_q; ++tq) { M[tq] = -1e30f; S[tq] = 0.0f; }
        std::memset(VKQ32.data(), 0, (size_t)tile_q * head_dim * sizeof(float));

        for (int ic = kv_lo; ic < kv_hi; ic += BLOCK_KV) {
            const int tile_kv = BLOCK_KV < (kv_hi - ic) ? BLOCK_KV : (kv_hi - ic);

            for (int tk = 0; tk < tile_kv; ++tk) {
                const float* kr = k_head + (size_t)(ic + tk) * kv_stride;
                for (int d = 0; d < head_dim; ++d) K_f32[(size_t)d * tile_kv + tk] = kr[d];
            }
            std::memset(KQ.data(), 0, (size_t)BLOCK_Q * BLOCK_KV * sizeof(float));
            forge::cpu::simd_gemm(KQ.data(), Q_q.data(), K_f32.data(), tile_q, head_dim, tile_kv);

            for (int tq = 0; tq < tile_q; ++tq) {
                const int q_abs = kv_len - q_len + (i + tq);
                float* row = KQ.data() + (size_t)tq * tile_kv;
                for (int tk = 0; tk < tile_kv; ++tk) {
                    float s = row[tk] * scale;
                    if (mask_data) {
                        float m = mask_data[(size_t)(i + tq) * kv_len + (ic + tk)];
                        if (m < -1e20f) { row[tk] = -1e30f; continue; }
                        s += m;
                    } else if (causal) {
                        if ((ic + tk) > q_abs) { row[tk] = -1e30f; continue; }
                    }
                    row[tk] = s;
                }
            }

            bool any_valid = false;
            for (int tq = 0; tq < tile_q; ++tq) {
                float* row = KQ.data() + (size_t)tq * tile_kv;
                float tile_max = -1e30f;
                for (int tk = 0; tk < tile_kv; ++tk) tile_max = std::max(tile_max, row[tk]);
                if (tile_max == -1e30f) continue;
                any_valid = true;
                const float Mold = M[tq];
                const float Mnew = std::max(Mold, tile_max);
                if (Mnew > Mold) {
                    float ms = std::exp(Mold - Mnew);
                    forge::cpu::scale_f32(VKQ32.data() + (size_t)tq * head_dim, head_dim, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;
                float row_sum = 0.0f;
                for (int tk = 0; tk < tile_kv; ++tk) {
                    float e = std::exp(row[tk] - Mnew);
                    row[tk] = e;
                    row_sum += e;
                }
                S[tq] += row_sum;
            }
            if (!any_valid) continue;

            for (int tk = 0; tk < tile_kv; ++tk) {
                const float* vr = v_head + (size_t)(ic + tk) * kv_stride;
                std::memcpy(V32.data() + (size_t)tk * head_dim, vr, (size_t)head_dim * sizeof(float));
            }
            forge::cpu::simd_gemm(VKQ32.data(), KQ.data(), V32.data(), tile_q, tile_kv, head_dim);
        }

        // Persist partials for this Q tile (absolute query rows).
        for (int tq = 0; tq < tile_q; ++tq) {
            const int64_t prow = (int64_t)(i + tq) * row_elems;
            partial_chunk[prow + 0] = M[tq];
            partial_chunk[prow + 1] = S[tq];
            const float* vk = VKQ32.data() + (size_t)tq * head_dim;
            for (int d = 0; d < head_dim; ++d) partial_chunk[prow + 2 + d] = vk[d];
        }
    }
}

// Prefill split-KV dispatcher (causal or non-causal): split the KV sequence
// across threads (each (head, kv-chunk) is one parallel unit, computing
// partials via flash_attn_prefill_head_partial), then merge per (head, query
// row) with the standard split-KV rule. num_kv_chunks is kept small
// (ceil(cores/heads)) so the partial buffer stays ~O(heads*chunks*q_len*head_dim);
// when heads already saturate the cores (or KV is tiny) we fall back to the
// original per-head tiled attention. For causal, blocks not yet visible to a
// query row are skipped inside the partial (see flash_attn_prefill_head_partial).
static void flash_attn_prefill_head_splitkv(const float* q_data, const float* k_data,
                                            const float* v_data, float* out_data, int q_len,
                                            int kv_len, int num_heads, int num_kv_heads,
                                            int head_dim, const float* mask_data, bool causal,
                                            float scale) {
    const int row_elems = head_dim + 2;
    const int nth = forge::cpu::get_num_threads();
    // Target ~3x the core count of parallel units. For causal, chunk 0 is visible
    // to every query row, so extra chunks shrink the dominant chunk's work and
    // keep all cores busy (avoids the load imbalance that hurts few-head configs
    // at ceil(nth/heads) chunks). For non-causal all chunks are equal work, so
    // finer chunks only improve utilization. The memory guard + min-chunk-size
    // below still cap this for large q_len / many heads.
    int num_kv_chunks = std::max(1, (nth * 3 + num_heads - 1) / num_heads);
    // Keep each chunk reasonably sized for the GEMM tiling (>= BLOCK_KV).
    while (num_kv_chunks > 1 && (kv_len / num_kv_chunks) < 64) --num_kv_chunks;
    // Memory guard (~256 MB) on the partial buffer.
    {
        const size_t budget_floats = (256ULL * 1024 * 1024) / sizeof(float);
        const size_t need = (size_t)num_heads * num_kv_chunks * q_len * row_elems;
        if (need > budget_floats) {
            int capped = (int)(budget_floats / ((size_t)num_heads * q_len * row_elems));
            num_kv_chunks = std::max(1, capped);
        }
    }
    if (num_kv_chunks <= 1) {
        // No extra parallelism vs per-head (heads already >= cores, or tiny
        // KV): use the original tiled attention, still parallel over heads.
        PERF_SCOPE("attention/qk_dot");
        forge::cpu::parallel_for_chunks(num_heads, 1, [&](int h0, int h1) {
            for (int h = h0; h < h1; ++h) {
                flash_attn_prefill_head(q_data, k_data, v_data, out_data, q_len, kv_len, num_heads,
                                        num_kv_heads, head_dim, h, mask_data, causal, scale);
            }
        });
        return;
    }

    std::vector<float> partials((size_t)num_heads * num_kv_chunks * q_len * row_elems, 0.0f);

    forge::cpu::parallel_for_chunks(num_heads * num_kv_chunks, 1, [&](int hc0, int hc1) {
        for (int hc = hc0; hc < hc1; ++hc) {
        const int h = hc / num_kv_chunks;
        const int c = hc % num_kv_chunks;
        const int chunk = (kv_len + num_kv_chunks - 1) / num_kv_chunks;
        const int kv_lo = c * chunk;
        const int kv_hi = std::min(kv_lo + chunk, kv_len);
        float* pchunk = partials.data() + (size_t)((h * num_kv_chunks + c) * q_len) * row_elems;
        flash_attn_prefill_head_partial(q_data, k_data, v_data, q_len, kv_len, num_heads,
                                        num_kv_heads, head_dim, h, kv_lo, kv_hi, pchunk,
                                        mask_data, causal, scale);
        }
    });

    // Merge partials per (head, query row).
    std::vector<float> acc(head_dim);
    for (int h = 0; h < num_heads; ++h) {
        float* out_head = out_data + (size_t)h * head_dim;
        for (int i = 0; i < q_len; ++i) {
            float m_g = -1e30f;
            for (int c = 0; c < num_kv_chunks; ++c) {
                const float* pc = partials.data() +
                    (size_t)((h * num_kv_chunks + c) * q_len + i) * row_elems;
                if (pc[0] > m_g) m_g = pc[0];
            }
            float* orow = out_head + (size_t)i * num_heads * head_dim;
            if (m_g == -1e30f) { std::memset(orow, 0, head_dim * sizeof(float)); continue; }
            std::memset(acc.data(), 0, head_dim * sizeof(float));
            float l_g = 0.0f;
            for (int c = 0; c < num_kv_chunks; ++c) {
                const float* pc = partials.data() +
                    (size_t)((h * num_kv_chunks + c) * q_len + i) * row_elems;
                const float m_c = pc[0];
                if (m_c == -1e30f) continue;
                const float w = std::exp(m_c - m_g);
                l_g += pc[1] * w;
                const float* ac = pc + 2;
                for (int d = 0; d < head_dim; ++d) acc[d] += ac[d] * w;
            }
            const float inv = 1.0f / (l_g + 1e-30f);
            for (int d = 0; d < head_dim; ++d) orow[d] = acc[d] * inv;
        }
    }
}

TensorPtr scaled_dot_product_attention(const TensorPtr& q, const TensorPtr& k, const TensorPtr& v,
                                       bool causal) {
    // 6-B-1 (P1): CPU attention reads Q/K/V as float* below -- BF16 has no
    // kernel, so reject loudly (FP16-on-CPU stays as-is: out of 6-B-1 scope).
    require_no_bf16("scaled_dot_product_attention/q", OpType::FLASH_ATTN_GQA, q->dtype());
    require_no_bf16("scaled_dot_product_attention/k", OpType::FLASH_ATTN_GQA, k->dtype());
    require_no_bf16("scaled_dot_product_attention/v", OpType::FLASH_ATTN_GQA, v->dtype());
    // 6-C-1: FP8 has no CPU attention kernel -- reject before the float* reads.
    require_no_fp8("scaled_dot_product_attention/q", OpType::FLASH_ATTN_GQA, q->dtype());
    require_no_fp8("scaled_dot_product_attention/k", OpType::FLASH_ATTN_GQA, k->dtype());
    require_no_fp8("scaled_dot_product_attention/v", OpType::FLASH_ATTN_GQA, v->dtype());
    int seq_len = static_cast<int>(q->shape()[0]);
    int num_heads = static_cast<int>(q->shape()[1]);
    int head_dim = static_cast<int>(q->shape()[2]);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto out = std::make_shared<Tensor>(DataType::FP32, q->shape(), q->device());

    if (q->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        auto q_2d =
            std::make_shared<Tensor>(q->view(std::vector<int64_t>{seq_len, num_heads * head_dim}));
        auto k_2d =
            std::make_shared<Tensor>(k->view(std::vector<int64_t>{seq_len, num_heads * head_dim}));
        auto v_2d =
            std::make_shared<Tensor>(v->view(std::vector<int64_t>{seq_len, num_heads * head_dim}));
        auto out_2d = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);

        cuda::launch_flash_attention(
            static_cast<const float*>(q_2d->data()), static_cast<const float*>(k_2d->data()),
            static_cast<const float*>(v_2d->data()), static_cast<float*>(out_2d->data()), seq_len,
            seq_len, num_heads, head_dim, nullptr, causal);

        auto out_3d = out_2d->view(std::vector<int64_t>{seq_len, num_heads, head_dim});
        *out = std::move(out_3d);
#endif
    } else {
        const float* q_data = static_cast<const float*>(q->data());
        const float* k_data = static_cast<const float*>(k->data());
        const float* v_data = static_cast<const float*>(v->data());

        std::vector<float> host_out(q->numel());

        auto attn_3d = [&](int h, int i) {
            std::vector<float> scores(seq_len);
            float max_val = -1e30f;
            for (int j = 0; j < seq_len; ++j) {
                if (causal && j > i) {
                    scores[j] = -1e30f;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_data[i * num_heads * head_dim + h * head_dim + d] *
                           k_data[j * num_heads * head_dim + h * head_dim + d];
                }
                scores[j] = dot * scale;
                max_val = std::max(max_val, scores[j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                scores[j] = std::exp(scores[j] - max_val);
                sum += scores[j];
            }
            float inv_sum = 1.0f / (sum + 1e-30f);
            for (int d = 0; d < head_dim; ++d) {
                float val = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    val += scores[j] * v_data[j * num_heads * head_dim + h * head_dim + d];
                }
                host_out[i * num_heads * head_dim + h * head_dim + d] = val * inv_sum;
            }
        };
        if ((int64_t)num_heads * seq_len <= 4) {
            for (int h = 0; h < num_heads; ++h)
                for (int i = 0; i < seq_len; ++i)
                    attn_3d(h, i);
        } else {
            forge::cpu::parallel_for2_dynamic(num_heads, seq_len, attn_3d);
        }

        std::memcpy(out->data(), host_out.data(), host_out.size() * sizeof(float));
    }

    return out;
}

TensorPtr scaled_dot_product_attention_2d(const TensorPtr& q, const TensorPtr& k,
                                          const TensorPtr& v, int seq_len, int num_heads,
                                          int head_dim, const TensorPtr& mask, bool causal) {
    return scaled_dot_product_attention_2d(q, k, v, seq_len, seq_len, num_heads, head_dim, mask, causal);
}

TensorPtr scaled_dot_product_attention_2d(const TensorPtr& q, const TensorPtr& k,
                                          const TensorPtr& v, int q_len, int kv_len, int num_heads,
                                          int head_dim, const TensorPtr& mask, bool causal) {
    // 6-B-1 (P1): reads Q/K/V (/mask) as float* below -- BF16 has no kernel.
    require_no_bf16("scaled_dot_product_attention_2d/q", OpType::FLASH_ATTN_GQA, q->dtype());
    require_no_bf16("scaled_dot_product_attention_2d/k", OpType::FLASH_ATTN_GQA, k->dtype());
    require_no_bf16("scaled_dot_product_attention_2d/v", OpType::FLASH_ATTN_GQA, v->dtype());
    if (mask) {
        require_no_bf16("scaled_dot_product_attention_2d/mask", OpType::FLASH_ATTN_GQA,
                        mask->dtype());
    }
    // 6-C-1: FP8 has no CPU attention kernel -- reject before the float* reads.
    require_no_fp8("scaled_dot_product_attention_2d/q", OpType::FLASH_ATTN_GQA, q->dtype());
    require_no_fp8("scaled_dot_product_attention_2d/k", OpType::FLASH_ATTN_GQA, k->dtype());
    require_no_fp8("scaled_dot_product_attention_2d/v", OpType::FLASH_ATTN_GQA, v->dtype());
    if (mask) {
        require_no_fp8("scaled_dot_product_attention_2d/mask", OpType::FLASH_ATTN_GQA,
                       mask->dtype());
    }
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{q_len, num_heads * head_dim}, q->device());

    if (q->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        // If mask is on CPU, copy it to CUDA first
        const float* mask_ptr = nullptr;
        auto mask_cuda = mask;
        if (mask && mask->device() == DeviceType::CPU) {
            mask_cuda = std::make_shared<Tensor>(DataType::FP32, mask->shape(), DeviceType::CUDA);
            mask_cuda->copy_from(*mask);
        }
        if (mask_cuda) {
            mask_ptr = static_cast<const float*>(mask_cuda->data());
        }
        cuda::launch_flash_attention(
            static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
            static_cast<const float*>(v->data()), static_cast<float*>(out->data()), q_len, kv_len,
            num_heads, head_dim, mask_ptr, causal);
        return out;
#endif
    }
    const float* q_data = static_cast<const float*>(q->data());
    const float* k_data = static_cast<const float*>(k->data());
    const float* v_data = static_cast<const float*>(v->data());
    const float* mask_data = mask ? static_cast<const float*>(mask->data()) : nullptr;

    std::vector<float> host_out(out->numel());

    // For decode (q_len=1), causal mask is irrelevant since all KV positions
    // are visible. Use online softmax to avoid allocating a scores vector
    // per head and reduce memory traffic.
    if (q_len == 1) {
        // Decode path (split-KV / FlashDecoding): split the KV sequence into
        // KV_CHUNK-sized chunks so the KV dimension is a parallel axis. Each
        // (head, kv-chunk) computes a partial online-softmax state, merged
        // per head in a cheap combine pass (see GQA path for details).
        PERF_SCOPE("attention/qk_dot");

        constexpr int KV_CHUNK = 256;
        const int num_kv_chunks = (kv_len + KV_CHUNK - 1) / KV_CHUNK;

        if (num_kv_chunks <= 1) {
            // Single-chunk fallback: serial per-head online softmax.
            forge::cpu::parallel_for_chunks(num_heads, 1, [&](int h0, int h1) {
                for (int h = h0; h < h1; ++h) {
                const float* q_row = q_data + h * head_dim;
                float* out_row = host_out.data() + h * head_dim;
                float max_val = -1e30f, sum_exp = 0.0f;
                std::memset(out_row, 0, head_dim * sizeof(float));
                for (int j = 0; j < kv_len; ++j) {
                    const float* k_row = k_data + (int64_t)j * num_heads * head_dim + h * head_dim;
                    float score = forge::cpu::dot_f32(q_row, k_row, head_dim) * scale;
                    if (mask_data) {
                        float m = mask_data[j];
                        if (m < -1e20f) continue;  // skip masked positions
                        score += m;
                    }
                    float new_max = std::max(max_val, score);
                    float rescale = std::exp(max_val - new_max);
                    if (new_max > max_val) {
                        sum_exp *= rescale;
                        forge::cpu::scale_f32(out_row, head_dim, rescale);
                    }
                    float exp_score = std::exp(score - new_max);
                    sum_exp += exp_score;
                    const float* v_row = v_data + (int64_t)j * num_heads * head_dim + h * head_dim;
                    forge::cpu::fmadd_f32(out_row, v_row, head_dim, exp_score);
                    max_val = new_max;
                }
                forge::cpu::scale_f32(out_row, head_dim, 1.0f / (sum_exp + 1e-30f));
                }
            });
        } else {
            const int64_t pstride = (int64_t)head_dim + 2;  // [m_c, l_c, acc_c[head_dim]]
            std::vector<float> partials((int64_t)num_heads * num_kv_chunks * pstride, 0.0f);

            forge::cpu::parallel_for_chunks(num_heads * num_kv_chunks, 1, [&](int hc0, int hc1) {
                for (int hc = hc0; hc < hc1; ++hc) {
                const int h = hc / num_kv_chunks;
                const int c = hc % num_kv_chunks;
                const float* q_row = q_data + h * head_dim;
                const int j0 = c * KV_CHUNK;
                const int j1 = std::min(j0 + KV_CHUNK, kv_len);
                const int64_t base = (int64_t)hc * pstride;

                float m_c = -1e30f;
                float l_c = 0.0f;
                float* acc = partials.data() + base + 2;
                std::memset(acc, 0, head_dim * sizeof(float));

                for (int j = j0; j < j1; ++j) {
                    const float* k_row = k_data + (int64_t)j * num_heads * head_dim + h * head_dim;
                    float dot = forge::cpu::dot_f32(q_row, k_row, head_dim);
                    float score = dot * scale;

                    if (mask_data) {
                        float m = mask_data[j];
                        if (m < -1e20f) continue;  // skip masked positions
                        score += m;
                    }

                    float new_max = std::max(m_c, score);
                    float rescale = std::exp(m_c - new_max);
                    if (new_max > m_c) {
                        l_c *= rescale;
                        forge::cpu::scale_f32(acc, head_dim, rescale);
                    }
                    float exp_score = std::exp(score - new_max);
                    l_c += exp_score;

                    const float* v_row = v_data + (int64_t)j * num_heads * head_dim + h * head_dim;
                    forge::cpu::fmadd_f32(acc, v_row, head_dim, exp_score);
                    m_c = new_max;
                }
                partials[base] = m_c;
                partials[base + 1] = l_c;
                }
            });

            // Combine partials per head
            for (int h = 0; h < num_heads; ++h) {
                float* out_row = host_out.data() + h * head_dim;
                std::memset(out_row, 0, head_dim * sizeof(float));

                float m_g = -1e30f;
                for (int c = 0; c < num_kv_chunks; ++c) {
                    const int64_t base = ((int64_t)h * num_kv_chunks + c) * pstride;
                    if (partials[base] > m_g) m_g = partials[base];
                }
                float l_g = 0.0f;
                for (int c = 0; c < num_kv_chunks; ++c) {
                    const int64_t base = ((int64_t)h * num_kv_chunks + c) * pstride;
                    const float m_c = partials[base];
                    if (m_c == -1e30f) continue;  // fully masked chunk
                    const float w = std::exp(m_c - m_g);
                    l_g += partials[base + 1] * w;
                    const float* acc = partials.data() + base + 2;
                    forge::cpu::fmadd_f32(out_row, acc, head_dim, w);
                }
                float inv = 1.0f / (l_g + 1e-30f);
                forge::cpu::scale_f32(out_row, head_dim, inv);
            }
        }
    } else {
        // General path: prefill / causal attention.
        // Both causal and non-causal use split-KV across KV chunks for extra
        // parallelism (causal masking is applied inside the per-chunk partial;
        // when heads already saturate the cores the dispatcher falls back to
        // the original per-head tiled attention).
        PERF_SCOPE("attention/qk_dot");
        flash_attn_prefill_head_splitkv(q_data, k_data, v_data, host_out.data(), q_len, kv_len,
                                        num_heads, num_heads, head_dim, mask_data, causal, scale);
    }

    std::memcpy(out->data(), host_out.data(), host_out.size() * sizeof(float));

    return out;
}

TensorPtr scaled_dot_product_attention_2d_masked(const TensorPtr& q, const TensorPtr& k,
                                                 const TensorPtr& v, int seq_len, int num_heads,
                                                 int head_dim, const TensorPtr& mask) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // For masked attention, we must use CPU path (flash attention doesn't support custom masks)
    auto q_cpu = q;
    auto k_cpu = k;
    auto v_cpu = v;
    auto mask_cpu = mask;
    if (q->device() == DeviceType::CUDA) {
        q_cpu = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
        k_cpu = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
        v_cpu = std::make_shared<Tensor>(DataType::FP32, v->shape(), DeviceType::CPU);
        mask_cpu = std::make_shared<Tensor>(DataType::FP32, mask->shape(), DeviceType::CPU);
        q_cpu->copy_from(*q);
        k_cpu->copy_from(*k);
        v_cpu->copy_from(*v);
        mask_cpu->copy_from(*mask);
    }

    const float* q_data = static_cast<const float*>(q_cpu->data());
    const float* k_data = static_cast<const float*>(k_cpu->data());
    const float* v_data = static_cast<const float*>(v_cpu->data());
    const float* mask_data = static_cast<const float*>(mask_cpu->data());

    auto out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CPU);
    float* out_data = static_cast<float*>(out->data());

    auto attn_masked = [&](int h, int i) {
        std::vector<float> scores(seq_len);
        float max_val = -1e30f;
        for (int j = 0; j < seq_len; ++j) {
            float m = mask_data[i * seq_len + j];
            if (m < -1e20f) {
                scores[j] = -1e30f;
                continue;
            }
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_data[i * num_heads * head_dim + h * head_dim + d] *
                       k_data[j * num_heads * head_dim + h * head_dim + d];
            }
            scores[j] = dot * scale + m;
            max_val = std::max(max_val, scores[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = std::exp(scores[j] - max_val);
            sum += scores[j];
        }
        float inv_sum = 1.0f / (sum + 1e-30f);
        for (int d = 0; d < head_dim; ++d) {
            float val = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                val += scores[j] * v_data[j * num_heads * head_dim + h * head_dim + d];
            }
            out_data[i * num_heads * head_dim + h * head_dim + d] = val * inv_sum;
        }
    };
    if ((int64_t)num_heads * seq_len <= 4) {
        for (int h = 0; h < num_heads; ++h)
            for (int i = 0; i < seq_len; ++i)
                attn_masked(h, i);
    } else {
        forge::cpu::parallel_for2_dynamic(num_heads, seq_len, attn_masked);
    }

    return out;
}

TensorPtr scaled_dot_product_attention_2d_gqa(const TensorPtr& q, const TensorPtr& k,
                                              const TensorPtr& v, int q_len, int kv_len,
                                              int num_heads, int num_kv_heads, int head_dim,
                                              const TensorPtr& mask, bool causal) {
    // 6-B-1 (P1): reads Q/K/V (/mask) as float* below -- BF16 has no kernel.
    require_no_bf16("scaled_dot_product_attention_2d_gqa/q", OpType::FLASH_ATTN_GQA, q->dtype());
    require_no_bf16("scaled_dot_product_attention_2d_gqa/k", OpType::FLASH_ATTN_GQA, k->dtype());
    require_no_bf16("scaled_dot_product_attention_2d_gqa/v", OpType::FLASH_ATTN_GQA, v->dtype());
    if (mask) {
        require_no_bf16("scaled_dot_product_attention_2d_gqa/mask", OpType::FLASH_ATTN_GQA,
                        mask->dtype());
    }
    // 6-C-1: FP8 has no CPU attention kernel -- reject before the float* reads.
    require_no_fp8("scaled_dot_product_attention_2d_gqa/q", OpType::FLASH_ATTN_GQA, q->dtype());
    require_no_fp8("scaled_dot_product_attention_2d_gqa/k", OpType::FLASH_ATTN_GQA, k->dtype());
    require_no_fp8("scaled_dot_product_attention_2d_gqa/v", OpType::FLASH_ATTN_GQA, v->dtype());
    if (mask) {
        require_no_fp8("scaled_dot_product_attention_2d_gqa/mask", OpType::FLASH_ATTN_GQA,
                       mask->dtype());
    }
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{q_len, num_heads * head_dim}, q->device());

    // CUDA path: use GQA flash attention kernels
    if (q->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        const float* q_data = static_cast<const float*>(q->data());
        const float* k_data = static_cast<const float*>(k->data());
        const float* v_data = static_cast<const float*>(v->data());
        float* out_data = static_cast<float*>(out->data());

        // If mask is on CPU, copy it to CUDA first
        const float* mask_ptr = nullptr;
        auto mask_cuda = mask;
        if (mask && mask->device() == DeviceType::CPU) {
            mask_cuda = std::make_shared<Tensor>(DataType::FP32, mask->shape(), DeviceType::CUDA);
            mask_cuda->copy_from(*mask);
        }
        if (mask_cuda) {
            mask_ptr = static_cast<const float*>(mask_cuda->data());
        }

        bool cuda_decode_supported = (head_dim == 64 || head_dim == 96 || head_dim == 128 ||
                                       head_dim == 256 || head_dim == 512);

        if (q_len == 1 && cuda_decode_supported) {
            // Decode path (supported head_dim)
            cuda::launch_flash_attention_gqa_decode(q_data, k_data, v_data, out_data,
                                                     kv_len, num_heads, num_kv_heads, head_dim,
                                                     mask_ptr);
            return out;
        } else if (q_len > 1) {
            // Prefill path
            cuda::launch_flash_attention_gqa(q_data, k_data, v_data, out_data,
                                              q_len, kv_len, num_heads, num_kv_heads, head_dim,
                                              mask_ptr, causal);
            return out;
        }
        // Fall through to CPU path for unsupported head_dim in decode mode
#endif
    }

    // CPU path
    const float* q_data = static_cast<const float*>(q->data());
    const float* k_data = static_cast<const float*>(k->data());
    const float* v_data = static_cast<const float*>(v->data());
    const float* mask_data = mask ? static_cast<const float*>(mask->data()) : nullptr;

    std::vector<float> host_out(out->numel());

    int kv_group_size = num_heads / num_kv_heads;  // e.g., 32/4 = 8

    if (q_len == 1) {
        // Decode path (split-KV / FlashDecoding): split the KV sequence into
        // KV_CHUNK-sized chunks so the KV dimension becomes a parallel axis
        // (each (head, kv-chunk) is one parallel unit). Each chunk computes a
        // partial online-softmax state (m_c, l_c, acc_c[d]); a cheap combine
        // pass merges partials per head using the standard split-KV rule:
        //   m = max_c m_c;  l = sum_c l_c*exp(m_c-m);
        //   acc = sum_c acc_c*exp(m_c-m);  out = acc / l.
        // When KV fits in a single chunk, splitting adds only buffer-alloc +
        // combine overhead with no extra parallelism, so we fall back to the
        // original serial per-head (still OpenMP-parallel over heads) loop.
        PERF_SCOPE("attention/qk_dot");

        constexpr int KV_CHUNK = 256;
        const int num_kv_chunks = (kv_len + KV_CHUNK - 1) / KV_CHUNK;

        if (num_kv_chunks <= 1) {
            // Single-chunk fallback: serial per-head online softmax.
            forge::cpu::parallel_for_chunks(num_heads, 1, [&](int h0, int h1) {
                for (int h = h0; h < h1; ++h) {
                const int kv_h = h / kv_group_size;
                const float* q_row = q_data + h * head_dim;
                float* out_row = host_out.data() + h * head_dim;
                float max_val = -1e30f, sum_exp = 0.0f;
                std::memset(out_row, 0, head_dim * sizeof(float));
                for (int j = 0; j < kv_len; ++j) {
                    const float* k_row =
                        k_data + (int64_t)j * num_kv_heads * head_dim + kv_h * head_dim;
                    float score = forge::cpu::dot_f32(q_row, k_row, head_dim) * scale;
                    if (mask_data) {
                        float m = mask_data[j];
                        if (m < -1e20f) continue;  // skip masked positions
                        score += m;
                    }
                    float new_max = std::max(max_val, score);
                    float rescale = std::exp(max_val - new_max);
                    if (new_max > max_val) {
                        sum_exp *= rescale;
                        forge::cpu::scale_f32(out_row, head_dim, rescale);
                    }
                    float exp_score = std::exp(score - new_max);
                    sum_exp += exp_score;
                    const float* v_row =
                        v_data + (int64_t)j * num_kv_heads * head_dim + kv_h * head_dim;
                    forge::cpu::fmadd_f32(out_row, v_row, head_dim, exp_score);
                    max_val = new_max;
                }
                forge::cpu::scale_f32(out_row, head_dim, 1.0f / (sum_exp + 1e-30f));
                }
            });
        } else {
            const int64_t pstride = (int64_t)head_dim + 2;  // [m_c, l_c, acc_c[head_dim]]
            std::vector<float> partials((int64_t)num_heads * num_kv_chunks * pstride, 0.0f);

            forge::cpu::parallel_for_chunks(num_heads * num_kv_chunks, 1, [&](int hc0, int hc1) {
                for (int hc = hc0; hc < hc1; ++hc) {
                const int h = hc / num_kv_chunks;
                const int c = hc % num_kv_chunks;
                const int kv_h = h / kv_group_size;
                const float* q_row = q_data + h * head_dim;
                const int j0 = c * KV_CHUNK;
                const int j1 = std::min(j0 + KV_CHUNK, kv_len);
                const int64_t base = (int64_t)hc * pstride;

                float m_c = -1e30f;
                float l_c = 0.0f;
                float* acc = partials.data() + base + 2;
                std::memset(acc, 0, head_dim * sizeof(float));

                for (int j = j0; j < j1; ++j) {
                    const float* k_row =
                        k_data + (int64_t)j * num_kv_heads * head_dim + kv_h * head_dim;
                    float dot = forge::cpu::dot_f32(q_row, k_row, head_dim);
                    float score = dot * scale;

                    if (mask_data) {
                        float m = mask_data[j];
                        if (m < -1e20f) continue;  // skip masked positions
                        score += m;
                    }

                    float new_max = std::max(m_c, score);
                    float rescale = std::exp(m_c - new_max);
                    if (new_max > m_c) {
                        l_c *= rescale;
                        forge::cpu::scale_f32(acc, head_dim, rescale);
                    }
                    float exp_score = std::exp(score - new_max);
                    l_c += exp_score;

                    const float* v_row =
                        v_data + (int64_t)j * num_kv_heads * head_dim + kv_h * head_dim;
                    forge::cpu::fmadd_f32(acc, v_row, head_dim, exp_score);
                    m_c = new_max;
                }
                partials[base] = m_c;
                partials[base + 1] = l_c;
                }
            });

            // Combine partials per head
            for (int h = 0; h < num_heads; ++h) {
                float* out_row = host_out.data() + h * head_dim;
                std::memset(out_row, 0, head_dim * sizeof(float));

                float m_g = -1e30f;
                for (int c = 0; c < num_kv_chunks; ++c) {
                    const int64_t base = ((int64_t)h * num_kv_chunks + c) * pstride;
                    if (partials[base] > m_g) m_g = partials[base];
                }
                float l_g = 0.0f;
                for (int c = 0; c < num_kv_chunks; ++c) {
                    const int64_t base = ((int64_t)h * num_kv_chunks + c) * pstride;
                    const float m_c = partials[base];
                    if (m_c == -1e30f) continue;  // fully masked chunk
                    const float w = std::exp(m_c - m_g);
                    l_g += partials[base + 1] * w;
                    const float* acc = partials.data() + base + 2;
                    forge::cpu::fmadd_f32(out_row, acc, head_dim, w);
                }
                float inv = 1.0f / (l_g + 1e-30f);
                forge::cpu::scale_f32(out_row, head_dim, inv);
            }
        }
    } else {
        // Prefill path with GQA.
        // Both causal and non-causal use split-KV across KV chunks for extra
        // parallelism (causal masking inside the per-chunk partial; fallback to
        // per-head tiled attention when heads saturate the cores).
        PERF_SCOPE("attention/qk_dot");
        flash_attn_prefill_head_splitkv(q_data, k_data, v_data, host_out.data(), q_len, kv_len,
                                        num_heads, num_kv_heads, head_dim, mask_data, causal,
                                        scale);
    }

    std::memcpy(out->data(), host_out.data(), host_out.size() * sizeof(float));

    return out;
}

}  // namespace ops
}  // namespace forge
