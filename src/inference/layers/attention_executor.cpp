#include "cpu/common/thread_pool.h"
#include "forge/inference/layers/attention_executor.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "cpu/simd.h"
#include "forge/attention_selector.h"
#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/kv_cache_dtype.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"


namespace forge {

AttentionExecutor::QKVResult AttentionExecutor::project_qkv(const TensorPtr& x,
                                                            const LayerWeights& lw, bool has_bias,
                                                            DeviceType dev, int seq_len) {
    auto wq = lw.wq();
    auto wk = lw.wk();
    auto wv = lw.wv();
    auto bq = has_bias ? lw.bq() : nullptr;
    auto bk = has_bias ? lw.bk() : nullptr;
    auto bv = has_bias ? lw.bv() : nullptr;

    QKVResult result;

    // CUDA Q4_0 fused path
    if (dev == DeviceType::CUDA && seq_len == 1 && wq->dtype() == DataType::Q4_0 &&
        wk->dtype() == DataType::Q4_0 && wv->dtype() == DataType::Q4_0) {
#ifdef USE_CUDA
        int K_proj = static_cast<int>(wq->shape()[1]);
        int N_q = static_cast<int>(wq->shape()[0]);
        int N_k = static_cast<int>(wk->shape()[0]);
        int N_v = static_cast<int>(wv->shape()[0]);

        result.q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q},
                                            DeviceType::CUDA);
        result.k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k},
                                            DeviceType::CUDA);
        result.v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v},
                                            DeviceType::CUDA);

        cuda::launch_qkv_fused_q4_0(
            static_cast<const float*>(x->data()), wq->data(), N_q, wk->data(), N_k, wv->data(), N_v,
            static_cast<float*>(result.q->data()), static_cast<float*>(result.k->data()),
            static_cast<float*>(result.v->data()), K_proj);

        // Add bias terms (fused kernel only computes GEMV, not bias)
        if (bq && bq->numel() > 0) {
            cuda::launch_add_bias(static_cast<const float*>(result.q->data()),
                                  static_cast<const float*>(bq->data()),
                                  static_cast<float*>(result.q->data()), N_q);
        }
        if (bk && bk->numel() > 0) {
            cuda::launch_add_bias(static_cast<const float*>(result.k->data()),
                                  static_cast<const float*>(bk->data()),
                                  static_cast<float*>(result.k->data()), N_k);
        }
        if (bv && bv->numel() > 0) {
            cuda::launch_add_bias(static_cast<const float*>(result.v->data()),
                                  static_cast<const float*>(bv->data()),
                                  static_cast<float*>(result.v->data()), N_v);
        }
#else
        (void)0;
#endif
    }
    // CUDA Q4_K fused path
    else if (dev == DeviceType::CUDA && seq_len == 1 && wq->dtype() == DataType::Q4_K &&
             wk->dtype() == DataType::Q4_K && wv->dtype() == DataType::Q4_K) {
#ifdef USE_CUDA
        int K_proj = static_cast<int>(wq->shape()[1]);
        int N_q = static_cast<int>(wq->shape()[0]);
        int N_k = static_cast<int>(wk->shape()[0]);
        int N_v = static_cast<int>(wv->shape()[0]);

        result.q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q},
                                            DeviceType::CUDA);
        result.k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k},
                                            DeviceType::CUDA);
        result.v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v},
                                            DeviceType::CUDA);

        cuda::launch_qkv_fused_q4_k(
            static_cast<const float*>(x->data()), wq->data(), N_q, wk->data(), N_k, wv->data(), N_v,
            static_cast<float*>(result.q->data()), static_cast<float*>(result.k->data()),
            static_cast<float*>(result.v->data()), K_proj);

        if (bq && bq->numel() > 0) {
            cuda::launch_add_bias(static_cast<const float*>(result.q->data()),
                                  static_cast<const float*>(bq->data()),
                                  static_cast<float*>(result.q->data()), N_q);
        }
        if (bk && bk->numel() > 0) {
            cuda::launch_add_bias(static_cast<const float*>(result.k->data()),
                                  static_cast<const float*>(bk->data()),
                                  static_cast<float*>(result.k->data()), N_k);
        }
        if (bv && bv->numel() > 0) {
            cuda::launch_add_bias(static_cast<const float*>(result.v->data()),
                                  static_cast<const float*>(bv->data()),
                                  static_cast<float*>(result.v->data()), N_v);
        }
#endif
    }
    // CUDA Q5_K fused path
    else if (dev == DeviceType::CUDA && seq_len == 1 && wq->dtype() == DataType::Q5_K &&
             wk->dtype() == DataType::Q5_K && wv->dtype() == DataType::Q5_K) {
#ifdef USE_CUDA
        int K_proj = static_cast<int>(wq->shape()[1]);
        int N_q = static_cast<int>(wq->shape()[0]);
        int N_k = static_cast<int>(wk->shape()[0]);
        int N_v = static_cast<int>(wv->shape()[0]);

        result.q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q},
                                            DeviceType::CUDA);
        result.k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k},
                                            DeviceType::CUDA);
        result.v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v},
                                            DeviceType::CUDA);

        cuda::launch_qkv_fused_q5_k(
            static_cast<const float*>(x->data()), wq->data(), N_q, wk->data(), N_k, wv->data(), N_v,
            static_cast<float*>(result.q->data()), static_cast<float*>(result.k->data()),
            static_cast<float*>(result.v->data()), K_proj);

        if (bq && bq->numel() > 0) {
            cuda::launch_add_bias(static_cast<const float*>(result.q->data()),
                                  static_cast<const float*>(bq->data()),
                                  static_cast<float*>(result.q->data()), N_q);
        }
        if (bk && bk->numel() > 0) {
            cuda::launch_add_bias(static_cast<const float*>(result.k->data()),
                                  static_cast<const float*>(bk->data()),
                                  static_cast<float*>(result.k->data()), N_k);
        }
        if (bv && bv->numel() > 0) {
            cuda::launch_add_bias(static_cast<const float*>(result.v->data()),
                                  static_cast<const float*>(bv->data()),
                                  static_cast<float*>(result.v->data()), N_v);
        }
#else
        (void)0;
#endif
    }
    // CPU Q4_0 fused path
    else if (dev == DeviceType::CPU && seq_len == 1 && wq->dtype() == DataType::Q4_0 &&
             wk->dtype() == DataType::Q4_0 && wv->dtype() == DataType::Q4_0) {
        int N_q = static_cast<int>(wq->shape()[0]);
        int N_k = static_cast<int>(wk->shape()[0]);
        int N_v = static_cast<int>(wv->shape()[0]);
        auto qkv = ops::matmul_transB_fused_qkv_q4_0(x, wq, wk, wv);
        result.q =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q}, DeviceType::CPU);
        result.k =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k}, DeviceType::CPU);
        result.v =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v}, DeviceType::CPU);
        float* src = static_cast<float*>(qkv->data());
        std::memcpy(result.q->data(), src, N_q * sizeof(float));
        std::memcpy(result.k->data(), src + N_q, N_k * sizeof(float));
        std::memcpy(result.v->data(), src + N_q + N_k, N_v * sizeof(float));
        if (bq && bq->numel() > 0) {
            float* qd = static_cast<float*>(result.q->data());
            const float* bd = static_cast<const float*>(bq->data());
            for (int i = 0; i < N_q; ++i)
                qd[i] += bd[i];
        }
        if (bk && bk->numel() > 0) {
            float* kd = static_cast<float*>(result.k->data());
            const float* bd = static_cast<const float*>(bk->data());
            for (int i = 0; i < N_k; ++i)
                kd[i] += bd[i];
        }
        if (bv && bv->numel() > 0) {
            float* vd = static_cast<float*>(result.v->data());
            const float* bd = static_cast<const float*>(bv->data());
            for (int i = 0; i < N_v; ++i)
                vd[i] += bd[i];
        }
    }
    // CPU Q4_K fused path
    else if (dev == DeviceType::CPU && seq_len == 1 && wq->dtype() == DataType::Q4_K &&
             wk->dtype() == DataType::Q4_K && wv->dtype() == DataType::Q4_K) {
        int N_q = static_cast<int>(wq->shape()[0]);
        int N_k = static_cast<int>(wk->shape()[0]);
        int N_v = static_cast<int>(wv->shape()[0]);
        auto qkv = ops::matmul_transB_fused_qkv_q4_k(x, wq, wk, wv);
        result.q =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q}, DeviceType::CPU);
        result.k =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k}, DeviceType::CPU);
        result.v =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v}, DeviceType::CPU);
        float* src = static_cast<float*>(qkv->data());
        std::memcpy(result.q->data(), src, N_q * sizeof(float));
        std::memcpy(result.k->data(), src + N_q, N_k * sizeof(float));
        std::memcpy(result.v->data(), src + N_q + N_k, N_v * sizeof(float));
        if (bq && bq->numel() > 0) {
            float* qd = static_cast<float*>(result.q->data());
            const float* bd = static_cast<const float*>(bq->data());
            for (int i = 0; i < N_q; ++i)
                qd[i] += bd[i];
        }
        if (bk && bk->numel() > 0) {
            float* kd = static_cast<float*>(result.k->data());
            const float* bd = static_cast<const float*>(bk->data());
            for (int i = 0; i < N_k; ++i)
                kd[i] += bd[i];
        }
        if (bv && bv->numel() > 0) {
            float* vd = static_cast<float*>(result.v->data());
            const float* bd = static_cast<const float*>(bv->data());
            for (int i = 0; i < N_v; ++i)
                vd[i] += bd[i];
        }
    }
    // CPU Q5_K fused path
    else if (dev == DeviceType::CPU && seq_len == 1 && wq->dtype() == DataType::Q5_K &&
             wk->dtype() == DataType::Q5_K && wv->dtype() == DataType::Q5_K) {
        int N_q = static_cast<int>(wq->shape()[0]);
        int N_k = static_cast<int>(wk->shape()[0]);
        int N_v = static_cast<int>(wv->shape()[0]);
        auto qkv = ops::matmul_transB_fused_qkv_q5_k(x, wq, wk, wv);
        result.q =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q}, DeviceType::CPU);
        result.k =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k}, DeviceType::CPU);
        result.v =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v}, DeviceType::CPU);
        float* src = static_cast<float*>(qkv->data());
        std::memcpy(result.q->data(), src, N_q * sizeof(float));
        std::memcpy(result.k->data(), src + N_q, N_k * sizeof(float));
        std::memcpy(result.v->data(), src + N_q + N_k, N_v * sizeof(float));
        if (bq && bq->numel() > 0) {
            float* qd = static_cast<float*>(result.q->data());
            const float* bd = static_cast<const float*>(bq->data());
            for (int i = 0; i < N_q; ++i)
                qd[i] += bd[i];
        }
        if (bk && bk->numel() > 0) {
            float* kd = static_cast<float*>(result.k->data());
            const float* bd = static_cast<const float*>(bk->data());
            for (int i = 0; i < N_k; ++i)
                kd[i] += bd[i];
        }
        if (bv && bv->numel() > 0) {
            float* vd = static_cast<float*>(result.v->data());
            const float* bd = static_cast<const float*>(bv->data());
            for (int i = 0; i < N_v; ++i)
                vd[i] += bd[i];
        }
    }
    // CPU Q2_K fused path
    else if (dev == DeviceType::CPU && seq_len == 1 && wq->dtype() == DataType::Q2_K &&
             wk->dtype() == DataType::Q2_K && wv->dtype() == DataType::Q2_K) {
        int N_q = static_cast<int>(wq->shape()[0]);
        int N_k = static_cast<int>(wk->shape()[0]);
        int N_v = static_cast<int>(wv->shape()[0]);
        auto qkv = ops::matmul_transB_fused_qkv_q2_k(x, wq, wk, wv);
        result.q =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q}, DeviceType::CPU);
        result.k =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k}, DeviceType::CPU);
        result.v =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v}, DeviceType::CPU);
        float* src = static_cast<float*>(qkv->data());
        std::memcpy(result.q->data(), src, N_q * sizeof(float));
        std::memcpy(result.k->data(), src + N_q, N_k * sizeof(float));
        std::memcpy(result.v->data(), src + N_q + N_k, N_v * sizeof(float));
        if (bq && bq->numel() > 0) {
            float* qd = static_cast<float*>(result.q->data());
            const float* bd = static_cast<const float*>(bq->data());
            for (int i = 0; i < N_q; ++i)
                qd[i] += bd[i];
        }
        if (bk && bk->numel() > 0) {
            float* kd = static_cast<float*>(result.k->data());
            const float* bd = static_cast<const float*>(bk->data());
            for (int i = 0; i < N_k; ++i)
                kd[i] += bd[i];
        }
        if (bv && bv->numel() > 0) {
            float* vd = static_cast<float*>(result.v->data());
            const float* bd = static_cast<const float*>(bv->data());
            for (int i = 0; i < N_v; ++i)
                vd[i] += bd[i];
        }
    }
    // CPU mixed-precision fused Q3_K+Q4_K QKV: Q,K are Q3_K, V is Q4_K
    // Shares Q8_K quantization of activation across all three projections.
    else if (dev == DeviceType::CPU && seq_len == 1 && wq->dtype() == DataType::Q3_K &&
             wk->dtype() == DataType::Q3_K && wv->dtype() == DataType::Q4_K) {
        int N_q = static_cast<int>(wq->shape()[0]);
        int N_k = static_cast<int>(wk->shape()[0]);
        int N_v = static_cast<int>(wv->shape()[0]);
        auto qkv = ops::matmul_transB_fused_qkv_q3_k_q4_k(x, wq, wk, wv);
        result.q =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q}, DeviceType::CPU);
        result.k =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k}, DeviceType::CPU);
        result.v =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v}, DeviceType::CPU);
        float* src = static_cast<float*>(qkv->data());
        std::memcpy(result.q->data(), src, N_q * sizeof(float));
        std::memcpy(result.k->data(), src + N_q, N_k * sizeof(float));
        std::memcpy(result.v->data(), src + N_q + N_k, N_v * sizeof(float));
        if (bq && bq->numel() > 0) {
            float* qd = static_cast<float*>(result.q->data());
            const float* bd = static_cast<const float*>(bq->data());
            for (int i = 0; i < N_q; ++i)
                qd[i] += bd[i];
        }
        if (bk && bk->numel() > 0) {
            float* kd = static_cast<float*>(result.k->data());
            const float* bd = static_cast<const float*>(bk->data());
            for (int i = 0; i < N_k; ++i)
                kd[i] += bd[i];
        }
        if (bv && bv->numel() > 0) {
            float* vd = static_cast<float*>(result.v->data());
            const float* bd = static_cast<const float*>(bv->data());
            for (int i = 0; i < N_v; ++i)
                vd[i] += bd[i];
        }
    }
    // CPU Q3_K fused Q+K path (wq=Q3_K, wk=Q3_K; wv is Q5_K or other)
    // Shares Q8_K quantization of activation across Q and K projections.
    else if (dev == DeviceType::CPU && seq_len == 1 && wq->dtype() == DataType::Q3_K &&
             wk->dtype() == DataType::Q3_K) {
        int N_q = static_cast<int>(wq->shape()[0]);
        int N_k = static_cast<int>(wk->shape()[0]);
        auto qk = ops::matmul_transB_fused_qk_q3_k(x, wq, wk);
        result.q =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q}, DeviceType::CPU);
        result.k =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k}, DeviceType::CPU);
        float* src = static_cast<float*>(qk->data());
        std::memcpy(result.q->data(), src, N_q * sizeof(float));
        std::memcpy(result.k->data(), src + N_q, N_k * sizeof(float));
        result.v = ops::matmul_transB(x, wv, bv);
        if (bq && bq->numel() > 0) {
            float* qd = static_cast<float*>(result.q->data());
            const float* bd = static_cast<const float*>(bq->data());
            for (int i = 0; i < N_q; ++i)
                qd[i] += bd[i];
        }
        if (bk && bk->numel() > 0) {
            float* kd = static_cast<float*>(result.k->data());
            const float* bd = static_cast<const float*>(bk->data());
            for (int i = 0; i < N_k; ++i)
                kd[i] += bd[i];
        }
    }
    // Generic fallback
    else {
        result.q = ops::matmul_transB(x, wq, bq);
        result.k = ops::matmul_transB(x, wk, bk);
        result.v = ops::matmul_transB(x, wv, bv);
    }

    return result;
}

// ---- Quantized / paged prefill dispatch helpers (Phase 2-B) ----
// The eight prefill cases below differ only in the KV dtype, the launch entry
// point and whether FP8 row scales are used, so they are generated from a macro.
// Non-CUDA builds fall back to materializing K/V and running the FP32 GQA SDPA.
#ifdef USE_CUDA
#define FORGE_QUANT_PREFILL_CASE(ENUM_PATH, DTYPE, LAUNCH_FN, SCALED)                          \
    case ENUM_PATH: {                                                                          \
        attn_out = std::make_shared<Tensor>(                                                    \
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim},               \
            DeviceType::CUDA);                                                                 \
        {                                                                                      \
            void* d_q_K = kv_cache_.d_q_key_cache(layer_idx);                                   \
            void* d_q_V = kv_cache_.d_q_value_cache(layer_idx);                                 \
            size_t q_row_size = KVCache::block_nbytes(DTYPE, num_kv_heads * head_dim);         \
            const float* k_sc =                                                                \
                (SCALED) ? static_cast<const float*>(kv_cache_.d_q_key_scale(layer_idx))       \
                         : nullptr;                                                             \
            const float* v_sc =                                                                \
                (SCALED) ? static_cast<const float*>(kv_cache_.d_q_value_scale(layer_idx))     \
                         : nullptr;                                                             \
            cuda::LAUNCH_FN(static_cast<const float*>(q->data()), d_q_K, d_q_V,                \
                            static_cast<float*>(attn_out->data()), seq_len, total_len,         \
                            num_heads, num_kv_heads, head_dim, q_row_size, mask_data, causal,  \
                            k_sc, v_sc, 0);                                                    \
        }                                                                                      \
        break;                                                                                 \
    }
#else
#define FORGE_QUANT_PREFILL_CASE(ENUM_PATH, DTYPE, LAUNCH_FN, SCALED)                          \
    case ENUM_PATH: {                                                                          \
        attn_out = ops::scaled_dot_product_attention_2d_gqa(                                   \
            q, kv_cache_.get_key_filled(layer_idx), kv_cache_.get_value_filled(layer_idx),     \
            seq_len, total_len, num_heads, num_kv_heads, head_dim, mask, true);                \
        break;                                                                                 \
    }
#endif

#ifdef USE_CUDA
// Asymmetric K/V: the two sides have different block geometries, so both the row
// strides and the per-head byte offsets differ. Passing one row size here (as the
// monomorphic macro does) is what made this configuration unservable.
#define FORGE_PAGED_MIXED_PREFILL_CASE(ENUM_PATH, DK, DV, LAUNCH_FN)                           \
    case ENUM_PATH: {                                                                          \
        attn_out = std::make_shared<Tensor>(                                                    \
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim},               \
            DeviceType::CUDA);                                                                 \
        {                                                                                      \
            const int32_t* d_page_ids = paged_tables(layer_idx, seq_id);                       \
            const size_t k_row_size =                                                          \
                KVCache::block_nbytes(DK, num_kv_heads * head_dim);                            \
            const size_t v_row_size =                                                          \
                KVCache::block_nbytes(DV, num_kv_heads * head_dim);                            \
            cuda::LAUNCH_FN(                                                                  \
                static_cast<const float*>(q->data()), storage_->d_key_page_ptrs(layer_idx),      \
                storage_->d_value_page_ptrs(layer_idx), d_page_ids,                            \
                static_cast<float*>(attn_out->data()), seq_len, total_len, num_heads,           \
                num_kv_heads, head_dim, storage_->page_size(), k_row_size, v_row_size,          \
                mask_data, causal, nullptr, nullptr, 0);                                       \
        }                                                                                      \
        break;                                                                                 \
    }

#define FORGE_PAGED_PREFILL_CASE(ENUM_PATH, DTYPE, LAUNCH_FN)                                  \
    case ENUM_PATH: {                                                                          \
        attn_out = std::make_shared<Tensor>(                                                    \
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim},               \
            DeviceType::CUDA);                                                                 \
        {                                                                                      \
            const int32_t* d_page_ids = paged_tables(layer_idx, seq_id);                       \
            size_t q_row_size = KVCache::block_nbytes(DTYPE, num_kv_heads * head_dim);         \
            cuda::LAUNCH_FN(static_cast<const float*>(q->data()),                              \
                            storage_->d_key_page_ptrs(layer_idx),                              \
                            storage_->d_value_page_ptrs(layer_idx), d_page_ids,                \
                            static_cast<float*>(attn_out->data()), seq_len, total_len,         \
                            num_heads, num_kv_heads, head_dim, storage_->page_size(),          \
                            q_row_size, mask_data, causal, nullptr, nullptr, 0);               \
        }                                                                                      \
        break;                                                                                 \
    }
#else
#define FORGE_PAGED_PREFILL_CASE(ENUM_PATH, DTYPE, LAUNCH_FN)                                  \
    case ENUM_PATH: {                                                                          \
        TensorPtr k_paged, v_paged;                                                            \
        get_kv_on_device(k_paged, v_paged);                                                    \
        attn_out = ops::scaled_dot_product_attention_2d_gqa(                                   \
            q, k_paged, v_paged, seq_len, total_len, num_heads, num_kv_heads, head_dim, mask,  \
            true);                                                                             \
        break;                                                                                 \
    }
#endif

TensorPtr AttentionExecutor::attend(const TensorPtr& q, const ModelConfig& cfg, int layer_idx,
                                    int seq_len, DeviceType dev, const TensorPtr& mask, int seq_id,
                                    bool causal, cudaStream_t stream) {
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;

    // Paged mode: read K/V from PagedKVStorage instead of KVCache
    bool paged = (storage_ != nullptr && storage_->is_paged());

    // Paged mode attends only to the current sequence's pages (per-seq).
    // This is critical when pages are shared across sequences (prefix cache),
    // since read_key/filled would otherwise duplicate shared-page data.
    // Contiguous mode keeps the all-sequence aggregate (mask handles per-seq).
    int total_len;
    if (paged && seq_id >= 0) {
        total_len = storage_->seq_filled(layer_idx, seq_id);
    } else if (paged) {
        total_len = storage_->filled(layer_idx);
    } else {
        total_len = kv_cache_.filled(layer_idx);
    }

    // Empty view: nothing in KV cache yet (first token of first sequence).
    // Return zero output — causal attention has no valid keys to attend to.
    if (total_len == 0) {
        auto out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, dev);
        out->zero_();
        return out;
    }

    // Helper: fetch and ensure FP32 K/V tensors on the target device.
    // Only needed for non-fused attention paths.
    auto get_kv_on_device = [&](TensorPtr& k_out, TensorPtr& v_out) {
        if (paged) {
            // Per-sequence read: avoids duplicating shared-page data.
            // Falls back to all-sequence read if seq_id is invalid.
            if (seq_id >= 0) {
                // Phase 4-B: a CPU-backed paged pool serving CUDA compute hands out its
                // persistent device row mirror, skipping the per-call host
                // materialization + full-window H2D copy (the dominant decode
                // cost at long context with partial weight offload). On a miss
                // the legacy per-sequence materialization path runs unchanged.
                if (dev == DeviceType::CUDA && storage_->device() == DeviceType::CPU) {
                    TensorPtr mk, mv;
                    if (storage_->read_window_cuda(layer_idx, seq_id, &mk, &mv)) {
                        k_out = mk;
                        v_out = mv;
                    }
                }
                if (!k_out) k_out = storage_->read_key_seq(layer_idx, seq_id);
                if (!k_out || !v_out) v_out = storage_->read_value_seq(layer_idx, seq_id);
            }
            if (!k_out || !v_out) {
                k_out = storage_->read_key(layer_idx);
                v_out = storage_->read_value(layer_idx);
            }
            // A CUDA paged storage cannot materialize rows (read_key* return
            // null by design). If we still ended up here, some path selected a
            // materialized kernel with paged device KV — fail loud instead of
            // dereferencing a null tensor downstream.
            if ((!k_out || !v_out) && dev == DeviceType::CUDA &&
                storage_->device() == DeviceType::CUDA) {
                throw std::runtime_error(
                    "paged CUDA storage cannot materialize K/V for this attention path "
                    "(layer=" + std::to_string(layer_idx) + ", seq=" + std::to_string(seq_id) +
                    ", seq_len=" + std::to_string(seq_len) + ")");
            }
        } else {
            // Quantized KV (Q4_0 / Q8_0 / Q4_K) is materialized to FP32 lazily.
            // update() invalidates dequantized_filled from the write position
            // onward, so the newest row(s) are stale in the FP32 tensor. If we
            // handed out the slice view without refreshing them, attention would
            // read uninitialized heap memory (nondeterministic under
            // multithreaded malloc). dequantize_layer() re-materializes only
            // rows [dequantized_filled, filled) and is a no-op when up to date;
            // it also allocates the FP32 tensor for engines that don't
            // pre-dequantize, so get_key_filled always returns a valid view.
            kv_cache_.dequantize_layer(layer_idx);
            k_out = kv_cache_.get_key_filled(layer_idx);
            v_out = kv_cache_.get_value_filled(layer_idx);
        }
        if (dev == DeviceType::CUDA && k_out && k_out->device() == DeviceType::CPU) {
            auto k_cuda =
                std::make_shared<Tensor>(DataType::FP32, k_out->shape(), DeviceType::CUDA);
            k_cuda->copy_from(*k_out);
            k_out = k_cuda;

            auto v_cuda =
                std::make_shared<Tensor>(DataType::FP32, v_out->shape(), DeviceType::CUDA);
            v_cuda->copy_from(*v_out);
            v_out = v_cuda;
        }
    };

    // Prepare mask data pointer for CUDA kernels
    const float* mask_data = nullptr;
    TensorPtr mask_on_dev = mask;
    if (mask && dev == DeviceType::CUDA && mask->device() == DeviceType::CPU) {
        mask_on_dev = std::make_shared<Tensor>(DataType::FP32, mask->shape(), DeviceType::CUDA);
        mask_on_dev->copy_from(*mask);
    }
    if (mask_on_dev) {
        mask_data = static_cast<const float*>(mask_on_dev->data());
    }

    // For decode with mask, the mask has already been sliced to a single row
    const float* mask_row = nullptr;
    if (mask && seq_len == 1 && mask->ndim() == 2) {
        mask_row = mask_data;
    }

    // Paged kernel dispatch helper: resolves the per-(seq, layer) device page
    // table and fails loud if it is unusable (e.g. storage not on CUDA, or the
    // sequence has no pages). Launching a paged kernel with null tables is an
    // illegal memory access — never do it silently.
    auto paged_tables = [&](int layer, int seq) -> const int32_t* {
        const void* k_pages = storage_->d_key_page_ptrs(layer);
        const void* v_pages = storage_->d_value_page_ptrs(layer);
        // Timed separately (FORGE_DEBUG/perf): this runs once per (layer, step)
        // on the decode hot path and syncs the sequence's page table to device.
        const int32_t* ids = [&] {
            PERF_SCOPE("attn/page_table_upload");
            return storage_->upload_seq_page_table(layer, seq);
        }();
        if (k_pages == nullptr || v_pages == nullptr || ids == nullptr) {
            throw std::runtime_error(
                "paged attention: null page table (storage not on CUDA or seq has no pages) "
                "layer=" +
                std::to_string(layer) + " seq=" + std::to_string(seq));
        }
        return ids;
    };

    // ---- Build problem descriptor and select path ----
    AttentionProblem problem;
    problem.device = dev;
    problem.seq_len = seq_len;
    problem.num_heads = num_heads;
    problem.num_kv_heads = num_kv_heads;
    problem.head_dim = head_dim;
    problem.q_dtype = q->dtype();  // 6-A-7: drives CUDA_F16_* selection

    // Quantized KV state for path selection.
    // - Paged CUDA decode: KV lives in pages; the storage reports the dtype.
    // - Contiguous CUDA decode: query the KVCache's quantized device pointers.
    // The paged fast path additionally requires the STORAGE to live on the
    // attention device. A paged storage on CPU (weights on GPU, KV on CPU) must
    // take the materialized paths below — the paged CUDA kernels would
    // otherwise receive null page tables (illegal memory access). It must also
    // NOT consult kv_cache_'s quantized pointers: in paged mode the contiguous
    // cache is allocated but never written (its d_q_* would be all-zero data).
    const bool paged_cuda =
        paged && dev == DeviceType::CUDA && storage_->device() == DeviceType::CUDA;
    if (dev == DeviceType::CUDA && seq_len == 1 && paged_cuda) {
        problem.paged = true;
        problem.seq_id = seq_id;
        problem.has_quantized_kv = true;  // paged CUDA always quantized-or-FP32
        problem.kv_type_k = storage_->kv_type_k();
        problem.kv_type_v = storage_->kv_type_v();
    } else if (dev == DeviceType::CUDA && seq_len == 1 && !paged) {
        void* d_q_K = kv_cache_.d_q_key_cache(layer_idx);
        void* d_q_V = kv_cache_.d_q_value_cache(layer_idx);
        problem.has_quantized_kv = (d_q_K != nullptr && d_q_V != nullptr);
        const auto& kv_cfg = kv_cache_.kv_config();
        problem.kv_type_k = kv_cfg.type_k;
        problem.kv_type_v = kv_cfg.type_v;
    } else if (dev == DeviceType::CUDA && seq_len > 1 && paged_cuda) {
        // Prefill (Phase 2-B): quantized KV is read directly by the fused
        // prefill kernel — no FP32 materialization. Paged KV resolves rows
        // through the sequence page table.
        problem.paged = true;
        problem.seq_id = seq_id;
        problem.has_quantized_kv = true;
        problem.kv_type_k = storage_->kv_type_k();
        problem.kv_type_v = storage_->kv_type_v();
    } else if (dev == DeviceType::CUDA && seq_len > 1 && !paged) {
        // Contiguous KV reads the cache's quantized device pointers.
        void* d_q_K = kv_cache_.d_q_key_cache(layer_idx);
        void* d_q_V = kv_cache_.d_q_value_cache(layer_idx);
        problem.has_quantized_kv = (d_q_K != nullptr && d_q_V != nullptr);
        const auto& kv_cfg = kv_cache_.kv_config();
        problem.kv_type_k = kv_cfg.type_k;
        problem.kv_type_v = kv_cfg.type_v;
    }
    // paged storage on CPU + CUDA attention (and any other combination) falls
    // through: plain FP32 materialized path via get_kv_on_device().

    AttentionPath path = choose_attention_path(problem);

    // ---- Dispatch ----
    TensorPtr attn_out;

    switch (path) {
    case AttentionPath::CPU_MHA: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
        attn_out = ops::scaled_dot_product_attention_2d(q, k_sliced, v_sliced, seq_len, total_len,
                                                        num_heads, head_dim, mask, causal);
        break;
    }
    case AttentionPath::CPU_GQA: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
        attn_out = ops::scaled_dot_product_attention_2d_gqa(q, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, num_kv_heads,
                                                            head_dim, mask, causal);
        break;
    }
    case AttentionPath::CUDA_GENERIC_MHA: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
        attn_out = ops::scaled_dot_product_attention_2d(q, k_sliced, v_sliced, seq_len, total_len,
                                                        num_heads, head_dim, mask, true);
        break;
    }
    // ---- Phase 2-C: FP32 contiguous prefill ----
    // Vectorized is ~2x faster than the tiled FP32 prefill on this hardware, so
    // only the quantized prefill paths use the tiled kernel: there it avoids
    // materializing the KV cache (the real win — no FP32 read of the whole cache).
    case AttentionPath::CUDA_FP32_PREFILL: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
        // 6-B-1: FP32 prefill reads Q/K/V as float* below -- BF16 falls through
        // to this default path (selector only recognizes FP16), so reject loudly.
        require_no_bf16("AttentionExecutor::CUDA_FP32_PREFILL/q", OpType::FLASH_ATTN_GQA,
                        q->dtype());
        require_no_bf16("AttentionExecutor::CUDA_FP32_PREFILL/k", OpType::FLASH_ATTN_GQA,
                        k_sliced->dtype());
        require_no_bf16("AttentionExecutor::CUDA_FP32_PREFILL/v", OpType::FLASH_ATTN_GQA,
                        v_sliced->dtype());
        // 6-C-1: FP8 has no attention kernel on this path -- reject before the
        // float* reads (FP8 Q falls through the selector like BF16 did).
        require_no_fp8("AttentionExecutor::CUDA_FP32_PREFILL/q", OpType::FLASH_ATTN_GQA,
                       q->dtype());
        require_no_fp8("AttentionExecutor::CUDA_FP32_PREFILL/k", OpType::FLASH_ATTN_GQA,
                       k_sliced->dtype());
        require_no_fp8("AttentionExecutor::CUDA_FP32_PREFILL/v", OpType::FLASH_ATTN_GQA,
                       v_sliced->dtype());
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        cuda::launch_flash_attention_gqa_vectorized(
            static_cast<const float*>(q->data()), static_cast<const float*>(k_sliced->data()),
            static_cast<const float*>(v_sliced->data()), static_cast<float*>(attn_out->data()),
            seq_len, total_len, num_heads, num_kv_heads, head_dim, mask_data, causal, 0);
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(q, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, num_kv_heads,
                                                            head_dim, mask, causal);
#endif
        break;
    }
    // ---- Phase 2-B: fused quantized prefill (KV stays quantized) ----
    FORGE_QUANT_PREFILL_CASE(AttentionPath::CUDA_FUSED_Q4_0_PREFILL, KVCacheDType::Q4_0,
                             launch_fused_flash_attention_gqa_prefill_q4_0, false)
    FORGE_QUANT_PREFILL_CASE(AttentionPath::CUDA_FUSED_F16_PREFILL, KVCacheDType::F16,
                             launch_fused_flash_attention_gqa_prefill_f16, false)
    FORGE_QUANT_PREFILL_CASE(AttentionPath::CUDA_FUSED_Q8_0_PREFILL, KVCacheDType::Q8_0,
                             launch_fused_flash_attention_gqa_prefill_q8_0, false)
    FORGE_QUANT_PREFILL_CASE(AttentionPath::CUDA_FUSED_FP8_E4M3_PREFILL, KVCacheDType::FP8_E4M3,
                             launch_fused_flash_attention_gqa_prefill_fp8_e4m3, true)
    FORGE_QUANT_PREFILL_CASE(AttentionPath::CUDA_FUSED_FP8_E5M2_PREFILL, KVCacheDType::FP8_E5M2,
                             launch_fused_flash_attention_gqa_prefill_fp8_e5m2, true)

    // ---- Phase 2-B: paged quantized prefill (rows via the page table) ----
    FORGE_PAGED_PREFILL_CASE(AttentionPath::CUDA_PAGED_FP32_PREFILL, KVCacheDType::FP32,
                             launch_paged_flash_attention_gqa_prefill_fp32)
    FORGE_PAGED_PREFILL_CASE(AttentionPath::CUDA_PAGED_Q4_0_PREFILL, KVCacheDType::Q4_0,
                             launch_paged_flash_attention_gqa_prefill_q4_0)
    // Asymmetric K/V prefill. One case, dispatching on the dtype pair inside:
    // six FORGE_PAGED_MIXED_PREFILL_CASE invocations would be six `case` labels
    // on the same enumerator, which does not compile.
    case AttentionPath::CUDA_PAGED_MIXED_PREFILL: {
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            const int32_t* d_page_ids = paged_tables(layer_idx, seq_id);
            // storage_, not kv_cache_: see the decode case above.
            const KVCacheDType tk = storage_->kv_type_k();
            const KVCacheDType tv = storage_->kv_type_v();
            const size_t k_row_size = KVCache::block_nbytes(tk, num_kv_heads * head_dim);
            const size_t v_row_size = KVCache::block_nbytes(tv, num_kv_heads * head_dim);
            const auto run = [&](auto fn) {
                fn(static_cast<const float*>(q->data()), storage_->d_key_page_ptrs(layer_idx),
                   storage_->d_value_page_ptrs(layer_idx), d_page_ids,
                   static_cast<float*>(attn_out->data()), seq_len, total_len, num_heads,
                   num_kv_heads, head_dim, storage_->page_size(), k_row_size, v_row_size,
                   mask_data, causal, nullptr, nullptr, 0);
            };
            if (tk == KVCacheDType::Q8_0 && tv == KVCacheDType::Q4_0)
                run(cuda::launch_paged_flash_attention_gqa_prefill_q8_0_q4_0);
            else if (tk == KVCacheDType::Q4_0 && tv == KVCacheDType::Q8_0)
                run(cuda::launch_paged_flash_attention_gqa_prefill_q4_0_q8_0);
            else if (tk == KVCacheDType::Q8_0 && tv == KVCacheDType::F16)
                run(cuda::launch_paged_flash_attention_gqa_prefill_q8_0_f16);
            else if (tk == KVCacheDType::F16 && tv == KVCacheDType::Q8_0)
                run(cuda::launch_paged_flash_attention_gqa_prefill_f16_q8_0);
            else if (tk == KVCacheDType::Q4_0 && tv == KVCacheDType::F16)
                run(cuda::launch_paged_flash_attention_gqa_prefill_q4_0_f16);
            else if (tk == KVCacheDType::F16 && tv == KVCacheDType::Q4_0)
                run(cuda::launch_paged_flash_attention_gqa_prefill_f16_q4_0);
            else
                throw std::runtime_error(
                    std::string("paged mixed prefill: unsupported KV dtype pair (K=") +
                    kv_cache_dtype_name(tk) + ", V=" + kv_cache_dtype_name(tv) + ")");
        }
        break;
    }
    FORGE_PAGED_PREFILL_CASE(AttentionPath::CUDA_PAGED_F16_PREFILL, KVCacheDType::F16,
                             launch_paged_flash_attention_gqa_prefill_f16)
    FORGE_PAGED_PREFILL_CASE(AttentionPath::CUDA_PAGED_Q8_0_PREFILL, KVCacheDType::Q8_0,
                             launch_paged_flash_attention_gqa_prefill_q8_0)
    FORGE_PAGED_PREFILL_CASE(AttentionPath::CUDA_PAGED_FP8_E4M3_PREFILL, KVCacheDType::FP8_E4M3,
                             launch_paged_flash_attention_gqa_prefill_fp8_e4m3)
    FORGE_PAGED_PREFILL_CASE(AttentionPath::CUDA_PAGED_FP8_E5M2_PREFILL, KVCacheDType::FP8_E5M2,
                             launch_paged_flash_attention_gqa_prefill_fp8_e5m2)

    case AttentionPath::CUDA_FP32_DECODE: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
        // 6-B-1: FP32 decode reads Q/K/V as float* below -- BF16 falls through
        // to this default path (selector only recognizes FP16), so reject loudly.
        require_no_bf16("AttentionExecutor::CUDA_FP32_DECODE/q", OpType::FLASH_ATTN_GQA,
                        q->dtype());
        require_no_bf16("AttentionExecutor::CUDA_FP32_DECODE/k", OpType::FLASH_ATTN_GQA,
                        k_sliced->dtype());
        require_no_bf16("AttentionExecutor::CUDA_FP32_DECODE/v", OpType::FLASH_ATTN_GQA,
                        v_sliced->dtype());
        // 6-C-1: FP8 has no attention kernel on this path -- reject before the
        // float* reads (FP8 Q falls through the selector like BF16 did).
        require_no_fp8("AttentionExecutor::CUDA_FP32_DECODE/q", OpType::FLASH_ATTN_GQA,
                       q->dtype());
        require_no_fp8("AttentionExecutor::CUDA_FP32_DECODE/k", OpType::FLASH_ATTN_GQA,
                       k_sliced->dtype());
        require_no_fp8("AttentionExecutor::CUDA_FP32_DECODE/v", OpType::FLASH_ATTN_GQA,
                       v_sliced->dtype());
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        cuda::launch_flash_attention_gqa_decode(
            static_cast<const float*>(q->data()), static_cast<const float*>(k_sliced->data()),
            static_cast<const float*>(v_sliced->data()), static_cast<float*>(attn_out->data()),
            total_len, num_heads, num_kv_heads, head_dim, mask_row, 0);
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(q, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, num_kv_heads,
                                                            head_dim, mask, causal);
#endif
        break;
    }
    // ---- 6-A-7: native-FP16 decode (all-FP16 QKV, FP32 math, FP16 out) ----
    // Locked rule: selected only when q is FP16 (selector). K/V must also be
    // FP16 here — no silent reinterpret and no implicit conversion. FP16 KV
    // cache integration is post-7.1; until then a non-FP16 KV throws loudly.
    case AttentionPath::CUDA_F16_DECODE: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
#ifdef USE_CUDA
        if (!k_sliced || !v_sliced || k_sliced->dtype() != DataType::FP16 ||
            v_sliced->dtype() != DataType::FP16) {
            throw std::runtime_error(
                "AttentionExecutor::CUDA_F16_DECODE requires FP16 K/V (native FP16 "
                "attention needs all-FP16 QKV; FP16 KV-cache integration is post-7.1).");
        }
        if (head_dim != 16 && head_dim != 32 && head_dim != 64 && head_dim != 96 &&
            head_dim != 128) {
            throw std::runtime_error("AttentionExecutor::CUDA_F16_DECODE: unsupported head_dim=" +
                                     std::to_string(head_dim) + " (6-A-7: 16/32/64/96/128 only).");
        }
        const __half* mask_row_fp16 = nullptr;
        if (mask) {
            if (mask_on_dev->dtype() != DataType::FP16) {
                throw std::runtime_error(
                    "AttentionExecutor::CUDA_F16_DECODE requires an FP16 mask (or none).");
            }
            mask_row_fp16 = reinterpret_cast<const __half*>(mask_on_dev->data());
        }
        attn_out = std::make_shared<Tensor>(
            DataType::FP16, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        cuda::launch_flash_attention_gqa_decode_fp16(
            reinterpret_cast<const __half*>(q->data()),
            reinterpret_cast<const __half*>(k_sliced->data()),
            reinterpret_cast<const __half*>(v_sliced->data()),
            reinterpret_cast<__half*>(attn_out->data()), total_len, num_heads, num_kv_heads,
            head_dim, mask_row_fp16, stream);
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(q, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, num_kv_heads,
                                                            head_dim, mask, causal);
#endif
        break;
    }
    // ---- 6-B-6: native BF16-Q decode (decode-first; BF16 Q + FP32 KV) ----
    // 6-B-6.7 extends K/V to BF16 (authoritative BF16 KV): the K/V dtype
    // selects the kernel (FP32 -> bf16q launch, BF16 -> bf16 launch); mixed
    // K/V dtypes keep loud rejection. Locked rule: selected only when q is
    // BF16 with seq_len == 1 (selector). No silent reinterpret, no implicit
    // conversion; BF16 prefill still throws loudly until its own 6-B step.
    case AttentionPath::CUDA_BF16_DECODE: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
#ifdef USE_CUDA
        if (q->dtype() != DataType::BF16) {
            throw std::runtime_error(
                "AttentionExecutor::CUDA_BF16_DECODE requires a BF16 Q (selector "
                "routes q-BF16 here; anything else is a dispatch bug).");
        }
        const bool kv_fp32 = k_sliced && v_sliced && k_sliced->dtype() == DataType::FP32 &&
                             v_sliced->dtype() == DataType::FP32;
        const bool kv_bf16 = k_sliced && v_sliced && k_sliced->dtype() == DataType::BF16 &&
                             v_sliced->dtype() == DataType::BF16;
        if (!kv_fp32 && !kv_bf16) {
            throw std::runtime_error(
                "AttentionExecutor::CUDA_BF16_DECODE requires all-FP32 or all-BF16 K/V "
                "(mixed K/V fail loudly).");
        }
        if (head_dim != 16 && head_dim != 32 && head_dim != 64 && head_dim != 96 &&
            head_dim != 128) {
            throw std::runtime_error("AttentionExecutor::CUDA_BF16_DECODE: unsupported head_dim=" +
                                     std::to_string(head_dim) + " (6-B-6: 16/32/64/96/128 only).");
        }
        attn_out = std::make_shared<Tensor>(
            DataType::BF16, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        if (kv_bf16) {
            const __nv_bfloat16* mask_row_bf16 = nullptr;
            if (mask) {
                if (mask_on_dev->dtype() != DataType::BF16) {
                    throw std::runtime_error(
                        "AttentionExecutor::CUDA_BF16_DECODE requires a BF16 mask with BF16 "
                        "K/V (or none).");
                }
                mask_row_bf16 = reinterpret_cast<const __nv_bfloat16*>(mask_on_dev->data());
            }
            cuda::launch_flash_attention_gqa_decode_bf16(
                reinterpret_cast<const __nv_bfloat16*>(q->data()),
                reinterpret_cast<const __nv_bfloat16*>(k_sliced->data()),
                reinterpret_cast<const __nv_bfloat16*>(v_sliced->data()),
                reinterpret_cast<__nv_bfloat16*>(attn_out->data()), total_len, num_heads,
                num_kv_heads, head_dim, mask_row_bf16, stream);
        } else {
            if (mask && mask_on_dev->dtype() != DataType::FP32) {
                throw std::runtime_error(
                    "AttentionExecutor::CUDA_BF16_DECODE requires an FP32 mask with FP32 "
                    "K/V (or none).");
            }
            cuda::launch_flash_attention_gqa_decode_bf16q(
                reinterpret_cast<const __nv_bfloat16*>(q->data()),
                static_cast<const float*>(k_sliced->data()),
                static_cast<const float*>(v_sliced->data()),
                reinterpret_cast<__nv_bfloat16*>(attn_out->data()), total_len, num_heads,
                num_kv_heads, head_dim, mask_row, stream);
        }
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(q, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, num_kv_heads,
                                                            head_dim, mask, causal);
#endif
        break;
    }
    // ---- 6-A-7.2: native-FP16 prefill (same dtype contract as decode) ----
    // Mirror of the vectorized FP32 prefill: Q tile in smem, strided KV loop,
    // local_acc kept. K/V must be FP16 (loud otherwise); mask FP16 or absent.
    case AttentionPath::CUDA_F16_PREFILL: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
#ifdef USE_CUDA
        if (!k_sliced || !v_sliced || k_sliced->dtype() != DataType::FP16 ||
            v_sliced->dtype() != DataType::FP16) {
            throw std::runtime_error(
                "AttentionExecutor::CUDA_F16_PREFILL requires FP16 K/V (native FP16 "
                "attention needs all-FP16 QKV; FP16 KV-cache integration is post-7.1).");
        }
        if (head_dim != 16 && head_dim != 32 && head_dim != 64 && head_dim != 96 &&
            head_dim != 128) {
            throw std::runtime_error("AttentionExecutor::CUDA_F16_PREFILL: unsupported head_dim=" +
                                     std::to_string(head_dim) + " (6-A-7: 16/32/64/96/128 only).");
        }
        const __half* mask_fp16 = nullptr;
        if (mask) {
            if (mask_on_dev->dtype() != DataType::FP16) {
                throw std::runtime_error(
                    "AttentionExecutor::CUDA_F16_PREFILL requires an FP16 mask (or none).");
            }
            mask_fp16 = reinterpret_cast<const __half*>(mask_on_dev->data());
        }
        attn_out = std::make_shared<Tensor>(
            DataType::FP16, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        cuda::launch_flash_attention_gqa_prefill_fp16(
            reinterpret_cast<const __half*>(q->data()),
            reinterpret_cast<const __half*>(k_sliced->data()),
            reinterpret_cast<const __half*>(v_sliced->data()),
            reinterpret_cast<__half*>(attn_out->data()), seq_len, total_len, num_heads,
            num_kv_heads, head_dim, mask_fp16, causal, stream);
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(q, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, num_kv_heads,
                                                            head_dim, mask, causal);
#endif
        break;
    }
    // ---- 6-B-7.0: native-BF16 prefill (E2E unlock; all-BF16 QKV) ----
    // Mirror of the F16 prefill above: Q tile in smem, strided KV loop,
    // local_acc kept, FP32 math, BF16 out. K/V must be BF16 (loud otherwise);
    // mask BF16 or absent. No wmma (decode-track parity).
    case AttentionPath::CUDA_BF16_PREFILL: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
#ifdef USE_CUDA
        if (q->dtype() != DataType::BF16) {
            throw std::runtime_error(
                "AttentionExecutor::CUDA_BF16_PREFILL requires a BF16 Q (selector "
                "routes q-BF16 prefill here; anything else is a dispatch bug).");
        }
        if (!k_sliced || !v_sliced || k_sliced->dtype() != DataType::BF16 ||
            v_sliced->dtype() != DataType::BF16) {
            throw std::runtime_error(
                "AttentionExecutor::CUDA_BF16_PREFILL requires BF16 K/V (native BF16 "
                "prefill needs all-BF16 QKV).");
        }
        if (head_dim != 16 && head_dim != 32 && head_dim != 64 && head_dim != 96 &&
            head_dim != 128) {
            throw std::runtime_error("AttentionExecutor::CUDA_BF16_PREFILL: unsupported head_dim=" +
                                     std::to_string(head_dim) + " (6-B-7: 16/32/64/96/128 only).");
        }
        const __nv_bfloat16* mask_bf16 = nullptr;
        if (mask) {
            if (mask_on_dev->dtype() != DataType::BF16) {
                throw std::runtime_error(
                    "AttentionExecutor::CUDA_BF16_PREFILL requires a BF16 mask (or none).");
            }
            mask_bf16 = reinterpret_cast<const __nv_bfloat16*>(mask_on_dev->data());
        }
        attn_out = std::make_shared<Tensor>(
            DataType::BF16, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        cuda::launch_flash_attention_gqa_prefill_bf16(
            reinterpret_cast<const __nv_bfloat16*>(q->data()),
            reinterpret_cast<const __nv_bfloat16*>(k_sliced->data()),
            reinterpret_cast<const __nv_bfloat16*>(v_sliced->data()),
            reinterpret_cast<__nv_bfloat16*>(attn_out->data()), seq_len, total_len, num_heads,
            num_kv_heads, head_dim, mask_bf16, causal, stream);
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(q, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, num_kv_heads,
                                                            head_dim, mask, causal);
#endif
        break;
    }
    case AttentionPath::CUDA_FUSED_Q4_0_DECODE:
#ifdef USE_CUDA
        // Fused decode reads quantized KV directly — no materialize.
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            void* d_q_K = kv_cache_.d_q_key_cache(layer_idx);
            void* d_q_V = kv_cache_.d_q_value_cache(layer_idx);
            size_t q_row_size = KVCache::block_nbytes(KVCacheDType::Q4_0, num_kv_heads * head_dim);
            cuda::launch_fused_flash_attention_gqa_decode_q4_0(
                static_cast<const float*>(q->data()), d_q_K, d_q_V,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                q_row_size, mask_row, 0);
        }
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, kv_cache_.get_key_filled(layer_idx), kv_cache_.get_value_filled(layer_idx), seq_len,
            total_len, num_heads, num_kv_heads, head_dim, mask, true);
#endif
        break;

    case AttentionPath::CUDA_FUSED_F16_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            void* d_q_K = kv_cache_.d_q_key_cache(layer_idx);
            void* d_q_V = kv_cache_.d_q_value_cache(layer_idx);
            size_t q_row_size = KVCache::block_nbytes(KVCacheDType::F16, num_kv_heads * head_dim);
            cuda::launch_fused_flash_attention_gqa_decode_f16(
                static_cast<const float*>(q->data()), d_q_K, d_q_V,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                q_row_size, mask_row, 0);
        }
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, kv_cache_.get_key_filled(layer_idx), kv_cache_.get_value_filled(layer_idx), seq_len,
            total_len, num_heads, num_kv_heads, head_dim, mask, true);
#endif
        break;

    case AttentionPath::CUDA_FUSED_Q8_0_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            void* d_q_K = kv_cache_.d_q_key_cache(layer_idx);
            void* d_q_V = kv_cache_.d_q_value_cache(layer_idx);
            size_t q_row_size = KVCache::block_nbytes(KVCacheDType::Q8_0, num_kv_heads * head_dim);
            cuda::launch_fused_flash_attention_gqa_decode_q8_0(
                static_cast<const float*>(q->data()), d_q_K, d_q_V,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                q_row_size, mask_row, 0);
        }
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, kv_cache_.get_key_filled(layer_idx), kv_cache_.get_value_filled(layer_idx), seq_len,
            total_len, num_heads, num_kv_heads, head_dim, mask, true);
#endif
        break;

    case AttentionPath::CUDA_FUSED_MIXED_DECODE: {
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            void* d_q_K = kv_cache_.d_q_key_cache(layer_idx);
            void* d_q_V = kv_cache_.d_q_value_cache(layer_idx);
            // Independent row sizes: the two sides are stored at different
            // precisions, so a shared row size would mis-stride one of them.
            const auto& kv_cfg = kv_cache_.kv_config();
            const int kv_dim = num_kv_heads * head_dim;
            const size_t k_row_size = KVCache::block_nbytes(kv_cfg.type_k, kv_dim);
            const size_t v_row_size = KVCache::block_nbytes(kv_cfg.type_v, kv_dim);
            const auto fmt = [](KVCacheDType d) {
                switch (d) {
                case KVCacheDType::Q8_0: return 0;
                case KVCacheDType::Q4_0: return 1;
                case KVCacheDType::F16:  return 2;
                default: return -1;
                }
            };
            const bool ok = cuda::launch_fused_flash_attention_gqa_decode_mixed(
                static_cast<const float*>(q->data()), d_q_K, d_q_V,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                k_row_size, v_row_size, fmt(kv_cfg.type_k), fmt(kv_cfg.type_v), mask_row, 0);
            // The selector already gated on this, so a false here means the two
            // disagree. Fail loudly rather than emit an uninitialised output.
            if (!ok) {
                throw std::runtime_error(
                    std::string("fused mixed-dtype attention: launcher refused the requested "
                                "combination (K=") +
                    kv_cache_dtype_name(kv_cfg.type_k) + ", V=" +
                    kv_cache_dtype_name(kv_cfg.type_v) +
                    ", head_dim=" + std::to_string(head_dim) + ")");
            }
        }
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, kv_cache_.get_key_filled(layer_idx), kv_cache_.get_value_filled(layer_idx), seq_len,
            total_len, num_heads, num_kv_heads, head_dim, mask, true);
#endif
        break;
    }

    case AttentionPath::CUDA_FUSED_FP8_E4M3_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            void* d_q_K = kv_cache_.d_q_key_cache(layer_idx);
            void* d_q_V = kv_cache_.d_q_value_cache(layer_idx);
            size_t q_row_size =
                KVCache::block_nbytes(KVCacheDType::FP8_E4M3, num_kv_heads * head_dim);
            const float* k_scales = static_cast<const float*>(kv_cache_.d_q_key_scale(layer_idx));
            const float* v_scales = static_cast<const float*>(kv_cache_.d_q_value_scale(layer_idx));
            cuda::launch_fused_flash_attention_gqa_decode_fp8_e4m3(
                static_cast<const float*>(q->data()), d_q_K, d_q_V,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                q_row_size, mask_row, k_scales, v_scales, 0);
        }
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, kv_cache_.get_key_filled(layer_idx), kv_cache_.get_value_filled(layer_idx), seq_len,
            total_len, num_heads, num_kv_heads, head_dim, mask, true);
#endif
        break;

    case AttentionPath::CUDA_FUSED_FP8_E5M2_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            void* d_q_K = kv_cache_.d_q_key_cache(layer_idx);
            void* d_q_V = kv_cache_.d_q_value_cache(layer_idx);
            size_t q_row_size =
                KVCache::block_nbytes(KVCacheDType::FP8_E5M2, num_kv_heads * head_dim);
            const float* k_scales = static_cast<const float*>(kv_cache_.d_q_key_scale(layer_idx));
            const float* v_scales = static_cast<const float*>(kv_cache_.d_q_value_scale(layer_idx));
            cuda::launch_fused_flash_attention_gqa_decode_fp8_e5m2(
                static_cast<const float*>(q->data()), d_q_K, d_q_V,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                q_row_size, mask_row, k_scales, v_scales, 0);
        }
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, kv_cache_.get_key_filled(layer_idx), kv_cache_.get_value_filled(layer_idx), seq_len,
            total_len, num_heads, num_kv_heads, head_dim, mask, true);
#endif
        break;

    // ---- Paged CUDA decode (Phase 4) ----
    // KV rows are resolved through the per-sequence page table; no materialize.
    // ---- Paged MIXED decode (asymmetric K/V) ----
    // Rows are resolved through the page table; K and V are templated
    // independently, so each side carries its own row stride and per-head byte
    // offset. Passing the page-pointer table to a contiguous kernel here is what
    // segfaulted this configuration before the paged mixed kernel existed.
    case AttentionPath::CUDA_PAGED_MIXED_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            const int32_t* d_page_ids = paged_tables(layer_idx, seq_id);
            // storage_, not kv_cache_: in paged mode the rows live in the
            // storage and kv_cache_ stays FP32, so asking kv_cache_ reports
            // (fp32, fp32) and the launcher refuses.
            const KVCacheDType tk = storage_->kv_type_k();
            const KVCacheDType tv = storage_->kv_type_v();
            const size_t k_row_size = KVCache::block_nbytes(tk, num_kv_heads * head_dim);
            const size_t v_row_size = KVCache::block_nbytes(tv, num_kv_heads * head_dim);
            const auto fmt = [](KVCacheDType d) {
                switch (d) {
                case KVCacheDType::Q8_0: return 0;
                case KVCacheDType::Q4_0: return 1;
                case KVCacheDType::F16:  return 2;
                default: return -1;
                }
            };
            const bool ok = cuda::launch_paged_flash_attention_gqa_decode_mixed(
                static_cast<const float*>(q->data()), storage_->d_key_page_ptrs(layer_idx),
                storage_->d_value_page_ptrs(layer_idx), d_page_ids,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads,
                head_dim, storage_->page_size(), k_row_size, v_row_size, fmt(tk), fmt(tv),
                mask_row, 0);
            // The selector gated on coverage, so false here means the two
            // disagree. Fail loudly rather than emit an uninitialised output.
            if (!ok) {
                throw std::runtime_error(
                    std::string("paged mixed-dtype attention: launcher refused (K=") +
                    kv_cache_dtype_name(tk) + ", V=" + kv_cache_dtype_name(tv) +
                    ", head_dim=" + std::to_string(head_dim) + ")");
            }
        }
#endif
        break;

    case AttentionPath::CUDA_PAGED_Q4_0_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            const int32_t* d_page_ids = paged_tables(layer_idx, seq_id);
            size_t q_row_size = KVCache::block_nbytes(KVCacheDType::Q4_0, num_kv_heads * head_dim);
            cuda::launch_paged_flash_attention_gqa_decode_q4_0(
                static_cast<const float*>(q->data()), storage_->d_key_page_ptrs(layer_idx),
                storage_->d_value_page_ptrs(layer_idx), d_page_ids,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                storage_->page_size(), q_row_size, mask_row, 0);
        }
#else
    {
        TensorPtr k_, v_;
        get_kv_on_device(k_, v_);
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, k_, v_, seq_len, total_len, num_heads, num_kv_heads, head_dim, mask, true);
    }
#endif
        break;

    case AttentionPath::CUDA_PAGED_F16_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            const int32_t* d_page_ids = paged_tables(layer_idx, seq_id);
            size_t q_row_size = KVCache::block_nbytes(KVCacheDType::F16, num_kv_heads * head_dim);
            cuda::launch_paged_flash_attention_gqa_decode_f16(
                static_cast<const float*>(q->data()), storage_->d_key_page_ptrs(layer_idx),
                storage_->d_value_page_ptrs(layer_idx), d_page_ids,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                storage_->page_size(), q_row_size, mask_row, 0);
        }
#else
    {
        TensorPtr k_, v_;
        get_kv_on_device(k_, v_);
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, k_, v_, seq_len, total_len, num_heads, num_kv_heads, head_dim, mask, true);
    }
#endif
        break;

    case AttentionPath::CUDA_PAGED_Q8_0_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            const int32_t* d_page_ids = paged_tables(layer_idx, seq_id);
            size_t q_row_size = KVCache::block_nbytes(KVCacheDType::Q8_0, num_kv_heads * head_dim);
            cuda::launch_paged_flash_attention_gqa_decode_q8_0(
                static_cast<const float*>(q->data()), storage_->d_key_page_ptrs(layer_idx),
                storage_->d_value_page_ptrs(layer_idx), d_page_ids,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                storage_->page_size(), q_row_size, mask_row, 0);
        }
#else
    {
        TensorPtr k_, v_;
        get_kv_on_device(k_, v_);
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, k_, v_, seq_len, total_len, num_heads, num_kv_heads, head_dim, mask, true);
    }
#endif
        break;

    case AttentionPath::CUDA_PAGED_FP32_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            const int32_t* d_page_ids = paged_tables(layer_idx, seq_id);
            size_t q_row_size = KVCache::block_nbytes(KVCacheDType::FP32, num_kv_heads * head_dim);
            cuda::launch_paged_flash_attention_gqa_decode_fp32(
                static_cast<const float*>(q->data()), storage_->d_key_page_ptrs(layer_idx),
                storage_->d_value_page_ptrs(layer_idx), d_page_ids,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                storage_->page_size(), q_row_size, mask_row, 0);
        }
#else
    {
        TensorPtr k_, v_;
        get_kv_on_device(k_, v_);
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, k_, v_, seq_len, total_len, num_heads, num_kv_heads, head_dim, mask, true);
    }
#endif
        break;

    case AttentionPath::CUDA_PAGED_FP8_E4M3_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            const int32_t* d_page_ids = paged_tables(layer_idx, seq_id);
            size_t q_row_size =
                KVCache::block_nbytes(KVCacheDType::FP8_E4M3, num_kv_heads * head_dim);
            cuda::launch_paged_flash_attention_gqa_decode_fp8_e4m3(
                static_cast<const float*>(q->data()), storage_->d_key_page_ptrs(layer_idx),
                storage_->d_value_page_ptrs(layer_idx), d_page_ids,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                storage_->page_size(), q_row_size, mask_row, 0);
        }
#else
    {
        TensorPtr k_, v_;
        get_kv_on_device(k_, v_);
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, k_, v_, seq_len, total_len, num_heads, num_kv_heads, head_dim, mask, true);
    }
#endif
        break;

    case AttentionPath::CUDA_PAGED_FP8_E5M2_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            const int32_t* d_page_ids = paged_tables(layer_idx, seq_id);
            size_t q_row_size =
                KVCache::block_nbytes(KVCacheDType::FP8_E5M2, num_kv_heads * head_dim);
            cuda::launch_paged_flash_attention_gqa_decode_fp8_e5m2(
                static_cast<const float*>(q->data()), storage_->d_key_page_ptrs(layer_idx),
                storage_->d_value_page_ptrs(layer_idx), d_page_ids,
                static_cast<float*>(attn_out->data()), total_len, num_heads, num_kv_heads, head_dim,
                storage_->page_size(), q_row_size, mask_row, 0);
        }
#else
    {
        TensorPtr k_, v_;
        get_kv_on_device(k_, v_);
        attn_out = ops::scaled_dot_product_attention_2d_gqa(
            q, k_, v_, seq_len, total_len, num_heads, num_kv_heads, head_dim, mask, true);
    }
#endif
        break;

    case AttentionPath::UNSUPPORTED:
    default:
        fprintf(stderr,
                "[ERROR] Unsupported attention path for device=%d seq_len=%d "
                "num_heads=%d num_kv_heads=%d head_dim=%d\n",
                static_cast<int>(dev), seq_len, num_heads, num_kv_heads, head_dim);
        break;
    }

    return attn_out;
}

TensorPtr AttentionExecutor::expand_kv_heads(const TensorPtr& kv, int seq_len, int num_heads,
                                             int num_kv_heads, int head_dim, DeviceType dev) {
    int kv_groups = num_heads / num_kv_heads;
    auto expanded = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, dev);

    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_expand_kv(static_cast<const float*>(kv->data()),
                               static_cast<float*>(expanded->data()), seq_len, num_heads,
                               num_kv_heads, head_dim);
#endif
    } else {
        const float* kv_data = static_cast<const float*>(kv->data());
        float* out_data = static_cast<float*>(expanded->data());
        forge::cpu::expand_kv_heads_f32(kv_data, out_data, seq_len, num_heads, num_kv_heads,
                                        head_dim);
    }

    return expanded;
}

}  // namespace forge
