#include "forge/inference/layers/attention_executor.h"

#include <cstring>

#include "cpu/simd.h"
#include "forge/attention_selector.h"
#include "forge/cuda_kernels.h"
#include "forge/operators.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

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

TensorPtr AttentionExecutor::attend(const TensorPtr& q, const ModelConfig& cfg, int layer_idx,
                                    int seq_len, DeviceType dev, const TensorPtr& mask, int seq_id,
                                    bool causal) {
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
                k_out = storage_->read_key_seq(layer_idx, seq_id);
                v_out = storage_->read_value_seq(layer_idx, seq_id);
            }
            if (!k_out || !v_out) {
                k_out = storage_->read_key(layer_idx);
                v_out = storage_->read_value(layer_idx);
            }
        } else {
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

    // ---- Build problem descriptor and select path ----
    AttentionProblem problem;
    problem.device = dev;
    problem.seq_len = seq_len;
    problem.num_heads = num_heads;
    problem.num_kv_heads = num_kv_heads;
    problem.head_dim = head_dim;

    // Quantized KV state for path selection.
    // - Paged CUDA decode: KV lives in pages; the storage reports the dtype.
    // - Contiguous CUDA decode: query the KVCache's quantized device pointers.
    if (dev == DeviceType::CUDA && seq_len == 1 && paged) {
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
    }

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
    case AttentionPath::CUDA_FP32_PREFILL: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        cuda::launch_flash_attention_gqa(
            static_cast<const float*>(q->data()), static_cast<const float*>(k_sliced->data()),
            static_cast<const float*>(v_sliced->data()), static_cast<float*>(attn_out->data()),
            seq_len, total_len, num_heads, num_kv_heads, head_dim, mask_data, causal);
#else
        attn_out = ops::scaled_dot_product_attention_2d_gqa(q, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, num_kv_heads,
                                                            head_dim, mask, causal);
#endif
        break;
    }
    case AttentionPath::CUDA_FP32_DECODE: {
        TensorPtr k_sliced, v_sliced;
        get_kv_on_device(k_sliced, v_sliced);
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
    case AttentionPath::CUDA_PAGED_Q4_0_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            const int32_t* d_page_ids = storage_->upload_seq_page_table(layer_idx, seq_id);
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
            const int32_t* d_page_ids = storage_->upload_seq_page_table(layer_idx, seq_id);
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
            const int32_t* d_page_ids = storage_->upload_seq_page_table(layer_idx, seq_id);
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

    case AttentionPath::CUDA_PAGED_FP8_E4M3_DECODE:
#ifdef USE_CUDA
        attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, DeviceType::CUDA);
        {
            const int32_t* d_page_ids = storage_->upload_seq_page_table(layer_idx, seq_id);
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
            const int32_t* d_page_ids = storage_->upload_seq_page_table(layer_idx, seq_id);
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
