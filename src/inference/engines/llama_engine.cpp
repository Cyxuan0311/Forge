#include "forge/engines/llama_engine.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

LlamaEngine::LlamaEngine(Model& model, InferenceContext& ctx) : TransformerEngine(model, ctx) {
    if (!init_weights()) {
        throw std::runtime_error("LlamaEngine: failed to initialize weights");
    }
}

bool LlamaEngine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

TensorPtr LlamaEngine::forward_layer(const TensorPtr& hidden, int layer_idx, int seq_len,
                                     int64_t start_pos, DeviceType dev) {
    const auto& cfg = model_.config();
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;
    const auto& lw = weights_.layers[layer_idx];

    TensorPtr normed;
    {
        PERF_SCOPE("layer/attn_norm");
        normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);
    }

    TensorPtr q, k, v;
    {
        PERF_SCOPE("layer/qkv_proj");
        if (dev == DeviceType::CUDA && seq_len == 1 && lw.wq()->dtype() == DataType::Q4_0 &&
            lw.wk()->dtype() == DataType::Q4_0 && lw.wv()->dtype() == DataType::Q4_0) {
#ifdef USE_CUDA
            int K_proj = static_cast<int>(lw.wq()->shape()[1]);
            int N_q = static_cast<int>(lw.wq()->shape()[0]);
            int N_k = static_cast<int>(lw.wk()->shape()[0]);
            int N_v = static_cast<int>(lw.wv()->shape()[0]);

            q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q},
                                         DeviceType::CUDA);
            k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k},
                                         DeviceType::CUDA);
            v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v},
                                         DeviceType::CUDA);

            cuda::launch_qkv_fused_q4_0(
                static_cast<const float*>(normed->data()), lw.wq()->data(), N_q, lw.wk()->data(),
                N_k, lw.wv()->data(), N_v, static_cast<float*>(q->data()),
                static_cast<float*>(k->data()), static_cast<float*>(v->data()), K_proj);

            // Add bias terms (fused kernel only computes GEMV, not bias)
            if (lw.bq() && lw.bq()->numel() > 0) {
                cuda::launch_add_bias(static_cast<const float*>(q->data()),
                                      static_cast<const float*>(lw.bq()->data()),
                                      static_cast<float*>(q->data()), N_q);
            }
            if (lw.bk() && lw.bk()->numel() > 0) {
                cuda::launch_add_bias(static_cast<const float*>(k->data()),
                                      static_cast<const float*>(lw.bk()->data()),
                                      static_cast<float*>(k->data()), N_k);
            }
            if (lw.bv() && lw.bv()->numel() > 0) {
                cuda::launch_add_bias(static_cast<const float*>(v->data()),
                                      static_cast<const float*>(lw.bv()->data()),
                                      static_cast<float*>(v->data()), N_v);
            }
#endif
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.wq()->dtype() == DataType::Q4_0 &&
                   lw.wk()->dtype() == DataType::Q4_0 && lw.wv()->dtype() == DataType::Q4_0) {
            int N_q = static_cast<int>(lw.wq()->shape()[0]);
            int N_k = static_cast<int>(lw.wk()->shape()[0]);
            int N_v = static_cast<int>(lw.wv()->shape()[0]);
            auto qkv = ops::matmul_transB_fused_qkv_q4_0(normed, lw.wq(), lw.wk(), lw.wv());
            q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q},
                                         DeviceType::CPU);
            k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k},
                                         DeviceType::CPU);
            v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v},
                                         DeviceType::CPU);
            float* src = static_cast<float*>(qkv->data());
            std::memcpy(q->data(), src, N_q * sizeof(float));
            std::memcpy(k->data(), src + N_q, N_k * sizeof(float));
            std::memcpy(v->data(), src + N_q + N_k, N_v * sizeof(float));
            if (lw.bq() && lw.bq()->numel() > 0) {
                float* qd = static_cast<float*>(q->data());
                const float* bd = static_cast<const float*>(lw.bq()->data());
                for (int i = 0; i < N_q; ++i)
                    qd[i] += bd[i];
            }
            if (lw.bk() && lw.bk()->numel() > 0) {
                float* kd = static_cast<float*>(k->data());
                const float* bd = static_cast<const float*>(lw.bk()->data());
                for (int i = 0; i < N_k; ++i)
                    kd[i] += bd[i];
            }
            if (lw.bv() && lw.bv()->numel() > 0) {
                float* vd = static_cast<float*>(v->data());
                const float* bd = static_cast<const float*>(lw.bv()->data());
                for (int i = 0; i < N_v; ++i)
                    vd[i] += bd[i];
            }
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.wq()->dtype() == DataType::Q4_K &&
                   lw.wk()->dtype() == DataType::Q4_K && lw.wv()->dtype() == DataType::Q4_K) {
            int N_q = static_cast<int>(lw.wq()->shape()[0]);
            int N_k = static_cast<int>(lw.wk()->shape()[0]);
            int N_v = static_cast<int>(lw.wv()->shape()[0]);
            auto qkv = ops::matmul_transB_fused_qkv_q4_k(normed, lw.wq(), lw.wk(), lw.wv());
            q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q},
                                         DeviceType::CPU);
            k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k},
                                         DeviceType::CPU);
            v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v},
                                         DeviceType::CPU);
            float* src = static_cast<float*>(qkv->data());
            std::memcpy(q->data(), src, N_q * sizeof(float));
            std::memcpy(k->data(), src + N_q, N_k * sizeof(float));
            std::memcpy(v->data(), src + N_q + N_k, N_v * sizeof(float));
            if (lw.bq() && lw.bq()->numel() > 0) {
                float* qd = static_cast<float*>(q->data());
                const float* bd = static_cast<const float*>(lw.bq()->data());
                for (int i = 0; i < N_q; ++i)
                    qd[i] += bd[i];
            }
            if (lw.bk() && lw.bk()->numel() > 0) {
                float* kd = static_cast<float*>(k->data());
                const float* bd = static_cast<const float*>(lw.bk()->data());
                for (int i = 0; i < N_k; ++i)
                    kd[i] += bd[i];
            }
            if (lw.bv() && lw.bv()->numel() > 0) {
                float* vd = static_cast<float*>(v->data());
                const float* bd = static_cast<const float*>(lw.bv()->data());
                for (int i = 0; i < N_v; ++i)
                    vd[i] += bd[i];
            }
        } else {
            q = ops::matmul_transB(normed, lw.wq(), lw.bq());
            k = ops::matmul_transB(normed, lw.wk(), lw.bk());
            v = ops::matmul_transB(normed, lw.wv(), lw.bv());
        }
    }

    // Apply QK-Norm (per-head RMSNorm) if enabled
    // On CUDA: reuse launch_rms_norm (per-head norm = per-row RMSNorm with head_dim cols)
    // On CPU: per-head RMSNorm via OpenMP loop
    bool qk_norm_on_cpu = false;
    if (cfg.use_qk_norm) {
        PERF_SCOPE("layer/qk_norm");
        // Q-norm
        if (lw.attn_q_norm()) {
            int q_rows = seq_len * num_heads;
            if (dev == DeviceType::CUDA && q->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
                auto q_norm_w = lw.attn_q_norm();
                // Ensure norm weight is on CUDA
                const float* w_ptr = nullptr;
                TensorPtr w_cuda_tmp;
                if (q_norm_w->dtype() == DataType::FP32) {
                    if (q_norm_w->device() == DeviceType::CUDA) {
                        w_ptr = static_cast<const float*>(q_norm_w->data());
                    } else {
                        w_cuda_tmp = std::make_shared<Tensor>(DataType::FP32, q_norm_w->shape(),
                                                              DeviceType::CUDA);
                        w_cuda_tmp->copy_from(*q_norm_w);
                        w_ptr = static_cast<const float*>(w_cuda_tmp->data());
                    }
                }
                if (w_ptr) {
                    // In-place: each row's reads complete before writes (due to __syncthreads)
                    auto q_out = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CUDA);
                    cuda::launch_rms_norm(static_cast<const float*>(q->data()), w_ptr,
                                         static_cast<float*>(q_out->data()), q_rows, head_dim,
                                         cfg.rms_norm_eps);
                    q = q_out;
                }
#endif
            } else {
                // CPU path
                float* qd = static_cast<float*>(q->data());
                auto q_norm_w = lw.attn_q_norm();
                std::vector<float> qn_w(head_dim);
                if (q_norm_w->dtype() == DataType::FP32) {
                    if (q_norm_w->device() == DeviceType::CUDA) {
                        auto tmp = std::make_shared<Tensor>(DataType::FP32, q_norm_w->shape(),
                                                            DeviceType::CPU);
                        tmp->copy_from(*q_norm_w);
                        std::memcpy(qn_w.data(), tmp->data(), head_dim * sizeof(float));
                    } else {
                        std::memcpy(qn_w.data(), q_norm_w->data(), head_dim * sizeof(float));
                    }
                } else {
                    std::fill(qn_w.begin(), qn_w.end(), 1.0f);
                }
#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_heads > 4)
                for (int s = 0; s < seq_len; ++s) {
                    for (int h = 0; h < num_heads; ++h) {
                        float* head_ptr = qd + s * num_heads * head_dim + h * head_dim;
                        float norm_sq = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                            norm_sq += head_ptr[d] * head_ptr[d];
                        float inv_rms = 1.0f / std::sqrt(norm_sq / head_dim + cfg.rms_norm_eps);
                        for (int d = 0; d < head_dim; ++d)
                            head_ptr[d] *= inv_rms * qn_w[d];
                    }
                }
                qk_norm_on_cpu = true;
            }
        }
        // K-norm
        if (lw.attn_k_norm()) {
            int k_rows = seq_len * num_kv_heads;
            if (dev == DeviceType::CUDA && k->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
                auto k_norm_w = lw.attn_k_norm();
                const float* w_ptr = nullptr;
                TensorPtr w_cuda_tmp;
                if (k_norm_w->dtype() == DataType::FP32) {
                    if (k_norm_w->device() == DeviceType::CUDA) {
                        w_ptr = static_cast<const float*>(k_norm_w->data());
                    } else {
                        w_cuda_tmp = std::make_shared<Tensor>(DataType::FP32, k_norm_w->shape(),
                                                              DeviceType::CUDA);
                        w_cuda_tmp->copy_from(*k_norm_w);
                        w_ptr = static_cast<const float*>(w_cuda_tmp->data());
                    }
                }
                if (w_ptr) {
                    auto k_out = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CUDA);
                    cuda::launch_rms_norm(static_cast<const float*>(k->data()), w_ptr,
                                         static_cast<float*>(k_out->data()), k_rows, head_dim,
                                         cfg.rms_norm_eps);
                    k = k_out;
                }
#endif
            } else {
                // CPU path
                if (k->device() != DeviceType::CPU) {
                    auto k_cpu = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
                    k_cpu->copy_from(*k);
                    k = k_cpu;
                }
                float* kd = static_cast<float*>(k->data());
                auto k_norm_w = lw.attn_k_norm();
                std::vector<float> kn_w(head_dim);
                if (k_norm_w->dtype() == DataType::FP32) {
                    if (k_norm_w->device() == DeviceType::CUDA) {
                        auto tmp = std::make_shared<Tensor>(DataType::FP32, k_norm_w->shape(),
                                                            DeviceType::CPU);
                        tmp->copy_from(*k_norm_w);
                        std::memcpy(kn_w.data(), tmp->data(), head_dim * sizeof(float));
                    } else {
                        std::memcpy(kn_w.data(), k_norm_w->data(), head_dim * sizeof(float));
                    }
                } else {
                    std::fill(kn_w.begin(), kn_w.end(), 1.0f);
                }
#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_kv_heads > 4)
                for (int s = 0; s < seq_len; ++s) {
                    for (int h = 0; h < num_kv_heads; ++h) {
                        float* head_ptr = kd + s * num_kv_heads * head_dim + h * head_dim;
                        float norm_sq = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                            norm_sq += head_ptr[d] * head_ptr[d];
                        float inv_rms = 1.0f / std::sqrt(norm_sq / head_dim + cfg.rms_norm_eps);
                        for (int d = 0; d < head_dim; ++d)
                            head_ptr[d] *= inv_rms * kn_w[d];
                    }
                }
                qk_norm_on_cpu = true;
            }
        }
    }

    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), dev);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), dev);

    {
        PERF_SCOPE("layer/rope");
        int n_rot = cfg.use_mrope ? cfg.rope_dimension_count : head_dim;
        if (n_rot <= 0) n_rot = head_dim;
        if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
            // For text-only MRoPE where n_rot == head_dim, all position counters are
            // identical, so MRoPE reduces to standard Neox-style RoPE — use CUDA kernel.
            // CPU fallback only needed when Q/K are already on CPU (QK-Norm CPU path)
            // or when n_rot < head_dim (partial MRoPE not yet supported on CUDA).
            bool use_cpu_rope = (cfg.use_mrope && n_rot < head_dim) || qk_norm_on_cpu;
            if (use_cpu_rope) {
                // Ensure Q/K are on CPU
                auto q_cpu = (q->device() == DeviceType::CPU)
                                 ? q
                                 : [&]() {
                                       auto t = std::make_shared<Tensor>(DataType::FP32, q->shape(),
                                                                         DeviceType::CPU);
                                       t->copy_from(*q);
                                       return t;
                                   }();
                auto k_cpu = (k->device() == DeviceType::CPU)
                                 ? k
                                 : [&]() {
                                       auto t = std::make_shared<Tensor>(DataType::FP32, k->shape(),
                                                                         DeviceType::CPU);
                                       t->copy_from(*k);
                                       return t;
                                   }();
                auto q_rope_cpu = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
                auto k_rope_cpu = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
                if (cfg.use_mrope) {
                    apply_rope_mrope_cpu(
                        static_cast<const float*>(q_cpu->data()),
                        static_cast<const float*>(k_cpu->data()),
                        static_cast<float*>(q_rope_cpu->data()),
                        static_cast<float*>(k_rope_cpu->data()),
                        seq_len, num_heads, num_kv_heads, head_dim, n_rot, start_pos, cfg.rope_theta);
                } else if (cfg.use_neox_rope) {
                    apply_rope_neox_cpu(
                        static_cast<const float*>(q_cpu->data()),
                        static_cast<const float*>(k_cpu->data()),
                        static_cast<float*>(q_rope_cpu->data()),
                        static_cast<float*>(k_rope_cpu->data()),
                        seq_len, num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta);
                } else {
                    apply_rope_standard(
                        static_cast<const float*>(q_cpu->data()),
                        static_cast<const float*>(k_cpu->data()),
                        static_cast<float*>(q_rope_cpu->data()),
                        static_cast<float*>(k_rope_cpu->data()),
                        seq_len, num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta);
                }
                q_rope->copy_from(*q_rope_cpu);
                k_rope->copy_from(*k_rope_cpu);
            } else {
                cuda::launch_rope_gqa(
                    static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                    static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), num_heads,
                    num_kv_heads, head_dim, seq_len, start_pos, cfg.rope_theta);
            }
#endif
        } else {
            if (cfg.use_mrope) {
                apply_rope_mrope_cpu(
                    static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                    static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                    seq_len, num_heads, num_kv_heads, head_dim, n_rot, start_pos, cfg.rope_theta);
            } else if (cfg.use_neox_rope) {
                apply_rope_neox_cpu(
                    static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                    static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                    seq_len, num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta);
            } else {
                apply_rope_standard(
                    static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                    static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                    seq_len, num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta);
            }
        }
    }

    {
        PERF_SCOPE("layer/kv_cache_update");
        kv_cache_.update(layer_idx, k_rope, v, seq_len);

        if (kv_cache_.kv_dtype() == KVCacheDType::Q4_0) {
            kv_cache_.dequantize_layer(layer_idx);
        }
    }

    int total_len = kv_cache_.filled(layer_idx);

    TensorPtr k_sliced = kv_cache_.get_key_filled(layer_idx);
    TensorPtr v_sliced = kv_cache_.get_value_filled(layer_idx);

    if (dev == DeviceType::CUDA && k_sliced->device() == DeviceType::CPU) {
        auto k_cuda = std::make_shared<Tensor>(DataType::FP32, k_sliced->shape(), DeviceType::CUDA);
        k_cuda->copy_from(*k_sliced);
        k_sliced = k_cuda;

        auto v_cuda = std::make_shared<Tensor>(DataType::FP32, v_sliced->shape(), DeviceType::CUDA);
        v_cuda->copy_from(*v_sliced);
        v_sliced = v_cuda;
    }

    TensorPtr attn_out;
    {
        PERF_SCOPE("layer/attention");
        if (dev == DeviceType::CUDA && num_kv_heads < num_heads) {
            attn_out = std::make_shared<Tensor>(DataType::FP32,
                                                std::vector<int64_t>{seq_len, num_heads * head_dim},
                                                DeviceType::CUDA);
            if (seq_len == 1) {
                // Decode path: use optimized single-pass online softmax kernel
#ifdef USE_CUDA
                cuda::launch_flash_attention_gqa_decode(static_cast<const float*>(q_rope->data()),
                                                        static_cast<const float*>(k_sliced->data()),
                                                        static_cast<const float*>(v_sliced->data()),
                                                        static_cast<float*>(attn_out->data()),
                                                        total_len, num_heads, num_kv_heads,
                                                        head_dim);
#endif
            } else {
                // Prefill path: use generic GQA flash attention
#ifdef USE_CUDA
                cuda::launch_flash_attention_gqa(static_cast<const float*>(q_rope->data()),
                                                 static_cast<const float*>(k_sliced->data()),
                                                 static_cast<const float*>(v_sliced->data()),
                                                 static_cast<float*>(attn_out->data()), seq_len,
                                                 total_len, num_heads, num_kv_heads, head_dim,
                                                 true);
#endif
            }
        } else if (dev == DeviceType::CUDA) {
            // Non-GQA: use standard flash attention
            attn_out = ops::scaled_dot_product_attention_2d(q_rope, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, head_dim, true);
        } else {
            // CPU path: use GQA attention with direct head mapping (no KV expand)
            if (num_kv_heads < num_heads) {
                attn_out = ops::scaled_dot_product_attention_2d_gqa(q_rope, k_sliced, v_sliced,
                                                                    seq_len, total_len, num_heads,
                                                                    num_kv_heads, head_dim, true);
            } else {
                attn_out = ops::scaled_dot_product_attention_2d(
                    q_rope, k_sliced, v_sliced, seq_len, total_len, num_heads, head_dim, true);
            }
        }
    }

    TensorPtr attn_proj;
    {
        PERF_SCOPE("layer/attn_proj");
        attn_proj = ops::matmul_transB(attn_out, lw.wo());
    }

    auto hidden_after_attn = ops::add(hidden, attn_proj);

    TensorPtr ffn_normed;
    {
        PERF_SCOPE("layer/ffn_norm");
        ffn_normed = ops::rms_norm(hidden_after_attn, lw.ffn_norm(), cfg.rms_norm_eps);
    }

    TensorPtr ffn_mid;
    {
        PERF_SCOPE("layer/ffn_up");
        if (dev == DeviceType::CUDA && lw.w1()->dtype() == DataType::Q4_0 &&
            lw.w3()->dtype() == DataType::Q4_0) {
            ffn_mid = ops::ffn_up_fused(ffn_normed, lw.w1(), lw.w3(), cfg.intermediate_dim);
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.w1() && lw.w3() &&
                   lw.w1()->dtype() == DataType::Q4_0 && lw.w3()->dtype() == DataType::Q4_0) {
            // Fused gate+up projection for Q4_0 CPU decode:
            // reads input once, computes SiLU(gate) * up in a single pass
            ffn_mid = ops::matmul_transB_fused_ffn_up_q4_0(ffn_normed, lw.w1(), lw.w3());
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.w1() && lw.w3() &&
                   lw.w1()->dtype() == DataType::Q4_K && lw.w3()->dtype() == DataType::Q4_K) {
            // Fused gate+up projection for Q4_K CPU decode:
            // reads input once, produces SiLU(gate)*up directly
            ffn_mid = ops::matmul_transB_fused_ffn_up_q4_k(ffn_normed, lw.w1(), lw.w3());
        } else {
            auto gate = ops::matmul_transB(ffn_normed, lw.w1());
            auto up = ops::matmul_transB(ffn_normed, lw.w3());
            ffn_mid = ops::silu_multiply(gate, up);
        }
    }

    TensorPtr ffn_out;
    {
        PERF_SCOPE("layer/ffn_down");
        // Fused down_proj + residual add for decode (M=1)
        if (dev == DeviceType::CUDA && seq_len == 1 && lw.w2()->dtype() == DataType::Q4_0) {
            int K_down = static_cast<int>(lw.w2()->shape()[1]);
            int N_down = static_cast<int>(lw.w2()->shape()[0]);
            ffn_out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_down},
                                               DeviceType::CUDA);
#ifdef USE_CUDA
            cuda::launch_ffn_down_fused_q4_0(static_cast<const float*>(ffn_mid->data()),
                                             lw.w2()->data(),
                                             static_cast<const float*>(hidden_after_attn->data()),
                                             static_cast<float*>(ffn_out->data()), K_down, N_down);
#endif
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.w2()->dtype() == DataType::Q4_0) {
            ffn_out = ops::matmul_transB_fused_ffn_down_residual_q4_0(ffn_mid, lw.w2(),
                                                                      hidden_after_attn);
        } else if (dev == DeviceType::CPU && seq_len == 1 && lw.w2()->dtype() == DataType::Q6_K) {
            // Fused down_proj + residual for Q6_K CPU decode
            ffn_out = ops::matmul_transB_fused_ffn_down_residual_q6_k(ffn_mid, lw.w2(),
                                                                      hidden_after_attn);
        } else {
            ffn_out = ops::matmul_transB(ffn_mid, lw.w2());
        }
    }

    TensorPtr output;
    // Only skip residual add if a fused kernel already added it:
    // - Q4_0 CUDA: launch_ffn_down_fused_q4_0 adds residual
    // - Q4_0 CPU: matmul_transB_fused_ffn_down_residual_q4_0 adds residual
    // - Q6_K CPU: matmul_transB_fused_ffn_down_residual_q6_k adds residual
    // - Q6_K CUDA: NO fused kernel, generic matmul_transB without residual
    if ((dev == DeviceType::CUDA && seq_len == 1 && lw.w2()->dtype() == DataType::Q4_0) ||
        (dev == DeviceType::CPU && seq_len == 1 &&
         (lw.w2()->dtype() == DataType::Q4_0 || lw.w2()->dtype() == DataType::Q6_K))) {
        output = ffn_out;  // residual already added in fused kernel
    } else {
        output = ops::add(hidden_after_attn, ffn_out);
    }

    return output;
}

void LlamaEngine::apply_rope_neox_cpu(const float* q_data, const float* k_data, float* q_out,
                                      float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                      int head_dim, int64_t start_pos, float theta) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = (start_pos + s) * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int q_idx0 = s * q_stride + h * head_dim + d;
                int q_idx1 = q_idx0 + half_dim;

                q_out[q_idx0] = q_data[q_idx0] * cos_a - q_data[q_idx1] * sin_a;
                q_out[q_idx1] = q_data[q_idx0] * sin_a + q_data[q_idx1] * cos_a;

                if (h < num_kv_heads) {
                    int k_idx0 = s * k_stride + h * head_dim + d;
                    int k_idx1 = k_idx0 + half_dim;

                    k_out[k_idx0] = k_data[k_idx0] * cos_a - k_data[k_idx1] * sin_a;
                    k_out[k_idx1] = k_data[k_idx0] * sin_a + k_data[k_idx1] * cos_a;
                }
            }
        }
    }
}

void LlamaEngine::apply_rope_mrope_cpu(const float* q_data, const float* k_data, float* q_out,
                                       float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                       int head_dim, int n_rot, int64_t start_pos, float theta) {
    // MRoPE: apply rotary embeddings to the first n_rot dimensions only.
    // For text-only inference, all position counters advance identically,
    // so this reduces to standard RoPE with n_rot rotary dimensions.
    int half_rot = n_rot / 2;
    float theta_scale = 1.0f / theta;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;

#pragma omp parallel for schedule(static) if (seq_len > 1)
    for (int s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;

        for (int h = 0; h < num_heads; ++h) {
            const float* q_src = q_data + s * q_stride + h * head_dim;
            float* q_dst = q_out + s * q_stride + h * head_dim;

            // Apply RoPE to first n_rot dimensions
            for (int d = 0; d < half_rot; ++d) {
                float freq = std::pow(theta_scale, 2.0f * d / n_rot);
                float angle = pos * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float x0 = q_src[d];
                float x1 = q_src[d + half_rot];

                q_dst[d] = x0 * cos_a - x1 * sin_a;
                q_dst[d + half_rot] = x0 * sin_a + x1 * cos_a;
            }
            // Copy remaining dimensions unchanged
            for (int d = n_rot; d < head_dim; ++d) {
                q_dst[d] = q_src[d];
            }
        }

        for (int h = 0; h < num_kv_heads; ++h) {
            const float* k_src = k_data + s * k_stride + h * head_dim;
            float* k_dst = k_out + s * k_stride + h * head_dim;

            for (int d = 0; d < half_rot; ++d) {
                float freq = std::pow(theta_scale, 2.0f * d / n_rot);
                float angle = pos * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float x0 = k_src[d];
                float x1 = k_src[d + half_rot];

                k_dst[d] = x0 * cos_a - x1 * sin_a;
                k_dst[d + half_rot] = x0 * sin_a + x1 * cos_a;
            }
            for (int d = n_rot; d < head_dim; ++d) {
                k_dst[d] = k_src[d];
            }
        }
    }
}

// Register engines for all GQA-based architectures
namespace {
static EngineAutoRegister _reg_llama("llama", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<LlamaEngine>(model, ctx);
});
static EngineAutoRegister _reg_mistral("mistral", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<LlamaEngine>(model, ctx);
});
static EngineAutoRegister _reg_qwen("qwen", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<LlamaEngine>(model, ctx);
});
static EngineAutoRegister _reg_qwen2("qwen2", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<LlamaEngine>(model, ctx);
});
static EngineAutoRegister _reg_qwen3vl("qwen3vl", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<LlamaEngine>(model, ctx);
});
static EngineAutoRegister _reg_yi("yi", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<LlamaEngine>(model, ctx);
});
static EngineAutoRegister _reg_deepseek_gqa("deepseek", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<LlamaEngine>(model, ctx);
});
}  // namespace

}  // namespace forge
