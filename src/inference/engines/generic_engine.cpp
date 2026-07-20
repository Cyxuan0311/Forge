#include "forge/engines/generic_engine.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/model.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"
#include "forge/quant_traits.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {

// ============================================================================
// File-local RoPE helpers (identical to LlamaEngine / FalconEngine versions)
// ============================================================================

static void apply_rope_neox_cpu(const float* q_data, const float* k_data, float* q_out,
                                float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                int head_dim, int64_t start_pos, float theta) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = pos * freq;
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

static void apply_rope_mrope_cpu(const float* q_data, const float* k_data, float* q_out,
                                 float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                 int head_dim, int n_rot, int64_t start_pos, float theta) {
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

// Proportional RoPE: same as NeoX but with per-dimension frequency scaling factors.
static void apply_rope_proportional_cpu(const float* q_data, const float* k_data, float* q_out,
                                        float* k_out, int seq_len, int num_heads,
                                        int num_kv_heads, int head_dim, int64_t start_pos,
                                        float theta, const float* freq_factors) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float base_freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float freq = freq_factors ? base_freq / freq_factors[d] : base_freq;
                float angle = pos * freq;
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

// ============================================================================
// GenericEngine implementation
// ============================================================================

GenericEngine::GenericEngine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx) {
    if (!init_weights()) {
        throw std::runtime_error("GenericEngine: failed to initialize weights");
    }
}

bool GenericEngine::init_weights() {
    if (!weights_.init(model_.weights(), model_.config())) {
        return false;
    }
    // Load proportional RoPE frequency factors if present
    const auto& cfg = model_.config();
    if (cfg.rope_type == RopeType::Proportional) {
        rope_freqs_ = model_.weights().get("rope_freqs");
        if (!rope_freqs_) {
            for (int i = 0; i < cfg.num_layers; ++i) {
                rope_freqs_ = weights_.layers[i].rope_freqs();
                if (rope_freqs_) break;
            }
        }
        if (rope_freqs_ && is_quantized_type(rope_freqs_->dtype())) {
            rope_freqs_ = ops::dequantize_weight(rope_freqs_);
        }
    }
    return true;
}

// ============================================================================
// forward() — override to handle embedding scaling and logit softcapping
// ============================================================================

TensorPtr GenericEngine::forward(const TensorPtr& input_ids, int64_t start_pos, int seq_id) {
    const auto& cfg = model_.config();
    int seq_len = static_cast<int>(input_ids->numel());

    // Decode path: use fewer threads (memory-bandwidth bound)
#ifdef _OPENMP
    omp_set_num_threads(ctx_.params().n_threads);
#endif

    init_kv_cache(cfg);

    DeviceType first_dev = layer_device(0);
    auto ids_on_dev = transfer_hidden(input_ids, first_dev);

    auto token_emb = model_.weights().get("token_embedding");
    TensorPtr hidden;
    {
        PERF_SCOPE("forward/embedding");
        hidden = ops::embedding(token_emb, ids_on_dev, weights_.token_embedding_fp32);
    }

    // Embedding scaling (Gemma/Gemma2/Gemma4: hidden *= sqrt(n_embd))
    // Check ArchCapability for the current architecture
    auto cap = ArchCapabilityRegistry::instance().get(cfg.arch_type);
    if (cap.embedding_scale) {
        PERF_SCOPE("forward/emb_scale");
        float scale = std::sqrt(static_cast<float>(cfg.hidden_dim));
        bool was_cuda = (hidden->device() == DeviceType::CUDA);
        if (was_cuda) {
            auto hidden_cpu =
                std::make_shared<Tensor>(DataType::FP32, hidden->shape(), DeviceType::CPU);
            hidden_cpu->copy_from(*hidden);
            hidden = hidden_cpu;
        }
        int n = static_cast<int>(hidden->numel());
        float* data = static_cast<float*>(hidden->data());
        for (int i = 0; i < n; ++i) {
            data[i] *= scale;
        }
        if (was_cuda) {
            auto hidden_back =
                std::make_shared<Tensor>(DataType::FP32, hidden->shape(), DeviceType::CUDA);
            hidden_back->copy_from(*hidden);
            hidden = hidden_back;
        }
    }

    auto logits = forward_layers(hidden, seq_len, start_pos);

    // Logit softcapping (Gemma2/Gemma4)
    if (cfg.f_final_logit_softcapping > 0.0f && logits) {
        PERF_SCOPE("forward/logit_softcap");
        DeviceType logits_dev = logits->device();
        // Need CPU access for scalar loop
        if (logits_dev == DeviceType::CUDA) {
            auto logits_cpu =
                std::make_shared<Tensor>(DataType::FP32, logits->shape(), DeviceType::CPU);
            logits_cpu->copy_from(*logits);
            logits = logits_cpu;
        }
        float cap_val = cfg.f_final_logit_softcapping;
        int n = static_cast<int>(logits->numel());
        float* data = static_cast<float*>(logits->data());
        for (int i = 0; i < n; ++i) {
            data[i] = std::tanh(data[i] / cap_val) * cap_val;
        }
        if (logits_dev == DeviceType::CUDA) {
            auto logits_back =
                std::make_shared<Tensor>(DataType::FP32, logits->shape(), DeviceType::CUDA);
            logits_back->copy_from(*logits);
            logits = logits_back;
        }
    }

    return logits;
}

// ============================================================================
// 1. norm_forward
// ============================================================================

TensorPtr GenericEngine::norm_forward(const TensorPtr& x, const TensorPtr& weight,
                                      const TensorPtr& bias, NormType type, float eps,
                                      DeviceType dev) {
    (void)dev;
    if (type == NormType::LayerNorm) {
        return ops::layer_norm(x, weight, bias, eps);
    }
    return ops::rms_norm(x, weight, eps);
}

// ============================================================================
// 2. qkv_proj_forward
// ============================================================================

GenericEngine::QKVProjResult GenericEngine::qkv_proj_forward(const TensorPtr& x, int layer_idx,
                                                             bool has_bias, DeviceType dev,
                                                             int seq_len) {
    const auto& lw = weights_.layers[layer_idx];
    auto wq = lw.wq();
    auto wk = lw.wk();
    auto wv = lw.wv();
    auto bq = has_bias ? lw.bq() : nullptr;
    auto bk = has_bias ? lw.bk() : nullptr;
    auto bv = has_bias ? lw.bv() : nullptr;

    QKVProjResult result;

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
            static_cast<const float*>(x->data()), wq->data(), N_q, wk->data(), N_k, wv->data(),
            N_v, static_cast<float*>(result.q->data()), static_cast<float*>(result.k->data()),
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
        // CUDA not available, fall through to generic path
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
        result.q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q},
                                            DeviceType::CPU);
        result.k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k},
                                            DeviceType::CPU);
        result.v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v},
                                            DeviceType::CPU);
        float* src = static_cast<float*>(qkv->data());
        std::memcpy(result.q->data(), src, N_q * sizeof(float));
        std::memcpy(result.k->data(), src + N_q, N_k * sizeof(float));
        std::memcpy(result.v->data(), src + N_q + N_k, N_v * sizeof(float));
        if (bq && bq->numel() > 0) {
            float* qd = static_cast<float*>(result.q->data());
            const float* bd = static_cast<const float*>(bq->data());
            for (int i = 0; i < N_q; ++i) qd[i] += bd[i];
        }
        if (bk && bk->numel() > 0) {
            float* kd = static_cast<float*>(result.k->data());
            const float* bd = static_cast<const float*>(bk->data());
            for (int i = 0; i < N_k; ++i) kd[i] += bd[i];
        }
        if (bv && bv->numel() > 0) {
            float* vd = static_cast<float*>(result.v->data());
            const float* bd = static_cast<const float*>(bv->data());
            for (int i = 0; i < N_v; ++i) vd[i] += bd[i];
        }
    }
    // CPU Q4_K fused path
    else if (dev == DeviceType::CPU && seq_len == 1 && wq->dtype() == DataType::Q4_K &&
             wk->dtype() == DataType::Q4_K && wv->dtype() == DataType::Q4_K) {
        int N_q = static_cast<int>(wq->shape()[0]);
        int N_k = static_cast<int>(wk->shape()[0]);
        int N_v = static_cast<int>(wv->shape()[0]);
        auto qkv = ops::matmul_transB_fused_qkv_q4_k(x, wq, wk, wv);
        result.q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_q},
                                            DeviceType::CPU);
        result.k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_k},
                                            DeviceType::CPU);
        result.v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N_v},
                                            DeviceType::CPU);
        float* src = static_cast<float*>(qkv->data());
        std::memcpy(result.q->data(), src, N_q * sizeof(float));
        std::memcpy(result.k->data(), src + N_q, N_k * sizeof(float));
        std::memcpy(result.v->data(), src + N_q + N_k, N_v * sizeof(float));
        if (bq && bq->numel() > 0) {
            float* qd = static_cast<float*>(result.q->data());
            const float* bd = static_cast<const float*>(bq->data());
            for (int i = 0; i < N_q; ++i) qd[i] += bd[i];
        }
        if (bk && bk->numel() > 0) {
            float* kd = static_cast<float*>(result.k->data());
            const float* bd = static_cast<const float*>(bk->data());
            for (int i = 0; i < N_k; ++i) kd[i] += bd[i];
        }
        if (bv && bv->numel() > 0) {
            float* vd = static_cast<float*>(result.v->data());
            const float* bd = static_cast<const float*>(bv->data());
            for (int i = 0; i < N_v; ++i) vd[i] += bd[i];
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

// ============================================================================
// 3. rope_forward
// ============================================================================

GenericEngine::RopeResult GenericEngine::rope_forward(const TensorPtr& q, const TensorPtr& k,
                                                      int layer_idx, int64_t start_pos,
                                                      int seq_len, DeviceType dev) {
    const auto& cfg = model_.config();
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;

    // Determine RopeType
    RopeType rope_type = cfg.rope_type;
    if (cfg.use_mrope) {
        rope_type = RopeType::MRoPE;
    } else if (cfg.use_neox_rope) {
        rope_type = RopeType::NeoX;
    }

    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), dev);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), dev);

    int n_rot = cfg.use_mrope ? cfg.rope_dimension_count : head_dim;
    if (n_rot <= 0) n_rot = head_dim;

    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        // For MRoPE with partial rotation or when QK-norm forced Q/K to CPU,
        // fall back to CPU RoPE
        bool use_cpu_rope = (rope_type == RopeType::MRoPE && n_rot < head_dim) ||
                            q->device() != DeviceType::CUDA || k->device() != DeviceType::CUDA;

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
            auto q_rope_cpu =
                std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
            auto k_rope_cpu =
                std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);

            if (rope_type == RopeType::MRoPE) {
                apply_rope_mrope_cpu(
                    static_cast<const float*>(q_cpu->data()),
                    static_cast<const float*>(k_cpu->data()),
                    static_cast<float*>(q_rope_cpu->data()),
                    static_cast<float*>(k_rope_cpu->data()), seq_len, num_heads, num_kv_heads,
                    head_dim, n_rot, start_pos, cfg.rope_theta);
            } else if (rope_type == RopeType::NeoX) {
                apply_rope_neox_cpu(
                    static_cast<const float*>(q_cpu->data()),
                    static_cast<const float*>(k_cpu->data()),
                    static_cast<float*>(q_rope_cpu->data()),
                    static_cast<float*>(k_rope_cpu->data()), seq_len, num_heads, num_kv_heads,
                    head_dim, start_pos, cfg.rope_theta);
            } else if (rope_type == RopeType::Proportional) {
                const float* freq_factors = nullptr;
                if (rope_freqs_) {
                    if (rope_freqs_->device() == DeviceType::CUDA) {
                        rope_freqs_cpu_ = std::make_shared<Tensor>(DataType::FP32,
                                                                   rope_freqs_->shape(),
                                                                   DeviceType::CPU);
                        rope_freqs_cpu_->copy_from(*rope_freqs_);
                    } else {
                        rope_freqs_cpu_ = rope_freqs_;
                    }
                    freq_factors = static_cast<const float*>(rope_freqs_cpu_->data());
                }
                apply_rope_proportional_cpu(
                    static_cast<const float*>(q_cpu->data()),
                    static_cast<const float*>(k_cpu->data()),
                    static_cast<float*>(q_rope_cpu->data()),
                    static_cast<float*>(k_rope_cpu->data()), seq_len, num_heads, num_kv_heads,
                    head_dim, start_pos, cfg.rope_theta, freq_factors);
            } else if (rope_type == RopeType::None) {
                q_rope_cpu->copy_from(*q_cpu);
                k_rope_cpu->copy_from(*k_cpu);
            } else {
                apply_rope_standard(
                    static_cast<const float*>(q_cpu->data()),
                    static_cast<const float*>(k_cpu->data()),
                    static_cast<float*>(q_rope_cpu->data()),
                    static_cast<float*>(k_rope_cpu->data()), seq_len, num_heads, num_kv_heads,
                    head_dim, start_pos, cfg.rope_theta);
            }
            q_rope->copy_from(*q_rope_cpu);
            k_rope->copy_from(*k_rope_cpu);
        } else {
            // CUDA kernel path (standard/NeoX — same kernel handles both)
            // Note: Proportional RoPE on CUDA falls through here only if no
            // freq_factors are needed; otherwise use_cpu_rope would be true.
            cuda::launch_rope_gqa(
                static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                num_heads, num_kv_heads, head_dim, seq_len, start_pos, cfg.rope_theta);
        }
#else
        // CUDA not compiled — use CPU fallback
        (void)layer_idx;
        // Should not reach here when CUDA is not available, but handle gracefully
        apply_rope_standard(
            static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
            static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), seq_len,
            num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta);
#endif
    } else {
        // CPU path
        switch (rope_type) {
            case RopeType::None:
                q_rope->copy_from(*q);
                k_rope->copy_from(*k);
                break;
            case RopeType::NeoX:
                apply_rope_neox_cpu(
                    static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                    static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                    seq_len, num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta);
                break;
            case RopeType::MRoPE:
                apply_rope_mrope_cpu(
                    static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                    static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                    seq_len, num_heads, num_kv_heads, head_dim, n_rot, start_pos, cfg.rope_theta);
                break;
            case RopeType::Proportional: {
                const float* freq_factors = nullptr;
                if (rope_freqs_) {
                    if (rope_freqs_->device() == DeviceType::CUDA) {
                        rope_freqs_cpu_ =
                            std::make_shared<Tensor>(DataType::FP32, rope_freqs_->shape(),
                                                     DeviceType::CPU);
                        rope_freqs_cpu_->copy_from(*rope_freqs_);
                    } else {
                        rope_freqs_cpu_ = rope_freqs_;
                    }
                    freq_factors = static_cast<const float*>(rope_freqs_cpu_->data());
                }
                apply_rope_proportional_cpu(
                    static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                    static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                    seq_len, num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta,
                    freq_factors);
                break;
            }
            default:
                // Standard, LinearScaling, NTK_Scaled all use the base class method
                apply_rope_standard(
                    static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                    static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                    seq_len, num_heads, num_kv_heads, head_dim, start_pos, cfg.rope_theta);
                break;
        }
    }

    // Apply Q scaling after RoPE (Gemma/Gemma2/Gemma4 pattern: rope_q_scale > 0)
    if (cfg.rope_q_scale > 0.0f) {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        // Ensure on CPU for scalar loop
        bool was_cuda = (q_rope->device() == DeviceType::CUDA);
        if (was_cuda) {
            auto q_cpu = std::make_shared<Tensor>(DataType::FP32, q_rope->shape(), DeviceType::CPU);
            q_cpu->copy_from(*q_rope);
            q_rope = q_cpu;
        }
        int n = static_cast<int>(q_rope->numel());
        float* data = static_cast<float*>(q_rope->data());
        for (int i = 0; i < n; ++i) {
            data[i] *= scale;
        }
        if (was_cuda) {
            auto q_back =
                std::make_shared<Tensor>(DataType::FP32, q_rope->shape(), DeviceType::CUDA);
            q_back->copy_from(*q_rope);
            q_rope = q_back;
        }
    }

    return {q_rope, k_rope};
}

// ============================================================================
// 4. attention_forward
// ============================================================================

TensorPtr GenericEngine::attention_forward(const TensorPtr& q, const TensorPtr& k,
                                           const TensorPtr& v, int layer_idx,
                                           int64_t start_pos, int seq_len, DeviceType dev,
                                           const TensorPtr& mask) {
    (void)start_pos;
    (void)v;  // v is already in KV cache; we use k_sliced/v_sliced from cache
    const auto& cfg = model_.config();
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;

    int total_len = kv_cache_.filled(layer_idx);
    auto k_sliced = kv_cache_.get_key_filled(layer_idx);
    auto v_sliced = kv_cache_.get_value_filled(layer_idx);

    // Ensure KV cache data is on the correct device
    if (dev == DeviceType::CUDA && k_sliced->device() == DeviceType::CPU) {
        auto k_cuda =
            std::make_shared<Tensor>(DataType::FP32, k_sliced->shape(), DeviceType::CUDA);
        k_cuda->copy_from(*k_sliced);
        k_sliced = k_cuda;

        auto v_cuda =
            std::make_shared<Tensor>(DataType::FP32, v_sliced->shape(), DeviceType::CUDA);
        v_cuda->copy_from(*v_sliced);
        v_sliced = v_cuda;
    }

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

    // For decode with mask, extract the single row for this sequence
    const float* mask_row = nullptr;
    TensorPtr mask_row_tensor;
    if (mask && seq_len == 1 && mask->ndim() == 2) {
        // mask is [n_seqs, kv_len] — extract the first (and only) row
        // When called per-sequence from forward_batch, the mask has already been sliced
        mask_row = mask_data;
    }

    TensorPtr attn_out;
    if (dev == DeviceType::CUDA && num_kv_heads < num_heads) {
        attn_out = std::make_shared<Tensor>(DataType::FP32,
                                            std::vector<int64_t>{seq_len, num_heads * head_dim},
                                            DeviceType::CUDA);
        if (seq_len == 1) {
#ifdef USE_CUDA
            // Fused path: read quantized KV cache directly (no dequantize_layer needed)
            const auto& kv_cfg = kv_cache_.kv_config();
            void* d_q_K = kv_cache_.d_q_key_cache(layer_idx);
            void* d_q_V = kv_cache_.d_q_value_cache(layer_idx);

            if (d_q_K && d_q_V && kv_cfg.type_k == kv_cfg.type_v) {
                if (kv_cfg.type_k == KVCacheDType::Q4_0) {
                    size_t q_row_size =
                        KVCache::block_nbytes(KVCacheDType::Q4_0, num_kv_heads * head_dim);
                    cuda::launch_fused_flash_attention_gqa_decode_q4_0(
                        static_cast<const float*>(q->data()), d_q_K, d_q_V,
                        static_cast<float*>(attn_out->data()),
                        total_len, num_heads, num_kv_heads, head_dim,
                        q_row_size, mask_row, 0);
                } else if (kv_cfg.type_k == KVCacheDType::F16) {
                    size_t q_row_size =
                        KVCache::block_nbytes(KVCacheDType::F16, num_kv_heads * head_dim);
                    cuda::launch_fused_flash_attention_gqa_decode_f16(
                        static_cast<const float*>(q->data()), d_q_K, d_q_V,
                        static_cast<float*>(attn_out->data()),
                        total_len, num_heads, num_kv_heads, head_dim,
                        q_row_size, mask_row, 0);
                } else if (kv_cfg.type_k == KVCacheDType::Q8_0) {
                    size_t q_row_size =
                        KVCache::block_nbytes(KVCacheDType::Q8_0, num_kv_heads * head_dim);
                    cuda::launch_fused_flash_attention_gqa_decode_q8_0(
                        static_cast<const float*>(q->data()), d_q_K, d_q_V,
                        static_cast<float*>(attn_out->data()),
                        total_len, num_heads, num_kv_heads, head_dim,
                        q_row_size, mask_row, 0);
                } else {
                    // Unsupported symmetric type — fallback
                    cuda::launch_flash_attention_gqa_decode(
                        static_cast<const float*>(q->data()),
                        static_cast<const float*>(k_sliced->data()),
                        static_cast<const float*>(v_sliced->data()),
                        static_cast<float*>(attn_out->data()),
                        total_len, num_heads, num_kv_heads, head_dim,
                        mask_row, 0);
                }
            } else {
                // Fallback: FP32 attention kernel with dequantized KV
                cuda::launch_flash_attention_gqa_decode(
                    static_cast<const float*>(q->data()),
                    static_cast<const float*>(k_sliced->data()),
                    static_cast<const float*>(v_sliced->data()),
                    static_cast<float*>(attn_out->data()),
                    total_len, num_heads, num_kv_heads, head_dim,
                    mask_row, 0);
            }
#endif
        } else {
#ifdef USE_CUDA
            cuda::launch_flash_attention_gqa(
                static_cast<const float*>(q->data()), static_cast<const float*>(k_sliced->data()),
                static_cast<const float*>(v_sliced->data()), static_cast<float*>(attn_out->data()),
                seq_len, total_len, num_heads, num_kv_heads, head_dim,
                mask_data,  // prefill mask
                true);
#endif
        }
    } else if (dev == DeviceType::CUDA) {
        attn_out = ops::scaled_dot_product_attention_2d(q, k_sliced, v_sliced, seq_len, total_len,
                                                        num_heads, head_dim, mask, true);
    } else {
        // CPU path
        if (num_kv_heads < num_heads) {
            attn_out = ops::scaled_dot_product_attention_2d_gqa(q, k_sliced, v_sliced, seq_len,
                                                                total_len, num_heads, num_kv_heads,
                                                                head_dim, mask, true);
        } else {
            attn_out = ops::scaled_dot_product_attention_2d(q, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, head_dim, mask, true);
        }
    }

    return attn_out;
}

// ============================================================================
// 5. ffn_forward
// ============================================================================

TensorPtr GenericEngine::ffn_forward(const TensorPtr& x, const TensorPtr& residual,
                                     int layer_idx, int seq_len, DeviceType dev) {
    const auto& cfg = model_.config();
    const auto& lw = weights_.layers[layer_idx];

    switch (cfg.ffn_type) {
        case FFNType::SiLUGated: {
            TensorPtr ffn_mid;
            {
                PERF_SCOPE("layer/ffn_up");
                // Fused CUDA Q4_0 gate+up
                if (dev == DeviceType::CUDA && lw.w1()->dtype() == DataType::Q4_0 &&
                    lw.w3()->dtype() == DataType::Q4_0) {
                    ffn_mid = ops::ffn_up_fused(x, lw.w1(), lw.w3(), cfg.intermediate_dim);
                }
                // Fused CPU Q4_0 gate+up
                else if (dev == DeviceType::CPU && seq_len == 1 && lw.w1() && lw.w3() &&
                         lw.w1()->dtype() == DataType::Q4_0 && lw.w3()->dtype() == DataType::Q4_0) {
                    ffn_mid = ops::matmul_transB_fused_ffn_up_q4_0(x, lw.w1(), lw.w3());
                }
                // Fused CPU Q4_K gate+up
                else if (dev == DeviceType::CPU && seq_len == 1 && lw.w1() && lw.w3() &&
                         lw.w1()->dtype() == DataType::Q4_K && lw.w3()->dtype() == DataType::Q4_K) {
                    ffn_mid = ops::matmul_transB_fused_ffn_up_q4_k(x, lw.w1(), lw.w3());
                } else {
                    auto gate = ops::matmul_transB(x, lw.w1());
                    auto up = ops::matmul_transB(x, lw.w3());
                    ffn_mid = ops::silu_multiply(gate, up);
                }
            }

            TensorPtr ffn_out;
            {
                PERF_SCOPE("layer/ffn_down");
                // Fused down_proj + residual add for decode (M=1)
                if (dev == DeviceType::CUDA && seq_len == 1 &&
                    lw.w2()->dtype() == DataType::Q4_0) {
                    int K_down = static_cast<int>(lw.w2()->shape()[1]);
                    int N_down = static_cast<int>(lw.w2()->shape()[0]);
                    ffn_out = std::make_shared<Tensor>(DataType::FP32,
                                                       std::vector<int64_t>{1, N_down},
                                                       DeviceType::CUDA);
#ifdef USE_CUDA
                    cuda::launch_ffn_down_fused_q4_0(
                        static_cast<const float*>(ffn_mid->data()), lw.w2()->data(),
                        static_cast<const float*>(residual->data()),
                        static_cast<float*>(ffn_out->data()), K_down, N_down);
#endif
                } else if (dev == DeviceType::CPU && seq_len == 1 &&
                           lw.w2()->dtype() == DataType::Q4_0) {
                    ffn_out = ops::matmul_transB_fused_ffn_down_residual_q4_0(ffn_mid, lw.w2(),
                                                                              residual);
                } else if (dev == DeviceType::CPU && seq_len == 1 &&
                           lw.w2()->dtype() == DataType::Q6_K) {
                    ffn_out = ops::matmul_transB_fused_ffn_down_residual_q6_k(ffn_mid, lw.w2(),
                                                                              residual);
                } else {
                    ffn_out = ops::matmul_transB(ffn_mid, lw.w2());
                }
            }
            return ffn_out;
        }

        case FFNType::GeGLU: {
            auto gate = ops::matmul_transB(x, lw.w1());
            auto up = ops::matmul_transB(x, lw.w3());
            auto gated = ops::gelu_multiply(gate, up);
            return ops::matmul_transB(gated, lw.w2());
        }

        case FFNType::SimpleGELU: {
            auto up = ops::matmul_transB(x, lw.w3());
            auto activated = ops::gelu(up);
            return ops::matmul_transB(activated, lw.w2());
        }

        case FFNType::MoE:
            // MoE is not handled by GenericEngine; should not reach here
            // for the architectures GenericEngine serves (Llama/Gemma/Gemma2/Falcon)
            throw std::runtime_error("GenericEngine: MoE FFN not supported; use Gemma4Engine");

        default:
            throw std::runtime_error("GenericEngine: unknown FFN type");
    }
}

// ============================================================================
// QK-Norm helper
// ============================================================================

TensorPtr GenericEngine::qk_norm_forward(const TensorPtr& x, const TensorPtr& norm_weight,
                                         int num_heads, int head_dim, DeviceType dev) {
    const auto& cfg = model_.config();
    int seq_len = static_cast<int>(x->shape()[0]);
    int rows = seq_len * num_heads;

    if (dev == DeviceType::CUDA && x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        const float* w_ptr = nullptr;
        TensorPtr w_cuda_tmp;
        if (norm_weight->dtype() == DataType::FP32) {
            if (norm_weight->device() == DeviceType::CUDA) {
                w_ptr = static_cast<const float*>(norm_weight->data());
            } else {
                w_cuda_tmp = std::make_shared<Tensor>(DataType::FP32, norm_weight->shape(),
                                                      DeviceType::CUDA);
                w_cuda_tmp->copy_from(*norm_weight);
                w_ptr = static_cast<const float*>(w_cuda_tmp->data());
            }
        }
        if (w_ptr) {
            auto x_out = std::make_shared<Tensor>(DataType::FP32, x->shape(), DeviceType::CUDA);
            cuda::launch_rms_norm(static_cast<const float*>(x->data()), w_ptr,
                                  static_cast<float*>(x_out->data()), rows, head_dim,
                                  cfg.rms_norm_eps);
            return x_out;
        }
#endif
    }

    // CPU path
    auto x_cpu = x;
    if (x_cpu->device() != DeviceType::CPU) {
        x_cpu = std::make_shared<Tensor>(DataType::FP32, x->shape(), DeviceType::CPU);
        x_cpu->copy_from(*x);
    }

    // Create mutable copy for in-place norm
    auto result = std::make_shared<Tensor>(DataType::FP32, x_cpu->shape(), DeviceType::CPU);
    std::memcpy(result->data(), x_cpu->data(), result->nbytes());

    float* data = static_cast<float*>(result->data());
    std::vector<float> nw(head_dim);
    if (norm_weight->dtype() == DataType::FP32) {
        if (norm_weight->device() == DeviceType::CUDA) {
            auto tmp = std::make_shared<Tensor>(DataType::FP32, norm_weight->shape(),
                                                DeviceType::CPU);
            tmp->copy_from(*norm_weight);
            std::memcpy(nw.data(), tmp->data(), head_dim * sizeof(float));
        } else {
            std::memcpy(nw.data(), norm_weight->data(), head_dim * sizeof(float));
        }
    } else {
        std::fill(nw.begin(), nw.end(), 1.0f);
    }

#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_heads > 4)
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            float* head_ptr = data + s * num_heads * head_dim + h * head_dim;
            float norm_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d) norm_sq += head_ptr[d] * head_ptr[d];
            float inv_rms = 1.0f / std::sqrt(norm_sq / head_dim + cfg.rms_norm_eps);
            for (int d = 0; d < head_dim; ++d) head_ptr[d] *= inv_rms * nw[d];
        }
    }

    return result;
}

// ============================================================================
// forward_layer — main orchestration
// ============================================================================

TensorPtr GenericEngine::forward_layer(const TensorPtr& hidden, int layer_idx, int seq_len,
                                       int64_t start_pos, DeviceType dev, int seq_id) {
    const auto& cfg = model_.config();
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;
    const auto& lw = weights_.layers[layer_idx];

    // Determine norm type and epsilon
    NormType norm_type = cfg.norm_type;
    float eps = (norm_type == NormType::LayerNorm) ? cfg.layer_norm_eps : cfg.rms_norm_eps;

    // Determine if QKV has bias
    bool has_qkv_bias = (lw.bq() && lw.bq()->numel() > 0);

    // ---- 1. Pre-attention norm ----
    TensorPtr pre_attn_norm;
    {
        PERF_SCOPE("layer/attn_norm");
        auto norm_w = lw.attn_norm();
        auto norm_b = (norm_type == NormType::LayerNorm) ? lw.get("attn_norm_bias") : nullptr;
        pre_attn_norm = norm_forward(hidden, norm_w, norm_b, norm_type, eps, dev);
    }

    // ---- 2. QKV projection ----
    TensorPtr q, k, v;
    {
        PERF_SCOPE("layer/qkv_proj");
        auto qkv_result = qkv_proj_forward(pre_attn_norm, layer_idx, has_qkv_bias, dev, seq_len);
        q = qkv_result.q;
        k = qkv_result.k;
        v = qkv_result.v;
    }

    // ---- 3. QK-Norm (per-head RMSNorm) ----
    if (cfg.use_qk_norm) {
        PERF_SCOPE("layer/qk_norm");
        if (lw.attn_q_norm()) {
            q = qk_norm_forward(q, lw.attn_q_norm(), num_heads, head_dim, dev);
        }
        if (lw.attn_k_norm()) {
            k = qk_norm_forward(k, lw.attn_k_norm(), num_kv_heads, head_dim, dev);
        }
    }

    // ---- 4. RoPE ----
    TensorPtr q_rope, k_rope;
    {
        PERF_SCOPE("layer/rope");
        auto rope_result = rope_forward(q, k, layer_idx, start_pos, seq_len, dev);
        q_rope = rope_result.q_rope;
        k_rope = rope_result.k_rope;
    }

    // ---- 5. KV cache update ----
    {
        PERF_SCOPE("layer/kv_cache_update");
        kv_cache_.update(layer_idx, seq_id, start_pos, k_rope, v, seq_len);

        // Fused decode path: skip dequantize_layer() when the fused attention
        // kernel will read the quantized KV cache directly (Q4_0/F16/Q8_0 on CUDA, seq_len==1).
        const auto& kv_cfg = kv_cache_.kv_config();
        bool use_fused_decode =
            (dev == DeviceType::CUDA && seq_len == 1 &&
             kv_cache_.d_q_key_cache(layer_idx) != nullptr &&
             ((kv_cfg.type_k == KVCacheDType::Q4_0 && kv_cfg.type_v == KVCacheDType::Q4_0) ||
              (kv_cfg.type_k == KVCacheDType::F16 && kv_cfg.type_v == KVCacheDType::F16) ||
              (kv_cfg.type_k == KVCacheDType::Q8_0 && kv_cfg.type_v == KVCacheDType::Q8_0)));

        // For non-fused paths with quantized KV, dequantize the layer into FP32 shadow cache.
        if (!use_fused_decode && kv_cache_.kv_dtype() != KVCacheDType::FP32) {
            kv_cache_.dequantize_layer(layer_idx);
        }
    }

    // ---- 6. Attention ----
    TensorPtr attn_out;
    {
        PERF_SCOPE("layer/attention");
        attn_out = attention_forward(q_rope, k_rope, v, layer_idx, start_pos, seq_len, dev);
    }

    // ---- 7. Attention output projection ----
    TensorPtr attn_proj;
    {
        PERF_SCOPE("layer/attn_proj");
        attn_proj = ops::matmul_transB(attn_out, lw.wo());
    }

    // ---- 8. Post-attention norm (Gemma2) ----
    if (cfg.has_post_attention_norm) {
        PERF_SCOPE("layer/post_attn_norm");
        auto post_attn_norm = lw.attn_post_norm();
        if (!post_attn_norm) post_attn_norm = lw.post_attention_norm();
        if (post_attn_norm) {
            attn_proj = norm_forward(attn_proj, post_attn_norm, nullptr, NormType::RMSNorm,
                                     cfg.rms_norm_eps, dev);
        }
    }

    // ---- 9. Compute output based on residual style ----
    TensorPtr output;

    if (cfg.use_parallel_residual) {
        // Falcon-style: ffn from the same norm, parallel add
        // Determine FFN input: use attn_norm_2 if present (Falcon-40B), else pre_attn_norm
        TensorPtr ffn_in = pre_attn_norm;
        auto attn_norm_2_w = lw.get("attn_norm_2");
        if (attn_norm_2_w) {
            PERF_SCOPE("layer/attn_norm_2");
            auto attn_norm_2_b = lw.get("attn_norm_2_bias");
            ffn_in = norm_forward(pre_attn_norm, attn_norm_2_w, attn_norm_2_b, NormType::LayerNorm,
                                  cfg.layer_norm_eps, dev);
        }

        TensorPtr ffn_out;
        {
            PERF_SCOPE("layer/ffn");
            ffn_out = ffn_forward(ffn_in, hidden, layer_idx, seq_len, dev);
        }

        // Parallel residual: attn_proj + ffn_out + hidden
        {
            PERF_SCOPE("layer/residual");
            output = ops::add(attn_proj, ffn_out);
            output = ops::add(output, hidden);
        }
    } else {
        // Sequential: norm→attn→residual→norm→ffn→residual
        auto hidden_after_attn = ops::add(hidden, attn_proj);

        TensorPtr ffn_normed;
        {
            PERF_SCOPE("layer/ffn_norm");
            ffn_normed = norm_forward(hidden_after_attn, lw.ffn_norm(), nullptr, norm_type, eps,
                                      dev);
        }

        TensorPtr ffn_out;
        {
            PERF_SCOPE("layer/ffn");
            ffn_out = ffn_forward(ffn_normed, hidden_after_attn, layer_idx, seq_len, dev);
        }

        // Post-FFN norm (Gemma2)
        if (cfg.has_post_ffn_norm) {
            PERF_SCOPE("layer/post_ffn_norm");
            auto post_ffn_norm = lw.post_ffn_norm();
            if (post_ffn_norm) {
                ffn_out = norm_forward(ffn_out, post_ffn_norm, nullptr, NormType::RMSNorm,
                                       cfg.rms_norm_eps, dev);
            }
        }

        // Only skip residual add if a fused SiLUGated kernel already added it
        bool fused_residual =
            (cfg.ffn_type == FFNType::SiLUGated) &&
            ((dev == DeviceType::CUDA && seq_len == 1 && lw.w2()->dtype() == DataType::Q4_0) ||
             (dev == DeviceType::CPU && seq_len == 1 &&
              (lw.w2()->dtype() == DataType::Q4_0 || lw.w2()->dtype() == DataType::Q6_K)));

        if (fused_residual) {
            output = ffn_out;  // residual already added in fused kernel
        } else {
            output = ops::add(hidden_after_attn, ffn_out);
        }
    }

    return output;
}

// ============================================================================
// init_kv_cache override (standard — same as base class)
// ============================================================================

void GenericEngine::init_kv_cache(const ModelConfig& cfg) {
    // Use the base class implementation (no per-layer dims needed for
    // Llama/Gemma/Gemma2/Falcon architectures)
    TransformerEngine::init_kv_cache(cfg);
}

}  // namespace forge
