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

    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), dev);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), dev);

    {
        PERF_SCOPE("layer/rope");
        if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
            cuda::launch_rope_gqa(
                static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()), num_heads,
                num_kv_heads, head_dim, seq_len, start_pos, cfg.rope_theta);
#endif
        } else {
            if (cfg.use_neox_rope) {
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
                // (no KV expand, Q cached in shared memory, vectorized float4 access)
                cuda::launch_flash_attention_gqa_decode(static_cast<const float*>(q_rope->data()),
                                                        static_cast<const float*>(k_sliced->data()),
                                                        static_cast<const float*>(v_sliced->data()),
                                                        static_cast<float*>(attn_out->data()),
                                                        total_len, num_heads, num_kv_heads,
                                                        head_dim);
            } else {
                // Prefill path: use generic GQA flash attention
                cuda::launch_flash_attention_gqa(static_cast<const float*>(q_rope->data()),
                                                 static_cast<const float*>(k_sliced->data()),
                                                 static_cast<const float*>(v_sliced->data()),
                                                 static_cast<float*>(attn_out->data()), seq_len,
                                                 total_len, num_heads, num_kv_heads, head_dim,
                                                 true);
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
            cuda::launch_ffn_down_fused_q4_0(static_cast<const float*>(ffn_mid->data()),
                                             lw.w2()->data(),
                                             static_cast<const float*>(hidden_after_attn->data()),
                                             static_cast<float*>(ffn_out->data()), K_down, N_down);
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
    if ((dev == DeviceType::CUDA || dev == DeviceType::CPU) && seq_len == 1 &&
        (lw.w2()->dtype() == DataType::Q4_0 || lw.w2()->dtype() == DataType::Q6_K)) {
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
static EngineAutoRegister _reg_yi("yi", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<LlamaEngine>(model, ctx);
});
static EngineAutoRegister _reg_deepseek_gqa("deepseek", [](Model& model, InferenceContext& ctx) {
    return std::make_unique<LlamaEngine>(model, ctx);
});
}  // namespace

}  // namespace forge
