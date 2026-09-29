#include "forge/engines/lfm2_engine.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "cpu/common/thread_pool.h"
#include "forge/cuda_kernels.h"
#include "forge/inference/layers/ffn_executor.h"
#include "forge/inference/layers/qwen35_linear_attention.h"  // conv1d_cpu (shared static)
#include "forge/inference/layers/qwen35_rope.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

namespace {

// Per-head RMS norm over [seq, heads, head_dim] (same math as the qwen35
// attention layer; duplicated here to keep that file's helpers private).
std::vector<float> load_head_norm_weight(const TensorPtr& w, int head_dim) {
    std::vector<float> out(head_dim, 1.0f);
    if (!w || w->dtype() != DataType::FP32)
        return out;
    auto cpu = ensure_cpu(w);
    std::memcpy(out.data(), cpu->data(), head_dim * sizeof(float));
    return out;
}

void per_head_rms_norm_cpu(float* data, const std::vector<float>& weight, int seq_len,
                           int num_heads, int head_dim, float eps) {
    cpu::parallel_for2(seq_len, num_heads, [&](int s, int h) {
        float* head_ptr = data + s * num_heads * head_dim + h * head_dim;
        float norm_sq = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            norm_sq += head_ptr[d] * head_ptr[d];
        float inv_rms = 1.0f / std::sqrt(norm_sq / head_dim + eps);
        for (int d = 0; d < head_dim; ++d)
            head_ptr[d] *= inv_rms * weight[d];
    });
}

}  // namespace

// =========================================================================
// Construction / lifecycle
// =========================================================================

Lfm2Engine::Lfm2Engine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx),
      attn_(kv_cache_, (kv_memory_ && kv_memory_->is_paged()) ? &kv_memory_->storage() : nullptr) {
    if (!init_weights()) {
        throw std::runtime_error("Lfm2Engine: failed to initialize weights");
    }
}

bool Lfm2Engine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

void Lfm2Engine::reset() {
    kv_cache_.reset();
    set_kv_cache_initialized(false);
    if (kv_memory_) {
        kv_memory_->storage().reset();
    }
    // Shortconv recurrent state is positional like the KV cache: dropping the
    // KV cache without dropping it would mix tokens across generations.
    conv_states_.clear();
}

int Lfm2Engine::conv_kernel(const ModelConfig& cfg, const LayerWeights& lw) const {
    if (cfg.shortconv_l_cache > 0)
        return cfg.shortconv_l_cache;
    auto w = lw.shortconv_conv();
    if (w && w->shape().size() == 2 && w->shape()[1] > 1)
        return static_cast<int>(w->shape()[1]);
    return 3;  // LFM2 family default (llama.cpp conv_L_cache)
}

// =========================================================================
// Forward
// =========================================================================

TensorPtr Lfm2Engine::forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;

    if (!lw.attn_norm()) {
        LOG_ERROR("Lfm2Engine: layer " + std::to_string(lctx.layer_idx) + " missing attn_norm");
        return hidden;
    }

    auto normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);

    TensorPtr attn_out;
    if (lw.layer_type == LayerType::FullAttention) {
        if (lctx.device.type == DeviceType::CUDA) {
#ifdef USE_CUDA
            attn_out = attention_cuda(normed, lctx);
#else
            attn_out = attention_cpu(normed, lctx);
#endif
        } else {
            attn_out = attention_cpu(normed, lctx);
        }
    } else {
        if (lctx.device.type == DeviceType::CUDA) {
#ifdef USE_CUDA
            attn_out = shortconv_cuda(normed, lctx);
#else
            attn_out = shortconv_cpu(normed, lctx);
#endif
        } else {
            attn_out = shortconv_cpu(normed, lctx);
        }
    }
    if (!attn_out) {
        return hidden;
    }

    auto hidden_after_attn = ops::add(hidden, attn_out);
    return apply_ffn(hidden_after_attn, lctx);
}

TensorPtr Lfm2Engine::apply_ffn(const TensorPtr& hidden_after_attn,
                                const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;

    if (!lw.w1() || !lw.w3() || !lw.w2()) {
        return hidden_after_attn;
    }

    // LFM2 block skeleton: ffn_norm is the pre-FFN norm (no post-attention norm).
    TensorPtr ffn_input = ops::rms_norm(hidden_after_attn, lw.ffn_norm(), cfg.rms_norm_eps);

    // FfnExecutor fuses decode (M=1) gate+up / down+residual into single OpenMP
    // regions; prefill goes through the batched path.
    int seq_len = static_cast<int>(ffn_input->shape()[0]);
    TensorPtr ffn_out;
    {
        PERF_SCOPE("layer/ffn");
        ffn_out = FfnExecutor::apply(ffn_input, hidden_after_attn, cfg, lw, seq_len,
                                     lctx.device.type);
    }
    if (FfnExecutor::residual_fused(cfg, lw, seq_len, lctx.device.type)) {
        return ffn_out;
    }
    return ops::add(hidden_after_attn, ffn_out);
}

// =========================================================================
// Shortconv operator layer
// =========================================================================

float* Lfm2Engine::conv_state(int seq_id, int layer_idx, int64_t start_pos, int seq_len,
                              DeviceType dev, int state_rows, int hidden_dim) {
    auto& per_layer = conv_states_[seq_id];
    if (per_layer.empty()) {
        per_layer.resize(static_cast<size_t>(model_.config().num_layers));
    }
    auto& entry = per_layer[static_cast<size_t>(layer_idx)];

    if (!entry.valid || entry.next_pos != start_pos) {
        // First touch, or a rewind/gap (e.g. KVMem re-prefill from an earlier
        // point): stale history rows would poison the window, restart clean.
        if (!entry.state || entry.state->shape()[0] != state_rows ||
            entry.state->shape()[1] != hidden_dim) {
            entry.state = std::make_shared<Tensor>(DataType::FP32,
                                                   std::vector<int64_t>{state_rows, hidden_dim},
                                                   dev);
        }
        entry.state->zero_();
        entry.valid = true;
    }
    entry.next_pos = start_pos + seq_len;
    return static_cast<float*>(entry.state->data());
}

TensorPtr Lfm2Engine::shortconv_cpu(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int T = lctx.seq_len();
    const int H = cfg.hidden_dim;
    const int KW = conv_kernel(cfg, lw);
    const int SR = KW - 1;  // recurrent state rows

    if (!lw.shortconv_in_proj() || !lw.shortconv_conv() || !lw.shortconv_out_proj()) {
        LOG_ERROR("Lfm2Engine: layer " + std::to_string(lctx.layer_idx) +
                  " missing shortconv weights");
        return nullptr;
    }

    // bcx = in_proj(normed): [T, 3H]; chunk order b|c|x (llama.cpp build_rs split).
    auto bcx = ensure_cpu(ops::matmul_transB(normed, lw.shortconv_in_proj()));

    auto bx = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{T, H}, DeviceType::CPU);
    auto gate = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{T, H}, DeviceType::CPU);

    const float* src = static_cast<const float*>(bcx->data());
    float* bx_data = static_cast<float*>(bx->data());
    float* gate_data = static_cast<float*>(gate->data());
    // bx = b ⊙ x; gate holds c for the post-conv multiply.
    cpu::parallel_for2(T, H, [&](int s, int c) {
        const float* row = src + s * 3 * H;
        bx_data[s * H + c] = row[c] * row[2 * H + c];
        gate_data[s * H + c] = row[H + c];
    });

    auto y = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{T, H}, DeviceType::CPU);
    float* state = conv_state(lctx.seq_id(), lctx.layer_idx, lctx.start_pos(), T, DeviceType::CPU,
                              SR, H);
    // Weight layout [H, KW] (weight[c * KW + k]) — identical to the qwen35 GDN
    // conv; the state slides forward in place.
    Qwen35LinearAttention::conv1d_cpu(bx_data, static_cast<const float*>(lw.shortconv_conv()->data()),
                                      static_cast<float*>(y->data()), state, T, H, KW);

    // y = c ⊙ conv_out
    const float* gate_ptr = static_cast<const float*>(gate->data());
    float* y_ptr = static_cast<float*>(y->data());
    cpu::parallel_for(T * H, [&](int i) { y_ptr[i] *= gate_ptr[i]; });

    return ops::matmul_transB(y, lw.shortconv_out_proj());
}

#ifdef USE_CUDA
TensorPtr Lfm2Engine::shortconv_cuda(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int T = lctx.seq_len();
    const int H = cfg.hidden_dim;
    const int KW = conv_kernel(cfg, lw);
    const int SR = KW - 1;

    if (!lw.shortconv_in_proj() || !lw.shortconv_conv() || !lw.shortconv_out_proj()) {
        LOG_ERROR("Lfm2Engine: layer " + std::to_string(lctx.layer_idx) +
                  " missing shortconv weights");
        return nullptr;
    }

    cudaStream_t stream = 0;

    auto bcx = ops::matmul_transB(normed, lw.shortconv_in_proj());  // [T, 3H] on CUDA
    auto bx = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{T, H}, DeviceType::CUDA);
    auto gate =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{T, H}, DeviceType::CUDA);
    forge::cuda::launch_lfm2_bcx_split(static_cast<const float*>(bcx->data()),
                                       static_cast<float*>(bx->data()),
                                       static_cast<float*>(gate->data()), T, H, stream);

    auto y = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{T, H}, DeviceType::CUDA);
    float* state = conv_state(lctx.seq_id(), lctx.layer_idx, lctx.start_pos(), T, DeviceType::CUDA,
                              SR, H);
    forge::cuda::launch_ssm_conv1d(static_cast<const float*>(bx->data()),
                                   static_cast<const float*>(lw.shortconv_conv()->data()), state,
                                   static_cast<float*>(y->data()), T, H, KW, stream);

    // y = c ⊙ conv_out
    auto yg = ops::multiply(gate, y);
    return ops::matmul_transB(yg, lw.shortconv_out_proj());
}
#endif

// =========================================================================
// GQA attention layer (QK-norm + NeoX RoPE, no output gate)
// =========================================================================

TensorPtr Lfm2Engine::attention_cpu(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;
    const int q_dim = num_heads * head_dim;

    TensorPtr q, k, v;
    if (lw.attn_q()->dtype() == DataType::Q4_K && lw.attn_k()->dtype() == DataType::Q4_K &&
        lw.attn_v()->dtype() == DataType::Q4_K) {
        // Shared input: one quantization pass + one OpenMP region.
        auto qkv_fused =
            ops::matmul_transB_fused_qkv_q4_k(normed, lw.attn_q(), lw.attn_k(), lw.attn_v());
        int Nq = static_cast<int>(lw.attn_q()->shape()[0]);
        int Nk = static_cast<int>(lw.attn_k()->shape()[0]);
        int Nv = static_cast<int>(lw.attn_v()->shape()[0]);
        const int row_stride = Nq + Nk + Nv;
        const float* src = static_cast<const float*>(qkv_fused->data());
        q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, Nq},
                                     DeviceType::CPU);
        k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, Nk},
                                     DeviceType::CPU);
        v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, Nv},
                                     DeviceType::CPU);
        for (int s = 0; s < seq_len; ++s) {
            const float* row = src + s * row_stride;
            std::memcpy(static_cast<float*>(q->data()) + s * Nq, row, Nq * sizeof(float));
            std::memcpy(static_cast<float*>(k->data()) + s * Nk, row + Nq, Nk * sizeof(float));
            std::memcpy(static_cast<float*>(v->data()) + s * Nv, row + Nq + Nk, Nv * sizeof(float));
        }
    } else {
        q = ensure_cpu(ops::matmul_transB(normed, lw.attn_q()));
        k = ensure_cpu(ops::matmul_transB(normed, lw.attn_k()));
        v = ensure_cpu(ops::matmul_transB(normed, lw.attn_v()));
    }

    if (lw.attn_q_norm()) {
        per_head_rms_norm_cpu(static_cast<float*>(q->data()),
                              load_head_norm_weight(lw.attn_q_norm(), head_dim), seq_len,
                              num_heads, head_dim, cfg.rms_norm_eps);
    }
    if (lw.attn_k_norm()) {
        per_head_rms_norm_cpu(static_cast<float*>(k->data()),
                              load_head_norm_weight(lw.attn_k_norm(), head_dim), seq_len,
                              num_kv_heads, head_dim, cfg.rms_norm_eps);
    }

    // NeoX rotate-half RoPE over the full head dim (LFM2 has no partial rope).
    const int n_rot = Qwen35Rope::rot_dim(cfg);
    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
    Qwen35Rope::apply_cpu(static_cast<const float*>(q->data()),
                          static_cast<const float*>(k->data()),
                          static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                          seq_len, num_heads, num_kv_heads, head_dim, n_rot, lctx.rope_pos(),
                          cfg.rope_theta);

    // ---- KV write (storage-aware: paged PagedKVStorage or contiguous KVCache) ----
    if (kv_memory_) {
        if (!kv_memory_->write_kv(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), seq_len, k_rope,
                                  v)) {
            throw std::runtime_error("lfm2: KV write failed at layer " +
                                     std::to_string(lctx.layer_idx) +
                                     " (paged page pool exhausted or invalid seq)");
        }
    } else {
        if (kv_cache_.update(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), k_rope, v, seq_len) <
            0) {
            throw std::runtime_error("lfm2: KV cache overflow at layer " +
                                     std::to_string(lctx.layer_idx) + ": write_pos=" +
                                     std::to_string(lctx.start_pos()) + " seq_len=" +
                                     std::to_string(seq_len));
        }
    }

    return ops::matmul_transB(
        attn_.attend(q_rope, cfg, lctx.layer_idx, seq_len, DeviceType::CPU, nullptr,
                     lctx.seq_id(), /*causal=*/true),
        lw.attn_output());
}

#ifdef USE_CUDA
TensorPtr Lfm2Engine::attention_cuda(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;

    cudaStream_t stream = 0;

    auto to_gpu = [](const TensorPtr& t) { return restore_device(t, DeviceType::CUDA); };

    auto q = ops::matmul_transB(normed, lw.attn_q());
    auto k = ops::matmul_transB(normed, lw.attn_k());
    auto v = ops::matmul_transB(normed, lw.attn_v());

    if (lw.attn_q_norm()) {
        auto q_norm_w = to_gpu(lw.attn_q_norm());
        forge::cuda::launch_rms_norm(static_cast<const float*>(q->data()),
                                     static_cast<const float*>(q_norm_w->data()),
                                     static_cast<float*>(q->data()), seq_len * num_heads, head_dim,
                                     cfg.rms_norm_eps, stream);
    }
    if (lw.attn_k_norm()) {
        auto k_norm_w = to_gpu(lw.attn_k_norm());
        forge::cuda::launch_rms_norm(static_cast<const float*>(k->data()),
                                     static_cast<const float*>(k_norm_w->data()),
                                     static_cast<float*>(k->data()), seq_len * num_kv_heads,
                                     head_dim, cfg.rms_norm_eps, stream);
    }

    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CUDA);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CUDA);
    forge::cuda::launch_rope_gqa(static_cast<const float*>(q->data()),
                                 static_cast<const float*>(k->data()),
                                 static_cast<float*>(q_rope->data()),
                                 static_cast<float*>(k_rope->data()), num_heads, num_kv_heads,
                                 head_dim, seq_len, lctx.rope_pos(), cfg.rope_theta, stream);

    // ---- KV write (storage-aware: paged PagedKVStorage or contiguous KVCache) ----
    if (kv_memory_) {
        if (!kv_memory_->write_kv(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), seq_len, k_rope,
                                  v)) {
            throw std::runtime_error("lfm2: KV write failed at layer " +
                                     std::to_string(lctx.layer_idx) +
                                     " (paged page pool exhausted or invalid seq)");
        }
    } else {
        if (kv_cache_.update(lctx.layer_idx, lctx.seq_id(), lctx.start_pos(), k_rope, v, seq_len) <
            0) {
            throw std::runtime_error("lfm2: KV cache overflow at layer " +
                                     std::to_string(lctx.layer_idx) + ": write_pos=" +
                                     std::to_string(lctx.start_pos()) + " seq_len=" +
                                     std::to_string(seq_len));
        }
    }

    TensorPtr attn_out = attn_.attend(q_rope, cfg, lctx.layer_idx, seq_len, DeviceType::CUDA,
                                      nullptr, lctx.seq_id(), /*causal=*/true, stream);
    return ops::matmul_transB(attn_out, lw.attn_output());
}
#endif

}  // namespace forge
