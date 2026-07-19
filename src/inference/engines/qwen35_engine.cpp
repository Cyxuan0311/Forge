#include "forge/engines/qwen35_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/operators.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

// Enable verbose debug logging (set to 0 to disable)
#define QWEN35_DEBUG 0

#if QWEN35_DEBUG
#    define QDEBUG(fmt, ...)                     \
        do {                                     \
            fprintf(stderr, fmt, ##__VA_ARGS__); \
            fflush(stderr);                      \
        } while (0)
#else
#    define QDEBUG(fmt, ...) ((void)0)
#endif

// FATAL errors always print
#define QFATAL(fmt, ...)                     \
    do {                                     \
        fprintf(stderr, fmt, ##__VA_ARGS__); \
        fflush(stderr);                      \
    } while (0)

namespace forge {

// ============================================================================
// Engine constructor
// ============================================================================
Qwen35Engine::Qwen35Engine(Model& model, InferenceContext& ctx) : TransformerEngine(model, ctx) {
    if (!init_weights()) {
        throw std::runtime_error("Qwen35Engine: failed to initialize weights");
    }
    init_recurrent_states();
}

bool Qwen35Engine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

// ============================================================================
// Initialize recurrent states for Gated Delta Net layers
// ============================================================================
void Qwen35Engine::init_recurrent_states() {
    const auto& cfg = model_.config();
    if (!cfg.use_ssm)
        return;

    // Parse SSM dimensions from config and weight shapes
    ssm_d_state_ = cfg.ssm_state_size;
    ssm_n_group_ = cfg.ssm_group_count;
    ssm_dt_rank_ = cfg.ssm_time_step_rank;
    ssm_d_inner_ = cfg.ssm_inner_size;
    ssm_d_conv_ = cfg.ssm_conv_kernel;

    // Derive from weight shapes if config values are zero
    for (int i = 0; i < cfg.num_layers; ++i) {
        if (weights_.layers[i].layer_type == LayerType::LinearAttention) {
            if (weights_.layers[i].ssm_conv1d() && ssm_d_conv_ == 0) {
                auto& shape = weights_.layers[i].ssm_conv1d()->shape();
                ssm_d_conv_ = static_cast<int>(shape[0]);
            }
            if (weights_.layers[i].ssm_a() && ssm_dt_rank_ == 0) {
                ssm_dt_rank_ = static_cast<int>(weights_.layers[i].ssm_a()->numel());
            }
            break;
        }
    }

    // Fallback defaults
    if (ssm_d_state_ == 0)
        ssm_d_state_ = 128;
    if (ssm_n_group_ == 0)
        ssm_n_group_ = 16;
    if (ssm_dt_rank_ == 0)
        ssm_dt_rank_ = 16;
    if (ssm_d_inner_ == 0)
        ssm_d_inner_ = 2 * cfg.hidden_dim;
    if (ssm_d_conv_ == 0)
        ssm_d_conv_ = 4;

    // Compute derived dimensions
    ssm_head_v_dim_ = ssm_d_inner_ / ssm_dt_rank_;
    ssm_conv_channels_ = ssm_d_inner_ + 2 * ssm_n_group_ * ssm_d_state_;

    LOG_INFO("Qwen35Engine: Gated Delta Net params:");
    LOG_INFO(
        "  d_inner=" + std::to_string(ssm_d_inner_) + ", d_state=" + std::to_string(ssm_d_state_) +
        ", n_group=" + std::to_string(ssm_n_group_) + ", dt_rank=" + std::to_string(ssm_dt_rank_));
    LOG_INFO("  head_v_dim=" + std::to_string(ssm_head_v_dim_) + ", conv_channels=" +
             std::to_string(ssm_conv_channels_) + ", d_conv=" + std::to_string(ssm_d_conv_));

    // Allocate states
    int state_size = ssm_head_v_dim_ * ssm_head_v_dim_ * ssm_dt_rank_;
    int conv_state_size = (ssm_d_conv_ - 1) * ssm_conv_channels_;

    int num_linear_layers = 0;
    for (int i = 0; i < cfg.num_layers; ++i) {
        if (weights_.layers[i].layer_type == LayerType::LinearAttention) {
            num_linear_layers++;
        }
    }

    size_t total_ssm_bytes =
        (size_t)num_linear_layers * (state_size + conv_state_size) * sizeof(float);
    LOG_INFO("Qwen35Engine: SSM state allocation:");
    LOG_INFO("  num_linear_layers=" + std::to_string(num_linear_layers) +
             ", state_size_per_layer=" + std::to_string(state_size) + " floats (" +
             std::to_string(state_size * sizeof(float) / (1024 * 1024)) + " MB)" +
             ", conv_state_per_layer=" + std::to_string(conv_state_size) + " floats (" +
             std::to_string(conv_state_size * sizeof(float) / 1024) + " KB)");
    LOG_INFO("  Total SSM state: " + std::to_string(total_ssm_bytes / (1024 * 1024)) + " MB");

    recurrent_states_.resize(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; ++i) {
        if (weights_.layers[i].layer_type == LayerType::LinearAttention) {
            recurrent_states_[i].conv_state.resize(conv_state_size, 0.0f);
            recurrent_states_[i].ssm_state.resize(state_size, 0.0f);
        }
    }
    states_initialized_ = true;
    LOG_INFO("Qwen35Engine: SSM states allocated successfully");
}

void Qwen35Engine::reset() {
    kv_cache_.reset();
    kv_cache_initialized_ = false;
    for (auto& state : recurrent_states_) {
        std::fill(state.conv_state.begin(), state.conv_state.end(), 0.0f);
        std::fill(state.ssm_state.begin(), state.ssm_state.end(), 0.0f);
    }
}

// ============================================================================
// Layer dispatch
// ============================================================================
TensorPtr Qwen35Engine::forward_layer(const TensorPtr& hidden, int layer_idx, int seq_len,
                                      int64_t start_pos, DeviceType dev, int seq_id) {
    const auto& lw = weights_.layers[layer_idx];

    if (lw.layer_type == LayerType::FullAttention) {
        return forward_full_attn_layer(hidden, layer_idx, seq_len, start_pos, dev, seq_id);
    } else {
        return forward_linear_attn_layer(hidden, layer_idx, seq_len, start_pos, dev, seq_id);
    }
}

// ============================================================================
// Full Attention Layer (with gated Q, Q/K norm, MRoPE)
// ============================================================================
TensorPtr Qwen35Engine::forward_full_attn_layer(const TensorPtr& hidden, int layer_idx, int seq_len,
                                                int64_t start_pos, DeviceType dev, int seq_id) {
    const auto& cfg = model_.config();
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;
    const auto& lw = weights_.layers[layer_idx];

    LOG_DEBUG("Layer " + std::to_string(layer_idx) + " (FullAttn): seq_len=" +
              std::to_string(seq_len) + ", start_pos=" + std::to_string(start_pos));

    // Validate required weights
    if (!lw.attn_norm()) {
        QFATAL("[FATAL] Layer %d (FullAttn): missing attn_norm\n", layer_idx);
        return hidden;
    }
    if (!lw.attn_q()) {
        QFATAL("[FATAL] Layer %d (FullAttn): missing attn_q\n", layer_idx);
        return hidden;
    }
    if (!lw.attn_k()) {
        QFATAL("[FATAL] Layer %d (FullAttn): missing attn_k\n", layer_idx);
        return hidden;
    }
    if (!lw.attn_v()) {
        QFATAL("[FATAL] Layer %d (FullAttn): missing attn_v\n", layer_idx);
        return hidden;
    }
    if (!lw.attn_output()) {
        QFATAL("[FATAL] Layer %d (FullAttn): missing attn_output\n", layer_idx);
        return hidden;
    }

    // Norm
    auto normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);

    // Q projection: outputs [num_heads * head_dim * 2] (Q + gate concatenated per head)
    auto q_full = ops::matmul_transB(normed, lw.attn_q());

    // Split Q and gate from the concatenated output
    int q_dim = num_heads * head_dim;

    // Move q_full to CPU for splitting
    auto q_full_cpu = q_full;
    if (q_full_cpu->device() == DeviceType::CUDA) {
        q_full_cpu = std::make_shared<Tensor>(DataType::FP32, q_full->shape(), DeviceType::CPU);
        q_full_cpu->copy_from(*q_full);
    }

    auto q = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, q_dim},
                                      DeviceType::CPU);
    auto gate = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, q_dim},
                                         DeviceType::CPU);

    const float* qf_data = static_cast<const float*>(q_full_cpu->data());
    float* q_data = static_cast<float*>(q->data());
    float* g_data = static_cast<float*>(gate->data());

    if (!qf_data) {
        QFATAL("[FATAL] Layer %d (FullAttn): qf_data is NULL!\n", layer_idx);
        return hidden;
    }

// Split per head: each head has [head_dim Q | head_dim gate]
#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_heads > 4)
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            const float* src = qf_data + s * num_heads * head_dim * 2 + h * head_dim * 2;
            float* dst_q = q_data + s * num_heads * head_dim + h * head_dim;
            float* dst_g = g_data + s * num_heads * head_dim + h * head_dim;
            std::memcpy(dst_q, src, head_dim * sizeof(float));
            std::memcpy(dst_g, src + head_dim, head_dim * sizeof(float));
        }
    }

    // K, V projections
    auto k = ops::matmul_transB(normed, lw.attn_k());
    auto v = ops::matmul_transB(normed, lw.attn_v());

    // Move K, V to CPU for per-head operations (RMSNorm, RoPE)
    if (k->device() == DeviceType::CUDA) {
        auto k_cpu = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
        k_cpu->copy_from(*k);
        k = k_cpu;
    }
    if (v->device() == DeviceType::CUDA) {
        auto v_cpu = std::make_shared<Tensor>(DataType::FP32, v->shape(), DeviceType::CPU);
        v_cpu->copy_from(*v);
        v = v_cpu;
    }

    // Apply Q/K RMSNorm
    if (lw.attn_q_norm()) {
        float* qd = static_cast<float*>(q->data());
        auto q_norm_w = lw.attn_q_norm();
        std::vector<float> qn_w(head_dim);
        if (q_norm_w->dtype() == DataType::FP32) {
            if (q_norm_w->device() == DeviceType::CUDA) {
                auto tmp =
                    std::make_shared<Tensor>(DataType::FP32, q_norm_w->shape(), DeviceType::CPU);
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
    }
    if (lw.attn_k_norm()) {
        float* kd = static_cast<float*>(k->data());
        auto k_norm_w = lw.attn_k_norm();
        std::vector<float> kn_w(head_dim);
        if (k_norm_w->dtype() == DataType::FP32) {
            if (k_norm_w->device() == DeviceType::CUDA) {
                auto tmp =
                    std::make_shared<Tensor>(DataType::FP32, k_norm_w->shape(), DeviceType::CPU);
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
    }

    // Apply MRoPE
    int n_rot = cfg.use_mrope ? cfg.rope_dimension_count : head_dim;
    if (n_rot <= 0)
        n_rot = head_dim;

    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
    auto k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);

    apply_rope_mrope(static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                     static_cast<float*>(q_rope->data()), static_cast<float*>(k_rope->data()),
                     seq_len, num_heads, num_kv_heads, head_dim, n_rot, start_pos, cfg.rope_theta);

    // KV cache
    kv_cache_.update(layer_idx, seq_id, start_pos, k_rope, v, seq_len);
    if (kv_cache_.kv_dtype() == KVCacheDType::Q4_0) {
        kv_cache_.dequantize_layer(layer_idx);
    }

    int total_len = kv_cache_.filled(layer_idx);

    TensorPtr k_sliced = kv_cache_.get_key_filled(layer_idx);
    TensorPtr v_sliced = kv_cache_.get_value_filled(layer_idx);

    // Expand KV heads if needed (GQA)
    TensorPtr k_expanded, v_expanded;
    if (num_kv_heads < num_heads) {
        auto to_cpu = [](const TensorPtr& t) -> TensorPtr {
            if (t->device() == DeviceType::CPU)
                return t;
            auto tmp = std::make_shared<Tensor>(DataType::FP32, t->shape(), DeviceType::CPU);
            tmp->copy_from(*t);
            return tmp;
        };
        auto k_cpu = to_cpu(k_sliced);
        auto v_cpu = to_cpu(v_sliced);
        k_expanded =
            expand_kv_heads(k_cpu, total_len, num_heads, num_kv_heads, head_dim, DeviceType::CPU);
        v_expanded =
            expand_kv_heads(v_cpu, total_len, num_heads, num_kv_heads, head_dim, DeviceType::CPU);
    } else {
        k_expanded = k_sliced;
        v_expanded = v_sliced;
    }

    // Move Q, K, V to target device for attention computation
    if (dev == DeviceType::CUDA) {
        auto to_dev = [dev](const TensorPtr& t) -> TensorPtr {
            if (t->device() == dev)
                return t;
            auto tmp = std::make_shared<Tensor>(DataType::FP32, t->shape(), dev);
            tmp->copy_from(*t);
            return tmp;
        };
        q_rope = to_dev(q_rope);
        k_expanded = to_dev(k_expanded);
        v_expanded = to_dev(v_expanded);
    }

    auto attn_out = ops::scaled_dot_product_attention_2d(q_rope, k_expanded, v_expanded, seq_len,
                                                         total_len, num_heads, head_dim, nullptr, true);

    // Move attn_out to CPU for gating
    auto attn_out_cpu = attn_out;
    if (attn_out_cpu->device() == DeviceType::CUDA) {
        attn_out_cpu = std::make_shared<Tensor>(DataType::FP32, attn_out->shape(), DeviceType::CPU);
        attn_out_cpu->copy_from(*attn_out);
    }

    // Gated attention: output = sigmoid(gate) * attn_out
    float* attn_data = static_cast<float*>(attn_out_cpu->data());
    const float* gate_data = static_cast<const float*>(gate->data());
    for (int i = 0; i < seq_len * num_heads * head_dim; ++i) {
        float g = 1.0f / (1.0f + std::exp(-gate_data[i]));
        attn_data[i] *= g;
    }

    // Output projection
    if (dev == DeviceType::CUDA) {
        auto tmp =
            std::make_shared<Tensor>(DataType::FP32, attn_out_cpu->shape(), DeviceType::CUDA);
        tmp->copy_from(*attn_out_cpu);
        attn_out_cpu = tmp;
    }
    auto attn_proj = ops::matmul_transB(attn_out_cpu, lw.attn_output());
    auto hidden_after_attn = ops::add(hidden, attn_proj);

    // Post-attention norm + FFN
    TensorPtr ffn_input;
    if (lw.post_attention_norm()) {
        ffn_input = ops::rms_norm(hidden_after_attn, lw.post_attention_norm(), cfg.rms_norm_eps);
    } else {
        ffn_input = hidden_after_attn;
    }

    if (lw.w1() && lw.w3() && lw.w2()) {
        auto gate_ffn = ops::matmul_transB(ffn_input, lw.w1());
        auto up = ops::matmul_transB(ffn_input, lw.w3());
        auto ffn_mid = ops::silu_multiply(gate_ffn, up);
        auto ffn_out = ops::matmul_transB(ffn_mid, lw.w2());
        return ops::add(hidden_after_attn, ffn_out);
    }

    return hidden_after_attn;
}

// ============================================================================
// Linear Attention Layer (Gated Delta Net)
// ============================================================================
TensorPtr Qwen35Engine::forward_linear_attn_layer(const TensorPtr& hidden, int layer_idx,
                                                  int seq_len, int64_t start_pos, DeviceType dev, int seq_id) {
    const auto& cfg = model_.config();
    const auto& lw = weights_.layers[layer_idx];
    int hidden_dim = cfg.hidden_dim;

    int head_k_dim = ssm_d_state_;
    int num_k_heads = ssm_n_group_;
    int num_v_heads = ssm_dt_rank_;
    int head_v_dim = ssm_head_v_dim_;
    int d_conv = ssm_d_conv_;
    int conv_channels = ssm_conv_channels_;

    LOG_DEBUG(
        "Layer " + std::to_string(layer_idx) + " (LinearAttn): seq_len=" + std::to_string(seq_len) +
        ", start_pos=" + std::to_string(start_pos) + ", hidden_dim=" + std::to_string(hidden_dim) +
        ", conv_channels=" + std::to_string(conv_channels));

    // Validate required weights
    if (!lw.attn_norm()) {
        QFATAL("[FATAL] Layer %d (LinearAttn): missing attn_norm\n", layer_idx);
        return hidden;
    }
    if (!lw.attn_qkv()) {
        QFATAL("[FATAL] Layer %d (LinearAttn): missing attn_qkv\n", layer_idx);
        return hidden;
    }
    if (!lw.attn_gate()) {
        QFATAL("[FATAL] Layer %d (LinearAttn): missing attn_gate\n", layer_idx);
        return hidden;
    }
    if (!lw.ssm_conv1d()) {
        QFATAL("[FATAL] Layer %d (LinearAttn): missing ssm_conv1d\n", layer_idx);
        return hidden;
    }
    if (!lw.ssm_alpha()) {
        QFATAL("[FATAL] Layer %d (LinearAttn): missing ssm_alpha\n", layer_idx);
        return hidden;
    }
    if (!lw.ssm_beta()) {
        QFATAL("[FATAL] Layer %d (LinearAttn): missing ssm_beta\n", layer_idx);
        return hidden;
    }
    if (!lw.ssm_norm()) {
        QFATAL("[FATAL] Layer %d (LinearAttn): missing ssm_norm\n", layer_idx);
        return hidden;
    }
    if (!lw.ssm_out()) {
        QFATAL("[FATAL] Layer %d (LinearAttn): missing ssm_out\n", layer_idx);
        return hidden;
    }

    // Step 1: Attention norm
    auto normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);

    // Step 2: Input projections
    int key_dim = head_k_dim * num_k_heads;
    int value_dim = head_v_dim * num_v_heads;

    auto qkv_mixed = ops::matmul_transB(normed, lw.attn_qkv());
    auto z = ops::matmul_transB(normed, lw.attn_gate());
    auto alpha = ops::matmul_transB(normed, lw.ssm_alpha());
    auto beta = ops::matmul_transB(normed, lw.ssm_beta());

    // Move results to CPU for recurrent computation
    auto to_cpu = [](const TensorPtr& t) -> TensorPtr {
        if (!t || t->device() == DeviceType::CPU)
            return t;
        auto cpu = std::make_shared<Tensor>(DataType::FP32, t->shape(), DeviceType::CPU);
        cpu->copy_from(*t);
        return cpu;
    };
    qkv_mixed = to_cpu(qkv_mixed);
    z = to_cpu(z);
    alpha = to_cpu(alpha);
    beta = to_cpu(beta);

    float* alpha_data = static_cast<float*>(alpha->data());
    float* beta_data = static_cast<float*>(beta->data());

    // Add dt bias and apply softplus to alpha
    if (lw.ssm_dt()) {
        auto dt_cpu = lw.ssm_dt();
        if (dt_cpu->device() == DeviceType::CUDA) {
            dt_cpu = std::make_shared<Tensor>(lw.ssm_dt()->dtype(), lw.ssm_dt()->shape(),
                                              DeviceType::CPU);
            dt_cpu->copy_from(*lw.ssm_dt());
        }
        const float* dt_bias = static_cast<const float*>(dt_cpu->data());
        for (int s = 0; s < seq_len; ++s) {
            for (int j = 0; j < num_v_heads; ++j) {
                alpha_data[s * num_v_heads + j] += dt_bias[j];
            }
        }
    }

    // Softplus on alpha
    for (int i = 0; i < seq_len * num_v_heads; ++i) {
        float x = alpha_data[i];
        alpha_data[i] = std::log(1.0f + std::exp(x));
    }

    // Multiply by ssm_a to get gate
    if (lw.ssm_a()) {
        auto a_cpu = lw.ssm_a();
        if (a_cpu->device() == DeviceType::CUDA) {
            a_cpu =
                std::make_shared<Tensor>(lw.ssm_a()->dtype(), lw.ssm_a()->shape(), DeviceType::CPU);
            a_cpu->copy_from(*lw.ssm_a());
        }
        const float* a_data = static_cast<const float*>(a_cpu->data());
        for (int s = 0; s < seq_len; ++s) {
            for (int j = 0; j < num_v_heads; ++j) {
                alpha_data[s * num_v_heads + j] *= a_data[j];
            }
        }
    }

    // Sigmoid on beta
    for (int i = 0; i < seq_len * num_v_heads; ++i) {
        beta_data[i] = 1.0f / (1.0f + std::exp(-beta_data[i]));
    }

    // Step 4: Causal conv1d on QKV
    const float* qkv_data = static_cast<const float*>(qkv_mixed->data());

    auto conv_weight_cpu = lw.ssm_conv1d();
    if (conv_weight_cpu && conv_weight_cpu->device() == DeviceType::CUDA) {
        conv_weight_cpu = std::make_shared<Tensor>(lw.ssm_conv1d()->dtype(),
                                                   lw.ssm_conv1d()->shape(), DeviceType::CPU);
        conv_weight_cpu->copy_from(*lw.ssm_conv1d());
    }

    std::vector<float> conv_w(d_conv * conv_channels, 0.0f);
    if (conv_weight_cpu) {
        if (conv_weight_cpu->dtype() == DataType::FP32) {
            std::memcpy(conv_w.data(), conv_weight_cpu->data(), conv_w.size() * sizeof(float));
        } else if (conv_weight_cpu->dtype() == DataType::FP16) {
            const uint16_t* fp16 = static_cast<const uint16_t*>(conv_weight_cpu->data());
            for (size_t i = 0; i < conv_w.size(); ++i) {
                uint16_t h = fp16[i];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1f;
                uint32_t mant = h & 0x3ff;
                float f;
                if (exp == 0)
                    f = mant == 0 ? 0.0f : std::ldexp(static_cast<float>(mant), -24);
                else if (exp == 31)
                    f = mant == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
                else
                    f = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
                conv_w[i] = sign ? -f : f;
            }
        }
    }

    auto conv_out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len, conv_channels}, DeviceType::CPU);

    ssm_conv1d_cpu(qkv_data, conv_w.data(), static_cast<float*>(conv_out->data()),
                   recurrent_states_[layer_idx].conv_state.data(), seq_len, conv_channels, d_conv);

    // Step 5: SiLU on conv output
    float* conv_data = static_cast<float*>(conv_out->data());
    for (int i = 0; i < seq_len * conv_channels; ++i) {
        float x = conv_data[i];
        conv_data[i] = x / (1.0f + std::exp(-x));
    }

    // Step 6: Split conv output into Q, K, V
    auto q_conv = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, key_dim},
                                           DeviceType::CPU);
    auto k_conv = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, key_dim},
                                           DeviceType::CPU);
    auto v_conv = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len, value_dim},
                                           DeviceType::CPU);

    float* q_conv_data = static_cast<float*>(q_conv->data());
    float* k_conv_data = static_cast<float*>(k_conv->data());
    float* v_conv_data = static_cast<float*>(v_conv->data());

    for (int s = 0; s < seq_len; ++s) {
        const float* src = conv_data + s * conv_channels;
        std::memcpy(q_conv_data + s * key_dim, src, key_dim * sizeof(float));
        std::memcpy(k_conv_data + s * key_dim, src + key_dim, key_dim * sizeof(float));
        std::memcpy(v_conv_data + s * value_dim, src + 2 * key_dim, value_dim * sizeof(float));
    }

// Step 7: L2 normalize Q and K (per head)
#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_k_heads > 4)
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_k_heads; ++h) {
            float* q_head = q_conv_data + s * key_dim + h * head_k_dim;
            float norm = 0.0f;
            for (int d = 0; d < head_k_dim; ++d)
                norm += q_head[d] * q_head[d];
            float inv_norm = 1.0f / std::sqrt(norm + cfg.rms_norm_eps);
            for (int d = 0; d < head_k_dim; ++d)
                q_head[d] *= inv_norm;

            float* k_head = k_conv_data + s * key_dim + h * head_k_dim;
            norm = 0.0f;
            for (int d = 0; d < head_k_dim; ++d)
                norm += k_head[d] * k_head[d];
            inv_norm = 1.0f / std::sqrt(norm + cfg.rms_norm_eps);
            for (int d = 0; d < head_k_dim; ++d)
                k_head[d] *= inv_norm;
        }
    }

    // Step 8: Gated Delta Net - autoregressive processing
    auto delta_net_out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len, value_dim}, DeviceType::CPU);
    float* dn_out_data = static_cast<float*>(delta_net_out->data());

    for (int s = 0; s < seq_len; ++s) {
        const float* q_s = q_conv_data + s * key_dim;
        const float* k_s = k_conv_data + s * key_dim;
        const float* v_s = v_conv_data + s * value_dim;
        const float* gate_s = alpha_data + s * num_v_heads;
        const float* beta_s = beta_data + s * num_v_heads;

        float* out_s = dn_out_data + s * value_dim;
        float* state = recurrent_states_[layer_idx].ssm_state.data();

        gated_delta_net_ar_cpu(q_s, k_s, v_s, gate_s, beta_s, state, out_s, head_k_dim, head_v_dim,
                               num_k_heads, num_v_heads);
    }

    // Step 9: Gated normalization
    auto z_cpu = z;
    const float* z_data = static_cast<const float*>(z_cpu->data());

    std::vector<float> ssm_norm_w(head_v_dim, 1.0f);
    if (lw.ssm_norm()) {
        auto norm_cpu = lw.ssm_norm();
        if (norm_cpu->device() == DeviceType::CUDA) {
            norm_cpu = std::make_shared<Tensor>(lw.ssm_norm()->dtype(), lw.ssm_norm()->shape(),
                                                DeviceType::CPU);
            norm_cpu->copy_from(*lw.ssm_norm());
        }
        if (norm_cpu->dtype() == DataType::FP32) {
            std::memcpy(ssm_norm_w.data(), norm_cpu->data(), head_v_dim * sizeof(float));
        }
    }

#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_v_heads > 4)
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_v_heads; ++h) {
            float* out_head = dn_out_data + s * value_dim + h * head_v_dim;
            const float* z_head = z_data + s * value_dim + h * head_v_dim;

            float norm_sq = 0.0f;
            for (int d = 0; d < head_v_dim; ++d)
                norm_sq += out_head[d] * out_head[d];
            float inv_rms = 1.0f / std::sqrt(norm_sq / head_v_dim + cfg.rms_norm_eps);
            for (int d = 0; d < head_v_dim; ++d)
                out_head[d] *= inv_rms * ssm_norm_w[d];

            for (int d = 0; d < head_v_dim; ++d) {
                float gz = z_head[d] / (1.0f + std::exp(-z_head[d]));
                out_head[d] *= gz;
            }
        }
    }

    // Step 10: SSM output projection
    TensorPtr ssm_output;
    if (lw.ssm_out()) {
        auto dn_on_dev = delta_net_out;
        if (dev == DeviceType::CUDA && dn_on_dev->device() == DeviceType::CPU) {
            dn_on_dev =
                std::make_shared<Tensor>(DataType::FP32, delta_net_out->shape(), DeviceType::CUDA);
            dn_on_dev->copy_from(*delta_net_out);
        }
        ssm_output = ops::matmul_transB(dn_on_dev, lw.ssm_out());
    } else {
        ssm_output = delta_net_out;
    }

    // Residual connection
    auto hidden_after_attn = ops::add(hidden, ssm_output);

    // Step 11: Post-attention norm + FFN
    TensorPtr ffn_input;
    if (lw.post_attention_norm()) {
        ffn_input = ops::rms_norm(hidden_after_attn, lw.post_attention_norm(), cfg.rms_norm_eps);
    } else {
        ffn_input = hidden_after_attn;
    }

    if (lw.w1() && lw.w3() && lw.w2()) {
        auto gate_ffn = ops::matmul_transB(ffn_input, lw.w1());
        auto up = ops::matmul_transB(ffn_input, lw.w3());
        auto ffn_mid = ops::silu_multiply(gate_ffn, up);
        auto ffn_out = ops::matmul_transB(ffn_mid, lw.w2());
        return ops::add(hidden_after_attn, ffn_out);
    }

    return hidden_after_attn;
}

// ============================================================================
// Gated Delta Net autoregressive step
// ============================================================================
void Qwen35Engine::gated_delta_net_ar_cpu(const float* q, const float* k, const float* v,
                                          const float* gate, const float* beta, float* state,
                                          float* output, int head_k_dim, int head_v_dim,
                                          int num_k_heads, int num_v_heads) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    int head_repeat = num_v_heads / num_k_heads;

#pragma omp parallel for schedule(static) if (num_v_heads > 1)
    for (int hv = 0; hv < num_v_heads; ++hv) {
        int hk = hv / head_repeat;

        float* S = state + hv * head_v_dim * head_v_dim;
        const float* q_h = q + hk * head_k_dim;
        const float* k_h = k + hk * head_k_dim;
        const float* v_h = v + hv * head_v_dim;
        float g = std::exp(gate[hv]);
        float b = beta[hv];

        for (int i = 0; i < head_v_dim * head_v_dim; ++i) {
            S[i] *= g;
        }

        // Compute delta = (v - S^T @ k) * beta
        // llama.cpp convention: sk[i] = sum_j(S[j,i] * k[j]), d[i] = (v[i] - sk[i]) * beta
        std::vector<float> delta(head_v_dim);
        for (int i = 0; i < head_v_dim; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < head_k_dim; ++j) {
                sum += S[j * head_v_dim + i] * k_h[j];
            }
            delta[i] = (v_h[i] - sum) * b;
        }

        // Update state: S += k * delta^T (outer product)
        // S[j,i] += k[j] * delta[i]
        for (int j = 0; j < head_v_dim; ++j) {
            for (int i = 0; i < head_k_dim; ++i) {
                S[j * head_v_dim + i] += k_h[j] * delta[i];
            }
        }

        // Compute output: o = S^T @ q * scale
        // o[i] = sum_j(S[j,i] * q[j]) * scale
        float* o = output + hv * head_v_dim;
        for (int i = 0; i < head_v_dim; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < head_k_dim; ++j) {
                sum += S[j * head_v_dim + i] * q_h[j];
            }
            o[i] = sum * scale;
        }
    }
}

// ============================================================================
// Causal conv1d (with persistent state)
// ============================================================================
void Qwen35Engine::ssm_conv1d_cpu(const float* x_data, const float* weight_data, float* y_data,
                                  float* conv_state, int seq_len, int conv_channels, int d_conv) {
    int state_len = d_conv - 1;

    for (int s = 0; s < seq_len; ++s) {
        const float* x_row = x_data + s * conv_channels;
        float* y_row = y_data + s * conv_channels;

#pragma omp parallel for schedule(static) if (conv_channels > 64)
        for (int c = 0; c < conv_channels; ++c) {
            float val = 0.0f;
            for (int k = 0; k < d_conv; ++k) {
                float x_val;
                if (k < state_len) {
                    x_val = conv_state[k * conv_channels + c];
                } else {
                    x_val = x_row[c];
                }
                val += x_val * weight_data[c * d_conv + k];
            }
            y_row[c] = val;

            for (int k = 0; k < state_len - 1; ++k) {
                conv_state[k * conv_channels + c] = conv_state[(k + 1) * conv_channels + c];
            }
            conv_state[(state_len - 1) * conv_channels + c] = x_row[c];
        }
    }
}

// ============================================================================
// MRoPE (Multi-dimensional RoPE) for Qwen3.5
// ============================================================================
void Qwen35Engine::apply_rope_mrope(const float* q_data, const float* k_data, float* q_out,
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

}  // namespace forge
