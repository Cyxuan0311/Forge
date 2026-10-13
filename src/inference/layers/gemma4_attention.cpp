#include "forge/inference/layers/gemma4_attention.h"

#include <cmath>

#include "forge/cuda_kernels.h"
#include "forge/inference/layers/per_head_norm.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/kv_memory.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

namespace forge {

namespace {

// Gemma4 RoPE: NeoX 风格 split-half + Q 的 sqrt(head_dim) 预缩放。
// freq_factors 非空时对每个维度做频率缩放 (Proportional RoPE)。
// k_data/k_out 为空时只处理 Q (无自有 KV 的层)。
void rope_gemma4_cpu(const float* q_data, const float* k_data, float* q_out, float* k_out,
                     int seq_len, int num_heads, int num_kv_heads, int head_dim, int64_t start_pos,
                     float theta, const float* freq_factors) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    float q_scale = std::sqrt(static_cast<float>(head_dim));

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

                float q0 = q_data[q_idx0];
                float q1 = q_data[q_idx1];
                q_out[q_idx0] = (q0 * cos_a - q1 * sin_a) * q_scale;
                q_out[q_idx1] = (q0 * sin_a + q1 * cos_a) * q_scale;

                if (k_data && h < num_kv_heads) {
                    int k_idx0 = s * k_stride + h * head_dim + d;
                    int k_idx1 = k_idx0 + half_dim;

                    float k0 = k_data[k_idx0];
                    float k1 = k_data[k_idx1];
                    k_out[k_idx0] = k0 * cos_a - k1 * sin_a;
                    k_out[k_idx1] = k0 * sin_a + k1 * cos_a;
                }
            }
        }
    }
}

}  // namespace

TensorPtr Gemma4Attention::attend(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int layer_idx = lctx.layer_idx;
    const int seq_len = lctx.seq_len();
    const int64_t start_pos = lctx.start_pos();
    const int64_t rope_pos = lctx.rope_pos();
    const DeviceType dev = lctx.device.type;

    bool is_swa_layer = (layer_idx < (int)cfg.swa_layers.size() && cfg.swa_layers[layer_idx] == 1);
    bool has_kv = (layer_idx < cfg.n_layer_kv_from_start);

    int head_dim = is_swa_layer ? cfg.head_dim_swa : cfg.head_dim;
    int num_heads = is_swa_layer ? cfg.num_heads_swa : cfg.num_heads;
    int num_kv_heads = cfg.kv_heads_for_layer(layer_idx, is_swa_layer);

    TensorPtr q = ops::matmul_transB(normed, lw.wq());

    // ---- Q-Norm (per-head RMSNorm) ----
    if (lw.attn_q_norm()) {
        PERF_SCOPE("layer/q_norm");
#ifdef USE_CUDA
        if (q->device() == DeviceType::CUDA && lw.attn_q_norm()->device() == DeviceType::CUDA) {
            auto q_out = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CUDA);
            cuda::launch_rms_norm(static_cast<const float*>(q->data()),
                                  static_cast<const float*>(lw.attn_q_norm()->data()),
                                  static_cast<float*>(q_out->data()), seq_len * num_heads, head_dim,
                                  cfg.rms_norm_eps);
            q = q_out;
        } else
#endif
        {
            q = ensure_cpu(q);
            auto qn_w = ensure_cpu(lw.attn_q_norm());
            per_head_rms_norm_cpu(static_cast<float*>(q->data()),
                                  static_cast<const float*>(qn_w->data()), seq_len, num_heads,
                                  head_dim, cfg.rms_norm_eps);
        }
    }

    float theta = is_swa_layer ? cfg.rope_theta_swa : cfg.rope_theta;

    // full-attention 层使用 Proportional RoPE 频率因子; SWA 层不使用。
    auto resolve_freq_factors = [&](const TensorPtr& ref, TensorPtr& gpu_temp) -> const float* {
        if (is_swa_layer || !rope_freqs_) return nullptr;
        if (ref->device() == DeviceType::CUDA) {
            if (rope_freqs_->device() == DeviceType::CUDA) {
                return static_cast<const float*>(rope_freqs_->data());
            }
            gpu_temp = std::make_shared<Tensor>(DataType::FP32, rope_freqs_->shape(),
                                                DeviceType::CUDA);
            gpu_temp->copy_from(*rope_freqs_);
            return static_cast<const float*>(gpu_temp->data());
        }
        rope_freqs_cpu_ = ensure_cpu(rope_freqs_);
        return static_cast<const float*>(rope_freqs_cpu_->data());
    };

    auto q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
    TensorPtr q_rope_out = q_rope;
    TensorPtr k_rope, v;

    if (has_kv) {
        TensorPtr k = ops::matmul_transB(normed, lw.wk());
        v = lw.wv() ? ops::matmul_transB(normed, lw.wv()) : k;

        // ---- K-Norm ----
        if (lw.attn_k_norm()) {
            PERF_SCOPE("layer/k_norm");
#ifdef USE_CUDA
            if (k->device() == DeviceType::CUDA && lw.attn_k_norm()->device() == DeviceType::CUDA) {
                auto k_out = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CUDA);
                cuda::launch_rms_norm(static_cast<const float*>(k->data()),
                                      static_cast<const float*>(lw.attn_k_norm()->data()),
                                      static_cast<float*>(k_out->data()), seq_len * num_kv_heads,
                                      head_dim, cfg.rms_norm_eps);
                k = k_out;
            } else
#endif
            {
                k = ensure_cpu(k);
                auto kn_w = ensure_cpu(lw.attn_k_norm());
                per_head_rms_norm_cpu(static_cast<float*>(k->data()),
                                      static_cast<const float*>(kn_w->data()), seq_len,
                                      num_kv_heads, head_dim, cfg.rms_norm_eps);
            }
        }

        // ---- V-Norm (无 learned weight) ----
        {
            PERF_SCOPE("layer/v_norm");
#ifdef USE_CUDA
            if (v->device() == DeviceType::CUDA) {
                auto v_out = std::make_shared<Tensor>(DataType::FP32, v->shape(), DeviceType::CUDA);
                cuda::launch_rms_norm_unweighted(static_cast<const float*>(v->data()),
                                                 static_cast<float*>(v_out->data()),
                                                 seq_len * num_kv_heads, head_dim,
                                                 cfg.rms_norm_eps);
                v = v_out;
            } else
#endif
            {
                per_head_rms_norm_cpu(static_cast<float*>(v->data()), nullptr, seq_len,
                                      num_kv_heads, head_dim, cfg.rms_norm_eps);
            }
        }

        TensorPtr rope_freqs_gpu_temp;
        const float* d_freq_factors = resolve_freq_factors(q, rope_freqs_gpu_temp);

#ifdef USE_CUDA
        if (q->device() == DeviceType::CUDA && k->device() == DeviceType::CUDA) {
            PERF_SCOPE("layer/rope");
            q_rope_out = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CUDA);
            k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CUDA);
            cuda::launch_rope_gemma4_gqa(
                static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                static_cast<float*>(q_rope_out->data()), static_cast<float*>(k_rope->data()),
                num_heads, num_kv_heads, head_dim, seq_len, rope_pos, theta, d_freq_factors);
        } else
#endif
        {
            PERF_SCOPE("layer/rope");
            q = ensure_cpu(q);
            k = ensure_cpu(k);
            k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
            rope_gemma4_cpu(static_cast<const float*>(q->data()),
                            static_cast<const float*>(k->data()),
                            static_cast<float*>(q_rope->data()),
                            static_cast<float*>(k_rope->data()), seq_len, num_heads, num_kv_heads,
                            head_dim, rope_pos, theta, d_freq_factors);
        }

        {
            PERF_SCOPE("layer/kv_cache_update");
            if (kv_mem_) {
                // Paged mode: KVMemory::write_kv is the single write facade -- it
                // stages K/V for the storage backend (host pools need the rows on
                // the host; quantization happens per page write) and appends to
                // the sequence's pages.
                if (!kv_mem_->write_kv(layer_idx, lctx.seq_id(), start_pos, seq_len, k_rope, v)) {
                    throw KVOverflowError("paged KV write failed at layer " +
                                          std::to_string(layer_idx) +
                                          ": pos=" + std::to_string(start_pos) +
                                          " seq_len=" + std::to_string(seq_len) +
                                          " (page pool exhausted -- reduce --ctx-size "
                                          "or free memory)");
                }
            } else if (kv_cache_.update(layer_idx, lctx.seq_id(), start_pos, k_rope, v, seq_len) <
                       0) {
                // KVOverflowError, not a bare runtime_error: the entry points
                // catch this type and print an actionable message, whereas an
                // uncaught std::runtime_error here aborted the process with a
                // core dump.
                throw KVOverflowError("context too long for the KV cache at layer " +
                                      std::to_string(layer_idx) + ": writing " +
                                      std::to_string(seq_len) + " token(s) at position " +
                                      std::to_string(start_pos) + ", which exceeds the " +
                                      std::to_string(kv_cache_.max_seq_len()) + "-token context");
            }
        }
    } else {
        // 无自有 KV 的层: 只对 Q 应用 RoPE, KV 复用其他层。
        TensorPtr rope_freqs_gpu_temp;
        const float* d_freq_factors = resolve_freq_factors(q, rope_freqs_gpu_temp);

#ifdef USE_CUDA
        if (q->device() == DeviceType::CUDA) {
            PERF_SCOPE("layer/rope_q_only");
            q_rope_out = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CUDA);
            cuda::launch_rope_gemma4_q_only(static_cast<const float*>(q->data()),
                                            static_cast<float*>(q_rope_out->data()), num_heads,
                                            head_dim, seq_len, rope_pos, theta, d_freq_factors);
        } else
#endif
        {
            PERF_SCOPE("layer/rope_q_only");
            q = ensure_cpu(q);
            rope_gemma4_cpu(static_cast<const float*>(q->data()), nullptr,
                            static_cast<float*>(q_rope->data()), nullptr, seq_len, num_heads,
                            num_kv_heads, head_dim, rope_pos, theta, d_freq_factors);
        }
    }

    q_rope_out = restore_device(q_rope_out, dev);

    // ---- Attention ----
    PERF_SCOPE("layer/attention");
    int total_len = 0;
    TensorPtr k_sliced, v_sliced;
    int attn_num_kv_heads = num_kv_heads;
    int attn_head_dim = head_dim;

    // KV actually read: own layer when it has KV, otherwise the reused layer.
    // SWA 层复用 n_layer_kv_from_start - 2, full-attention 层复用 - 1,
    // 与 llama.cpp 参考实现一致。
    int read_layer = layer_idx;
    bool read_is_swa = is_swa_layer;
    if (!has_kv) {
        read_layer =
            is_swa_layer ? (cfg.n_layer_kv_from_start - 2) : (cfg.n_layer_kv_from_start - 1);
        read_is_swa = (read_layer < (int)cfg.swa_layers.size() && cfg.swa_layers[read_layer] == 1);
        attn_num_kv_heads = cfg.kv_heads_for_layer(read_layer, read_is_swa);
        attn_head_dim = read_is_swa ? cfg.head_dim_swa : cfg.head_dim;
    }

    if (kv_mem_) {
        // Paged mode: resolve K/V per sequence from the storage backend. The
        // row-mirror fast paths (device mirror for CUDA compute, host mirror for
        // CPU compute) hand back persistent views; the fallback materializes the
        // sequence's rows (dequantizing quantized pages on the fly).
        KVStorage& st = kv_mem_->storage();
        const int sid = lctx.seq_id();
        total_len = sid >= 0 ? st.seq_filled(read_layer, sid) : st.filled(read_layer);
        if (total_len <= 0) {
            // Empty view: nothing written yet -- causal attention has no valid
            // keys. Return zeros (mirrors AttentionExecutor's guard).
            auto out = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, dev);
            out->zero_();
            return out;
        }
        if (sid >= 0) {
            if (dev == DeviceType::CUDA && st.device() == DeviceType::CPU)
                st.read_window_cuda(read_layer, sid, &k_sliced, &v_sliced);
            else if (dev == DeviceType::CPU && st.device() == DeviceType::CPU)
                st.read_window_host(read_layer, sid, &k_sliced, &v_sliced);
            if (!k_sliced)
                k_sliced = st.read_key_seq(read_layer, sid);
            if (!v_sliced)
                v_sliced = st.read_value_seq(read_layer, sid);
        }
        if (!k_sliced)
            k_sliced = st.read_key(read_layer);
        if (!v_sliced)
            v_sliced = st.read_value(read_layer);
        if (!k_sliced || !v_sliced) {
            throw std::runtime_error(
                "paged KV storage cannot materialize K/V for attention (layer=" +
                std::to_string(read_layer) + ", seq=" + std::to_string(lctx.seq_id()) + ")");
        }
    } else {
        // Contiguous mode. Quantized stores have no live FP32 rows --
        // dequantize_layer() refreshes the FP32 shadow incrementally and is a
        // no-op for FP32 caches (and for ring layers, whose shadow is the native
        // store itself).
        kv_cache_.dequantize_layer(read_layer);
        total_len = kv_cache_.filled(read_layer);
        k_sliced = kv_cache_.get_key_filled(read_layer);
        v_sliced = kv_cache_.get_value_filled(read_layer);
    }

    // SWA 的滑动窗口由 KV cache 的 ring buffer 处理, get_key_filled()/filled()
    // 已只返回最后 window_size 个位置。当 ring 未启用时 (量化 store 无实时 FP32
    // 镜像, ring 回绕会读到陈旧行; 分页后端没有 ring), 窗口语义在读取侧执行:
    // attention 只看最后 n_swa 个 token。
    const bool ring_handled = !kv_mem_ && kv_cache_.use_ring_buffer(read_layer);
    if (read_is_swa && cfg.n_swa > 0 && !ring_handled && total_len > cfg.n_swa) {
        const int64_t start_row = static_cast<int64_t>(total_len) - cfg.n_swa;
        k_sliced = std::make_shared<Tensor>(k_sliced->slice(0, start_row, total_len));
        v_sliced = std::make_shared<Tensor>(v_sliced->slice(0, start_row, total_len));
        total_len = cfg.n_swa;
    }

    // KV reuse crosses layer devices under mixed offload (-ngl < n_layers): a
    // CPU layer reuses KV held by a CUDA layer (and vice versa). Transfer the
    // slice to this layer's device, or the attention kernel dereferences a
    // foreign pointer (host SEGV for CPU layers, sticky illegal access for
    // CUDA layers).
    if (k_sliced && k_sliced->device() != dev) {
        auto k_dev = std::make_shared<Tensor>(DataType::FP32, k_sliced->shape(), dev);
        k_dev->copy_from(*k_sliced);
        k_sliced = k_dev;
        auto v_dev = std::make_shared<Tensor>(DataType::FP32, v_sliced->shape(), dev);
        v_dev->copy_from(*v_sliced);
        v_sliced = v_dev;
    }

    if (attn_num_kv_heads < num_heads) {
        auto out = ops::scaled_dot_product_attention_2d_gqa(q_rope_out, k_sliced, v_sliced, seq_len,
                                                            total_len, num_heads, attn_num_kv_heads,
                                                            attn_head_dim, nullptr, true);
        return out;
    }
    auto out2 =
        ops::scaled_dot_product_attention_2d(q_rope_out, k_sliced, v_sliced, seq_len, total_len,
                                             num_heads, attn_head_dim, nullptr, true);
    return out2;
}

}  // namespace forge
