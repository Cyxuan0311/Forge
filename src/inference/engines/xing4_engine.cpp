#include "forge/engines/xing4_engine.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cpu/common/thread_pool.h"
#include "forge/engine.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/logger.h"
#include "forge/operator_elementwise.h"
#include "forge/operator_embedding.h"
#include "forge/operator_matmul.h"
#include "forge/operator_norm.h"
#include "forge/perf_profiler.h"
#include "forge/quant_traits.h"

// CPU absorbed-MLA attention op (shared with bailingmoe3; gate may be null).
#include "cpu/gdn.h"

// CUDA backends: absorbed MLA kernels (gate-aware, null gate fine) + grouped
// IQ MoE + Xing4.0 mHC residual-stream kernels.
#include "cuda/cuda_gdn.h"
#include "cuda/cuda_gemv.h"
#include "cuda/cuda_xing4.h"
#include "forge/cuda_kernels.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace forge {

// ============================================================================
// Construction / lifecycle
// ============================================================================

Xing4_0Engine::Xing4_0Engine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx),
      mhc_(model.config().hc_sinkhorn_iters, model.config().rms_norm_eps,
           model.config().hidden_dim) {
    if (!init_weights())
        throw std::runtime_error("Xing4_0Engine: failed to initialize weights");
    n_layers_ = model_.config().num_layers;
    cache_.resize(static_cast<size_t>(n_layers_));
    mla_derived_hist_.resize(static_cast<size_t>(n_layers_));
}

bool Xing4_0Engine::init_weights() {
    build_yarn_table();
    return weights_.init(model_.weights(), model_.config());
}

// Per-dimension YaRN frequency scale s_i such that
// theta_i = pos * inv_freq_i * s_i, plus the rotation magnitude mscale.
// Mirrors ggml rope_yarn() + ggml_rope_yarn_corr_dims().
void Xing4_0Engine::build_yarn_table() {
    const auto& cfg = model_.config();
    yarn_s_.clear();
    yarn_mscale_ = 1.0f;
    yarn_s_dev_ = nullptr;
    if (!cfg.rope_yarn_enabled)
        return;

    const int dim = cfg.rope_dimension_count > 0 ? cfg.rope_dimension_count : 0;
    if (dim <= 0)
        return;

    auto corr_dim = [&](float n_rot) {
        return static_cast<float>(dim) *
               std::log(static_cast<float>(cfg.rope_yarn_orig_ctx) /
                        (n_rot * 2.0f * 3.14159265358979323846f)) /
               (2.0f * std::log(cfg.rope_theta));
    };
    const float low = std::max(0.0f, std::floor(corr_dim(cfg.rope_yarn_beta_fast)));
    const float high =
        std::min(static_cast<float>(dim - 1), std::ceil(corr_dim(cfg.rope_yarn_beta_slow)));
    const float denom = std::max(0.001f, high - low);

    yarn_mscale_ = (cfg.rope_yarn_ext_factor != 0.0f)
                       ? cfg.rope_yarn_attn_factor *
                             (1.0f + 0.1f * std::log(1.0f / cfg.rope_yarn_freq_scale))
                       : cfg.rope_yarn_attn_factor;

    yarn_s_.assign(static_cast<size_t>(dim), 1.0f);
    for (int i0 = 0; i0 + 1 < dim; i0 += 2) {
        const float yr = (static_cast<float>(i0 / 2) - low) / denom;
        const float ramp = 1.0f - std::min(1.0f, std::max(0.0f, yr));
        const float ramp_mix = ramp * cfg.rope_yarn_ext_factor;
        const float s = cfg.rope_yarn_freq_scale * (1.0f - ramp_mix) + ramp_mix;
        yarn_s_[static_cast<size_t>(i0)] = s;
        yarn_s_[static_cast<size_t>(i0) + 1] = s;
    }

}

const float* Xing4_0Engine::yarn_s_dev_ptr() {
    if (yarn_s_.empty())
        return nullptr;
#ifdef USE_CUDA
    if (!yarn_s_dev_) {
        const int dim = static_cast<int>(yarn_s_.size());
        auto host = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{dim},
                                             DeviceType::CPU);
        std::memcpy(host->data(), yarn_s_.data(), sizeof(float) * dim);
        yarn_s_dev_ = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{dim},
                                               DeviceType::CUDA);
        yarn_s_dev_->copy_from(*host);
    }
    return static_cast<const float*>(yarn_s_dev_->data());
#else
    return nullptr;
#endif
}

void Xing4_0Engine::init_kv_cache(const ModelConfig& cfg) {
    // No unified KV cache: the absorbed-MLA latent history is engine-owned
    // (mla_derived_hist_), like bailingmoe3. Just mark initialized so the base
    // does not allocate a num_kv_heads*head_dim (== 576*1) KV buffer.
    (void)cfg;
    set_kv_cache_initialized(true);
}

void Xing4_0Engine::reset() {
    for (auto& m : mla_derived_hist_)
        m.clear();
    for (auto& c : cache_)
        c = LayerCache{};
    moe_ = MoeCudaScratch{};
    moe_idx_h_ = nullptr;
    moe_w_h_ = nullptr;
    TransformerEngine::reset();
}

// ============================================================================
// forward_request: embedding -> 4-way stream -> layers -> hc_mean -> head
// ============================================================================

TensorPtr Xing4_0Engine::forward_request(const ForwardRequest& req) {
    cpu::set_num_threads(ctx_.params().n_threads);
    init_kv_cache(model_.config());

    DeviceTarget first_dev = layer_device_target(0);
#ifdef USE_CUDA
    if (first_dev.is_cuda())
        cudaSetDevice(first_dev.device_id);
#endif

    const auto& cfg = model_.config();
    TensorPtr hidden;
    if (req.input_embeddings) {
        // Hidden-state entry (draft engines): single stream [nt, H].
        hidden = req.input_embeddings;
        if (hidden->device() != first_dev.type)
            hidden = transfer_hidden(hidden, first_dev);
    } else {
        auto token_emb = model_.weights().get("token_embedding");
        if (!token_emb) {
            LOG_ERROR("Xing4_0Engine: token_embedding is NULL");
            return nullptr;
        }
        DeviceType emb_dev = token_emb->device();
        auto ids = transfer_hidden(req.input_ids,
                                   (emb_dev != first_dev.type && emb_dev == DeviceType::CPU)
                                       ? DeviceTarget::cpu()
                                       : first_dev);
        hidden = ops::embedding(token_emb, ids, weights_.token_embedding_fp32);
        if (!hidden)
            return nullptr;
        if (emb_dev != first_dev.type)
            hidden = transfer_hidden(hidden, first_dev);
    }

    // mHC: replicate the embedding across the 4 residual streams [nt, 4*H].
    hidden = mhc_.expand(hidden, first_dev.type);
    if (!hidden)
        return nullptr;

    for (int layer = 0; layer < n_layers_; ++layer) {
        DeviceTarget layer_dev = layer_device_target(layer);
#ifdef USE_CUDA
        if (layer_dev.is_cuda())
            cudaSetDevice(layer_dev.device_id);
#endif
        hidden = transfer_hidden(hidden, layer_dev);
        hidden = forward_layer(hidden, make_layer_context(layer, req, layer_dev));
        if (!hidden) {
            LOG_ERROR("Xing4_0Engine: layer " + std::to_string(layer) + " returned NULL");
            return nullptr;
        }
    }

    // Average the residual streams -> single stream for the head.
    DeviceTarget last_dev = layer_device_target(n_layers_ - 1);
    hidden = mhc_.mean(hidden, last_dev.type);
    if (!hidden)
        return nullptr;

    auto output_norm = weights_.output_norm;
    if (output_norm && hidden->device() != output_norm->device())
        hidden = transfer_hidden(hidden, DeviceTarget(output_norm->device()));
    last_hidden_ = ops::rms_norm(hidden, output_norm, cfg.rms_norm_eps);
    hidden = last_hidden_;

    auto output_weight = weights_.output_weight;
    if (!output_weight && cfg.tie_embeddings)
        output_weight = weights_.token_embedding;
    if (output_weight && hidden->device() != output_weight->device())
        hidden = transfer_hidden(hidden, DeviceTarget(output_weight->device()));

    const int seq_len = req.n_tokens;
    TensorPtr logits;
    if (output_weight && output_weight->device() == DeviceType::CUDA && seq_len == 1) {
#ifdef USE_CUDA
        // Specialized decode output_proj kernel (M == 1), matching the base.
        int K = static_cast<int>(output_weight->shape()[1]);
        int N = static_cast<int>(output_weight->shape()[0]);
        logits = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N},
                                          DeviceType::CUDA);
        auto dtype = output_weight->dtype();
        if (dtype == DataType::Q4_0) {
            cuda::launch_output_proj_q4_0(static_cast<const float*>(hidden->data()),
                                          output_weight->data(),
                                          static_cast<float*>(logits->data()), K, N);
        } else if (dtype == DataType::Q4_K) {
            cuda::launch_output_proj_q4_k(static_cast<const float*>(hidden->data()),
                                          output_weight->data(),
                                          static_cast<float*>(logits->data()), K, N);
        } else if (dtype == DataType::Q5_K) {
            cuda::launch_output_proj_q5_k(static_cast<const float*>(hidden->data()),
                                          output_weight->data(),
                                          static_cast<float*>(logits->data()), K, N);
        } else if (dtype == DataType::Q6_K) {
            cuda::launch_output_proj_q6_k(static_cast<const float*>(hidden->data()),
                                          output_weight->data(),
                                          static_cast<float*>(logits->data()), K, N);
        } else {
            logits = ops::matmul_transB(hidden, output_weight);
        }
#endif
    } else if (!req.return_all_logits && seq_len > 1) {
        // Prefill only ever consumes the final row's logits.
        auto last_row = std::make_shared<Tensor>(hidden->slice(0, seq_len - 1, seq_len));
        logits = ops::matmul_transB(last_row, output_weight);
    } else {
        logits = ops::matmul_transB(hidden, output_weight);
    }
    return logits;
}

// forward_batch: per-item delegation through forward_request (which owns the
// mHC head/tail). Result contract matches the base: [n_seq, vocab] CPU (one
// last-token row per logits-enabled item), or [total_tokens, vocab] CPU for
// all_logits.
TensorPtr Xing4_0Engine::forward_batch(const InferenceBatch& batch) {
    if (batch.empty())
        return nullptr;

    const int n_seq = batch.size();
    int vocab = -1;
    std::vector<std::vector<float>> rows(static_cast<size_t>(n_seq));
    int64_t all_rows = 0;
    TensorPtr all_logits;

    for (int i = 0; i < n_seq; ++i) {
        const auto& item = batch.items[i];
        auto ids = std::make_shared<Tensor>(
            DataType::INT32, std::vector<int64_t>{static_cast<int64_t>(item.tokens.size())},
            DeviceType::CPU);
        std::memcpy(ids->data(), item.tokens.data(), item.tokens.size() * sizeof(int32_t));
        ForwardRequest req = ForwardRequest::from_ids(ids, item.start_pos, item.seq_id);
        req.return_all_logits = batch.all_logits;

        TensorPtr logits = ensure_cpu(forward_request(req));
        if (!logits)
            return nullptr;
        const int64_t v = logits->shape().back();
        vocab = static_cast<int>(v);
        const int64_t r = logits->shape()[0];
        const float* src = static_cast<const float*>(logits->data());

        if (batch.all_logits) {
            auto part = std::make_shared<Tensor>(DataType::FP32,
                                                 std::vector<int64_t>{r, v}, DeviceType::CPU);
            std::memcpy(part->data(), src,
                        static_cast<size_t>(r) * v * sizeof(float));
            if (!all_logits) {
                all_logits = part;
            } else {
                const int64_t prev = all_logits->shape()[0];
                auto merged = std::make_shared<Tensor>(
                    DataType::FP32, std::vector<int64_t>{prev + r, v}, DeviceType::CPU);
                std::memcpy(merged->data(), all_logits->data(),
                            static_cast<size_t>(prev) * v * sizeof(float));
                std::memcpy(static_cast<float*>(merged->data()) + prev * v, part->data(),
                            static_cast<size_t>(r) * v * sizeof(float));
                all_logits = merged;
            }
            all_rows += r;
        } else if (item.logits) {
            rows[i].assign(src + (r - 1) * v, src + r * v);
        }
    }

    if (batch.all_logits)
        return all_logits;
    if (vocab <= 0)
        return nullptr;

    auto result = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{n_seq, static_cast<int64_t>(vocab)},
        DeviceType::CPU);
    float* dst = static_cast<float*>(result->data());
    std::memset(dst, 0, static_cast<size_t>(n_seq) * vocab * sizeof(float));
    for (int i = 0; i < n_seq; ++i)
        if (!rows[i].empty())
            std::memcpy(dst + static_cast<int64_t>(i) * vocab, rows[i].data(),
                        vocab * sizeof(float));
    return result;
}

// ============================================================================
// forward_layer: mhc.pre(attn) -> norm -> MLA -> mhc.post -> mhc.pre(ffn) ->
//                norm -> FFN -> mhc.post
// ============================================================================

TensorPtr Xing4_0Engine::forward_layer(const TensorPtr& hidden,
                                       const LayerExecutionContext& lctx) {
    const DeviceType dev = lctx.device.type;
    const auto& lw = lctx.weights;

    // hidden is the 4-way residual stream [nt, 4H].
    TensorPtr x4 = hidden;

    // ---- attention block ----
    TensorPtr post_a, comb_a;
    TensorPtr x_in = mhc_.pre(x4, lw.get("hc_attn_fn"), lw.get("hc_attn_scale"),
                            lw.get("hc_attn_base"), post_a, comb_a, dev, lctx.layer_idx);
    if (!x_in)
        return nullptr;
    TensorPtr normed = ops::rms_norm(x_in, lw.attn_norm(), lctx.config.rms_norm_eps);
    TensorPtr attn_out = apply_mla(normed, lctx);
    if (!attn_out)
        return nullptr;
    x4 = mhc_.post(attn_out, hidden, post_a, comb_a, dev);
    if (!x4)
        return nullptr;

    // ---- ffn block ----
    TensorPtr post_f, comb_f;
    x_in = mhc_.pre(x4, lw.get("hc_ffn_fn"), lw.get("hc_ffn_scale"), lw.get("hc_ffn_base"),
                  post_f, comb_f, dev, lctx.layer_idx);
    if (!x_in)
        return nullptr;
    normed = ops::rms_norm(x_in, lw.ffn_norm(), lctx.config.rms_norm_eps);
    TensorPtr ffn_out = (lctx.layer_idx < lctx.config.leading_dense_block_count)
                            ? apply_dense_ffn(normed, lctx)
                            : apply_moe_ffn(normed, lctx);
    if (!ffn_out)
        return nullptr;
    return mhc_.post(ffn_out, x4, post_f, comb_f, dev);
}

// ============================================================================
// Absorbed MLA (deepseek2 style, no output gate)
// ============================================================================

// Dequantize a (possibly multi-dim) quantized tensor to FP32 row-major (flat).
static TensorPtr dequant_lead2(const TensorPtr& t) {
    auto cpu = ensure_cpu(t);
    if (!is_quantized_type(cpu->dtype()))
        return cpu;
    const int64_t total = cpu->numel();
    Tensor view_t = cpu->view(std::vector<int64_t>{1, total});
    return ops::dequantize_weight(std::make_shared<Tensor>(std::move(view_t)));
}

// Concatenate two 2-D tensors along dim-0 (device-aware).
static TensorPtr append_rows(const TensorPtr& prev, const TensorPtr& cur) {
    if (!prev) return cur;
    if (!cur) return prev;
    const int64_t p = prev->shape()[0];
    const int64_t c = cur->shape()[0];
    const int64_t rest = prev->numel() / p;
    auto out = std::make_shared<Tensor>(prev->dtype(),
                                        std::vector<int64_t>{p + c, rest},
                                        prev->device());
    const size_t pbytes = static_cast<size_t>(p) * rest * sizeof(float);
    const size_t cbytes = static_cast<size_t>(c) * rest * sizeof(float);
    if (prev->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cudaMemcpy(out->data(), prev->data(), pbytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(static_cast<char*>(out->data()) + pbytes, cur->data(), cbytes,
                   cudaMemcpyDeviceToDevice);
#endif
    } else {
        std::memcpy(out->data(), prev->data(), pbytes);
        std::memcpy(static_cast<char*>(out->data()) + pbytes, cur->data(), cbytes);
    }
    return out;
}

TensorPtr Xing4_0Engine::apply_mla(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const DeviceType dev = lctx.device.type;
    const bool on_cuda = (dev == DeviceType::CUDA);
    const int seq_len = lctx.seq_len();
    const int64_t start_pos = lctx.start_pos();
    const int64_t rope_pos = lctx.rope_pos();
    const int seq_id = lctx.seq_id();
    const int layer_idx = lctx.layer_idx;

    if (!lw.wq_a() || !lw.wq_b() || !lw.attn_q_a_norm() || !lw.kv_a_proj() ||
        !lw.kv_a_norm() || !lw.attn_k_b() || !lw.attn_v_b() || !lw.wo()) {
        LOG_ERROR("Xing4_0Engine: MLA layer " + std::to_string(layer_idx) +
                  " missing required weights");
        return nullptr;
    }

    PERF_SCOPE("layer/mla");

    const int n_heads = cfg.num_heads;
    const int qk_head_dim = cfg.mla_key_dim > 0 ? cfg.mla_key_dim : 192;
    const int v_dim = cfg.mla_value_dim > 0 ? cfg.mla_value_dim : 128;
    const int rope_dim = cfg.rope_dimension_count;
    const int qk_nope_dim = qk_head_dim - rope_dim;
    const int kv_lora = cfg.kv_lora_rank;
    const int latent_dim = kv_lora + rope_dim;
    const float theta = cfg.rope_theta;

    // YaRN: llama.cpp xing4_0.cpp build_graph pre-scales kq_scale with mscale^2
    // and applies the interpolated rope frequencies at every position.
    ops::MlaRopeYarn yarn;
    float scale = 1.0f / std::sqrt(static_cast<float>(qk_head_dim));
    if (cfg.rope_yarn_enabled) {
        yarn.enabled = true;
        yarn.freq_scale = cfg.rope_yarn_freq_scale;
        yarn.ext_factor = cfg.rope_yarn_ext_factor;
        yarn.attn_factor = cfg.rope_yarn_attn_factor;
        yarn.beta_fast = cfg.rope_yarn_beta_fast;
        yarn.beta_slow = cfg.rope_yarn_beta_slow;
        yarn.n_ctx_orig = cfg.rope_yarn_orig_ctx;

        const float fs = cfg.rope_yarn_freq_scale;
        const float ln_inv = std::log(1.0f / fs);
        const float attn_factor_org = cfg.rope_yarn_attn_factor * (1.0f + 0.1f * ln_inv);
        const float mscale =
            attn_factor_org * (1.0f + 0.1f * cfg.rope_yarn_log_mul * ln_inv);
        scale = mscale * mscale / std::sqrt(static_cast<float>(qk_head_dim));
    }

    // Q low-rank path: q = wq_b @ RMSNorm(x @ wq_a; attn_q_a_norm)
    TensorPtr q_low = ops::matmul_transB(normed, lw.wq_a());
    if (!q_low)
        return nullptr;
    TensorPtr q_low_norm = ops::rms_norm(q_low, lw.attn_q_a_norm(), cfg.rms_norm_eps);
    auto& wq_cache = cache_[layer_idx];
    if (!wq_cache.wq_b_flat)
        wq_cache.wq_b_flat = ops::flatten_rows(lw.wq_b());
    TensorPtr q_flat = ops::matmul_transB(q_low_norm, wq_cache.wq_b_flat);
    if (!q_flat)
        return nullptr;

    // K/V latent: c = kv_a_proj @ x, c = [kv_lora (nope) | rope_dim (rope)]
    TensorPtr c_cur = ops::matmul_transB(normed, lw.kv_a_proj());
    if (!c_cur)
        return nullptr;
    const int cur_n = static_cast<int>(c_cur->shape()[0]);
    const float eps = cfg.rms_norm_eps;

    // New rows: split nope/rope, RMSNorm, rotate key, contract value. Fixed
    // once written at the absolute position -> incremental append below.
    auto c_nope_new = std::make_shared<Tensor>(DataType::FP32,
                                               std::vector<int64_t>{cur_n, kv_lora}, dev);
    auto c_rope_new = std::make_shared<Tensor>(DataType::FP32,
                                               std::vector<int64_t>{cur_n, rope_dim}, dev);
    if (on_cuda) {
#ifdef USE_CUDA
        cuda::mla_split_latent(static_cast<const float*>(c_cur->data()),
                               static_cast<float*>(c_nope_new->data()),
                               static_cast<float*>(c_rope_new->data()), cur_n, kv_lora,
                               rope_dim);
#endif
    } else {
        const float* lat = static_cast<const float*>(c_cur->data());
        float* nope = static_cast<float*>(c_nope_new->data());
        float* rope = static_cast<float*>(c_rope_new->data());
        for (int r = 0; r < cur_n; ++r) {
            std::memcpy(nope + static_cast<int64_t>(r) * kv_lora,
                        lat + static_cast<int64_t>(r) * latent_dim, kv_lora * sizeof(float));
            std::memcpy(rope + static_cast<int64_t>(r) * rope_dim,
                        lat + static_cast<int64_t>(r) * latent_dim + kv_lora,
                        rope_dim * sizeof(float));
        }
    }
    TensorPtr c_nope_norm_new = ops::rms_norm(c_nope_new, lw.kv_a_norm(), eps);

    auto& c = cache_[layer_idx];
    if (on_cuda) {
#ifdef USE_CUDA
        if (!c.vb_fp32_dev) {
            if (!c.vb_fp32)
                c.vb_fp32 = dequant_lead2(lw.attn_v_b());
            c.vb_fp32->to_device(DeviceType::CUDA);
            c.vb_fp32_dev = c.vb_fp32;
        }
        if (!c.kb_fp32_dev) {
            if (!c.kb_fp32)
                c.kb_fp32 = dequant_lead2(lw.attn_k_b());
            c.kb_fp32->to_device(DeviceType::CUDA);
            c.kb_fp32_dev = c.kb_fp32;
        }
#endif
    } else {
        if (!c.vb_fp32)
            c.vb_fp32 = dequant_lead2(lw.attn_v_b());
        if (!c.kb_fp32)
            c.kb_fp32 = dequant_lead2(lw.attn_k_b());
    }
    const TensorPtr& vb = on_cuda ? c.vb_fp32_dev : c.vb_fp32;
    const TensorPtr& kb = on_cuda ? c.kb_fp32_dev : c.kb_fp32;

    // k_pe_rot: rotate new c_rope rows to their absolute positions.
    auto k_pe_rot_new = std::make_shared<Tensor>(DataType::FP32,
                                                 std::vector<int64_t>{cur_n, rope_dim}, dev);
    if (on_cuda) {
#ifdef USE_CUDA
        const float* ys = yarn_s_dev_ptr();
        cuda::mla_rope_launch(c_rope_new, k_pe_rot_new, rope_pos, cur_n, rope_dim, theta,
                              ys, yarn_mscale_);
#endif
    } else {
        const float* cr = static_cast<const float*>(c_rope_new->data());
        float* kr = static_cast<float*>(k_pe_rot_new->data());
        for (int r = 0; r < cur_n; ++r) {
            float* dst = kr + static_cast<int64_t>(r) * rope_dim;
            std::memcpy(dst, cr + static_cast<int64_t>(r) * rope_dim,
                        rope_dim * sizeof(float));
            ops::apply_rope_yarn(dst, rope_dim, rope_pos + r, theta, yarn);
        }
    }

    // v_all: v[j][h*v_dim+d] = attn_v_b[h*v_dim+d] · c_nope_norm[j], new rows only.
    auto v_all_new = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{cur_n, static_cast<int64_t>(n_heads) * v_dim},
        dev);
    if (on_cuda) {
#ifdef USE_CUDA
        cuda::mla_value_contract_launch(vb, c_nope_norm_new, v_all_new, cur_n, n_heads,
                                        kv_lora, v_dim);
#endif
    } else {
        const float* cnp = static_cast<const float*>(c_nope_norm_new->data());
        const float* vb_d = static_cast<const float*>(vb->data());
        float* va = static_cast<float*>(v_all_new->data());
        for (int j = 0; j < cur_n; ++j) {
            const float* cj = cnp + static_cast<int64_t>(j) * kv_lora;
            float* vj = va + static_cast<int64_t>(j) * n_heads * v_dim;
            for (int h = 0; h < n_heads; ++h) {
                const float* vbh = vb_d + static_cast<int64_t>(h) * v_dim * kv_lora;
                float* vjh = vj + h * v_dim;
                for (int d = 0; d < v_dim; ++d) {
                    const float* vbk = vbh + static_cast<int64_t>(d) * kv_lora;
                    float acc = 0.0f;
                    for (int k = 0; k < kv_lora; ++k) acc += vbk[k] * cj[k];
                    vjh[d] = acc;
                }
            }
        }
    }

    // Append to this (layer, seq) derived history.
    auto& hist = mla_derived_hist_[layer_idx];
    if (start_pos == 0)
        hist.erase(seq_id);  // fresh sequence for a reused seq id
    auto hit = hist.find(seq_id);
    TensorPtr c_nope_norm_full, k_pe_rot_full, v_all_full;
    int total = 0;
    if (hit != hist.end() && hit->second.c_nope_norm) {
        const int prev_n = static_cast<int>(hit->second.c_nope_norm->shape()[0]);
        total = prev_n + cur_n;
        c_nope_norm_full = append_rows(hit->second.c_nope_norm, c_nope_norm_new);
        k_pe_rot_full = append_rows(hit->second.k_pe_rot, k_pe_rot_new);
        v_all_full = append_rows(hit->second.v_all, v_all_new);
        hist[seq_id] = MlaDerived{c_nope_norm_full, k_pe_rot_full, v_all_full};
    } else {
        total = cur_n;
        c_nope_norm_full = c_nope_norm_new;
        k_pe_rot_full = k_pe_rot_new;
        v_all_full = v_all_new;
        hist[seq_id] = MlaDerived{c_nope_norm_full, k_pe_rot_full, v_all_full};
    }

    // Absorbed MLA attention (no output gate on xing4_0).
    TensorPtr out;
    if (on_cuda) {
#ifdef USE_CUDA
        const float* ys = yarn_s_dev_ptr();
        cuda::mla_attention_launch(q_flat, c_nope_norm_full, k_pe_rot_full, kb, v_all_full,
                                   nullptr, start_pos, seq_len, total, n_heads, kv_lora,
                                   qk_nope_dim, rope_dim, v_dim, theta, scale, out, ys,
                                   yarn_mscale_);
#endif
    } else {
        out = ops::bailingmoe3_mla_attention(
            q_flat, c_nope_norm_full, k_pe_rot_full, kb, v_all_full, nullptr, start_pos,
            seq_len, total, n_heads, kv_lora, qk_nope_dim, rope_dim, v_dim, theta, scale,
            cfg.rope_yarn_enabled ? &yarn : nullptr);
    }
    if (!out)
        return nullptr;
    return ops::matmul_transB(out, lw.wo());
}

// ============================================================================
// FFN branches (body only; the mHC residual is applied by hc_post)
// ============================================================================

TensorPtr Xing4_0Engine::apply_dense_ffn(const TensorPtr& normed,
                                         const LayerExecutionContext& lctx) {
    const auto& lw = lctx.weights;
    if (!lw.w1() || !lw.w2() || !lw.w3()) {
        LOG_ERROR("Xing4_0Engine: dense layer " + std::to_string(lctx.layer_idx) +
                  " missing w1/w2/w3");
        return nullptr;
    }
    TensorPtr gate = ops::matmul_transB(normed, lw.w1());
    TensorPtr up = ops::matmul_transB(normed, lw.w3());
    TensorPtr gated = ops::silu_multiply(gate, up);
    return ops::matmul_transB(gated, lw.w2());
}

TensorPtr Xing4_0Engine::apply_moe_ffn(const TensorPtr& normed,
                                       const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const DeviceType dev = lctx.device.type;
    const bool on_cuda = (dev == DeviceType::CUDA);
    const int seq_len = lctx.seq_len();
    PERF_SCOPE("layer/moe");

    // Shared expert (SiLU FFN).
    TensorPtr shared_out;
    {
        TensorPtr s_gate = ops::matmul_transB(normed, lw.ffn_gate_shexp());
        TensorPtr s_up = ops::matmul_transB(normed, lw.ffn_up_shexp());
        TensorPtr s_gated = ops::silu_multiply(s_gate, s_up);
        shared_out = ops::matmul_transB(s_gated, lw.ffn_down_shexp());
    }

    // Routed experts: sigmoid gating + (degenerate) group selection + unbiased
    // weight normalization. Mirrors llama.cpp build_moe_ffn.
    const int n_expert = cfg.n_expert;
    const int n_expert_used = cfg.n_expert_used > 0 ? cfg.n_expert_used : 1;
    const int hidden_dim = cfg.hidden_dim;
    const int n_group = cfg.moe_group_count;
    const int n_group_used = cfg.moe_group_used_count;
    const int n_per_group = n_group > 1 ? n_expert / n_group : 0;

    TensorPtr router_logits = ops::matmul_transB(normed, lw.ffn_gate_inp());
    if (!router_logits)
        return nullptr;
    TensorPtr bias_cpu;  // keep-alive: a dying ensure_cpu temp dangles the bias
    const float* bias = nullptr;
    if (lw.ffn_exp_probs_b()) {
        bias_cpu = ensure_cpu(lw.ffn_exp_probs_b());
        bias = static_cast<const float*>(bias_cpu->data());
    }

    const int n_ids = seq_len * n_expert_used;
    if (!moe_idx_h_ || moe_idx_h_->numel() < n_ids) {
        moe_idx_h_ = std::make_shared<Tensor>(DataType::INT32,
                                              std::vector<int64_t>{n_ids}, DeviceType::CPU);
        moe_w_h_ = std::make_shared<Tensor>(DataType::FP32,
                                            std::vector<int64_t>{n_ids}, DeviceType::CPU);
    }

    auto gate3d = lw.ffn_gate_exps();
    auto up3d = lw.ffn_up_exps();
    auto down3d = lw.ffn_down_exps();
    const DataType moe_dt = gate3d ? gate3d->dtype() : DataType::FP32;
    // Grouped path: three expert stacks same dtype, a verified IQ type, all on
    // CUDA. IQ3_XXS gold-standard verified (bailingmoe3).
    const bool grouped =
        on_cuda && gate3d && up3d && down3d &&
        gate3d->device() == DeviceType::CUDA && up3d->device() == DeviceType::CUDA &&
        down3d->device() == DeviceType::CUDA && up3d->dtype() == moe_dt &&
        down3d->dtype() == moe_dt &&
        (moe_dt == DataType::IQ3_XXS || moe_dt == DataType::IQ1_M);

    if (grouped) {
#ifdef USE_CUDA
        const float* bias_dev = nullptr;
        if (lw.ffn_exp_probs_b())
            bias_dev = static_cast<const float*>(lw.ffn_exp_probs_b()->data());
        if (!moe_.idx_d || moe_.idx_d->numel() < n_ids) {
            moe_.idx_d = std::make_shared<Tensor>(DataType::INT32,
                                                  std::vector<int64_t>{n_ids},
                                                  DeviceType::CUDA);
            moe_.w_d = std::make_shared<Tensor>(DataType::FP32,
                                                std::vector<int64_t>{n_ids},
                                                DeviceType::CUDA);
        }
        {
            PERF_SCOPE("moe/router");
            cuda::launch_moe_router_sigmoid(
                static_cast<const float*>(router_logits->data()), bias_dev,
                static_cast<int*>(moe_.idx_d->data()),
                static_cast<float*>(moe_.w_d->data()),
                n_expert, n_expert_used, n_group, n_group_used,
                cfg.moe_weights_norm ? 1 : 0, cfg.moe_weights_scale, seq_len, 0);
        }

        const int n_ff = static_cast<int>(gate3d->shape()[1]);
        const size_t n_gated = static_cast<size_t>(n_ids) * n_ff;
        if (!moe_.gate_out || static_cast<size_t>(moe_.gate_out->numel()) < n_gated) {
            std::vector<int64_t> shp{n_ids, n_ff};
            moe_.gate_out = std::make_shared<Tensor>(DataType::FP32, shp, DeviceType::CUDA);
            moe_.up_out = std::make_shared<Tensor>(DataType::FP32, shp, DeviceType::CUDA);
            moe_.gated_out = std::make_shared<Tensor>(DataType::FP32, shp, DeviceType::CUDA);
        }
        if (!moe_.expert_out ||
            moe_.expert_out->numel() < static_cast<int64_t>(seq_len) * hidden_dim) {
            moe_.expert_out = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{seq_len, hidden_dim}, DeviceType::CUDA);
        }
        moe_.expert_out->zero_();

        {
            PERF_SCOPE("moe/gateup");
            cuda::launch_moe_expert_iq_gateup_dt(
                moe_dt, static_cast<const float*>(normed->data()), gate3d->data(),
                up3d->data(), static_cast<float*>(moe_.gate_out->data()),
                static_cast<float*>(moe_.up_out->data()),
                static_cast<const int*>(moe_.idx_d->data()), hidden_dim, n_ff, n_expert,
                n_expert_used, seq_len, 0);
        }
        {
            PERF_SCOPE("moe/silu");
            cuda::launch_silu_multiply(static_cast<const float*>(moe_.gate_out->data()),
                                       static_cast<const float*>(moe_.up_out->data()),
                                       static_cast<float*>(moe_.gated_out->data()),
                                       static_cast<int>(n_gated));
        }
        {
            PERF_SCOPE("moe/down");
            cuda::launch_moe_expert_iq_down_dt(
                moe_dt, static_cast<const float*>(moe_.gated_out->data()), down3d->data(),
                static_cast<float*>(moe_.expert_out->data()),
                static_cast<const int*>(moe_.idx_d->data()),
                static_cast<const float*>(moe_.w_d->data()), n_ff, hidden_dim, n_expert,
                n_expert_used, seq_len, 0);
        }
        return shared_out ? ops::add(shared_out, moe_.expert_out) : moe_.expert_out;
#endif
    }

    // ---- Host routing (CPU path + CUDA per-expert fallback) ----
    TensorPtr rl_host = on_cuda ? ensure_cpu(router_logits) : router_logits;
    const float* rl = static_cast<const float*>(rl_host->data());
    int* h_idx = static_cast<int*>(moe_idx_h_->data());
    float* h_w = static_cast<float*>(moe_w_h_->data());

    for (int s = 0; s < seq_len; ++s) {
        const float* lrow = rl + static_cast<int64_t>(s) * n_expert;
        std::vector<float> probs(n_expert);
        std::vector<float> sel(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            probs[e] = 1.0f / (1.0f + std::exp(-lrow[e]));
            sel[e] = bias ? probs[e] + bias[e] : probs[e];
        }

        if (n_group > 1 && n_group_used > 0 && n_group_used < n_group) {
            std::vector<float> group_scores(n_group);
            for (int g = 0; g < n_group; ++g) {
                float top1 = -std::numeric_limits<float>::infinity();
                float top2 = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < n_per_group; ++j) {
                    const float v = sel[g * n_per_group + j];
                    if (v > top1) {
                        top2 = top1;
                        top1 = v;
                    } else if (v > top2) {
                        top2 = v;
                    }
                }
                group_scores[g] = top1 + top2;
            }
            std::vector<int> gidx(n_group);
            for (int g = 0; g < n_group; ++g)
                gidx[g] = g;
            std::partial_sort(gidx.begin(), gidx.begin() + n_group_used, gidx.end(),
                              [&](int a, int b) { return group_scores[a] > group_scores[b]; });
            std::vector<char> keep_group(n_group, 0);
            for (int gi = 0; gi < n_group_used; ++gi)
                keep_group[gidx[gi]] = 1;
            for (int e = 0; e < n_expert; ++e)
                if (!keep_group[e / n_per_group])
                    sel[e] = -std::numeric_limits<float>::infinity();
        }

        std::vector<int> idx(n_expert);
        for (int e = 0; e < n_expert; ++e)
            idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + n_expert_used, idx.end(),
                          [&](int a, int b) {
                              if (sel[a] != sel[b]) return sel[a] > sel[b];
                              return a < b;
                          });
        float topk_sum = 0.0f;
        for (int k = 0; k < n_expert_used; ++k)
            topk_sum += probs[idx[k]];

        for (int k = 0; k < n_expert_used; ++k) {
            int e = idx[k];
            float w;
            if (cfg.moe_weights_norm) {
                const float denom = std::max(topk_sum, 6.103515625e-5f);
                w = probs[e] / denom;
            } else {
                w = probs[e];
            }
            if (cfg.moe_weights_scale != 0.0f && cfg.moe_weights_scale != 1.0f)
                w *= cfg.moe_weights_scale;
            h_idx[static_cast<int64_t>(s) * n_expert_used + k] = e;
            h_w[static_cast<int64_t>(s) * n_expert_used + k] = w;
        }
    }

    auto expert_out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len, hidden_dim}, dev);
    float* eo = static_cast<float*>(expert_out->data());
    if (on_cuda) {
#ifdef USE_CUDA
        cudaMemset(eo, 0, static_cast<size_t>(seq_len) * hidden_dim * sizeof(float));
#endif
    } else {
        std::fill_n(eo, static_cast<size_t>(seq_len) * hidden_dim, 0.0f);
    }

    for (int s = 0; s < seq_len; ++s) {
        float* orow = eo + static_cast<int64_t>(s) * hidden_dim;
        for (int k = 0; k < n_expert_used; ++k) {
            int e = h_idx[static_cast<int64_t>(s) * n_expert_used + k];
            float w = h_w[static_cast<int64_t>(s) * n_expert_used + k];

            // Expert stacks are [n_expert, rows, cols] contiguous; borrowed
            // quantized views feed matmul_transB's fused dot kernels directly.
            auto gw = ops::expert_quant_view(lw.ffn_gate_exps(), e, n_expert);
            auto uw = ops::expert_quant_view(lw.ffn_up_exps(), e, n_expert);
            auto dw = ops::expert_quant_view(lw.ffn_down_exps(), e, n_expert);
            if (!gw || !uw || !dw)
                continue;

            TensorPtr th = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{1, hidden_dim}, dev);
            const char* row_src = static_cast<const char*>(normed->data()) +
                                  static_cast<int64_t>(s) * hidden_dim * sizeof(float);
            if (on_cuda) {
#ifdef USE_CUDA
                cudaMemcpy(th->data(), row_src, hidden_dim * sizeof(float),
                           cudaMemcpyDeviceToDevice);
#endif
            } else {
                std::memcpy(th->data(), row_src, hidden_dim * sizeof(float));
            }
            TensorPtr g = ops::matmul_transB(th, gw);
            TensorPtr u = ops::matmul_transB(th, uw);
            TensorPtr gated = ops::silu_multiply(g, u);
            TensorPtr er = ops::matmul_transB(gated, dw);
            if (on_cuda) {
#ifdef USE_CUDA
                cuda::moe_accumulate_launch(orow, static_cast<const float*>(er->data()),
                                            w, hidden_dim);
#endif
            } else {
                const float* erd = static_cast<const float*>(er->data());
                for (int d = 0; d < hidden_dim; ++d)
                    orow[d] += w * erd[d];
            }
        }
    }

    return shared_out ? ops::add(shared_out, expert_out) : expert_out;
}

}  // namespace forge
