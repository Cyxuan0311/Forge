#include "forge/engines/bailingmoe3_engine.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "cpu/common/thread_pool.h"
#include "forge/engine.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/logger.h"
#include "forge/operator_elementwise.h"
#include "forge/operator_matmul.h"
#include "forge/quant_traits.h"
#include "forge/operator_embedding.h"
#include "forge/operator_matmul.h"
#include "forge/operator_norm.h"
#include "forge/perf_profiler.h"

// Model-specific BailingMoE3 kernels live in the CPU operator tree, not the
// public include/forge API: depthwise conv1d + SiLU, and the GDN recurrence.
#include "cpu/conv.h"
#include "cpu/gdn.h"

// CUDA 后端的自定义 GDN/MLA 内核 (与 cpu/gdn.cpp 数学严格对齐)。
#include "cuda/cuda_gdn.h"
#include "cuda/cuda_gemv.h"        // grouped IQ MoE (gateup/down _dt)
#include "forge/cuda_kernels.h" // launch_moe_router_sigmoid / launch_silu_multiply

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace forge {

#include <algorithm>
#include <cstdio>
#include <cstdlib>


// ============================================================================
// Construction / registration
// ============================================================================

Bailingmoe3Engine::Bailingmoe3Engine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx), recurrent_memory_(*memory_.bailingmoe3_recurrent()) {
    if (!memory_.bailingmoe3_recurrent()) {
        throw std::runtime_error("Bailingmoe3Engine: memory is not a Bailingmoe3Memory");
    }
    if (!init_weights()) {
        throw std::runtime_error("Bailingmoe3Engine: failed to initialize weights");
    }
    const auto& cfg = model_.config();
    n_layers_ = cfg.num_layers;
    mla_derived_hist_.resize(n_layers_);
    mla_cache_.resize(n_layers_);
    gdn_cache_.resize(n_layers_);
    recurrent_memory_.init(cfg, weights_);

    // CPU-first: 全部权重搬到 CPU, 避免逐次 ensure_cpu 拷贝 (尤其量化权重)。
    for (auto& lw : weights_.layers) {
        for (auto& [name, t] : lw.weights) {
            (void)name;
            if (t && t->device() != DeviceType::CPU)
                t->to_device(DeviceType::CPU);
        }
    }
    auto to_cpu = [](TensorPtr& t) {
        if (t && t->device() != DeviceType::CPU)
            t->to_device(DeviceType::CPU);
    };
    to_cpu(weights_.output_norm);
    to_cpu(weights_.output_weight);
    to_cpu(weights_.token_embedding);
}

bool Bailingmoe3Engine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

void Bailingmoe3Engine::init_kv_cache(const ModelConfig& cfg) {
    // BailingMoE3 不使用统一 KV cache: MLA latent 历史由引擎自管, GDN 状态由
    // recurrent_memory_ 管理。这里仅标记已初始化, 避免基类按 num_kv_heads*head_dim
    // (bailingmoe3 下为 0) 分配无意义的 KV 缓冲。
    (void)cfg;
    set_kv_cache_initialized(true);
}

void Bailingmoe3Engine::set_gpu_layers(int gpu_layers) {
    // gpu_layers < 0 表示自动探测: 有 GPU 则全量上 CUDA, 否则回退 CPU。
    bool cuda_available = false;
#ifdef USE_CUDA
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0)
        cuda_available = true;
#endif
    cuda_mode_ = (gpu_layers < 0) ? cuda_available : (gpu_layers > 0);
    if (cuda_mode_ && !cuda_available) {
        cuda_mode_ = false;
        LOG_WARN(
            "Bailingmoe3Engine: CUDA requested but no GPU is available; "
            "running on CPU.");
    }

    gpu_layers_ = cuda_mode_ ? n_layers_ : 0;
    layer_devices_.assign(static_cast<size_t>(n_layers_),
                          cuda_mode_ ? DeviceTarget::cuda() : DeviceTarget::cpu());

    // 后端状态 (递归 conv / SSM 状态) 必须和权重同一设备。
    recurrent_memory_.set_cuda_mode(cuda_mode_);

    if (!cuda_mode_ && gpu_layers < 0) {
        LOG_WARN(
            "Bailingmoe3Engine: CUDA offload not available; running on CPU. "
            "Pass -ngl 0 to skip the pointless GPU weight upload/round-trip.");
    }

    // 权重与持久状态整体搬到一个设备 (CPU 或 CUDA), 保持放置原子性。
    const DeviceType target = cuda_mode_ ? DeviceType::CUDA : DeviceType::CPU;
    for (auto& lw : weights_.layers) {
        for (auto& [name, t] : lw.weights) {
            (void)name;
            if (t && t->device() != target)
                t->to_device(target);
        }
    }
    auto move = [target](TensorPtr& t) {
        if (t && t->device() != target)
            t->to_device(target);
    };
    move(weights_.output_norm);
    move(weights_.output_weight);
    move(weights_.token_embedding);
    if (weights_.token_embedding_fp32 && weights_.token_embedding_fp32->device() != target)
        weights_.token_embedding_fp32->to_device(target);
}

void Bailingmoe3Engine::reset() {
    for (auto& m : mla_derived_hist_)
        m.clear();
    for (auto& cache : mla_cache_)
        cache = MlaCache{};
    for (auto& cache : gdn_cache_)
        cache = GdnCache{};
    recurrent_memory_.reset();
    set_kv_cache_initialized(false);
}

// ============================================================================
// forward_request (CPU head)
// ============================================================================

TensorPtr Bailingmoe3Engine::forward_request(const ForwardRequest& req) {
    if (cuda_mode_) {
        // GPU 路径交由基类编排: 逐层 transfer_hidden + 虚 forward_layer +
        // 最终 norm/logits, 全部按 tensor->device() 派发。本 engine 只需把
        // apply_mla/apply_gdn 改成设备感知 (见 *_cuda 分支)。
        return TransformerEngine::forward_request(req);
    }
    cpu::set_num_threads(ctx_.params().n_threads);
    init_kv_cache(model_.config());

    DeviceType dev = DeviceType::CPU;
    auto ids = transfer_hidden(req.input_ids, dev);
    auto token_emb = weights_.token_embedding ? weights_.token_embedding
                                             : model_.weights().get("token_embedding");
    TensorPtr hidden = ops::embedding(ensure_cpu(token_emb), ids, weights_.token_embedding_fp32);
    if (!hidden)
        return nullptr;


    // 输出头: RMSNorm + lm_head。
    auto out_w = weights_.output_weight;
    if (!out_w && model_.config().tie_embeddings)
        out_w = weights_.token_embedding;

    for (int layer = 0; layer < n_layers_; ++layer) {
        hidden = forward_layer(hidden, make_layer_context(layer, req, DeviceTarget::cpu()));
        if (!hidden)
            return nullptr;
    }
    auto normed = ops::rms_norm(hidden, ensure_cpu(weights_.output_norm), model_.config().rms_norm_eps);
    auto logits = ops::matmul_transB(normed, ensure_cpu(out_w));
    return logits;
}

// ============================================================================
// forward_layer dispatch
// ============================================================================

TensorPtr Bailingmoe3Engine::forward_layer(const TensorPtr& hidden,
                                           const LayerExecutionContext& lctx) {
    const auto& lw = lctx.weights;
    if (!lw.attn_norm()) {
        LOG_ERROR("Bailingmoe3Engine: layer " + std::to_string(lctx.layer_idx) +
                  " missing attn_norm");
        return hidden;
    }
    TensorPtr normed =
        ops::rms_norm(hidden, lw.attn_norm(), lctx.config.rms_norm_eps);

    bool is_recurrent = (lctx.layer_idx < static_cast<int>(lctx.config.head_count_kv.size())) &&
                        lctx.config.head_count_kv[lctx.layer_idx] == 0;


#ifdef USE_CUDA
    TensorPtr attn_out = is_recurrent ? (cuda_mode_ ? apply_gdn_cuda(normed, lctx)
                                                    : apply_gdn(normed, lctx))
                                      : (cuda_mode_ ? apply_mla_cuda(normed, lctx)
                                                    : apply_mla(normed, lctx));
#else
    // CPU-only 构建: CUDA 分支 (apply_*_cuda / cuda::* 符号) 未编译, 直接走
    // 与之一一对应的 CPU 实现 (数学严格对齐, 见 apply_mla / apply_gdn)。
    TensorPtr attn_out = is_recurrent ? apply_gdn(normed, lctx)
                                      : apply_mla(normed, lctx);
#endif
    if (!attn_out)
        return hidden;

    auto attn_residual = ops::add(hidden, attn_out);
    auto f = apply_ffn(attn_residual, lctx);
    return f;
}

// ============================================================================
// Absorbed MLA branch
// ============================================================================

// Dequantize a (possibly multi-dim) quantized tensor to FP32 in row-major order.
// We flatten to [1, numel] so the stock 2-D dequant path processes the whole
// tensor as one row (numel is a multiple of the quant block size). The resulting
// flat row-major order is what ops::bailingmoe3_mla_attention consumes.
static TensorPtr dequant_lead2(const TensorPtr& t) {
    auto cpu = ensure_cpu(t);
    if (!is_quantized_type(cpu->dtype()))
        return cpu;
    const int64_t total = cpu->numel();
    Tensor view_t = cpu->view(std::vector<int64_t>{1, total});
    return ops::dequantize_weight(std::make_shared<Tensor>(std::move(view_t)));
}

// Concatenate two 2-D tensors along dim-0 (prev first, cur second). Device-aware:
// on CUDA the copy goes device-to-device; on CPU it is a plain memcpy. Used to
// append the new MLA-derived rows to the per-(layer,seq) incremental cache.
static TensorPtr append_rows(const TensorPtr& prev, const TensorPtr& cur) {
    if (!prev) return cur;
    if (!cur) return prev;
    const int64_t p = prev->shape()[0];
    const int64_t c = cur->shape()[0];
    const int64_t rest = prev->numel() / p;  // 2-D tensor: columns
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

TensorPtr Bailingmoe3Engine::apply_mla(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const int64_t start_pos = lctx.start_pos();
    const int64_t rope_pos = lctx.rope_pos();
    const int seq_id = lctx.seq_id();
    const int layer_idx = lctx.layer_idx;

    if (!lw.wq_a() || !lw.wq_b() || !lw.attn_q_a_norm() || !lw.kv_a_proj() ||
        !lw.kv_a_norm() || !lw.attn_k_b() || !lw.attn_v_b() || !lw.attn_gate() || !lw.wo()) {
        LOG_ERROR("Bailingmoe3Engine: MLA layer " + std::to_string(layer_idx) +
                  " missing required weights");
        return nullptr;
    }

    PERF_SCOPE("layer/mla");

    // Dims per llama.cpp: attention.key_length_mla is the FULL query head dim
    // (rope tail included), since wq_b projects to n_heads * qk_head_dim = 3072.
    const int n_heads = cfg.num_heads;
    const int qk_head_dim = cfg.mla_key_dim > 0 ? cfg.mla_key_dim : 192;
    const int v_dim = cfg.mla_value_dim > 0 ? cfg.mla_value_dim : 128;
    const int rope_dim = cfg.rope_dimension_count;
    const int qk_nope_dim = qk_head_dim - rope_dim;  // 128
    const int kv_lora = cfg.kv_lora_rank;            // 512
    const int latent_dim = kv_lora + rope_dim;       // 576
    const float theta = cfg.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(qk_head_dim));

    // Q low-rank path: q = wq_b @ RMSNorm(x @ wq_a; attn_q_a_norm)
    TensorPtr q_low = ops::matmul_transB(normed, lw.wq_a());  // [seq, q_lora_rank]
    if (!q_low)
        return nullptr;
    TensorPtr q_low_norm = ops::rms_norm(q_low, lw.attn_q_a_norm(), cfg.rms_norm_eps);
    // wq_b 展平视图按层缓存: Tensor::view() 共享权重存储并保留原设备,
    // 既无数据拷贝也无 host 往返, 仅省掉每 token 重建视图对象的开销。
    auto& wq_cache = mla_cache_[layer_idx];
    if (!wq_cache.wq_b_flat)
        wq_cache.wq_b_flat = ops::flatten_rows(lw.wq_b());
    TensorPtr q_flat = ops::matmul_transB(q_low_norm, wq_cache.wq_b_flat);
    if (!q_flat)
        return nullptr;

    // K/V latent: c = kv_a_proj @ x, c = [kv_lora (nope) | rope_dim (rope)]
    TensorPtr c_cur = ops::matmul_transB(normed, lw.kv_a_proj());  // [cur_n, latent_dim]
    const int cur_n = static_cast<int>(c_cur->shape()[0]);
    const float eps = cfg.rms_norm_eps;

    // 本 token 的新行: 拆分 nope/rope、RMSNorm、旋转 key、收缩 value。这些量仅
    // 取决于绝对位置, 入库即固定 —— 历史部分直接从 mla_derived_hist_ 复用, 不再每
    // token 全量重算 RoPE 与收缩值 (见 ops::bailingmoe3_mla_attention)。
    auto c_nope_new = std::make_shared<Tensor>(DataType::FP32,
                                              std::vector<int64_t>{cur_n, kv_lora},
                                              DeviceType::CPU);
    auto c_rope_new = std::make_shared<Tensor>(DataType::FP32,
                                              std::vector<int64_t>{cur_n, rope_dim},
                                              DeviceType::CPU);
    {
        const float* lat = static_cast<const float*>(c_cur->data());
        float* nope = static_cast<float*>(c_nope_new->data());
        float* rope = static_cast<float*>(c_rope_new->data());
        for (int r = 0; r < cur_n; ++r) {
            std::memcpy(nope + r * kv_lora, lat + r * latent_dim, kv_lora * sizeof(float));
            std::memcpy(rope + r * rope_dim, lat + r * latent_dim + kv_lora,
                       rope_dim * sizeof(float));
        }
    }
    TensorPtr c_nope_norm_new = ops::rms_norm(c_nope_new, lw.kv_a_norm(), eps);

    // k_pe_rot: 旋转新行 c_rope 到其绝对位置 (start_pos .. start_pos+cur_n-1)。
    auto k_pe_rot_new = std::make_shared<Tensor>(DataType::FP32,
                                                std::vector<int64_t>{cur_n, rope_dim},
                                                DeviceType::CPU);
    {
        const float* cr = static_cast<const float*>(c_rope_new->data());
        float* kr = static_cast<float*>(k_pe_rot_new->data());
        for (int r = 0; r < cur_n; ++r) {
            float* dst = kr + r * rope_dim;
            std::memcpy(dst, cr + r * rope_dim, rope_dim * sizeof(float));
            ops::apply_rope_interleaved(dst, rope_dim, rope_pos + r, theta);
        }
    }

    // 权重 (attn_k_b / attn_v_b) 按层缓存为 FP32。
    auto& weight_cache = mla_cache_[layer_idx];
    if (!weight_cache.vb_fp32)
        weight_cache.vb_fp32 = dequant_lead2(lw.attn_v_b());
    if (!weight_cache.kb_fp32)
        weight_cache.kb_fp32 = dequant_lead2(lw.attn_k_b());

    // v_all: 收缩值 v[j][h*v_dim+d] = attn_v_b[h*v_dim+d] · c_nope_norm[j], 仅新行。
    auto v_all_new = std::make_shared<Tensor>(DataType::FP32,
                                             std::vector<int64_t>{cur_n, n_heads * v_dim},
                                             DeviceType::CPU);
    {
        const float* cnp = static_cast<const float*>(c_nope_norm_new->data());
        const float* vb_d = static_cast<const float*>(weight_cache.vb_fp32->data());
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

    // 追加到本 (layer, seq) 的派生历史, 得到覆盖全部绝对位置的完整张量。
    auto& hist = mla_derived_hist_[layer_idx];
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

    // Output gate: sigmoid(x @ attn_gate) — one scalar per (token, head).
    TensorPtr gate_t = ops::matmul_transB(normed, lw.attn_gate());  // [seq_len, n_heads]
    ops::sigmoid_inplace(gate_t);

    // Absorbed MLA attention: low-rank contraction + causal softmax + output gate.
    TensorPtr out = ops::bailingmoe3_mla_attention(
        q_flat, c_nope_norm_full, k_pe_rot_full, weight_cache.kb_fp32, v_all_full,
        gate_t, start_pos, seq_len, total, n_heads, kv_lora, qk_nope_dim, rope_dim,
        v_dim, theta, scale);

    return ops::matmul_transB(out, lw.wo());
}

#ifdef USE_CUDA
// CUDA 后端 MLA 分支: 与 apply_mla 数学严格对齐, 所有张量/状态均驻留设备。
TensorPtr Bailingmoe3Engine::apply_mla_cuda(const TensorPtr& normed,
                                           const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const int64_t start_pos = lctx.start_pos();
    const int64_t rope_pos = lctx.rope_pos();
    const int seq_id = lctx.seq_id();
    const int layer_idx = lctx.layer_idx;

    if (!lw.wq_a() || !lw.wq_b() || !lw.attn_q_a_norm() || !lw.kv_a_proj() ||
        !lw.kv_a_norm() || !lw.attn_k_b() || !lw.attn_v_b() || !lw.attn_gate() || !lw.wo()) {
        LOG_ERROR("Bailingmoe3Engine: MLA layer " + std::to_string(layer_idx) +
                  " missing required weights");
        return nullptr;
    }

    const int n_heads = cfg.num_heads;
    const int qk_head_dim = cfg.mla_key_dim > 0 ? cfg.mla_key_dim : 192;
    const int v_dim = cfg.mla_value_dim > 0 ? cfg.mla_value_dim : 128;
    const int rope_dim = cfg.rope_dimension_count;
    const int qk_nope_dim = qk_head_dim - rope_dim;
    const int kv_lora = cfg.kv_lora_rank;
    const float theta = cfg.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(qk_head_dim));

    TensorPtr q_low = ops::matmul_transB(normed, lw.wq_a());
    if (!q_low)
        return nullptr;
    TensorPtr q_low_norm = ops::rms_norm(q_low, lw.attn_q_a_norm(), cfg.rms_norm_eps);
    TensorPtr q_flat = ops::matmul_transB(q_low_norm, ops::flatten_rows(lw.wq_b()));
    if (!q_flat)
        return nullptr;

    TensorPtr c_cur = ops::matmul_transB(normed, lw.kv_a_proj());  // [cur_n, latent_dim]
    const int cur_n = static_cast<int>(c_cur->shape()[0]);
    const float eps = cfg.rms_norm_eps;

    // 本 token 的新行 (设备端): 拆分 nope/rope、RMSNorm、旋转 key、收缩 value。
    // 这些量仅取决于绝对位置, 入库即固定 —— 历史部分直接从 mla_derived_hist_ 复用,
    // 不再每 token 全量重算 RoPE 与收缩值 (见 cuda::mla_attention_launch)。
    auto c_nope_new = std::make_shared<Tensor>(DataType::FP32,
                                              std::vector<int64_t>{cur_n, kv_lora},
                                              DeviceType::CUDA);
    auto c_rope_new = std::make_shared<Tensor>(DataType::FP32,
                                              std::vector<int64_t>{cur_n, rope_dim},
                                              DeviceType::CUDA);
    cuda::mla_split_latent(static_cast<const float*>(c_cur->data()),
                           static_cast<float*>(c_nope_new->data()),
                           static_cast<float*>(c_rope_new->data()), cur_n, kv_lora,
                           rope_dim);
    TensorPtr c_nope_norm_new = ops::rms_norm(c_nope_new, lw.kv_a_norm(), eps);

    auto& weight_cache = mla_cache_[layer_idx];
    if (!weight_cache.vb_fp32_dev) {
        if (!weight_cache.vb_fp32)
            weight_cache.vb_fp32 = dequant_lead2(lw.attn_v_b());
        weight_cache.vb_fp32->to_device(DeviceType::CUDA);
        weight_cache.vb_fp32_dev = weight_cache.vb_fp32;
    }
    if (!weight_cache.kb_fp32_dev) {
        if (!weight_cache.kb_fp32)
            weight_cache.kb_fp32 = dequant_lead2(lw.attn_k_b());
        weight_cache.kb_fp32->to_device(DeviceType::CUDA);
        weight_cache.kb_fp32_dev = weight_cache.kb_fp32;
    }

    // k_pe_rot: 旋转新行 c_rope 到其绝对位置 (仅在设备上算 cur_n 行)。
    auto k_pe_rot_new = std::make_shared<Tensor>(DataType::FP32,
                                                std::vector<int64_t>{cur_n, rope_dim},
                                                DeviceType::CUDA);
    cuda::mla_rope_launch(c_rope_new, k_pe_rot_new, rope_pos, cur_n, rope_dim, theta);

    // v_all: 收缩值 v[j][h*v_dim+d] = attn_v_b[h*v_dim+d] · c_nope_norm[j], 仅新行。
    auto v_all_new = std::make_shared<Tensor>(DataType::FP32,
                                             std::vector<int64_t>{cur_n, n_heads * v_dim},
                                             DeviceType::CUDA);
    cuda::mla_value_contract_launch(weight_cache.vb_fp32_dev, c_nope_norm_new,
                                    v_all_new, cur_n, n_heads, kv_lora, v_dim);

    // 追加到本 (layer, seq) 的派生历史。新序列起点先清空旧历史 (CUDA 基类 warmup)。
    auto& hist = mla_derived_hist_[layer_idx];
    if (start_pos == 0)
        hist.erase(seq_id);
    auto hit = hist.find(seq_id);
    TensorPtr c_nope_norm_full, k_pe_rot_full, v_all_full;
    int total = 0;
    if (hit != hist.end() && hit->second.c_nope_norm) {
        const int prev_n =
            static_cast<int>(hit->second.c_nope_norm->shape()[0]);
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

    TensorPtr gate_t = ops::matmul_transB(normed, lw.attn_gate());  // [seq_len, n_heads]
    cuda::sigmoid_launch(gate_t);

    TensorPtr out = nullptr;
    cuda::mla_attention_launch(
        q_flat, c_nope_norm_full, k_pe_rot_full, weight_cache.kb_fp32_dev, v_all_full,
        gate_t, start_pos, seq_len, total, n_heads, kv_lora, qk_nope_dim, rope_dim,
        v_dim, theta, scale, out);

    return ops::matmul_transB(out, lw.wo());
}
#endif  // USE_CUDA

// ============================================================================
// GDN / KDA recurrent branch
// ============================================================================
// ============================================================================
// KDA / GDN recurrent branch
// ============================================================================
// Ported from llama.cpp src/models/bailingmoe3.cpp (the `is_recr` branch), which
// is the authoritative Ling 3.0 formulation:
//   head_dim = kda.head_dim (128), n_head = attention.head_count (16),
//   d_inner  = n_head * head_dim (2048)
//   1) q/k/v : dense projection -> depthwise causal conv1d -> SiLU
//   2) decay : kda_gate_lower_bound * sigmoid(ssm_a[h] * (ssm_f_a(x) + ssm_dt_bias))
//   3) beta  : sigmoid(ssm_beta(x))                       (one scalar per head)
//   4) q,k   : L2 normalized per (token, head)
//   5) per-head gated delta rule -> attn / sqrt(head_dim)
//   6) out   : RMSNorm(ssm_norm) * sigmoid(ssm_g_a(x))
//   7) out   : wo projection back to hidden_dim

TensorPtr Bailingmoe3Engine::apply_gdn(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const int layer_idx = lctx.layer_idx;
    const int seq_id = lctx.seq_id();

    const auto& dims = recurrent_memory_.dims();
    const int head_dim = dims.kd;
    const int num_heads = dims.num_v_heads;
    const int d_conv = dims.d_conv;
    const int d_inner = dims.d_inner();

    if (!lw.attn_q() || !lw.attn_k() || !lw.attn_v() || !lw.ssm_conv1d_q() ||
        !lw.ssm_conv1d_k() || !lw.ssm_conv1d_v() || !lw.ssm_f_a() || !lw.ssm_g_a() ||
        !lw.ssm_beta() || !lw.ssm_a() || !lw.ssm_dt() || !lw.ssm_norm() || !lw.wo()) {
        LOG_ERROR("Bailingmoe3Engine: GDN layer " + std::to_string(layer_idx) +
                  " missing required weights");
        return nullptr;
    }

    PERF_SCOPE("layer/gdn");

    // ---- 1) projections + depthwise causal conv + SiLU -> [seq_len, d_inner] ----
    TensorPtr q0 = ops::matmul_transB(normed, lw.attn_q());
    TensorPtr k0 = ops::matmul_transB(normed, lw.attn_k());
    TensorPtr v0 = ops::matmul_transB(normed, lw.attn_v());
    if (!q0 || !k0 || !v0)
        return nullptr;

    const auto conv_w_q = ops::load_conv_weight(lw.ssm_conv1d_q(), d_conv, d_inner);
    const auto conv_w_k = ops::load_conv_weight(lw.ssm_conv1d_k(), d_conv, d_inner);
    const auto conv_w_v = ops::load_conv_weight(lw.ssm_conv1d_v(), d_conv, d_inner);

    float* conv_q = recurrent_memory_.conv_state_q_cpu(seq_id, layer_idx);
    float* conv_k = recurrent_memory_.conv_state_k_cpu(seq_id, layer_idx);
    float* conv_v = recurrent_memory_.conv_state_v_cpu(seq_id, layer_idx);
    if (!conv_q || !conv_k || !conv_v)
        return nullptr;

    ops::conv1d_depthwise_silu(q0, conv_w_q, conv_q, seq_len, d_inner, d_conv);
    ops::conv1d_depthwise_silu(k0, conv_w_k, conv_k, seq_len, d_inner, d_conv);
    ops::conv1d_depthwise_silu(v0, conv_w_v, conv_v, seq_len, d_inner, d_conv);

    // ---- 2) decay gate, per (head, dim) ----
    TensorPtr decay = ops::matmul_transB(normed, lw.ssm_f_a());  // [seq_len, d_inner]
    ops::build_decay_gate(decay, lw.ssm_dt(), lw.ssm_a(), seq_len, num_heads, head_dim,
                          cfg.kda_gate_lower_bound);

    // ---- 3) beta input gate, per head ----
    TensorPtr beta = ops::matmul_transB(normed, lw.ssm_beta());  // [seq_len, num_heads]
    ops::sigmoid_inplace(beta);

    // ---- 4+5) per-head gated delta rule ----
    float* state = recurrent_memory_.ssm_state_cpu(seq_id, layer_idx);
    if (!state)
        return nullptr;
    TensorPtr attn = ops::gdn_recurrence(q0, k0, v0, decay, beta, state, seq_len,
                                         num_heads, head_dim, cfg.rms_norm_eps);
    if (!attn)
        return nullptr;

    // ---- 6) per-head RMSNorm + output gate ----
    TensorPtr out_gate = ops::matmul_transB(normed, lw.ssm_g_a());  // [seq_len, d_inner]
    TensorPtr out = ops::gdn_output_gate(attn, out_gate, lw.ssm_norm(), seq_len,
                                         num_heads, head_dim, cfg.rms_norm_eps);
    if (!out)
        return nullptr;

    // ---- 7) back to hidden_dim ----
    return ops::matmul_transB(out, lw.wo());
}

// CUDA 后端 GDN 分支: 与 apply_gdn 数学严格对齐, 所有张量/状态均驻留设备。
#ifdef USE_CUDA
TensorPtr Bailingmoe3Engine::apply_gdn_cuda(const TensorPtr& normed,
                                            const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const int layer_idx = lctx.layer_idx;
    const int seq_id = lctx.seq_id();

    const auto& dims = recurrent_memory_.dims();
    const int head_dim = dims.kd;
    const int num_heads = dims.num_v_heads;
    const int d_conv = dims.d_conv;
    const int d_inner = dims.d_inner();

    if (!lw.attn_q() || !lw.attn_k() || !lw.attn_v() || !lw.ssm_conv1d_q() ||
        !lw.ssm_conv1d_k() || !lw.ssm_conv1d_v() || !lw.ssm_f_a() || !lw.ssm_g_a() ||
        !lw.ssm_beta() || !lw.ssm_a() || !lw.ssm_dt() || !lw.ssm_norm() || !lw.wo()) {
        LOG_ERROR("Bailingmoe3Engine: GDN layer " + std::to_string(layer_idx) +
                  " missing required weights");
        return nullptr;
    }

    PERF_SCOPE("layer/gdn");

    // Decode 专用融合投影: 6 个共享 normed 的量化 GEMV 按 dtype 分 2 组
    // (q,k 一组; v,f_a,g_a,beta 一组), 各一次 q8 量化 + 单 launch (逐位对齐
    // 单权重路径, 见 launch_gdn_proj_fused_dt)。prefill / dtype 不满足时回退
    // 逐权重 matmul_transB。
    TensorPtr q0, k0, v0;
    TensorPtr decay, beta, out_gate;
    bool fused_done = false;
    if (seq_len == 1 && false) {  // 数据实测: fused 负优化 (10.5 vs 12.8 tok/s), 回退逐权重
        auto group_fusable = [](const std::vector<TensorPtr>& ws) {
            if (ws.empty()) return false;
            const DataType dt = ws[0]->dtype();
            if (dt != DataType::IQ2_XXS && dt != DataType::IQ2_S &&
                dt != DataType::IQ3_XXS && dt != DataType::IQ1_M)
                return false;
            const int64_t K = ws[0]->shape()[1];
            for (const auto& w : ws) {
                if (!w || w->dtype() != dt || w->device() != DeviceType::CUDA ||
                    w->shape()[1] != K)
                    return false;
            }
            return true;
        };
        const std::vector<TensorPtr> grp_a = {lw.attn_q(), lw.attn_k()};
        const std::vector<TensorPtr> grp_b = {lw.attn_v(), lw.ssm_f_a(), lw.ssm_g_a(),
                                              lw.ssm_beta()};
        if (group_fusable(grp_a) && group_fusable(grp_b)) {
            auto alloc_out = [](int64_t N) {
                return std::make_shared<Tensor>(DataType::FP32,
                                                std::vector<int64_t>{1, N},
                                                DeviceType::CUDA);
            };
            q0 = alloc_out(lw.attn_q()->shape()[0]);
            k0 = alloc_out(lw.attn_k()->shape()[0]);
            v0 = alloc_out(lw.attn_v()->shape()[0]);
            decay = alloc_out(lw.ssm_f_a()->shape()[0]);
            beta = alloc_out(lw.ssm_beta()->shape()[0]);
            out_gate = alloc_out(lw.ssm_g_a()->shape()[0]);

            const void* wa[4] = {lw.attn_q()->data(), lw.attn_k()->data(), nullptr, nullptr};
            float* oa[4] = {static_cast<float*>(q0->data()), static_cast<float*>(k0->data()),
                            nullptr, nullptr};
            const int na[4] = {static_cast<int>(lw.attn_q()->shape()[0]),
                               static_cast<int>(lw.attn_k()->shape()[0]), 0, 0};
            {
                PERF_SCOPE("gdn/proj_a");
                cuda::launch_gdn_proj_fused_dt(
                    lw.attn_q()->dtype(), static_cast<const float*>(normed->data()), wa, oa, na,
                    2, static_cast<int>(lw.attn_q()->shape()[1]), 0);
            }

            const void* wb[4] = {lw.attn_v()->data(), lw.ssm_f_a()->data(),
                                 lw.ssm_g_a()->data(), lw.ssm_beta()->data()};
            float* ob[4] = {static_cast<float*>(v0->data()), static_cast<float*>(decay->data()),
                            static_cast<float*>(out_gate->data()),
                            static_cast<float*>(beta->data())};
            const int nb[4] = {static_cast<int>(lw.attn_v()->shape()[0]),
                               static_cast<int>(lw.ssm_f_a()->shape()[0]),
                               static_cast<int>(lw.ssm_g_a()->shape()[0]),
                               static_cast<int>(lw.ssm_beta()->shape()[0])};
            {
                PERF_SCOPE("gdn/proj_b");
                cuda::launch_gdn_proj_fused_dt(
                    lw.attn_v()->dtype(), static_cast<const float*>(normed->data()), wb, ob, nb,
                    4, static_cast<int>(lw.attn_v()->shape()[1]), 0);
            }
            fused_done = true;
        }
    }
    if (!fused_done) {
        {
            PERF_SCOPE("gdn/mm_q");
            q0 = ops::matmul_transB(normed, lw.attn_q());
        }
        {
            PERF_SCOPE("gdn/mm_k");
            k0 = ops::matmul_transB(normed, lw.attn_k());
        }
        {
            PERF_SCOPE("gdn/mm_v");
            v0 = ops::matmul_transB(normed, lw.attn_v());
        }
        if (!q0 || !k0 || !v0)
            return nullptr;
    }

    // 逐层缓存: 卷积权重与 bias/a/nw 极小常量全部 init 时一次性驻留设备端。
    // 否则每步每层 3 次阻塞 H2D (decay bias+a, ogate nw) 会把默认流流水线
    // 整条排空 (decode 每步 54 次)。
    auto& gc = gdn_cache_[layer_idx];
    if (!gc.init) {
        gc.conv_w_q = ops::load_conv_weight(lw.ssm_conv1d_q(), d_conv, d_inner);
        gc.conv_w_k = ops::load_conv_weight(lw.ssm_conv1d_k(), d_conv, d_inner);
        gc.conv_w_v = ops::load_conv_weight(lw.ssm_conv1d_v(), d_conv, d_inner);
        gc.conv_w_q->to_device(DeviceType::CUDA);
        gc.conv_w_k->to_device(DeviceType::CUDA);
        gc.conv_w_v->to_device(DeviceType::CUDA);

        const int64_t d_inner_i64 = static_cast<int64_t>(num_heads) * head_dim;
        std::vector<float> bias(static_cast<size_t>(d_inner_i64), 0.0f);
        if (lw.ssm_dt()) {
            const float* b = static_cast<const float*>(ensure_cpu(lw.ssm_dt())->data());
            for (int64_t i = 0; i < d_inner_i64; ++i) bias[i] = b[i];
        }
        std::vector<float> a(static_cast<size_t>(num_heads), 1.0f);
        if (lw.ssm_a()) {
            const float* av = static_cast<const float*>(ensure_cpu(lw.ssm_a())->data());
            for (int h = 0; h < num_heads; ++h) a[h] = av[h];
        }
        std::vector<float> nw(static_cast<size_t>(head_dim), 1.0f);
        if (lw.ssm_norm()) {
            const float* w = static_cast<const float*>(ensure_cpu(lw.ssm_norm())->data());
            for (int d = 0; d < head_dim; ++d) nw[d] = w[d];
        }
        auto upload_dev = [](const std::vector<float>& v) {
            auto t = std::make_shared<Tensor>(DataType::FP32,
                                              std::vector<int64_t>{(int64_t)v.size()},
                                              DeviceType::CPU);
            if (!v.empty())
                std::memcpy(t->data(), v.data(), v.size() * sizeof(float));
            t->to_device(DeviceType::CUDA);
            return t;
        };
        gc.bias_d = upload_dev(bias);
        gc.a_d = upload_dev(a);
        gc.nw_d = upload_dev(nw);
        gc.init = true;
    }

    float* conv_q = recurrent_memory_.conv_state_q_dev(seq_id, layer_idx);
    float* conv_k = recurrent_memory_.conv_state_k_dev(seq_id, layer_idx);
    float* conv_v = recurrent_memory_.conv_state_v_dev(seq_id, layer_idx);
    float* state_dev = recurrent_memory_.ssm_state_dev(seq_id, layer_idx);
    if (!conv_q || !conv_k || !conv_v || !state_dev)
        return nullptr;

    // 新序列起点: 清零该 (seq, layer) 的卷积历史与递归状态, 丢弃 CUDA warmup
    // 留下的数据 (与 CPU 路径每个请求从零状态开始保持一致)。
    if (lctx.start_pos() == 0) {
        const size_t sq = static_cast<size_t>(dims.conv_state_size_q()) * sizeof(float);
        const size_t sk = static_cast<size_t>(dims.conv_state_size_k()) * sizeof(float);
        const size_t sv = static_cast<size_t>(dims.conv_state_size_v()) * sizeof(float);
        const size_t ss = static_cast<size_t>(dims.state_size()) * sizeof(float);
#ifdef USE_CUDA
        if (sq) cudaMemset(conv_q, 0, sq);
        if (sk) cudaMemset(conv_k, 0, sk);
        if (sv) cudaMemset(conv_v, 0, sv);
        if (ss) cudaMemset(state_dev, 0, ss);
#endif
    }

    {
        PERF_SCOPE("gdn/conv");
        cuda::conv3_depthwise_silu_launch(q0, gc.conv_w_q, conv_q, k0, gc.conv_w_k,
                                          conv_k, v0, gc.conv_w_v, conv_v, seq_len,
                                          d_inner, d_conv);
    }

    {
        PERF_SCOPE("gdn/decay_gate");
        if (!fused_done)
            decay = ops::matmul_transB(normed, lw.ssm_f_a());  // [seq_len, d_inner]
        // bias / a 取自逐层 init 时上传的设备端常量 (零拷贝)。
        cuda::build_decay_gate_launch(decay, static_cast<const float*>(gc.bias_d->data()),
                                     static_cast<const float*>(gc.a_d->data()), seq_len,
                                     num_heads, head_dim, cfg.kda_gate_lower_bound);
    }

    {
        PERF_SCOPE("gdn/beta_sigmoid");
        if (!fused_done)
            beta = ops::matmul_transB(normed, lw.ssm_beta());  // [seq_len, num_heads]
        cuda::sigmoid_launch(beta);
    }

    TensorPtr attn = nullptr;
    {
        PERF_SCOPE("gdn/recurrence");
        cuda::gdn_recurrence_launch(q0, k0, v0, decay, beta, state_dev, seq_len, num_heads,
                                   head_dim, cfg.rms_norm_eps, attn);
    }
    if (!attn)
        return nullptr;

    TensorPtr out = nullptr;
    {
        PERF_SCOPE("gdn/out_gate");
        if (!fused_done)
            out_gate = ops::matmul_transB(normed, lw.ssm_g_a());  // [seq_len, d_inner]
        // nw 取自逐层 init 时上传的设备端常量 (零拷贝)。
        cuda::gdn_output_gate_launch(attn, out_gate, static_cast<const float*>(gc.nw_d->data()),
                                    seq_len, num_heads, head_dim, cfg.rms_norm_eps, out);
    }
    if (!out)
        return nullptr;

    {
        PERF_SCOPE("gdn/wo");
        return ops::matmul_transB(out, lw.wo());
    }
}
#endif  // USE_CUDA

// ============================================================================
// FFN
// ============================================================================

TensorPtr Bailingmoe3Engine::apply_ffn(const TensorPtr& attn_residual,
                                       const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    if (lctx.layer_idx < cfg.leading_dense_block_count) {
        if (!lctx.weights.w1() || !lctx.weights.w2() || !lctx.weights.w3()) {
            LOG_WARN("Bailingmoe3Engine: dense FFN layer " + std::to_string(lctx.layer_idx) +
                     " missing w1/w2/w3");
            return attn_residual;
        }
        return apply_dense_ffn(attn_residual, lctx);
    }
    if (!lctx.weights.ffn_gate_exps() && !lctx.weights.ffn_gate_inp()) {
        LOG_WARN("Bailingmoe3Engine: MoE FFN layer " + std::to_string(lctx.layer_idx) +
                 " missing MoE weights");
        return attn_residual;
    }
    return apply_moe_ffn(attn_residual, lctx);
}

TensorPtr Bailingmoe3Engine::apply_dense_ffn(const TensorPtr& x,
                                             const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    PERF_SCOPE("layer/ffn_dense");
    TensorPtr ffn_normed = ops::rms_norm(x, lw.ffn_norm(), cfg.rms_norm_eps);
    TensorPtr gate = ops::matmul_transB(ffn_normed, lw.w1());
    TensorPtr up = ops::matmul_transB(ffn_normed, lw.w3());
    TensorPtr gated = ops::silu_multiply(gate, up);
    TensorPtr out = ops::matmul_transB(gated, lw.w2());
    return ops::add(x, out);
}

TensorPtr Bailingmoe3Engine::apply_moe_ffn(const TensorPtr& x,
                                           const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    PERF_SCOPE("layer/moe");

    TensorPtr ffn_normed = ops::rms_norm(x, lw.ffn_norm(), cfg.rms_norm_eps);

    // 共享专家 (SiLU FFN)。
    TensorPtr shared_out;
    {
        TensorPtr s_gate = ops::matmul_transB(ffn_normed, lw.ffn_gate_shexp());
        TensorPtr s_up = ops::matmul_transB(ffn_normed, lw.ffn_up_shexp());
        TensorPtr s_gated = ops::silu_multiply(s_gate, s_up);
        shared_out = ops::matmul_transB(s_gated, lw.ffn_down_shexp());
    }

    // 路由专家: sigmoid 门控 + DeepSeek-V3 式分组选择 + 无偏权重归一化。
    // Mirrors llama.cpp build_moe_ffn (llama-graph.cpp): probs = sigmoid(logits)
    // stay unbiased and are later used as expert weights; only selection_probs
    // get the exp_probs_b bias and the group mask.
    const int n_expert = cfg.n_expert;
    const int n_expert_used = cfg.n_expert_used > 0 ? cfg.n_expert_used : 1;
    const int hidden_dim = cfg.hidden_dim;
    const int n_group = cfg.moe_group_count;
    const int n_group_used = cfg.moe_group_used_count;
    const int n_per_group = n_group > 1 ? n_expert / n_group : 0;

    TensorPtr router_logits = ops::matmul_transB(ffn_normed, lw.ffn_gate_inp());  // [seq, n_expert]
    TensorPtr bias_cpu;  // 保活: ensure_cpu 的临时张量死亡会让 bias 悬空 (UB)
    const float* bias = nullptr;
    if (lw.ffn_exp_probs_b()) {
        bias_cpu = ensure_cpu(lw.ffn_exp_probs_b());
        bias = static_cast<const float*>(bias_cpu->data());
    }

    // 路由 scratch (host 收集数组供回退路径; device 数组供两条路径共用)。
    const int n_ids = seq_len * n_expert_used;
    if (!moe_.idx_h || moe_.idx_h->numel() < n_ids) {
        moe_.idx_h = std::make_shared<Tensor>(DataType::INT32,
                                              std::vector<int64_t>{n_ids}, DeviceType::CPU);
        moe_.w_h = std::make_shared<Tensor>(DataType::FP32,
                                            std::vector<int64_t>{n_ids}, DeviceType::CPU);
        if (cuda_mode_) {
            moe_.idx_d = std::make_shared<Tensor>(DataType::INT32,
                                                  std::vector<int64_t>{n_ids}, DeviceType::CUDA);
            moe_.w_d = std::make_shared<Tensor>(DataType::FP32,
                                                std::vector<int64_t>{n_ids}, DeviceType::CUDA);
        }
    }

    auto gate3d = lw.ffn_gate_exps();
    auto up3d = lw.ffn_up_exps();
    auto down3d = lw.ffn_down_exps();
    const DataType moe_dt = gate3d ? gate3d->dtype() : DataType::FP32;
    // Grouped 路径: 三个专家堆矩阵同 dtype 且为已验证的 IQ 类型、权重在 CUDA 上。
    // IQ3_XXS / IQ1_M: 金标准逐字验证通过。IQ2_S 的 vec_dot_iq2_s_q8_1 ksigns
    // 语义存疑 (实测输出乱码, 上轮已标记 unpack_ksigns 疑点), 在定性修复前
    // 保持逐专家回退路径。
    const bool grouped =
        cuda_mode_ && gate3d && up3d && down3d &&
        gate3d->device() == DeviceType::CUDA && up3d->device() == DeviceType::CUDA &&
        down3d->device() == DeviceType::CUDA && up3d->dtype() == moe_dt &&
        down3d->dtype() == moe_dt &&
        (moe_dt == DataType::IQ3_XXS || moe_dt == DataType::IQ1_M);

    if (grouped) {
#ifdef USE_CUDA
        // 设备端路由: sigmoid + bias + 分组 top-k + renorm, 与 host 逐位同序。
        // 消掉每层 ensure_cpu(router_logits) 的同步 D2H (decode 每步 23 次管线排空)。
        const float* bias_dev = nullptr;
        if (lw.ffn_exp_probs_b())
            bias_dev = static_cast<const float*>(lw.ffn_exp_probs_b()->data());
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

        // gate/up: 激活量化一次, 每 (token, slot) 两个投影。
        {
            PERF_SCOPE("moe/gateup");
            cuda::launch_moe_expert_iq_gateup_dt(
                moe_dt, static_cast<const float*>(ffn_normed->data()), gate3d->data(),
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
        // down: 逐 token 加权累加, slot 顺序确定性覆盖写 (内部重量化 gated)。
        {
            PERF_SCOPE("moe/down");
            cuda::launch_moe_expert_iq_down_dt(
                moe_dt, static_cast<const float*>(moe_.gated_out->data()), down3d->data(),
                static_cast<float*>(moe_.expert_out->data()),
                static_cast<const int*>(moe_.idx_d->data()),
                static_cast<const float*>(moe_.w_d->data()), n_ff, hidden_dim, n_expert,
                n_expert_used, seq_len, 0);
        }

        // Residual: llama.cpp does `cur = add(moe, shared); cur = add(cur, ffn_inp)`.
        auto ffn_out = shared_out ? ops::add(shared_out, moe_.expert_out) : moe_.expert_out;
        return ops::add(x, ffn_out);
#endif
    }

    // ---- Host 路由 (CPU 路径与 CUDA 逐专家回退路径) ----
    // 分组 top-k 是 host 控制流, 需要读 logits: CUDA 模式下把 [seq, n_expert]
    // 这个极小的张量拉回 host, 数值与 CPU 路径完全一致。
    TensorPtr rl_host = cuda_mode_ ? ensure_cpu(router_logits) : router_logits;
    const float* rl = static_cast<const float*>(rl_host->data());
    int* h_idx = static_cast<int*>(moe_.idx_h->data());
    float* h_w = static_cast<float*>(moe_.w_h->data());

    for (int s = 0; s < seq_len; ++s) {
        const float* lrow = rl + s * n_expert;
        std::vector<float> probs(n_expert);
        std::vector<float> sel(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            probs[e] = 1.0f / (1.0f + std::exp(-lrow[e]));
            sel[e] = bias ? probs[e] + bias[e] : probs[e];
        }

        // Group routing: score each group by the sum of its top-2 selection
        // probs, keep the top n_group_used groups, mask all other experts out.
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
            // norm_w: divide by the sum of top-k unbiased probs, clamped to the
            // smallest F16 (llama.cpp: 6.103515625e-5), then scale.
            float w;
            if (cfg.moe_weights_norm) {
                const float denom = std::max(topk_sum, 6.103515625e-5f);
                w = probs[e] / denom;
            } else {
                w = probs[e];
            }
            if (cfg.moe_weights_scale != 0.0f && cfg.moe_weights_scale != 1.0f)
                w *= cfg.moe_weights_scale;
            h_idx[s * n_expert_used + k] = e;
            h_w[s * n_expert_used + k] = w;
        }
    }

    // 回退路径: 逐专家 (非 IQ dtype / 权重不在 CUDA)。路由结果读 h_idx/h_w。
    auto expert_out = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len, hidden_dim},
        cuda_mode_ ? DeviceType::CUDA : DeviceType::CPU);
    float* eo = static_cast<float*>(expert_out->data());
    if (cuda_mode_) {
#ifdef USE_CUDA
        cudaMemset(eo, 0, static_cast<size_t>(seq_len) * hidden_dim * sizeof(float));
#endif
    } else {
        std::fill_n(eo, static_cast<size_t>(seq_len) * hidden_dim, 0.0f);
    }

    for (int s = 0; s < seq_len; ++s) {
        float* orow = eo + s * hidden_dim;
        for (int k = 0; k < n_expert_used; ++k) {
            int e = h_idx[s * n_expert_used + k];
            float w = h_w[s * n_expert_used + k];

            // Expert weights are already [n_expert, rows, cols] contiguous; do NOT
            // flatten_rows() here — that collapses to [n_expert, rows*cols] and
            // extract_expert() can no longer recover the per-expert sub-shape.
            // Borrowed quantized views feed matmul_transB's fused dot kernels
            // directly (zero dequant per token; llama.cpp mul_mat semantics).
            auto gw = ops::expert_quant_view(lw.ffn_gate_exps(), e, n_expert);
            auto uw = ops::expert_quant_view(lw.ffn_up_exps(), e, n_expert);
            auto dw = ops::expert_quant_view(lw.ffn_down_exps(), e, n_expert);
            if (!gw || !uw || !dw)
                continue;

            // 每 token 输入行: CUDA 模式下在设备内做 D2D 拷贝, 让专家 GEMM
            // 全部留在 GPU (专家权重也在设备上)。
            TensorPtr th = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{1, hidden_dim},
                cuda_mode_ ? DeviceType::CUDA : DeviceType::CPU);
            const char* row_src =
                static_cast<const char*>(ffn_normed->data()) +
                static_cast<int64_t>(s) * hidden_dim * sizeof(float);
            if (cuda_mode_) {
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
#ifdef USE_CUDA
            if (cuda_mode_) {
                // 设备端加权累加: expert_out[s,:] += w * er
                cuda::moe_accumulate_launch(orow, static_cast<const float*>(er->data()),
                                            w, hidden_dim);
            } else
#endif
            {
                const float* erd = static_cast<const float*>(er->data());
                for (int d = 0; d < hidden_dim; ++d)
                    orow[d] += w * erd[d];
            }
        }
    }

    // Residual: llama.cpp does `cur = add(moe, shared); cur = add(cur, ffn_inp)`.
    // expert_out 已在对应设备上 (CUDA 模式下全程设备累加), 直接与设备张量相加。
    auto ffn_out = shared_out ? ops::add(shared_out, expert_out) : expert_out;
    return ops::add(x, ffn_out);
}

}  // namespace forge

// Registration lives in src/model/arch_registrations.cpp, alongside every other
// architecture. Registering centrally (not via FORGE_REGISTER_ENGINE here) is
// what references Bailingmoe3Engine's constructor and force-links this
// translation unit out of the static library.
