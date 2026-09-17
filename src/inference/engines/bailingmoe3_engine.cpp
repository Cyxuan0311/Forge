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

namespace forge {

#include <algorithm>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// BailingMoE3 debug probe (FORGE_BM3_PROBE=1): dump per-layer tensor statistics
// to stderr so a bad stage can be bisected without a debugger.
// ---------------------------------------------------------------------------
namespace {
bool bm3_probe_on() {
    static int v = -1;
    if (v < 0) {
        const char* s = std::getenv("FORGE_BM3_PROBE");
        v = (s && s[0] == '1') ? 1 : 0;
    }
    return v == 1;
}
void bm3_dump(const char* tag, int layer, const TensorPtr& t) {
    if (!bm3_probe_on()) return;
    if (!t) {
        std::fprintf(stderr, "[BM3] %-10s L%-2d = <null>\n", tag, layer);
        return;
    }
    auto cpu = ensure_cpu(t);
    // Quantized tensors pack several elements per block: reinterpreting their
    // bytes as float* would read past the buffer. Dequantize first.
    if (is_quantized_type(cpu->dtype())) {
        int64_t total = cpu->numel();
        Tensor view_t = cpu->view(std::vector<int64_t>{1, total});
        cpu = ops::dequantize_weight(std::make_shared<Tensor>(std::move(view_t)));
        if (!cpu) {
            std::fprintf(stderr, "[BM3] %-10s L%-2d = <dequant failed>\n", tag, layer);
            return;
        }
    }
    const float* p = static_cast<const float*>(cpu->data());
    const int64_t n = cpu->numel();
    double ss = 0.0;
    float mx = 0.0f;
    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float v = p[i];
        if (!std::isfinite(v)) { ++bad; continue; }
        ss += static_cast<double>(v) * v;
        const float av = std::fabs(v);
        if (av > mx) mx = av;
    }
    std::fprintf(stderr,
                 "[BM3] %-10s L%-2d n=%-7lld rms=%.6e absmax=%.6e nonfinite=%lld head=[%.5f %.5f %.5f %.5f]\n",
                 tag, layer, static_cast<long long>(n),
                 std::sqrt(ss / static_cast<double>(n > 0 ? n : 1)), mx,
                 static_cast<long long>(bad),
                 n > 0 ? p[0] : 0.0f, n > 1 ? p[1] : 0.0f,
                 n > 2 ? p[2] : 0.0f, n > 3 ? p[3] : 0.0f);
}
}  // namespace

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
    latent_hist_.resize(n_layers_);
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

void Bailingmoe3Engine::set_gpu_layers(int /*gpu_layers*/) {
    gpu_layers_ = 0;
    layer_devices_.assign(static_cast<size_t>(n_layers_), DeviceTarget::cpu());
}

void Bailingmoe3Engine::reset() {
    for (auto& m : latent_hist_)
        m.clear();
    recurrent_memory_.reset();
    set_kv_cache_initialized(false);
}

// ============================================================================
// forward_request (CPU head)
// ============================================================================

TensorPtr Bailingmoe3Engine::forward_request(const ForwardRequest& req) {
    cpu::set_num_threads(ctx_.params().n_threads);
    init_kv_cache(model_.config());

    DeviceType dev = DeviceType::CPU;
    auto ids = transfer_hidden(req.input_ids, dev);
    auto token_emb = weights_.token_embedding ? weights_.token_embedding
                                             : model_.weights().get("token_embedding");
    TensorPtr hidden = ops::embedding(ensure_cpu(token_emb), ids, weights_.token_embedding_fp32);
    if (!hidden)
        return nullptr;

    bm3_dump("embd", -1, hidden);

    // 输出头: RMSNorm + lm_head。
    auto out_w = weights_.output_weight;
    if (!out_w && model_.config().tie_embeddings)
        out_w = weights_.token_embedding;

    // Logit-lens probe: project each deep layer's hidden through the lm_head and
    // report the argmax of the last position. The layer where the prediction
    // diverges from a sensible continuation is the layer to investigate.
    static const bool lens_on = (std::getenv("FORGE_BM3_LENS") != nullptr);

    for (int layer = 0; layer < n_layers_; ++layer) {
        hidden = forward_layer(hidden, make_layer_context(layer, req, DeviceTarget::cpu()));
        if (!hidden)
            return nullptr;

        if (lens_on && layer >= 12) {
            auto ln = ops::rms_norm(hidden, ensure_cpu(weights_.output_norm),
                                    model_.config().rms_norm_eps);
            auto lg = ops::matmul_transB(ln, ensure_cpu(out_w));
            if (lg) {
                const float* lp = static_cast<const float*>(ensure_cpu(lg)->data());
                const int64_t vocab = lg->shape().back();
                const int64_t last = (lg->numel() / vocab) - 1;
                const float* row = lp + last * vocab;
                int best = 0;
                float bv = row[0];
                for (int64_t v = 1; v < vocab; ++v) {
                    if (row[v] > bv) {
                        bv = row[v];
                        best = (int)v;
                    }
                }
                std::fprintf(stderr, "[LENS] L%d top1=%d(%.4f)\n", layer, best, bv);
            }
        }
    }
    bm3_dump("final_hid", n_layers_, hidden);
    auto normed = ops::rms_norm(hidden, ensure_cpu(weights_.output_norm), model_.config().rms_norm_eps);
    bm3_dump("final_nrm", n_layers_, normed);
    auto logits = ops::matmul_transB(normed, ensure_cpu(out_w));
    bm3_dump("logits", n_layers_, logits);
    if (bm3_probe_on() && logits) {
        const float* lp = static_cast<const float*>(ensure_cpu(logits)->data());
        const int64_t vocab = logits->shape().back();
        const int64_t last = (logits->numel() / vocab) - 1;
        const float* row = lp + last * vocab;
        int best[5] = {0, 0, 0, 0, 0};
        float bv[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
        for (int64_t v = 0; v < vocab; ++v) {
            for (int k = 0; k < 5; ++k) {
                if (row[v] > bv[k]) {
                    for (int q = 4; q > k; --q) { bv[q] = bv[q - 1]; best[q] = best[q - 1]; }
                    bv[k] = row[v]; best[k] = (int)v;
                    break;
                }
            }
        }
        std::fprintf(stderr, "[BM3] logits_argmax last_row top5:");
        for (int k = 0; k < 5; ++k) std::fprintf(stderr, " %d(%.3f)", best[k], bv[k]);
        std::fprintf(stderr, "\n");

        // Optional: report score/rank of a specific token (e.g. the reference
        // continuation) to quantify how far the prediction has drifted.
        if (const char* tgt = std::getenv("FORGE_BM3_TARGET")) {
            const int tid = std::atoi(tgt);
            if (tid >= 0 && tid < (int)vocab) {
                const float sv = row[tid];
                int rank = 0;
                for (int64_t v = 0; v < vocab; ++v)
                    if (row[v] > sv) ++rank;
                std::fprintf(stderr, "[BM3] target %d score=%.4f rank=%d (top1=%.4f)\n",
                             tid, sv, rank, bv[0]);
            }
        }
    }
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
        ops::rms_norm(ensure_cpu(hidden), lw.attn_norm(), lctx.config.rms_norm_eps);

    bool is_recurrent = (lctx.layer_idx < static_cast<int>(lctx.config.head_count_kv.size())) &&
                        lctx.config.head_count_kv[lctx.layer_idx] == 0;

    bm3_dump("hidden_in", lctx.layer_idx, hidden);
    bm3_dump("normed", lctx.layer_idx, normed);
    bm3_dump(is_recurrent ? "gdn_attn" : "mla_attn", lctx.layer_idx, nullptr);  // marker

    TensorPtr attn_out = is_recurrent ? apply_gdn(normed, lctx) : apply_mla(normed, lctx);
    if (!attn_out)
        return hidden;
    bm3_dump("attn_out", lctx.layer_idx, attn_out);

    auto attn_residual = ops::add(ensure_cpu(hidden), attn_out);
    bm3_dump("attn_res", lctx.layer_idx, attn_residual);
    auto f = apply_ffn(attn_residual, lctx);
    bm3_dump("layer_out", lctx.layer_idx, f);
    return f;
}

// ============================================================================
// Absorbed MLA branch
// ============================================================================

// out[head * dim_out + dout] = Σ_din T[din, dout, head] * x[din]
// T: dequantized FP32, stored [dim_in, dim_out, n_head] -> flat [dim_in*dim_out, n_head].
// Contract a per-head weight tensor T with one input vector x:
//   out[h][dout] = sum_din T[h][dout][din] * x[din]
// T's logical shape is [n_head, dim_out, dim_in]. Forge stores tensor shapes
// outermost-first (reversed vs. the GGUF ne[] array), so the flat row-major
// offset of T[h][dout][din] is ((h * dim_out) + dout) * dim_in + din.
static void contracted_per_head(const float* T, int dim_in, int dim_out, int n_head,
                                 const float* x, float* out) {
    for (int h = 0; h < n_head; ++h)
        for (int dout = 0; dout < dim_out; ++dout) {
            float acc = 0.0f;
            const int64_t base = (static_cast<int64_t>(h) * dim_out + dout) * dim_in;
            for (int din = 0; din < dim_in; ++din)
                acc += T[base + din] * x[din];
            out[static_cast<int64_t>(h) * dim_out + dout] = acc;
        }
}

// Dequantize a (possibly multi-dim) quantized tensor to FP32 in row-major order.
// We flatten to [1, numel] so the stock 2-D dequant path processes the whole
// tensor as one row (numel is a multiple of the quant block size); the resulting
// flat order matches T[h][dout][din] addressed via contracted_per_head().
static TensorPtr dequant_lead2(const TensorPtr& t) {
    auto cpu = ensure_cpu(t);
    if (!is_quantized_type(cpu->dtype()))
        return cpu;
    const int64_t total = cpu->numel();
    Tensor view_t = cpu->view(std::vector<int64_t>{1, total});
    return ops::dequantize_weight(std::make_shared<Tensor>(std::move(view_t)));
}

TensorPtr Bailingmoe3Engine::apply_mla(const TensorPtr& normed, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const int64_t start_pos = lctx.start_pos();
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
    TensorPtr q_flat = ops::matmul_transB(q_low_norm, ops::flatten_rows(lw.wq_b()));
    if (!q_flat)
        return nullptr;

    // K/V latent: c = kv_a_proj @ x, c = [kv_lora (nope) | rope_dim (rope)]
    TensorPtr c_cur = ops::matmul_transB(normed, lw.kv_a_proj());  // [seq, latent_dim]

    // 拼接历史 latent (绝对位置 0..total-1)。
    TensorPtr latent_all;
    auto& hist = latent_hist_[layer_idx];
    auto hit = hist.find(seq_id);
    int total = 0;
    {
        const int cur_n = static_cast<int>(c_cur->shape()[0]);
        if (hit != hist.end() && hit->second) {
            const int prev_n = static_cast<int>(hit->second->shape()[0]);
            total = prev_n + cur_n;
            latent_all = std::make_shared<Tensor>(DataType::FP32,
                                                 std::vector<int64_t>{total, latent_dim},
                                                 DeviceType::CPU);
            float* dst = static_cast<float*>(latent_all->data());
            const float* prev = static_cast<const float*>(hit->second->data());
            const float* cur = static_cast<const float*>(c_cur->data());
            std::memcpy(dst, prev, static_cast<size_t>(prev_n) * latent_dim * sizeof(float));
            std::memcpy(dst + static_cast<size_t>(prev_n) * latent_dim, cur,
                       static_cast<size_t>(cur_n) * latent_dim * sizeof(float));
            hit->second = latent_all;
        } else {
            total = cur_n;
            latent_all = c_cur;
            hist[seq_id] = c_cur;
        }
    }

    // 拆分 nope / rope 部分。
    auto c_nope = std::make_shared<Tensor>(DataType::FP32,
                                          std::vector<int64_t>{total, kv_lora}, DeviceType::CPU);
    auto c_rope = std::make_shared<Tensor>(DataType::FP32,
                                          std::vector<int64_t>{total, rope_dim}, DeviceType::CPU);
    {
        const float* lat = static_cast<const float*>(latent_all->data());
        float* nope = static_cast<float*>(c_nope->data());
        float* rope = static_cast<float*>(c_rope->data());
        for (int r = 0; r < total; ++r) {
            std::memcpy(nope + r * kv_lora, lat + r * latent_dim, kv_lora * sizeof(float));
            std::memcpy(rope + r * rope_dim, lat + r * latent_dim + kv_lora,
                       rope_dim * sizeof(float));
        }
    }

    TensorPtr c_nope_norm = ops::rms_norm(c_nope, lw.kv_a_norm(), cfg.rms_norm_eps);

    // Absorbed MLA (llama.cpp): rather than materialising per-head keys of width
    // qk_nope_dim, the query's nope slice is pushed through attn_k_b[h] so both
    // sides meet in the shared kv_lora space. The value side stays per head:
    //   v[head][token] = Σ_m attn_v_b[m, d, head] * c_nope_norm[token][m]
    // attn_v_b is stored [kv_lora, v_head_dim, n_head] (head last) and is
    // quantized, so dequantize to FP32 [kv_lora*v_dim, n_head] first.
    auto vb_fp32 = dequant_lead2(lw.attn_v_b());  // [kv_lora*v_dim, n_head] FP32
    const float* vb_d = static_cast<const float*>(vb_fp32->data());
    const float* cnp = static_cast<const float*>(ensure_cpu(c_nope_norm)->data());

    auto v_all = std::make_shared<Tensor>(DataType::FP32,
                                          std::vector<int64_t>{total, n_heads * v_dim},
                                          DeviceType::CPU);
    float* vall = static_cast<float*>(v_all->data());
    std::vector<float> v_row(static_cast<size_t>(n_heads) * v_dim);
    for (int j = 0; j < total; ++j) {
        const float* cj = cnp + static_cast<int64_t>(j) * kv_lora;
        contracted_per_head(vb_d, kv_lora, v_dim, n_heads, cj, v_row.data());
        std::memcpy(vall + static_cast<int64_t>(j) * (n_heads * v_dim), v_row.data(),
                    static_cast<size_t>(n_heads) * v_dim * sizeof(float));
    }

    const float* qf = static_cast<const float*>(q_flat->data());
    const float* vV = static_cast<const float*>(v_all->data());
    const float* kR = static_cast<const float*>(c_rope->data());
    // attn_k_b is stored [n_head, kv_lora, qk_nope] (head first), quantized.
    if (bm3_probe_on() && lw.attn_k_b()) {
        const auto& sh = lw.attn_k_b()->shape();
        std::fprintf(stderr, "[BM3] attn_k_b shape ndim=%zu:", sh.size());
        for (size_t i = 0; i < sh.size(); ++i) std::fprintf(stderr, " %lld", (long long) sh[i]);
        std::fprintf(stderr, "  (expect n_head=%d dim_out=kv_lora=%d dim_in=qk_nope=%d)\n",
                     n_heads, kv_lora, qk_nope_dim);
    }
    if (bm3_probe_on() && lw.attn_v_b()) {
        const auto& sh = lw.attn_v_b()->shape();
        std::fprintf(stderr, "[BM3] attn_v_b shape ndim=%zu:", sh.size());
        for (size_t i = 0; i < sh.size(); ++i) std::fprintf(stderr, " %lld", (long long) sh[i]);
        std::fprintf(stderr, "  (expect n_head=%d dim_out=v_dim=%d dim_in=kv_lora=%d)\n",
                     n_heads, v_dim, kv_lora);
    }
    auto kb_fp32 = dequant_lead2(lw.attn_k_b());  // [n_head, kv_lora, qk_nope] FP32
    const float* kb_d = static_cast<const float*>(kb_fp32->data());
    const int v_stride = n_heads * v_dim;
    const int q_stride = n_heads * qk_head_dim;

    auto out = std::make_shared<Tensor>(DataType::FP32,
                                       std::vector<int64_t>{seq_len, v_stride}, DeviceType::CPU);
    float* odata = static_cast<float*>(out->data());

    // Output gate: sigmoid(x @ attn_gate) — one scalar per (token, head), then
    // broadcast over that head's v_dim entries (llama.cpp: mul(cur, sigmoid(gate))).
    TensorPtr gate_t = ops::matmul_transB(normed, lw.attn_gate());  // [seq_len, n_heads]
    ops::sigmoid_inplace(gate_t);
    const float* gptr = gate_t ? static_cast<const float*>(gate_t->data()) : nullptr;

    std::vector<float> scores(total);
    std::vector<float> q_abs_full(static_cast<size_t>(n_heads) * kv_lora);
    std::vector<float> q_rope_rot(static_cast<size_t>(rope_dim));
    std::vector<float> k_pe_rot(static_cast<size_t>(rope_dim));

    for (int s = 0; s < seq_len; ++s) {
        const int64_t qi = start_pos + s;
        const float* q_heads = qf + static_cast<int64_t>(s) * q_stride;
        float* o_heads = odata + static_cast<int64_t>(s) * v_stride;
        for (int h = 0; h < n_heads; ++h) {
            const float* q_nope = q_heads + h * qk_head_dim;
            const float* q_rope = q_heads + h * qk_head_dim + qk_nope_dim;
            std::memcpy(q_rope_rot.data(), q_rope, q_rope_rot.size() * sizeof(float));
            ops::apply_rope_interleaved(q_rope_rot.data(), rope_dim, qi, theta);

            // q_abs = attn_k_b[h] @ q_nope : [kv_lora, qk_nope_dim] x [qk_nope_dim]
            contracted_per_head(kb_d, qk_nope_dim, kv_lora, n_heads, q_nope,
                                q_abs_full.data());
            const float* q_abs = q_abs_full.data() + static_cast<int64_t>(h) * kv_lora;

            const float* vV_h = vV + h * v_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < total; ++j) {
                if (j > qi)
                    continue;  // 因果掩码 (latent_all 按绝对位置索引)
                float dot_nope = 0.0f, dot_rope = 0.0f;
                const float* cj = cnp + static_cast<int64_t>(j) * kv_lora;
                for (int m = 0; m < kv_lora; ++m)
                    dot_nope += q_abs[m] * cj[m];
                // Each history key gets RoPE at its own absolute position.
                const float* k_pe = kR + static_cast<int64_t>(j) * rope_dim;
                std::memcpy(k_pe_rot.data(), k_pe, k_pe_rot.size() * sizeof(float));
                ops::apply_rope_interleaved(k_pe_rot.data(), rope_dim, j, theta);
                for (int d = 0; d < rope_dim; ++d)
                    dot_rope += q_rope_rot[d] * k_pe_rot[d];
                float sc = (dot_nope + dot_rope) * scale;
                scores[j] = sc;
                if (sc > max_score)
                    max_score = sc;
            }
            float sum = 0.0f;
            for (int j = 0; j < total; ++j) {
                if (j > qi)
                    continue;
                float e = std::exp(scores[j] - max_score);
                scores[j] = e;
                sum += e;
            }
            const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            float* o = o_heads + h * v_dim;
            for (int d = 0; d < v_dim; ++d)
                o[d] = 0.0f;
            for (int j = 0; j < total; ++j) {
                if (j > qi)
                    continue;
                float w = scores[j] * inv_sum;
                const float* vv = vV_h + j * v_stride;
                for (int d = 0; d < v_dim; ++d)
                    o[d] += w * vv[d];
            }
            // Self-check: at s == 0 and qi == 0 the causal mask admits exactly
            // one key (j == 0), so softmax weight is 1 and o must equal v[0].
            if (bm3_probe_on() && s == 0 && qi == 0 && h == 0) {
                std::fprintf(stderr,
                             "[BM3] mla_selfchk L%d o[0..3]=%g %g %g %g  v0[0..3]=%g %g %g %g\n",
                             layer_idx, o[0], o[1], o[2], o[3], vV_h[0], vV_h[1], vV_h[2], vV_h[3]);
            }

            if (gptr) {
                const float gv = gptr[static_cast<int64_t>(s) * n_heads + h];
                for (int d = 0; d < v_dim; ++d)
                    o[d] *= gv;
            }
        }
    }
    return ops::matmul_transB(out, lw.wo());
}

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
    bm3_dump("gdn_q_raw", layer_idx, q0);
    bm3_dump("gdn_decay_w", layer_idx, lw.ssm_f_a());
    bm3_dump("gdn_beta_w", layer_idx, lw.ssm_beta());
    bm3_dump("gdn_ga_w", layer_idx, lw.ssm_g_a());
    bm3_dump("gdn_norm_w", layer_idx, lw.ssm_norm());
    bm3_dump("gdn_wo_w", layer_idx, lw.wo());
    bm3_dump("gdn_conv_q_w", layer_idx, lw.ssm_conv1d_q());

    ops::conv1d_depthwise_silu(q0, conv_w_q, conv_q, seq_len, d_inner, d_conv);
    ops::conv1d_depthwise_silu(k0, conv_w_k, conv_k, seq_len, d_inner, d_conv);
    ops::conv1d_depthwise_silu(v0, conv_w_v, conv_v, seq_len, d_inner, d_conv);
    bm3_dump("gdn_q_conv", layer_idx, q0);
    bm3_dump("gdn_v_conv", layer_idx, v0);

    // ---- 2) decay gate, per (head, dim) ----
    TensorPtr decay = ops::matmul_transB(normed, lw.ssm_f_a());  // [seq_len, d_inner]
    ops::build_decay_gate(decay, lw.ssm_dt(), lw.ssm_a(), seq_len, num_heads, head_dim,
                          cfg.kda_gate_lower_bound);
    bm3_dump("gdn_decay", layer_idx, decay);

    // ---- 3) beta input gate, per head ----
    TensorPtr beta = ops::matmul_transB(normed, lw.ssm_beta());  // [seq_len, num_heads]
    ops::sigmoid_inplace(beta);
    bm3_dump("gdn_beta", layer_idx, beta);

    // ---- 4+5) per-head gated delta rule ----
    float* state = recurrent_memory_.ssm_state_cpu(seq_id, layer_idx);
    if (!state)
        return nullptr;
    TensorPtr attn = ops::gdn_recurrence(q0, k0, v0, decay, beta, state, seq_len,
                                         num_heads, head_dim, cfg.rms_norm_eps);
    if (!attn)
        return nullptr;
    bm3_dump("gdn_rec", layer_idx, attn);

    // ---- 6) per-head RMSNorm + output gate ----
    TensorPtr out_gate = ops::matmul_transB(normed, lw.ssm_g_a());  // [seq_len, d_inner]
    bm3_dump("gdn_ologit", layer_idx, out_gate);
    TensorPtr out = ops::gdn_output_gate(attn, out_gate, lw.ssm_norm(), seq_len,
                                         num_heads, head_dim, cfg.rms_norm_eps);
    if (!out)
        return nullptr;
    bm3_dump("gdn_ogate", layer_idx, out);

    // ---- 7) back to hidden_dim ----
    return ops::matmul_transB(out, lw.wo());
}

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
    const float* bias = nullptr;
    if (lw.ffn_exp_probs_b())
        bias = static_cast<const float*>(ensure_cpu(lw.ffn_exp_probs_b())->data());

    auto expert_out = std::make_shared<Tensor>(DataType::FP32,
                                              std::vector<int64_t>{seq_len, hidden_dim},
                                              DeviceType::CPU);
    float* eo = static_cast<float*>(expert_out->data());
    std::fill_n(eo, static_cast<size_t>(seq_len) * hidden_dim, 0.0f);

    const float* rl = static_cast<const float*>(router_logits->data());
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

        const float* moe_in =
            static_cast<const float*>(ffn_normed->data()) + s * hidden_dim;
        float* orow = eo + s * hidden_dim;
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

            // Expert weights are already [n_expert, rows, cols] contiguous; do NOT
            // flatten_rows() here — that collapses to [n_expert, rows*cols] and
            // extract_expert() can no longer recover the per-expert sub-shape.
            auto gw = ops::extract_expert(lw.ffn_gate_exps(), e, n_expert);
            auto uw = ops::extract_expert(lw.ffn_up_exps(), e, n_expert);
            auto dw = ops::extract_expert(lw.ffn_down_exps(), e, n_expert);
            if (!gw || !uw || !dw)
                continue;

            TensorPtr th = std::make_shared<Tensor>(DataType::FP32,
                                                   std::vector<int64_t>{1, hidden_dim},
                                                   DeviceType::CPU);
            std::memcpy(th->data(), moe_in, hidden_dim * sizeof(float));
            TensorPtr g = ops::matmul_transB(th, gw);
            TensorPtr u = ops::matmul_transB(th, uw);
            TensorPtr gated = ops::silu_multiply(g, u);
            TensorPtr er = ops::matmul_transB(gated, dw);
            const float* erd = static_cast<const float*>(er->data());
            for (int d = 0; d < hidden_dim; ++d)
                orow[d] += w * erd[d];
        }
    }

    // Residual: llama.cpp does `cur = add(moe, shared); cur = add(cur, ffn_inp)`.
    auto ffn_out = shared_out ? ops::add(shared_out, expert_out) : expert_out;
    return ops::add(x, ffn_out);
}

}  // namespace forge

// Registration lives in src/model/arch_registrations.cpp, alongside every other
// architecture. Registering centrally (not via FORGE_REGISTER_ENGINE here) is
// what references Bailingmoe3Engine's constructor and force-links this
// translation unit out of the static library.
