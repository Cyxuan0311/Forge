#include "forge/engines/dflash_engine.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "forge/fp16_compat.h"
#include "forge/inference/layers/ffn_executor.h"
#include "forge/inference/layers/norm_executor.h"
#include "forge/inference/layers/qwen35_recurrent_memory.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
using forge::SyncScopeTimer;
#endif
#include "forge/quant_traits.h"

namespace forge {

namespace {

// Stage an activation/weight for the CPU conv kernel: copy to CPU and convert
// FP16 → FP32 when needed (dtype must be FP32 or FP16).
TensorPtr stage_cpu_f32(const TensorPtr& t) {
    if (!t)
        return nullptr;
    TensorPtr cur = t;
    if (cur->device() != DeviceType::CPU) {
        auto tmp = std::make_shared<Tensor>(cur->dtype(), cur->shape(), DeviceType::CPU);
        tmp->copy_from(*cur);
        cur = tmp;
    }
    if (cur->dtype() == DataType::FP32)
        return cur;
    if (cur->dtype() == DataType::FP16) {
        auto out = std::make_shared<Tensor>(DataType::FP32, cur->shape(), DeviceType::CPU);
        const auto* in = static_cast<const uint16_t*>(cur->data());
        auto* o = static_cast<float*>(out->data());
        for (int64_t i = 0; i < cur->numel(); ++i)
            o[i] = fp16_detail::float_from_bits(fp16_detail::half_to_float_bits(in[i]));
        return out;
    }
    throw std::runtime_error("DFlash2 stage_cpu_f32: unsupported dtype " +
                                 dtype_name(cur->dtype()));
}

// Score-descending top-k over one logits row (bounded insertion; top_k is tiny
// so the vocab scan dominates). Ties keep the earlier candidate.
void dflash2_top_k(const float* row, int64_t vocab, int top_k, std::vector<int32_t>& cand,
                   std::vector<float>& vals) {
    cand.assign(top_k, 0);
    vals.assign(top_k, -std::numeric_limits<float>::infinity());
    for (int64_t v = 0; v < vocab; ++v) {
        const float s = row[v];
        if (s <= vals[top_k - 1])
            continue;
        int p = top_k - 1;
        while (p > 0 && vals[p - 1] < s) {
            vals[p] = vals[p - 1];
            cand[p] = cand[p - 1];
            --p;
        }
        vals[p] = s;
        cand[p] = static_cast<int32_t>(v);
    }
}

}  // namespace

// DFlash2 block-causal dynamic depthwise conv (llama.cpp dflash.cpp
// build_dflash2_conv, row-major port).
//
// Applied twice per attn/ffn sublayer: side 0 transforms the block input (the
// post-norm activation feeding QKV / FFN), side 1 transforms the sublayer
// output (attn proj / FFN out) before the residual add. Weights:
//   dynamic: conv_proj output [n_tokens, 2*kernel*n_groups]
//   base_f32: conv base [2, kernel, n_embd] FP32 (engine-cached)
// For token t, channel c (group g = c / group_size) and tap k:
//   out[t][c] = sum_k (dynamic[t][side*kernel*n_groups + k*n_groups + g]
//                      + base[side][k][c]) * hidden[t-k][c]   (0 when t-k < 0)
// matching the ggml col-major channel order j = g + k*n_groups + side*kernel*n_groups.
// The drafter decodes a single query block per call, so block_size == n_tokens
// and the taps never shift across sequences.
TensorPtr build_dflash2_conv(const TensorPtr& hidden, const TensorPtr& dynamic,
                             const TensorPtr& base_f32, int side, const ModelConfig& cfg) {
    const int64_t n_tokens = hidden->shape()[0];
    const int64_t n_embd = hidden->shape()[1];
    const int64_t kernel = cfg.dflash_conv_kernel_size;
    const int64_t group_size = cfg.dflash_conv_group_size;
    if (group_size <= 0 || n_embd % group_size != 0)
        throw std::runtime_error("DFlash2 conv: n_embd not divisible by conv_group_size");
    const int64_t n_groups = n_embd / group_size;
    if (dynamic->shape()[0] != n_tokens || dynamic->shape()[1] != 2 * kernel * n_groups)
        throw std::runtime_error("DFlash2 conv: dynamic shape mismatch");

    auto h = stage_cpu_f32(hidden);
    auto dyn = stage_cpu_f32(dynamic);
    auto base = stage_cpu_f32(base_f32);
    const float* h_data = static_cast<const float*>(h->data());
    const float* dyn_data = static_cast<const float*>(dyn->data());
    const float* base_data = static_cast<const float*>(base->data());

    auto out = std::make_shared<Tensor>(DataType::FP32,
                                        std::vector<int64_t>{n_tokens, n_embd},
                                        DeviceType::CPU);
    float* out_data = static_cast<float*>(out->data());
    std::memset(out_data, 0, static_cast<size_t>(out->nbytes()));

    const int64_t proj_dim = dyn->shape()[1];
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float* dyn_row = dyn_data + t * proj_dim;
        float* out_row = out_data + t * n_embd;
        for (int64_t k = 0; k < kernel; ++k) {
            const int64_t src = t - k;  // causal: taps stay inside the block
            if (src < 0)
                continue;
            const float* h_row = h_data + src * n_embd;
            const float* base_row = base_data + side * (kernel * n_embd) + k * n_embd;
            for (int64_t c = 0; c < n_embd; ++c) {
                const float coeff =
                    dyn_row[side * (kernel * n_groups) + k * n_groups + c / group_size];
                out_row[c] += (coeff + base_row[c]) * h_row[c];
            }
        }
    }

    // Preserve the caller's layout: keep FP16 chains intact (CUDA native FP16
    // GEMM path), otherwise return FP32 on the input's device.
    if (hidden->dtype() == DataType::FP16) {
        auto h16 = std::make_shared<Tensor>(DataType::FP16, out->shape(), DeviceType::CPU);
        const auto* in = static_cast<const float*>(out->data());
        auto* o = static_cast<uint16_t*>(h16->data());
        for (int64_t i = 0; i < out->numel(); ++i)
            o[i] = fp16_detail::float_to_half_bits(fp16_detail::float_bits(in[i]));
        if (hidden->device() != DeviceType::CPU)
            h16->to_device(hidden->device());
        return h16;
    }
    if (hidden->device() != DeviceType::CPU)
        out->to_device(hidden->device());
    return out;
}

// DFlash2 selector (llama.cpp build_dflash2_selector, CPU port): packs the
// greedy-walk lattice consumed by the provider. Row pos of the result holds
//   [0, top_k)                  candidate token ids (float-cast, score-descending)
//   [top_k, top_k + top_k^2)    transition scores indexed [pred][succ]
// with score[p][s] = <selector_next[cand_s], selector_prev[pred_p] * gate[pos]>
// + unary[s]. Position 1's predecessor is the anchor token id; positions >= 2
// are scored against every candidate of the previous position. Position 0 (the
// anchor itself) is zero-filled — the walk starts at row 1.
TensorPtr build_dflash2_selector(const TensorPtr& logits, const TensorPtr& t_embd,
                                 const ModelConfig& cfg, const TensorPtr& sel_prev,
                                 const TensorPtr& sel_next, const TensorPtr& sel_hidden,
                                 int32_t anchor_id) {
    // CPU-port contract: this builder reads raw host pointers (sel_prev/next)
    // and runs the gate projection on the CPU GEMV path. Stage every input to
    // host FP32 so a fully-GPU drafter (selector weights load onto
    // last_layer_dev = CUDA) cannot trigger a cross-device dereference — the
    // same bug class that hit embed_via_target / write_context_kv. Hot-path
    // callers pass cached host tensors, making this a no-op for them.
    auto logits_h = stage_cpu_f32(logits);
    auto t_embd_h = stage_cpu_f32(t_embd);
    auto prev_h = stage_cpu_f32(sel_prev);
    auto next_h = stage_cpu_f32(sel_next);
    auto hidden_h = stage_cpu_f32(sel_hidden);
    if (!prev_h || !next_h || !hidden_h)
        throw std::runtime_error("DFlash2 selector: selector weights missing");

    const int top_k = cfg.dflash_selector_top_k;
    const int64_t n_tokens = logits_h->shape()[0];
    const int64_t vocab = logits_h->shape()[1];
    if (!t_embd_h || t_embd_h->shape()[0] != n_tokens)
        throw std::runtime_error("DFlash2 selector: t_embd/logits row mismatch");
    if (prev_h->shape()[0] != vocab || prev_h->shape()[1] != next_h->shape()[1] ||
        next_h->shape()[0] != vocab || hidden_h->shape()[0] != prev_h->shape()[1] ||
        hidden_h->shape()[1] != t_embd_h->shape()[1])
        throw std::runtime_error("DFlash2 selector: weight shape mismatch");

    // gate[pos][r] = <selector_hidden[r], t_embd[pos]> : [n_tokens, rank]
    TensorPtr gate = ops::matmul_transB(t_embd_h, hidden_h);
    const int64_t rank = gate->shape()[1];
    const int64_t row_used = top_k + static_cast<int64_t>(top_k) * top_k;
    auto lattice = std::make_shared<Tensor>(DataType::FP32,
                                            std::vector<int64_t>{n_tokens, row_used},
                                            DeviceType::CPU);
    float* lat = static_cast<float*>(lattice->data());
    std::memset(lat, 0, static_cast<size_t>(lattice->nbytes()));

    const float* lg = static_cast<const float*>(logits_h->data());
    const float* gt = static_cast<const float*>(gate->data());
    const float* prev = static_cast<const float*>(prev_h->data());
    const float* next = static_cast<const float*>(next_h->data());

    std::vector<int32_t> cand, prev_cand;
    std::vector<float> vals(top_k), succ(static_cast<size_t>(top_k) * rank), cond(rank);
    for (int64_t pos = 1; pos < n_tokens; ++pos) {
        dflash2_top_k(lg + pos * vocab, vocab, top_k, cand, vals);

        float* row = lat + pos * row_used;
        for (int s = 0; s < top_k; ++s)
            row[s] = static_cast<float>(cand[s]);
        for (int s = 0; s < top_k; ++s)
            std::memcpy(&succ[static_cast<size_t>(s) * rank],
                        next + static_cast<int64_t>(cand[s]) * rank,
                        static_cast<size_t>(rank) * sizeof(float));

        const int32_t anchor_pred[1] = {anchor_id};
        const int32_t* preds = (pos == 1) ? anchor_pred : prev_cand.data();
        const int n_pred = (pos == 1) ? 1 : top_k;
        const float* gate_row = gt + pos * rank;
        float* scores = row + top_k;
        for (int p = 0; p < n_pred; ++p) {
            const float* pr = prev + static_cast<int64_t>(preds[p]) * rank;
            for (int64_t r = 0; r < rank; ++r)
                cond[r] = pr[r] * gate_row[r];
            const float* sr = succ.data();
            for (int s = 0; s < top_k; ++s) {
                float acc = 0.0f;
                for (int64_t r = 0; r < rank; ++r)
                    acc += sr[r] * cond[r];
                scores[p * top_k + s] = acc + vals[s];
                sr += rank;
            }
        }
        prev_cand = cand;
    }
    return lattice;
}

DflashEngine::DflashEngine(Model& model, InferenceContext& ctx)
    : GenericEngine(model, ctx),
      dflash_attn_(kv_cache_, kv_memory_ ? &kv_memory_->storage() : nullptr) {}

TensorPtr DflashEngine::forward_request(const ForwardRequest& req) {
    { SyncScopeTimer t("dfr/0_init_kv_cache"); init_kv_cache(model_.config()); }
    DeviceType first_dev = layer_device(0);
    TensorPtr hidden;
    if (req.input_embeddings) {
        hidden = transfer_hidden(req.input_embeddings, first_dev);
    } else {
        if (!target_) {
            throw std::runtime_error(
                "DflashEngine::forward_request: target not set, "
                "cannot embed tokens");
        }
        auto ids = transfer_hidden(req.input_ids, first_dev);
        hidden = embed_via_target(ids, first_dev);
    }
    // Reuse the base forward_layers: it runs the draft layers (our forward_layer
    // override) and the final norm + lm_head projection (weights_.output_weight
    // is borrowed from the target via set_target()).
    return TransformerEngine::forward_layers(hidden, req);
}

TensorPtr DflashEngine::embed_via_target(const TensorPtr& ids, DeviceType dev) {
    // 查表必须在 embedding 权重所在设备上进行: drafter 层设备与目标权重设备
    // 可能不同 (如 drafter 全 CPU 而目标部分在 GPU, 或继承到不一致的 gpu_layers),
    // ops::embedding 的 CPU/CUDA 分支都假定 weight 与 indices 同设备, 混置时会
    // 在主机端解引用另一侧指针导致段错误。查表后把结果搬到调用方请求的设备。
    auto& tw = static_cast<TransformerEngine*>(target_)->weights();
    const DeviceType wdev = tw.token_embedding ? tw.token_embedding->device() : dev;
    auto ids_on_wdev = transfer_hidden(ids, wdev);
    auto h = ops::embedding(tw.token_embedding, ids_on_wdev, tw.token_embedding_fp32);
    if (h && h->device() != dev) {
        h->to_device(dev);
    }
    return h;
}

TensorPtr DflashEngine::encode(const TensorPtr& target_layer_hiddens) {
    // 方案 A: encode() 只做 fc 投影, encoder 输出 norm (output_norm_enc) 由
    // write_context_kv 统一应用, precompute / append 两条路径行为一致, 避免
    // double norm。take_layer_hiddens 返回 CPU 拼接 hidden, fc 投影要求同设备,
    // 先搬运到 fc 权重所在设备 (权重缺失时退化到引擎首层设备)。
    const DeviceType wdev =
        weights_.dflash_fc ? weights_.dflash_fc->device() : layer_device(0);
    TensorPtr h = ops::matmul_transB(transfer_hidden(target_layer_hiddens, wdev),
                                     weights_.dflash_fc);
    context_feature_ = h;
    return h;
}

void DflashEngine::write_context_kv(const TensorPtr& prefix_embd, int64_t start_pos) {
    const auto& cfg = model_.config();
    // KV cache 采用首次 forward 惰性分配, 但增量追加路径 (append_context_kv)
    // 先于任何 decode 执行, 必须在此确保已分配 (幂等: 已初始化则立即返回),
    // 否则 kv_cache_ 是空壳, update() 报 invalid layer / max_seq_len=0。
    init_kv_cache(cfg);
    // 输入是"已编码 context feature" (append: encode() 的 fc 输出; precompute:
    // raw prefix)。按 vLLM dflash _project_context_kv 语义统一在此应用 encoder
    // 输出 norm (hidden_norm), 再逐层投影。权重缺失时原样透传。
    TensorPtr feature = prefix_embd;
    DeviceType dev = prefix_embd->device();
    if (weights_.dflash_output_norm_enc) {
        TensorPtr enc_norm = weights_.dflash_output_norm_enc;
        if (enc_norm->device() != dev) {
            auto tmp = std::make_shared<Tensor>(enc_norm->dtype(), enc_norm->shape(), dev);
            tmp->copy_from(*enc_norm);
            enc_norm = tmp;
        }
        feature = NormExecutor::apply(feature, enc_norm, nullptr, NormType::RMSNorm,
                                      cfg.rms_norm_eps);
        dev = feature->device();
    }
    const int prefix_len = static_cast<int>(feature->shape()[0]);
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const auto& lw = weights_.layers[layer];
        // 逐层设备对齐: drafter 部分卸载时各层权重分处 CPU/CUDA, 而 feature 只有
        // 一个设备。沿用 forward_layer 的做法, 每层先把 feature 搬到该层权重所在
        // 设备再投影 —— ops::matmul_transB 的两个分支都假定两侧同设备, 混置时会
        // 在主机端解引用对侧指针导致段错误 (drafter --spec-draft-ngl 1/5 时崩溃)。
        const DeviceType ldev = lw.wk() ? lw.wk()->device() : layer_device(layer);
        TensorPtr feat_l = transfer_hidden(feature, ldev);
        TensorPtr K, V;
        {
            PERF_SCOPE("dflash/ctx_proj");
            K = ops::matmul_transB(feat_l, lw.wk());
            V = ops::matmul_transB(feat_l, lw.wv());
        }
        // 逐层 k_norm (per-head RMSNorm), 与 forward_layer 的 QK-Norm 分支一致,
        // 置于 RoPE 之前 (vLLM dflash: fused GEMM -> k_norm -> RoPE)。
        if (cfg.use_qk_norm && lw.attn_k_norm()) {
            PERF_SCOPE("dflash/ctx_kvnorm");
            K = NormExecutor::apply_qk_norm(K, lw.attn_k_norm(), num_kv_heads, head_dim,
                                            cfg.rms_norm_eps, ldev);
        }
        // RoPE applies to K (and Q); build a dummy Q so we can reuse apply() and
        // take the rotated K. V is intentionally left un-rotated. The dummy Q is
        // allocated once and reused across layers: the ROPE result is read from
        // k_rope only, so a zeroed buffer of the right shape is sufficient and
        // saves a per-layer allocation + memset.
        TensorPtr q_dummy = dummy_q(prefix_len, num_heads * head_dim, K->dtype(), ldev);
        TensorPtr k_rope;
        {
            PERF_SCOPE("dflash/ctx_rope");
            auto rope = dflash_rope_.apply(q_dummy, K, cfg, start_pos, prefix_len, ldev);
            k_rope = rope.k_rope;
        }
        {
            PERF_SCOPE("dflash/ctx_cache");
            if (kv_cache_.update(layer, 0, start_pos, k_rope, V, prefix_len) < 0) {
                throw std::runtime_error(
                    "KV cache overflow at layer " + std::to_string(layer) +
                    ": write_pos=" + std::to_string(start_pos) +
                    " seq_len=" + std::to_string(prefix_len) +
                    " exceeds max_seq_len=" + std::to_string(kv_cache_.max_seq_len()));
            }
        }
    }
}

TensorPtr DflashEngine::dummy_q(int prefix_len, int64_t width, DataType dtype, DeviceType dev) {
    if (!dummy_q_ || dummy_q_->shape()[1] != width || dummy_q_->device() != dev) {
        dummy_q_ = std::make_shared<Tensor>(dtype, std::vector<int64_t>{prefix_len, width}, dev);
    }
    if (prefix_len > dummy_q_->shape()[0]) {
        dummy_q_ = std::make_shared<Tensor>(dtype, std::vector<int64_t>{prefix_len, width}, dev);
    }
    dummy_q_->zero_();
    return dummy_q_;
}

void DflashEngine::precompute_context_kv(const TensorPtr& prefix_embd, int64_t start_pos) {
    kv_cache_.reset();
    write_context_kv(prefix_embd, start_pos);
}

void DflashEngine::append_context_kv(const std::vector<int32_t>& delta_ids,
                                     int64_t start_pos) {
    if (delta_ids.empty())
        return;
    if (!target_) {
        throw std::runtime_error(
            "DflashEngine::append_context_kv: target not set, cannot embed delta");
    }
    const auto& cfg = model_.config();
    const int64_t n = static_cast<int64_t>(delta_ids.size());
    auto ids =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1, n}, DeviceType::CPU);
    std::memcpy(ids->data(), delta_ids.data(), static_cast<size_t>(n) * sizeof(int32_t));

    // vLLM DFlash 增量上下文: 在 target 上 forward 这批增量 token, 取目标层
    // hidden, 经 encode() (fc) 编码成 context feature, 再投影进 drafter KV。
    //
    // ---- SSM 状态纪律 (Qwen3.5 hybrid target) ----
    // 这里的 delta forward 是一次"旁路重放": 为 drafter 重新走过 target 已确认
    // 的 token, 绝不能污染 live conv/ssm 状态。状态起点按场景选取:
    //   fresh (start_pos==0): live 状态已被 generator 的 prefill 推进到 prompt 末,
    //     直接重放会双重推进 —— 先 reset_seq 清零再从头走;
    //   增量: live 状态停在 anchor (committed 末位, 上一轮 verify 的终点), 而
    //     delta 必须从 as-of (start_pos-1) 的状态起步 —— 回滚到引擎私有的
    //     "上下文前沿" 快照 (slot 1, 与 executor 的接受点快照 slot 0 互不干扰)。
    // forward 结束后 live 状态 = as-of (start_pos + n - 1), 恰好是下一轮 verify
    // 前向 (从 anchor 位置重跑) 所需的起点, 无需再恢复; 同时把它刷新进 slot 1,
    // 供下一轮增量重放使用。
    auto* t = static_cast<TransformerEngine*>(target_);
    auto* rmem = t->recurrent_memory();
    if (rmem) {
        if (start_pos == 0) {
            rmem->reset_seq(0);
        } else {
            rmem->rollback(0, /*slot=*/1);
        }
    }
    // capture 置位期间 forward_layers 自动退回 imperative (不触发 CUDA graph),
    // 取完立刻还原标志, 避免拖慢后续主流程 forward。
    const bool was_capturing = t->captures_layer_hiddens();
    t->set_capture_layer_hiddens(true);
    t->forward_request(ForwardRequest::from_ids(ids, start_pos));
    TensorPtr hiddens = t->take_layer_hiddens(cfg.target_layers);
    t->set_capture_layer_hiddens(was_capturing);
    if (!hiddens) {
        LOG_WARN("DflashEngine::append_context_kv: no target layer hiddens captured, "
                 "skipping context injection");
        return;
    }
    // 不 reset: 增量追加本批 token 的 K/V (vLLM dflash 的增量上下文模式)。
    encode_and_write_context_kv(hiddens, start_pos);
    if (rmem)
        rmem->snapshot(0, /*slot=*/1);
}

void DflashEngine::append_context_kv_from_hiddens(const TensorPtr& hiddens, int64_t start_pos) {
    if (!hiddens || hiddens->shape()[0] == 0)
        return;
    // Hiddens come from the target's own verify forward, so there is no replay:
    // the live conv/ssm state was already advanced past these positions by that
    // forward and must not be touched here.
    encode_and_write_context_kv(hiddens, start_pos);
}

void DflashEngine::encode_and_write_context_kv(const TensorPtr& hiddens, int64_t start_pos) {
    write_context_kv(encode(hiddens), start_pos);
}

void DflashEngine::precompute_context_kv(const std::vector<int32_t>& prefix_ids,
                                         int64_t start_pos) {
    if (!target_) {
        throw std::runtime_error(
            "DflashEngine::precompute_context_kv(ids): target not set, cannot embed prefix");
    }
    DeviceType dev = layer_device(0);
    const int64_t n = static_cast<int64_t>(prefix_ids.size());
    auto ids =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1, n}, DeviceType::CPU);
    std::memcpy(ids->data(), prefix_ids.data(), static_cast<size_t>(n) * sizeof(int32_t));
    TensorPtr emb = embed_via_target(ids, dev);
    precompute_context_kv(emb, start_pos);
}

TensorPtr DflashEngine::decode(const std::vector<int32_t>& query_ids, int64_t start_pos) {
    DeviceType dev = layer_device(0);
    const int64_t n = static_cast<int64_t>(query_ids.size());
    TensorPtr ids =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1, n}, DeviceType::CPU);
    std::memcpy(ids->data(), query_ids.data(), static_cast<size_t>(n) * sizeof(int32_t));
    TensorPtr emb = embed_via_target(ids, dev);
    // Query-block decode is non-causal: every query attends to the full prefix
    // (injected via precompute_context_kv) plus the entire block.
    decode_noncausal_ = true;
    auto req = ForwardRequest::from_embedding(emb, static_cast<int>(n), start_pos, 0);
    // Greedy sampling and the DFlash2 selector read every row of the block; the
    // sparse-logits fast path (last row only) would truncate the output.
    req.return_all_logits = true;
    TensorPtr logits;
    {
        PERF_SCOPE("dflash/fwd_forward");
        logits = forward_request(req);
    }
    decode_noncausal_ = false;
    // Query-block 行是一次性 KV 占用: block forward 期间行间经各层写出的 K/V
    // 互相可见 (non-causal), forward 结束后 cache 里的 MASK 残行对后续 forward
    // 纯有害 —— attention 以 filled() 为读界且残行带合法 pos 戳, 而下一轮的
    // delta 上下文重放 (append_context_kv) 发生在下一个 query block 之前, 其
    // rows 会把残行 MASK 垃圾当作 context 读入, 污染 captured hiddens → 编码
    // 进 drafter 上下文 KV → 提案崩坏, 在 rejected / adaptive 收缩 (每轮仅
    // 推进 1 token, 残行宽 n_block) 时恶性循环到 0% 接受率。回滚到 anchor 行
    // 之后, 只保留 anchor (下一轮 delta 从 anchor 位置重写, 与增量注入约定
    // 一致)。DFlash / DSPark 的 query block 共用本路径。
    kv_cache_.rollback(start_pos + 1);
    dflash2_lattice_ = nullptr;

    const bool dflash2 = is_dflash2();

    if (dflash2 && logits && logits->ndim() >= 2 && n >= 2) {
        PERF_SCOPE("dflash/stage_cpu_logits");
        const auto& cfg = model_.config();
        // DFlash2 feeds these logits to the selector, so they carry the target's
        // output transforms (upstream dflash.cpp: logit scale, then softcap).
        logits = stage_cpu_f32(logits);
        float* data = static_cast<float*>(logits->data());
        const int64_t total = logits->numel();
        if (cfg.f_logit_scale != 0.0f) {
            for (int64_t i = 0; i < total; ++i)
                data[i] *= cfg.f_logit_scale;
        }
        if (cfg.f_final_logit_softcapping > 0.0f) {
            const float c = cfg.f_final_logit_softcapping;
            for (int64_t i = 0; i < total; ++i)
                data[i] = std::tanh(data[i] / c) * c;
        }
        // Selector lattice: top-k candidates + pairwise transition scores per
        // block position, consumed by the provider's CPU greedy walk.
        TensorPtr t_embd = take_last_hidden();
        if (t_embd) {
            TensorPtr sel_prev = dflash2_selector_weight_cpu(weights_.dflash_selector_prev);
            TensorPtr sel_next = dflash2_selector_weight_cpu(weights_.dflash_selector_next);
            TensorPtr sel_hidden = dflash2_selector_weight_cpu(weights_.dflash_selector_hidden);
            TensorPtr t_embd_cpu = stage_cpu_f32(t_embd);
            dflash2_lattice_ = build_dflash2_selector(
                logits, t_embd_cpu, cfg, sel_prev, sel_next, sel_hidden, query_ids[0]);
        }
    }

    // Expose the draft-model hidden states for the DSpark confidence head. Staged to
    // CPU since the confidence head is CPU-only; valid until the next decode() call.
    TensorPtr h = take_last_hidden();
    if (h && h->device() != DeviceType::CPU) {
        PERF_SCOPE("dflash/d2h_hidden");
        h->to_device(DeviceType::CPU);
    }
    last_hidden_states_ = h;
    return logits;
}

TensorPtr DflashEngine::forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) {
    PERF_SCOPE("dflash/layer_total");
    const auto& cfg = lctx.config;
    const int layer_idx = lctx.layer_idx;
    const int seq_len = lctx.seq_len();
    const int64_t start_pos = lctx.start_pos();
    const DeviceType dev = lctx.device.type;
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;
    const auto& lw = lctx.weights;

    const NormType norm_type = cfg.norm_type;
    const float eps = (norm_type == NormType::LayerNorm) ? cfg.layer_norm_eps : cfg.rms_norm_eps;
    const bool has_qkv_bias = (lw.bq() && lw.bq()->numel() > 0);

    // 1. Pre-attention norm
    TensorPtr pre_attn_norm;
    { SyncScopeTimer t("dfl/1_attn_norm");
      pre_attn_norm = NormExecutor::apply(hidden, lw.attn_norm(), nullptr, norm_type, eps); }

    // DFlash2: block-causal dynamic depthwise conv around each attn/ffn sublayer
    // (upstream dflash.cpp). The dynamic coefficients are projected from the
    // pre-conv activation and shared by both sides of the sublayer.
    const bool d2_attn_conv =
        cfg.dflash_selector_top_k > 0 && lw.dflash_attn_conv_proj() != nullptr;
    TensorPtr attn_dynamic;
    if (d2_attn_conv) {
        attn_dynamic =
            ops::matmul_transB(pre_attn_norm, dflash2_weight_f32(lw.dflash_attn_conv_proj()));
        pre_attn_norm = build_dflash2_conv(pre_attn_norm, attn_dynamic,
                                           dflash2_weight_f32(lw.dflash_attn_conv_base()), 0,
                                           cfg);
    }

    // 2. QKV projection
    AttentionExecutor::QKVResult qkv;
    { SyncScopeTimer t("dfl/2_qkv_proj");
      qkv = AttentionExecutor::project_qkv(pre_attn_norm, lw, has_qkv_bias, dev, seq_len); }
    TensorPtr q = qkv.q, k = qkv.k, v = qkv.v;

    // 3. QK-Norm (per-head RMSNorm)
    { SyncScopeTimer t("dfl/3_qk_norm");
    if (cfg.use_qk_norm) {
        if (lw.attn_q_norm()) {
            q = NormExecutor::apply_qk_norm(q, lw.attn_q_norm(), num_heads, head_dim,
                                            cfg.rms_norm_eps, dev);
        }
        if (lw.attn_k_norm()) {
            k = NormExecutor::apply_qk_norm(k, lw.attn_k_norm(), num_kv_heads, head_dim,
                                            cfg.rms_norm_eps, dev);
        }
    } }

    // 4. RoPE
    RopeExecutor::Result rope;
    { SyncScopeTimer t("dfl/4_rope");
      rope = dflash_rope_.apply(q, k, cfg, start_pos, seq_len, dev); }
    TensorPtr q_rope = rope.q_rope, k_rope = rope.k_rope;

    // 5. KV cache update (contiguous; the drafter owns a non-paged cache)
    { SyncScopeTimer t("dfl/5_kv_update");
    if (kv_cache_.update(layer_idx, lctx.seq_id(), start_pos, k_rope, v, seq_len) < 0) {
        throw std::runtime_error(
            "KV cache overflow at layer " + std::to_string(layer_idx) +
            ": write_pos=" + std::to_string(start_pos) +
            " seq_len=" + std::to_string(seq_len) +
            " exceeds max_seq_len=" + std::to_string(kv_cache_.max_seq_len()));
    } }

    // 6. Attention. Non-causal during the query-block decode.
    const bool causal = !decode_noncausal_;
    TensorPtr attn_out;
    { SyncScopeTimer t("dfl/6_attention");
      attn_out = dflash_attn_.attend(q_rope, cfg, layer_idx, seq_len, dev, nullptr, lctx.seq_id(), causal); }

    // 7. Attention output projection
    TensorPtr attn_proj;
    { SyncScopeTimer t("dfl/7_attn_proj");
      attn_proj = ops::matmul_transB(attn_out, lw.wo()); }
    if (d2_attn_conv) {
        attn_proj = build_dflash2_conv(attn_proj, attn_dynamic,
                                       dflash2_weight_f32(lw.dflash_attn_conv_base()), 1, cfg);
    }

    // 8/9. Sequential residual: norm -> attn -> residual -> ffn -> residual
    TensorPtr hidden_after_attn;
    { SyncScopeTimer t("dfl/8_resid1");
      hidden_after_attn = ops::add(hidden, attn_proj); }
    TensorPtr ffn_normed;
    { SyncScopeTimer t("dfl/9_ffn_norm");
      ffn_normed = NormExecutor::apply(hidden_after_attn, lw.ffn_norm(), nullptr, norm_type, eps); }

    const bool d2_ffn_conv =
        cfg.dflash_selector_top_k > 0 && lw.dflash_ffn_conv_proj() != nullptr;
    TensorPtr ffn_dynamic;
    if (d2_ffn_conv) {
        ffn_dynamic =
            ops::matmul_transB(ffn_normed, dflash2_weight_f32(lw.dflash_ffn_conv_proj()));
        ffn_normed = build_dflash2_conv(ffn_normed, ffn_dynamic,
                                        dflash2_weight_f32(lw.dflash_ffn_conv_base()), 0, cfg);
    }
    TensorPtr ffn_out;
    // No probe around the FFN: it would wrap the ffn/* leaves. Those are the
    // real breakdown of this region.
    ffn_out = FfnExecutor::apply(ffn_normed, hidden_after_attn, cfg, lw, seq_len, dev);
    if (d2_ffn_conv) {
        ffn_out = build_dflash2_conv(ffn_out, ffn_dynamic,
                                     dflash2_weight_f32(lw.dflash_ffn_conv_base()), 1, cfg);
    }
    { SyncScopeTimer t("dfl/11_resid2");
      return ops::add(hidden_after_attn, ffn_out); }
}

TensorPtr DflashEngine::dflash2_weight_f32(const TensorPtr& w) {
    if (!w)
        return nullptr;
    if (w->dtype() == DataType::FP32)
        return w;  // matmul runs natively on any device; the conv stages a CPU copy
    auto it = dflash2_f32_.find(w.get());
    if (it != dflash2_f32_.end())
        return it->second;
    TensorPtr cur = w;
    if (cur->device() != DeviceType::CPU) {
        auto tmp = std::make_shared<Tensor>(cur->dtype(), cur->shape(), DeviceType::CPU);
        tmp->copy_from(*cur);
        cur = tmp;
    }
    TensorPtr f32;
    if (is_quantized_type(cur->dtype())) {
        f32 = ops::dequantize_weight(cur);
        if (!f32 || f32->dtype() != DataType::FP32)
            throw std::runtime_error("DFlash2 conv: cannot dequantize weight dtype " +
                                     dtype_name(cur->dtype()));
    } else if (cur->dtype() == DataType::FP16) {
        f32 = std::make_shared<Tensor>(DataType::FP32, cur->shape(), DeviceType::CPU);
        const auto* in = static_cast<const uint16_t*>(cur->data());
        auto* o = static_cast<float*>(f32->data());
        for (int64_t i = 0; i < cur->numel(); ++i)
            o[i] = fp16_detail::float_from_bits(fp16_detail::half_to_float_bits(in[i]));
    } else if (cur->dtype() == DataType::BF16) {
        // The loader keeps BF16 weights native on CUDA (expanding them doubled the
        // drafter's footprint -- see GgufModel::get_tensor), so a BF16 drafter
        // hands these over as BF16. The conv kernels are scalar CPU loops.
        f32 = std::make_shared<Tensor>(DataType::FP32, cur->shape(), DeviceType::CPU);
        const auto* in = static_cast<const uint16_t*>(cur->data());
        auto* o = static_cast<float*>(f32->data());
        for (int64_t i = 0; i < cur->numel(); ++i) {
            const uint32_t bits = static_cast<uint32_t>(in[i]) << 16;
            std::memcpy(o + i, &bits, sizeof(float));
        }
    } else {
        throw std::runtime_error("DFlash2 conv: unsupported weight dtype " +
                                 dtype_name(cur->dtype()));
    }
    if (w->device() != DeviceType::CPU)
        f32->to_device(w->device());  // keep the conv_proj matmul on the weight's device
    dflash2_f32_.emplace(w.get(), f32);
    return f32;
}

TensorPtr DflashEngine::dflash2_selector_weight_cpu(const TensorPtr& w) {
    if (!w)
        return nullptr;
    auto it = dflash2_f32_cpu_.find(w.get());
    if (it != dflash2_f32_cpu_.end())
        return it->second;
    // dflash2_weight_f32() keeps its result on the source device (the CUDA
    // conv-proj matmul relies on that); the selector lattice is a CPU-only
    // algorithm, so cache a separate host-FP32 copy per source weight. The
    // cross-device transfer is paid once per weight, not on every decode step.
    TensorPtr cpu = stage_cpu_f32(dflash2_weight_f32(w));
    dflash2_f32_cpu_.emplace(w.get(), cpu);
    return cpu;
}

bool DflashEngine::has_markov_head() const {
    return weights_.dflash_markov_embed && weights_.dflash_markov_bias;
}

bool DflashEngine::is_dflash2() const {
    return model_.config().dflash_selector_top_k > 0 &&
           weights_.dflash_selector_hidden != nullptr;
}

TensorPtr DflashEngine::dflash2_lattice() const {
    auto t = dflash2_lattice_;
    if (t && t->device() != DeviceType::CPU)
        t->to_device(DeviceType::CPU);  // built on CPU; kept for symmetry with markov accessors
    return t;
}

static TensorPtr ensure_cpu(const TensorPtr& t) {
    if (!t)
        return nullptr;
    if (t->device() != DeviceType::CPU)
        t->to_device(DeviceType::CPU);  // in-place
    return t;
}

TensorPtr DflashEngine::markov_embed() const {
    return ensure_cpu(weights_.dflash_markov_embed);
}
TensorPtr DflashEngine::markov_bias() const {
    return ensure_cpu(weights_.dflash_markov_bias);
}
TensorPtr DflashEngine::draft_id_map() const {
    return ensure_cpu(weights_.dflash_draft_id_to_target_id);
}

bool DflashEngine::has_confidence_head() const {
    return weights_.dflash_conf_proj_weight && weights_.dflash_conf_proj_bias;
}
TensorPtr DflashEngine::conf_proj_weight() const {
    return ensure_cpu(weights_.dflash_conf_proj_weight);  // [conf_in_dim, 1]
}
TensorPtr DflashEngine::conf_proj_bias() const {
    return ensure_cpu(weights_.dflash_conf_proj_bias);  // [1]
}
int DflashEngine::conf_in_dim() const {
    return weights_.dflash_conf_proj_weight
               ? static_cast<int>(weights_.dflash_conf_proj_weight->shape()[0])
               : 0;
}
TensorPtr DflashEngine::last_hidden_states() const {
    return last_hidden_states_;
}

// Scalar dot product of two length-n float vectors (markov bias term). Kept scalar
// and simple: the matvec is memory-bandwidth bound (the [V, d] bias tensor is ~130MB
// and read once per query position), so SIMD on the inner product is not on the
// critical path. markov_d is a multiple of 8 but the tail loop keeps it correct.
static inline float dot_markov(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

std::vector<int32_t> dspark_sequential_sample(
    const std::vector<const float*>& rows, int vocab,
    const std::vector<const float*>& hidden_rows, int hidden_dim,
    const float* markov_embed, int markov_d,
    const float* markov_bias, const int32_t* draft_id_map,
    int32_t anchor_token, int n,
    int cand_top_k,
    const float* conf_weight, const float* conf_bias, float conf_threshold) {
    std::vector<int32_t> out;
    out.reserve(n);
    int32_t prev = anchor_token;  // target vocab id (markov_embed is keyed by it)
    const bool use_markov = markov_embed && markov_bias && markov_d > 0;
    const bool use_conf = conf_weight && hidden_dim > 0 &&
                          hidden_rows.size() >= static_cast<size_t>(n);
    const int top_k = (cand_top_k > 0 && cand_top_k < vocab) ? cand_top_k : 0;

    // Scratch for the truncated candidate re-ranking (vLLM apply_bias_gathered).
    std::vector<int32_t> cand_idx;
    std::vector<float> cand_val;
    if (top_k > 0) {
        cand_idx.resize(top_k);
        cand_val.resize(top_k);
    }
    std::vector<float> confs;
    if (use_conf)
        confs.reserve(n);

    for (int j = 0; j < n; ++j) {
        const float* row = (j < static_cast<int>(rows.size())) ? rows[j] : rows.back();
        // markov bias vector for this step: mbuf[v] = dot(markov_bias[v], embed[prev]).
        // embed[prev] is reused across all v, so hoist it out of the vocab loop.
        const float* eb = use_markov ? markov_embed + static_cast<int64_t>(prev) * markov_d : nullptr;

        int best = 0;
        float best_score = 0.0f;
        if (top_k > 0) {
            // Truncated candidate re-ranking: rank base logits, add markov bias only to
            // the top-k candidates, then argmax within that set (others stay -inf).
            dflash2_top_k(row, vocab, top_k, cand_idx, cand_val);
            best = cand_idx[0];
            best_score = cand_val[0] + (eb ? dot_markov(markov_bias + static_cast<int64_t>(best) * markov_d, eb, markov_d) : 0.0f);
            for (int c = 1; c < top_k; ++c) {
                const int v = cand_idx[c];
                const float score = cand_val[c] + (eb ? dot_markov(markov_bias + static_cast<int64_t>(v) * markov_d, eb, markov_d) : 0.0f);
                if (score > best_score) {
                    best_score = score;
                    best = v;
                }
            }
        } else {
            best = 0;
            best_score = row[0];
            for (int v = 1; v < vocab; ++v) {
                const float score = row[v] + (eb ? dot_markov(markov_bias + static_cast<int64_t>(v) * markov_d, eb, markov_d) : 0.0f);
                if (score > best_score) {
                    best_score = score;
                    best = v;
                }
            }
        }

        if (draft_id_map) {
            // vLLM's map_draft_to_target is an ADDITIVE offset table, not an
            // index map:
            //     return draft_ids + self.draft_id_to_target_id[draft_ids]
            // Reading it as an index (`target = table[draft]`) silently emits a
            // different token for every reduced-vocab draft, which is invisible
            // until acceptance collapses. Offset 0 means "same id", so this is
            // also a no-op for a table full of zeros.
            //
            // No bounds check: the result is a TARGET id and the target vocab is
            // not in scope here (`vocab` is the draft logits width), so clamping
            // against it would corrupt legitimate mappings. vLLM does not clamp
            // either -- a bad offset is a malformed checkpoint.
            best += draft_id_map[best];
        }
        out.push_back(best);

        if (use_conf) {
            // Confidence head: sigmoid(conf_proj([hidden; markov_embed])). Input is the
            // concatenation of this position's draft hidden state and the markov embed
            // of the previously sampled token; conf_proj is [conf_in_dim, 1].
            const float* h = hidden_rows[j];
            float acc = conf_bias ? conf_bias[0] : 0.0f;
            for (int k = 0; k < hidden_dim; ++k)
                acc += conf_weight[k] * h[k];
            if (eb)
                for (int k = 0; k < markov_d; ++k)
                    acc += conf_weight[hidden_dim + k] * eb[k];
            confs.push_back(1.0f / (1.0f + std::exp(-acc)));
        }
        prev = best;  // next markov step keys on the (mapped) target vocab id
    }

    // Confidence-head adaptive draft truncation (vLLM adaptive verification, single-
    // request form): drop the tail once a position's confidence falls below the
    // threshold, keeping the longest high-confidence prefix.
    if (use_conf && conf_threshold > 0.0f) {
        size_t keep = out.size();
        for (size_t j = 0; j < confs.size(); ++j) {
            if (confs[j] < conf_threshold) {
                keep = j;
                break;
            }
        }
        // If the very first position is already below threshold, keep 0 proposals
        // rather than proposing the unfiltered run: the whole point of the gate is
        // to not spend the verify pass on tokens we expect to be rejected, and the
        // executor's --spec-min-acc floor is the right lever for "stop drafting
        // altogether". Truncating to a single token here instead would be worse
        // than useless -- one row of speculation cannot pay for the verify forward
        // it triggers, while a 0-length draft returns control to plain decode
        // without one.
        out.resize(keep);
    }
    return out;
}

std::vector<int32_t> dflash2_lattice_walk(const float* lattice, int64_t row_width, int top_k,
                                          int n_block_tokens, float p_min) {
    std::vector<int32_t> out;
    if (!lattice || top_k <= 0 || n_block_tokens <= 1)
        return out;
    out.reserve(n_block_tokens - 1);
    int32_t predecessor = 0;  // position 1 reads the anchor-scored row (index 0)
    for (int32_t i = 1; i < n_block_tokens; ++i) {
        const float* row = lattice + static_cast<int64_t>(i) * row_width;
        const float* scores = row + top_k + static_cast<int64_t>(predecessor) * top_k;
        int32_t best = 0;
        for (int k = 1; k < top_k; ++k)
            if (scores[k] > scores[best])
                best = k;
        if (p_min > 0.0f) {
            // softmax confidence at the argmax: 1 / sum(exp(s_k - s_max));
            // scores[best] is the max, so every exponent is <= 0.
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
                sum += std::exp(scores[k] - scores[best]);
            if (sum > 0.0f && 1.0f / sum < p_min)
                break;
        }
        predecessor = best;
        out.push_back(static_cast<int32_t>(row[predecessor]));
    }
    return out;
}

}  // namespace forge
