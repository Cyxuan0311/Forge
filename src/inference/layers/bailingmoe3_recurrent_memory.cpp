#include "forge/inference/layers/bailingmoe3_recurrent_memory.h"

#include "forge/logger.h"

namespace forge {

Bailingmoe3RecurrentMemory::~Bailingmoe3RecurrentMemory() = default;

void Bailingmoe3RecurrentMemory::alloc_state(CpuState& s) {
    s.conv_q.assign(static_cast<size_t>(dims_.conv_state_size_q()), 0.0f);
    s.conv_k.assign(static_cast<size_t>(dims_.conv_state_size_k()), 0.0f);
    s.conv_v.assign(static_cast<size_t>(dims_.conv_state_size_v()), 0.0f);
    s.ssm_state.assign(static_cast<size_t>(dims_.state_size()), 0.0f);
}

void Bailingmoe3RecurrentMemory::init(const ModelConfig& cfg, const ModelWeights& /*weights*/) {
    num_layers_ = cfg.num_layers;

    // 维度严格对齐 llama.cpp 的 bailingmoe3:
    //   head_dim = kda.head_dim, 每个 attention head 一个 KDA 头,
    //   q/k/v/ssm_f_a/ssm_g_a 都投影到 d_inner = num_heads * head_dim。
    // 单个 KDA 头的 k 维与 v 维同为 head_dim (状态方阵 head_dim²)。
    const int num_heads = cfg.num_heads > 0 ? cfg.num_heads : 1;
    const int head_dim = cfg.kda_head_dim > 0 ? cfg.kda_head_dim : 128;
    const int d_conv = cfg.ssm_conv_kernel > 0 ? cfg.ssm_conv_kernel : 4;

    dims_.kd = head_dim;
    dims_.vd = head_dim;
    dims_.d_conv = d_conv;
    dims_.num_k_heads = num_heads;
    dims_.num_v_heads = num_heads;

    seqs_.clear();
    initialized_ = true;
    LOG_DEBUG("Bailingmoe3RecurrentMemory: heads=" + std::to_string(dims_.num_v_heads) +
              " head_dim=" + std::to_string(dims_.kd) +
              " d_inner=" + std::to_string(dims_.d_inner()) +
              " d_conv=" + std::to_string(dims_.d_conv) +
              " conv_state=" + std::to_string(dims_.conv_state_size_q()) +
              " state=" + std::to_string(dims_.state_size()) +
              " layers=" + std::to_string(num_layers_));
}

void Bailingmoe3RecurrentMemory::reset() {
    for (auto& [seq_id, st] : seqs_) {
        (void)seq_id;
        for (auto& s : st.cur) alloc_state(s);
        for (auto& s : st.snap) alloc_state(s);
        st.snap_valid = false;
    }
}

Bailingmoe3RecurrentMemory::PerSeq& Bailingmoe3RecurrentMemory::ensure_seq(int seq_id) {
    auto it = seqs_.find(seq_id);
    if (it != seqs_.end())
        return it->second;
    PerSeq st;
    st.cur.resize(num_layers_);
    st.snap.resize(num_layers_);
    for (auto& s : st.cur) alloc_state(s);
    for (auto& s : st.snap) alloc_state(s);
    auto res = seqs_.emplace(seq_id, std::move(st));
    return res.first->second;
}

#define BM3_STATE_ACCESSOR(field, getter)                                             \
    float* Bailingmoe3RecurrentMemory::getter(int seq_id, int layer_idx) {             \
        if (layer_idx < 0 || layer_idx >= num_layers_) return nullptr;                 \
        auto& st = ensure_seq(seq_id);                                                 \
        return st.cur[layer_idx].field.data();                                         \
    }

BM3_STATE_ACCESSOR(conv_q, conv_state_q_cpu)
BM3_STATE_ACCESSOR(conv_k, conv_state_k_cpu)
BM3_STATE_ACCESSOR(conv_v, conv_state_v_cpu)
BM3_STATE_ACCESSOR(ssm_state, ssm_state_cpu)

#undef BM3_STATE_ACCESSOR

void Bailingmoe3RecurrentMemory::snapshot(int seq_id) {
    auto& st = ensure_seq(seq_id);
    for (int i = 0; i < num_layers_; ++i) {
        st.snap[i].conv_q = st.cur[i].conv_q;
        st.snap[i].conv_k = st.cur[i].conv_k;
        st.snap[i].conv_v = st.cur[i].conv_v;
        st.snap[i].ssm_state = st.cur[i].ssm_state;
    }
    st.snap_valid = true;
}

void Bailingmoe3RecurrentMemory::rollback(int seq_id) {
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end())
        return;
    auto& st = it->second;
    if (!st.snap_valid)
        return;  // 从未 snapshot 的序列: 必须是 no-op
    for (int i = 0; i < num_layers_; ++i) {
        st.cur[i].conv_q = st.snap[i].conv_q;
        st.cur[i].conv_k = st.snap[i].conv_k;
        st.cur[i].conv_v = st.snap[i].conv_v;
        st.cur[i].ssm_state = st.snap[i].ssm_state;
    }
}

void Bailingmoe3RecurrentMemory::reset_seq(int seq_id) {
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end())
        return;
    for (auto& s : it->second.cur) alloc_state(s);
    it->second.snap_valid = false;
}

}  // namespace forge
