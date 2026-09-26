// DeepSeek-MTP style draft provider: drives the target engine's trailing
// nextn head (qwen35) as an autoregressive drafter.
//
// Discipline mirrors ModelDraftProvider (committed-prefix invariant):
//   - MTP KV slot (layer index == cfg.num_layers) holds exactly the confirmed
//     prefix after every accept()/begin().
//   - pending_h_ is the hidden state of the last confirmed token, chained
//     into the next mtp_step.
//   - accept() trims both trunk and the MTP slot to the new confirmed length
//     with a single rollback, then replays the confirmed tokens through
//     mtp_step to regenerate the hidden chain (partial acceptance makes the
//     speculative chain diverge, so unconditional replay keeps it exact).

#include "forge/mtp_draft_provider.h"

#include <algorithm>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "forge/engines/qwen35_engine.h"
#include "forge/logger.h"

namespace forge {

MtpDraftProvider::MtpDraftProvider(InferenceContext& ctx, float p_min)
    : ctx_(ctx),
      sampler_([] {
          SamplerConfig cfg;
          cfg.do_sample = false;
          cfg.temperature = 0.0f;
          cfg.repeat_penalty = 1.0f;
          return cfg;
      }()),
      p_min_(p_min) {
    auto* q35 = dynamic_cast<Qwen35Engine*>(ctx_.engine());
    if (!q35) {
        LOG_WARN("MtpDraftProvider: target engine is not qwen35");
        return;
    }
    if (!ctx_.model().config().n_nextn_layers) {
        LOG_WARN("MtpDraftProvider: model carries no nextn layers");
        return;
    }
    eng_ = q35;
}

bool MtpDraftProvider::valid() const {
    return eng_ && eng_->mtp_valid();
}

void MtpDraftProvider::begin(const std::vector<int32_t>& prompt) {
    if (!valid() || prompt.empty()) return;
    committed_len_ = static_cast<int64_t>(prompt.size());
    draft_active_ = false;
    drafted_tail_ = 0;

    // Seed hidden = post-final-norm hidden of the LAST prompt token, captured
    // from the prefill forward that just completed.
    auto full = ctx_.engine()->take_last_hidden();
    if (!full || full->shape()[0] < 1) {
        LOG_WARN("MtpDraftProvider: no hidden state available from prefill");
        return;
    }
    const int64_t vocab_rows = full->shape()[0];
    const int64_t H = full->shape()[1];
    pending_h_ = std::make_shared<Tensor>(
        full->slice(0, vocab_rows - 1, vocab_rows));
    last_committed_tok_ = prompt.back();
}

std::vector<int32_t> MtpDraftProvider::draft(int32_t /*last_token*/, int n_draft) {
    std::vector<int32_t> out;
    if (!valid() || n_draft <= 0) return out;

    sampler_.clear_history();

    // Discard any speculative tail from a previous round before extending.
    if (draft_active_) {
        if (auto* kv = ctx_.engine()->kv_cache()) kv->rollback(committed_len_);
        draft_active_ = false;
    }

    TensorPtr h = pending_h_;
    int32_t tok_in = last_committed_tok_;
    chain_h_.clear();
    chain_h_.reserve(static_cast<size_t>(n_draft));
    for (int i = 0; i < n_draft; ++i) {
        TensorPtr logits;
        auto h_next = eng_->mtp_step(tok_in, h, committed_len_ + i, /*seq_id=*/0,
                                     &logits);
        if (!h_next || !logits) break;

        chain_h_.push_back(h_next);

        // Keep logits on the device. Sampler::sample() performs a GPU argmax
        // and transfers only the resulting int32 token, avoiding a full
        // vocab-sized device-to-host copy on every MTP step. p_min is skipped
        // on CUDA because evaluating it would require another reduction; the
        // default is zero and token selection remains exact greedy argmax.
        const int32_t next = static_cast<int32_t>(sampler_.sample(logits, committed_len_ + i));
        out.push_back(next);
        tok_in = next;
        h = h_next;
    }

    if (!out.empty()) {
        drafted_tail_ = out.size();
        draft_active_ = true;
        // The chain is intentionally not committed until accept(); partial
        // acceptance must replay the confirmed suffix to rebuild it.
    }
    return out;
}

void MtpDraftProvider::accept(const std::vector<int32_t>& tokens) {
    if (!valid() || tokens.empty()) return;

    const int64_t base = committed_len_;
    const int64_t new_len = base + static_cast<int64_t>(tokens.size());

    // Realign BOTH trunk and the MTP slot to the confirmed length. The
    // executor already trimmed the trunk on partial acceptance, so for it this
    // is a no-op; on full acceptance it drops the un-confirmed draft rows.
    if (auto* kv = ctx_.engine()->kv_cache()) kv->rollback(new_len);

    // ---- 链复用 (替代整段重放) ----
    // 重放步骤 i 的输入是 (tokens[i-1], h_{i-1}, pos=base+i)。对已接受的
    // draft 前缀, tokens[i-1] == d_{i-1}, 即重放步骤 0..n_accepted 与本轮
    // 草稿链步骤 0..n_accepted 同输入、确定性同输出 —— 直接复用 chain_h_,
    // 最多补算 1 步 (全接受时的 bonus 末位)。部分接受时 0 步补算。
    // (draft_active_ == false 时无本轮链, 回退为从 pending_h_ 逐步重放,
    //  与旧行为一致; 该路径通常每轮只喂 1 个 token。)
    TensorPtr h;
    int32_t tok_in = -1;
    size_t i_start = 0;
    if (draft_active_ && !chain_h_.empty()) {
        const size_t len = tokens.size();
        // 可复用的链步骤数: min(len, chain_h_.size())
        //   - 部分接受: len = n_accepted+1 <= chain 步数 -> 全部复用, 0 补算
        //   - 全部接受: len = k+1, chain 有 k 步 -> 复用 k 步, 补 1 步 (bonus)
        const size_t reuse = std::min(len, chain_h_.size());
        if (reuse > 0) {
            h = chain_h_[reuse - 1];
            i_start = reuse;
            if (reuse >= 1)
                tok_in = tokens[reuse - 1];
        }
    }
    if (i_start == 0) {
        h = pending_h_;
        tok_in = last_committed_tok_;
    }

    for (size_t i = i_start; i < tokens.size(); ++i) {
        auto h_next = eng_->mtp_step(tok_in, h, base + static_cast<int64_t>(i),
                                     /*seq_id=*/0, nullptr);
        if (!h_next) {
            LOG_WARN("MtpDraftProvider: replay step failed at " + std::to_string(i));
            return;  // keep old pending state; next begin()/reset recovers
        }
        tok_in = tokens[i];
        h = h_next;
    }
    pending_h_ = h;
    last_committed_tok_ = tokens.back();
    committed_len_ = new_len;
    draft_active_ = false;
    drafted_tail_ = 0;
    chain_h_.clear();
}

void MtpDraftProvider::reset() {
    committed_len_ = 0;
    draft_active_ = false;
    drafted_tail_ = 0;
    pending_h_ = nullptr;
    last_committed_tok_ = -1;
    chain_h_.clear();
}

}  // namespace forge
