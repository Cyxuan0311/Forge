#include "forge/kv_working_set.h"

#include <algorithm>

#include "forge/logger.h"

namespace forge {

KVWorkingSetManager::KVWorkingSetManager(KVemConfig config, int page_size, int pool_tokens)
    : cfg_(std::move(config)), page_size_(page_size > 0 ? page_size : 16) {
    budget_ = cfg_.budget(page_size_, pool_tokens);
    // A sink larger than the budget would make every window overflow before a
    // single token is written; clamp it so the tier degrades to plain sliding
    // rather than failing the request.
    if (cfg_.sink_tokens > budget_)
        cfg_.sink_tokens = budget_;
    if (cfg_.gen_reserve < 0)
        cfg_.gen_reserve = 0;
    // A reserve that leaves no room for the prompt is self-defeating: the plan
    // would drop the entire turn and forward zero tokens. Cap it so at least
    // half the budget stays available for context.
    if (cfg_.gen_reserve > budget_ / 2)
        cfg_.gen_reserve = budget_ / 2;
    LOG_INFO("KVMem tiered KV enabled: budget=" + std::to_string(budget_) +
             " sink=" + std::to_string(cfg_.sink_tokens) +
             " gen_reserve=" + std::to_string(cfg_.gen_reserve) +
             " retrieval=" + cfg_.retrieval + " page_size=" + std::to_string(page_size_));
}

int64_t KVWorkingSetManager::archived_tokens() const {
    // Everything before the window start that has not already been dropped.
    const int64_t begin = std::max<int64_t>(0, rope_off_);
    const int64_t dropped = stats_.dropped_tokens;
    return std::max<int64_t>(0, begin - dropped);
}

int64_t KVWorkingSetManager::archive_begin() const {
    return stats_.dropped_tokens;
}

int64_t KVWorkingSetManager::archive_end() const {
    return std::max<int64_t>(0, rope_off_);
}

KVemTurnPlan KVWorkingSetManager::on_prompt(const std::vector<int32_t>& prompt, int gen_reserve) {
    KVemTurnPlan plan;

    // ---- 1. Common-prefix match against the previous logical history ----
    // The match is measured in *logical* space against the shifted prompt:
    // logical position p corresponds to prompt index p + head_shift_ (the shift
    // is 0 unless a previous single-prompt overflow clipped a prompt head).
    // It survives window eviction: what matters is whether the KV for logical
    // position p is still resident in a window slot, tracked separately below.
    int64_t match = 0;
    {
        const int64_t avail =
            static_cast<int64_t>(prompt.size()) - head_shift_;
        const int64_t n = std::min<int64_t>(
            static_cast<int64_t>(logical_hist_.size()), std::max<int64_t>(0, avail));
        while (match < n && logical_hist_[static_cast<size_t>(match)] ==
                                prompt[static_cast<size_t>(head_shift_ + match)])
            ++match;
    }

    // Tokens whose KV is still in the window are the ones we can keep. The
    // window covers logical [rope_off_, rope_off_ + last_win_len_).
    const int64_t resident_begin = last_rope_off_;
    const int64_t resident_end = last_rope_off_ + last_win_len_;
    const int64_t resident_until = std::clamp<int64_t>(match, resident_begin, resident_end);

    // ---- 2. Sink + recalled prefix stay resident ----
    // Window slot w holds logical position old_rope_off_ + w for
    // w < last_win_len_. Keeping a prefix of the window therefore keeps a
    // *contiguous* logical range starting at old_rope_off_.
    int64_t keep_window = 0;
    if (resident_until > resident_begin && last_win_len_ > 0) {
        // Never keep past the divergence point.
        keep_window = resident_until - resident_begin;
        // Cap the keep at the window position of the divergence.
        keep_window = std::min(keep_window, match - resident_begin);
        keep_window = std::max<int64_t>(keep_window, 0);
    }

    // The sink region [0, sink) is the highest-value prefix; if the window was
    // evicted past it we lose it, which is why the sink is pinned by the
    // caller's budget computation rather than here. Report the loss.
    if (cfg_.sink_tokens > 0 && resident_begin > cfg_.sink_tokens) {
        LOG_WARN("KVMem: window start (logical " + std::to_string(resident_begin) +
                 ") is past the sink (" + std::to_string(cfg_.sink_tokens) +
                 " tokens) -- the pinned prefix was evicted by budget pressure");
    }

    // kept_logical_len is the logical position of the first token the forward
    // must produce. When the prompt diverges before the window start (match <
    // resident_begin: a new conversation on a server that already archived
    // history), every resident slot is stale -- recompute from the divergence
    // point instead of claiming the stale window start, which would silently
    // drop the prompt tokens in between.
    plan.kept_logical_len = std::min(resident_begin, match) + keep_window;
    plan.kept_window_len = keep_window;
    plan.kept_rope_off = resident_begin;

    // ---- 3. Prompt side: everything after the common prefix is new ----
    // NOTE: tokens in [match, resident_begin) are *not* resident (they were
    // archived or evicted) and also no longer match the current prompt, so
    // they are recomputed from the earliest non-resident position. The prompt
    // index of logical position p is p + head_shift_.
    int64_t recompute_from = head_shift_ + plan.kept_logical_len;

    // ---- 4. Window assembly + pressure handling ----
    // The window must hold: kept prefix + prompt tail + gen_reserve.
    int64_t tail = static_cast<int64_t>(prompt.size()) - recompute_from;
    int64_t room = static_cast<int64_t>(budget_) - plan.kept_window_len;
    if (room < 0)
        room = 0;

    // Generation reserve is a hard floor: the window must not fill before the
    // reply completes. It is per-request (the caller passes the request's
    // max_new_tokens); clamp it to what the window can actually spare so a huge
    // generation budget never starves the prompt tail below one page.
    int64_t gr = (gen_reserve > 0) ? gen_reserve : cfg_.gen_reserve;
    int64_t max_reserve = room - page_size_;
    if (max_reserve < 0) max_reserve = 0;
    if (gr > max_reserve) gr = max_reserve;
    // The per-request reserve is the client's max_new_tokens, which can be huge
    // (opencode sends 32000). Capping it to the server-configured headroom keeps
    // the bulk of the window for the prompt instead of reserving the whole
    // window for generation. Decode still hard-stops at plan.max_generate, so
    // generation is bounded by the pool regardless of the client's request.
    if (gr > cfg_.gen_reserve) gr = cfg_.gen_reserve;
    if (gr < 0) gr = 0;
    int64_t usable = room - gr;
    if (usable < 0) usable = 0;

    if (tail > usable) {
        // Pressure: window + generation reserve exceeds the budget. The keep
        // size does not change feasibility (an evicted resident token just
        // moves into the recompute set -- the window always spans from the
        // earliest surviving logical token to the prompt end), so the knobs
        // are the reserve and, as a last resort, clipping the oldest history
        // through head_shift_.
        const int64_t window_need =
            static_cast<int64_t>(prompt.size()) - head_shift_;
        const int64_t gr_fit = static_cast<int64_t>(budget_) - window_need;
        if (gr_fit >= 0) {
            // Shrink the reserve instead of losing context: decode hard-stops
            // at plan.max_generate, so a smaller reserve only caps the reply
            // length. No prompt token and no resident token is lost.
            LOG_WARN("KVMem: prompt tail (" + std::to_string(tail) +
                     " tokens) exceeds window room (" + std::to_string(usable) +
                     ") -- shrinking gen_reserve " + std::to_string(gr) + " -> " +
                     std::to_string(gr_fit) + " (budget=" +
                     std::to_string(budget_) + ")");
            gr = gr_fit;
        } else {
            // Last resort: even a zero reserve cannot hold the prompt. Clip
            // the oldest history (head_shift_ += clip) and evict the whole
            // window: without a physical page-table shift the kept slot
            // prefix cannot represent a clipped head, and a stale prefix
            // would desync the next turn's match.
            const int64_t reserve =
                std::min<int64_t>(cfg_.gen_reserve, budget_ / 4);
            const int64_t clip = window_need + reserve - budget_;
            head_shift_ += clip;
            stats_.dropped_tokens += clip;
            if (plan.kept_window_len > 0) {
                plan.dropped_archived +=
                    static_cast<int>(plan.kept_window_len);
                stats_.dropped_tokens += plan.kept_window_len;
                plan.kept_window_len = 0;
                plan.kept_logical_len = resident_begin;
            }
            gr = reserve;
            plan.truncated_prompt = true;
            LOG_WARN("KVMem: prompt (" + std::to_string(prompt.size()) +
                     " tokens, head_shift " + std::to_string(head_shift_ - clip) +
                     " -> " + std::to_string(head_shift_) +
                     ") exceeds the whole window budget -- clipping " +
                     std::to_string(clip) +
                     " oldest history tokens (budget=" +
                     std::to_string(budget_) +
                     ", reserve=" + std::to_string(gr) + ")");
        }
        room = static_cast<int64_t>(budget_) - plan.kept_window_len;
        usable = room - gr;
        if (usable < 0)
            usable = 0;
        recompute_from = head_shift_ + plan.kept_logical_len;
        tail = static_cast<int64_t>(prompt.size()) - recompute_from;
    }

    // The engine samples the first token from the last prefill row, so the
    // forward needs at least one token (the untiered prefix-reuse path makes
    // the same guarantee by clamping prefix_len to prompt_len-1). A
    // byte-identical consecutive prompt -- a client retrying after a transport
    // error -- would otherwise forward an empty batch; recompute the resident
    // window's last token instead.
    if (recompute_from >= static_cast<int64_t>(prompt.size())) {
        if (plan.kept_window_len > 0)
            --plan.kept_window_len;
        --plan.kept_logical_len;
        --recompute_from;
    }

    plan.new_to_compute.assign(prompt.begin() + static_cast<size_t>(recompute_from), prompt.end());
    plan.new_window_len = plan.kept_window_len + static_cast<int64_t>(plan.new_to_compute.size());
    plan.new_rope_off = plan.kept_rope_off + plan.kept_window_len;
    // Remaining window slots for this turn's decode; the generator caps the reply
    // here. Clamped >= 0 so a full window yields "generate nothing" rather than
    // writing past the pool.
    plan.max_generate = std::max<int64_t>(0, budget_ - plan.new_window_len);

    if (plan.new_window_len > budget_) {
        // Should not happen given the clamp above; guard anyway so the caller
        // never writes outside the pool.
        plan.window_overflow = true;
        stats_.window_overflow++;
        LOG_ERROR("KVMem: window overflow after planning: window=" +
                  std::to_string(plan.new_window_len) + " budget=" + std::to_string(budget_));
    }

    // ---- 5. Commit state ----
    // logical_hist_ mirrors the window content exactly: slot w holds the token
    // logical_hist_[w], so next turn's common-prefix match measures precisely
    // which resident slots the new prompt can reuse. Kept slots stay, the newly
    // computed tail extends it. Tokens generated during decode land past
    // new_window_len and are deliberately NOT folded in: they are not reusable
    // (the next prompt re-encodes them inside its tail) and folding them would
    // claim slots the plan no longer tracks.
    logical_hist_.resize(static_cast<size_t>(plan.kept_window_len));
    logical_hist_.insert(logical_hist_.end(), plan.new_to_compute.begin(),
                         plan.new_to_compute.end());
    last_win_len_ = plan.new_window_len;
    last_rope_off_ = plan.kept_rope_off;
    window_len_ = plan.new_window_len;
    rope_off_ = plan.kept_rope_off;
    stats_.turns++;

    LOG_INFO("KVMem turn " + std::to_string(stats_.turns) + ": prompt=" +
             std::to_string(prompt.size()) + " match=" + std::to_string(match) +
             " keep_window=" + std::to_string(plan.kept_window_len) +
             " compute=" + std::to_string(plan.new_to_compute.size()) +
             " window=" + std::to_string(plan.new_window_len) + "/" +
             std::to_string(budget_) + " rope_off=" + std::to_string(plan.new_rope_off));

    return plan;
}

}  // namespace forge
