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

int64_t KVWorkingSetManager::common_prefix_len(const std::vector<int32_t>& a,
                                               const std::vector<int32_t>& b) const {
    const int64_t n = std::min<int64_t>(static_cast<int64_t>(a.size()),
                                        static_cast<int64_t>(b.size()));
    int64_t i = 0;
    while (i < n && a[static_cast<size_t>(i)] == b[static_cast<size_t>(i)])
        ++i;
    return i;
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
    // The match is measured in *logical* space, so it survives window
    // eviction: what matters is whether the KV for logical position p is still
    // resident in a window slot, which we track separately below.
    const int64_t match = common_prefix_len(logical_hist_, prompt);

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

    plan.kept_logical_len = resident_begin + keep_window;
    plan.kept_window_len = keep_window;
    plan.kept_rope_off = resident_begin;

    // ---- 3. Prompt side: everything after the common prefix is new ----
    // NOTE: tokens in [match, resident_begin) are *not* resident (they were
    // archived or evicted) and also no longer match the current prompt, so
    // they are recomputed from the earliest non-resident position.
    int64_t recompute_from = plan.kept_logical_len;

    // ---- 4. Window assembly + staleness drop ----
    // The window must hold: recall(optional) + prompt tail + gen_reserve.
    const int64_t tail = static_cast<int64_t>(prompt.size()) - recompute_from;
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
        plan.dropped_archived = static_cast<int>(tail - usable);
        recompute_from = static_cast<int64_t>(prompt.size()) - usable;
        plan.truncated_prompt = true;
        stats_.dropped_tokens += plan.dropped_archived;
        LOG_WARN("KVMem: prompt tail (" + std::to_string(tail) +
                 " tokens) exceeds window room (" + std::to_string(usable) +
                 ") -- dropping " + std::to_string(plan.dropped_archived) +
                 " tokens of stale history (budget=" + std::to_string(budget_) +
                 ", sink=" + std::to_string(cfg_.sink_tokens) +
                 ", gen_reserve=" + std::to_string(gr) + ")");
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

    if (plan.dropped_archived > 0)
        stats_.dropped_tokens += plan.dropped_archived;

    // ---- 5. Commit state ----
    // rope_off_ / last_rope_off_ are the logical position of *window slot 0*,
    // which is kept_rope_off — NOT new_rope_off (the logical position of the
    // first newly-written token). Next turn's on_prompt() reads last_rope_off_
    // as "slot 0 maps to this logical position"; committing new_rope_off would
    // shift every resident token's rope position by kept_window_len and desync
    // multi-turn reuse after the first turn.
    logical_hist_ = prompt;
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
