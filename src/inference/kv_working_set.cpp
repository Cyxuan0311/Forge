#include "forge/kv_working_set.h"

#include <algorithm>
#include <unordered_map>

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
    // An empty window is a valid layout (nothing resident yet): a zero-length Tail
    // segment at slot 0, which tiles [0, 0) vacuously.
    segs_.push_back({0, 0, 0, WindowSeg::Kind::Tail});
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
    const int64_t prompt_len = static_cast<int64_t>(prompt.size());

    // ---- 0. Current window layout ----
    // segs_ tiles [0, window_slots()) in slot order. Reading it as
    // [ Sink ][ Recalled... ][ Tail ]: the sink is the conversation's first
    // tokens (attention needs a stable anchor), recalled pages are archived
    // history pulled back in, and the tail is the most recent context.
    const int64_t prev_slots = window_slots();
    int64_t sink_slots = 0;
    for (const auto& s : segs_) {
        if (s.kind == WindowSeg::Kind::Sink) {
            sink_slots = s.slot_len;
            break;
        }
    }
    int64_t tail_base = head_shift_;
    for (const auto& s : segs_) {
        if (s.kind == WindowSeg::Kind::Tail) {
            tail_base = s.logical_start;
            break;
        }
    }
    auto logical_at = [this](int64_t slot) { return logical_of_slot(slot); };

    // ---- 1. Common-prefix match ----
    // Slot by slot, against the prompt index that slot logically represents. A
    // slot whose logical position is past the end of this prompt cannot match (the
    // prompt shrank) and ends the run.
    int64_t match = 0;
    {
        const int64_t n =
            std::min<int64_t>(prev_slots, static_cast<int64_t>(logical_hist_.size()));
        while (match < n) {
            const int64_t lp = logical_at(match);
            if (lp < 0 || lp >= prompt_len)
                break;
            if (logical_hist_[static_cast<size_t>(match)] != prompt[static_cast<size_t>(lp)])
                break;
            ++match;
        }
    }

    // Reuse is always a contiguous slot prefix, because the new tokens are
    // written right after it. The sink portion can only be reused if the
    // divergence is at or after the sink's end.
    int64_t sink_reuse = std::min<int64_t>(match, sink_slots);
    int64_t tail_keep = match - sink_reuse;
    // How much of the sink must be (re)forwarded. Only when the divergence falls
    // inside the sink region; in that case tail_keep is necessarily 0, so the two
    // never contend for the same slots.
    int64_t sink_forward = std::max<int64_t>(0, sink_slots - sink_reuse);
    // Where the tail forward starts reading from the prompt. `match` counts *reused*
    // slots, so the first slot to compute is `match` itself -- which is inside the
    // window only when the match stopped short of the end. When it ran to the end
    // (the common multi-turn case: the whole prompt so far matches), the next
    // logical position is one past the last resident one.
    auto next_logical = [&](int64_t m) -> int64_t {
        if (m < prev_slots)
            return logical_at(m);
        const int64_t lp = logical_at(prev_slots - 1);
        return lp < 0 ? tail_base : lp + 1;
    };
    // Divergence inside the pinned sink: the sink re-forward covers prompt
    // [sink_reuse, sink_slots), so the tail's prompt-side start is the sink's
    // end. tail_base would point at the OLD tail's logical start and silently
    // skip every prompt token in [sink_slots, tail_base) -- the model would
    // answer from a window that never saw most of the new turn.
    int64_t tail_src = (tail_keep > 0) ? next_logical(match)
                     : (sink_forward > 0 ? sink_reuse + sink_forward : tail_base);
    if (tail_src < 0)
        tail_src = 0;
    if (tail_src > prompt_len)
        tail_src = prompt_len;

    plan.sink_reuse = sink_reuse;
    plan.sink_forward = sink_forward;
    plan.kept_window_len = match;
    plan.kept_logical_len = tail_src;
    plan.kept_rope_off = 0;

    // ---- 2. Prompt side ----
    int64_t tail = prompt_len - tail_src;

    // ---- 3. Budget ----
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
    // window for generation. Decode hard-stops at plan.max_generate, so
    // generation is bounded by the pool regardless of the client's request.
    if (gr > cfg_.gen_reserve) gr = cfg_.gen_reserve;
    if (gr < 0) gr = 0;
    int64_t usable = room - gr;
    if (usable < 0) usable = 0;

    const int64_t sink_slots_before_clip = sink_slots;
    bool clipped = false;
    if (tail > usable) {
        // This is where the window loses its head, which is exactly where the
        // attention sink has to be materialised: rebuild as [sink | tail] and
        // forward prompt[0, sink) into slots [0, sink). StreamingLLM's result is
        // that the model keeps a stable anchor even when nearly all history is
        // gone, instead of degrading into confident nonsense. The sink is free in
        // slot terms -- those slots held tokens this branch drops anyway.
        //
        // Dropped history goes to the host archive rather than being discarded,
        // so a later turn can pull the relevant pages back.
        const int64_t sink_want =
            std::clamp<int64_t>(cfg_.sink_tokens, 0, std::max<int64_t>(0, budget_ - 1));
        const int64_t window_need = sink_want + (prompt_len - tail_base);
        const int64_t gr_fit = static_cast<int64_t>(budget_) - window_need;
        if (gr_fit >= 0 && sink_slots == 0) {
            // Shrink the reserve instead of losing context: decode hard-stops at
            // plan.max_generate, so a smaller reserve only caps the reply length.
            gr = gr_fit;
            LOG_WARN("KVMem: prompt tail (" + std::to_string(tail) +
                     " tokens) exceeds window room (" + std::to_string(usable) +
                     ") -- shrinking gen_reserve " + std::to_string(gr) + " -> " +
                     std::to_string(gr_fit) + " (budget=" + std::to_string(budget_) + ")");
        } else {
            // Rebuild as [sink | tail]: clip the oldest history so the tail fits,
            // pin a sink, forward both. The kept prefix goes to the archive -- a
            // contiguous prefix cannot simultaneously represent a clipped head
            // and a pinned sink, and a stale prefix would desync the next match.
            const int64_t reserve = std::min<int64_t>(cfg_.gen_reserve, budget_ / 4);
            const int64_t tail_room =
                std::max<int64_t>(1, static_cast<int64_t>(budget_) - reserve - sink_want);
            const int64_t new_tail_base = std::max<int64_t>(sink_want, prompt_len - tail_room);
            const int64_t clip = new_tail_base - tail_base;

            // Archive the window pages that this clip is about to discard. This has to
            // happen BEFORE the new layout is committed, while the old segments
            // still describe what the pages contain. Skipping pages already in the
            // archive: only pages that are resident right now can be offloaded.
            //
            // Note this is a no-op on the very first clipped turn, when the window
            // is still empty -- there is nothing resident to archive yet. History
            // that was never resident in the first place has no KV to keep, so it
            // is genuinely unrecoverable rather than merely archived.
            if (storage_ && prev_slots > 0)
                archive_pages(prev_slots);

            if (plan.kept_window_len > 0)
                stats_.dropped_tokens += plan.kept_window_len;
            stats_.dropped_tokens += std::max<int64_t>(0, clip);
            plan.kept_window_len = 0;
            plan.kept_logical_len = 0;
            plan.sink_reuse = 0;
            // Leave at least one slot for the tail, or there is no "recent" left.
            plan.sink_forward = std::min<int64_t>(sink_want, tail_room);
            sink_slots = plan.sink_forward;
            tail_base = new_tail_base;
            tail_src = new_tail_base;
            tail = prompt_len - tail_src;
            gr = reserve;
            clipped = true;
            plan.truncated_prompt = true;
            LOG_WARN("KVMem: prompt (" + std::to_string(prompt.size()) +
                     " tokens) exceeds the window budget -- clipping " +
                     std::to_string(clip) + " oldest history tokens (budget=" +
                     std::to_string(budget_) + ", reserve=" + std::to_string(reserve) +
                     "), pinning a " + std::to_string(plan.sink_forward) +
                     "-token attention sink in slots [0, " +
                     std::to_string(plan.sink_forward) + ")");
        }
        room = static_cast<int64_t>(budget_) - plan.kept_window_len;
        usable = room - gr;
        if (usable < 0)
            usable = 0;
    }

    // ---- 4. Recall pass ----
    // Off by default: recalling competes with the recent tail for the same slots,
    // and the relevance heuristic is unvalidated. Enable with
    // FORGE_KVMEM_RECALL=<max pages> [FORGE_KVMEM_RECALL_MIN_SCORE=<float>].
    // Only worth attempting on a turn that dropped history, since otherwise
    // nothing left the window.
    if (clipped && storage_ && recall_max_pages_ > 0)
        recall_pages(prompt, plan.sink_forward);

    // The engine samples the first token from the last prefill row, so the tail
    // forward needs at least one token. A byte-identical consecutive prompt (a
    // client retrying after a transport error) would otherwise forward an empty
    // batch; pull the boundary back a slot so one token is recomputed.
    if (tail <= 0) {
        if (plan.kept_window_len > 0)
            --plan.kept_window_len;
        if (tail_src > 0)
            --tail_src;
        tail = prompt_len - tail_src;
    }
    if (tail < 0)
        tail = 0;

    plan.new_to_compute.assign(prompt.begin() + static_cast<size_t>(tail_src), prompt.end());

    // ---- 5. Commit the layout ----
    // Sink length as it stood *before* this turn's clip, so the commit can tell a
    // reused sink from a freshly forwarded one.
    const int64_t prev_sink_slots = sink_slots_before_clip;

    // Rebuild the segment table so it tiles the new window with no holes, and
    // keep logical_hist_ in the same slot order so next turn's match is exact.
    std::vector<WindowSeg> next;
    int64_t cursor = 0;
    // --- Sink: either reused (keep its slots) or freshly forwarded ---
    const int64_t sink_reused = std::min<int64_t>(plan.sink_reuse, prev_sink_slots);
    const int64_t sink_len = sink_reused + plan.sink_forward;
    if (sink_len > 0) {
        // A reused sink keeps its own logical range (always [0, ...)); a forwarded
        // one restarts at logical 0. Both are [0, sink_len) in logical space, so
        // one segment covers them.
        next.push_back({cursor, sink_len, 0, WindowSeg::Kind::Sink});
        cursor += sink_len;
    }
    // --- Recalled pages (still resident, KV not rewritten) ---
    for (const auto& s : segs_) {
        if (s.kind != WindowSeg::Kind::Recalled)
            continue;
        WindowSeg c = s;
        c.slot_start = cursor;
        next.push_back(c);
        cursor += c.slot_len;
    }
    // --- Reused tail prefix: keep_window_len - sink_reused slots ---
    // These are still resident from the previous turn; their KV is valid, so they
    // stay as the head of the new tail. This is what makes multi-turn prefix reuse
    // free: the planner forwards only the divergent tail.
    const int64_t tail_reused = std::max<int64_t>(0, plan.kept_window_len - sink_reused);
    if (tail_reused > 0) {
        // The reused tail prefix starts right after the reused sink, so its logical
        // base is the logical position of slot sink_reused -- or, when the reused
        // prefix runs to the end of the old window, one past its last slot.
        int64_t logical_start = sink_reused < prev_slots
                                    ? logical_at(sink_reused)
                                    : next_logical(prev_slots);
        if (logical_start < 0)
            logical_start = tail_base;
        next.push_back({cursor, tail_reused, logical_start, WindowSeg::Kind::Tail});
        cursor += tail_reused;
    }
    // --- Freshly forwarded tail ---
    plan.tail_slot_start = cursor;
    const int64_t tail_len = static_cast<int64_t>(plan.new_to_compute.size());
    if (tail_len > 0) {
        next.push_back({cursor, tail_len, tail_src, WindowSeg::Kind::Tail});
        cursor += tail_len;
    }
    segs_ = std::move(next);

    plan.sink_slots = plan.sink_forward;
    // The fresh tail's logical base is tail_src, not the resident tail's base:
    // on an unclipped multi-turn turn the resident tail still starts at logical
    // 0 while the divergence -- and the fresh tail -- sit at the match point.
    // The two only coincide on a clipped turn. plan.tail_base therefore carries
    // the *forwarded* tail's base, which is what the rope offset derives from.
    plan.tail_base = tail_src;
    // A forward writing tail slots needs this shift so RoPE lands on the logical
    // position: slot + tail_rope_offset == logical_of_slot(slot).
    int64_t sink_slots_final = 0;
    for (const auto& s : segs_)
        if (s.kind == WindowSeg::Kind::Sink)
            sink_slots_final = s.slot_len;
    plan.tail_rope_offset = tail_src - plan.tail_slot_start;
    plan.new_window_len = cursor;
    plan.new_rope_off = tail_src;
    plan.max_generate = std::max<int64_t>(0, budget_ - plan.new_window_len);

    if (plan.new_window_len > budget_) {
        // Should not happen given the clamp above; guard anyway so the caller
        // never writes outside the pool.
        plan.window_overflow = true;
        stats_.window_overflow++;
        LOG_ERROR("KVMem: window overflow after planning: window=" +
                  std::to_string(plan.new_window_len) + " budget=" + std::to_string(budget_) +
                  " sink=" + std::to_string(plan.sink_slots));
    }

    // ---- 6. Commit token history ----
    // logical_hist_ mirrors the window slot by slot, so next turn's match measures
    // exactly which resident slots this prompt can reuse. Tokens generated during
    // decode land past new_window_len and are deliberately NOT folded in: they
    // are not reusable (the next prompt re-encodes them inside its tail).
    logical_hist_.clear();
    for (const auto& s : segs_) {
        if (s.kind == WindowSeg::Kind::Recalled) {
            // Recalled tokens live in archived_tokens_; find this chunk's copy.
            for (const auto& c : archived_chunks_) {
                if (c.logical_start == s.logical_start && c.slot_len == s.slot_len) {
                    logical_hist_.insert(logical_hist_.end(), c.tokens.begin(), c.tokens.end());
                    break;
                }
            }
            continue;
        }
        const int64_t from = s.logical_start;
        const int64_t to = std::min<int64_t>(prompt_len, from + s.slot_len);
        if (from < 0 || from >= prompt_len)
            continue;
        logical_hist_.insert(logical_hist_.end(),
                             prompt.begin() + static_cast<size_t>(from),
                             prompt.begin() + static_cast<size_t>(to));
    }
    head_shift_ = tail_base;
    last_win_len_ = plan.new_window_len;
    last_rope_off_ = 0;
    window_len_ = plan.new_window_len;
    rope_off_ = 0;
    stats_.turns++;

    LOG_INFO("KVMem turn " + std::to_string(stats_.turns) + ": prompt=" +
             std::to_string(prompt.size()) + " match=" + std::to_string(match) +
             " sink=" + std::to_string(plan.sink_slots) + "(reuse " +
             std::to_string(plan.sink_reuse) + ", fwd " + std::to_string(plan.sink_forward) +
             ") recalled=" + std::to_string(stats_.recalled_pages) + " tail_base=" +
             std::to_string(plan.tail_base) + " tail_slot=" + std::to_string(plan.tail_slot_start) +
             " keep_window=" + std::to_string(plan.kept_window_len) + " compute=" +
             std::to_string(plan.new_to_compute.size()) + " window=" +
             std::to_string(plan.new_window_len) + "/" + std::to_string(budget_) +
             " rope_off=" + std::to_string(plan.new_rope_off));

    // Explicit, actionable visibility for the steady-state loss. History before
    // the tail's logical base is not in the window; warn when that first happens
    // and again each time it grows materially, rather than leaving the user to
    // notice a tok/s cliff and confidently-wrong answers.
    if (head_shift_ > 0) {
        const int64_t step = std::max<int64_t>(1, budget_ / 10);
        if (last_warned_head_shift_ == 0 || head_shift_ - last_warned_head_shift_ >= step) {
            LOG_WARN("KVMem: context truncated -- " + std::to_string(head_shift_) +
                     " of " + std::to_string(prompt.size()) +
                     " prompt tokens are outside the GPU window (budget=" +
                     std::to_string(budget_) +
                     "). The window keeps the most recent tokens plus a " +
                     std::to_string(plan.sink_slots) +
                     "-token attention sink" +
                     (storage_ && recall_max_pages_ > 0
                          ? "; older pages are archived on the host and pulled back by "
                            "lexical relevance (FORGE_KVMEM_RECALL)."
                          : "; no host archive backs the dropped history. Raise "
                            "--kvmem-budget if earlier context matters.") );
            last_warned_head_shift_ = head_shift_;
        }
    }

    return plan;
}


// ---------------------------------------------------------------------------
// Host archive + keyword recall
// ---------------------------------------------------------------------------
// Recalled pages land in window slots [sink_pages, sink_pages+n). Their KV
// carries logical RoPE positions baked in at write time, so the slot order is
// invisible to attention (order-independent over keys); only the contiguous
// [0, logical_len) prefix that seq_filled() reports has to stay intact.
int64_t KVWorkingSetManager::logical_of_slot(int64_t slot) const {
    for (const auto& s : segs_) {
        if (slot >= s.slot_start && slot < s.slot_start + s.slot_len)
            return s.logical_start + (slot - s.slot_start);
    }
    return -1;
}

int64_t KVWorkingSetManager::window_slots() const {
    int64_t n = 0;
    for (const auto& s : segs_)
        n += s.slot_len;
    return n;
}

void KVWorkingSetManager::compact_segments() {
    int64_t cursor = 0;
    std::vector<WindowSeg> out;
    out.reserve(segs_.size());
    for (auto s : segs_) {
        if (s.slot_len <= 0)
            continue;
        s.slot_start = cursor;
        out.push_back(s);
        cursor += s.slot_len;
    }
    segs_.swap(out);
}

// ---------------------------------------------------------------------------
// Host archive + keyword recall
// ---------------------------------------------------------------------------
// Recalled pages land in window slots between the sink and the tail. Their KV
// carries logical RoPE positions baked in at write time, so the slot order is
// invisible to attention (order-independent over keys); only the hole-free
// [0, logical_len) prefix that seq_filled() reports has to stay intact.
int64_t KVWorkingSetManager::recall_pages(const std::vector<int32_t>& prompt,
                                          int64_t sink_slots) {
    if (!storage_ || archived_chunks_.empty() || recall_max_pages_ <= 0)
        return 0;

    const int64_t sink_pages = (sink_slots + page_size_ - 1) / page_size_;
    // Spare room only: recall competes with the recent tail for the same slots,
    // and the recent tail is what keeps the answer coherent.
    const int64_t win_pages = window_slots() / page_size_;
    const int64_t room_pages = budget_ / page_size_ - sink_pages - win_pages;
    if (room_pages <= 0)
        return 0;

    // Score each archived page by lexical overlap with this prompt, weighted by
    // inverse page frequency so filler cannot dominate: a token present in every
    // archived page carries no retrieval signal, one unique to a single page
    // carries all of it. Frequency is counted over the archive only -- the prompt
    // is the query, not a document.
    std::unordered_map<int32_t, int> df;
    std::vector<std::vector<int32_t>> distinct(archived_chunks_.size());
    for (size_t i = 0; i < archived_chunks_.size(); ++i) {
        auto& toks = archived_chunks_[i].tokens;
        std::sort(toks.begin(), toks.end());
        toks.erase(std::unique(toks.begin(), toks.end()), toks.end());
        distinct[i] = toks;
        for (int32_t t : distinct[i])
            ++df[t];
    }

    std::vector<int32_t> query(prompt);
    std::sort(query.begin(), query.end());
    query.erase(std::unique(query.begin(), query.end()), query.end());

    struct Cand {
        size_t idx;
        double score;
    };
    std::vector<Cand> cands;
    cands.reserve(distinct.size());
    for (size_t i = 0; i < distinct.size(); ++i) {
        double score = 0.0;
        for (int32_t t : distinct[i]) {
            if (std::binary_search(query.begin(), query.end(), t))
                score += 1.0 / (1.0 + static_cast<double>(df[t]));
        }
        if (score >= recall_min_score_)
            cands.push_back({i, score});
    }
    if (cands.empty())
        return 0;

    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        return a.score != b.score ? a.score > b.score : a.idx > b.idx;
    });
    const int64_t take = std::min<int64_t>(
        {static_cast<int64_t>(cands.size()), recall_max_pages_, room_pages});

    // Restore most-relevant-first, recording each as its own segment. Storage's
    // archive index shifts as chunks are erased, so map each chosen chunk to its
    // position among the *remaining* ones rather than reusing the score order.
    int64_t recalled = 0;
    std::vector<WindowSeg> added;
    std::vector<bool> taken(archived_chunks_.size(), false);
    for (int64_t k = 0; k < take; ++k) {
        // Next best candidate that is still untaken.
        size_t pick = static_cast<size_t>(-1);
        for (const auto& c : cands) {
            if (!taken[c.idx]) {
                pick = c.idx;
                break;
            }
        }
        if (pick == static_cast<size_t>(-1))
            break;

        const int64_t insert_at = static_cast<int64_t>(added.size());
        // Storage's archive index counts untaken chunks before this one.
        int64_t remaining = 0;
        for (size_t j = 0; j < archived_chunks_.size(); ++j) {
            if (taken[j])
                continue;
            if (j == pick)
                break;
            ++remaining;
        }
        bool ok = true;
        for (int l = 0; l < num_layers_ && ok; ++l) {
            const int64_t got = storage_->seq_window_restore_page(
                seq_id_, l, static_cast<int>(remaining),
                static_cast<int>(sink_pages) + static_cast<int>(insert_at));
            if (got <= 0)
                ok = false;
            else
                recalled += got;
        }
        if (!ok) {
            // Undo the partial restore so the page table stays consistent.
            for (int l = 0; l < num_layers_; ++l)
                storage_->seq_window_archive_page(seq_id_, l,
                                                  static_cast<int>(sink_pages) +
                                                      static_cast<int>(insert_at));
            break;
        }
        taken[pick] = true;
        added.push_back({0, archived_chunks_[pick].slot_len,
                         archived_chunks_[pick].logical_start,
                         WindowSeg::Kind::Recalled});
        ++stats_.recalled_pages;
    }

    if (added.empty())
        return 0;
    // Drop the recalled chunks from the archive index.
    std::vector<ArchivedChunk> keep;
    keep.reserve(archived_chunks_.size());
    for (size_t i = 0; i < archived_chunks_.size(); ++i)
        if (!taken[i])
            keep.push_back(archived_chunks_[i]);
    archived_chunks_.swap(keep);
    // Insert the new segments after the sink; the commit step re-lays out slots.
    std::vector<WindowSeg> next;
    for (const auto& s : segs_)
        if (s.kind == WindowSeg::Kind::Sink)
            next.push_back(s);
    for (const auto& s : added)
        next.push_back(s);
    for (const auto& s : segs_)
        if (s.kind != WindowSeg::Kind::Sink)
            next.push_back(s);
    segs_.swap(next);
    compact_segments();

    stats_.recalled_tokens += recalled;
    LOG_INFO("KVMem: recalled " + std::to_string(added.size()) + " archived page(s), " +
             std::to_string(recalled) + " tokens (sink_pages=" + std::to_string(sink_pages) +
             ", room_pages=" + std::to_string(room_pages) + ")");
    return recalled;
}

void KVWorkingSetManager::attach_storage(KVStorage* storage, int num_layers, int seq_id) {
    storage_ = storage;
    num_layers_ = num_layers;
    seq_id_ = seq_id;
    // Experimental knobs live in env, not in the CLI flag set: recall is unproven
    // (it competes with the recent tail for slots), so it must not change default
    // behaviour, and promoting it to a flag later is a one-line change here.
    if (const char* e = std::getenv("FORGE_KVMEM_RECALL")) {
        const int n = std::atoi(e);
        if (n > 0)
            recall_max_pages_ = n;
    }
    if (const char* e = std::getenv("FORGE_KVMEM_RECALL_MIN_SCORE")) {
        const double v = std::atof(e);
        if (v > 0.0)
            recall_min_score_ = v;
    }
    if (storage_) {
        storage_->attach_working_set(seq_id, true);
        LOG_INFO("KVMem: host archive armed for seq " + std::to_string(seq_id) + " across " +
                 std::to_string(num_layers_) + " layer(s); clipped pages offload to the host "
                 "pool. Recall " +
                 (recall_max_pages_ > 0
                      ? "ON (max " + std::to_string(recall_max_pages_) + " pages, min_score " +
                            std::to_string(recall_min_score_) + ")"
                      : std::string("OFF -- enable with FORGE_KVMEM_RECALL=<max pages>")));
    }
}

// Archive the window pages at slots [0, from_slot), skipping the pinned sink.
// Those pages are about to be overwritten; their KV stays in the host pool and
// their token ids in archived_chunks_ so a later turn can recall them. Whole pages
// only, so the sink/tail boundary never straddles a page.
int64_t KVWorkingSetManager::archive_pages(int64_t from_slot) {
    if (!storage_ || from_slot <= 0)
        return 0;
    const int64_t first_page = (from_slot + page_size_ - 1) / page_size_;
    if (first_page <= 0)
        return 0;

    int64_t moved = 0;
    // Erase shifts the tail left, so archiving repeatedly at index `first_page - 1`
    // walks the page range downwards from its top.
    for (int64_t k = 0; k < first_page; ++k) {
        const int64_t idx = first_page - 1 - k;
        if (idx < 0 || idx >= static_cast<int64_t>(segs_.size()) + first_page)
            break;
        // Token ids for this page, straight out of the slot-ordered history.
        ArchivedChunk chunk;
        chunk.logical_start = logical_of_slot(idx * page_size_);
        chunk.slot_len = page_size_;
        const size_t lo = static_cast<size_t>(idx * page_size_);
        if (chunk.logical_start >= 0 && lo < logical_hist_.size()) {
            const size_t hi =
                std::min(logical_hist_.size(), lo + static_cast<size_t>(page_size_));
            chunk.tokens.assign(logical_hist_.begin() + static_cast<long>(lo),
                                logical_hist_.begin() + static_cast<long>(hi));
            chunk.slot_len = static_cast<int64_t>(hi - lo);
        } else {
            continue;  // nothing resident here; not worth archiving
        }

        bool ok = true;
        for (int l = 0; l < num_layers_ && ok; ++l) {
            const int64_t got =
                storage_->seq_window_archive_page(seq_id_, l, static_cast<int>(idx));
            if (got <= 0)
                ok = false;
            else
                moved += got;
        }
        if (!ok) {
            // Partial archive: put the pages back where they were.
            for (int l = 0; l < num_layers_; ++l)
                storage_->seq_window_restore_page(seq_id_, l, static_cast<int>(idx),
                                                  static_cast<int>(idx));
            break;
        }
        archived_chunks_.push_back(std::move(chunk));
        ++stats_.archived_pages;
    }
    return moved;
}
int64_t KVWorkingSetManager::append_generated(const std::vector<int32_t>& tokens,
                                              int64_t start_slot) {
    if (tokens.empty() || start_slot != last_win_len_)
        return 0;  // only the tokens that directly continue the window
    WindowSeg* tail = nullptr;
    for (auto& s : segs_)
        if (s.kind == WindowSeg::Kind::Tail && s.slot_start <= start_slot &&
            start_slot == s.slot_start + s.slot_len)
            tail = &s;
    if (!tail)
        return 0;

    // The window is bounded; anything past the budget would be written nowhere.
    const int64_t room = budget_ - window_slots() - cfg_.gen_reserve;
    if (room <= 0)
        return 0;
    const int64_t take = std::min<int64_t>(room, static_cast<int64_t>(tokens.size()));
    if (take <= 0)
        return 0;

    // Growing the Tail keeps its logical mapping intact (logical_start + offset),
    // which is exactly right: the generated tokens are the logical successors of
    // the prompt tail.
    tail->slot_len += take;
    logical_hist_.insert(logical_hist_.end(), tokens.begin(),
                         tokens.begin() + static_cast<long>(take));
    last_win_len_ += take;
    window_len_ = last_win_len_;

    LOG_INFO("KVMem: folded " + std::to_string(take) +
             " generated token(s) into the logical history (window=" +
             std::to_string(window_slots()) + "/" + std::to_string(budget_) + ")");
    return take;
}

}  // namespace forge
