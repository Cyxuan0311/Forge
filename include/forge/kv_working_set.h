#pragma once

// KVWorkingSetManager — the KVMem-style tiered KV policy layer
// (arXiv:2609.04852, https://github.com/kvmem/kvmem-llama.cpp).
//
// Motivation
// ----------
// Dense attention pays O(n^2) in the resident history: at 256K the attention
// term alone swamps the prefill budget. KVMem's central observation is that a
// working set of only a few thousand tokens carries most of the information, so
// the KV cache should be *tiered* rather than grown linearly:
//
//   logical context (-c)   : how much conversation the user can hold, unbounded
//                            by device memory
//   GPU working set        : bounded, resident, what attention actually scans
//   host archive           : cold history kept off-device, recallable on demand
//
// The manager is the policy layer between the two. It decides, once per turn,
// which prefix of the logical history stays resident and which moves to the
// archive, and it emits the position-remap metadata the engine needs so that
// RoPE keeps the *logical* position while the KV write cursor and the
// attention mask use the *window* slot cursor.
//
// What it owns
// ------------
// * The per-turn plan (on_prompt): common-prefix match, stale-history
//   truncation, window reassembly, recall.
// * The position remap (logical<->window) consumed by the generator to set
//   ForwardRequest::start_pos / rope_pos_offset.
//
// What it does NOT own
// --------------------
// * Physical pages. The window/archive split of PagedKVStorage is the
//   mechanism; this manager is the policy. It calls the storage through
//   KVMemory and never touches pages directly.
//
// Short-circuit guarantee
// -----------------------
// The manager is attached only when tiered KV is enabled. When disabled
// (the default) KVMemory::working_set() is nullptr and every legacy code path
// keeps the invariant "window == logical", so prefix reuse, rollback and
// speculative decode are all bit-for-bit unchanged.

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "forge/kv_storage.h"

namespace forge {

class KVMemory;  // forward: the manager is attached to KVMemory, not built on it

// Policy knobs, mirroring the kvmem-llama.cpp flag surface.
//
// budget_tokens  : resident working-set budget. 0 = auto (sink + gen_reserve +
//                  the remainder of the GPU pool the caller reports). The pool
//                  cap is what turns the paged pool into a working set instead
//                  of a logical-context-sized arena.
// sink_tokens    : the first N tokens of the logical history stay resident
//                  forever. Prompt templates, chat delimiters and system
//                  prompts land here — they are referenced by every later
//                  position and the cheapest thing to keep hot.
// gen_reserve    : slots reserved at the tail of the window for the tokens this
//                  turn will generate. The window must never fill before the
//                  reply is done, otherwise mid-generation eviction would
//                  destroy the KV of a sequence we are currently writing.
// retrieval      : how archived history is brought back when the next prompt
//                  is mostly stale history.
//                  "recent" = take the most recent window-sized chunk
//                             (positional recall, what kvmem defaults to);
//                  "embed"  = reserve the hook for embedding-based recall; the
//                             manager keeps the same positional fallback until
//                             a retriever is wired in.
struct KVemConfig {
    bool enabled = false;
    int budget_tokens = 0;
    int sink_tokens = 256;
    int gen_reserve = 0;
    std::string retrieval = "recent";

    // Normalized budget: at least the sink, at least a page.
    int budget(int page_size, int pool_tokens) const {
        if (budget_tokens > 0)
            return budget_tokens;
        // Auto: the whole working set is the pool the caller sized, floored at
        // the sink so a tiny pool never disables the tier.
        int b = pool_tokens > 0 ? pool_tokens : 4096;
        if (b < sink_tokens)
            b = sink_tokens;
        if (b < page_size)
            b = page_size;
        return b;
    }
};

// Outcome of a turn-boundary plan. Consumed by the generator, which sets
// ForwardRequest::start_pos and rope_pos_offset from it.
//
//   kept_logical_len : the longest common prefix between the previous logical
//                      history and the new prompt; its KV is still resident at
//                      window slot [rope_off, kept_window_len) and needs no
//                      recompute.
//   kept_window_len  : window slots already holding valid KV after the plan
//                      (sink kept + recalled prefix). The prefill cursor starts
//                      here.
//   kept_rope_off    : the logical position of window slot 0. Window slot w
//                      maps to logical position kept_rope_off + w.
//   new_to_compute   : prompt tokens in [head_shift + kept_logical_len,
//                      prompt.size()) to forward, written to window slots
//                      [kept_window_len, kept_window_len + new_to_compute).
//   new_window_len   : window slots occupied once the prefill lands
//                      (== kept_window_len + new_to_compute).
//   new_rope_off     : kept_rope_off + kept_window_len.
//   dropped_archived : logical tokens lost to the archive budget this turn.
//   window_overflow  : the prompt could not fit even after dropping; the caller
//                      must grow the budget or shrink the prompt.
struct KVemTurnPlan {
    int64_t kept_logical_len = 0;
    int64_t kept_window_len = 0;
    int64_t kept_rope_off = 0;

    // ---- Attention-sink prefix (StreamingLLM-style) ----
    // The window is two contiguous slot segments: slots
    // [0, sink_reuse + sink_forward) hold logical [0, sink_reuse+sink_forward) --
    // the conversation's first tokens, which attention needs as a stable sink --
    // and the slots after them hold the most recent tail. The logical gap
    // between the two ranges is the history that was dropped, deliberately.
    //
    //   sink_reuse    sink slots still resident from the previous turn (no forward)
    //   sink_forward  sink slots to (re)forward into slots [sink_reuse, ...) from
    //                 prompt[0, sink_reuse+sink_forward)
    //   new_to_compute the tail tokens, written at slots [tail_slot_start, ...)
    //   tail_base     logical position of the first forwarded tail token; the
    //                 resident (reused) tail keeps the logical positions it was
    //                 written with
    //
    // With both sink counters at 0 the window is one contiguous run and the layout
    // is exactly the un-sunk one, so prompts that fit are unaffected.
    int64_t sink_reuse = 0;
    int64_t sink_forward = 0;
    int64_t tail_base = 0;
    // Window slot at which the tail segment starts. Normally sink_reuse +
    // sink_forward, but recalled archived pages sit between them, so the tail
    // forward must start after those too. The generator uses this as the tail
    // forward's start_pos.
    int64_t tail_slot_start = 0;
    // Total size of the sink segment after this turn (sink_reuse+sink_forward on
    // the turn that builds it; carried over unchanged on later turns). Needed by
    // the caller to turn a window slot into a logical position:
    //   slot <  sink_slots -> logical = slot
    //   slot >= sink_slots -> logical = tail_base + (slot - tail_slot_start)
    int64_t sink_slots = 0;
    // rope_pos_offset to use for a forward that writes *tail* slots. RoPE must
    // see the logical position, not the slot, so that the distance between the
    // sink and the recent tail is the true one rather than compressed by the
    // dropped middle. A forward that writes *sink* slots must use offset 0.
    int64_t tail_rope_offset = 0;

    std::vector<int32_t> new_to_compute;
    int64_t new_window_len = 0;
    int64_t new_rope_off = 0;
    int64_t max_generate = 0;  // remaining window slots for generation this turn
                               // (budget - new_window_len); the generator caps the
                               // reply here so the pool never fills mid-token.
    int dropped_archived = 0;
    bool window_overflow = false;
    bool truncated_prompt = false;  // prompt was clipped to the logical cap
};

class KVWorkingSetManager {
public:
    KVWorkingSetManager(KVemConfig config, int page_size, int pool_tokens);

    // ---- Turn boundary ----
    // Fold `prompt` into the logical history and rebuild the window. Must be
    // called before each generate(); the returned plan is what the generator
    // forwards. Not thread-safe: one conversation at a time, matching the
    // single-sequence (seq_id == 0) contract of the KV storage.
    // `gen_reserve` is the per-request generation budget (typically the request's
    // max_new_tokens). <= 0 means "use the configured default". It is clamped so
    // the window always keeps at least one page for the prompt tail.
    KVemTurnPlan on_prompt(const std::vector<int32_t>& prompt, int gen_reserve = -1);

    // The window length after the last on_prompt() — the logical length that
    // the engine sees (seq_filled).
    int64_t window_len() const { return window_len_; }

    // Window slot 0's logical position; the generator turns this into
    // ForwardRequest::rope_pos_offset = logical_pos - window_pos.
    int64_t rope_offset() const { return rope_off_; }

    // How many archived logical tokens are held, if anything, and where they
    // sit. Used by callers that want to know whether there is history worth
    // recalling.
    int64_t archived_tokens() const;
    int64_t archive_begin() const;
    int64_t archive_end() const;

    // ---- Configuration ----
    const KVemConfig& config() const { return cfg_; }
    int page_size() const { return page_size_; }
    int budget() const { return budget_; }
    bool sink_enabled() const { return cfg_.sink_tokens > 0; }
    int sink_tokens() const { return cfg_.sink_tokens; }
    int gen_reserve() const { return cfg_.gen_reserve; }

    // ---- Statistics ----
    struct Stats {
        int64_t turns = 0;
        int64_t recalled_tokens = 0;
        int64_t dropped_tokens = 0;
        int64_t window_overflow = 0;
        int64_t archived_pages = 0;
        int64_t recalled_pages = 0;
    };
    const Stats& stats() const { return stats_; }

    // Give the manager the storage it must move pages between, and arm the
    // per-sequence window ops. Until this is called the manager is pure policy:
    // it plans a window but never archives or recalls anything.
    //
    // The manager owns the slot layout while attached, so it must be detached
    // (or the sequence released) before anything else touches that sequence's
    // page table.
    void attach_storage(KVStorage* storage, int num_layers, int seq_id);

    // Keyword ("recent"-style) recall settings. `recall_pages` caps how many
    // archived pages one turn may pull back; `recall_min_score` is the minimum
    // per-page relevance below which a page is not worth a slot.
    void set_recall_policy(int max_pages, double min_score) {
        recall_max_pages_ = max_pages;
        recall_min_score_ = min_score;
    }

// Fold the tokens produced by decode back into the logical history.
    //
    // Decode writes them into seq 0 starting at the window slot just past
    // new_window_len, i.e. contiguous with the Tail segment, and the next turn's
    // prompt begins with exactly this sequence. Without folding them in the
    // manager's match can never reach them, so every turn re-prefilled the whole
    // previous assistant message -- pure waste, since the KV was already resident
    // and already carried the right logical RoPE positions.
    //
    // Returns the number of tokens adopted (0 when there is no manager, the tail
    // is absent, or the window has no room left).
    int64_t append_generated(const std::vector<int32_t>& tokens, int64_t start_slot);

private:
    // Archive the window pages at slots [0, from_slot) to the host pool, skipping
    // the pinned sink. Returns the number of tokens moved.
    int64_t archive_pages(int64_t from_slot);
    // Pull up to recall_max_pages_ archived pages back into the window, choosing by
    // lexical relevance to `prompt`. Appends a WindowSeg per restored page. Returns
    // the tokens restored.
    int64_t recall_pages(const std::vector<int32_t>& prompt, int64_t sink_slots);
    KVemConfig cfg_;
    int page_size_ = 16;
    int budget_ = 0;

    // Previous turn's logical history; source of the common-prefix match.
    // Invariant: window slot w holds the token logical_hist_[w] (the window
    // always starts at logical position 0 of the shifted space below).
    std::vector<int32_t> logical_hist_;
    int64_t last_win_len_ = 0;
    int64_t last_rope_off_ = 0;

    // Logical position of the *tail* segment's first slot. 0 unless a prompt
    // exceeded the whole budget and its head was clipped; then it is the first
    // logical position still covered by the tail, so the common-prefix match and
    // the recall source ranges stay aligned with the next prompt. Mirrors the
    // Tail segment's logical_start.
    int64_t head_shift_ = 0;

    // ---- Window layout: explicit segment table ----
    // The window is a contiguous run of slots [0, window_len) tiled by segs_ in
    // slot order, with NO holes. That single invariant is what makes everything
    // else safe:
    //   * seq_filled() (= the storage's logical_len, advanced as max(old, pos+n)
    //     over slot writes) equals sum(slot_len), so attention sees exactly the
    //     resident window;
    //   * the paged kernels derive a query's slot as kv_len - q_len + qi, which
    //     with a hole-free range is exactly the slot the token was written to, so
    //     the causal mask never mis-binds;
    //   * a recalled page can be *inserted* at any slot index without the
    //     kernels caring about order, because attention is order-independent over
    //     keys and each K already carries its logical RoPE position.
    //
    // Slots:  [ Sink ][ Recalled ][ Recalled ][ Tail ]
    // Logical:[0,S)  [A1,A1+n1)  [A2,A2+n2)   [base, ...)
    // The gaps between the logical ranges are history that was dropped on
    // purpose; they are what the archive holds.
    struct WindowSeg {
        int64_t slot_start = 0;
        int64_t slot_len = 0;
        int64_t logical_start = 0;
        enum class Kind { Sink, Recalled, Tail };
        Kind kind = Kind::Tail;
    };
    std::vector<WindowSeg> segs_;

    // Logical position of a window slot, or -1 when the slot is past the window.
    int64_t logical_of_slot(int64_t slot) const;
    // Sum of segment slot lengths; must always equal last_win_len_.
    int64_t window_slots() const;
    // Drop segments that no longer hold a slot range (after archiving), keeping
    // slot_start/slot_len consistent for the survivors.
    void compact_segments();

    // ---- Host archive + keyword recall ----
    // Storage handle, armed by attach_storage(). Null => the planner is pure
    // policy and never moves pages.
    KVStorage* storage_ = nullptr;
    int num_layers_ = 0;
    int seq_id_ = 0;

    // One archived window page: its token ids plus the logical range it covers,
    // so a recalled segment can be described exactly (and its tokens rebuilt into
    // logical_hist_ without re-reading the prompt).
    struct ArchivedChunk {
        int64_t logical_start = 0;
        int64_t slot_len = 0;
        std::vector<int32_t> tokens;
    };
    // Archived pages, in archive order. Kept in step with the storage's own
    // archived_pages list so a page can be scored without touching its KV.
    std::vector<ArchivedChunk> archived_chunks_;
    int recall_max_pages_ = 0;  // 0 = recall disabled (the default)
    double recall_min_score_ = 1.0;

    // Current turn's window state.
    int64_t window_len_ = 0;
    int64_t rope_off_ = 0;
    // head_shift_ value at which the truncation warning was last emitted, so the
    // steady-state loss is reported when it appears and when it grows materially
    // rather than once per turn.
    int64_t last_warned_head_shift_ = 0;

    Stats stats_;
};

}  // namespace forge
