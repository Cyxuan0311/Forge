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
    };
    const Stats& stats() const { return stats_; }

private:
    KVemConfig cfg_;
    int page_size_ = 16;
    int budget_ = 0;

    // Previous turn's logical history; source of the common-prefix match.
    // Invariant: window slot w holds the token logical_hist_[w] (the window
    // always starts at logical position 0 of the shifted space below).
    std::vector<int32_t> logical_hist_;
    int64_t last_win_len_ = 0;
    int64_t last_rope_off_ = 0;

    // Logical position p corresponds to prompt index p + head_shift_.
    // > 0 only after a single prompt exceeded the whole budget and its head
    // was clipped (the last-resort path); the shift keeps the logical history
    // aligned with the *next* prompt so the common-prefix match survives.
    int64_t head_shift_ = 0;

    // Current turn's window state.
    int64_t window_len_ = 0;
    int64_t rope_off_ = 0;

    Stats stats_;
};

}  // namespace forge
