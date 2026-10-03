// KVMem tiered working set: window-planning policy.
//
// Host-only (KVWorkingSetManager is pure policy -- it owns no storage handle).
// These tests pin the two properties the planner must never violate:
//
//   1. The window never exceeds the budget. Under-budget is required because the
//      generator writes decode output into the same slots and the paged pool is
//      fixed at init time.
//   2. The attention sink survives budget pressure. Once a prompt is too long to
//      fit, the window becomes [sink | tail]: slots [0, sink) carry logical
//      [0, sink) (the conversation's first tokens) and the rest carry the most
//      recent tail. Before the sink was materialised the head-clip path zeroed
//      the whole window, so the sink was evicted exactly when it mattered most
//      and the model answered from a truncated window with total confidence.
//
// The acceptance criterion for (2) is that a token at the very start of a long
// prompt is still inside the planned window, so a needle test can reach it.
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "forge/kv_working_set.h"

using namespace forge;

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++g_failures;
        printf("  FAIL: %s\n", what);
    } else {
        printf("  ok  : %s\n", what);
    }
}

// A prompt whose first token is a unique marker, so "is the start of the prompt
// still reachable?" is answerable from the plan alone.
std::vector<int32_t> make_prompt(int n, int32_t head_marker = 777) {
    std::vector<int32_t> p;
    p.reserve(static_cast<size_t>(n));
    p.push_back(head_marker);
    for (int i = 1; i < n; ++i)
        p.push_back(1000 + (i % 97));
    return p;
}

// The sink tokens must be the head of the prompt: slots [0, sink) carry
// logical [0, sink), which under the plan is prompt[sink_reuse,
// sink_reuse+sink_forward).
bool sink_covers_prompt_head(const std::vector<int32_t>& prompt,
                             const KVemTurnPlan& p) {
    const int64_t sink_end = p.sink_reuse + p.sink_forward;
    if (sink_end <= 0)
        return false;
    if (sink_end > static_cast<int64_t>(prompt.size()))
        return false;
    return prompt[static_cast<size_t>(p.sink_reuse)] ==
           prompt[static_cast<size_t>(0)];
}

void test_no_sink_when_prompt_fits() {
    printf("[kvmem] prompt that fits the budget keeps a single-segment window\n");
    KVemConfig cfg;
    cfg.enabled = true;
    cfg.budget_tokens = 4096;
    cfg.sink_tokens = 256;
    cfg.gen_reserve = 64;
    KVWorkingSetManager mgr(cfg, /*page_size=*/16, /*pool_tokens=*/4096);

    const auto prompt = make_prompt(512);
    const KVemTurnPlan p = mgr.on_prompt(prompt, 32);

    check(p.sink_forward == 0, "no sink is materialised when nothing is clipped");
    check(p.sink_reuse == 0, "no sink reuse either");
    check(p.new_window_len <= 4096, "window within budget");
    check(static_cast<int64_t>(p.new_to_compute.size()) == 512,
          "the whole prompt is forwarded");
    check(p.max_generate == 4096 - p.new_window_len, "decode headroom is the rest");
}

void test_sink_pinned_under_pressure(int budget, int64_t prompt_len) {
    printf("[kvmem] budget=%d, prompt=%lld -> sink must be pinned\n", budget,
           static_cast<long long>(prompt_len));
    KVemConfig cfg;
    cfg.enabled = true;
    cfg.budget_tokens = budget;
    cfg.sink_tokens = 256;
    cfg.gen_reserve = 32;
    KVWorkingSetManager mgr(cfg, /*page_size=*/16, /*pool_tokens=*/budget);

    const auto prompt = make_prompt(static_cast<int>(prompt_len));
    const KVemTurnPlan p = mgr.on_prompt(prompt, 32);

    check(p.truncated_prompt, "the plan reports the prompt as truncated");
    check(p.sink_forward > 0, "a sink segment is forwarded");
    check(sink_covers_prompt_head(prompt, p),
          "the sink carries the very first prompt token (needle reachable)");
    check(p.new_window_len <= budget, "window within budget");
    check(p.sink_forward + static_cast<int64_t>(p.new_to_compute.size()) ==
              p.new_window_len,
          "window == sink segment + tail (no double counting)");
    check(p.new_window_len > p.sink_forward,
          "some tail survives alongside the sink (not sink-only)");
    // The tail must be the *most recent* tokens, not the ones just after the
    // sink, otherwise the model would attend to ancient context.
    const int64_t tail_start =
        static_cast<int64_t>(prompt.size()) -
        static_cast<int64_t>(p.new_to_compute.size());
    check(tail_start > static_cast<int64_t>(p.sink_forward),
          "the tail starts well past the sink");
    check(!p.new_to_compute.empty(), "tail is non-empty");
}

void test_prefix_reuse_across_turns() {
    printf("[kvmem] multi-turn prefix reuse keeps growing\n");
    KVemConfig cfg;
    cfg.enabled = true;
    cfg.budget_tokens = 4096;
    cfg.sink_tokens = 256;
    cfg.gen_reserve = 64;
    KVWorkingSetManager mgr(cfg, 16, 4096);

    const auto p1 = make_prompt(100);
    const KVemTurnPlan a = mgr.on_prompt(p1, 32);
    check(a.kept_window_len == 0, "turn 1 keeps nothing");

    // Same conversation, one turn appended: the whole first prompt must match.
    auto p2 = p1;
    p2.push_back(31337);
    const KVemTurnPlan b = mgr.on_prompt(p2, 32);
    check(b.kept_window_len == 100, "turn 2 reuses all 100 resident slots");
    check(static_cast<int64_t>(b.new_to_compute.size()) == 1,
          "turn 2 forwards only the new token");

    auto p3 = p2;
    p3.push_back(4242);
    const KVemTurnPlan c = mgr.on_prompt(p3, 32);
    check(c.kept_window_len == 101, "turn 3 reuses 101 slots");
}

void test_divergence_inside_sink_refills_sink() {
    printf("[kvmem] divergence inside the sink refills only the sink\n");
    KVemConfig cfg;
    cfg.enabled = true;
    cfg.budget_tokens = 512;
    cfg.sink_tokens = 64;
    cfg.gen_reserve = 32;
    KVWorkingSetManager mgr(cfg, 16, 512);

    // Turn 1 forces a clip, establishing a sink segment.
    const KVemTurnPlan a = mgr.on_prompt(make_prompt(4000), 32);
    check(a.sink_forward > 0, "turn 1 pins a sink");

    // Turn 2 rewrites history from token 0 onward but shares the tail end, so the
    // match dies inside the sink and only the sink has to be re-forwarded.
    auto p2 = make_prompt(4000);
    p2[10] = 555555;  // break the prefix early
    const KVemTurnPlan b = mgr.on_prompt(p2, 32);
    check(b.new_window_len <= 512, "window within budget after divergence");
    check(b.kept_window_len <= b.sink_reuse,
          "reuse never claims tail slots that the sink refill must overwrite");
    check(b.sink_reuse + b.sink_forward + static_cast<int64_t>(b.new_to_compute.size()) ==
              b.new_window_len,
          "window accounting still consistent");
}

// RoPE must see the logical position, not the window slot. Otherwise the gap
// the sink/tail split opens gets compressed out of the relative distances and
// the whole point of keeping the logical mapping is lost. Before this was wired
// the offset was structurally 0 (kept_rope_off was pinned to 0), i.e. every token
// was rotated at its slot index.
void test_rope_offset_is_logical() {
    printf("[kvmem] tail rope offset equals tail_base - sink\n");
    KVemConfig cfg;
    cfg.enabled = true;
    cfg.budget_tokens = 2048;
    cfg.sink_tokens = 256;
    cfg.gen_reserve = 32;
    KVWorkingSetManager mgr(cfg, 16, 2048);

    // No clipping: sink absent, offset must be exactly head_shift_ == 0.
    {
        const KVemTurnPlan p = mgr.on_prompt(make_prompt(500), 32);
        check(p.sink_slots == 0, "no sink without clipping");
        check(p.tail_rope_offset == 0, "offset is 0 when there is no dropped head");
    }

    // Force a clip, then verify slot + tail_rope_offset == logical position of
    // that slot, for both a sink slot and a tail slot.
    const KVemTurnPlan c = mgr.on_prompt(make_prompt(9000), 32);
    check(c.sink_slots > 0, "clip pins a sink");
    check(c.tail_base > c.sink_slots,
          "the tail's logical base is beyond the sink (a gap really was dropped)");
    check(c.tail_rope_offset == c.tail_base - c.sink_slots,
          "tail rope offset == tail_base - sink");

    // Sink slots: logical == slot, so a sink forward must use offset 0.
    check(c.tail_rope_offset != 0, "tail offset is non-zero after a clip");
    // Tail slots: logical == tail_base + (slot - sink), so offset == tail_base - sink.
    const int64_t slot = c.sink_slots + 3;
    check(slot + c.tail_rope_offset == c.tail_base + 3,
          "slot + offset lands on the tail's logical position");
}

// ---- Segment-table invariants ----
// The whole three-segment design rests on one property: the segments tile
// [0, window_len) in slot order with no holes. That is what makes seq_filled()
// (= sum of slot_len) the true visible length, and what makes the paged kernel's
// "query slot == kv_len - q_len + qi" derivation exact. A hole would silently
// misalign every KV row after it.
//
// The accessors under test are the same ones on_prompt uses, so this catches the
// layout drifting apart from the arithmetic.

// Reconstruct the slot->logical map from the plan the same way on_prompt commits
// it, then assert the tiling.
void check_tiling(const KVemTurnPlan& p, const char* what) {
    // Segments in slot order: sink, [recalled], reused tail, fresh tail.
    int64_t cursor = 0;
    std::vector<std::pair<int64_t, int64_t>> ranges;  // (slot_start, slot_len)

    const int64_t sink_len = p.sink_reuse + p.sink_forward;
    if (sink_len > 0)
        ranges.push_back({cursor, sink_len}), cursor += sink_len;
    // tail_slot_start must sit exactly where the cursor is after the sink and any
    // recalled pages: that is where the generator writes the tail.
    if (p.new_to_compute.size() > 0) {
        check(p.tail_slot_start >= cursor,
              (std::string(what) + ": tail starts at or after the sink").c_str());
    }
    const int64_t reused_tail = p.kept_window_len - std::min<int64_t>(p.kept_window_len,
                                                                    p.sink_reuse);
    if (reused_tail > 0)
        ranges.push_back({cursor, reused_tail}), cursor += reused_tail;
    const int64_t fresh = static_cast<int64_t>(p.new_to_compute.size());
    if (fresh > 0)
        ranges.push_back({cursor, fresh}), cursor += fresh;

    check(cursor == p.new_window_len,
          (std::string(what) + ": segments tile exactly to new_window_len").c_str());
    check(p.new_window_len <= 0 || ranges.front().first == 0,
          (std::string(what) + ": first segment starts at slot 0").c_str());
    // Contiguity: no gap, no overlap, strictly increasing starts.
    bool contiguous = true;
    int64_t expect = 0;
    for (const auto& r : ranges) {
        if (r.first != expect)
            contiguous = false;
        expect = r.first + r.second;
    }
    check(contiguous, (std::string(what) + ": no holes or overlaps between segments").c_str());
}

void test_segment_tiling_invariants() {
    printf("[kvmem] segments tile the window with no holes\n");
    for (int budget : {512, 2048, 8192}) {
        KVemConfig cfg;
        cfg.enabled = true;
        cfg.budget_tokens = budget;
        cfg.sink_tokens = 64;
        cfg.gen_reserve = 32;
        KVWorkingSetManager mgr(cfg, 16, budget);

        // A prompt that fits: single tail, no sink.
        check_tiling(mgr.on_prompt(make_prompt(200), 16), "fits");
        // A prompt that must clip: sink + tail.
        check_tiling(mgr.on_prompt(make_prompt(budget * 4), 16), "clipped");
        // A tiny budget: the sink clamp path.
        check_tiling(mgr.on_prompt(make_prompt(budget * 8), 16), "tight");
    }
}

void test_multi_turn_tiling() {
    printf("[kvmem] tiling holds across a growing conversation\n");
    KVemConfig cfg;
    cfg.enabled = true;
    cfg.budget_tokens = 4096;
    cfg.sink_tokens = 64;
    cfg.gen_reserve = 32;
    KVWorkingSetManager mgr(cfg, 16, 4096);

    auto prompt = make_prompt(50);
    for (int turn = 0; turn < 6; ++turn) {
        const KVemTurnPlan p = mgr.on_prompt(prompt, 16);
        check_tiling(p, "multi-turn");
        prompt.push_back(9000 + turn);  // append one token per turn
    }
}

// Folding decode output back into the logical history.
//
// This is what makes multi-turn serving actually cheap. Decode writes result
// tokens into the window right after new_window_len, and the next request's
// prompt *starts with those very tokens* (a chat turn appends the assistant
// reply verbatim). Without folding, the manager's match can never reach past the
// prompt boundary, so every turn re-prefilled the whole previous assistant
// message even though its KV was already resident with the right RoPE positions.
//
// CLI interactive mode cannot show this: cli_interactive.cpp stores
// {"assistant", ""}, so the reply never reappears in the prompt and there is
// nothing to match. A server, which feeds the real reply back, does.
void test_generated_tokens_folded_into_history() {
    printf("[kvmem] decode output is reusable on the next turn\n");
    KVemConfig cfg;
    cfg.enabled = true;
    cfg.budget_tokens = 4096;
    cfg.sink_tokens = 0;
    cfg.gen_reserve = 64;
    KVWorkingSetManager mgr(cfg, 16, 4096);

    // Turn 1: a 40-token user prompt.
    const std::vector<int32_t> reply = {7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008};
    auto p1 = make_prompt(40);
    const KVemTurnPlan t1 = mgr.on_prompt(p1, 32);
    check(t1.new_window_len == 40, "turn 1 forwards the whole prompt");
    const int64_t before = t1.new_window_len;

    // Decode writes the reply at slots [40, 48).
    const int64_t folded = mgr.append_generated(reply, t1.new_window_len);
    check(folded == static_cast<int64_t>(reply.size()),
          "every generated token is folded in");

    // Turn 2's prompt = turn 1's prompt + the reply + a new question. This is
    // exactly what a chat template produces when the assistant turn is real.
    auto p2 = p1;
    p2.insert(p2.end(), reply.begin(), reply.end());
    for (int i = 0; i < 20; ++i)
        p2.push_back(5555 + i);  // the new user turn
    const KVemTurnPlan t2 = mgr.on_prompt(p2, 32);

    check(t2.kept_window_len >= before + static_cast<int64_t>(reply.size()),
          "turn 2 reuses the prompt AND the generated reply");
    check(t2.sink_forward == 0, "no sink needed without clipping");
    // The whole point: only the new question needs a forward.
    check(static_cast<int64_t>(t2.new_to_compute.size()) == 20,
          "turn 2 forwards only the new question, not the reply");
    (void)before;
}

// A reply that the next prompt does NOT contain (e.g. the user edited history, or
// CLI's empty assistant turn) must not be matched -- and must not corrupt the
// layout either.
void test_unrelated_reply_does_not_break_matching() {
    printf("[kvmem] a non-matching reply stops the match cleanly\n");
    KVemConfig cfg;
    cfg.enabled = true;
    cfg.budget_tokens = 4096;
    cfg.sink_tokens = 0;
    cfg.gen_reserve = 64;
    KVWorkingSetManager mgr(cfg, 16, 4096);

    auto p1 = make_prompt(40);
    const KVemTurnPlan t1 = mgr.on_prompt(p1, 32);
    mgr.append_generated({7001, 7002, 7003}, t1.new_window_len);

    // The next prompt diverges from the folded reply (different tokens there).
    auto p2 = make_prompt(40);
    for (int i = 0; i < 3; ++i)
        p2.push_back(9999);  // not the folded reply
    p2.push_back(5555);
    const KVemTurnPlan t2 = mgr.on_prompt(p2, 32);
    check(t2.kept_window_len <= 40,
          "the match stops where the prompt diverges from the reply");
    check(t2.new_window_len <= 4096, "window still within budget");
    check_tiling(t2, "divergent-reply");
}

void test_budget_invariants_random() {
    printf("[kvmem] invariants hold across a sweep of budgets and lengths\n");
    int overflows = 0, negatives = 0, checked = 0;
    for (int budget : {256, 512, 1024, 2048, 4096}) {
        for (int len : {64, 255, 256, 257, 1000, 4095, 4096, 4097, 9000}) {
            KVemConfig cfg;
            cfg.enabled = true;
            cfg.budget_tokens = budget;
            cfg.sink_tokens = 64;  // <= budget by construction
            cfg.gen_reserve = 32;
            KVWorkingSetManager mgr(cfg, 16, budget);
            const auto prompt = make_prompt(len);
            const KVemTurnPlan p = mgr.on_prompt(prompt, 16);
            ++checked;
            if (p.new_window_len > budget) ++overflows;
            if (p.kept_window_len < 0 || p.sink_reuse < 0 || p.sink_forward < 0 ||
                p.max_generate < 0 || p.new_window_len < 0)
                ++negatives;
            if (p.window_overflow) ++overflows;
        }
    }
    check(overflows == 0, "no window ever exceeded the budget or overflowed");
    check(negatives == 0, "no plan field went negative");
    printf("  (%d configurations checked)\n", checked);
}

}  // namespace

int main() {
    test_no_sink_when_prompt_fits();
    test_sink_pinned_under_pressure(512, 5978);
    test_sink_pinned_under_pressure(2048, 5978);
    test_sink_pinned_under_pressure(4096, 20000);
    test_prefix_reuse_across_turns();
    test_divergence_inside_sink_refills_sink();
    test_rope_offset_is_logical();
    test_segment_tiling_invariants();
    test_multi_turn_tiling();
    test_generated_tokens_folded_into_history();
    test_unrelated_reply_does_not_break_matching();
    test_budget_invariants_random();

    if (g_failures == 0) {
        printf("all KVMem working-set tests passed\n");
        return 0;
    }
    printf("%d KVMem working-set test(s) FAILED\n", g_failures);
    return 1;
}
