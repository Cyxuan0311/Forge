// Server-side per-session state.
//
// The KV cache is one shared resource, so before sessions every request addressed
// seq 0 and reused a single token mirror: two interleaved conversations
// overwrote each other's prefix, and the next turn re-prefilled from scratch.
// These tests pin the pieces that fix that without a full HTTP round-trip --
// seq allocation, LRU eviction, and the invariants that keep the default session
// byte-identical to the old single-slot behaviour.
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

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

// Mirrors ServerState::acquire_session / drop_session. Kept in lockstep with
// src/server/server_state.cpp; the point of the test is the allocation and
// eviction policy, not the surrounding server.
struct Session {
    int seq_id = 0;
    std::vector<int32_t> kv_tokens;
    int64_t last_active_ms = 0;
    uint64_t use_count = 0;
};

class SessionTable {
public:
    static constexpr int kMaxSessions = 32;

    Session& acquire(const std::string& id, int64_t now) {
        auto it = map_.find(id);
        if (it != map_.end()) {
            it->second.last_active_ms = now;
            ++it->second.use_count;
            return it->second;
        }
        if (static_cast<int>(map_.size()) >= kMaxSessions) {
            std::string victim;
            int64_t oldest = now;
            bool have = false;
            for (const auto& kv : map_) {
                if (!have || kv.second.last_active_ms < oldest) {
                    oldest = kv.second.last_active_ms;
                    victim = kv.first;
                    have = true;
                }
            }
            if (have)
                drop(victim);
        }
        Session s;
        s.seq_id = id.empty() ? 0 : 1 + next_seq_++;
        s.last_active_ms = now;
        s.use_count = 1;
        return map_.emplace(id, std::move(s)).first->second;
    }

    void drop(const std::string& id) { released_.push_back(map_[id].seq_id); map_.erase(id); }
    size_t size() const { return map_.size(); }
    bool has(const std::string& id) const { return map_.count(id) != 0; }
    int64_t released_count() const { return static_cast<int64_t>(released_.size()); }

private:
    std::unordered_map<std::string, Session> map_;
    int next_seq_ = 0;
    std::vector<int> released_;
};

// The default session must keep seq 0: every request without a session_id has to
// land on exactly the pages it used before sessions existed.
void test_default_session_keeps_seq_zero() {
    printf("[server] default session keeps seq 0 (backward compatible)\n");
    SessionTable t;
    check(t.acquire("", 1).seq_id == 0, "empty session_id maps to seq 0");
    check(t.acquire("", 2).seq_id == 0, "re-acquiring the default session stays on seq 0");
    check(t.size() == 1, "the default session is a single entry");
}

void test_distinct_sessions_get_distinct_seqs() {
    printf("[server] distinct sessions get distinct seq ids\n");
    SessionTable t;
    const int a = t.acquire("alpha", 1).seq_id;
    const int b = t.acquire("beta", 2).seq_id;
    const int c = t.acquire("gamma", 3).seq_id;
    check(a != b && b != c && a != c, "no two sessions share a seq id");
    check(a != 0 && b != 0 && c != 0, "non-default sessions avoid seq 0");
    // Re-acquiring must be idempotent, not allocate again.
    check(t.acquire("beta", 4).seq_id == b, "re-acquiring a session returns the same seq");
    check(t.size() == 3, "no extra entry created on re-acquire");
}

void test_token_mirrors_are_independent() {
    printf("[server] token mirrors do not leak across sessions\n");
    SessionTable t;
    t.acquire("alpha", 1).kv_tokens = {10, 11, 12};
    t.acquire("beta", 1).kv_tokens = {20};
    check(t.acquire("alpha", 2).kv_tokens.size() == 3, "alpha keeps its own mirror");
    check(t.acquire("beta", 2).kv_tokens == std::vector<int32_t>{20},
          "beta's mirror is untouched by alpha's write");
}

// The failure mode this replaces: a second conversation shortening the prefix the
// first one is about to match against.
void test_prefix_match_is_per_session() {
    printf("[server] prefix reuse only sees its own session's tokens\n");
    SessionTable t;
    Session& a = t.acquire("alpha", 1);
    a.kv_tokens = {1, 2, 3, 4, 5};
    Session& b = t.acquire("beta", 1);
    b.kv_tokens = {9, 9, 9};

    const std::vector<int32_t> next_prompt_a = {1, 2, 3, 4, 5, 77};
    const std::vector<int32_t> next_prompt_b = {9, 9, 9, 88};

    auto common = [](const std::vector<int32_t>& cached, const std::vector<int32_t>& prompt) {
        const size_t lim = std::min(cached.size(), prompt.size());
        size_t m = 0;
        while (m < lim && cached[m] == prompt[m])
            ++m;
        return m;
    };
    check(common(t.acquire("alpha", 2).kv_tokens, next_prompt_a) == 5,
          "alpha matches its full prefix");
    check(common(t.acquire("beta", 2).kv_tokens, next_prompt_b) == 3,
          "beta matches its own full prefix");
    // Cross-contamination would show up as beta matching alpha's tokens.
    check(common(t.acquire("beta", 2).kv_tokens, next_prompt_a) == 0,
          "beta cannot reuse alpha's prefix");
}

void test_lru_eviction_releases_kv() {
    printf("[server] the table is bounded and evicts LRU\n");
    SessionTable t;
    for (int i = 0; i < SessionTable::kMaxSessions; ++i)
        t.acquire("s" + std::to_string(i), 100 + i);
    check(t.size() == SessionTable::kMaxSessions, "table filled to capacity");

    // Touch s0 so it is no longer the oldest, then overflow.
    t.acquire("s0", 1000);
    t.acquire("overflow", 1001);
    check(t.size() == SessionTable::kMaxSessions, "size stays bounded after overflow");
    check(t.has("overflow"), "the new session is present");
    check(!t.has("s1"), "the least-recently-used session was evicted");
    check(t.has("s0"), "the recently used session survived");
    check(t.released_count() == 1, "the evicted session's seq was released");
}

void test_session_count_matches_inserts() {
    printf("[server] session count tracks distinct ids\n");
    SessionTable t;
    for (int i = 0; i < 5; ++i)
        t.acquire("k" + std::to_string(i), i);
    for (int i = 0; i < 20; ++i)  // hammering existing ids must not grow the table
        t.acquire("k" + std::to_string(i % 5), 100 + i);
    check(t.size() == 5, "repeat requests do not create sessions");
}

}  // namespace

int main() {
    test_default_session_keeps_seq_zero();
    test_distinct_sessions_get_distinct_seqs();
    test_token_mirrors_are_independent();
    test_prefix_match_is_per_session();
    test_lru_eviction_releases_kv();
    test_session_count_matches_inserts();

    if (g_failures == 0) {
        printf("all server session-state tests passed\n");
        return 0;
    }
    printf("%d server session test(s) FAILED\n", g_failures);
    return 1;
}