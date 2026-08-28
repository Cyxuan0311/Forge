#include "forge/speculative.h"

#include <algorithm>
#include <mutex>

#include "forge/sampler.h"

namespace forge {

// =========================================================================
// SpeculativeStats
// =========================================================================

void SpeculativeStats::reset() { *this = SpeculativeStats{}; }

double SpeculativeStats::acceptance_rate() const {
    return n_draft_tokens > 0
               ? static_cast<double>(n_accepted_tokens) / static_cast<double>(n_draft_tokens)
               : 0.0;
}

double SpeculativeStats::tokens_per_step() const {
    return n_spec_steps > 0
               ? static_cast<double>(n_output_tokens) / static_cast<double>(n_spec_steps)
               : 0.0;
}

// =========================================================================
// NgramDraftProvider -- rolling-hash indexed suffix matching
// =========================================================================

NgramDraftProvider::NgramDraftProvider(int ngram_n, int ngram_min)
    : ngram_n_(ngram_n), ngram_min_(ngram_min) {}

uint64_t NgramDraftProvider::hash_window(const int32_t* ptr, int len) {
    // FNV-1a 64 with length salt to separate different L values
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(ptr[i]) + 0x9e3779b9u);
        h *= 1099511628211ull;
    }
    h ^= static_cast<uint64_t>(len) * 0x9e3779b97f4a7c15ull;
    h ^= h >> 33;
    return h;
}

void NgramDraftProvider::index_range(int old_len, int new_len) {
    for (int L = ngram_min_; L <= ngram_n_; ++L) {
        const int s_begin = std::max(0, old_len - L + 1);
        const int s_end = new_len - L;
        for (int s = s_begin; s <= s_end; ++s) {
            uint64_t h = hash_window(history_.data() + s, L);
            index_[h].push_back(s);
        }
    }
}

void NgramDraftProvider::begin(const std::vector<int32_t>& prompt) {
    history_ = prompt;
    index_.clear();
    index_range(0, static_cast<int>(history_.size()));
}

std::vector<int32_t> NgramDraftProvider::draft(int32_t last_token, int n_draft) {
    (void)last_token;
    std::vector<int32_t> result;
    const int len = static_cast<int>(history_.size());
    if (len < ngram_min_ || n_draft <= 0) return result;

    const int max_L = std::min(ngram_n_, len);
    for (int L = max_L; L >= ngram_min_; --L) {
        uint64_t h = hash_window(history_.data() + len - L, L);
        auto it = index_.find(h);
        if (it == index_.end()) continue;
        // Verify hash hit to guard against rare collision
        const auto& positions = it->second;
        for (auto p = positions.rbegin(); p != positions.rend(); ++p) {
            const int s = *p;
            if (s + L > len - 1) continue;
            // Collision check: full memcmp on candidate key
            bool eq = true;
            const int32_t* cand = history_.data() + s;
            const int32_t* suffix = history_.data() + len - L;
            for (int k = 0; k < L; ++k) if (cand[k] != suffix[k]) { eq = false; break; }
            if (!eq) continue;
            const int avail = std::min(n_draft, len - s - L);
            result.assign(history_.begin() + s + L,
                          history_.begin() + s + L + avail);
            return result;
        }
    }

    return result;
}

void NgramDraftProvider::accept(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return;
    const int old_len = static_cast<int>(history_.size());
    history_.insert(history_.end(), tokens.begin(), tokens.end());
    index_range(old_len, static_cast<int>(history_.size()));
}

void NgramDraftProvider::reset() {
    history_.clear();
    index_.clear();
}

// =========================================================================
// NgramModProvider -- global hash-pool (llama.cpp ngram-mod style)
// =========================================================================

size_t NgramModProvider::pool_size_ = 4 * 1024 * 1024;

NgramModProvider::SharedPool& NgramModProvider::shared_pool() {
    static SharedPool pool;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        pool.table.assign(pool_size_, -1);
        pool.used = 0;
    });
    // Resize if config requests larger pool than default
    if (pool.table.size() != pool_size_) {
        std::lock_guard<std::mutex> lk(pool.mutex);
        if (pool.table.size() != pool_size_) {
            pool.table.assign(pool_size_, -1);
            pool.used = 0;
        }
    }
    return pool;
}

uint64_t NgramModProvider::lcg_hash(const int32_t* tokens, int n) {
    uint64_t h = 0;
    for (int i = 0; i < n; ++i) {
        h = h * 6364136223846793005ULL + static_cast<uint64_t>(static_cast<uint32_t>(tokens[i]));
    }
    return h;
}

NgramModProvider::NgramModProvider(int n_match, int n_min, int n_max, size_t pool_size)
    : n_match_(n_match), n_min_(n_min), n_max_(n_max) {
    if (pool_size != pool_size_) {
        pool_size_ = pool_size;
    }
    // Ensure pool allocated
    (void)shared_pool();
}

void NgramModProvider::begin(const std::vector<int32_t>& prompt) {
    history_ = prompt;
    i_last_ = 0;
    n_draft_last_ = 0;
    n_low_streak_ = 0;
    // Feed prompt[0 .. size-n_match-1] into pool: each window's next token
    auto& pool = shared_pool();
    std::lock_guard<std::mutex> lk(pool.mutex);
    int limit = static_cast<int>(history_.size()) - n_match_;
    for (; i_last_ < limit; ++i_last_) {
        uint64_t h = lcg_hash(history_.data() + i_last_, n_match_) % pool.table.size();
        int32_t nxt = history_[i_last_ + n_match_];
        if (pool.table[h] == -1) ++pool.used;
        pool.table[h] = nxt;
    }
}

std::vector<int32_t> NgramModProvider::draft(int32_t last_token, int n_draft) {
    (void)last_token;
    // Lazy indexing: push history that has become indexable since last call
    {
        auto& pool = shared_pool();
        std::lock_guard<std::mutex> lk(pool.mutex);
        int limit = static_cast<int>(history_.size()) - n_match_;
        // Index every 32 tokens batch like llama.cpp, but we do all pending to keep simple
        // (still O(n) overall)
        for (; i_last_ < limit; ++i_last_) {
            // Avoid indexing windows that overlap with not-yet-committed tail
            // (history_.size() is confirmed only, so safe)
            uint64_t h = lcg_hash(history_.data() + i_last_, n_match_) % pool.table.size();
            int32_t nxt = history_[i_last_ + n_match_];
            if (pool.table[h] == -1) ++pool.used;
            pool.table[h] = nxt;
        }
    }

    const int len = static_cast<int>(history_.size());
    if (len < n_match_ || n_draft <= 0) return {};

    int want = std::min({n_draft, n_max_, static_cast<int>(history_.size())}); // bounded
    // Iteratively pull from pool
    std::vector<int32_t> result;
    result.reserve(want);
    // Temporary working suffix: we grow result and use last n_match tokens
    // Seed suffix is history tail
    std::vector<int32_t> work;
    work.reserve(n_match_ + want);
    work.assign(history_.end() - n_match_, history_.end());

    auto& pool = shared_pool();
    for (int i = 0; i < want; ++i) {
        uint64_t h;
        {
            std::lock_guard<std::mutex> lk(pool.mutex);
            h = lcg_hash(work.data() + work.size() - n_match_, n_match_) % pool.table.size();
            int32_t nxt = pool.table[h];
            if (nxt == -1) break;
            result.push_back(nxt);
            work.push_back(nxt);
        }
    }
    if (static_cast<int>(result.size()) < n_min_) {
        result.clear();
    }
    n_draft_last_ = static_cast<int>(result.size());
    return result;
}

void NgramModProvider::accept(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return;
    // Update low-accept heuristic: f = n_accepted / n_draft_last
    // llama.cpp triggers reset after 5 consecutive <0.25 rounds (shared pool!)
    if (n_draft_last_ > 0) {
        double f = static_cast<double>(tokens.size()) / static_cast<double>(n_draft_last_);
        // For mod, tokens == accepted+resampled/bonus (up to n_draft_last+1)
        // Approximate acceptance as min(tokens.size(), n_draft_last)/n_draft_last
        double acc = std::min(static_cast<int>(tokens.size()), n_draft_last_) / static_cast<double>(n_draft_last_);
        (void)f;
        if (acc < 0.25) ++n_low_streak_;
        else n_low_streak_ = 0;
        if (n_low_streak_ >= 5) {
            auto& pool = shared_pool();
            std::lock_guard<std::mutex> lk(pool.mutex);
            std::fill(pool.table.begin(), pool.table.end(), -1);
            pool.used = 0;
            n_low_streak_ = 0;
        }
    }
    history_.insert(history_.end(), tokens.begin(), tokens.end());
    // i_last_ will be advanced lazily in next draft()/begin()
}

void NgramModProvider::reset() {
    history_.clear();
    i_last_ = 0;
    n_draft_last_ = 0;
    n_low_streak_ = 0;
}

// =========================================================================
// verify_draft_tokens -- resample-consistency verification
// =========================================================================

SpecVerifyResult verify_draft_tokens(
    const float* logits_all, int vocab_size,
    const std::vector<int32_t>& draft_tokens,
    Sampler& sampler, const SpeculativeConfig& config) {
    (void)config;

    SpecVerifyResult result;
    const int n_draft = static_cast<int>(draft_tokens.size());

    // Use the sampler's full chain via sample_ptr so repeat penalty,
    // logit softcapping, top_k/top_p/top_p and temperature are all
    // honoured. This matches llama.cpp common_sampler_sample_and_accept_n
    // (which runs the full sampler chain per row) and guarantees
    // lossless resample-consistency when do_sample=true.
    auto sample_row = [&](int row) -> int32_t {
        const float* row_logits =
            logits_all + static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
        int32_t tok = sampler.sample_ptr(row_logits, vocab_size);
        sampler.add_token_to_history(tok);
        return tok;
    };

    for (int i = 0; i < n_draft; ++i) {
        const int32_t tok = sample_row(i);
        if (tok == draft_tokens[i]) {
            result.accepted_tokens.push_back(draft_tokens[i]);
            result.n_accepted++;
        } else {
            result.resampled = tok;
            return result;
        }
    }

    if (n_draft > 0) {
        result.bonus = sample_row(n_draft);
    }

    return result;
}

}  // namespace forge
