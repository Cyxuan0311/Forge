/**
 * ReasoningFilter implementation — byte-oriented <think> state machine.
 *
 * Tags are pure ASCII, so matching byte-by-byte is safe even when chunks cut
 * mid-UTF-8-sequence: those bytes simply pass through the matching state into
 * whichever output the current state targets, exactly as they arrived.
 *
 * Mismatch handling: while `hold_` is a strict prefix of the current state's
 * candidate tag we keep holding; on the first mismatching byte we emit the
 * longest suffix of `hold_` that is *still* a prefix of the tag back into
 * hold_ (a "<<" can itself open a tag) and flush the rest as literal content.
 */

#include "forge/agent/reasoning_filter.h"

namespace forge {

namespace {

constexpr const char* kOpenTag = "<think>";
constexpr const char* kCloseTag = "</think>";

/// True when `hold` is a prefix of (or equal to) `tag`.
bool is_tag_prefix(const std::string& hold, const char* tag) {
    size_t tag_len = 0;
    while (tag[tag_len] != '\0')
        ++tag_len;
    if (hold.size() >= tag_len)
        return false;  // longer than the tag cannot be a prefix
    return std::char_traits<char>::compare(tag, hold.data(), hold.size()) == 0;
}

/// Length of the longest suffix of `hold` that is a non-empty prefix of `tag`.
size_t longest_tag_suffix(const std::string& hold, const char* tag) {
    const size_t tag_len = std::string(tag).size();
    const size_t max = std::min(hold.size(), tag_len - 1);
    for (size_t len = max; len >= 1; --len) {
        if (hold.compare(hold.size() - len, len, tag, 0, len) == 0)
            return len;
    }
    return 0;
}

}  // namespace

void ReasoningFilter::flush_hold(std::string* visible, std::string* reasoning) {
    if (hold_.empty())
        return;
    const char* tag = (state_ == State::InThink) ? kCloseTag : kOpenTag;
    const size_t keep = longest_tag_suffix(hold_, tag);
    std::string literal = hold_.substr(0, hold_.size() - keep);
    hold_ = hold_.substr(hold_.size() - keep);
    if (state_ == State::InThink) {
        if (reasoning)
            *reasoning += literal;
    } else {
        if (visible)
            *visible += literal;
    }
}

void ReasoningFilter::feed_byte(char c, std::string* visible, std::string* reasoning) {
    hold_ += c;
    if (state_ == State::InThink) {
        if (hold_ == kCloseTag) {
            hold_.clear();
            state_ = State::Passthrough;
            return;
        }
        if (is_tag_prefix(hold_, kCloseTag))
            return;  // still ambiguous: keep holding
        flush_hold(nullptr, reasoning);
        return;
    }
    // Passthrough
    if (hold_ == kOpenTag) {
        hold_.clear();
        state_ = State::InThink;
        return;
    }
    if (is_tag_prefix(hold_, kOpenTag))
        return;
    flush_hold(visible, reasoning);
}

void ReasoningFilter::feed(const std::string& delta, std::string* visible,
                           std::string* reasoning) {
    for (char c : delta)
        feed_byte(c, visible, reasoning);
}

void ReasoningFilter::flush(std::string* visible, std::string* reasoning) {
    if (hold_.empty())
        return;
    if (state_ == State::InThink) {
        // Unclosed think block: the held bytes are reasoning (the tag never
        // completed, so nothing is ambiguous anymore).
        if (reasoning)
            *reasoning += hold_;
    } else {
        if (visible)
            *visible += hold_;
    }
    hold_.clear();
}

void ReasoningFilter::strip(const std::string& text, std::string* visible,
                            std::string* reasoning) {
    static const std::string open = kOpenTag;
    static const std::string close = kCloseTag;
    std::string vis;
    std::string rea;
    size_t pos = 0;
    while (true) {
        const size_t t = text.find(open, pos);
        if (t == std::string::npos) {
            vis += text.substr(pos);
            break;
        }
        vis += text.substr(pos, t - pos);
        const size_t e = text.find(close, t + open.size());
        if (e == std::string::npos) {
            // Unclosed block: the whole tail is reasoning.
            rea += text.substr(t + open.size());
            pos = text.size();
            break;
        }
        rea += text.substr(t + open.size(), e - (t + open.size()));
        pos = e + close.size();
    }
    if (visible)
        *visible = std::move(vis);
    if (reasoning)
        *reasoning = std::move(rea);
}

}  // namespace forge
