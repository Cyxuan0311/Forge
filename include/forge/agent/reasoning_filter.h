/**
 * Agent module — streaming-safe <think> reasoning separator.
 *
 * Thinking models (Qwen3.x, MiniCPM5, ...) open their reply with a
 * `<think>...</think>` block. The block must never reach the client-visible
 * content: it is separated into the OpenAI-compatible `reasoning_content`
 * channel instead.
 *
 * The filter is a small byte-oriented state machine so a tag split across
 * streaming chunks ("<th", "ink>", ...) is still recognized; bytes that only
 * *look* like the start of a tag (e.g. a literal "a < b" in prose) are held
 * back briefly and flushed on mismatch.
 */

#pragma once

#include <string>

namespace forge {

class ReasoningFilter {
public:
    /// Feed one decoded delta. Splits it into client-visible content and
    /// reasoning bytes. Either out-param may be null.
    void feed(const std::string& delta, std::string* visible, std::string* reasoning);

    /// End of stream: classify held-back bytes. An unclosed `<think>` block
    /// yields everything after the open tag as reasoning.
    void flush(std::string* visible, std::string* reasoning);

    /// One-shot pass over a complete text (non-streaming / final authority).
    /// Every balanced `<think>...</think>` block is moved to `reasoning`; an
    /// unclosed `<think>` moves the whole tail. Text outside blocks is kept
    /// verbatim in `visible`.
    static void strip(const std::string& text, std::string* visible, std::string* reasoning);

private:
    enum class State { Passthrough, InThink };
    void feed_byte(char c, std::string* visible, std::string* reasoning);
    // Flush `hold_` to the current state's output after removing the longest
    // suffix that is still a prefix of the state's candidate tag.
    void flush_hold(std::string* visible, std::string* reasoning);

    State state_ = State::Passthrough;
    std::string hold_;  // bytes that may belong to a tag; resolved ASAP
};

}  // namespace forge
