/**
 * ChatTemplateEngine — Jinja-powered chat template rendering
 *
 * Reads the GGUF `tokenizer.chat_template` metadata, compiles it via the
 * vendored jinja engine, and renders conversation messages into a prompt
 * string (or directly into token IDs).
 *
 * Fallback: if the model lacks a GGUF chat_template (or jinja compilation
 * fails), the engine falls back to the legacy hardcoded templates in
 * cli_chat_template.cpp.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace forge {

class Tokenizer;

// ---- Chat message ----

struct ChatTemplateMessage {
    std::string role;     // "system", "user", "assistant", "tool"
    std::string content;  // text content
};

// ---- Input to ChatTemplateEngine ----

struct ChatTemplateInput {
    std::vector<ChatTemplateMessage> messages;
    bool add_generation_prompt = true;
    std::string system_prompt;  // if non-empty, prepended as a system message
};

// ---- ChatTemplateEngine ----

class ChatTemplateEngine {
public:
    ChatTemplateEngine();
    ~ChatTemplateEngine();
    ChatTemplateEngine(ChatTemplateEngine&&) noexcept;
    ChatTemplateEngine& operator=(ChatTemplateEngine&&) noexcept;

    /// Build the engine from a tokenizer.
    /// Reads `tokenizer.chat_template()` from GGUF metadata, compiles it
    /// via the jinja engine, and sets up the fallback path.
    static ChatTemplateEngine from_tokenizer(const Tokenizer& tok);

    /// Whether the engine is using the jinja path (vs. fallback).
    bool uses_jinja() const;

    /// Whether the template supports a system role.
    bool supports_system_role() const;

    /// Render messages into a prompt string.
    /// @throws std::runtime_error on rendering failure.
    std::string render(const ChatTemplateInput& input) const;

    /// Render + encode: render the prompt string, then tokenize it with
    /// parse_special=true so that special tokens inside the rendered string
    /// (e.g. <|im_start|>, [INST]) are recognized as single tokens.
    std::vector<int32_t> apply(const ChatTemplateInput& input) const;

private:
    const Tokenizer* tok_ = nullptr;

    // Jinja path
    std::string tmpl_source_;
    struct JinjaProgram;  // opaque, defined in .cpp
    std::unique_ptr<JinjaProgram> jinja_prog_;

    // Capability flags
    bool jinja_supports_system_ = false;
    bool jinja_supports_tools_ = false;

    // Fallback path
    enum class FallbackType { ChatML, DeepSeek, Llama, Plain };
    FallbackType fallback_type_ = FallbackType::Plain;

    std::string render_jinja(const ChatTemplateInput& input) const;
    std::string render_fallback(const ChatTemplateInput& input) const;
};

}  // namespace forge