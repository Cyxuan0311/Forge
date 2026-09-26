/**
 * Agent module — pluggable tool-call dialects.
 *
 * A "dialect" is one model-side tool-calling convention: how the tool
 * catalogue is taught in the system prompt and how the model is expected to
 * spell out a call. The server no longer hardcodes its own `###TOOL###`
 * protocol; models trained on a native format (XML function blocks,
 * `<tool_call>` JSON, ...) are handled by registering the matching dialect
 * and letting the auto-detector pick whichever actually parses.
 *
 * Adding support for a new model/agent format = implement ToolCallDialect +
 * register it in ensure_default_tool_dialects() (or via register_tool_dialect
 * from a test). No server-body edits needed.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "forge/agent/tool_call.h"

namespace forge {

class ToolCallDialect {
public:
    virtual ~ToolCallDialect() = default;

    /// Registry key (also the FORGE_TOOL_DIALECT value), e.g. "forge_text".
    virtual const char* name() const = 0;

    /// Render the tool catalogue into a system-prompt block that teaches the
    /// model this dialect's call format.
    virtual std::string render_catalogue(const std::vector<ToolSpec>& tools,
                                         const std::string& tool_choice) const = 0;

    /// Extract tool calls from generated text. When `allowed` is non-null and
    /// non-empty, emitted names are normalized/fuzzy-matched against it and
    /// calls that cannot be resolved are dropped.
    virtual std::vector<ToolCall> parse(const std::string& text,
                                        const std::vector<std::string>* allowed) const = 0;

    /// Remove this dialect's call blocks from the text (result is trimmed).
    virtual std::string strip(const std::string& text) const = 0;
};

// ---- Registry ----

/// Register a dialect (takes ownership). Call before find/parse use.
void register_tool_dialect(std::unique_ptr<ToolCallDialect> dialect);

/// Idempotently register the built-in dialects:
///   xml_function, qwen_json, forge_text.
void ensure_default_tool_dialects();

/// Look a dialect up by name; nullptr when unknown.
const ToolCallDialect* find_tool_dialect(const std::string& name);

/// Auto-detection: try the built-in dialects in priority order
/// (xml_function -> qwen_json -> forge_text) and return the first non-empty
/// parse. `used`, when non-null, receives the dialect that matched (nullptr
/// when nothing parsed).
std::vector<ToolCall> parse_tool_calls_auto(const std::string& text,
                                            const std::vector<std::string>* allowed,
                                            const ToolCallDialect** used = nullptr);

/// Purge every recognized tool-call syntax from `text` by applying each
/// registered dialect's strip in turn. Unlike parse+strip, this runs regardless
/// of whether a call was successfully resolved, so a tool attempt whose name
/// cannot be matched (or any stray marker) can never leak its raw syntax into
/// the visible message. No-op for text that contains no tool-call markers.
std::string strip_all_tool_syntax(const std::string& text);

}  // namespace forge
