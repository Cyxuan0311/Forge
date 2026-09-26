/**
 * Agent module — per-request orchestration facade.
 *
 * Wires the three concerns the server used to inline so that ServerState
 * only orchestrates:
 *   1. inject_tools()  — render the tool catalogue with the configured
 *                        dialect and splice it into the system message;
 *   2. process_delta() — streaming think/reasoning separation;
 *   3. finalize()      — one-shot reasoning strip + tool-call extraction
 *                        over the complete generated text.
 *
 * Cheap to construct (a dialect pointer + a filter): ServerState builds one
 * per request instead of sharing mutable state.
 */

#pragma once

#include <string>
#include <vector>

#include "forge/agent/reasoning_filter.h"
#include "forge/agent/tool_call.h"
#include "forge/agent/tool_dialect.h"
#include "forge/chat_template.h"

namespace forge {

struct AgentPipelineConfig {
    // "auto" | "forge_text" | "xml_function" | "qwen_json" (registry names).
    // auto: catalogue is rendered with forge_text (backward compatible) and
    //       parsing auto-detects across all registered dialects;
    // explicit name: that dialect renders AND parses.
    std::string dialect = "auto";
    // True when the model's jinja chat template renders the tool catalogue
    // itself (templates with a tools branch): inject_tools then passes the
    // messages through untouched and the server hands the tools to the
    // template. Parsing of the model's output stays dialect-agnostic.
    bool native_template_tools = false;
};

class AgentPipeline {
public:
    explicit AgentPipeline(AgentPipelineConfig cfg = {});

    /// Splice the dialect-rendered tool catalogue into the system message.
    /// Empty `tools` passes messages through unchanged and clears `*active`.
    std::vector<ChatTemplateMessage> inject_tools(
        const std::vector<ChatTemplateMessage>& msgs, const std::vector<ToolSpec>& tools,
        const std::string& tool_choice, bool* active);

    /// Streaming path: split one decoded delta. Returns the client-visible
    /// part; the reasoning part is written to `reasoning_out` (may be null).
    std::string process_delta(const std::string& delta, std::string* reasoning_out);

    /// Final pass over the complete generated text. `allowed` (the registered
    /// tool names) also gates tool-call extraction: pass nullptr to skip
    /// parsing entirely (no-tools request) while still separating reasoning.
    struct Final {
        std::string text;
        std::string reasoning;
        std::vector<ToolCall> tool_calls;
    };
    Final finalize(const std::string& raw_text, const std::vector<std::string>* allowed);

    const AgentPipelineConfig& config() const { return cfg_; }

private:
    AgentPipelineConfig cfg_;
    const ToolCallDialect* prompt_dialect_ = nullptr;  // catalogue rendering
    ReasoningFilter filter_;                           // streaming state
};

/// Build a config from the environment: FORGE_TOOL_DIALECT selects the
/// dialect ("auto" when unset), mirroring the FORGE_MATMUL_PATH override
/// convention.
AgentPipelineConfig agent_pipeline_config_from_env();

}  // namespace forge
