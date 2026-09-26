/**
 * AgentPipeline implementation — per-request orchestration facade.
 *
 * The pipeline owns no shared mutable state: ServerState constructs one per
 * request (a dialect pointer + a streaming filter), so concurrent requests
 * never contend beyond ServerState's own inference mutex.
 */

#include "forge/agent/agent_pipeline.h"

#include <cstdlib>
#include <utility>

namespace forge {

namespace {

// Auto mode renders the catalogue with the server's own protocol: existing
// deployments and models already conditioned on it keep their prompt shape,
// while parsing still auto-detects every registered dialect.
constexpr const char* kDefaultPromptDialect = "forge_text";

}  // namespace

AgentPipeline::AgentPipeline(AgentPipelineConfig cfg) : cfg_(std::move(cfg)) {
    ensure_default_tool_dialects();
    prompt_dialect_ = find_tool_dialect(
        cfg_.dialect == "auto" ? kDefaultPromptDialect : cfg_.dialect);
    if (!prompt_dialect_)  // unknown FORGE_TOOL_DIALECT value: fall back
        prompt_dialect_ = find_tool_dialect(kDefaultPromptDialect);
}

std::vector<ChatTemplateMessage> AgentPipeline::inject_tools(
    const std::vector<ChatTemplateMessage>& msgs, const std::vector<ToolSpec>& tools,
    const std::string& tool_choice, bool* active) {
    if (active)
        *active = !tools.empty();
    if (tools.empty())
        return msgs;
    // Native-template mode: the jinja template renders the catalogue from the
    // `tools` context variable; a dialect block here would teach a second,
    // conflicting call format.
    if (cfg_.native_template_tools)
        return msgs;
    std::vector<ChatTemplateMessage> out = msgs;
    const std::string block = prompt_dialect_ ? prompt_dialect_->render_catalogue(tools, tool_choice)
                                              : "";
    if (!out.empty() && out[0].role == "system") {
        out[0].content += "\n\n" + block;
    } else {
        ChatTemplateMessage sys;
        sys.role = "system";
        sys.content = block;
        out.insert(out.begin(), std::move(sys));
    }
    return out;
}

std::string AgentPipeline::process_delta(const std::string& delta, std::string* reasoning_out) {
    std::string visible;
    std::string reasoning;
    filter_.feed(delta, &visible, &reasoning);
    if (reasoning_out)
        *reasoning_out = std::move(reasoning);
    return visible;
}

AgentPipeline::Final AgentPipeline::finalize(const std::string& raw_text,
                                             const std::vector<std::string>* allowed) {
    Final f;
    ReasoningFilter::strip(raw_text, &f.text, &f.reasoning);
    // A stripped leading think block usually leaves blank lines behind.
    if (raw_text.find("<think>") != std::string::npos) {
        const size_t a = f.text.find_first_not_of(" \t\r\n");
        f.text = (a == std::string::npos) ? "" : f.text.substr(a);
    }
    if (allowed) {
        if (cfg_.dialect == "auto") {
            const ToolCallDialect* used = nullptr;
            f.tool_calls = parse_tool_calls_auto(f.text, allowed, &used);
            if (!f.tool_calls.empty() && used)
                f.text = used->strip(f.text);
        } else if (prompt_dialect_) {
            f.tool_calls = prompt_dialect_->parse(f.text, allowed);
            if (!f.tool_calls.empty())
                f.text = prompt_dialect_->strip(f.text);
        }
        // Always purge recognized tool-call syntax from the visible text. This
        // runs regardless of whether a call resolved: a tool attempt whose name
        // can't be matched against the catalogue (or any stray marker) must
        // never leak its raw syntax into the message. Resolved calls already
        // live in f.tool_calls and are unaffected by the strip.
        f.text = strip_all_tool_syntax(f.text);
    }
    return f;
}

AgentPipelineConfig agent_pipeline_config_from_env() {
    AgentPipelineConfig cfg;
    if (const char* e = std::getenv("FORGE_TOOL_DIALECT")) {
        if (*e)
            cfg.dialect = e;
    }
    return cfg;
}

}  // namespace forge
