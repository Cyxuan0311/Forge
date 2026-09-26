// =========================================================================
// test_tool_dialects.cpp — forge_agent module: tool-call dialects,
// ReasoningFilter and AgentPipeline.
//
// Pure host logic (all headers are header-only; nlohmann json is vendored):
// no GPU, no model files. Compiled directly against src/agent/*.cpp so the
// executable stays fast to build (same pattern as test_dtype_contract.cpp).
//
// Covers:
//   1. forge_text  : single/multiple calls, dropped-'#' tolerance, unquoted
//                    keys, invalid-escape repair, fuzzy name resolution,
//                    unknown-tool drop, strip.
//   2. xml_function: the MiniCPM5 shape observed in the wild (CDATA params),
//                    no-CDATA params, JSON-typed values, multi-param /
//                    multi-call, strip.
//   3. qwen_json   : <tool_call> blocks, whole-message bare JSON / array,
//                    strip (wrapper + bare).
//   4. Auto-detect : dispatch priority (xml -> qwen -> forge_text) and the
//                    no-match case.
//   5. ReasoningFilter: closed/unclosed blocks, tags split across streaming
//                    chunks, literal '<' prose, "<<" reopen edge.
//   6. AgentPipeline  : inject_tools splice, streaming process_delta,
//                    finalize (reasoning + parse + strip), env override.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "forge/agent/agent_pipeline.h"
#include "forge/agent/reasoning_filter.h"
#include "forge/agent/tool_dialect.h"

namespace {

using json = nlohmann::json;
using forge::AgentPipeline;
using forge::AgentPipelineConfig;
using forge::ReasoningFilter;
using forge::ToolCall;
using forge::ToolCallDialect;
using forge::ToolSpec;

int failures = 0;

// Variadic so expressions containing commas (json{{"a", 1}}) stay one arg.
#define CHECK(...)                                                        \
    do {                                                                  \
        if (!(__VA_ARGS__)) {                                             \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                        #__VA_ARGS__);                                    \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

// ---------------------------------------------------------------------------
// 1. forge_text — the server's own ###TOOL### protocol
// ---------------------------------------------------------------------------

void test_forge_text() {
    const ToolCallDialect* d = forge::find_tool_dialect("forge_text");
    CHECK(d != nullptr);
    if (!d)
        return;

    std::string stripped;
    // Single call, well-formed JSON.
    auto calls = d->parse(
        "###TOOL###{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}###END###", nullptr);
    CHECK(calls.size() == 1);
    if (calls.size() == 1) {
        CHECK(calls[0].name == "bash");
        CHECK(calls[0].type == "function");
        CHECK(calls[0].id.rfind("call-", 0) == 0);
        CHECK(json::parse(calls[0].arguments) == json{{"command", "ls"}});
    }

    // Quantized models drop a '#': "###TOOL##" still opens a block.
    calls = d->parse("###TOOL##{\"name\":\"a\",\"arguments\":{}}###END###", nullptr);
    CHECK(calls.size() == 1);
    if (calls.size() == 1)
        CHECK(calls[0].name == "a");

    // The MiniCPM5 shape observed live: the name is written directly before
    // the object (no "name" key) and the close marker lost a '#'. The
    // tolerant fallback must yield a resolvable call with VALID JSON args.
    const std::vector<std::string> bash_only = {"bash"};
    calls = d->parse("###TOOL###bash{command:\"ls -la\"}###END##", &bash_only);
    CHECK(calls.size() == 1);
    if (calls.size() == 1) {
        CHECK(calls[0].name == "bash");
        CHECK(json::parse(calls[0].arguments) == json{{"command", "ls -la"}});
    }

    // Extra '#' on the close marker also closes the block.
    calls = d->parse("###TOOL###{\"name\":\"a\",\"arguments\":{}}###END####", nullptr);
    CHECK(calls.size() == 1);

    // strip removes the name-prefixed shape as well.
    stripped = d->strip("Go. ###TOOL###bash{command:\"ls -la\"}###END##");
    CHECK(stripped == "Go.");

    // Unquoted keys recover via the tolerant path.
    calls = d->parse("###TOOL###{name: bash, arguments: {command: ls}}###END###", nullptr);
    CHECK(calls.size() == 1);
    if (calls.size() == 1) {
        CHECK(calls[0].name == "bash");
        CHECK(calls[0].arguments == "{command: ls}");
    }

    // Invalid escapes (e.g. \') are repaired so the payload still parses.
    calls = d->parse(
        "###TOOL###{\"name\": \"bash\", \"arguments\": {\"command\": \"it\\'s fine\"}}###END###",
        nullptr);
    CHECK(calls.size() == 1);
    if (calls.size() == 1) {
        json args = json::parse(calls[0].arguments);  // must not throw
        CHECK(args["command"].get<std::string>() == "it\\'s fine");
    }

    // Multiple calls.
    calls = d->parse("###TOOL###{\"name\":\"a\",\"arguments\":{}}###END###mid"
                     "###TOOL###{\"name\":\"b\",\"arguments\":{}}###END###",
                     nullptr);
    CHECK(calls.size() == 2);
    if (calls.size() == 2) {
        CHECK(calls[0].name == "a");
        CHECK(calls[1].name == "b");
    }

    // Fuzzy name resolution against the registered catalogue.
    const std::vector<std::string> allowed = {"get_weather", "bash"};
    calls = d->parse(
        "###TOOL###{\"name\":\"getweather\",\"arguments\":{\"city\":\"Paris\"}}###END###",
        &allowed);
    CHECK(calls.size() == 1);
    if (calls.size() == 1)
        CHECK(calls[0].name == "get_weather");

    // Unknown tool: dropped when a catalogue is given.
    calls = d->parse("###TOOL###{\"name\":\"nope\",\"arguments\":{}}###END###", &allowed);
    CHECK(calls.empty());

    // strip removes the block and trims; unclosed block erases to the end.
    stripped = d->strip("Sure. ###TOOL###{\"name\":\"a\",\"arguments\":{}}###END###");
    CHECK(stripped == "Sure.");
    stripped = d->strip("Prefix ###TOOL###{\"name\":\"a\"");
    CHECK(stripped == "Prefix");
}

// ---------------------------------------------------------------------------
// 2. xml_function — native XML function blocks (MiniCPM5 in the wild)
// ---------------------------------------------------------------------------

void test_xml_function() {
    const ToolCallDialect* d = forge::find_tool_dialect("xml_function");
    CHECK(d != nullptr);
    if (!d)
        return;

    // The exact output observed from MiniCPM5-2B via opencode.
    const std::string sample =
        "<function name=\"bash\"><param name=\"command\">"
        "<![CDATA[ls -la && git status --short]]></param></function>";
    std::vector<std::string> allowed = {"bash"};
    auto calls = d->parse(sample, &allowed);
    CHECK(calls.size() == 1);
    if (calls.size() == 1) {
        CHECK(calls[0].name == "bash");
        CHECK(json::parse(calls[0].arguments) ==
              json{{"command", "ls -la && git status --short"}});
    }

    // Single-quoted attributes and no CDATA wrapper.
    calls = d->parse("<function name='echo'><param name=\"text\">hello world</param></function>",
                     nullptr);
    CHECK(calls.size() == 1);
    if (calls.size() == 1)
        CHECK(json::parse(calls[0].arguments) == json{{"text", "hello world"}});

    // JSON-typed parameter values keep their type.
    calls = d->parse("<function name=\"f\"><param name=\"n\">42</param>"
                     "<param name=\"arr\">[1,2]</param></function>",
                     nullptr);
    CHECK(calls.size() == 1);
    if (calls.size() == 1) {
        json args = json::parse(calls[0].arguments);
        CHECK(args["n"] == 42);
        CHECK(args["arr"] == json{1, 2});
    }

    // Multiple params and calls, with prose around them.
    const std::string two =
        "Running now.\n<function name=\"bash\"><param name=\"command\">pwd</param>"
        "<param name=\"timeout\">30</param></function>\n"
        "<function name=\"echo\"><param name=\"text\">done</param></function>\nAll set.";
    const std::vector<std::string> allowed2 = {"bash", "echo"};
    calls = d->parse(two, &allowed2);
    CHECK(calls.size() == 2);
    if (calls.size() == 2) {
        CHECK(calls[0].name == "bash");
        CHECK(calls[1].name == "echo");
        CHECK(json::parse(calls[0].arguments)["timeout"] == 30);
    }

    // Unknown tool dropped; without a catalogue the emitted name is kept.
    calls = d->parse(sample, nullptr);
    CHECK(calls.size() == 1);
    if (calls.size() == 1)
        CHECK(calls[0].name == "bash");
    const std::vector<std::string> other = {"other"};
    calls = d->parse(sample, &other);
    CHECK(calls.empty());

    // strip keeps the prose and removes every block.
    const std::string stripped = d->strip(two);
    CHECK(stripped.find("Running now.") != std::string::npos);
    CHECK(stripped.find("All set.") != std::string::npos);
    CHECK(stripped.find("<function") == std::string::npos);
    CHECK(stripped.find("<param") == std::string::npos);
}

// ---------------------------------------------------------------------------
// 3. qwen_json — <tool_call>{json}</tool_call> + bare-JSON fallback
// ---------------------------------------------------------------------------

void test_qwen_json() {
    const ToolCallDialect* d = forge::find_tool_dialect("qwen_json");
    CHECK(d != nullptr);
    if (!d)
        return;

    auto calls = d->parse(
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>",
        nullptr);
    CHECK(calls.size() == 1);
    if (calls.size() == 1) {
        CHECK(calls[0].name == "get_weather");
        CHECK(json::parse(calls[0].arguments) == json{{"city", "Paris"}});
    }

    // Whole-message bare object (no wrapper).
    calls = d->parse("{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}", nullptr);
    CHECK(calls.size() == 1);
    if (calls.size() == 1)
        CHECK(calls[0].name == "get_weather");

    // Bare array of calls; missing "arguments" becomes {}.
    const std::vector<std::string> allowed = {"a", "b"};
    calls = d->parse("[{\"name\": \"a\", \"arguments\": {}}, {\"name\": \"b\"}]", &allowed);
    CHECK(calls.size() == 2);
    if (calls.size() == 2) {
        CHECK(calls[0].name == "a");
        CHECK(calls[1].name == "b");
        CHECK(calls[1].arguments == "{}");
    }

    // strip: wrapper blocks removed, prose kept (inner spacing preserved).
    std::string stripped = d->strip("text <tool_call>{\"name\":\"a\"}</tool_call> tail");
    CHECK(stripped == "text  tail");
    // Whole-message bare JSON strips to nothing.
    stripped = d->strip("{\"name\": \"a\"}");
    CHECK(stripped.empty());
    // Non-JSON prose starting with '{' is left alone.
    stripped = d->strip("{not json}");
    CHECK(stripped == "{not json}");
}

// ---------------------------------------------------------------------------
// 4. Auto-detection dispatch
// ---------------------------------------------------------------------------

void test_auto_detect() {
    forge::ensure_default_tool_dialects();
    const std::vector<std::string> allowed = {"bash", "get_weather"};

    const ToolCallDialect* used = nullptr;
    auto calls = forge::parse_tool_calls_auto(
        "<function name=\"bash\"><param name=\"command\">ls</param></function>", &allowed, &used);
    CHECK(calls.size() == 1);
    CHECK(used != nullptr && std::string(used->name()) == "xml_function");

    calls = forge::parse_tool_calls_auto(
        "<tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>", &allowed, &used);
    CHECK(calls.size() == 1);
    CHECK(used != nullptr && std::string(used->name()) == "qwen_json");

    calls = forge::parse_tool_calls_auto(
        "###TOOL###{\"name\":\"bash\",\"arguments\":{}}###END###", &allowed, &used);
    CHECK(calls.size() == 1);
    CHECK(used != nullptr && std::string(used->name()) == "forge_text");

    // Plain prose matches nothing.
    calls = forge::parse_tool_calls_auto("Just answering in prose.", &allowed, &used);
    CHECK(calls.empty());
    CHECK(used == nullptr);
}

// ---------------------------------------------------------------------------
// 5. ReasoningFilter — <think> separation (one-shot + streaming)
// ---------------------------------------------------------------------------

void test_reasoning_filter() {
    // One-shot strip: balanced block.
    std::string vis, rea;
    ReasoningFilter::strip("Hi <think>deep thought</think>there", &vis, &rea);
    CHECK(vis == "Hi there");
    CHECK(rea == "deep thought");

    // Unclosed block: the whole tail is reasoning.
    ReasoningFilter::strip("Hi <think>abc", &vis, &rea);
    CHECK(vis == "Hi ");
    CHECK(rea == "abc");

    // No think block at all.
    ReasoningFilter::strip("plain text", &vis, &rea);
    CHECK(vis == "plain text");
    CHECK(rea.empty());

    // Streaming: the tag is split across three chunks.
    ReasoningFilter f;
    std::string sv, sr;
    f.feed("<th", &sv, &sr);
    f.feed("ink>re", &sv, &sr);
    f.feed("ason</th", &sv, &sr);
    f.feed("ink> done", &sv, &sr);
    f.flush(&sv, &sr);
    CHECK(sv == " done");
    CHECK(sr == "reason");

    // Streaming: unclosed think at end of stream.
    f = ReasoningFilter();
    sv.clear();
    sr.clear();
    f.feed("answer<think>hidden tail", &sv, &sr);
    f.flush(&sv, &sr);
    CHECK(sv == "answer");
    CHECK(sr == "hidden tail");

    // Literal '<' in prose passes through.
    f = ReasoningFilter();
    sv.clear();
    sr.clear();
    f.feed("a < b > c", &sv, &sr);
    f.flush(&sv, &sr);
    CHECK(sv == "a < b > c");
    CHECK(sr.empty());

    // "<<" — the first '<' cannot open the tag and is flushed; the second
    // completes it, so the visible output keeps one '<'.
    f = ReasoningFilter();
    sv.clear();
    sr.clear();
    f.feed("<<think>secret</think>", &sv, &sr);
    f.flush(&sv, &sr);
    CHECK(sv == "<");
    CHECK(sr == "secret");
}

// ---------------------------------------------------------------------------
// 6. AgentPipeline — inject / stream / finalize orchestration
// ---------------------------------------------------------------------------

void test_agent_pipeline() {
    // inject_tools: appends the catalogue to an existing system message.
    AgentPipeline p;
    const std::vector<forge::ChatTemplateMessage> msgs = {{"system", "Be terse."},
                                                          {"user", "list files"}};
    const std::vector<ToolSpec> tools = {{"bash", "run a shell command", "{}"}};
    bool active = false;
    auto out = p.inject_tools(msgs, tools, "auto", &active);
    CHECK(active);
    CHECK(out.size() == 2);
    if (out.size() == 2) {
        CHECK(out[0].role == "system");
        CHECK(out[0].content.rfind("Be terse.", 0) == 0);
        // auto renders the catalogue with forge_text (backward compatible).
        CHECK(out[0].content.find("###TOOL###") != std::string::npos);
        CHECK(out[0].content.find("bash") != std::string::npos);
        CHECK(out[1].content == "list files");
    }

    // No system message: one is inserted.
    const std::vector<forge::ChatTemplateMessage> no_sys = {{"user", "hi"}};
    out = p.inject_tools(no_sys, tools, "auto", &active);
    CHECK(active);
    CHECK(out.size() == 2);
    if (out.size() == 2)
        CHECK(out[0].role == "system");

    // No tools: passthrough, inactive.
    out = p.inject_tools(no_sys, {}, "auto", &active);
    CHECK(!active);
    CHECK(out.size() == 1);
    if (out.size() == 1)
        CHECK(out[0].content == "hi");

    // finalize: think separation + XML extraction — the exact MiniCPM5 shape.
    const std::vector<std::string> allowed = {"bash"};
    AgentPipeline xml_p{{"xml_function"}};
    const std::string raw =
        "<think>The user wants files.</think><function name=\"bash\">"
        "<param name=\"command\">ls -la</param></function>";

    // No tools offered (allowed == nullptr): reasoning separated, no parsing.
    auto fin = xml_p.finalize(raw, nullptr);
    CHECK(fin.tool_calls.empty());
    CHECK(fin.text == "<function name=\"bash\"><param name=\"command\">ls -la</param></function>");
    CHECK(fin.reasoning == "The user wants files.");

    // With the catalogue: the call is extracted and stripped from the text.
    fin = xml_p.finalize(raw, &allowed);
    CHECK(fin.tool_calls.size() == 1);
    if (fin.tool_calls.size() == 1) {
        CHECK(fin.tool_calls[0].name == "bash");
        CHECK(json::parse(fin.tool_calls[0].arguments) == json{{"command", "ls -la"}});
    }
    CHECK(fin.text.empty());
    CHECK(fin.reasoning == "The user wants files.");

    // Auto mode: forge_text protocol parsed out of the think-stripped text.
    AgentPipeline auto_p;
    fin = auto_p.finalize(
        "<think>reason</think>###TOOL###{\"name\":\"bash\",\"arguments\":{}}###END###", &allowed);
    CHECK(fin.tool_calls.size() == 1);
    if (fin.tool_calls.size() == 1)
        CHECK(fin.tool_calls[0].name == "bash");
    CHECK(fin.text.empty());
    CHECK(fin.reasoning == "reason");

    // Prose-only answer under tools: no calls, text kept.
    fin = auto_p.finalize("<think>hmm</think>Here is the answer.", &allowed);
    CHECK(fin.tool_calls.empty());
    CHECK(fin.text == "Here is the answer.");
    CHECK(fin.reasoning == "hmm");

    // Streaming process_delta: per-delta reasoning chunks (overwritten each
    // call, matching the SSE emit-per-delta contract).
    AgentPipeline sp;
    std::string reasoning;
    std::string visible = sp.process_delta("<think>a", &reasoning);
    CHECK(visible.empty());
    CHECK(reasoning == "a");
    visible = sp.process_delta("b</think>c", &reasoning);
    CHECK(visible == "c");
    CHECK(reasoning == "b");
    visible = sp.process_delta("!", &reasoning);
    CHECK(visible == "!");
    CHECK(reasoning.empty());

    // Env override: FORGE_TOOL_DIALECT selects the dialect, unset == auto.
    unsetenv("FORGE_TOOL_DIALECT");
    AgentPipelineConfig cfg = forge::agent_pipeline_config_from_env();
    CHECK(cfg.dialect == "auto");
    setenv("FORGE_TOOL_DIALECT", "xml_function", 1);
    cfg = forge::agent_pipeline_config_from_env();
    CHECK(cfg.dialect == "xml_function");
    unsetenv("FORGE_TOOL_DIALECT");
}

void test_unresolved_name_no_leak() {
    // Regression: a model that emits a tool-call whose name cannot be resolved
    // against the catalogue must NOT leak the raw syntax into the message. The
    // strip runs regardless of resolution (this is exactly the opencode case:
    // it drives the model to emit native <function> XML whose name differs from
    // the registered tool set).
    const std::vector<std::string> allowed = {"bash"};
    AgentPipeline auto_p;  // auto -> xml_function first

    const std::string raw =
        "Sure, running that.\n"
        "<function name=\"totally_unknown_tool\">"
        "<param name=\"command\">rm -rf /</param></function>\n"
        "Done.";

    auto fin = auto_p.finalize(raw, &allowed);
    // No resolvable call -> tool_calls empty.
    CHECK(fin.tool_calls.empty());
    // But the raw XML must be purged from the visible text.
    CHECK(fin.text.find("<function") == std::string::npos);
    CHECK(fin.text.find("<param") == std::string::npos);
    // Prose around it survives.
    CHECK(fin.text.find("Sure, running that.") != std::string::npos);
    CHECK(fin.text.find("Done.") != std::string::npos);

    // Continue past an unresolvable block and still extract the valid call
    // that follows it (the parse loop must not break on the first bad name).
    const std::string two =
        "<function name=\"unknown\"><param name=\"x\">1</param></function>"
        "<function name=\"bash\"><param name=\"command\">ls</param></function>";
    fin = auto_p.finalize(two, &allowed);
    CHECK(fin.tool_calls.size() == 1);
    if (fin.tool_calls.size() == 1)
        CHECK(fin.tool_calls[0].name == "bash");
    CHECK(fin.text.find("unknown") == std::string::npos);
}

}  // namespace

int main() {
    test_forge_text();
    test_xml_function();
    test_qwen_json();
    test_auto_detect();
    test_reasoning_filter();
    test_agent_pipeline();
    test_unresolved_name_no_leak();

    if (failures == 0) {
        std::printf("tool-dialects: all checks passed\n");
        return 0;
    }
    std::printf("tool-dialects: %d check(s) FAILED\n", failures);
    return 1;
}
