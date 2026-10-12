/**
 * Agent module — tool executor registry.
 *
 * The dialect layer (agent_pipeline / tool_dialect) is deliberately protocol
 * only: the model emits tool calls, and whoever drives the conversation
 * executes them and feeds the results back. This registry supplies the server
 * side of that split: a name -> executor table, a JSON-schema-lite check so a
 * hallucinated argument is rejected before it reaches a subprocess or an
 * outbound HTTP call, and the catalogue the web UI queries to decide which
 * modes it may offer.
 *
 * Everything here is pure host logic (nlohmann/json + std), like the rest of
 * src/agent, so it links into tests without an engine.
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace forge {

/// Result of one tool invocation. `ok == false` is a normal outcome the model
/// is meant to read and react to (a failed search, a syntax error in the code
/// it wrote) -- it is reported back as a tool-role message, not as a 5xx.
struct ToolResult {
    bool ok = false;
    std::string content;  // plain text handed back to the model
    std::string error;    // short machine-ish reason, safe to log
};

/// Credentials a tool may need. Resolved server-side from the environment, and
/// optionally overridden per request so the browser can supply its own key
/// without it ever being baked into the binary.
struct ToolContext {
    // Key supplied by the client for this call. Empty means "use the env".
    std::string api_key;
    // Provider hint from the client ("tavily", "serper", ...). Empty means env.
    std::string provider;
    // Hard ceiling for wall-clock execution, seconds. 0 -> tool default.
    int timeout_sec = 0;
    // Byte ceiling for the result handed back to the model.
    size_t max_output_bytes = 0;
};

/// One executable tool.
struct ToolDefinition {
    std::string name;
    std::string description;
    std::string parameters_json;  // JSON schema, same shape the model sees
    /// Whether the tool can run right now (e.g. a search key is configured).
    bool available = true;
    /// Why `available` is false, for the UI to show a hint instead of failing.
    std::string unavailable_reason;
    ToolContext ctx;  // per-call overrides, threaded into exec

    using ExecFn = std::function<ToolResult(const nlohmann::json& args, const ToolContext&)>;
    ExecFn exec;
};

/// Look up a tool by exact name. Returns nullptr when unknown.
ToolDefinition* find_tool(const std::string& name);

/// Every registered tool, in registration order.
const std::vector<std::unique_ptr<ToolDefinition>>& all_tools();

/// Catalogue for GET /v1/tools: the tool list plus per-tool availability.
nlohmann::json tool_catalogue();

/// Execute `name` with a JSON arguments object.
///
/// Returns a failed ToolResult (never throws) for: unknown tool, arguments
/// that fail the schema check, or a throwing executor. `result.error` carries
/// a short reason suitable for the model to see.
ToolResult execute_tool(const std::string& name, const nlohmann::json& args,
                         const ToolContext& ctx);

/// Schema check for `args` against `schema_json`. Returns false and fills
/// `err` on the first problem. Deliberately narrow: it understands the subset
/// a tool catalogue actually uses (type/properties/required/enum/items) rather
/// than being a general JSON Schema validator.
bool validate_arguments(const std::string& schema_json, const nlohmann::json& args,
                         std::string* err);

/// Clamp `text` to `max_bytes` on a UTF-8 boundary, appending a truncation
/// marker. Tool output goes straight into the prompt, so it must be bounded.
std::string clamp_output(const std::string& text, size_t max_bytes);

// ---- registration (implemented by the individual tool translation units) ----

/// Insert into the registry. Ignores a duplicate name (first registration
/// wins). Used by the tool TUs; not part of the stable surface.
namespace tool_registry_detail {
void add_tool(std::unique_ptr<ToolDefinition> def);
}  // namespace tool_registry_detail

/// Whether python execution is enabled. Resolved once from the environment
/// (FORGE_ALLOW_CODE_EXEC) unless the server overrides it explicitly via
/// set_code_exec_enabled() (the --enable-code-exec flag).
bool code_exec_enabled();
void set_code_exec_enabled(bool enabled);

/// web_search provider in use, e.g. "tavily", or "" when none is configured.
std::string search_provider();
std::string search_api_key();

void register_python_tool();
void register_web_search_tool();

/// Register everything. Idempotent.
void ensure_default_tools();

}  // namespace forge