/**
 * Agent module — tool executor registry (see include/forge/agent/tool_registry.h).
 */

#include "forge/agent/tool_registry.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>

namespace forge {

using json = nlohmann::json;

namespace {

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

std::vector<std::unique_ptr<ToolDefinition>>& registry() {
    static std::vector<std::unique_ptr<ToolDefinition>> tools;
    return tools;
}

// ---- schema check -----------------------------------------------------------

// Type name for diagnostics. JSON Schema allows an array of names; we report
// the first so the message stays short.
std::string json_type_name(const json& v) {
    if (v.is_null()) return "null";
    if (v.is_boolean()) return "boolean";
    if (v.is_number_integer() || v.is_number_float()) return "number";
    if (v.is_string()) return "string";
    if (v.is_array()) return "array";
    if (v.is_object()) return "object";
    return "unknown";
}

bool type_matches(const std::string& expected, const json& v) {
    if (expected == "string") return v.is_string();
    if (expected == "integer") return v.is_number_integer();
    // JSON Schema "number" accepts ints too.
    if (expected == "number") return v.is_number_integer() || v.is_number_float();
    if (expected == "boolean") return v.is_boolean();
    if (expected == "array") return v.is_array();
    if (expected == "object") return v.is_object();
    if (expected == "null") return v.is_null();
    // Unknown constraint word: accept rather than reject a valid call.
    return true;
}

// Recursive check for one schema node against one value.
bool check_node(const json& schema, const json& value, const std::string& path, std::string* err) {
    if (!schema.is_object())
        return true;

    if (schema.contains("type")) {
        const json& t = schema["type"];
        bool ok = false;
        if (t.is_string()) {
            ok = type_matches(t.get<std::string>(), value);
        } else if (t.is_array()) {
            for (const auto& alt : t) {
                if (alt.is_string() && type_matches(alt.get<std::string>(), value)) {
                    ok = true;
                    break;
                }
            }
        }
        if (!ok) {
            if (err) {
                *err = path + ": expected " + t.dump() + ", got " + json_type_name(value);
            }
            return false;
        }
    }

    if (schema.contains("enum") && schema["enum"].is_array()) {
        bool found = false;
        for (const auto& allowed : schema["enum"]) {
            if (allowed == value) {
                found = true;
                break;
            }
        }
        if (!found) {
            if (err) *err = path + ": value not in enum " + schema["enum"].dump();
            return false;
        }
    }

    if (schema.contains("properties") && schema["properties"].is_object() && value.is_object()) {
        for (auto it = schema["properties"].begin(); it != schema["properties"].end(); ++it) {
            const auto found = value.find(it.key());
            if (found == value.end())
                continue;
            if (!check_node(it.value(), *found, path.empty() ? it.key() : path + "." + it.key(),
                            err))
                return false;
        }
    }

    if (schema.contains("required") && schema["required"].is_array() && value.is_object()) {
        for (const auto& req : schema["required"]) {
            if (!req.is_string())
                continue;
            if (value.find(req.get<std::string>()) == value.end()) {
                if (err) *err = "missing required argument '" + req.get<std::string>() + "'";
                return false;
            }
        }
    }

    if (schema.contains("items") && value.is_array()) {
        for (const auto& item : value) {
            if (!check_node(schema["items"], item, path + "[]", err))
                return false;
        }
    }

    return true;
}

}  // namespace

bool validate_arguments(const std::string& schema_json, const json& args, std::string* err) {
    if (err)
        err->clear();
    if (schema_json.empty())
        return true;
    json schema = json::parse(schema_json, nullptr, false);
    if (schema.is_discarded() || !schema.is_object()) {
        // A malformed schema is a bug in our catalogue, not a user error: let the
        // call through rather than failing every invocation.
        return true;
    }
    return check_node(schema, args.is_object() ? args : json::object(), "", err);
}

std::string clamp_output(const std::string& text, size_t max_bytes) {
    if (max_bytes == 0 || text.size() <= max_bytes)
        return text;
    static const std::string kMarker = "\n... [output truncated]";
    size_t keep = max_bytes > kMarker.size() ? max_bytes - kMarker.size() : max_bytes / 2;
    // Do not split a UTF-8 sequence: walk back to a lead byte.
    while (keep > 0 && (static_cast<unsigned char>(text[keep]) & 0xC0) == 0x80)
        --keep;
    return text.substr(0, keep) + kMarker;
}

ToolDefinition* find_tool(const std::string& name) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    for (auto& t : registry()) {
        if (t->name == name)
            return t.get();
    }
    return nullptr;
}

const std::vector<std::unique_ptr<ToolDefinition>>& all_tools() {
    std::lock_guard<std::mutex> lock(registry_mutex());
    return registry();
}

json tool_catalogue() {
    json out = json::array();
    for (const auto& t : all_tools()) {
        json entry;
        entry["name"] = t->name;
        entry["description"] = t->description;
        entry["available"] = t->available;
        if (!t->unavailable_reason.empty())
            entry["unavailable_reason"] = t->unavailable_reason;
        json schema = json::parse(t->parameters_json, nullptr, false);
        entry["parameters"] = schema.is_discarded() ? json::object() : schema;
        out.push_back(std::move(entry));
    }
    return out;
}

ToolResult execute_tool(const std::string& name, const json& args, const ToolContext& ctx) {
    ToolResult r;

    ToolDefinition* tool = find_tool(name);
    if (!tool) {
        r.error = "unknown tool '" + name + "'";
        r.content = "Error: no such tool '" + name + "'.";
        return r;
    }
    // A tool can be "unavailable" at the server level (no search key, code exec
    // off) and still usable when the caller supplies what it was missing. Let
    // the executor try rather than refusing here -- web_search resolves its
    // key from the per-request context, so it may well succeed. If it cannot,
    // the executor's own error is the better message anyway (it names the
    // provider, not just "unavailable").
    const bool caller_supplied_credentials = !ctx.api_key.empty();
    if (!tool->available && !caller_supplied_credentials) {
        r.error = tool->unavailable_reason.empty() ? "tool unavailable" : tool->unavailable_reason;
        r.content = "Error: tool '" + name + "' is not available: " + r.error;
        return r;
    }
    if (!tool->exec) {
        r.error = "tool has no executor";
        r.content = "Error: tool '" + name + "' has no executor.";
        return r;
    }

    const json effective_args = args.is_object() ? args : json::object();
    std::string verr;
    if (!validate_arguments(tool->parameters_json, effective_args, &verr)) {
        r.error = "invalid arguments: " + verr;
        r.content = "Error: invalid arguments for '" + name + "': " + verr +
                    ". Call the tool again with arguments matching the schema.";
        return r;
    }

    // Per-call overrides win, else the tool's resolved default, else the
    // registry-wide default.
    ToolContext effective = tool->ctx;
    if (!ctx.api_key.empty())
        effective.api_key = ctx.api_key;
    if (!ctx.provider.empty())
        effective.provider = ctx.provider;
    if (ctx.timeout_sec > 0)
        effective.timeout_sec = ctx.timeout_sec;
    if (ctx.max_output_bytes > 0)
        effective.max_output_bytes = ctx.max_output_bytes;
    if (effective.max_output_bytes == 0)
        effective.max_output_bytes = 64 * 1024;

    try {
        r = tool->exec(effective_args, effective);
    } catch (const std::exception& e) {
        r = ToolResult{false, "Error: tool '" + name + "' failed: " + e.what(),
                       std::string("exception: ") + e.what()};
    } catch (...) {
        r = ToolResult{false, "Error: tool '" + name + "' failed with an unknown error.",
                       "unknown exception"};
    }

    r.content = clamp_output(r.content, effective.max_output_bytes);
    return r;
}

namespace {

void register_tool(std::unique_ptr<ToolDefinition> def) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    for (auto& t : registry()) {
        if (t->name == def->name)
            return;  // already registered; first wins
    }
    registry().push_back(std::move(def));
}

}  // namespace

void ensure_default_tools() {
    static std::once_flag once;
    std::call_once(once, [] {
        register_python_tool();
        register_web_search_tool();
    });
}

}  // namespace forge

// Exposed for the individual tool TUs without leaking registry internals.
namespace forge {
namespace tool_registry_detail {
void add_tool(std::unique_ptr<ToolDefinition> def) { register_tool(std::move(def)); }
}  // namespace tool_registry_detail
}  // namespace forge