/**
 * Tool-call dialect implementations + registry.
 *
 * forge_text : the server's own text protocol, ###TOOL###{name,arguments}###END###
 *              (tolerant of dropped '#', unquoted JSON keys, invalid escapes,
 *              and mangled tool names). Logic moved verbatim from
 *              src/server/server_state.cpp.
 * xml_function : native XML function blocks, e.g.
 *              <function name="bash"><param name="command"><![CDATA[ls]]></param></function>
 *              (tolerant of missing CDATA, multiple params/calls).
 * qwen_json : <tool_call>{json}</tool_call> blocks and a bare-JSON fallback
 *              (Qwen/GLM-style).
 *
 * Shared tolerance helpers (JSON repair, field extraction, fuzzy tool-name
 * resolution) live in the anonymous namespace and are reused across dialects.
 */

#include "forge/agent/tool_dialect.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <sstream>

#include <nlohmann/json.hpp>

namespace forge {

namespace {

// ---------------------------------------------------------------------------
// Shared helpers (moved from server_state.cpp, logic unchanged)
// ---------------------------------------------------------------------------

// Unique id for a parsed tool call (call-<hex-time>-<counter>).
std::string make_tool_call_id() {
    static std::atomic<uint64_t> counter{0};
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::ostringstream oss;
    oss << "call-" << std::hex << now << std::dec << "-" << ++counter;
    return oss.str();
}

// Normalize a tool name for fuzzy matching against the registered catalogue:
// lowercase and strip everything but alphanumerics, so "get_weather",
// "getWeather" and "getweather" all collapse to "getweather" and match.
std::string normalize_tool_name(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)))
            o += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return o;
}

// Resolve a (possibly mangled) model-emitted name against the registered
// catalogue. Returns the registered spelling, or "" when unresolvable.
std::string resolve_tool_name(const std::string& emitted,
                              const std::vector<std::string>* allowed) {
    if (!allowed || allowed->empty())
        return emitted;
    const std::string norm = normalize_tool_name(emitted);
    if (!norm.empty()) {
        for (const auto& a : *allowed) {
            if (a == emitted)
                return a;
        }
        for (const auto& a : *allowed) {
            if (normalize_tool_name(a) == norm)
                return a;
        }
    }
    return "";
}

// Read a scalar field from a possibly-unquoted-key object body, so that a
// model emitting {name: fn, arguments: {...}} still yields a usable call.
std::string extract_tool_field(const std::string& body, const std::string& key) {
    size_t k = body.find(key);
    if (k == std::string::npos)
        return "";
    size_t c = body.find(':', k + key.size());
    if (c == std::string::npos)
        return "";
    size_t s = c + 1;
    while (s < body.size() && (body[s] == ' ' || body[s] == '\t'))
        ++s;
    if (s < body.size() && body[s] == '"') {
        size_t end = body.find('"', s + 1);
        if (end == std::string::npos)
            return "";
        return body.substr(s + 1, end - s - 1);
    }
    size_t end = body.find_first_of(",}", s);
    if (end == std::string::npos)
        end = body.size();
    std::string v = body.substr(s, end - s);
    size_t a = v.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    size_t b = v.find_last_not_of(" \t\r\n");
    return v.substr(a, b - a + 1);
}

// Read a balanced JSON object field from a possibly-unquoted-key object body.
std::string extract_tool_object(const std::string& body, const std::string& key) {
    size_t k = body.find(key);
    if (k == std::string::npos)
        return "";
    size_t s = body.find('{', k + key.size());
    if (s == std::string::npos)
        return "";
    int depth = 0;
    for (size_t i = s; i < body.size(); ++i) {
        if (body[i] == '{')
            ++depth;
        else if (body[i] == '}') {
            --depth;
            if (depth == 0)
                return body.substr(s, i - s + 1);
        }
    }
    return "";
}

// Lenient JSON repair: some quantized models emit invalid string escapes
// (e.g. \' or \. inside a JSON string) that strict parsers reject with
// "Invalid escape character". We cannot re-train the model, so repair the
// text: any backslash that does NOT begin a valid JSON escape is turned into
// a literal backslash (\\), which is valid JSON and preserves the character
// content. Only string literals are touched; structure outside quotes is left
// untouched so balanced-brace recovery still works.
std::string repair_json_invalid_escapes(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_str = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (in_str) {
            if (c == '\\') {
                bool valid = false;
                if (i + 1 < s.size()) {
                    char n = s[i + 1];
                    valid = (n == '"' || n == '\\' || n == '/' || n == 'b' ||
                             n == 'f' || n == 'n' || n == 'r' || n == 't');
                    if (n == 'u' && i + 5 < s.size() &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 2])) &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 3])) &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 4])) &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 5]))) {
                        valid = true;
                    }
                }
                if (!valid) {
                    // Escape the offending backslash so the following character
                    // stays literal; the result is valid JSON and survives parse.
                    out += '\\';
                    out += '\\';
                    if (i + 1 < s.size()) {
                        out += s[i + 1];
                        ++i;  // consume the escaped character
                    }
                    continue;
                }
            } else if (c == '"') {
                in_str = false;
            }
            out += c;
        } else {
            if (c == '"')
                in_str = true;
            out += c;
        }
    }
    return out;
}

// Parse one tool-call payload into `tc`: strict JSON first (after escape
// repair), then the tolerant path for unquoted keys. Returns false when the
// payload yields no name.
bool parse_call_payload(const std::string& inner, ToolCall& tc) {
    std::string repaired = repair_json_invalid_escapes(inner);
    nlohmann::json j;
    bool parsed = false;
    try {
        j = nlohmann::json::parse(repaired);
        parsed = true;
    } catch (...) {
        size_t b = repaired.find('{');
        size_t e = repaired.rfind('}');
        if (b != std::string::npos && e != std::string::npos && e > b) {
            try {
                j = nlohmann::json::parse(repaired.substr(b, e - b + 1));
                parsed = true;
            } catch (...) {
            }
        }
    }
    if (parsed && j.is_object()) {
        if (j.contains("name") && j["name"].is_string())
            tc.name = j["name"].get<std::string>();
        if (j.contains("arguments")) {
            const auto& a = j["arguments"];
            tc.arguments = a.is_string() ? a.get<std::string>() : a.dump();
        }
    } else {
        // Tolerant path: models often emit unquoted keys, e.g.
        // {name: fn, arguments: {...}}. Recover name/arguments manually.
        tc.name = extract_tool_field(repaired, "name");
        std::string args = extract_tool_object(repaired, "arguments");
        tc.arguments = args.empty() ? "{}" : args;
    }
    return !tc.name.empty();
}

// Quote bare object keys ({command:"ls"} -> {"command":"ls"}) so the
// tolerant fallback yields valid JSON arguments. String literals are
// respected; only ident chars followed by ':' right after '{' or ',' are
// treated as keys.
std::string quote_bare_keys(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    size_t i = 0;
    bool in_str = false;
    while (i < s.size()) {
        const char c = s[i];
        if (in_str) {
            out += c;
            if (c == '\\' && i + 1 < s.size())
                out += s[++i];
            else if (c == '"')
                in_str = false;
            ++i;
            continue;
        }
        if (c == '"') {
            in_str = true;
            out += c;
            ++i;
            continue;
        }
        const bool key_pos = (c == '{' || c == ',');
        out += c;
        ++i;
        if (!key_pos)
            continue;
        // Lookahead: ws* ident ws* ':'  =>  quote the ident.
        size_t j = i;
        while (j < s.size() && std::isspace(static_cast<unsigned char>(s[j])))
            ++j;
        size_t k = j;
        while (k < s.size() &&
               (std::isalnum(static_cast<unsigned char>(s[k])) || s[k] == '_'))
            ++k;
        if (k > j && k < s.size()) {
            size_t m = k;
            while (m < s.size() && std::isspace(static_cast<unsigned char>(s[m])))
                ++m;
            if (m < s.size() && s[m] == ':') {
                out += s.substr(i, j - i);
                out += '"';
                out += s.substr(j, k - j);
                out += '"';
                i = k;
            }
        }
    }
    return out;
}

// Some quantized models collapse the taught JSON into "name{json}"
// (###TOOL###bash{command:"ls -la"}###END##): no "name" key at all. Recover
// the name from the text before the first '{' and the arguments from the
// balanced object starting there.
std::string trim_copy(const std::string& s);
bool parse_name_prefixed(const std::string& payload, ToolCall& tc) {
    const size_t brace = payload.find('{');
    if (brace == std::string::npos || brace == 0)
        return false;
    const std::string name = trim_copy(payload.substr(0, brace));
    if (name.empty())
        return false;
    size_t end = std::string::npos;
    int depth = 0;
    for (size_t i = brace; i < payload.size(); ++i) {
        if (payload[i] == '{')
            ++depth;
        else if (payload[i] == '}' && --depth == 0) {
            end = i;
            break;
        }
    }
    if (end == std::string::npos)
        return false;
    tc.name = name;
    const std::string args = payload.substr(brace, end - brace + 1);
    try {
        tc.arguments = nlohmann::json::parse(quote_bare_keys(repair_json_invalid_escapes(args)))
                           .dump();
    } catch (...) {
        tc.arguments = args;  // keep the raw tolerant form
    }
    return true;
}

std::string trim_copy(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Extract attribute `attr` from a tag body, tolerating "..." and '...'
// quoting and unquoted values: `<function name="bash" x=y>`.
std::string extract_attr(const std::string& tag, const std::string& attr) {
    const std::string pat = attr + "=";
    size_t p = tag.find(pat);
    if (p == std::string::npos)
        return "";
    size_t v = p + pat.size();
    while (v < tag.size() && (tag[v] == ' ' || tag[v] == '\t'))
        ++v;
    if (v < tag.size() && (tag[v] == '"' || tag[v] == '\'')) {
        char q = tag[v];
        size_t e = tag.find(q, v + 1);
        if (e == std::string::npos)
            return "";
        return tag.substr(v + 1, e - v - 1);
    }
    size_t e = v;
    while (e < tag.size() && !std::isspace(static_cast<unsigned char>(tag[e])) &&
           tag[e] != '>' && tag[e] != '/')
        ++e;
    return tag.substr(v, e - v);
}

// ---------------------------------------------------------------------------
// forge_text — the original server protocol (###TOOL###...###END###)
// ---------------------------------------------------------------------------

class ForgeTextDialect final : public ToolCallDialect {
public:
    const char* name() const override { return "forge_text"; }

    std::string render_catalogue(const std::vector<ToolSpec>& tools,
                                 const std::string& tool_choice) const override {
        std::string p =
            "You have access to tools. To call one, respond with exactly one block of the form:\n"
            "###TOOL###{name: <tool_name>, arguments: <json_object>}###END###\n"
            "Only call a tool when it is genuinely needed to answer the user.";
        if (!tool_choice.empty() && tool_choice != "auto")
            p += " You MUST call the requested tool.";
        p += "\n\n# Tools\n";
        for (const auto& t : tools) {
            p += "- name: " + t.name + "\n";
            if (!t.description.empty())
                p += "  description: " + t.description + "\n";
            p += "  parameters: " + (t.parameters_json.empty() ? "{}" : t.parameters_json) + "\n";
        }
        return p;
    }

    std::vector<ToolCall> parse(const std::string& text,
                                const std::vector<std::string>* allowed) const override {
        std::vector<ToolCall> calls;
        size_t pos = 0;
        while (true) {
            size_t s = 0, inner_start = 0;
            s = find_tool_open(text, pos, inner_start);
            if (s == std::string::npos)
                break;
            size_t after = 0;
            const size_t e = find_tool_close(text, inner_start, after);
            if (e == std::string::npos)
                break;
            const std::string payload = text.substr(inner_start, e - inner_start);
            ToolCall tc;
            if (!parse_call_payload(payload, tc) && !parse_name_prefixed(payload, tc))
                break;
            // Resolve the (possibly mangled) name against the registered
            // catalogue; unknown tools are dropped.
            const std::string resolved = resolve_tool_name(tc.name, allowed);
            if (allowed && !allowed->empty() && resolved.empty()) {
                pos = after;  // skip this unresolvable block, keep scanning
                continue;
            }
            tc.name = resolved;
            tc.id = make_tool_call_id();
            calls.push_back(std::move(tc));
            pos = after;
        }
        return calls;
    }

    std::string strip(const std::string& text) const override {
        std::string out = text;
        size_t pos = 0;
        while (true) {
            size_t s = 0, inner_start = 0;
            s = find_tool_open(out, pos, inner_start);
            if (s == std::string::npos)
                break;
            size_t after = 0;
            const size_t e = find_tool_close(out, inner_start, after);
            if (e == std::string::npos) {
                out.erase(s);
                break;
            }
            out.erase(s, after - s);
            pos = s;
        }
        return trim_copy(out);
    }

private:
    // Find a tool-call open marker. The catalogue instructs the model to emit
    // "###TOOL###", but quantized models routinely drop a '#' and emit
    // "###TOOL##" (or add extras). We therefore match "###TOOL" followed by
    // one or more '#'. Returns the position of the "###TOOL" prefix, or npos;
    // `inner_start` is set just past the trailing '#'s (where the JSON begins).
    static size_t find_tool_open(const std::string& t, size_t from, size_t& inner_start) {
        static const std::string prefix = "###TOOL";
        size_t p = t.find(prefix, from);
        if (p == std::string::npos)
            return std::string::npos;
        size_t i = p + prefix.size();
        while (i < t.size() && t[i] == '#')
            ++i;
        inner_start = i;
        return p;
    }

    // Mirror of find_tool_open for the close marker: "###END" followed by one
    // or more '#' ("###END##" from a dropped '#' still closes). Returns the
    // position of the "###END" prefix, or npos; `after` is set just past the
    // trailing '#'s.
    static size_t find_tool_close(const std::string& t, size_t from, size_t& after) {
        static const std::string prefix = "###END";
        const size_t p = t.find(prefix, from);
        if (p == std::string::npos)
            return std::string::npos;
        size_t i = p + prefix.size();
        while (i < t.size() && t[i] == '#')
            ++i;
        after = i;
        return p;
    }
};

// ---------------------------------------------------------------------------
// xml_function — native XML function blocks (observed in the wild, e.g.
// MiniCPM5 emitting <function name="bash"><param name="command">
// <![CDATA[...]]></param></function> instead of the taught protocol)
// ---------------------------------------------------------------------------

class XmlFunctionDialect final : public ToolCallDialect {
public:
    const char* name() const override { return "xml_function"; }

    std::string render_catalogue(const std::vector<ToolSpec>& tools,
                                 const std::string& tool_choice) const override {
        std::string p =
            "You have access to tools. To call one, respond with exactly one block of the form:\n"
            "<function name=\"<tool_name>\"><param name=\"<param_name>\">value</param>...</function>\n"
            "Only call a tool when it is genuinely needed to answer the user.";
        if (!tool_choice.empty() && tool_choice != "auto")
            p += " You MUST call the requested tool.";
        p += "\n\n# Tools\n";
        for (const auto& t : tools) {
            p += "- name: " + t.name + "\n";
            if (!t.description.empty())
                p += "  description: " + t.description + "\n";
            p += "  parameters: " + (t.parameters_json.empty() ? "{}" : t.parameters_json) + "\n";
        }
        return p;
    }

    std::vector<ToolCall> parse(const std::string& text,
                                const std::vector<std::string>* allowed) const override {
        std::vector<ToolCall> calls;
        static const std::string open = "<function";
        static const std::string close = "</function>";
        size_t pos = 0;
        while (true) {
            const size_t s = text.find(open, pos);
            if (s == std::string::npos)
                break;
            // Open tag ends at the first '>' (attribute values with '>' are
            // pathological; skip the block if the tag never terminates).
            const size_t tag_end = text.find('>', s);
            if (tag_end == std::string::npos)
                break;
            const std::string open_tag = text.substr(s, tag_end - s);
            const std::string emitted_name = extract_attr(open_tag, "name");
            if (emitted_name.empty()) {
                pos = s + open.size();  // not a tool block: keep scanning
                continue;
            }
            const size_t body_start = tag_end + 1;
            const size_t e = text.find(close, body_start);
            if (e == std::string::npos)
                break;  // incomplete block: nothing more to extract
            ToolCall tc;
            tc.name = emitted_name;
            tc.arguments = parse_params(text.substr(body_start, e - body_start));
            const std::string resolved = resolve_tool_name(tc.name, allowed);
            if (allowed && !allowed->empty() && resolved.empty()) {
                pos = e + close.size();  // skip this unresolvable block
                continue;
            }
            tc.name = resolved;
            tc.id = make_tool_call_id();
            calls.push_back(std::move(tc));
            pos = e + close.size();
        }
        return calls;
    }

    std::string strip(const std::string& text) const override {
        std::string out = text;
        static const std::string open = "<function";
        static const std::string close = "</function>";
        size_t pos = 0;
        while (true) {
            const size_t s = out.find(open, pos);
            if (s == std::string::npos)
                break;
            const size_t tag_end = out.find('>', s);
            if (tag_end == std::string::npos) {
                out.erase(s);
                break;
            }
            const size_t e = out.find(close, tag_end + 1);
            if (e == std::string::npos) {
                out.erase(s);
                break;
            }
            out.erase(s, e + close.size() - s);
            pos = s;
        }
        return trim_copy(out);
    }

private:
    // Parse <param name="P">value</param> pairs into an arguments JSON object.
    // A CDATA wrapper is unwrapped; a value that itself parses as JSON keeps
    // its type, everything else becomes a JSON string.
    static std::string parse_params(const std::string& body) {
        nlohmann::json args = nlohmann::json::object();
        static const std::string open = "<param";
        static const std::string close = "</param>";
        static const std::string cdata_open = "<![CDATA[";
        static const std::string cdata_close = "]]>";
        size_t pos = 0;
        while (true) {
            const size_t s = body.find(open, pos);
            if (s == std::string::npos)
                break;
            const size_t tag_end = body.find('>', s);
            if (tag_end == std::string::npos)
                break;
            const std::string key = extract_attr(body.substr(s, tag_end - s), "name");
            const size_t body_start = tag_end + 1;
            const size_t e = body.find(close, body_start);
            if (e == std::string::npos)
                break;
            std::string value = body.substr(body_start, e - body_start);
            if (value.compare(0, cdata_open.size(), cdata_open) == 0) {
                const size_t cend = value.find(cdata_close, cdata_open.size());
                value = (cend == std::string::npos)
                            ? value.substr(cdata_open.size())
                            : value.substr(cdata_open.size(), cend - cdata_open.size());
            }
            if (!key.empty()) {
                // Shell commands and other free text stay strings; JSON
                // payloads keep their structure.
                try {
                    args[key] = nlohmann::json::parse(trim_copy(value));
                } catch (...) {
                    args[key] = value;
                }
            }
            pos = e + close.size();
        }
        return args.dump();
    }
};

// ---------------------------------------------------------------------------
// qwen_json — <tool_call>{json}</tool_call> (Qwen / GLM style) with a
// bare-JSON fallback for models that emit the object without any wrapper.
// ---------------------------------------------------------------------------

class QwenJsonDialect final : public ToolCallDialect {
public:
    const char* name() const override { return "qwen_json"; }

    std::string render_catalogue(const std::vector<ToolSpec>& tools,
                                 const std::string& tool_choice) const override {
        std::string p =
            "You have access to tools. To call one, respond with exactly one block of the form:\n"
            "<tool_call>{\"name\": \"<tool_name>\", \"arguments\": {<json_object>}}</tool_call>\n"
            "Only call a tool when it is genuinely needed to answer the user.";
        if (!tool_choice.empty() && tool_choice != "auto")
            p += " You MUST call the requested tool.";
        p += "\n\n# Tools\n";
        for (const auto& t : tools) {
            p += "- name: " + t.name + "\n";
            if (!t.description.empty())
                p += "  description: " + t.description + "\n";
            p += "  parameters: " + (t.parameters_json.empty() ? "{}" : t.parameters_json) + "\n";
        }
        return p;
    }

    std::vector<ToolCall> parse(const std::string& text,
                                const std::vector<std::string>* allowed) const override {
        std::vector<ToolCall> calls;
        static const std::string open = "<tool_call>";
        static const std::string close = "</tool_call>";
        size_t pos = 0;
        while (true) {
            const size_t s = text.find(open, pos);
            if (s == std::string::npos)
                break;
            const size_t e = text.find(close, s + open.size());
            if (e == std::string::npos)
                break;
            ToolCall tc;
            if (!parse_call_payload(text.substr(s + open.size(), e - (s + open.size())), tc))
                break;
            const std::string resolved = resolve_tool_name(tc.name, allowed);
            if (allowed && !allowed->empty() && resolved.empty()) {
                pos = e + close.size();  // skip this unresolvable block
                continue;
            }
            tc.name = resolved;
            tc.id = make_tool_call_id();
            calls.push_back(std::move(tc));
            pos = e + close.size();
        }
        if (calls.empty())
            return parse_bare_json(text, allowed);
        return calls;
    }

    std::string strip(const std::string& text) const override {
        static const std::string open = "<tool_call>";
        static const std::string close = "</tool_call>";
        // A whole-message bare-JSON call (no wrapper) strips to nothing.
        if (text.find(open) == std::string::npos) {
            const std::string trimmed = trim_copy(text);
            if (is_bare_json_candidate(trimmed) && !parse_bare_json(trimmed, nullptr).empty())
                return "";
            return trimmed;
        }
        std::string out = text;
        size_t pos = 0;
        while (true) {
            const size_t s = out.find(open, pos);
            if (s == std::string::npos)
                break;
            const size_t e = out.find(close, s + open.size());
            if (e == std::string::npos) {
                out.erase(s);
                break;
            }
            out.erase(s, e + close.size() - s);
            pos = s;
        }
        return trim_copy(out);
    }

private:
    static bool is_bare_json_candidate(const std::string& t) {
        return !t.empty() && (t[0] == '{' || t[0] == '[');
    }

    // Whole-message JSON: {"name":...} or [{"name":...}, ...].
    std::vector<ToolCall> parse_bare_json(const std::string& text,
                                          const std::vector<std::string>* allowed) const {
        const std::string trimmed = trim_copy(text);
        if (!is_bare_json_candidate(trimmed))
            return {};
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(repair_json_invalid_escapes(trimmed));
        } catch (...) {
            return {};
        }
        std::vector<nlohmann::json> objs;
        if (j.is_object()) {
            objs.push_back(j);
        } else if (j.is_array()) {
            for (const auto& item : j) {
                if (item.is_object())
                    objs.push_back(item);
            }
        }
        std::vector<ToolCall> calls;
        for (const auto& o : objs) {
            ToolCall tc;
            if (o.contains("name") && o["name"].is_string())
                tc.name = o["name"].get<std::string>();
            if (tc.name.empty())
                continue;
            if (o.contains("arguments")) {
                const auto& a = o["arguments"];
                tc.arguments = a.is_string() ? a.get<std::string>() : a.dump();
            } else {
                tc.arguments = "{}";
            }
            const std::string resolved = resolve_tool_name(tc.name, allowed);
            if (allowed && !allowed->empty() && resolved.empty())
                continue;
            tc.name = resolved;
            tc.id = make_tool_call_id();
            calls.push_back(std::move(tc));
        }
        return calls;
    }
};

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

std::vector<std::unique_ptr<ToolCallDialect>>& registry() {
    static std::vector<std::unique_ptr<ToolCallDialect>> r;
    return r;
}

}  // namespace

void register_tool_dialect(std::unique_ptr<ToolCallDialect> dialect) {
    registry().push_back(std::move(dialect));
}

void ensure_default_tool_dialects() {
    static bool done = false;
    if (done)
        return;
    done = true;
    register_tool_dialect(std::make_unique<XmlFunctionDialect>());
    register_tool_dialect(std::make_unique<QwenJsonDialect>());
    register_tool_dialect(std::make_unique<ForgeTextDialect>());
}

const ToolCallDialect* find_tool_dialect(const std::string& name) {
    ensure_default_tool_dialects();
    for (const auto& d : registry()) {
        if (name == d->name())
            return d.get();
    }
    return nullptr;
}

std::vector<ToolCall> parse_tool_calls_auto(const std::string& text,
                                            const std::vector<std::string>* allowed,
                                            const ToolCallDialect** used) {
    ensure_default_tool_dialects();
    // Priority: native XML (observed in the wild) -> qwen json -> the server
    // protocol. First dialect that yields a resolvable call wins.
    static const char* const kPriority[] = {"xml_function", "qwen_json", "forge_text"};
    for (const char* n : kPriority) {
        const ToolCallDialect* d = find_tool_dialect(n);
        if (!d)
            continue;
        auto calls = d->parse(text, allowed);
        if (!calls.empty()) {
            if (used)
                *used = d;
            return calls;
        }
    }
    if (used)
        *used = nullptr;
    return {};
}

std::string strip_all_tool_syntax(const std::string& text) {
    ensure_default_tool_dialects();
    // Apply every registered dialect's strip in turn. Each only removes its
    // own marker format, so unconditionally chaining them is safe: text with no
    // tool-call syntax is returned unchanged.
    std::string out = text;
    for (const auto& d : registry())
        out = d->strip(out);
    return out;
}

}  // namespace forge
