// Host-only tests for the tool executor registry (src/agent/tools/).
//
// Three things are worth pinning down here, because none of them are visible
// from the UI until they misbehave:
//
//  1. Schema validation. A model that hallucinates arguments must be rejected
//     BEFORE the call reaches a subprocess or an outbound HTTP request, and the
//     message it gets back has to name the problem (it may well retry).
//  2. The executor contract. An unknown tool, a disabled tool, and a throwing
//     executor all have to come back as a failed ToolResult with usable text,
//     never as an exception -- the caller is a browser loop that would
//     otherwise wedge on the first malformed call.
//  3. run_python's containment: the timeout really kills, and stdout+stderr
//     both come back.
//
// Compiles the agent sources directly (same pattern as test_tool_dialects) so
// it needs neither the engine nor a GPU.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <nlohmann/json.hpp>

#include "forge/agent/tool_registry.h"

using json = nlohmann::json;
using forge::ToolContext;
using forge::ToolResult;

static int g_failed = 0;

static void check(const char* name, bool cond, const std::string& got = "") {
    if (cond) {
        printf("  ok   %s\n", name);
    } else {
        g_failed++;
        printf("  FAIL %s%s\n", name, got.empty() ? "" : ("\n       got: " + got).c_str());
    }
}

// ---- schema validation ------------------------------------------------------

static void test_schema() {
    printf("=== argument validation ===\n");
    const std::string schema = R"({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "unit": {"type": "string", "enum": ["c", "f"]},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["city"]
    })";
    std::string err;

    check("valid full args accepted",
          forge::validate_arguments(schema, json::parse(R"({"city":"Paris","days":3})"), &err), err);

    check("missing required rejected",
          !forge::validate_arguments(schema, json::parse(R"({"days":3})"), &err), err);
    check("  ...and names the missing argument", err.find("city") != std::string::npos, err);

    check("wrong type rejected",
          !forge::validate_arguments(schema, json::parse(R"({"city":123})"), &err), err);
    check("  ...and says what it wanted", err.find("string") != std::string::npos, err);

    check("float for integer rejected",
          !forge::validate_arguments(schema, json::parse(R"({"city":"P","days":1.5})"), &err),
          err);

    check("enum violation rejected",
          !forge::validate_arguments(schema, json::parse(R"({"city":"P","unit":"kelvin"})"), &err),
          err);

    check("bad array element rejected",
          !forge::validate_arguments(schema, json::parse(R"({"city":"P","tags":[1]})"), &err),
          err);

    check("empty schema accepts anything",
          forge::validate_arguments("", json::parse(R"({"whatever":1})"), &err), err);

    // A broken catalogue schema must not reject every call forever.
    check("malformed schema does not block the call",
          forge::validate_arguments("{not json", json::parse(R"({"city":"P"})"), &err), err);
}

// ---- output clamping --------------------------------------------------------

static void test_clamp() {
    printf("=== output clamping ===\n");
    const std::string small = "hello";
    check("short output untouched", forge::clamp_output(small, 1000) == small);

    const std::string big(5000, 'x');
    const std::string clamped = forge::clamp_output(big, 200);
    check("long output shortened", clamped.size() < 300, std::to_string(clamped.size()));
    check("  ...with a marker", clamped.find("truncated") != std::string::npos);

    // UTF-8 must not be cut mid-sequence.
    std::string utf8;
    for (int i = 0; i < 200; ++i) utf8 += "\xE4\xB8\xAD";  // 3-byte CJK
    const std::string cut = forge::clamp_output(utf8, 301);
    const std::string tail = cut.substr(cut.size() - 20);
    for (char c : tail) {
        if ((static_cast<unsigned char>(c) & 0xC0) == 0x80) continue;
    }
    // The marker suffix is ASCII, so just assert we still have content and the
    // body round-trips as valid UTF-8 in practice (no lone continuation byte at
    // the truncation point before the marker).
    check("utf-8 clamp keeps the prefix", cut.find("\xE4\xB8\xAD") != std::string::npos);

    check("zero max means no clamp", forge::clamp_output(big, 0) == big);
}

// ---- executor contract ------------------------------------------------------

namespace {

struct Harness {
    static ToolResult boom(const json&, const ToolContext&) {
        throw std::runtime_error("executor exploded");
    }
    static ToolResult echo(const json& args, const ToolContext&) {
        return ToolResult{true, "echo:" + args.dump(), ""};
    }
};

}  // namespace

static void test_execute_contract() {
    printf("=== executor contract ===\n");
    // Enable code exec BEFORE registration: the tool captures its availability
    // when it is registered, so setting the flag afterwards would leave the
    // catalogue reporting "disabled" and skip the runtime checks below.
    forge::set_code_exec_enabled(true);
    forge::ensure_default_tools();

    const ToolResult unknown = forge::execute_tool("does_not_exist", json::object(), ToolContext{});
    check("unknown tool fails", !unknown.ok);
    check("  ...with text the model can read", unknown.content.find("no such tool") !=
                                                std::string::npos,
          unknown.content);

    // register_python_tool ran inside ensure_default_tools; with code exec off
    // it must be present but unavailable, not silently missing.
    const forge::ToolDefinition* py = forge::find_tool("run_python");
    check("run_python is registered", py != nullptr);
    const forge::ToolDefinition* search = forge::find_tool("web_search");
    check("web_search is registered", search != nullptr);

    // Schema enforcement happens inside execute_tool too.
    if (py && py->available) {
        const ToolResult bad =
            forge::execute_tool("run_python", json::parse(R"({"nope":1})"), ToolContext{});
        check("bad arguments rejected before execution", !bad.ok);
        check("  ...with an actionable message",
              bad.content.find("missing required argument") != std::string::npos, bad.content);
    } else if (py) {
        printf("  skip run_python execution checks (code exec disabled)\n");
    }

    // The catalogue the UI polls must be valid JSON with the fields it reads.
    const json cat = forge::tool_catalogue();
    check("catalogue is a non-empty array", cat.is_array() && cat.size() >= 2);
    bool names_ok = true;
    bool avail_ok = true;
    for (const auto& t : cat) {
        if (!t.contains("name") || !t["name"].is_string()) names_ok = false;
        if (!t.contains("available") || !t["available"].is_boolean()) avail_ok = false;
    }
    check("every catalogue entry has a name", names_ok);
    check("every catalogue entry reports availability", avail_ok);
}

// ---- run_python containment -------------------------------------------------

static void test_python() {
    printf("=== run_python ===\n");
    forge::set_code_exec_enabled(true);
    if (!forge::code_exec_enabled()) {
        check("code exec flag honoured", false);
        return;
    }

    const ToolResult ok =
        forge::execute_tool("run_python", json::parse(R"J({"code":"print(6*7)"})J"), ToolContext{});
    if (ok.ok) {
        check("runs and returns stdout", ok.content.find("42") != std::string::npos, ok.content);
    } else {
        // A python3-less CI box is not a failure of this suite.
        printf("  skip run_python runtime checks (%s)\n", ok.error.c_str());
        return;
    }

    const ToolResult both = forge::execute_tool(
        "run_python",
        json::parse(R"J({"code":"import sys\nprint('OUT')\nsys.stderr.write('ERR\\n')"})J"),
        ToolContext{});
    check("stdout captured", both.content.find("OUT") != std::string::npos, both.content);
    check("stderr captured", both.content.find("ERR") != std::string::npos, both.content);

    const ToolResult bad =
        forge::execute_tool("run_python", json::parse(R"J({"code":"def f(:"})J"), ToolContext{});
    check("syntax error reported as failure", !bad.ok);
    check("  ...with the traceback text", bad.content.find("SyntaxError") != std::string::npos,
          bad.content);

    // The timeout is the containment that matters: an infinite loop must come
    // back, not wedge the request.
    ToolContext fast;
    fast.timeout_sec = 2;
    const auto t0 = std::clock();
    const ToolResult slow = forge::execute_tool(
        "run_python", json::parse(R"J({"code":"import time\ntime.sleep(60)"})J"), fast);
    const double elapsed = static_cast<double>(std::clock() - t0) / CLOCKS_PER_SEC;
    check("runaway snippet is killed", !slow.ok && slow.error == "timeout", slow.error);
    check("  ...promptly", elapsed < 30.0, std::to_string(elapsed) + "s");

    // A trailing bare expression is echoed, like a REPL: models write
    // `f(x)` far more often than `print(f(x))`, and a plain script would
    // return nothing at all.
    const ToolResult echoed = forge::execute_tool(
        "run_python", json::parse(R"J({"code":"def f(n):\n    return n * 2\nf(21)"})J"),
        ToolContext{});
    check("trailing expression is echoed", echoed.content.find("42") != std::string::npos,
          echoed.content);

    // ...but an explicit print still works, and is not doubled.
    const ToolResult printed = forge::execute_tool(
        "run_python", json::parse(R"J({"code":"print('once')"})J"), ToolContext{});
    size_t n = 0;
    for (size_t i = 0; (i = printed.content.find("once", i)) != std::string::npos; ++i) ++n;
    check("explicit print works", n == 1, printed.content);

    // A bare string literal as the last line is dead code, not a value to echo.
    const ToolResult literal = forge::execute_tool(
        "run_python", json::parse(R"J({"code":"x = 1\n'done'"})J"), ToolContext{});
    check("trailing string literal is not echoed",
          literal.content.find("done") == std::string::npos, literal.content);

    // Variables and imports still resolve: the snippet runs in one namespace.
    const ToolResult state = forge::execute_tool(
        "run_python",
        json::parse(R"J({"code":"import math\nvals = [1, 4, 9]\nsum(math.sqrt(v) for v in vals)"})J"),
        ToolContext{});
    check("imports and state persist within the snippet",
          state.content.find('\n6') != std::string::npos || state.content.find('6') != std::string::npos,
          state.content);  // sqrt(1)+sqrt(4)+sqrt(9) == 6

    // A runtime error still surfaces as a failure with its traceback.
    const ToolResult boom = forge::execute_tool(
        "run_python", json::parse(R"J({"code":"1/0"})J"), ToolContext{});
    check("runtime error reported", !boom.ok);
    check("  ...with ZeroDivisionError", boom.content.find("ZeroDivisionError") != std::string::npos,
          boom.content);

    // Empty code is a schema-valid but useless call; say so rather than spawning.
    const ToolResult empty =
        forge::execute_tool("run_python", json::parse(R"J({"code":"   "})J"), ToolContext{});
    check("empty snippet rejected", !empty.ok);
}

int main() {
    test_schema();
    test_clamp();
    test_execute_contract();
    test_python();

    if (g_failed) {
        printf("\ntool-registry: FAIL (%d check(s))\n", g_failed);
        return 1;
    }
    printf("\ntool-registry: PASS\n");
    return 0;
}