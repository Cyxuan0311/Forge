/**
 * Agent module — OpenAI function-calling contract types.
 *
 * Moved verbatim from src/server/server.h so the tool-call handling
 * (dialect parsing, reasoning separation) can live and be tested outside
 * the HTTP server. `namespace forge` is unchanged; existing includes of
 * server.h keep seeing these types.
 */

#pragma once

#include <string>
#include <vector>

namespace forge {

struct ToolCall {
    std::string id;          // e.g. "call-<hex>-<n>"
    std::string type = "function";
    std::string name;        // function name
    std::string arguments;   // JSON string (arguments object serialized)
};

// A tool the client exposes; parsed from the OpenAI `tools` array.
struct ToolSpec {
    std::string name;
    std::string description;
    std::string parameters_json;  // JSON schema, serialized
};

}  // namespace forge
