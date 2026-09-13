/**
 * forge-server - HTTP route registration (OpenAI-compatible payloads)
 *
 * Response shapes follow the OpenAI API so that any OpenAI-compatible client
 * (e.g. OpenCode with @ai-sdk/openai-compatible) can consume the server.
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "server.h"

namespace fsrv = forge;

namespace {

using json = nlohmann::json;

int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string gen_id(const char* prefix) {
    static std::atomic<uint64_t> counter{0};
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    std::ostringstream oss;
    oss << prefix << "-" << std::hex << ms << std::dec << "-" << ++counter;
    return oss.str();
}

// OpenAI error shape: {"error": {"message", "type", "code"}}
void send_error(httplib::Response& res, int status, const std::string& message,
                const char* type, const char* code) {
    json j;
    j["error"]["message"] = message;
    j["error"]["type"] = type;
    j["error"]["code"] = code;
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

// Build the OpenAI tool_calls array from parsed tool calls.
json build_tool_calls_array(const std::vector<fsrv::ToolCall>& calls) {
    json tcs = json::array();
    for (const auto& tc : calls) {
        tcs.push_back({{"id", tc.id},
                       {"type", tc.type},
                       {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
    }
    return tcs;
}

// Build the assistant message: tool_calls when present, else plain content.
json build_assistant_message(const fsrv::ChatCompletionResult& r) {
    json msg = json::object();
    msg["role"] = "assistant";
    if (!r.tool_calls.empty()) {
        msg["tool_calls"] = build_tool_calls_array(r.tool_calls);
        msg["content"] = r.text.empty() ? json(nullptr) : json(r.text);
    } else {
        msg["content"] = r.text;
    }
    return msg;
}

// ---- Request parsing -------------------------------------------------------

bool parse_chat_request(const httplib::Request& http_req, fsrv::ChatCompletionRequest& out,
                        std::string& err) {
    json body = json::parse(http_req.body, nullptr, false);
    if (body.is_discarded()) {
        err = "Invalid JSON in request body";
        return false;
    }

    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        err = "'messages' must be a non-empty array";
        return false;
    }
    for (const auto& m : body["messages"]) {
        if (!m.is_object() || !m.contains("role") || !m.contains("content")) {
            err = "Each message must be an object with 'role' and 'content'";
            return false;
        }
        fsrv::ChatTemplateMessage msg;
        msg.role = m["role"].get<std::string>();
        if (m["content"].is_string()) {
            msg.content = m["content"].get<std::string>();
        } else if (m["content"].is_array()) {
            // Multimodal-style content parts: Phase 1 concatenates text parts only.
            for (const auto& part : m["content"]) {
                if (part.is_object() && part.value("type", "") == "text" &&
                    part.contains("text") && part["text"].is_string()) {
                    msg.content += part["text"].get<std::string>();
                }
            }
        } else {
            err = "'content' must be a string or an array of parts";
            return false;
        }
        out.messages.push_back(std::move(msg));
    }

    if (body.contains("temperature") && body["temperature"].is_number())
        out.temperature = body["temperature"].get<float>();
    if (body.contains("top_p") && body["top_p"].is_number())
        out.top_p = body["top_p"].get<float>();
    if (body.contains("top_k") && body["top_k"].is_number_integer())
        out.top_k = body["top_k"].get<int>();
    if (body.contains("repeat_penalty") && body["repeat_penalty"].is_number())
        out.repeat_penalty = body["repeat_penalty"].get<float>();
    if (body.contains("repeat_last_n") && body["repeat_last_n"].is_number_integer())
        out.repeat_last_n = body["repeat_last_n"].get<int>();
    if (body.contains("max_tokens") && body["max_tokens"].is_number_integer())
        out.max_tokens = body["max_tokens"].get<int>();
    if (body.contains("seed") && body["seed"].is_number_unsigned())
        out.seed = body["seed"].get<uint64_t>();
    if (body.contains("stream") && body["stream"].is_boolean())
        out.stream = body["stream"].get<bool>();
    if (body.contains("stop")) {
        if (body["stop"].is_string()) {
            std::string s = body["stop"].get<std::string>();
            if (!s.empty())
                out.stop.push_back(std::move(s));
        } else if (body["stop"].is_array()) {
            // OpenAI spec: up to 4 stop sequences.
            for (const auto& item : body["stop"]) {
                if (!item.is_string())
                    continue;
                std::string s = item.get<std::string>();
                if (!s.empty())
                    out.stop.push_back(std::move(s));
                if (out.stop.size() >= 4)
                    break;
            }
        } else {
            err = "'stop' must be a string or an array of strings";
            return false;
        }
    }

    // ---- Function calling: tools / tool_choice ----
    // OpenAI shape: tools = [{"type":"function","function":{"name",
    // "description","parameters"}}]; tool_choice = "auto" | "none" |
    // {"type":"function",...} (a forced function).
    if (body.contains("tools") && body["tools"].is_array()) {
        for (const auto& t : body["tools"]) {
            if (!t.is_object())
                continue;
            const json& fn = (t.contains("function") && t["function"].is_object())
                                 ? t["function"]
                                 : t;
            fsrv::ToolSpec spec;
            spec.name = fn.value("name", "");
            spec.description = fn.value("description", "");
            if (fn.contains("parameters") && fn["parameters"].is_object())
                spec.parameters_json = fn["parameters"].dump();
            if (!spec.name.empty())
                out.tools.push_back(std::move(spec));
        }
    }
    if (body.contains("tool_choice")) {
        if (body["tool_choice"].is_string()) {
            out.tool_choice = body["tool_choice"].get<std::string>();
        } else if (body["tool_choice"].is_object()) {
            // Forced function: keep the raw JSON so the prompt can request it.
            out.tool_choice = body["tool_choice"].dump();
        }
    }
    return true;
}

// ---- Handlers --------------------------------------------------------------

// RAII release for the admission slot (exception-safe).
struct SlotGuard {
    fsrv::ServerState& state;
    ~SlotGuard() { state.release_slot(); }
};

void handle_health(httplib::Response& res) {
    res.set_content(R"({"status":"ok"})", "application/json");
}

void handle_models(const fsrv::ServerState& state, httplib::Response& res) {
    json out;
    out["object"] = "list";
    out["data"] = json::array({json{{"id", state.model_id()},
                                    {"object", "model"},
                                    {"created", unix_now()},
                                    {"owned_by", "forge"}}});
    res.set_content(out.dump(), "application/json");
}

void handle_chat_completions(const httplib::Request& http_req, httplib::Response& res,
                             fsrv::ServerState& state) {
    // Debug: FORGE_DUMP_REQ=/path.json dumps the raw request body (e.g. what
    // opencode actually sends) for server-side diagnosis.
    if (const char* dump_path = std::getenv("FORGE_DUMP_REQ")) {
        if (FILE* fp = fopen(dump_path, "w")) {
            fwrite(http_req.body.data(), 1, http_req.body.size(), fp);
            fclose(fp);
        }
    }

    fsrv::ChatCompletionRequest creq;
    std::string err;
    if (!parse_chat_request(http_req, creq, err)) {
        send_error(res, 400, err, "invalid_request_error", "400");
        return;
    }

    // Admission control: reject when the inference queue is full.
    if (!state.try_acquire_slot()) {
        send_error(res, 429, "Server is busy: inference queue is full, retry later.",
                   "server_error", "429");
        return;
    }

    // ---- non-streaming ----
    if (!creq.stream) {
        SlotGuard guard{state};
        fsrv::ChatCompletionResult r;
        try {
            r = state.chat_completion(creq, nullptr);
        } catch (const std::exception& e) {
            send_error(res, 500, std::string("Inference failed: ") + e.what(), "server_error",
                       "500");
            return;
        }

        json out;
        out["id"] = gen_id("chatcmpl");
        out["object"] = "chat.completion";
        out["created"] = unix_now();
        out["model"] = state.model_id();
        out["choices"] = json::array({json{
            {"index", 0},
            {"message", build_assistant_message(r)},
            {"finish_reason", r.finish_reason}}});
        out["usage"] = json{{"prompt_tokens", r.prompt_tokens},
                            {"completion_tokens", r.completion_tokens},
                            {"total_tokens", r.prompt_tokens + r.completion_tokens}};
        res.set_content(out.dump(), "application/json");
        return;
    }

    // ---- streaming (SSE) ----
    const std::string id = gen_id("chatcmpl");
    const int64_t created = unix_now();

    // The admission slot must live until generation finishes inside the chunk
    // provider (the handler returns before the provider runs), so the guard is
    // shared into the provider lambda and released when httplib destroys it.
    auto slot = std::shared_ptr<SlotGuard>(new SlotGuard{state});

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [&state, creq, id, created, slot](size_t, httplib::DataSink& sink) -> bool {
            auto make_chunk = [&](json delta, const char* finish) {
                json chunk;
                chunk["id"] = id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = created;
                chunk["model"] = state.model_id();
                chunk["choices"] = json::array({json{
                    {"index", 0},
                    {"delta", std::move(delta)},
                    {"finish_reason", finish ? json(finish) : json(nullptr)}}});
                return "data: " + chunk.dump() + "\n\n";
            };

            // First chunk carries the assistant role (OpenAI convention).
            std::string first = make_chunk(json{{"role", "assistant"}, {"content", ""}}, nullptr);
            if (!sink.write(first.data(), first.size())) {
                sink.done();
                return true;
            }

            fsrv::ChatCompletionResult r;
            std::string error_text;
            const bool tools_stream = !creq.tools.empty() && creq.tool_choice != "none";
            auto on_token = [&](const std::string& delta) -> bool {
                if (!sink.is_writable())
                    return false;
                if (tools_stream) {
                    // Tool requests: do not stream raw content (it may contain
                    // the tool-call wrapper); the tool_calls are delivered in
                    // the final chunk instead.
                    return true;
                }
                std::string payload = make_chunk(json{{"content", delta}}, nullptr);
                return sink.write(payload.data(), payload.size());
            };
            try {
                r = state.chat_completion(creq, on_token);
            } catch (const std::exception& e) {
                error_text = e.what();
            }

            if (sink.is_writable()) {
                if (!error_text.empty()) {
                    json jerr;
                    jerr["error"]["message"] = "Inference failed: " + error_text;
                    jerr["error"]["type"] = "server_error";
                    std::string payload = "data: " + jerr.dump() + "\n\n";
                    sink.write(payload.data(), payload.size());
                } else {
                    json delta = json::object();
                    if (!r.tool_calls.empty()) {
                        delta["tool_calls"] = build_tool_calls_array(r.tool_calls);
                        delta["content"] = json(nullptr);
                    } else if (tools_stream) {
                        // Tools were offered but the model answered in prose.
                        delta["content"] = r.text;
                    }
                    std::string fin = make_chunk(delta, r.finish_reason.c_str());
                    sink.write(fin.data(), fin.size());
                }
                const char* done = "data: [DONE]\n\n";
                sink.write(done, std::strlen(done));
            }
            sink.done();
            return true;
        });
}

}  // namespace

namespace forge {

void setup_middleware(httplib::Server& svr, const ServerOptions& opt) {
    // CORS: local dev server, allow any origin (llama.cpp server style).
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
    });

    const std::string expect = opt.api_key.empty() ? "" : ("Bearer " + opt.api_key);

    svr.set_pre_routing_handler([expect](const httplib::Request& req,
                                         httplib::Response& res) {
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!expect.empty() && req.get_header_value("Authorization") != expect) {
            json j;
            j["error"]["message"] =
                "Invalid API key. Provide it as 'Authorization: Bearer <key>'.";
            j["error"]["type"] = "invalid_request_error";
            j["error"]["code"] = "invalid_api_key";
            res.status = 401;
            res.set_content(j.dump(), "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
}

void register_routes(httplib::Server& svr, ServerState& state) {
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) { handle_health(res); });

    svr.Get("/v1/models",
            [&state](const httplib::Request&, httplib::Response& res) { handle_models(state, res); });

    svr.Post("/v1/chat/completions", [&state](const httplib::Request& req,
                                              httplib::Response& res) {
        handle_chat_completions(req, res, state);
    });
}

}  // namespace forge
