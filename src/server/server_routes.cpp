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
// Separated <think> reasoning travels in the DeepSeek-style
// `reasoning_content` field, which OpenAI-compatible clients that do not use
// it simply ignore.
json build_assistant_message(const fsrv::ChatCompletionResult& r) {
    json msg = json::object();
    msg["role"] = "assistant";
    if (!r.reasoning.empty())
        msg["reasoning_content"] = r.reasoning;
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
        if (!m.is_object() || !m.contains("role")) {
            err = "Each message must be an object with 'role'";
            return false;
        }
        fsrv::ChatTemplateMessage msg;
        msg.role = m["role"].get<std::string>();
        // OpenAI spec: an assistant message that only calls tools has
        // content=null; tool_calls must still reach the template replay.
        if (m.contains("reasoning_content") && m["reasoning_content"].is_string())
            msg.reasoning_content = m["reasoning_content"].get<std::string>();
        if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
            for (const auto& tc : m["tool_calls"]) {
                if (!tc.is_object())
                    continue;
                const json& fn = (tc.contains("function") && tc["function"].is_object())
                                     ? tc["function"]
                                     : tc;
                fsrv::ChatTemplateToolCall call;
                call.id = tc.value("id", "");
                call.name = fn.value("name", "");
                if (fn.contains("arguments")) {
                    if (fn["arguments"].is_string())
                        call.arguments_json = fn["arguments"].get<std::string>();
                    else if (fn["arguments"].is_object())
                        call.arguments_json = fn["arguments"].dump();
                }
                if (!call.name.empty())
                    msg.tool_calls.push_back(std::move(call));
            }
        }
        if (!m.contains("content") || m["content"].is_null()) {
            if (msg.role != "assistant" || msg.tool_calls.empty()) {
                err = "'content' must be a string, an array of parts, or null "
                      "for an assistant tool_calls message";
                return false;
            }
        } else if (m["content"].is_string()) {
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
        // Under KVMem the queue is capped at one by construction (the working set
        // is a single-conversation policy object), so a rejection there is a
        // configuration property, not transient load -- say so, and do not invite
        // a blind retry that will keep failing.
        if (state.tiered_kv_active()) {
            send_error(res, 429,
                       "Server runs with --kvmem, whose tiered KV working set serves one "
                       "conversation at a time. Retry once the current request finishes, or "
                       "start the server without --kvmem to allow concurrent requests.",
                       "server_error", "429");
        } else {
            send_error(res, 429, "Server is busy: inference queue is full, retry later.",
                       "server_error", "429");
        }
        return;
    }

    // ---- non-streaming ----
    if (!creq.stream) {
        SlotGuard guard{state};
        fsrv::ChatCompletionResult r;
        try {
            r = state.chat_completion(creq, nullptr);
        } catch (const fsrv::ContextLengthExceededError& e) {
            // Client-fixable request: reject with 400 before any prefill work,
            // mirroring vLLM's max-context admission error.
            send_error(res, 400, e.what(), "invalid_request_error", "400");
            return;
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
            std::string error_type;  // non-empty => client-fixable (mapped verbatim)
            const bool tools_stream = !creq.tools.empty() && creq.tool_choice != "none";
            // Reasoning deltas (separated <think> content) stream in the
            // DeepSeek-style reasoning_content field; clients that do not
            // render reasoning ignore the extra delta key.
            auto on_reasoning = [&](const std::string& delta) -> bool {
                if (!sink.is_writable())
                    return false;
                std::string payload = make_chunk(json{{"reasoning_content", delta}}, nullptr);
                return sink.write(payload.data(), payload.size());
            };
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
                r = state.chat_completion(creq, on_token, on_reasoning);
            } catch (const fsrv::ContextLengthExceededError& e) {
                error_text = e.what();
                error_type = "invalid_request_error";
            } catch (const std::exception& e) {
                error_text = e.what();
            }

            if (sink.is_writable()) {
                if (!error_text.empty()) {
                    json jerr;
                    jerr["error"]["message"] =
                        error_type.empty() ? ("Inference failed: " + error_text) : error_text;
                    jerr["error"]["type"] =
                        error_type.empty() ? "server_error" : error_type;
                    std::string payload = "data: " + jerr.dump() + "\n\n";
                    sink.write(payload.data(), payload.size());
                } else {
                    json delta = json::object();
                    if (!r.tool_calls.empty()) {
                        // Streaming shape requires a per-call `index` (OpenAI
                        // spec); clients accumulate deltas by index.
                        json tcs = json::array();
                        for (size_t i = 0; i < r.tool_calls.size(); ++i) {
                            const auto& tc = r.tool_calls[i];
                            tcs.push_back({{"index", i},
                                           {"id", tc.id},
                                           {"type", tc.type},
                                           {"function", {{"name", tc.name},
                                                         {"arguments", tc.arguments}}}});
                        }
                        delta["tool_calls"] = tcs;
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

// ---- OpenAI Responses API (used by opencode's chat models) ----
// opencode drives chat through the Responses API (/responses) rather than
// /chat/completions. We translate the request into the existing
// ChatCompletionRequest, reuse the same generation pipeline, and emit the
// Responses wire format (SSE) so no client-side change is needed.

// Convert a Responses request body into a ChatCompletionRequest.
bool parse_responses_request(const httplib::Request& http_req,
                             fsrv::ChatCompletionRequest& out, std::string& err) {
    json body = json::parse(http_req.body, nullptr, false);
    if (body.is_discarded()) {
        err = "Invalid JSON in request body";
        return false;
    }

    // instructions -> system prompt
    if (body.contains("instructions") && body["instructions"].is_string() &&
        !body["instructions"].get<std::string>().empty()) {
        fsrv::ChatTemplateMessage sys;
        sys.role = "system";
        sys.content = body["instructions"].get<std::string>();
        out.messages.push_back(std::move(sys));
    }

    auto extract_text = [](const json& content) -> std::string {
        if (content.is_string())
            return content.get<std::string>();
        if (content.is_array()) {
            std::string s;
            for (const auto& part : content) {
                if (!part.is_object())
                    continue;
                std::string t = part.value("type", "");
                if (t == "input_text" || t == "output_text" || t == "text") {
                    if (part.contains("text") && part["text"].is_string())
                        s += part["text"].get<std::string>();
                }
            }
            return s;
        }
        return "";
    };

    auto add_item = [&](const json& item) {
        if (!item.is_object())
            return;
        const std::string type = item.value("type", "");
        fsrv::ChatTemplateMessage msg;
        if (type == "function_call") {
            // Assistant turn replay: name + JSON-string arguments (+ call id).
            msg.role = "assistant";
            fsrv::ChatTemplateToolCall call;
            call.name = item.value("name", "");
            call.id = item.value("call_id", "");
            if (item.contains("arguments") && item["arguments"].is_string())
                call.arguments_json = item["arguments"].get<std::string>();
            if (!call.name.empty())
                msg.tool_calls.push_back(std::move(call));
            out.messages.push_back(std::move(msg));
            return;
        }
        if (type == "function_call_output" || item.value("role", "") == "tool") {
            msg.role = "tool";
            if (item.contains("output") && item["output"].is_string())
                msg.content = item["output"].get<std::string>();
            else
                msg.content = extract_text(item.value("content", json("")));
            out.messages.push_back(std::move(msg));
            return;
        }
        msg.role = item.value("role", "user");
        msg.content = extract_text(item.value("content", json("")));
        if (item.contains("reasoning_content") && item["reasoning_content"].is_string())
            msg.reasoning_content = item["reasoning_content"].get<std::string>();
        out.messages.push_back(std::move(msg));
    };

    if (body.contains("input")) {
        const json& input = body["input"];
        if (input.is_string()) {
            fsrv::ChatTemplateMessage msg;
            msg.role = "user";
            msg.content = input.get<std::string>();
            out.messages.push_back(std::move(msg));
        } else if (input.is_array()) {
            for (const auto& item : input)
                add_item(item);
        } else {
            err = "'input' must be a string or an array of items";
            return false;
        }
    }

    if (out.messages.empty()) {
        err = "'input' must contain at least one message";
        return false;
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
    if (body.contains("seed") && body["seed"].is_number_unsigned())
        out.seed = body["seed"].get<uint64_t>();
    // Responses API uses max_output_tokens; fall back to max_tokens.
    if (body.contains("max_output_tokens") && body["max_output_tokens"].is_number_integer())
        out.max_tokens = body["max_output_tokens"].get<int>();
    else if (body.contains("max_tokens") && body["max_tokens"].is_number_integer())
        out.max_tokens = body["max_tokens"].get<int>();
    if (body.contains("stream") && body["stream"].is_boolean())
        out.stream = body["stream"].get<bool>();

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
        if (body["tool_choice"].is_string())
            out.tool_choice = body["tool_choice"].get<std::string>();
        else if (body["tool_choice"].is_object())
            out.tool_choice = body["tool_choice"].dump();
    }
    return true;
}

// Build the Responses "output" array from a chat result. The message item
// reuses the id that was announced via response.output_item.added so streaming
// clients can correlate the final item with the deltas they already received.
// function_call items carry explicit `id`s: the @ai-sdk/openai Responses
// schema (zod) rejects items without `id`, which silently drops the tool call
// and leaves the client with an empty "stop" turn.
json build_responses_output(const fsrv::ChatCompletionResult& r,
                            const std::string& message_item_id,
                            const std::string& reasoning_item_id,
                            const std::vector<std::string>& fc_ids) {
    json items = json::array();
    // Reasoning item first (mirrors OpenAI ordering: reasoning -> message ->
    // function_call) when <think> content was separated during generation.
    if (!reasoning_item_id.empty() && !r.reasoning.empty()) {
        items.push_back(json{{"id", reasoning_item_id},
                             {"type", "reasoning"},
                             {"summary", json::array()}});
    }
    // A response may carry both a prose message and function-call items.
    if (!r.text.empty()) {
        items.push_back(json{{"id", message_item_id},
                             {"type", "message"},
                             {"status", "completed"},
                             {"role", "assistant"},
                             {"content", json::array({json{{"type", "output_text"},
                                                           {"text", r.text}}})}});
    }
    for (size_t i = 0; i < r.tool_calls.size(); ++i) {
        const auto& tc = r.tool_calls[i];
        items.push_back(json{{"id", i < fc_ids.size() ? fc_ids[i] : gen_id("fc")},
                             {"type", "function_call"},
                             {"status", "completed"},
                             {"name", tc.name},
                             {"arguments", tc.arguments},
                             {"call_id", tc.id}});
    }
    return items;
}

json responses_usage(const fsrv::ChatCompletionResult& r) {
    return json{{"input_tokens", r.prompt_tokens},
                {"output_tokens", r.completion_tokens},
                {"total_tokens", r.prompt_tokens + r.completion_tokens}};
}

void handle_responses(const httplib::Request& http_req, httplib::Response& res,
                      fsrv::ServerState& state) {
    if (const char* dump_path = std::getenv("FORGE_DUMP_REQ")) {
        if (FILE* fp = fopen(dump_path, "w")) {
            fwrite(http_req.body.data(), 1, http_req.body.size(), fp);
            fclose(fp);
        }
    }

    fsrv::ChatCompletionRequest creq;
    std::string err;
    // Debug: FORGE_DUMP_REQ=/path.json dumps the raw /responses body (same
    // hook as handle_chat_completions) for client-side diagnosis.
    if (const char* dump_path = std::getenv("FORGE_DUMP_REQ")) {
        if (FILE* fp = fopen(dump_path, "w")) {
            fwrite(http_req.body.data(), 1, http_req.body.size(), fp);
            fclose(fp);
        }
    }
    if (!parse_responses_request(http_req, creq, err)) {
        send_error(res, 400, err, "invalid_request_error", "400");
        return;
    }

    if (!state.try_acquire_slot()) {
        // Under KVMem the queue is capped at one by construction (the working set
        // is a single-conversation policy object), so a rejection there is a
        // configuration property, not transient load -- say so, and do not invite
        // a blind retry that will keep failing.
        if (state.tiered_kv_active()) {
            send_error(res, 429,
                       "Server runs with --kvmem, whose tiered KV working set serves one "
                       "conversation at a time. Retry once the current request finishes, or "
                       "start the server without --kvmem to allow concurrent requests.",
                       "server_error", "429");
        } else {
            send_error(res, 429, "Server is busy: inference queue is full, retry later.",
                       "server_error", "429");
        }
        return;
    }

    // ---- non-streaming ----
    if (!creq.stream) {
        SlotGuard guard{state};
        fsrv::ChatCompletionResult r;
        try {
            r = state.chat_completion(creq, nullptr);
        } catch (const fsrv::ContextLengthExceededError& e) {
            send_error(res, 400, e.what(), "invalid_request_error", "400");
            return;
        } catch (const std::exception& e) {
            send_error(res, 500, std::string("Inference failed: ") + e.what(),
                       "server_error", "500");
            return;
        }
        json out;
        out["id"] = gen_id("resp");
        out["object"] = "response";
        out["created_at"] = unix_now();
        out["model"] = state.model_id();
        out["status"] = "completed";
        std::vector<std::string> fc_ids;
        fc_ids.reserve(r.tool_calls.size());
        for (size_t i = 0; i < r.tool_calls.size(); ++i)
            fc_ids.push_back(gen_id("fc"));
        out["output"] = build_responses_output(r, "", "", fc_ids);
        out["usage"] = responses_usage(r);
        res.set_content(out.dump(), "application/json");
        return;
    }

    // ---- streaming (Responses SSE) ----
    const std::string id = gen_id("resp");
    const std::string item_id = gen_id("msg");
    auto slot = std::shared_ptr<SlotGuard>(new SlotGuard{state});

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [&state, creq, id, item_id, slot](size_t, httplib::DataSink& sink) -> bool {
            auto sse = [&](const std::string& type, json data) -> bool {
                if (!sink.is_writable())
                    return false;
                data["type"] = type;
                std::string s = "event: " + type + "\ndata: " + data.dump() + "\n\n";
                return sink.write(s.data(), s.size());
            };

            json base = json{{"id", id},
                             {"object", "response"},
                             {"status", "in_progress"},
                             {"model", state.model_id()},
                             {"output", json::array()}};
            // Responses-API SSE wraps the response object under a "response"
            // field for lifecycle events (created/in_progress/completed/failed);
            // the item/delta events below are flat.
            sse("response.created", json{{"response", base}});
            sse("response.in_progress", json{{"response", base}});

            // When tools are offered, visible deltas are held back until the
            // full text has been parsed for tool calls: a tool call is only
            // recognizable at finalize, so streaming it live would leak the
            // raw call syntax (###TOOL###/<function>/<tool_call>) into the
            // client as message text. Same policy as /chat/completions.
            const bool tools_stream = !creq.tools.empty() && creq.tool_choice != "none";
            std::string held;
            fsrv::ChatCompletionResult r;
            std::string error_text, error_type;

            // The assistant message item must exist before any text deltas so
            // streaming clients can attach deltas to it as they arrive. This
            // holds even when tools are offered (opencode streams prose and
            // only emits a function_call output item if the model actually
            // calls a tool).
            sse("response.output_item.added",
                json{{"output_index", 0},
                     {"item", json{{"id", item_id},
                                   {"type", "message"},
                                   {"status", "in_progress"},
                                   {"role", "assistant"},
                                   {"content", json::array()}}}});
            sse("response.content_part.added",
                json{{"output_index", 0},
                     {"content_index", 0},
                     {"part", json{{"type", "text"}, {"text", ""}}}});

            auto on_token = [&](const std::string& delta) -> bool {
                if (!sink.is_writable())
                    return false;
                if (tools_stream) {
                    held += delta;  // replayed after finalize when it is prose
                    return true;
                }
                // item_id is required by the OpenAI Responses SSE schema; the
                // @ai-sdk/openai parser uses it to attach the delta to the
                // message item (it ignores output_index for text deltas).
                return sse("response.output_text.delta",
                           json{{"item_id", item_id},
                                {"output_index", 0},
                                {"content_index", 0},
                                {"delta", delta}});
            };

            // Reasoning (<think>) deltas stream as a separate Responses
            // reasoning item + summary events, the same wire shape OpenAI uses
            // for o-series models. The item is announced lazily on the first
            // delta (most turns have no thinking) at the next free output
            // index; the message item keeps index 0.
            const std::string rs_id = gen_id("rs");
            int next_output_index = 1;  // 0 is the message item
            int rs_output_index = -1;
            bool rs_part_added = false;
            auto on_reasoning = [&](const std::string& delta) -> bool {
                if (!sink.is_writable())
                    return false;
                if (rs_output_index < 0) {
                    rs_output_index = next_output_index++;
                    sse("response.output_item.added",
                        json{{"output_index", rs_output_index},
                             {"item", json{{"id", rs_id},
                                           {"type", "reasoning"},
                                           {"summary", json::array()}}}});
                }
                if (!rs_part_added) {
                    rs_part_added = true;
                    sse("response.reasoning_summary_part.added",
                        json{{"item_id", rs_id},
                             {"output_index", rs_output_index},
                             {"summary_index", 0},
                             {"part", json{{"type", "summary_text"}, {"text", ""}}}});
                }
                return sse("response.reasoning_summary_text.delta",
                           json{{"item_id", rs_id},
                                {"output_index", rs_output_index},
                                {"summary_index", 0},
                                {"delta", delta}});
            };

            try {
                r = state.chat_completion(creq, on_token, on_reasoning);
            } catch (const fsrv::ContextLengthExceededError& e) {
                error_text = e.what();
                error_type = "invalid_request_error";
            } catch (const std::exception& e) {
                error_text = e.what();
            }

            if (!sink.is_writable())
                return true;

            if (!error_text.empty()) {
                json f = base;
                f["status"] = "failed";
                json err = json{{"message", error_type.empty()
                                                  ? ("Inference failed: " + error_text)
                                                  : error_text},
                                 {"type", error_type.empty() ? "server_error" : error_type}};
                sse("response.failed", json{{"response", f}, {"error", err}});
                sink.done();
                return true;
            }

            // Text content: streamed above (or replayed here after the tool
            // parse proved it is prose). Replay the FINALIZED text (r.text), not
            // the raw `held` buffer: r.text has had its tool-call syntax stripped
            // and <think> reasoning separated, so the delta a streaming client
            // (opencode/@ai-sdk) renders can never contain raw <function>/###TOOL###
            // markers. The /chat/completions path already uses r.text for the
            // equivalent final chunk; this keeps both endpoints consistent.
            if (tools_stream && r.tool_calls.empty() && !r.text.empty()) {
                (void)sse("response.output_text.delta",
                          json{{"item_id", item_id},
                               {"output_index", 0},
                               {"content_index", 0},
                               {"delta", r.text}});
            }
            // Close the reasoning item (if it streamed) before finishing the
            // message item, mirroring OpenAI's event order.
            if (rs_output_index >= 0) {
                sse("response.reasoning_summary_text.done",
                    json{{"item_id", rs_id},
                         {"output_index", rs_output_index},
                         {"summary_index", 0},
                         {"text", r.reasoning}});
                sse("response.reasoning_summary_part.done",
                    json{{"item_id", rs_id},
                         {"output_index", rs_output_index},
                         {"summary_index", 0},
                         {"part", json{{"type", "summary_text"}, {"text", r.reasoning}}}});
                sse("response.output_item.done",
                    json{{"output_index", rs_output_index},
                         {"item", json{{"id", rs_id},
                                       {"type", "reasoning"},
                                       {"summary", json::array({json{
                                            {"type", "summary_text"},
                                            {"text", r.reasoning}}})}}}});
            }

            sse("response.output_text.done",
                json{{"item_id", item_id},
                     {"output_index", 0},
                     {"content_index", 0},
                     {"text", r.text}});
            sse("response.output_item.done",
                json{{"output_index", 0},
                     {"item", json{{"id", item_id},
                                   {"type", "message"},
                                   {"status", "completed"},
                                   {"role", "assistant"},
                                   {"content", json::array({json{{"type", "output_text"},
                                                                 {"text", r.text}}})}}}});

            // Function-call output items. `id` is mandatory: the @ai-sdk/openai
            // zod schema rejects id-less items, which silently dropped tool
            // calls and turned agent turns into empty "stop" answers.
            std::vector<std::string> fc_ids;
            fc_ids.reserve(r.tool_calls.size());
            for (const auto& tc : r.tool_calls) {
                json item = json{{"id", gen_id("fc")},
                                 {"type", "function_call"},
                                 {"name", tc.name},
                                 {"arguments", tc.arguments},
                                 {"call_id", tc.id},
                                 {"status", "completed"}};
                fc_ids.push_back(item["id"].get<std::string>());
                sse("response.output_item.added",
                    json{{"output_index", next_output_index}, {"item", item}});
                sse("response.output_item.done",
                    json{{"output_index", next_output_index}, {"item", item}});
                ++next_output_index;
            }

            json done_obj = json{{"id", id},
                                 {"object", "response"},
                                 {"status", "completed"},
                                 {"model", state.model_id()},
                                 {"output", build_responses_output(
                                                r, item_id,
                                                rs_output_index >= 0 ? rs_id : std::string(),
                                                fc_ids)},
                                 {"usage", responses_usage(r)}};
            sse("response.completed", json{{"response", done_obj}});
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

    // OpenAI Responses API (opencode drives chat through this endpoint).
    svr.Post("/responses", [&state](const httplib::Request& req,
                                    httplib::Response& res) {
        handle_responses(req, res, state);
    });
    svr.Post("/v1/responses", [&state](const httplib::Request& req,
                                       httplib::Response& res) {
        handle_responses(req, res, state);
    });
}

}  // namespace forge
