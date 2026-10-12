/**
 * Agent module — web_search tool.
 *
 * Fetches results from one of four supported providers and flattens them into a
 * compact numbered list the model can cite.
 *
 * TRANSPORT: outbound HTTPS goes through the `curl` binary rather than an
 * in-process client. cpp-httplib is compiled without CPPHTTPLIB_OPENSSL_SUPPORT
 * (see CMakeLists.txt), so httplib::Client can only speak plain HTTP, and every
 * supported provider is HTTPS-only. Shelling out to curl avoids adding OpenSSL
 * to the server build. argv is passed as an array (no shell), so a query string
 * can never be interpolated into a command line.
 *
 * KEY RESOLUTION: server environment first, then a per-request override from
 * the client (ToolContext::api_key), so a browser can supply its own key without
 * it being baked into the binary or logged server-side.
 */

#include "forge/agent/tool_registry.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace forge {

using json = nlohmann::json;

namespace {

constexpr int kDefaultTimeoutSec = 20;
constexpr size_t kMaxHttpBytes = 2 * 1024 * 1024;

std::string env_or_empty(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string();
}

std::string lower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

/// Run curl with argv, capturing stdout. Returns false on spawn/transport error.
bool run_curl(const std::vector<std::string>& args, const std::string& url, int timeout_sec,
              std::string* out, std::string* err) {
#if defined(_WIN32)
    (void)args; (void)url; (void)timeout_sec;
    *err = "web_search is only supported on POSIX platforms";
    return false;
#else
    int fds[2];
    if (pipe(fds) != 0) {
        *err = std::string("pipe() failed: ") + std::strerror(errno);
        return false;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        *err = std::string("fork() failed: ") + std::strerror(errno);
        return false;
    }

    if (pid == 0) {
        setpgid(0, 0);
        dup2(fds[1], STDOUT_FILENO);
        // stderr must NOT share the body pipe: `curl: (22) The requested URL
        // returned error: 400` would land in the same buffer as the JSON error
        // document and make it unparseable. --fail-with-body already puts the
        // useful explanation in the body.
        const int devnull_err = open("/dev/null", O_WRONLY);
        if (devnull_err >= 0) {
            dup2(devnull_err, STDERR_FILENO);
            if (devnull_err > STDERR_FILENO)
                close(devnull_err);
        }
        close(fds[0]);
        close(fds[1]);
        const int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            if (devnull > STDERR_FILENO)
                close(devnull);
        }
        setenv("PATH", "/usr/local/bin:/usr/bin:/bin", 1);
        unsetenv("LD_PRELOAD");
        // Start from a clean proxy environment. Two reasons:
        //  * correctness -- a proxy configured for the server's own traffic may
        //    not be reachable from this context, and a stale one turns every
        //    search into "could not connect";
        //  * a literal `no_proxy=<local>` (some WSL images set this as a
        //    placeholder rather than a host list) makes curl try to resolve
        //    the *search host* as a proxy and fail with "Connection refused".
        // curl already takes per-request overrides, so honour an explicit
        // proxy only when the operator set one to a real value.
        for (const char* v : {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy"}) {
            const char* val = std::getenv(v);
            if (val && *val && std::string(val) != "<local>")
                setenv(v, val, 1);
            else
                unsetenv(v);
        }
        for (const char* v : {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
                              "NO_PROXY", "no_proxy"})
            unsetenv(v);

        // Route the request explicitly rather than letting curl guess from the
        // inherited environment: either through the operator's proxy (if they
        // set a real one) or straight out with --noproxy "*".
        const char* proxy = nullptr;
        for (const char* v : {"https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"}) {
            const char* val = std::getenv(v);
            if (val && *val && std::string(val) != "<local>") {
                proxy = val;
                break;
            }
        }

        std::vector<char*> argv;
        argv.reserve(args.size() + 5);
        argv.push_back(const_cast<char*>("curl"));
        if (proxy) {
            argv.push_back(const_cast<char*>("--proxy"));
            argv.push_back(const_cast<char*>(proxy));
        } else {
            argv.push_back(const_cast<char*>("--noproxy"));
            argv.push_back(const_cast<char*>("*"));
        }
        for (const auto& a : args)
            argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execvp("curl", argv.data());
        _exit(127);
    }

    close(fds[1]);
    fcntl(fds[0], F_SETFL, O_NONBLOCK);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    std::string buf;
    bool closed = false;
    bool overflow = false;

    while (!closed) {
        if (std::chrono::steady_clock::now() > deadline) {
            kill(-pid, SIGKILL);
            kill(pid, SIGKILL);
            *err = "curl timed out";
            break;
        }
        struct pollfd p = {fds[0], POLLIN, 0};
        const int n = poll(&p, 1, 200);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        char tmp[8192];
        const ssize_t got = read(fds[0], tmp, sizeof(tmp));
        if (got > 0) {
            if (buf.size() + static_cast<size_t>(got) > kMaxHttpBytes) {
                // Keep the head; a truncated JSON body will fail to parse and
                // surface as a provider error rather than silently looping.
                overflow = true;
                closed = true;
                break;
            }
            buf.append(tmp, static_cast<size_t>(got));
        } else if (got == 0) {
            closed = true;
        } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            closed = true;
        }
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);

    if (!err->empty())
        return false;
    if (overflow) {
        *err = "response exceeded 2 MB";
        return false;
    }
    if (!closed) {
        *err = "request did not complete";
        return false;
    }
    // curl's exit status: 0 = fine, 22 = HTTP error under --fail-with-body (the
    // body is still the API's explanation), 127 = curl missing, anything else
    // is a transport failure. Only the last two are hard errors here; an HTTP
    // error still returns the body so the parser can surface the API's message.
    if (WIFEXITED(status)) {
        const int code = WEXITSTATUS(status);
        if (code == 127) {
            *err = "curl binary not found on PATH";
            return false;
        }
        if (code != 0 && code != 22) {
            *err = "curl exited with status " + std::to_string(code);
            return false;
        }
    } else if (WIFSIGNALED(status)) {
        *err = "curl was killed (signal " + std::to_string(WTERMSIG(status)) + ")";
        return false;
    }
    *out = std::move(buf);
    return true;
#endif
}

// ---- provider response flattening ------------------------------------------

/// Escape for embedding raw text inside a JSON string value we are building.
std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    return o;
}

/// Percent-encode for a query-string value (RFC 3986 unreserved set).
std::string url_encode(const std::string& s) {
    std::string out;
    out.reserve(s.size() * 3 / 2);
    char buf[4];
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            std::snprintf(buf, sizeof(buf), "%%%02X", c);
            out += buf;
        }
    }
    return out;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a])))
        ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
        --b;
    return s.substr(a, b - a);
}

struct Hit {
    std::string title;
    std::string url;
    std::string snippet;
};

/// Emit the hits as a numbered list. This is what the model actually reads, so
/// it is plain text with an explicit "[n]" citation key rather than JSON.
std::string format_hits(const std::vector<Hit>& hits, const std::string& query) {
    std::string out;
    out += "Search results for \"" + query + "\":\n\n";
    if (hits.empty()) {
        out += "(no results)";
        return out;
    }
    for (size_t i = 0; i < hits.size(); ++i) {
        out += "[" + std::to_string(i + 1) + "] " + hits[i].title + "\n";
        if (!hits[i].url.empty())
            out += "    URL: " + hits[i].url + "\n";
        if (!hits[i].snippet.empty())
            out += "    " + hits[i].snippet + "\n";
    }
    out += "\nCite sources as [1], [2], ... when you use them.";
    return out;
}

/**
 * If the provider answered with an error document, extract its message.
 *
 * This matters: the HTTP status is discarded by the time we get here (curl
 * exits 22 on 4xx but --fail-with-body still hands us the body), and an error
 * document has no "results" key. Without this check an expired API key parses
 * as an empty result set, and the model tells the user "no results" instead of
 * "your search key is invalid".
 */
bool extract_api_error(const json& j, std::string* err) {
    if (!j.is_object())
        return false;
    for (const char* key : {"error", "detail", "message"}) {
        auto it = j.find(key);
        if (it == j.end())
            continue;
        if (it->is_string()) {
            *err = it->get<std::string>();
            return true;
        }
        // Tavily/OpenAI-style nested form: {"detail": {"error": "..."}}
        if (it->is_object()) {
            for (const char* inner : {"error", "message"}) {
                auto i2 = it->find(inner);
                if (i2 != it->end() && i2->is_string()) {
                    *err = i2->get<std::string>();
                    return true;
                }
            }
        }
    }
    return false;
}

std::vector<Hit> parse_tavily(const std::string& body, std::string* err) {
    std::vector<Hit> hits;
    json j = json::parse(body, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        *err = "tavily returned a body that is not JSON";
        return hits;
    }
    if (extract_api_error(j, err)) {
        *err = "search API error: " + *err;
        return hits;
    }
    if (j.contains("results") && j["results"].is_array()) {
        for (const auto& r : j["results"]) {
            if (!r.is_object())
                continue;
            Hit h;
            h.title = r.value("title", "");
            h.url = r.value("url", "");
            h.snippet = trim(r.value("content", ""));
            hits.push_back(std::move(h));
        }
    }
    return hits;
}

std::vector<Hit> parse_serper(const std::string& body, std::string* err) {
    std::vector<Hit> hits;
    json j = json::parse(body, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        *err = "serper returned a body that is not JSON";
        return hits;
    }
    if (extract_api_error(j, err)) {
        *err = "search API error: " + *err;
        return hits;
    }
    if (j.contains("organic") && j["organic"].is_array()) {
        for (const auto& r : j["organic"]) {
            if (!r.is_object())
                continue;
            Hit h;
            h.title = r.value("title", "");
            h.url = r.value("link", "");
            h.snippet = trim(r.value("snippet", ""));
            hits.push_back(std::move(h));
        }
    }
    return hits;
}

std::vector<Hit> parse_brave(const std::string& body, std::string* err) {
    std::vector<Hit> hits;
    json j = json::parse(body, nullptr, false);
    if (j.is_discarded()) {
        *err = "brave returned a body that is not JSON";
        return hits;
    }
    if (extract_api_error(j, err)) {
        *err = "search API error: " + *err;
        return hits;
    }
    if (j.contains("web") && j["web"].is_object() && j["web"].contains("results") &&
        j["web"]["results"].is_array()) {
        for (const auto& r : j["web"]["results"]) {
            if (!r.is_object())
                continue;
            Hit h;
            h.title = r.value("title", "");
            h.url = r.value("url", "");
            // Brave wraps the description in <strong> highlight tags.
            std::string snip = trim(r.value("description", ""));
            std::string clean;
            for (size_t i = 0; i < snip.size(); ++i) {
                if (snip.compare(i, 8, "<strong>") == 0) {
                    i += 7;
                    continue;
                }
                if (snip.compare(i, 9, "</strong>") == 0) {
                    i += 8;
                    continue;
                }
                clean += snip[i];
            }
            h.snippet = clean;
            hits.push_back(std::move(h));
        }
    }
    return hits;
}

std::vector<Hit> parse_searxng(const std::string& body, std::string* err) {
    std::vector<Hit> hits;
    json j = json::parse(body, nullptr, false);
    if (j.is_discarded() || !j.is_array()) {
        *err = "searxng returned a body that is not a JSON array";
        return hits;
    }
    // SearXNG reports rate limiting as a bare object rather than an array.
    if (extract_api_error(j, err)) {
        *err = "search API error: " + *err;
        return hits;
    }
    for (const auto& r : j) {
        if (!r.is_object())
            continue;
        Hit h;
        h.title = r.value("title", "");
        h.url = r.value("url", "");
        h.snippet = trim(r.value("content", ""));
        hits.push_back(std::move(h));
    }
    return hits;
}

struct Provider {
    std::string name;
    std::string env_key;   // env var holding the API key
    std::string endpoint;  // "" -> build the URL from env (searxng base)
};

const std::vector<Provider>& providers() {
    static const std::vector<Provider> p = {
        {"tavily", "FORGE_SEARCH_API_KEY", "https://api.tavily.com/search"},
        {"serper", "SERPER_API_KEY", "https://google.serper.dev/search"},
        {"brave", "BRAVE_API_KEY", "https://api.search.brave.com/res/v1/web/search"},
        {"searxng", "SEARXNG_API_KEY", ""},
    };
    return p;
}

const Provider* find_provider(const std::string& name) {
    const std::string want = lower(name);
    for (const auto& p : providers()) {
        if (p.name == want)
            return &p;
    }
    return nullptr;
}

}  // namespace

std::string search_provider() {
    if (const char* p = std::getenv("FORGE_SEARCH_PROVIDER")) {
        if (*p && find_provider(p))
            return lower(p);
    }
    for (const auto& p : providers()) {
        if (!env_or_empty(p.env_key.c_str()).empty())
            return p.name;
    }
    return "";
}

std::string search_api_key() {
    for (const auto& p : providers()) {
        const std::string k = env_or_empty(p.env_key.c_str());
        if (!k.empty())
            return k;
    }
    return "";
}

void register_web_search_tool() {
    auto def = std::make_unique<ToolDefinition>();
    def->name = "web_search";
    def->description =
        "Search the web for current information and return a ranked list of results with "
        "titles, URLs and snippets. Use it for facts that may have changed recently or "
        "that you are unsure about; do not use it for stable general knowledge.";
    def->parameters_json = R"({
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {"type": "integer", "description": "How many results to return (1-10)."}
        },
        "required": ["query"]
    })";

    def->ctx.timeout_sec = kDefaultTimeoutSec;

    const std::string configured = search_provider();
    def->available = !configured.empty() || !search_api_key().empty() ||
                      !env_or_empty("FORGE_SEARCH_PROVIDER").empty() ||
                      !env_or_empty("FORGE_SEARCH_BASE_URL").empty();
    if (!def->available) {
        def->unavailable_reason =
            "no search provider configured; set FORGE_SEARCH_API_KEY (tavily/serper/brave) "
            "or FORGE_SEARCH_BASE_URL for a SearXNG instance, or supply a key from the UI";
    }

    def->exec = [](const json& args, const ToolContext& ctx) -> ToolResult {
        ToolResult r;
        const std::string query = trim(args.value("query", ""));
        if (query.empty()) {
            r.error = "empty query";
            r.content = "Error: 'query' was empty.";
            return r;
        }
        int max_results = 5;
        if (args.contains("max_results") && args["max_results"].is_number_integer())
            max_results = args["max_results"].get<int>();
        max_results = std::max(1, std::min(10, max_results));

        // Provider: per-request hint, then env, then infer from which key is set.
        std::string provider_name = ctx.provider;
        if (provider_name.empty())
            provider_name = search_provider();
        if (provider_name.empty() && !ctx.api_key.empty())
            provider_name = "tavily";  // default shape for a client-supplied key
        const Provider* prov = find_provider(provider_name);
        if (!prov) {
            r.error = "unknown provider";
            r.content = "Error: unknown search provider '" + provider_name +
                        "'. Supported: tavily, serper, brave, searxng.";
            return r;
        }

        std::string api_key = ctx.api_key;
        if (api_key.empty())
            api_key = env_or_empty(prov->env_key.c_str());
        // SearXNG usually needs no key; only insist when the provider does.
        if (api_key.empty() && prov->name != "searxng") {
            r.error = "missing api key";
            r.content = "Error: no API key for provider '" + prov->name + "'.";
            return r;
        }

        // ---- build argv (no shell, so the query cannot inject anything) ----
        std::string url;
        std::vector<std::string> curl_args = {
            "-sS", "--compressed", "--fail-with-body",
            "--max-time", std::to_string(ctx.timeout_sec > 0 ? ctx.timeout_sec : kDefaultTimeoutSec),
        };

        if (prov->name == "tavily") {
            json payload = {{"query", query}, {"max_results", max_results}};
            url = prov->endpoint;
            curl_args.push_back("-X");
            curl_args.push_back("POST");
            curl_args.push_back("-H");
            curl_args.push_back("Content-Type: application/json");
            curl_args.push_back("-H");
            curl_args.push_back("Authorization: Bearer " + api_key);
            curl_args.push_back("--data-binary");
            curl_args.push_back(payload.dump());
        } else if (prov->name == "serper") {
            url = prov->endpoint + "?q=" + url_encode(query);
            curl_args.push_back("-H");
            curl_args.push_back("X-API-KEY: " + api_key);
        } else if (prov->name == "brave") {
            url = prov->endpoint + "?q=" + url_encode(query);
            curl_args.push_back("-H");
            curl_args.push_back("X-Subscription-Token: " + api_key);
        } else {
            // searxng: self-hosted base URL, format=json
            std::string base = env_or_empty("FORGE_SEARCH_BASE_URL");
            if (base.empty()) {
                r.error = "missing searxng base url";
                r.content = "Error: FORGE_SEARCH_BASE_URL is not set for the searxng provider.";
                return r;
            }
            if (!base.empty() && base.back() == '/')
                base.pop_back();
            url = base + "/search?q=" + url_encode(query) + "&format=json&language=en";
            if (!api_key.empty()) {
                curl_args.push_back("-H");
                curl_args.push_back("Authorization: Bearer " + api_key);
            }
        }

        curl_args.push_back(url);

        std::string body, err;
        if (!run_curl(curl_args, url, ctx.timeout_sec > 0 ? ctx.timeout_sec : kDefaultTimeoutSec, &body,
                      &err)) {
            r.error = err;
            r.content = "Error: the web search request failed (" + err +
                        "). Do not invent results; tell the user the search was unavailable.";
            return r;
        }

        std::string perr;
        std::vector<Hit> hits;
        if (prov->name == "tavily")
            hits = parse_tavily(body, &perr);
        else if (prov->name == "serper")
            hits = parse_serper(body, &perr);
        else if (prov->name == "brave")
            hits = parse_brave(body, &perr);
        else
            hits = parse_searxng(body, &perr);

        if (!perr.empty()) {
            r.error = perr;
            r.content = "Error: could not parse the search response (" + perr + ").";
            return r;
        }
        if (static_cast<int>(hits.size()) > max_results)
            hits.resize(static_cast<size_t>(max_results));

        r.ok = true;
        r.content = format_hits(hits, query);
        return r;
    };

    tool_registry_detail::add_tool(std::move(def));
}

}  // namespace forge