/**
 * Web-dashboard flags, shared by forge-cli and forge-server.
 *
 * These live in forge_infer for the same reason the common flags and the
 * --spec* flags do (see cli_common_args.h and cli_spec_args.h): the two
 * binaries share one parser so their defaults cannot drift. A dashboard that
 * binds a different port or forgets --no-open in one binary but not the other
 * would be exactly the kind of half-shared behaviour those files exist to
 * prevent.
 *
 * Note the deliberate asymmetry with the OTHER --perf flags: those act through
 * environment variables and an end-of-run dump, while --dashboard is a live
 * server. Same reasoning as cli_spec_args.h -- different mechanisms do not get
 * unified just because the names rhyme.
 */

#pragma once

#include <string>

namespace forge {

struct DashboardArgs {
    // Serve the dashboard on its own loopback listener (never on the API port,
    // so --api-key does not have to be bypassed and the OpenAI surface stays
    // exactly as documented).
    bool enabled = false;
    std::string host = "127.0.0.1";
    // 0 = let the OS pick a free port (recommended when the default may be
    // taken). A positive value is a PREFERENCE: if it is busy the dashboard
    // walks upward rather than failing, because losing a run to "port 9090 is
    // in use" is a worse outcome than landing on 9091.
    int port = 9090;
    // Launch a system browser once the listener is up.
    bool open_browser = true;
    // Populate the per-operator breakdown panel.
    //
    // Opt-in rather than automatic, and that is a performance decision, not an
    // oversight: recording an operator scope is not free, and the repo already
    // documents that arming PerfProfiler around the decode loop costs real
    // throughput (see the --perf-cuda-events help text). A dashboard that
    // silently halved decode speed to fill a chart nobody asked for would be a
    // bad trade. Only meaningful on a -DFORGE_PROFILING=ON build; elsewhere the
    // panel reports that it is unavailable.
    bool capture_operators = false;
};

// Parse one dashboard flag. NotMine for anything else, Ok on success, Error
// with `err` filled on a bad value. `i` advances past the value when the flag
// takes one.
enum class DashboardParseResult { NotMine, Ok, Error };

DashboardParseResult parse_dashboard_arg(const std::string& arg, int argc, char** argv, int& i,
                                         DashboardArgs& dash, std::string* err);

// True if `arg` is one of the dashboard flags (used for arity checks and help
// text generation).
bool is_dashboard_arg(const std::string& arg);

}  // namespace forge