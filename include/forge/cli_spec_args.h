/**
 * Speculative-decoding flags, shared by forge-cli and forge-server.
 *
 * forge-server had no speculative decoding at all: not one of the ~24 --spec*
 * flags, and no SpeculativeConfig ever reached ContextParams, so
 * Generator's `if (ctx_.params().speculative_config.enabled)` was never true and
 * the server's own `result.spec_stats` reporting was unreachable dead code. The
 * flags live here so the server can get the same surface and the two cannot
 * drift.
 *
 * The profiler flags (--perf / --perf-cuda-events / --sync-prof) stay CLI-only:
 * they act through environment variables and an end-of-run dump, whereas the
 * server's --perf is a per-request stdout summary. Different mechanisms under
 * the same name, deliberately not unified.
 */

#pragma once

#include <string>

#include "forge/speculative.h"

namespace forge {

// Parse one --spec* flag into `spec`. Returns NotMine for anything else (so the
// caller keeps its own chain), Ok on success, Error with `err` filled on a bad
// value. `i` is advanced past the value when the flag takes one.
enum class SpecParseResult { NotMine, Ok, Error };

SpecParseResult parse_spec_arg(const std::string& arg, int argc, char** argv, int& i,
                               SpeculativeConfig& spec, std::string* err);

// True if `arg` is a --spec* flag (used for arity checks and help text).
bool is_spec_arg(const std::string& arg);

}  // namespace forge
