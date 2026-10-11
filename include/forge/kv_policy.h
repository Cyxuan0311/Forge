#pragma once

// kv_policy.h — environment-driven knobs for the multi-level KV cache.
//
//   FORGE_KV_WRITE_POLICY = write_back (default) | write_through | off
//       write_back: host copies are made lazily at eviction time (a device
//                   victim is backed up, then its device pages are released).
//       write_through: host copies are made eagerly at prefix registration.
//       off: legacy single-tier behavior (eviction destroys nodes outright).
//
//   FORGE_KV_HOST_POOL_BYTES = <int>  (default 1073741824 = 1 GiB, 0 = grow-only)
//       Byte budget of the prefix blob pool (host tier L1). When the pool is
//       over budget, the oldest fully host-resident subtrees are dropped.

#include <cstddef>
#include <cstdlib>
#include <string>

#include "prefix_cache.h"

namespace forge {

inline KVWritePolicy kv_write_policy_from_env() {
    const char* env = std::getenv("FORGE_KV_WRITE_POLICY");
    if (!env || !*env)
        return KVWritePolicy::WriteBack;
    const std::string v = env;
    if (v == "write_through" || v == "write-through")
        return KVWritePolicy::WriteThrough;
    if (v == "off" || v == "none" || v == "disabled")
        return KVWritePolicy::Off;
    return KVWritePolicy::WriteBack;  // "write_back" / "write-back" / default
}

// Strict parse for the --kv-write-policy flag: accepts the same spellings as
// the env var. Returns false on an unknown value so the caller can fail loudly
// instead of silently downgrading to the default (out may be null).
inline bool kv_write_policy_from_string(const std::string& v, KVWritePolicy* out) {
    if (v == "write_back" || v == "write-back") {
        if (out) *out = KVWritePolicy::WriteBack;
        return true;
    }
    if (v == "write_through" || v == "write-through") {
        if (out) *out = KVWritePolicy::WriteThrough;
        return true;
    }
    if (v == "off" || v == "none" || v == "disabled") {
        if (out) *out = KVWritePolicy::Off;
        return true;
    }
    return false;
}

// CLI flag over env: an explicit --kv-write-policy value wins; otherwise the
// env var (then the built-in default) applies.
inline KVWritePolicy kv_write_policy_resolved(const std::string& flag) {
    KVWritePolicy p = KVWritePolicy::WriteBack;
    if (!flag.empty() && kv_write_policy_from_string(flag, &p))
        return p;
    return kv_write_policy_from_env();
}

// Phase 2 — asynchronous backup and restore prefetch.
//
//   FORGE_KV_BACKUP_ASYNC = 1 (default) | 0
//       Run the device -> host backup copy on a worker thread. The victim keeps
//       its device pages until the copy is committed, so the eviction still
//       frees memory (just a moment later).
//   FORGE_KV_PREFETCH = 0 (default) | 1
//       Warm a submitted request's host-resident prefix on the worker while
//       earlier requests are still running, so its lookup is a device hit.
//   FORGE_KV_RESTORE_WAIT_MS = <int> (default 200)
//       How long a lookup waits for an in-flight restore before serving a miss.

inline bool kv_async_backup_from_env() {
    const char* env = std::getenv("FORGE_KV_BACKUP_ASYNC");
    if (!env || !*env)
        return true;
    const std::string v = env;
    return !(v == "0" || v == "off" || v == "false" || v == "no");
}

inline bool kv_prefetch_from_env() {
    const char* env = std::getenv("FORGE_KV_PREFETCH");
    if (!env || !*env)
        return false;
    const std::string v = env;
    return !(v == "0" || v == "off" || v == "false" || v == "no");
}

inline int kv_restore_wait_ms_from_env() {
    const char* env = std::getenv("FORGE_KV_RESTORE_WAIT_MS");
    if (!env || !*env)
        return 200;
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    if (end == env || v < 0)
        return 200;
    return static_cast<int>(v);
}

inline size_t kv_host_pool_bytes_from_env() {
    const char* env = std::getenv("FORGE_KV_HOST_POOL_BYTES");
    if (!env || !*env)
        return static_cast<size_t>(1073741824);  // 1 GiB
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    if (end == env || v < 0)
        return static_cast<size_t>(1073741824);
    return static_cast<size_t>(v);
}

}  // namespace forge
