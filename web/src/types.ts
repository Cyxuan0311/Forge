/**
 * types.ts - the /api/metrics response shape.
 *
 * Kept as a hand-written mirror of DashboardServer::snapshot_to_json() in
 * src/dashboard/dashboard_server.cpp. Optional fields are marked `?` where the
 * server can legitimately omit them (a non-streaming request has no TTFT, an
 * operator panel is empty without FORGE_PROFILING=ON), so a missing key reads
 * as `undefined` at the call site instead of silently becoming NaN.
 */

/** A percentile plus how many samples it was computed from. */
export interface Percentile {
  value: number
  count: number
}

export interface EngineGauges {
  kv_active_bytes: number
  kv_total_bytes: number
  /** -1 on the contiguous backend, which has no page pool. */
  kv_free_pages: number
  kv_pool_tokens: number
  kv_page_size: number
  sessions: number
  active_requests: number
  prefix_cache_hits: number
  prefix_cache_misses: number
  tiered_kv: boolean
  paged_kv: boolean
}

export interface HostGauges {
  vram_total_bytes: number
  vram_used_bytes: number
  rss_bytes: number
  cpu_pct: number
}

/**
 * Progress of the request currently in flight.
 *
 * Present so the dashboard animates DURING a generation. Everything else in the
 * payload is only written when a request completes, which left every chart flat
 * for the whole duration of a long decode.
 */
export interface Live {
  active: boolean
  tokens: number
  prompt_tokens: number
  elapsed_ms: number
  /** Trailing ~1 s window, not a lifetime average. */
  tok_per_s: number
  itl_ms: number
  start_ms: number
  streaming: boolean
  session: number
}

export interface Totals {
  requests: number
  errors: number
  rejected: number
  prompt_tokens: number
  completion_tokens: number
  prefill_ms: number
  decode_ms: number
  queue_ms: number
  lifetime_tok_per_s: number
  prefix_hit_rate: number
}

export interface Latency {
  ttft_p50: Percentile
  ttft_p90: Percentile
  ttft_p99: Percentile
  itl_p50: Percentile
  itl_p90: Percentile
  itl_p99: Percentile
}

export interface Series {
  t: number[]
  tok_per_s: number[]
  ttft_ms: number[]
  itl_p50_ms: number[]
  vram_used: number[]
  kv_active: number[]
}

export interface RequestRow {
  start_ms: number
  end_ms: number
  queue_ms: number
  /** null for a request that produced no token (rejected / errored). */
  ttft_ms: number | null
  prefill_ms: number
  decode_ms: number
  total_ms: number
  prompt_tokens: number
  completion_tokens: number
  prefix_reused: number
  tok_per_s: number
  itl_p50_ms: number
  itl_p90_ms: number
  /** Negative means speculative decoding was not active. */
  spec_accept: number
  status: 'ok' | 'error' | 'aborted' | 'rejected' | 'bad_request' | 'unknown'
  streaming: boolean
  /** FNV-1a of the session id; 0 is the default (stateless) session. */
  session: number
}

export interface OperatorRow {
  name: string
  total_ms: number
  count: number
  min_ms: number
  max_ms: number
  avg_ms: number
}

export interface Metrics {
  meta: {
    app: string
    model: string
    arch: string
    start_ms: number
    now_ms: number
    uptime_s: number
    profiling: boolean
  }
  engine: EngineGauges
  host: HostGauges
  live: Live
  totals: Totals
  latency: Latency
  series: Series
  requests: RequestRow[]
  operators: OperatorRow[]
  operators_available: boolean
}