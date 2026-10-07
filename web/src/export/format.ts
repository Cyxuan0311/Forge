/**
 * format.ts - the four export serialisers.
 *
 * Pure string-in/string-out, no DOM, so scripts/smoke.ts can assert on their
 * output without a browser. The download/clipboard plumbing lives separately in
 * io.ts.
 *
 * All four serialise the SAME /api/metrics snapshot the charts are drawing, so
 * an export always agrees with what was on screen when the button was pressed.
 */

import type { Metrics, RequestRow } from '../types'

/* ------------------------------------------------------------------ CSV -- */

/**
 * CSV.
 *
 * Strict CSV is a single rectangular table, so "everything in one file" cannot
 * literally be one table. The layout is the common multi-table convention: the
 * time series first, then a blank line and a `# name` header before each other
 * block. Excel and LibreOffice show these as separate regions; `pandas.read_csv`
 * will NOT parse the file -- use the JSON export for that.
 */
export function toCsv(m: Metrics): string {
  const out: string[] = []
  const n = m.series.t.length

  out.push('# time_series')
  out.push('t_ms,tok_per_s,ttft_ms,itl_p50_ms,vram_used_bytes,kv_active_bytes')
  for (let i = 0; i < n; i++) {
    out.push(
      [
        m.series.t[i],
        m.series.tok_per_s[i],
        m.series.ttft_ms[i],
        m.series.itl_p50_ms[i],
        m.series.vram_used[i],
        m.series.kv_active[i],
      ]
        .map(csvCell)
        .join(','),
    )
  }

  out.push('')
  out.push('# requests')
  out.push(REQUEST_HEADER.join(','))
  for (const r of m.requests) out.push(REQUEST_COLUMNS.map((f) => csvCell(f(r))).join(','))

  out.push('')
  out.push('# operators')
  out.push('name,total_ms,count,min_ms,max_ms,avg_ms')
  for (const o of m.operators) {
    out.push([o.name, o.total_ms, o.count, o.min_ms, o.max_ms, o.avg_ms].map(csvCell).join(','))
  }

  out.push('')
  out.push('# snapshot')
  out.push('key,value')
  out.push('exported_at,' + csvCell(new Date(m.meta.now_ms).toISOString()))
  out.push('app,' + csvCell(m.meta.app))
  out.push('model,' + csvCell(m.meta.model))
  out.push('arch,' + csvCell(m.meta.arch))
  out.push('uptime_s,' + m.meta.uptime_s)
  for (const [k, v] of scalarSnapshot(m)) out.push(`${k},${csvCell(String(v))}`)

  return out.join('\n') + '\n'
}

const REQUEST_HEADER = [
  'start_ms',
  'end_ms',
  'queue_ms',
  'ttft_ms',
  'prefill_ms',
  'decode_ms',
  'total_ms',
  'prompt_tokens',
  'completion_tokens',
  'prefix_reused',
  'tok_per_s',
  'itl_p50_ms',
  'itl_p90_ms',
  'spec_accept',
  'status',
  'streaming',
  'session',
] as const

const REQUEST_COLUMNS: ((r: RequestRow) => string | number | boolean | null)[] = [
  (r) => r.start_ms,
  (r) => r.end_ms,
  (r) => r.queue_ms,
  // null, not 0: an absent TTFT is a different fact from a zero one.
  (r) => r.ttft_ms,
  (r) => r.prefill_ms,
  (r) => r.decode_ms,
  (r) => r.total_ms,
  (r) => r.prompt_tokens,
  (r) => r.completion_tokens,
  (r) => r.prefix_reused,
  (r) => r.tok_per_s,
  (r) => r.itl_p50_ms,
  (r) => r.itl_p90_ms,
  (r) => r.spec_accept,
  (r) => r.status,
  (r) => r.streaming,
  (r) => r.session,
]

/**
 * RFC 4180 cell escaping.
 *
 * Quoting on `,`/`"`/newline, and doubling embedded quotes. Both are needed for
 * real data: a model path or session label may legitimately contain either.
 * `null` becomes empty, which is what makes "no TTFT" distinguishable from 0.
 */
function csvCell(v: string | number | boolean | null): string {
  if (v === null) return ''
  const s = String(v)
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s
}

/* ----------------------------------------------------------------- JSON -- */

/**
 * JSON.
 *
 * Wrapped rather than dumping /api/metrics verbatim, so a file saved months
 * later says when and by what it was produced without reference to the server
 * it came from. `metrics` is the untouched snapshot.
 */
export function toJson(m: Metrics): string {
  return (
    JSON.stringify(
      {
        exported_at: new Date(m.meta.now_ms).toISOString(),
        generator: `forge-dashboard/${m.meta.app}`,
        metrics: m,
      },
      null,
      2,
    ) + '\n'
  )
}

/* ------------------------------------------------------------- Markdown -- */

/** A human-readable summary, sized for pasting into a bug report or a PR. */
export function toMarkdown(m: Metrics): string {
  const t = m.totals
  const out: string[] = []
  const label = `${m.meta.app}${m.meta.model ? ` — ${m.meta.model}` : ''}`

  out.push(`# Forge dashboard snapshot`, ``, `${label}`, ``)
  out.push(`- arch: \`${m.meta.arch || '—'}\``)
  out.push(`- uptime: ${m.meta.uptime_s.toFixed(1)} s`)
  out.push(`- exported: ${new Date(m.meta.now_ms).toISOString()}`)
  out.push('')

  out.push('## Totals', '')
  out.push('| metric | value |')
  out.push('| --- | --- |')
  const rows: [string, string][] = [
    ['requests', String(t.requests)],
    ['errors', String(t.errors)],
    ['rejected', String(t.rejected)],
    ['prompt tokens', String(t.prompt_tokens)],
    ['completion tokens', String(t.completion_tokens)],
    ['prefix hit rate', `${(t.prefix_hit_rate * 100).toFixed(1)}%`],
    ['lifetime tok/s', t.lifetime_tok_per_s.toFixed(2)],
    ['prefill ms', t.prefill_ms.toFixed(1)],
    ['decode ms', t.decode_ms.toFixed(1)],
    ['queue ms', t.queue_ms.toFixed(1)],
  ]
  for (const [k, v] of rows) out.push(`| ${k} | ${v} |`)
  out.push('')

  out.push('## Latency percentiles', '')
  out.push('| percentile | TTFT (ms) | ITL (ms) | samples |')
  out.push('| --- | --- | --- | --- |')
  for (const q of ['p50', 'p90', 'p99'] as const) {
    const ttft = m.latency[`ttft_${q}`]
    const itl = m.latency[`itl_${q}`]
    out.push(
      `| ${q} | ${ttft.count ? ttft.value.toFixed(1) : '—'} | ${
        itl.count ? itl.value.toFixed(2) : '—'
      } | ${ttft.count} / ${itl.count} |`,
    )
  }
  out.push('')

  out.push('## Memory', '')
  out.push('| gauge | used | total |')
  out.push('| --- | --- | --- |')
  out.push(
    `| VRAM | ${mib(m.host.vram_used_bytes)} MiB | ${mib(m.host.vram_total_bytes)} MiB |`,
  )
  out.push(`| KV cache | ${mib(m.engine.kv_active_bytes)} MiB | ${mib(m.engine.kv_total_bytes)} MiB |`)
  out.push('')

  if (m.live.active) {
    out.push(
      `> A request was **in flight** at export time: ${m.live.tokens} tokens over ` +
        `${(m.live.elapsed_ms / 1000).toFixed(1)} s at ${m.live.tok_per_s.toFixed(2)} tok/s.`,
      '',
    )
  }

  if (m.requests.length) {
    out.push('## Recent requests', '')
    out.push('| # | status | total ms | TTFT ms | tok/s | itl p50 ms | prompt | completion |')
    out.push('| --- | --- | --- | --- | --- | --- | --- | --- |')
    // Newest last, matching the on-screen table.
    m.requests.slice(-10).forEach((r, i) => {
      out.push(
        `| ${i + 1} | ${r.status} | ${r.total_ms.toFixed(1)} | ${
          r.ttft_ms === null ? '—' : r.ttft_ms.toFixed(1)
        } | ${r.tok_per_s.toFixed(2)} | ${r.itl_p50_ms.toFixed(2)} | ${r.prompt_tokens} | ${
          r.completion_tokens
        } |`,
      )
    })
    out.push('')
  }

  if (m.operators_available && m.operators.length) {
    out.push('## Top operators', '')
    out.push('| operator | total ms | count | avg ms |')
    out.push('| --- | --- | --- | --- |')
    for (const o of m.operators.slice(0, 12)) {
      out.push(`| ${o.name} | ${o.total_ms.toFixed(2)} | ${o.count} | ${o.avg_ms.toFixed(2)} |`)
    }
    out.push('')
  }

  return out.join('\n')
}

function mib(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1)
}

/* ------------------------------------------------------------ Prometheus -- */

/**
 * Prometheus text exposition format, TYPE 0.0.4.
 *
 * IMPORTANT: this is a FILE, not an endpoint. It suits the node_exporter
 * textfile collector or a manual `curl`-equivalent import; nothing scrapes it.
 * Live scraping would need a server route (e.g. /api/export), which this
 * dashboard deliberately does not have.
 *
 * Gauges are prefixed `forge_<app>_`, sanitised to [a-zA-Z0-9_]. Labels carry
 * the model and arch so two dashboards can share one scrape target without
 * colliding.
 */
export function toPrometheus(m: Metrics): string {
  const app = sanitize(m.meta.app || 'forge')
  const labels = `{app="${escLabel(m.meta.app)}",model="${escLabel(m.meta.model)}",arch="${escLabel(
    m.meta.arch,
  )}"}`
  const out: string[] = []

  // `extra` is a pre-formatted comma-separated list of EXTRA LABELS, which have
  // to be spliced INSIDE the brace -- appending them after it produces a line no
  // Prometheus parser accepts.
  const gauge = (name: string, help: string, value: number, extra = '') => {
    const extraLabels = extra ? extra.slice(1) : ''
    const withExtra = extraLabels ? labels.slice(0, -1) + ',' + extraLabels + '}' : labels
    out.push(`# HELP forge_${app}_${name} ${help}`)
    out.push(`# TYPE forge_${app}_${name} gauge`)
    out.push(`forge_${app}_${name}${withExtra} ${finite(value)}`)
  }

  gauge('uptime_seconds', 'Seconds since the engine started.', m.meta.uptime_s)
  gauge('requests_total', 'Requests completed since start.', m.totals.requests)
  gauge('errors_total', 'Requests that ended in an error.', m.totals.errors)
  gauge('rejected_total', 'Requests rejected before admission.', m.totals.rejected)
  gauge('prompt_tokens_total', 'Prompt tokens processed.', m.totals.prompt_tokens)
  gauge('completion_tokens_total', 'Completion tokens generated.', m.totals.completion_tokens)
  gauge(
    'prefill_ms_total',
    'Cumulative time spent prefilling.',
    m.totals.prefill_ms,
  )
  gauge('decode_ms_total', 'Cumulative time spent decoding.', m.totals.decode_ms)
  gauge('queue_ms_total', 'Cumulative time spent queued.', m.totals.queue_ms)
  gauge(
    'lifetime_tokens_per_second',
    'Lifetime average output rate.',
    m.totals.lifetime_tok_per_s,
  )
  gauge(
    'prefix_hit_rate',
    'Fraction of prompt tokens served from the prefix cache.',
    m.totals.prefix_hit_rate,
  )

  // Percentiles as gauges with a quantile label rather than summaries: this is a
  // point-in-time snapshot, so there is no window to aggregate over. The label
  // is the numeric quantile (0.5), which is what Prometheus tooling expects --
  // note this is NOT the "quantile" label convention for summaries, which we
  // cannot use here.
  const quantiles = [
    ['p50', 0.5],
    ['p90', 0.9],
    ['p99', 0.99],
  ] as const
  for (const [q, quantile] of quantiles) {
    const ttft = m.latency[`ttft_${q}`]
    const itl = m.latency[`itl_${q}`]
    if (ttft.count > 0) {
      gauge(
        'ttft_ms',
        'Time to first token, from the current sample window.',
        ttft.value,
        `,quantile="${quantile}"`,
      )
    }
    if (itl.count > 0) {
      gauge(
        'itl_ms',
        'Inter-token latency, from the current sample window.',
        itl.value,
        `,quantile="${quantile}"`,
      )
    }
  }

  const host: [string, string, number][] = [
    ['vram_used_bytes', 'Host VRAM in use.', m.host.vram_used_bytes],
    ['vram_total_bytes', 'Total host VRAM.', m.host.vram_total_bytes],
    ['rss_bytes', 'Process resident set size.', m.host.rss_bytes],
    ['cpu_pct', 'Host CPU utilisation, percent.', m.host.cpu_pct],
  ]
  for (const [k, h, v] of host) gauge(k, h, v)

  const engine: [string, string, number][] = [
    ['kv_active_bytes', 'Active KV cache bytes.', m.engine.kv_active_bytes],
    ['kv_total_bytes', 'Total KV cache bytes.', m.engine.kv_total_bytes],
    ['sessions', 'Live sessions.', m.engine.sessions],
    ['active_requests', 'Requests currently in flight.', m.engine.active_requests],
    ['prefix_cache_hits', 'Prefix cache hits.', m.engine.prefix_cache_hits],
    ['prefix_cache_misses', 'Prefix cache misses.', m.engine.prefix_cache_misses],
  ]
  for (const [k, h, v] of engine) gauge(k, h, v)

  const live: [string, string, number][] = [
    ['live_active', '1 while a request is in flight.', m.live.active ? 1 : 0],
    ['live_tokens', 'Tokens emitted by the in-flight request.', m.live.tokens],
    ['live_prompt_tokens', 'Prompt tokens in the in-flight request.', m.live.prompt_tokens],
    ['live_elapsed_ms', 'Milliseconds since the in-flight request started.', m.live.elapsed_ms],
    ['live_tokens_per_second', 'Output rate of the in-flight request.', m.live.tok_per_s],
    ['live_itl_ms', 'Inter-token latency of the in-flight request.', m.live.itl_ms],
  ]
  for (const [k, h, v] of live) gauge(k, h, v)

  // The most recent series sample: the full history is in the CSV/JSON exports.
  const i = m.series.t.length - 1
  if (i >= 0) {
    gauge(
      'series_tokens_per_second',
      'Output rate at the newest sample.',
      m.series.tok_per_s[i] ?? 0,
    )
    gauge('series_ttft_ms', 'TTFT at the newest sample.', m.series.ttft_ms[i] ?? 0)
    gauge('series_itl_p50_ms', 'Inter-token latency p50 at the newest sample.', m.series.itl_p50_ms[i] ?? 0)
  }

  return out.join('\n') + '\n'
}

function scalarSnapshot(m: Metrics): [string, string | number | boolean][] {
  return [
    ['requests', m.totals.requests],
    ['errors', m.totals.errors],
    ['rejected', m.totals.rejected],
    ['prompt_tokens', m.totals.prompt_tokens],
    ['completion_tokens', m.totals.completion_tokens],
    ['prefix_hit_rate', m.totals.prefix_hit_rate],
    ['lifetime_tok_per_s', m.totals.lifetime_tok_per_s],
    ['vram_used_bytes', m.host.vram_used_bytes],
    ['kv_active_bytes', m.engine.kv_active_bytes],
    ['sessions', m.engine.sessions],
  ]
}

/** Metric-name safe: Prometheus allows only [a-zA-Z0-9_] here. */
function sanitize(s: string): string {
  return s.replace(/[^a-zA-Z0-9_]/g, '_').toLowerCase() || 'forge'
}

/**
 * Label-value escaping: backslash, double quote and newline, in that order
 * (backslash first, or the escapes introduced by the later passes get doubled).
 */
function escLabel(s: string): string {
  return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n')
}

/**
 * Prometheus text format has no literal for NaN or Infinity, and a `-1`
 * sentinel (kv_free_pages on the contiguous backend) would be read as a real
 * measurement. Non-finite becomes 0 so the file always parses.
 */
function finite(v: number): number {
  return Number.isFinite(v) ? v : 0
}