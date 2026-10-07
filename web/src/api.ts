/**
 * api.ts - polling client for GET /api/metrics.
 *
 * Polling rather than SSE. cpp-httplib 0.51 has no set_sse_handler, and its
 * worker pool is 8 threads: one long-lived streaming response per open tab
 * would starve the API server's own workers. A 1 Hz poll costs one short
 * request per second and cannot wedge anything.
 */
import type { Metrics } from './types'

/** Matches kGaugeIntervalMs in src/dashboard/dashboard_server.cpp. */
const POLL_MS = 1000

/**
 * Backoff on failure. Without it, a server restart turns into a tight retry
 * loop that fills the console with stack traces; with it, a dashboard opened
 * before the engine finishes loading recovers on its own.
 */
const MAX_BACKOFF_MS = 15_000

export interface MetricsPoller {
  stop(): void
}

export function pollMetrics(
  onData: (m: Metrics) => void,
  onError: (err: Error) => void,
): MetricsPoller {
  let stopped = false
  let backoff = 0
  let timer: ReturnType<typeof setTimeout> | undefined

  const tick = async () => {
    if (stopped) return
    try {
      const res = await fetch('/api/metrics', { cache: 'no-store' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = (await res.json()) as Metrics
      backoff = 0
      onData(data)
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)))
      backoff = backoff === 0 ? 1000 : Math.min(backoff * 2, MAX_BACKOFF_MS)
    }
    if (!stopped) timer = setTimeout(tick, backoff || POLL_MS)
  }

  void tick()
  return {
    stop() {
      stopped = true
      if (timer !== undefined) clearTimeout(timer)
    },
  }
}

// ---- formatting helpers ---------------------------------------------------

export function fmtBytes(n: number): string {
  if (!n || n < 0) return '—'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`
}

export function fmtMs(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !isFinite(n) || n < 0) return '—'
  return `${n.toFixed(digits)} ms`
}

export function fmtTok(n: number): string {
  if (!isFinite(n) || n <= 0) return '—'
  return `${n.toFixed(1)}`
}

export function fmtCount(n: number): string {
  if (!isFinite(n)) return '—'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 10_000) return `${(n / 1000).toFixed(1)}k`
  return String(n)
}

export function fmtUptime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '—'
  const s = Math.floor(seconds % 60)
  const m = Math.floor((seconds / 60) % 60)
  const h = Math.floor(seconds / 3600)
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

export function fmtClock(ms: number): string {
  if (!ms) return '—'
  const d = new Date(ms)
  return d.toLocaleTimeString(undefined, { hour12: false })
}

export function fmtPercent(ratio: number, digits = 1): string {
  if (!isFinite(ratio) || ratio < 0) return '—'
  return `${(ratio * 100).toFixed(digits)}%`
}