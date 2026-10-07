/**
 * io.ts - the DOM side of exporting: turning serialised text into a file the
 * browser saves, or onto the clipboard.
 *
 * Separate from format.ts so the serialisers stay testable headlessly.
 */

import { toCsv, toJson, toMarkdown, toPrometheus } from './format'
import type { Metrics } from '../types'

export type ExportFormat = 'csv' | 'json' | 'markdown' | 'prometheus'

export interface ExportSpec {
  format: ExportFormat
  label: string
  /** One-line explanation shown in the menu. */
  hint: string
  extension: string
  mime: string
  /** Markdown is summarised, so it goes to the clipboard instead of a file. */
  toClipboard: boolean
  serialise: (m: Metrics) => string
}

export const EXPORT_FORMATS: ExportSpec[] = [
  {
    format: 'json',
    label: 'JSON',
    hint: 'Full snapshot, lossless',
    extension: 'json',
    mime: 'application/json',
    toClipboard: false,
    serialise: toJson,
  },
  {
    format: 'csv',
    label: 'CSV',
    hint: 'Series, requests and operators',
    extension: 'csv',
    mime: 'text/csv',
    toClipboard: false,
    serialise: toCsv,
  },
  {
    format: 'markdown',
    label: 'Markdown',
    hint: 'Summary — copies to clipboard',
    extension: 'md',
    mime: 'text/markdown',
    toClipboard: true,
    serialise: toMarkdown,
  },
  {
    format: 'prometheus',
    label: 'Prometheus',
    hint: 'Snapshot text — not a scrape endpoint',
    extension: 'prom',
    mime: 'text/plain; version=0.0.4',
    toClipboard: false,
    serialise: toPrometheus,
  },
]

/** Slugged filename stem: `forge-<app>-<model>-<timestamp>`. */
export function exportFilename(m: Metrics, spec: ExportSpec): string {
  const parts = ['forge', m.meta.app || 'app', m.meta.model, m.meta.arch]
  const stem = parts
    .filter(Boolean)
    .join('-')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  // 2026-10-07t00-15-11: lowercase so the whole name matches ^[a-z0-9-]+$, which
  // is also what makes it safe to paste into a shell unquoted.
  const stamp = new Date(m.meta.now_ms)
    .toISOString()
    .toLowerCase()
    .replace(/[:.]/g, '-')
    .slice(0, 19)
  return `${stem || 'forge'}-${stamp}.${spec.extension}`
}

/**
 * Save text as a file.
 *
 * The object URL is revoked on the next macrotask rather than synchronously:
 * revoking it inside the click handler can race the download in some browsers.
 */
export function download(filename: string, text: string, mime: string): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.rel = 'noopener'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

/**
 * Copy to the clipboard.
 *
 * navigator.clipboard needs a secure context. http://127.0.0.1 counts as one,
 * so the default loopback dashboard is fine; reaching the dashboard over a LAN
 * address is not, hence the execCommand fallback.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // Fall through to the textarea path.
  }

  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.setAttribute('readonly', '')
    // Off-screen rather than display:none, which would make it unselectable.
    ta.style.position = 'fixed'
    ta.style.top = '-1000px'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

/** Dispatch one export. Returns the filename, for the confirmation message. */
export async function runExport(m: Metrics, spec: ExportSpec): Promise<string> {
  const text = spec.serialise(m)
  const name = exportFilename(m, spec)
  if (spec.toClipboard) {
    const ok = await copyToClipboard(text)
    if (!ok) {
      // Clipboard refused: a file is worse than nothing, but better than
      // silently doing nothing, so download it anyway and say so.
      download(name, text, spec.mime)
      return name + ' (clipboard unavailable — downloaded instead)'
    }
    return name + ' (copied)'
  }
  download(name, text, spec.mime)
  return name
}