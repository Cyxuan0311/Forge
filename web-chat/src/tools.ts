/**
 * Client for the server's tool endpoints.
 *
 * forge-server returns tool_calls but never runs them — the caller owns
 * execution by design. These two wrappers are that caller side: `listTools`
 * reads the catalogue (to decide which chat modes to offer), `execTool` runs
 * one call.
 *
 * A tool failure is NOT an exception: the server returns 200 with
 * `ok: false` and an error string, because the model has to read that text and
 * react to it ("the search failed, answer without it"). Only transport problems
 * throw.
 */
import type { Connection } from './types'
import { apiBase, headers } from './api'

export interface ToolSpecInfo {
  name: string
  description: string
  parameters: Record<string, unknown>
  available: boolean
  unavailable_reason?: string
}

export interface ToolListInfo {
  object: string
  tools: ToolSpecInfo[]
  code_exec_enabled: boolean
  search_provider: string
  search_configured: boolean
}

export interface ToolExecResult {
  name: string
  ok: boolean
  content: string
  error?: string
}

/** Per-call overrides the server applies on top of its own configuration. */
export interface ToolExecOptions {
  /** Search provider hint ("tavily", "serper", "brave", "searxng"). */
  provider?: string
  /** API key supplied by the browser when the server has none configured. */
  apiKey?: string
  timeoutSec?: number
}

export async function listTools(conn: Connection): Promise<ToolListInfo> {
  const resp = await fetch(apiBase(conn) + '/v1/tools', { headers: headers(conn) })
  if (!resp.ok) throw new Error('HTTP ' + resp.status)
  const j = await resp.json()
  return {
    object: j.object || 'list',
    tools: Array.isArray(j.tools) ? j.tools : [],
    code_exec_enabled: !!j.code_exec_enabled,
    search_provider: j.search_provider || '',
    search_configured: !!j.search_configured,
  }
}

/**
 * Run one tool. `argumentsJson` is the raw JSON string from the model's
 * tool_calls entry — it is forwarded verbatim so nothing is lost to a
 * re-serialization round trip (and so a malformed value produces the server's
 * own schema error rather than a JS exception here).
 */
export async function execTool(
  conn: Connection,
  name: string,
  argumentsJson: string,
  opts: ToolExecOptions = {},
): Promise<ToolExecResult> {
  let args: Record<string, unknown> = {}
  try {
    const parsed = JSON.parse(argumentsJson || '{}')
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      args = parsed as Record<string, unknown>
    }
  } catch {
    // Malformed arguments are the model's mistake, not ours: send an empty
    // object and let the server's schema check produce the message the model
    // needs to see.
    args = {}
  }

  const body: Record<string, unknown> = { name, arguments: args }
  if (opts.provider) body.provider = opts.provider
  if (opts.apiKey) body.api_key = opts.apiKey
  if (opts.timeoutSec) body.timeout_sec = opts.timeoutSec

  const resp = await fetch(apiBase(conn) + '/v1/tools/exec', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers(conn) },
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    let msg = 'HTTP ' + resp.status
    try {
      const ej = await resp.json()
      if (ej && ej.error && ej.error.message) msg = ej.error.message
    } catch {
      /* not JSON */
    }
    throw new Error(msg)
  }
  const j = await resp.json()
  return {
    name: j.name || name,
    ok: !!j.ok,
    content: typeof j.content === 'string' ? j.content : '',
    error: j.error || undefined,
  }
}