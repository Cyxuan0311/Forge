/**
 * API client for the OpenAI-compatible surface served by forge-server.
 *
 * The UI is normally embedded and served from /chat/ on the SAME origin, in
 * which case baseUrl is '' and every call is relative (/v1/...). A non-empty
 * baseUrl lets the same bundle talk to a server on another port -- useful for
 * `npm run dev`, where Vite proxies /v1 instead.
 */
import type { Connection } from './types'

export function apiBase(conn: Connection): string {
  return (conn.baseUrl || '').replace(/\/+$/, '')
}

export function headers(conn: Connection): Record<string, string> {
  const h: Record<string, string> = {}
  if (conn.apiKey) h['Authorization'] = 'Bearer ' + conn.apiKey
  return h
}

export interface Delta {
  content?: string
  reasoning?: string
  /**
   * Tool calls requested by the model. forge-server emits them as a single
   * terminal delta (it buffers call syntax while tools are offered, so no raw
   * markers reach the client), so this arrives at most once per turn, after the
   * last content delta.
   */
  toolCalls?: ToolCallWire[]
  /** "tool_calls" means the loop must run tools and send the results back. */
  finishReason?: string
}

/** One entry of the OpenAI `message.tool_calls` array. */
export interface ToolCallWire {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

/**
 * Token counts plus the server-side timings. forge-server fills the timing
 * fields on both the plain response and the streaming usage chunk, so the chat
 * UI shows measured numbers rather than counting deltas off its own clock.
 */
export interface Usage {
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  prefill_ms?: number
  decode_ms?: number
  tokens_per_sec?: number
}

export async function listModels(conn: Connection): Promise<string[]> {
  const resp = await fetch(apiBase(conn) + '/v1/models', { headers: headers(conn) })
  if (!resp.ok) throw new Error('HTTP ' + resp.status)
  const j = await resp.json()
  return j && j.data ? j.data.map((m: { id: string }) => m.id) : []
}

/**
 * One chat completion. Streams when payload.stream is true, calling onDelta
 * per SSE chunk; otherwise fires onDelta once with the full message. Raises
 * on HTTP errors with the server's error message when one is present.
 */
export async function chatCompletion(
  conn: Connection,
  payload: Record<string, unknown>,
  signal: AbortSignal,
  onDelta: (d: Delta) => void,
  onUsage?: (u: Usage) => void,
): Promise<void> {
  const resp = await fetch(apiBase(conn) + '/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers(conn) },
    body: JSON.stringify(payload),
    signal,
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

  if (!(payload.stream === true && resp.body)) {
    const j = await resp.json()
    const ch = j.choices && j.choices[0]
    const d: Delta = {
      content: (ch && ch.message && ch.message.content) || '',
      reasoning: (ch && ch.message && ch.message.reasoning_content) || '',
    }
    const calls = ch && ch.message && ch.message.tool_calls
    if (Array.isArray(calls) && calls.length) d.toolCalls = calls as ToolCallWire[]
    if (ch && ch.finish_reason) d.finishReason = ch.finish_reason
    onDelta(d)
    if (onUsage && j.usage) onUsage(j.usage as Usage)
    return
  }

  // SSE pump: one line at a time, "data: {...}" frames ended by [DONE].
  const reader = (resp.body as ReadableStream<Uint8Array>).getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).trim()
      buf = buf.slice(idx + 1)
      if (!line.startsWith('data:')) continue
      const data = line.slice(5).trim()
      if (data === '[DONE]') return
      try {
        const j = JSON.parse(data)
        // Usage-only tail chunk: `choices` is an empty array (OpenAI's
        // stream_options.include_usage shape), so there is no delta to read.
        if (j.usage && (!j.choices || j.choices.length === 0)) {
          if (onUsage) onUsage(j.usage as Usage)
          continue
        }
        const choice = (j.choices && j.choices[0]) || {}
        const delta = choice.delta || {}
        const d: Delta = {}
        if (delta.content) d.content = delta.content
        if (delta.reasoning_content) d.reasoning = delta.reasoning_content
        // Terminal chunk: the whole tool_calls array plus finish_reason land
        // together, so one delta carries both.
        if (Array.isArray(delta.tool_calls) && delta.tool_calls.length) {
          d.toolCalls = delta.tool_calls as ToolCallWire[]
        }
        if (choice.finish_reason) d.finishReason = choice.finish_reason
        if (d.content || d.reasoning || d.toolCalls) onDelta(d)
      } catch {
        /* partial JSON line */
      }
    }
  }
}
