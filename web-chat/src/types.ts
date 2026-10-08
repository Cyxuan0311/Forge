/**
 * Shared types for the chat UI. Everything persisted to localStorage goes
 * through JSON here, so the shapes double as the storage schema.
 */

export interface ImageAttachment {
  name: string
  dataUrl: string
}

export interface FileAttachment {
  name: string
  size: number
}

/**
 * Per-turn performance, shown under the answer.
 *
 * prompt_tokens / completion_tokens and the *_ms figures come from the server
 * (`usage` in the response, or the streaming usage chunk) so they measure the
 * model rather than the browser. `ttft_ms` and `wall_ms` are the only
 * client-side numbers: time-to-first-token as the UI experienced it, and the
 * full round trip including the request itself.
 */
export interface MessageStats {
  prompt_tokens?: number
  completion_tokens?: number
  prefill_ms?: number
  decode_ms?: number
  tokens_per_sec?: number
  ttft_ms?: number
  wall_ms?: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'error'
  text: string
  /** <think>-style reasoning content, shown in a collapsible block. */
  reasoning?: string
  images?: ImageAttachment[]
  files?: FileAttachment[]
  /** Present once the turn finished: server timings + client TTFT. */
  stats?: MessageStats
  ts: number
}

export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  ts: number
}

export interface Connection {
  /** '' = same origin as the UI (the embedded /chat/ case). */
  baseUrl: string
  apiKey: string
  model: string
}

export interface GenParams {
  temperature: number
  max_tokens: number
  top_p: number
  top_k: number
  repeat_penalty: number
  system: string
  session_id: string
  stream: boolean
}

/**
 * Server-side anti-degeneration defaults (see server.h ChatCompletionRequest):
 * small models loop indefinitely without a repeat penalty, so these mirror the
 * server's defaults rather than OpenAI's bare ones.
 */
export const DEFAULT_PARAMS: GenParams = {
  temperature: 0.7,
  max_tokens: 512,
  top_p: 1.0,
  top_k: 40,
  repeat_penalty: 1.1,
  system: '',
  session_id: '',
  stream: true,
}
