/**
 * Chat store - all state and actions, one reactive object.
 *
 * Deliberately not split into per-component state: a conversation being
 * streamed mutates from the SSE pump while the sidebar, thread and composer
 * all read it, and localStorage persistence has to stay in exactly one place.
 * Components stay presentational; every mutation goes through the actions
 * exported here.
 */
import { computed, reactive } from 'vue'
import { chatCompletion, listModels } from './api'
import type { Delta, ToolCallWire, Usage } from './api'
import type {
  ChatMessage,
  Connection,
  Conversation,
  FileAttachment,
  GenParams,
  ImageAttachment,
  MessageStats,
  ToolInvocation,
} from './types'
import { DEFAULT_PARAMS } from './types'
import { DEFAULT_MODE, fetchAvailability, modeById, MODES } from './modes'
import type { ModeDef, ToolAvailability } from './modes'
import { UNKNOWN_AVAILABILITY } from './modes'
import { execTool } from './tools'

/** One file staged for sending (image -> dataUrl, text file -> text). */
export interface Attachment {
  id: string
  name: string
  kind: 'image' | 'file'
  size: number
  dataUrl?: string
  text?: string
}

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

function persist(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* quota / private mode: the UI still works for this session */
  }
}

export function uid(): string {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID()
  return 'id-' + Math.random().toString(36).slice(2) + Date.now().toString(36)
}

export const store = reactive({
  conversations: load<Conversation[]>('forge.chat.conversations', []),
  activeId: load<string>('forge.chat.activeId', ''),
  connection: load<Connection>('forge.chat.connection', {
    baseUrl: '',
    apiKey: '',
    model: '',
    searchApiKey: '',
    searchProvider: '',
  }),
  params: load<GenParams>('forge.chat.params', DEFAULT_PARAMS),
  theme: load<string>('forge.chat.theme', 'light'),
  /** Which top bar panel is open. The two panels share one slot on purpose. */
  panel: 'none' as 'none' | 'connection' | 'settings',
  /** Sidebar visibility, remembered across reloads like the model and theme.
   *  Defaults open on a real desktop (where it is a column, not an overlay). */
  sidebarOpen: load<boolean>(
    'forge.chat.sidebarOpen',
    typeof window !== 'undefined' && window.innerWidth > 900,
  ),
  connState: 'idle' as 'idle' | 'ok' | 'bad',
  models: [] as string[],
  streaming: false,
  draft: '',
  pending: [] as Attachment[],
  /** Active chat mode (see modes.ts). Persisted per browser, not per thread. */
  mode: load<string>('forge.chat.mode', DEFAULT_MODE),
  /** What GET /v1/tools reports; drives which modes are offered. */
  toolAvail: UNKNOWN_AVAILABILITY as ToolAvailability,
})

let abortCtrl: AbortController | null = null

export const activeConv = computed<Conversation | null>(
  () => store.conversations.find((c) => c.id === store.activeId) ?? null,
)

export function save(): void {
  persist('forge.chat.conversations', store.conversations)
  persist('forge.chat.activeId', store.activeId)
  persist('forge.chat.connection', store.connection)
  persist('forge.chat.params', store.params)
  persist('forge.chat.mode', store.mode)
  persist('forge.chat.sidebarOpen', store.sidebarOpen)
}

// ---- chat modes ------------------------------------------------------------

/** The active mode definition (falls back to plain chat for an unknown id). */
export function currentMode(): ModeDef {
  return modeById(store.mode)
}

/** Every mode the server can actually serve right now. */
export function availableModes(): ModeDef[] {
  return MODES.filter((m) => {
    if (!m.tools.length || m.optionalTools) return true
    if (!store.toolAvail.ready) return true
    return m.tools.every((n) => store.toolAvail.available[n] !== false)
  })
}

export function setMode(id: string): void {
  if (id === store.mode) return
  store.mode = id
  persist('forge.chat.mode', id)
}

/** Probe the server for tool availability so the mode menu can hide dead modes. */
export async function refreshToolAvailability(): Promise<void> {
  store.toolAvail = await fetchAvailability(store.connection)
}

// ---- theme -----------------------------------------------------------------

export function applyTheme(): void {
  if (store.theme === 'dark') document.documentElement.setAttribute('data-theme', 'dark')
  else document.documentElement.removeAttribute('data-theme')
}

export function toggleTheme(): void {
  store.theme = store.theme === 'dark' ? 'light' : 'dark'
  persist('forge.chat.theme', store.theme)
  applyTheme()
}

// ---- conversations ---------------------------------------------------------

export function newChat(): void {
  const c: Conversation = {
    id: uid(),
    title: 'New chat',
    messages: [],
    ts: Date.now(),
  }
  store.conversations.unshift(c)
  store.activeId = c.id
  // Switching away aborts an in-flight stream; its partial text is kept.
  if (store.streaming && abortCtrl) abortCtrl.abort()
  save()
}

export function selectChat(id: string): void {
  if (id === store.activeId) return
  if (store.streaming && abortCtrl) abortCtrl.abort()
  store.activeId = id
  save()
}

export function deleteChat(id: string): void {
  store.conversations = store.conversations.filter((c) => c.id !== id)
  if (store.activeId === id) {
    store.activeId = store.conversations.length ? store.conversations[0].id : ''
  }
  save()
}

function titleFrom(text: string): string {
  const t = text.trim().replace(/\s+/g, ' ').slice(0, 40)
  return t || 'New chat'
}

// ---- attachments -----------------------------------------------------------

const TEXT_FILE_RE =
  /\.(md|txt|json|py|js|ts|tsx|vue|cpp|c|h|hpp|go|rs|java|sh|yml|yaml|toml|csv|log|xml|html|css)$/i

export function addFiles(fileList: FileList | File[]): void {
  for (const f of Array.from(fileList)) {
    const isImage = !!f.type && f.type.startsWith('image/')
    const att: Attachment = {
      id: uid(),
      name: f.name,
      kind: isImage ? 'image' : 'file',
      size: f.size,
      text: '',
    }
    if (isImage) {
      const rd = new FileReader()
      rd.onload = () => {
        att.dataUrl = String(rd.result)
      }
      rd.readAsDataURL(f)
    } else if (
      f.size <= 65536 &&
      ((!!f.type && f.type.startsWith('text/')) || TEXT_FILE_RE.test(f.name))
    ) {
      const rt = new FileReader()
      rt.onload = () => {
        att.text = String(rt.result)
      }
      rt.readAsText(f)
    }
    store.pending.push(att)
  }
}

export function removePending(id: string): void {
  store.pending = store.pending.filter((a) => a.id !== id)
}

export function onDrop(e: DragEvent): void {
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
    e.preventDefault()
    addFiles(e.dataTransfer.files)
  }
}

// ---- connection ------------------------------------------------------------

export async function testConnection(): Promise<void> {
  try {
    const models = await listModels(store.connection)
    store.models = models
    store.connState = 'ok'
    if (!store.connection.model && models.length) {
      store.connection.model = models[0]
      save()
    }
  } catch {
    store.connState = 'bad'
  }
}

// ---- sending ---------------------------------------------------------------

/**
 * OpenAI vision content: plain string when the message is text-only, typed
 * parts as soon as images or file contents join in.
 */
function buildContent(m: ChatMessage, sent: Attachment[]): string | Array<Record<string, unknown>> {
  const parts: Array<Record<string, unknown>> = []
  if (m.text) parts.push({ type: 'text', text: m.text })
  for (const im of m.images ?? []) {
    parts.push({ type: 'image_url', image_url: { url: im.dataUrl } })
  }
  for (const f of m.files ?? []) {
    const att = sent.find((a) => a.name === f.name && a.kind === 'file')
    if (att && att.text) {
      parts.push({
        type: 'text',
        text: '<file name="' + f.name + '">\n' + att.text + '\n</file>',
      })
    } else {
      parts.push({
        type: 'text',
        text: '[attached file: ' + f.name + ' (' + f.size + ' bytes)]',
      })
    }
  }
  if (parts.length > 1 || (m.images && m.images.length)) return parts
  return m.text || ''
}
export async function send(): Promise<void> {
  if (store.streaming) return
  const text = store.draft.trim()
  if (!text && store.pending.length === 0) return
  if (!activeConv.value) newChat()
  const conv = activeConv.value
  if (!conv) return

  const images: ImageAttachment[] = store.pending
    .filter((a) => a.kind === 'image' && a.dataUrl)
    .map((a) => ({ name: a.name, dataUrl: a.dataUrl as string }))
  const files: FileAttachment[] = store.pending
    .filter((a) => a.kind === 'file')
    .map((a) => ({ name: a.name, size: a.size }))

  const userMsg: ChatMessage = {
    id: uid(),
    role: 'user',
    text,
    images,
    files,
    ts: Date.now(),
  }
  conv.messages.push(userMsg)
  const isFirstUserMsg = conv.messages.filter((m) => m.role === 'user').length === 1
  if (conv.title === 'New chat' || isFirstUserMsg) {
    conv.title = titleFrom(text || (images.length ? images[0].name : 'image'))
  }
  store.draft = ''
  const sent = store.pending
  store.pending = []

  await runAgentTurn(conv, buildRequest(conv, sent))
}

/**
 * Turn the conversation (optionally prefixed by the system prompt) into the
 * OpenAI-shaped message list. Split out of send() so regenerate() can replay an
 * existing turn without re-staging any attachment.
 */
function buildRequest(
  conv: Conversation,
  sent: Attachment[],
): Array<{ role: string; content: string | Array<Record<string, unknown>> }> {
  const reqMsgs: Array<{
    role: string
    content: string | Array<Record<string, unknown>>
  }> = []
  const mode = currentMode()
  const parts: string[] = []
  // Mode instructions come first so the user's own system prompt stays the
  // last word (it is the more specific, per-conversation instruction).
  if (mode.systemPrompt) parts.push(mode.systemPrompt)
  if (store.params.system && store.params.system.trim()) parts.push(store.params.system.trim())
  if (parts.length) reqMsgs.push({ role: 'system', content: parts.join('\n\n') })
  for (const m of conv.messages) {
    if (m.role === 'error') continue
    reqMsgs.push({ role: m.role, content: buildContent(m, sent) })
  }
  return reqMsgs
}

/**
 * Tool names the UI knows how to describe before the server catalogue has
 * loaded. The probe normally wins; these keep the prompt sensible on the very
 * first request of a fresh page.
 */
const TOOL_FALLBACK_DESC: Record<string, string> = {
  web_search: 'Search the web for current information and return ranked results.',
  run_python: 'Execute a short Python 3 snippet and return what it prints.',
}

function toInvocation(c: ToolCallWire): ToolInvocation {
  return { id: c.id, name: c.function.name, args: c.function.arguments, status: 'running' }
}

/** Ceiling on tool rounds in one turn, so a call-obsessed model cannot spin. */
const MAX_TOOL_ROUNDS = 6

/**
 * Run the tools the model asked for, recording each outcome on the assistant
 * message. Returns the tool-role messages that must be fed back so the model
 * can continue. Runs them sequentially: parallel execution would need a second
 * assistant message to interleave, and the common case is one call.
 */
async function runTools(
  calls: ToolCallWire[],
  live: ChatMessage,
): Promise<Array<{ role: string; content: string }>> {
  const offset = live.tools ? live.tools.length : 0
  live.tools = (live.tools ?? []).concat(calls.map(toInvocation))
  const feedBack: Array<{ role: string; content: string }> = []

  for (let i = 0; i < calls.length; i++) {
    const call = calls[i]
    const inv = live.tools![offset + i]
    const t0 = performance.now()
    try {
      const r = await execTool(store.connection, call.function.name, call.function.arguments, {
        // Only sent when the server has no key of its own, so an operator-wide
        // key is never shadowed by whatever a browser happens to have cached.
        ...(store.connection.searchApiKey ? { apiKey: store.connection.searchApiKey } : {}),
        ...(store.connection.searchProvider ? { provider: store.connection.searchProvider } : {}),
      })
      inv.ms = performance.now() - t0
      if (r.ok) {
        inv.status = 'ok'
        inv.result = r.content
      } else {
        inv.status = 'error'
        inv.error = r.error || 'tool failed'
        inv.result = r.content
      }
    } catch (e) {
      // Transport-level failure (server down, key rejected). Surfaced to the
      // model so it degrades gracefully instead of the turn just stalling.
      inv.ms = performance.now() - t0
      inv.status = 'error'
      inv.error = (e as Error).message || String(e)
      inv.result = 'The tool could not be reached: ' + inv.error
    }
    // The model reads this, so keep the framing explicit about failures.
    feedBack.push({
      role: 'tool',
      content: inv.status === 'ok' ? inv.result || '(no output)' : 'ERROR: ' + (inv.error || '') + '\n' + (inv.result || ''),
    })
    save()
  }
  return feedBack
}

/**
 * One full assistant turn, including any tool rounds.
 *
 * forge-server keeps the assistant's tool_calls reply in its own session
 * history (it cannot be reproduced byte-for-byte by a client), so after running
 * the tools we do NOT replay that message -- we send only the tool results as
 * `delta_messages` against the same session id, and the server re-inserts the
 * assistant turn itself. That also keeps the context prefix cache warm.
 */
async function runAgentTurn(
  conv: Conversation,
  reqMsgs: Array<{ role: string; content: string | Array<Record<string, unknown>> }>,
): Promise<void> {
  // One assistant message spans the whole turn, tool rounds included.
  let live: ChatMessage | null = null
  let pendingDelta: Array<{ role: string; content: string }> | null = null

  for (let round = 0; round <= MAX_TOOL_ROUNDS; round++) {
    const turn = await streamCompletion(conv, reqMsgs, pendingDelta, live)
    pendingDelta = null
    live = conv.messages[conv.messages.length - 1] as ChatMessage
    if (!turn.ok) return
    if (!turn.toolCalls.length) return

    pendingDelta = await runTools(turn.toolCalls, live)

    if (round === MAX_TOOL_ROUNDS) {
      // Out of rounds: stop rather than ask again, but tell the user what
      // happened instead of leaving the loop looking like it just ended.
      live.text +=
        '\n\n_(stopped after ' + MAX_TOOL_ROUNDS + ' tool rounds without a final answer)_'
      save()
      return
    }
  }
}

/** What one model turn produced, so the agent loop can decide whether to loop. */
interface TurnResult {
  /** Tool calls the model asked for (empty when it just answered). */
  toolCalls: ToolCallWire[]
  /** False when the turn was aborted or errored. */
  ok: boolean
}

/**
 * Append an assistant message and stream ONE model turn into it. Owns the
 * streaming flag, the abort handle, and the save() that has to happen even if
 * the page is reloaded mid-stream.
 *
 * When the active mode declares tools, `payload.tools` is filled from the
 * server's catalogue and the returned toolCalls are handed back to the caller
 * instead of being discarded -- the model, not this function, decides to call.
 *
 * `into` reuses an existing assistant message: a tool round produces two model
 * turns (the call, then the answer), and without it the caller would get two
 * bubbles, the first one empty.
 */
async function streamCompletion(
  conv: Conversation,
  reqMsgs: Array<{
    role: string
    content: string | Array<Record<string, unknown>>
  }>,
  deltaMessages?: Array<{ role: string; content: string }> | null,
  into?: ChatMessage | null,
): Promise<TurnResult> {
  const assistant: ChatMessage = into ?? {
    id: uid(),
    role: 'assistant',
    text: '',
    reasoning: '',
    ts: Date.now(),
  }
  if (!into) conv.messages.push(assistant)
  // Hold the REACTIVE proxy, not the raw object pushed above: the SSE pump
  // mutates this message in place, and Vue effects only fire on proxy
  // writes -- mutating the raw handle would leave the thread frozen at "...".
  const live = conv.messages[conv.messages.length - 1] as ChatMessage

  store.streaming = true
  abortCtrl = new AbortController()
  // TTFT and wall time are the two numbers only the client can measure: the
  // clock starts before fetch() (request + queue + prefill) and stops at [DONE].
  const t0 = performance.now()
  let ttft: number | undefined
  let stats: Usage | undefined
  const toolCalls: ToolCallWire[] = []

  const payload: Record<string, unknown> = {
    model: store.connection.model || 'local',
    messages: reqMsgs,
    temperature: store.params.temperature,
    max_tokens: store.params.max_tokens,
    top_p: store.params.top_p,
    top_k: store.params.top_k,
    repeat_penalty: store.params.repeat_penalty,
    stream: store.params.stream,
  }
  if (store.params.session_id && store.params.session_id.trim()) {
    payload.session_id = store.params.session_id.trim()
  }
  if (deltaMessages && deltaMessages.length) {
    // Tool round: append the results to the server-side session instead of
    // resending the history. The server re-inserts the assistant tool_calls
    // turn itself (see ServerState::SessionState::last_reply).
    payload.delta_messages = deltaMessages
  }
  // Only advertise the tools this mode actually declares, and only the ones the
  // server says it can run: a disabled tool in the catalogue makes the model
  // call something that can only fail.
  const mode = currentMode()
  if (mode.tools.length) {
    const specs = mode.tools
      .filter((n) => store.toolAvail.available[n] !== false)
      .map((n) => {
        const info = store.toolAvail.specs?.[n]
        return {
          type: 'function',
          function: {
            name: n,
            description: info?.description || TOOL_FALLBACK_DESC[n] || n,
            parameters: info?.parameters || { type: 'object', properties: {} },
          },
        }
      })
    if (specs.length) {
      payload.tools = specs
      payload.tool_choice = 'auto'
    }
  }
  if (store.params.stream) {
    // Ask for the usage-only tail chunk: that is where the server's token counts
    // and prefill/decode timings come from. A server that does not implement it
    // ignores the field, and the UI falls back to its own measurements.
    payload.stream_options = { include_usage: true }
  }

  // Persist the user turn BEFORE the stream starts: the final save() in the
  // try/finally below only runs when the fetch settles, and a page unloaded
  // mid-stream would otherwise lose the whole conversation.
  save()

  try {
    await chatCompletion(
      store.connection,
      payload,
      abortCtrl.signal,
      (d: Delta) => {
        if (d.content) {
          if (ttft === undefined) ttft = performance.now() - t0
          live.text += d.content
        }
        if (d.reasoning) live.reasoning = (live.reasoning || '') + d.reasoning
        if (d.toolCalls && d.toolCalls.length) {
          toolCalls.push(...d.toolCalls)
          // Show the calls immediately: the executor marks each one 'running'
          // and fills in its result, so the user sees progress per call rather
          // than one lump at the end.
          live.tools = d.toolCalls.map(toInvocation)
        }
      },
      (u) => {
        stats = u
      },
    )
  } catch (e) {
    const err = e as Error
    if (err && err.name === 'AbortError') {
      // User pressed Stop (or switched conversations): keep what streamed.
    } else {
      live.role = 'error'
      live.text = 'Error: ' + (err && err.message ? err.message : String(e))
    }
  } finally {
    // Always record a line, even for an aborted or failed turn: a half-answer
    // that took 40s is exactly when the numbers are worth seeing.
    const ms: MessageStats = { wall_ms: performance.now() - t0 }
    if (ttft !== undefined) ms.ttft_ms = ttft
    if (stats) {
      if (stats.prompt_tokens !== undefined) ms.prompt_tokens = stats.prompt_tokens
      if (stats.completion_tokens !== undefined) ms.completion_tokens = stats.completion_tokens
      if (stats.prefill_ms !== undefined) ms.prefill_ms = stats.prefill_ms
      if (stats.decode_ms !== undefined) ms.decode_ms = stats.decode_ms
      if (stats.tokens_per_sec !== undefined) ms.tokens_per_sec = stats.tokens_per_sec
    }
    live.stats = ms
    store.streaming = false
    abortCtrl = null
    save()
  }
  return { toolCalls, ok: live.role !== 'error' }
}

// ---- message actions ------------------------------------------------------

/**
 * Copy to the clipboard. forge-server serves the UI over plain http, where
 * navigator.clipboard is unavailable, so the execCommand path is the one that
 * actually runs in the embedded case.
 */
export async function copyText(s: string): Promise<boolean> {
  if (!s) return false
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(s)
      return true
    }
  } catch {
    /* fall through to the textarea path */
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = s
    ta.style.position = 'fixed'
    ta.style.top = '0'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

/**
 * Remove one message. Deleting a user turn also drops the answer that replied
 * to it: keeping the orphan would put an assistant message with no question in
 * front of it into the next request.
 */
export function deleteMessage(convId: string, msgId: string): void {
  const conv = store.conversations.find((c) => c.id === convId)
  if (!conv) return
  const i = conv.messages.findIndex((m) => m.id === msgId)
  if (i < 0) return
  // Deleting the message being streamed: stop the pump first, otherwise it
  // keeps appending tokens to an object no longer in the thread.
  if (store.streaming && i === conv.messages.length - 1 && abortCtrl) abortCtrl.abort()
  const role = conv.messages[i].role
  conv.messages.splice(i, 1)
  if (role === 'user' && conv.messages[i] && conv.messages[i].role === 'assistant') {
    conv.messages.splice(i, 1)
  }
  save()
}

/**
 * Re-run the turn that produced `msgId`. The stale answer and everything after
 * it are dropped first: those later messages were written against text that is
 * about to change, so keeping them would splice two branches together.
 */
export async function regenerate(msgId: string): Promise<void> {
  if (store.streaming) return
  const conv = activeConv.value
  if (!conv) return
  const i = conv.messages.findIndex((m) => m.id === msgId)
  if (i < 0) return
  // A user message is the prompt being re-run, not a result: keep it and drop
  // only what followed. Cutting from `i` there would send the request with an
  // empty message list -- and silently lose what the user typed.
  const cut = conv.messages[i].role === 'user' ? i + 1 : i
  conv.messages.splice(cut)
  // Attachments staged for the original send are long gone; images are read
  // back off the user message itself, and text files degrade to the same
  // "[attached file: ...]" placeholder any later turn would send.
  await runAgentTurn(conv, buildRequest(conv, []))
}

export function stop(): void {
  if (abortCtrl) abortCtrl.abort()
}
