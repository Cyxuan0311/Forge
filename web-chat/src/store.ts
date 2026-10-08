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
import type { Usage } from './api'
import type {
  ChatMessage,
  Connection,
  Conversation,
  FileAttachment,
  GenParams,
  ImageAttachment,
  MessageStats,
} from './types'
import { DEFAULT_PARAMS } from './types'

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
  }),
  params: load<GenParams>('forge.chat.params', DEFAULT_PARAMS),
  theme: load<string>('forge.chat.theme', 'light'),
  /** Which top bar panel is open. The two panels share one slot on purpose. */
  panel: 'none' as 'none' | 'connection' | 'settings',
  connState: 'idle' as 'idle' | 'ok' | 'bad',
  models: [] as string[],
  streaming: false,
  draft: '',
  pending: [] as Attachment[],
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

  await streamCompletion(conv, buildRequest(conv, sent))
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
  if (store.params.system && store.params.system.trim()) {
    reqMsgs.push({ role: 'system', content: store.params.system.trim() })
  }
  for (const m of conv.messages) {
    if (m.role === 'error') continue
    reqMsgs.push({ role: m.role, content: buildContent(m, sent) })
  }
  return reqMsgs
}

/**
 * Append an assistant message and stream into it. Owns the streaming flag, the
 * abort handle, and the save() that has to happen even if the page is reloaded
 * mid-stream.
 */
async function streamCompletion(
  conv: Conversation,
  reqMsgs: Array<{
    role: string
    content: string | Array<Record<string, unknown>>
  }>,
): Promise<void> {
  const assistant: ChatMessage = {
    id: uid(),
    role: 'assistant',
    text: '',
    reasoning: '',
    ts: Date.now(),
  }
  conv.messages.push(assistant)
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
      (d) => {
        if (d.content) {
          if (ttft === undefined) ttft = performance.now() - t0
          live.text += d.content
        }
        if (d.reasoning) live.reasoning = (live.reasoning || '') + d.reasoning
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
  await streamCompletion(conv, buildRequest(conv, []))
}

export function stop(): void {
  if (abortCtrl) abortCtrl.abort()
}
