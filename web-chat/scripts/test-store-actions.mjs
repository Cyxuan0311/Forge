// Behaviour checks for the per-message actions in src/store.ts.
//
// These touch thread surgery (delete / regenerate), which is easy to get subtly
// wrong and impossible to eyeball in the UI: a delete that leaves an orphan
// answer in the history sends an assistant message with no question in front of
// it on the next request. So they run here against the real store module with
// fetch stubbed out, and assert on the actual request payloads.
//
// Run with: node scripts/test-store-actions.mjs
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { rmSync } from 'node:fs'
import { build } from 'esbuild'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '..')
const tmp = join(here, '.store-actions.bundle.mjs')

// Bundle the real store module (vue resolves from node_modules). The browser
// globals it touches are stubbed below, before the import runs.
await build({
  entryPoints: [join(root, 'src', 'store.ts')],
  outfile: tmp,
  bundle: true,
  format: 'esm',
  platform: 'node',
  logLevel: 'silent',
})
process.on('exit', () => {
  try {
    rmSync(tmp, { force: true })
  } catch {}
})

// ---- minimal browser surface the store touches --------------------------
const persisted = new Map()
globalThis.localStorage = {
  getItem: (k) => (persisted.has(k) ? persisted.get(k) : null),
  setItem: (k, v) => persisted.set(k, String(v)),
}
globalThis.window = {
  crypto: { randomUUID: () => 'id-' + Math.random().toString(36).slice(2) },
  isSecureContext: false,
}
globalThis.document = {
  createElement: () => ({ style: {}, select() {}, value: '' }),
  body: { appendChild() {}, removeChild() {} },
  execCommand: () => true,
}

// addFiles() reads text via FileReader. The fakes below carry their payload on
// `__text`/`__dataUrl`, so the stub just echoes it back and fires onload (which
// addFiles has already assigned before calling readAsText/readAsDataURL).
globalThis.FileReader = class {
  constructor() {
    this.result = ''
  }
  readAsText(blob) {
    this.result = blob && blob.__text !== undefined ? blob.__text : ''
    if (this.onload) this.onload()
  }
  readAsDataURL(blob) {
    this.result = blob && blob.__dataUrl !== undefined ? blob.__dataUrl : ''
    if (this.onload) this.onload()
  }
}

const calls = []
let replyText = ''
// What the server would report in `usage` for the current reply.
let serverUsage = null
globalThis.fetch = async (url, init) => {
  const req = JSON.parse(init.body)
  calls.push({ url, body: req })
  if (req.stream === true) {
    // One content frame, then the usage-only tail chunk (choices: []) that
    // stream_options.include_usage asks for, then [DONE].
    const frames = [
      { choices: [{ delta: { content: replyText } }] },
      ...(req.stream_options?.include_usage && serverUsage
        ? [{ choices: [], usage: serverUsage }]
        : []),
    ]
    const sse =
      frames.map((f) => 'data: ' + JSON.stringify(f) + '\n\n').join('') + 'data: [DONE]\n\n'
    return {
      ok: true,
      status: 200,
      body: {
        getReader() {
          let sent = false
          return {
            read: async () => {
              if (sent) return { done: true }
              sent = true
              return { done: false, value: new TextEncoder().encode(sse) }
            },
          }
        },
      },
    }
  }
  const payload = { choices: [{ message: { content: replyText } }] }
  if (serverUsage) payload.usage = serverUsage
  return {
    ok: true,
    status: 200,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
    body: null,
  }
}

const mod = await import(tmp)
const { store, newChat, send, deleteMessage, regenerate, copyText, uid, addFiles } = mod

let failed = 0
function check(name, cond, got) {
  if (cond) console.log('  ok   ' + name)
  else {
    failed++
    console.log('  FAIL ' + name + (got === undefined ? '' : '\n       got: ' + got))
  }
}
const eq = (name, a, b) =>
  check(
    name,
    JSON.stringify(a) === JSON.stringify(b),
    JSON.stringify(a) + '  expected  ' + JSON.stringify(b),
  )

store.connection.baseUrl = ''
store.connection.model = 'test-model'
store.params.stream = false
store.params.system = ''

/** Fresh conversation whose turns are [ask, answer] pairs. */
async function thread(...turns) {
  newChat()
  const conv = store.conversations.find((c) => c.id === store.activeId)
  for (const [ask, answer] of turns) {
    replyText = answer
    store.draft = ask
    await send()
  }
  return conv
}
const texts = (conv) => conv.messages.map((m) => m.text)
const roles = (conv) => conv.messages.map((m) => m.role)
const lastRequest = () => calls[calls.length - 1].body

console.log('=== send() still works after the refactor ===')
{
  const conv = await thread(['first question', 'an answer'])
  eq('two messages after one turn', roles(conv), ['user', 'assistant'])
  eq('assistant text streamed in', conv.messages[1].text, 'an answer')
  eq('one request issued', calls.length, 1)
  eq('request carried the question', lastRequest().messages[0].content, 'first question')
  eq('title taken from the question', conv.title, 'first question')
}

console.log('=== delete ===')
{
  const conv = await thread(['q1', 'a1'], ['q2', 'a2'])
  const convId = conv.id
  const [q1, a1] = conv.messages
  eq('four messages after two turns', conv.messages.length, 4)

  deleteMessage(convId, a1.id)
  eq('deleting an answer leaves its question', texts(conv), ['q1', 'q2', 'a2'])

  deleteMessage(convId, q1.id)
  // The orphaned answer to the deleted question goes with it; the newer turn stays.
  eq('deleting a question also drops its answer', texts(conv), ['q2', 'a2'])
  eq('surviving roles are still a valid pair', roles(conv), ['user', 'assistant'])
}

console.log('=== regenerate (target = the answer) ===')
{
  const conv = await thread(['q1', 'a1'])
  const before = calls.length
  replyText = 'a1-again'
  await regenerate(conv.messages[1].id)
  eq('message count unchanged (answer replaced)', texts(conv), ['q1', 'a1-again'])
  eq('exactly one request issued', calls.length - before, 1)
  eq('replayed the question as a user message', lastRequest().messages, [
    { role: 'user', content: 'q1' },
  ])
}

console.log('=== regenerate on a user message keeps the prompt ===')
{
  const conv = await thread(['keep me', 'a1'])
  replyText = 'a1-again'
  await regenerate(conv.messages[0].id)
  eq('the user prompt survived', texts(conv), ['keep me', 'a1-again'])
  eq(
    'the request was not empty',
    lastRequest().messages.map((m) => m.content),
    ['keep me'],
  )
}

console.log('=== regenerate truncates the branch after it ===')
{
  const conv = await thread(['q1', 'a1'], ['q2', 'a2'], ['q3', 'a3'])
  eq('six messages after three turns', texts(conv), ['q1', 'a1', 'q2', 'a2', 'q3', 'a3'])
  replyText = 'a1-again'
  await regenerate(conv.messages[1].id)
  eq('later turns are gone', texts(conv), ['q1', 'a1-again'])
  eq(
    'the replayed request carries the turns up to that point',
    lastRequest().messages.map((m) => m.content),
    ['q1'],
  )
}

console.log('=== regenerate re-sends images from the message ===')
{
  newChat()
  const conv = store.conversations.find((c) => c.id === store.activeId)
  store.draft = 'what is this'
  store.pending.push({
    id: uid(),
    name: 'lion.png',
    kind: 'image',
    size: 10,
    dataUrl: 'data:image/png;base64,AAA',
  })
  replyText = 'a lion'
  await send()
  eq('image part reached the request', lastRequest().messages[0].content[1].type, 'image_url')
  replyText = 'still a lion'
  await regenerate(conv.messages[1].id)
  eq(
    'regenerated request still carries the image',
    lastRequest().messages[0].content[1].image_url.url,
    'data:image/png;base64,AAA',
  )
}

console.log('=== file attachment with no typed text still ships ===')
{
  newChat()
  const conv = store.conversations.find((c) => c.id === store.activeId)
  // Simulate a small text file that addFiles() has already read into text.
  store.pending.push({
    id: uid(),
    name: 'notes.txt',
    kind: 'file',
    size: 11,
    text: 'hello world',
  })
  store.draft = '' // no typed text: the classic "attach and send" flow
  replyText = 'got it'
  await send()
  const content = lastRequest().messages[0].content
  check('content is an array (not collapsed to empty)', Array.isArray(content), JSON.stringify(content))
  check(
    'file text reached the request',
    Array.isArray(content) &&
      content.some((p) => typeof p.text === 'string' && p.text.includes('<file name="notes.txt">')),
    JSON.stringify(content),
  )
  eq('user message recorded the file', conv.messages[0].files, [{ name: 'notes.txt', size: 11 }])
}

console.log('=== addFiles inlines text past the old 64 KiB cap ===')
{
  newChat()
  // A .md larger than the old 64 KiB limit must still inline its text.
  addFiles([{ name: 'notes.md', size: 200 * 1024, type: 'text/markdown', __text: 'hello notes' }])
  check('file staged', store.pending.length === 1)
  store.draft = 'describe this file'
  replyText = 'ok'
  await send()
  const content = lastRequest().messages[0].content
  check(
    'oversized .md content reached the request',
    Array.isArray(content) &&
      content.some(
        (p) =>
          typeof p.text === 'string' &&
          p.text.includes('<file name="notes.md">') &&
          p.text.includes('hello notes'),
      ),
    JSON.stringify(content),
  )

  // A binary file with no readable text still degrades to a filename note.
  newChat()
  addFiles([{ name: 'paper.pdf', size: 4096, type: 'application/pdf', __text: 'PDFBYTES' }])
  store.draft = 'read this'
  replyText = 'ok'
  await send()
  const c2 = lastRequest().messages[0].content
  check(
    'binary file becomes a placeholder note',
    Array.isArray(c2) &&
      c2.some(
        (p) => typeof p.text === 'string' && p.text.includes('[attached file: paper.pdf (4096 bytes)]'),
      ),
    JSON.stringify(c2),
  )
}

console.log('=== per-turn stats ===')
{
  store.params.stream = true
  newChat()
  const conv = store.conversations.find((c) => c.id === store.activeId)
  replyText = 'a lion'
  serverUsage = {
    prompt_tokens: 599,
    completion_tokens: 10,
    total_tokens: 609,
    prefill_ms: 1694.8,
    decode_ms: 350.8,
    tokens_per_sec: 28.5,
  }
  store.draft = 'what is this'
  await send()
  const st = conv.messages[1].stats
  check('stats were recorded', !!st, JSON.stringify(conv.messages[1]))
  eq('prompt tokens from the server', st.prompt_tokens, 599)
  eq('completion tokens from the server', st.completion_tokens, 10)
  eq('prefill ms from the server', Math.round(st.prefill_ms), 1695)
  eq('decode ms from the server', Math.round(st.decode_ms), 351)
  eq('tok/s from the server', Math.round(st.tokens_per_sec), 29)
  check('client measured ttft', typeof st.ttft_ms === 'number' && st.ttft_ms >= 0, st.ttft_ms)
  check('client measured wall time', typeof st.wall_ms === 'number' && st.wall_ms >= 0, st.wall_ms)
  eq('streaming asked for the usage chunk', lastRequest().stream_options, { include_usage: true })
  eq(
    'stats are persisted with the conversation',
    JSON.parse(persisted.get('forge.chat.conversations'))[0].messages[1].stats.completion_tokens,
    10,
  )
}

console.log('=== stats survive a server that sends no usage ===')
{
  serverUsage = null
  newChat()
  const conv = store.conversations.find((c) => c.id === store.activeId)
  replyText = 'no timings here'
  store.draft = 'hello'
  await send()
  const st = conv.messages[1].stats
  check(
    'client-only stats still recorded',
    !!st && typeof st.wall_ms === 'number',
    JSON.stringify(st),
  )
  eq('no invented token counts', st.prompt_tokens, undefined)
  eq('no invented tok/s', st.tokens_per_sec, undefined)
  check('ttft still measured', typeof st.ttft_ms === 'number')
}

console.log('=== copy ===')
eq('copyText succeeds on the execCommand path', await copyText('hello'), true)
eq('copyText refuses an empty string', await copyText(''), false)

if (failed) {
  console.log('\nstore-actions: FAIL (' + failed + ' check(s))')
  process.exit(1)
}
console.log('\nstore-actions: PASS')
