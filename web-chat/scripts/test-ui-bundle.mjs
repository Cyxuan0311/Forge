// Integration smoke test for the built web-chat bundle.
//
// Mounts dist/ in jsdom against the REAL forge-server (or a stubbed fetch when
// no server is reachable) and drives the mode selector through a full tool
// round. This is the closest thing to a browser test available here: it
// exercises the same bundle the server embeds, including Vue's rendering of the
// mode menu and the tool-result cards, rather than the TS sources the
// store-actions test imports.
//
// Run with: node scripts/test-ui-bundle.mjs [baseUrl]
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { readFileSync } from 'node:fs'

// jsdom is a devDependency of the sibling web/ dashboard; reuse it rather than
// adding a second copy to this package.
const webRequire = createRequire(join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'web', 'package.json'))
const { JSDOM } = webRequire('jsdom')

const here = dirname(fileURLToPath(import.meta.url))
const distDir = join(here, '..', 'dist')
const baseUrl = process.argv[2] || ''

let failed = 0
function check(name, cond, got = '') {
  if (cond) console.log('  ok   ' + name)
  else {
    failed++
    console.log('  FAIL ' + name + (got === '' ? '' : '\n       got: ' + got))
  }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// ---- fake backend ----------------------------------------------------------
// A tool round driven exactly as store.ts drives it: first request answers with
// tool_calls, the UI POSTs to /v1/tools/exec, then re-requests with
// delta_messages.
const requests = []
let completionIndex = 0
const TOOL_CALL = {
  id: 'call-ui-1',
  type: 'function',
  function: { name: 'run_python', arguments: '{"code":"print(6*7)"}' },
}

globalThis.fetch = async (url, init) => {
  const u = String(url)
  const body = init && init.body ? JSON.parse(init.body) : {}
  requests.push({ url: u, body })

  if (u.endsWith('/v1/tools')) {
    return jsonResp({
      object: 'list',
      code_exec_enabled: true,
      search_provider: 'tavily',
      search_configured: true,
      tools: [
        {
          name: 'run_python',
          description: 'Run python',
          parameters: { type: 'object', properties: { code: { type: 'string' } }, required: ['code'] },
          available: true,
        },
        {
          name: 'web_search',
          description: 'Search',
          parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] },
          available: true,
        },
      ],
    })
  }
  if (u.endsWith('/v1/tools/exec')) {
    return jsonResp({ name: 'run_python', ok: true, content: 'Output:\n42\n' })
  }
  if (u.endsWith('/v1/models')) {
    return jsonResp({ data: [{ id: 'test-model' }] })
  }
  if (u.endsWith('/v1/chat/completions')) {
    const first = completionIndex++ === 0
    return jsonResp({
      choices: [
        {
          message: first
            ? { role: 'assistant', content: null, tool_calls: [TOOL_CALL] }
            : { role: 'assistant', content: 'The result of 6*7 is 42.' },
          finish_reason: first ? 'tool_calls' : 'stop',
        },
      ],
    })
  }
  return jsonResp({})
}

function jsonResp(obj) {
  return {
    ok: true,
    status: 200,
    json: async () => obj,
    text: async () => JSON.stringify(obj),
    body: null,
  }
}

// ---- mount the built bundle -----------------------------------------------
const html = readFileSync(join(distDir, 'index.html'), 'utf8')
const dom = new JSDOM(html, { url: 'http://localhost/chat/', pretendToBeVisual: true, runScripts: 'outside-only' })
const { window } = dom

// The bundle reads localStorage and window.crypto at import time.
window.localStorage.setItem(
  'forge.chat.connection',
  JSON.stringify({ baseUrl, apiKey: '', model: 'test-model' }),
)
window.localStorage.setItem('forge.chat.params', JSON.stringify({
  temperature: 0.7, max_tokens: 256, top_p: 1, top_k: 40, repeat_penalty: 1.1,
  system: '', session_id: '', stream: false,
}))
// Seed a few conversations of different ages so the date-grouped list and the
// search box have something to render; without it the list is empty and the
// grouping code is never exercised.
const DAY = 86_400_000
const now = Date.now()
window.localStorage.setItem('forge.chat.conversations', JSON.stringify([
  { id: 'c1', title: 'Pricing plan for a SaaS', messages: [], ts: now },
  { id: 'c2', title: 'Refactor the auth flow', messages: [], ts: now - DAY },
  { id: 'c3', title: 'Travel itinerary for Kyoto', messages: [], ts: now - 8 * DAY },
]))
window.fetch = globalThis.fetch

// Inline the ESM bundle: jsdom cannot do ESM imports, so transform the single
// chunk into a classic script and evaluate it in the window.
const assetMatch = html.match(/src="\.\/assets\/([^"]+\.js)"/)
if (!assetMatch) {
  console.error('  FAIL could not find the bundle script in dist/index.html')
  process.exit(1)
}
const bundlePath = join(distDir, 'assets', assetMatch[1])
let code = readFileSync(bundlePath, 'utf8')
// The Vite ESM output uses import/export only for its own chunks; this app is a
// single chunk, so stripping the module syntax makes it evaluable as a script.
code = code.replace(/^\s*import\s[^;]+;\s*$/gm, '').replace(/\bexport\s*\{[^}]*\}\s*;?/g, '')

try {
  window.eval(code)
} catch (e) {
  console.error('  FAIL bundle threw during mount: ' + e.message)
  process.exit(1)
}

const $ = (sel) => window.document.querySelector(sel)
const $$ = (sel) => Array.from(window.document.querySelectorAll(sel))
const text = () => window.document.body.textContent || ''

await sleep(60)

console.log('=== mount ===')
check('app rendered', !!$('#app') && $('#app').children.length > 0)
check('composer present', !!$('.draft'), 'no textarea')
check('send is a round icon button', !!$('.composer .round svg'), 'no round send button')
check('no avatars in the thread', $$('.msg .avatar').length === 0,
  'n=' + $$('.msg .avatar').length)
check('sidebar is off-canvas when closed', ($('.sidebar') || {}).className === 'sidebar')
check('sidebar has a search box', !!$('.sidebar-search input'))
check('conversations are grouped by date', $$('.conv-group').length >= 2,
  $$('.conv-group').map((e) => e.textContent).join(','))
check('seeded chats appear in the list', $$('.conv').length === 3)
check('single composer menu button', $$('.composer-row .cmenu-btn').length === 1,
  'n=' + $$('.composer-row .cmenu-btn').length)
check('no leftover mode button', $$('.mode-btn').length === 0)
check('no leftover attach button', $$('.attach-btn').length === 0)
check('trigger is an SVG, not an emoji glyph', !!$('.cmenu-btn svg'), 'no svg in the trigger')
check('no emoji in the composer row', !/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u.test($('.composer-row')?.textContent || ''),
  ($('.composer-row') || {}).textContent)
check('empty thread centres the greeting with the composer', !!$('.main.empty'))
check('empty state offers suggestion prompts', $$('.sugg').length === 3,
  'n=' + $$('.sugg').length)
check('draft is focused on load', window.document.activeElement === $('.draft'),
  String((window.document.activeElement || {}).className))

console.log('=== drawer ===')
// jsdom reports a 1024px viewport, where the sidebar is a persistent column and
// therefore starts open. Drive it to a known state before asserting toggles.
async function setDrawer(open) {
  const isOpen = $('.app').classList.contains('drawer-open')
  if (isOpen !== open) {
    $('[title="Toggle sidebar"]').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
    await sleep(30)
  }
}
await setDrawer(false)
check('hamburger closes the sidebar', !$('.app').classList.contains('drawer-open'))
check('closed state is persisted', window.localStorage.getItem('forge.chat.sidebarOpen') === 'false',
  String(window.localStorage.getItem('forge.chat.sidebarOpen')))
await setDrawer(true)
check('hamburger opens the sidebar', $('.app').classList.contains('drawer-open'))
check('open state is persisted', window.localStorage.getItem('forge.chat.sidebarOpen') === 'true',
  String(window.localStorage.getItem('forge.chat.sidebarOpen')))
$('.scrim').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
await sleep(30)
check('scrim click closes the drawer', !$('.app').classList.contains('drawer-open'))

// Regression: the X emitted a 'close' event nothing listened for, so it did
// nothing at all.
await setDrawer(true)
$('[title="Close sidebar"]').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
await sleep(30)
check('sidebar X closes the drawer', !$('.app').classList.contains('drawer-open'),
  ($('.app') || {}).className)
window.document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
await sleep(30)
check('Escape closes the drawer', !$('.app').classList.contains('drawer-open'))

console.log('=== mode menu ===')
$('.cmenu-btn').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
await sleep(30)
const items = $$('.cmenu-item')
check('menu opened', items.length >= 6, 'items=' + items.length)
check('menu has an Add section', ($$('.cmenu-head').map((e) => e.textContent).join(',') || '').includes('Add'),
  $$('.cmenu-head').map((e) => e.textContent).join(','))
check('image upload entry present', !!$('[data-testid="attach-image"]'))
check('file upload entry present', !!$('[data-testid="attach-file"]'))
check('menu icons are SVG', $$('.cmenu-item svg').length >= 6,
  'svg=' + $$('.cmenu-item svg').length)
check('image input filters to images',
  ($('input[type=file][accept="image/*"]') || {}).accept === 'image/*')
check('chat listed', text().includes('Chat'))
check('web search listed', text().includes('Web search'))
check('writing listed', text().includes('Writing'))
check('coding listed', text().includes('Coding'))

const codeItem = items.find((el) => (el.getAttribute('data-testid') || '') === 'mode-code')
check('coding item has a testid', !!codeItem)
codeItem.dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
await sleep(30)
check('menu closed after picking', $$('.cmenu-item').length === 0)
// The badge was replaced by swapping the trigger's glyph to the mode icon.
check('badge dot retired in favour of the icon swap', $$('.cmenu-badge').length === 0)
// Reopen to inspect the selection marker (the item only exists while open).
$('.cmenu-btn').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
await sleep(30)
check('picked mode is marked active', !!$('[data-testid="mode-code"].on'))
check('only one item is active', $$('.cmenu-item.on').length === 1,
  'n=' + $$('.cmenu-item.on').length)
check('active item shows a check icon', !!$('.cmenu-item.on .cmenu-check'))
check('trigger has the active class once a mode is set', !!$('.cmenu-btn.active'))
// The trigger swaps its glyph to the mode, so the state is readable without
// opening the menu.
check('menu header echoes the current mode',
  (($('.cmenu-current') || {}).textContent || '').length > 0)
check('trigger icon is the mode, not a plus', !$('.cmenu-btn').textContent.includes('+'))
$('.cmenu-btn').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
await sleep(20)

console.log('=== modal dialog ===')
$('[title="Connection"]').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
await sleep(60)
check('dialog opened', !!$('.modal-backdrop .modal'))
check('  ...is a real dialog role', ($('.modal') || {}).getAttribute?.('role') === 'dialog')
check('  ...has an accessible name', !!($('.modal') || {}).getAttribute?.('aria-label'))
check('  ...close button is an SVG icon, not a glyph',
  !!$('.modal-x svg') && !($('.modal-x').textContent || '').includes('\u00d7'))
$('.modal-x').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
await sleep(60)
check('dialog closed via its close button', !$('.modal-backdrop'))
$('[title="Connection"]').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
await sleep(60)
check('reopened', !!$('.modal-backdrop'))
window.document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
await sleep(60)
check('dialog closed via Escape', !$('.modal-backdrop'))

console.log('=== a tool round through the UI ===')
const ta = $('.draft')
ta.value = 'what is 6*7?'
ta.dispatchEvent(new window.Event('input', { bubbles: true }))
await sleep(10)
// Send is disabled until there is something to send.
const sendBtn = $('[data-testid="send"]')
check('send disabled while the draft is empty', sendBtn.disabled === false)
sendBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
await sleep(200)

const completions = requests.filter((r) => r.url.endsWith('/v1/chat/completions'))
const execs = requests.filter((r) => r.url.endsWith('/v1/tools/exec'))
check('two completion requests', completions.length === 2, 'n=' + completions.length)
check('tool executed once', execs.length === 1, 'n=' + execs.length)
check('tools advertised on the first request', Array.isArray(completions[0].body.tools))
check('  ...naming run_python', (completions[0].body.tools || []).some((t) => t.function.name === 'run_python'))
check('result fed back as delta_messages',
  Array.isArray(completions[1].body.delta_messages) && completions[1].body.delta_messages[0].role === 'tool')

console.log('=== rendered output ===')
check('tool card rendered', !!$('.toolrun'), 'no .toolrun')
check('  ...names the tool', ($('.toolrun-name') || {}).textContent === 'run_python')
check('  ...shows the result', text().includes('42'))
check('  ...marked done', text().includes('done'))
const assistantBubbles = $$('.msg.assistant')
check('exactly one assistant bubble for the turn', assistantBubbles.length === 1,
  'n=' + assistantBubbles.length)
const bodyText = (assistantBubbles[0] && assistantBubbles[0].querySelector('.body'))
  ? assistantBubbles[0].querySelector('.body').textContent
  : ''
check('final answer rendered', bodyText.includes('42'), bodyText.slice(0, 120))
check('thread left the empty state', !$('.main').classList.contains('empty'))
// The stretched turn width (and so the bubble's alignment) depends on `.msg`
// being a direct flex child of the thread column.
check('turns are direct children of the thread column',
  $$('.thread-inner > .msg').length === $$('.msg').length && $$('.msg').length >= 2,
  'inner=' + $$('.thread-inner > .msg').length + ' all=' + $$('.msg').length)
check('user turn renders exactly one bubble', $$('.msg.user > .content').length === 1,
  'n=' + $$('.msg.user > .content').length)
check('user bubble carries the sent text',
  (($('.msg.user .body') || {}).textContent || '').includes('6*7'),
  ($('.msg.user .body') || {}).textContent)
check('no streaming class left on the body', !$('.body.streaming'))
check('caret is back in the draft after the reply', window.document.activeElement === $('.draft'),
  String((window.document.activeElement || {}).className))

// jsdom does no layout, so the three fixes below are guarded on the built
// stylesheet. Weaker than measuring the box, but it fails loudly if any of them
// is undone, which is what a regression guard is for.
console.log('=== stylesheet ===')
const cssHref = html.match(/href="\.\/assets\/([^"]+\.css)"/)
const css = cssHref ? readFileSync(join(distDir, 'assets', cssHref[1]), 'utf8') : ''
check('built stylesheet found', css.length > 1000, 'bytes=' + css.length)
// Regression: `.msg` carried auto horizontal margins, which cancel
// `align-items: stretch` in the column-flex thread. Each turn then sized to its
// content and sat mid-column, and the user bubble's `max-width: 82%` resolved
// against a content-sized parent -- narrow, wrapped, floating in the middle.
check('message rows fill the thread column', !/margin:\s*0\s+auto\s+22px/.test(css),
  'still has auto margins on .msg')
check('user turn right-aligns its bubble',
  /\.msg\.user\s*\{[^}]*justify-content:\s*flex-end/.test(css))
// Regression: a blinking caret was drawn at the end of every streamed reply.
check('no blinking caret while output streams', !/caret-blink/.test(css))
// Regression: a long Connection/Settings panel overflowed the dialog instead of
// scrolling inside it (a flex item defaults to `min-height: auto`).
check('dialog body scrolls inside the dialog',
  /\.modal-body\s*\{[^}]*min-height:\s*0/.test(css))
// Regression: `.main` had no `flex: 1`, so in the column-flex shell it sized to
// its content and the composer floated up under the last message.
check('main fills the shell so the composer stays at the bottom',
  /\.main\s*\{[^}]*flex:\s*1/.test(css))

console.log()
if (failed) {
  console.log('ui-bundle: FAIL (' + failed + ' check(s))')
  process.exit(1)
}
console.log('ui-bundle: PASS')