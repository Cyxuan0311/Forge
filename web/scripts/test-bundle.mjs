/**
 * Bundle test: mounts the REAL built asset in a DOM.
 *
 * Why a second test, when scripts/smoke.ts already renders every option
 * ----------------------------------------------------------------------
 * smoke.ts runs against the source module graph with ECharts' SSR painter, and
 * it cannot catch a failure that only exists in the bundled artifact. The bug
 * this was written for is exactly such a failure:
 *
 *   Uncaught Error: componentType " markLine" illegal
 *       at ... registerComponentView ... install ... use
 *
 * thrown at INSTALL time -- i.e. while the ECharts chunk was being imported --
 * and it did not reproduce under SSR at all. It is a property of how the
 * library got bundled, not of how the options were written. src/charts/*.ts
 * being renderable headlessly says nothing about whether `echarts.init` and the
 * component registrations survive Rollup's chunk splitting.
 *
 * (The bundling config that produced it, `manualChunks: { echarts: ['echarts'] }`
 *  combined with `import * as echarts from 'echarts'`, was demonstrably broken:
 *  it collapsed the library to a single mangled export and it also made
 *  `vite build --ssr` fail outright. Both are gone now, and this test is what
 *  keeps them gone.)
 *
 * So this file loads dist/assets/*.js -- the exact bytes forge-cli serves -- into
 * a jsdom document and mounts a chart. If the namespace import collapses, a
 * component is unregistered, or the renderer is missing, it fails here instead
 * of in a browser.
 *
 * Run with:  npm run test:bundle   (requires `npm run build` first)
 */
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { JSDOM } from 'jsdom'

const DIST = 'dist'
const ASSETS = join(DIST, 'assets')

let failures = 0
const fail = (msg) => {
  failures++
  console.log(`  FAIL  ${msg}`)
}
const ok = (msg) => console.log(`  ok    ${msg}`)

function expect(name, cond, detail = '') {
  cond ? ok(name) : fail(`${name}${detail ? ` -- ${detail}` : ''}`)
}

console.log('\n[bundle]')

// ---- 1. the assets the binary embeds must exist ----------------------------

let files
try {
  files = readdirSync(ASSETS)
} catch {
  console.error(`  FAIL  ${ASSETS} not found -- run \`npm run build\` first`)
  process.exit(1)
}
const html = readFileSync(join(DIST, 'index.html'), 'utf-8')
expect('dist/index.html exists', html.includes('<div id="app">'))
expect(
  'index.html references a relative ./assets/ path (works under /dashboard/)',
  /src="\.\/assets\//.test(html),
  html.match(/src="[^"]*"/)?.[0] ?? 'no script tag',
)

// Every asset index.html points at must actually be present.
const referenced = [...html.matchAll(/(?:src|href)="\.\/(assets\/[^"]+)"/g)].map((m) => m[1])
expect('index.html references at least one asset', referenced.length > 0)
for (const r of referenced) {
  expect(`referenced asset exists: ${r}`, files.includes(r.replace('assets/', '')))
}

// Bundle budget. The whole point of the modular ECharts build is that this
// stays small; it is embedded in forge-cli / forge-server, so a regression here
// is a regression in the binary size.
const totalBytes = files.reduce((n, f) => n + readFileSync(join(ASSETS, f)).length, 0)
const totalKb = Math.round(totalBytes / 1024)
expect(`bundle is under 900 KB (is ${totalKb} KB)`, totalKb < 900)
console.log(`        total dist/assets = ${totalKb} KB across ${files.length} file(s)`)

// ---- 2. load the built bundle into a DOM and mount a real chart ------------

const entryName = referenced.find((r) => r.endsWith('.js'))
if (!entryName) {
  fail('no JS entry chunk referenced by index.html')
  process.exit(1)
}

const dom = new JSDOM('<!doctype html><html><body><div id="app"></div></body></html>', {
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:9090/dashboard/index.html',
})
// Node >= 21 exposes some of these as read-only accessors on globalThis
// (navigator in particular), so plain assignment throws. defineProperty with
// configurable:true is the portable way to install them.
const define = (name, value) =>
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true })

define('window', dom.window)
define('document', dom.window.document)
define('navigator', dom.window.navigator)
// Vue's reactivity layer touches MutationObserver at module scope, and ECharts
// touches the rest of the DOM surface, so install the whole window surface
// rather than playing whack-a-mole one global per run.
for (const k of [
  'HTMLElement', 'HTMLCanvasElement', 'SVGElement', 'Element', 'Node', 'Image',
  'MutationObserver', 'ResizeObserver', 'IntersectionObserver', 'Text', 'Comment',
  'DocumentFragment', 'Event', 'CustomEvent', 'MouseEvent', 'KeyboardEvent',
  'getBoundingClientRect', 'matchMedia',
]) {
  if (dom.window[k] !== undefined) define(k, dom.window[k])
}
// jsdom implements neither of these, and both are called during a mount.
// jsdom has localStorage, but it is origin-scoped and this document's url IS a
// real origin, so it works as-is. Blob/URL.createObjectURL are absent and the
// export path needs them: a download is stubbed and captured rather than served.
const downloads = []
if (!globalThis.URL.createObjectURL) {
  dom.window.URL.createObjectURL = () => 'blob:stub'
}
dom.window.URL.revokeObjectURL = () => {}
const store = new Map()
const shimStorage = () => ({
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => void store.set(k, String(v)),
  removeItem: (k) => void store.delete(k),
  clear: () => store.clear(),
  key: (i) => [...store.keys()][i] ?? null,
  get length() {
    return store.size
  },
})
// localStorage is a getter-only accessor on jsdom's Window, so it has to be
// replaced rather than assigned.
for (const target of [dom.window, globalThis]) {
  Object.defineProperty(target, 'localStorage', {
    value: shimStorage(),
    configurable: true,
    writable: true,
  })
}
// Capture the anchor click instead of navigating. jsdom does not implement
// downloads at all, so without this an export would throw on a.click().
dom.window.HTMLAnchorElement.prototype.click = function () {
  if (this.download) downloads.push({ name: this.download, href: this.href })
}
if (!globalThis.ResizeObserver) {
  define(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  )
}
// jsdom implements neither matchMedia nor localStorage. The bundle reads
// `window.matchMedia` and bare `localStorage`, so both globals need the shim --
// and matches:false means the OS prefers LIGHT. That is deliberate: with a
// dark-preferring OS the default `system` mode already resolves to dark and the
// theme-flip assertions below could pass without anything having happened.
const fakeMq = (q) => ({
  matches: false,
  media: q,
  onchange: null,
  addEventListener() {},
  removeEventListener() {},
  addListener() {},
  removeListener() {},
  dispatchEvent: () => false,
})
dom.window.matchMedia = fakeMq
define('matchMedia', fakeMq)
define('getComputedStyle', dom.window.getComputedStyle.bind(dom.window))
define('requestAnimationFrame', (cb) => setTimeout(() => cb(Date.now()), 0))
define('cancelAnimationFrame', (id) => clearTimeout(id))
define('devicePixelRatio', 1)
// jsdom has no canvas backend; ECharts only needs getContext() to exist and
// return an object with the 2d methods it calls during a render.
const ctx2d = new Proxy(
  { canvas: { width: 800, height: 400 }, measureText: () => ({ width: 42 }) },
  { get: (t, k) => (k in t ? t[k] : () => undefined) },
)
dom.window.HTMLCanvasElement.prototype.getContext = () => ctx2d

// Anything Vue or ECharts logs as an error during import is a failure. Without
// this the test would happily pass while the app silently failed to mount --
// Vue catches errors in lifecycle hooks and reports them through the console
// rather than rejecting the module promise.
const consoleErrors = []
for (const level of ['error', 'warn']) {
  const original = dom.window.console[level].bind(dom.window.console)
  dom.window.console[level] = (...args) => {
    consoleErrors.push(`[${level}] ${args.map(String).join(' ')}`)
    original(...args)
  }
}

// Serve a realistic /api/metrics payload so the app renders its charts rather
// than the "waiting for the first sample" placeholder. Without data there are
// no <canvas> elements to assert on and the whole mount path is skipped.
const METRICS = {
  meta: {
    app: 'forge-cli', model: 'qwen3-4b', arch: 'qwen35',
    start_ms: 0, now_ms: Date.now(), uptime_s: 42, profiling: true,
  },
  engine: {
    kv_active_bytes: 52_428_800, kv_total_bytes: 46_137_344, kv_free_pages: -1,
    kv_pool_tokens: 0, kv_page_size: 0, sessions: 1, active_requests: 0,
    prefix_cache_hits: 7, prefix_cache_misses: 3, tiered_kv: false, paged_kv: false,
  },
  host: { vram_total_bytes: 6_442_529_280, vram_used_bytes: 4_100_000_000, rss_bytes: 2_000_000_000, cpu_pct: 37 },
  live: {
    active: false, tokens: 0, prompt_tokens: 0, elapsed_ms: 0,
    tok_per_s: 0, itl_ms: 0, start_ms: 0, streaming: false, session: 0,
  },
  totals: {
    requests: 12, errors: 0, rejected: 1, prompt_tokens: 640, completion_tokens: 1400,
    prefill_ms: 1200, decode_ms: 38_000, queue_ms: 220,
    lifetime_tok_per_s: 36.8, prefix_hit_rate: 0.7,
  },
  latency: {
    ttft_p50: { value: 42, count: 12 }, ttft_p90: { value: 88, count: 12 }, ttft_p99: { value: 140, count: 12 },
    itl_p50: { value: 27, count: 1400 }, itl_p90: { value: 31, count: 1400 }, itl_p99: { value: 40, count: 1400 },
  },
  series: {
    t: Array.from({ length: 40 }, (_, i) => Date.now() - (40 - i) * 1000),
    tok_per_s: Array.from({ length: 40 }, (_, i) => 34 + (i % 6)),
    ttft_ms: Array.from({ length: 40 }, (_, i) => 40 + (i % 9)),
    itl_p50_ms: Array.from({ length: 40 }, (_, i) => 27 + (i % 4)),
    vram_used: Array.from({ length: 40 }, (_, i) => 4_000_000_000 + i * 1_000_000),
    kv_active: Array.from({ length: 40 }, (_, i) => 50_000_000 + i * 200_000),
  },
  requests: Array.from({ length: 5 }, (_, i) => ({
    start_ms: Date.now() - i * 2000, end_ms: Date.now() - i * 2000 + 900,
    queue_ms: i === 0 ? 210 : 0, ttft_ms: 42 + i,
    prefill_ms: 50 + i, decode_ms: 900 + i * 10, total_ms: 1000 + i * 10,
    prompt_tokens: 50 + i, completion_tokens: 120 + i, prefix_reused: i % 2 ? 0 : 20,
    tok_per_s: 36 + (i % 4), itl_p50_ms: 27 + (i % 5), itl_p90_ms: 33 + (i % 5),
    spec_accept: -1, status: 'ok', streaming: i % 2 === 0, session: 0,
  })),
  operators: [
    { name: 'decode/forward', total_ms: 860, count: 12, min_ms: 60, max_ms: 80, avg_ms: 71 },
    { name: 'layer/ffn', total_ms: 713, count: 286, min_ms: 2, max_ms: 4, avg_ms: 2.5 },
  ],
  operators_available: true,
}

define('fetch', async (url) => {
  const href = String(url)
  if (href.includes('/api/metrics')) {
    return {
      ok: true, status: 200,
      json: async () => METRICS,
      text: async () => JSON.stringify(METRICS),
    }
  }
  throw new Error(`unexpected fetch in test: ${href}`)
})

try {
  // The entry chunk calls createApp(App).mount('#app') at module scope, so
  // importing it RUNS the dashboard. That is the whole point: a broken chart
  // only reveals itself during a real mount.
  await import(pathToFileURL(join(ASSETS, entryName.replace('assets/', ''))).href)
  ok(`built entry chunk imported and executed: ${entryName}`)
} catch (e) {
  fail(`importing the built chunk threw: ${e.message}`)
  console.log(`        ${(e.stack || '').split('\n')[1]?.trim() || ''}`)
}

// ---- 3. did it actually mount? ----------------------------------------------

// The first poll is a promise; give it a few microtask/macrotask turns to
// resolve and let Vue flush the render.
for (let i = 0; i < 20; i++) await new Promise((r) => setTimeout(r, 5))

const app = dom.window.document.getElementById('app')
expect(
  'the app mounted into #app',
  !!app && app.childNodes.length > 0,
  `#app has ${app ? app.childNodes.length : 'n/a'} child node(s)`,
)
// Each chart panel creates its own <canvas> via echarts.init(). If the echarts
// namespace import collapsed, init would be undefined and onMounted would throw --
// so a canvas count of zero means the page would render blank.
const canvases = dom.window.document.querySelectorAll('canvas').length
expect(
  `charts initialised (${canvases} canvas element(s))`,
  canvases > 0,
  'no canvas: echarts.init() never ran, so the page would be blank',
)
expect(
  'the header shows the model id from /api/metrics',
  (app?.textContent ?? '').includes('Model') || (app?.textContent ?? '').includes('model'),
)

// ---- 4. the header controls exist and actually do something -----------------
//
// Existence is asserted separately from behaviour on purpose: a button that
// renders but does nothing is the more likely regression, and a presence-only
// check would not notice it.

console.log('\n[controls]')
const q = (sel) => dom.window.document.querySelector(sel)
const themeBtn = q('[data-testid="theme-toggle"]')
const exportBtn = q('[data-testid="export-toggle"]')
const pauseBtn = q('[data-testid="pause-toggle"]')
expect('header has a theme control', !!themeBtn)
expect('header has an export control', !!exportBtn)
expect('header has a pause control', !!pauseBtn)

// ---- 5. theme switching is behavioural -------------------------------------

console.log('\n[theme]')
const root = dom.window.document.documentElement
// main.ts calls applyTheme() before mount, so this is already set from the
// stored mode (absent here) resolved through matchMedia -> light.
expect('a theme is applied before first paint', root.dataset.theme === 'light', `got ${root.dataset.theme}`)
expect('color-scheme follows the theme', root.style.colorScheme === 'light', `got ${root.style.colorScheme}`)
expect('theme control starts on system', themeBtn?.dataset.mode === 'system', `got ${themeBtn?.dataset.mode}`)

// The slider has one stop per mode. Selecting a stop must move BOTH the
// data-mode attribute and the persisted mode, and both must agree with the
// resolved colour on <html> (system re-resolves from matchMedia -> light).
const clickStop = (mode) => {
  const stop = q(`[data-testid="theme-stop-${mode}"]`)
  expect(`slider has a ${mode} stop`, !!stop, 'missing stop')
  stop?.click()
}
const wantTheme = (mode) => (mode === 'system' ? 'light' : mode)
for (const mode of ['light', 'dark', 'system']) {
  clickStop(mode)
  await new Promise((r) => setTimeout(r, 5))
  expect(
    `selecting ${mode} moves the slider`,
    themeBtn.dataset.mode === mode,
    `data-mode=${themeBtn.dataset.mode}`,
  )
  expect(`data-theme follows ${mode}`, root.dataset.theme === wantTheme(mode), `got ${root.dataset.theme}`)
  expect(
    `${mode} is persisted`,
    dom.window.localStorage.getItem('forge.dashboard.theme') === mode,
    `got ${dom.window.localStorage.getItem('forge.dashboard.theme')}`,
  )
}

// A light theme must not leave any hardcoded dark literal behind in the CSS the
// bundle shipped, which is what would show as dark panels on a light page.
const cssName = files.find((f) => f.endsWith('.css'))
const css = readFileSync(join(ASSETS, cssName), 'utf-8')
expect('css defines a light theme block', css.includes('[data-theme=light]'), cssName)
expect('css has no hardcoded dark background left', !/background:\s*#0f1115/.test(css.replace(/--bg:[^;]+;/g, '')))

// ---- 6. export is behavioural ----------------------------------------------

console.log('\n[export]')
// Disabled until the first poll resolves; data is already in by now.
expect('export control is enabled once data arrives', exportBtn && !exportBtn.disabled)
exportBtn.click()
await new Promise((r) => setTimeout(r, 5))
const items = [...dom.window.document.querySelectorAll('[data-testid^="export-"]')]
  .filter((el) => el.dataset.testid !== 'export-toggle' && el.dataset.testid !== 'export-status')
expect('the export menu offers all four formats', items.length === 4, `got ${items.length}`)
for (const f of ['json', 'csv', 'markdown', 'prometheus']) {
  expect(`export menu offers ${f}`, items.some((el) => el.dataset.testid === `export-${f}`))
}

// Pick Prometheus: a download, so the anchor stub above captures it.
items.find((el) => el.dataset.testid === 'export-prometheus').click()
await new Promise((r) => setTimeout(r, 20))
expect('choosing a format triggers a download', downloads.length === 1, `${downloads.length} download(s)`)
const dl = downloads[0]
expect('the download is named sanely', /^forge-[a-z0-9-]+\.prom$/.test(dl?.name ?? ''), dl?.name)
expect('the export menu closes after choosing', !q('[data-testid="export-csv"]'))

// Markdown goes to the clipboard rather than to a file.
//
// BOTH paths are installed, and only the modern one is expected to be used:
// navigator.clipboard works when `window.isSecureContext` is true, which it is
// for the dashboard's default http://127.0.0.1 origin. So the execCommand
// fallback below must NOT fire here -- asserting that keeps the fallback from
// silently taking over on a real secure origin.
let copiedFallback = false
dom.window.document.execCommand = () => {
  copiedFallback = true
  return true
}
let clipboardText = null
const clipboardShim = {
  writeText: async (t) => {
    clipboardText = t
  },
}
// navigator.clipboard is a getter-only accessor on jsdom's Window.
Object.defineProperty(dom.window.navigator, 'clipboard', {
  value: clipboardShim,
  configurable: true,
  writable: true,
})
Object.defineProperty(dom.window.window, 'isSecureContext', {
  value: true,
  configurable: true,
  writable: true,
})
// Bare `navigator` in the bundle resolves to globalThis.navigator, which was
// defined above from dom.window.navigator -- the same object, so the shim above
// is visible either way. Assert it rather than assume.
exportBtn.click()
await new Promise((r) => setTimeout(r, 5))
q('[data-testid="export-markdown"]').click()
await new Promise((r) => setTimeout(r, 20))
const status = q('[data-testid="export-status"]')
expect('markdown reports a result', !!status, 'no status element')
expect('markdown did not become a download', downloads.length === 1, `${downloads.length} downloads`)
expect('markdown reached the clipboard', clipboardText !== null, 'writeText was never called')
expect(
  'the clipboard payload is the markdown report',
  typeof clipboardText === 'string' && clipboardText.startsWith('# Forge dashboard snapshot'),
  String(clipboardText).slice(0, 40),
)
expect('the insecure-context fallback was not used on a secure origin', !copiedFallback)
expect('the status confirms the copy', (status?.textContent ?? '').includes('copied'), status?.textContent)

// ---- 7. manual pause is behavioural -----------------------------------------

console.log('\n[pause]')
expect(
  'pause control starts as Pause',
  (pauseBtn?.textContent ?? '').trim() === 'Pause',
  pauseBtn?.textContent ?? 'missing',
)
expect(
  'no manual-paused indicator before click',
  !(app?.textContent ?? '').includes('paused (manual)'),
)
pauseBtn?.click()
await new Promise((r) => setTimeout(r, 5))
expect(
  'pause control flips to Resume',
  (pauseBtn?.textContent ?? '').trim() === 'Resume',
  pauseBtn?.textContent ?? 'missing',
)
expect(
  'manual-paused indicator appears',
  (app?.textContent ?? '').includes('paused (manual)'),
)
pauseBtn?.click()
await new Promise((r) => setTimeout(r, 5))
expect(
  'pause control flips back to Pause',
  (pauseBtn?.textContent ?? '').trim() === 'Pause',
  pauseBtn?.textContent ?? 'missing',
)
expect(
  'manual-paused indicator clears',
  !(app?.textContent ?? '').includes('paused (manual)'),
)

if (consoleErrors.length > 0) {
  for (const e of consoleErrors.slice(0, 8)) fail(`console during mount: ${e}`)
  if (consoleErrors.length > 8) console.log(`        ...and ${consoleErrors.length - 8} more`)
} else {
  ok('no errors logged to the console while mounting')
}

console.log('')
if (failures > 0) {
  console.log(`bundle test FAILED: ${failures} problem(s)`)
  process.exit(1)
}
console.log('bundle test passed: the embedded assets are importable and correctly sized')

// The mounted app keeps its 1 s /api/metrics poll and ECharts' rAF loop alive,
// so the process would otherwise hang after the assertions finish.
process.exit(0)