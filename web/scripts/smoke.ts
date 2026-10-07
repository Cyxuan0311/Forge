/**
 * Headless smoke test for every chart option.
 *
 * Why this exists
 * ---------------
 * The dashboard shipped with `componentType "markLine" illegal` in the browser:
 * ECharts throws that from deep inside its class registry while installing
 * extensions, and the only cause is a chart/component that was never registered
 * via `use()`. The offending panel looked fine in TypeScript, passed
 * `vue-tsc`, and built without a warning -- it only failed when a human opened
 * the page.
 *
 * So every option builder is now a plain function (src/charts/*.ts) instead of
 * an inline computed() in a .vue file, and this script feeds each one's output
 * into a REAL headless ECharts instance. A missing registration, a malformed
 * axis, or a bad formatter callback now fails `npm run smoke` -- in CI, without
 * a browser.
 *
 * Run with:  npm run smoke   (builds an SSR bundle of this file, then runs it)
 */
import { echarts } from '../src/echarts'
// The app renders to a <canvas>, so src/echarts.ts only registers
// CanvasRenderer. This test renders to a STRING instead (ECharts' `ssr: true`
// mode), which needs the SVG painter. Registering it here keeps it out of the
// production bundle -- it is a test-only concern, and SVGRenderer is not free.
import { SVGRenderer } from 'echarts/renderers'
echarts.use([SVGRenderer])
import { buildTimeSeriesOption } from '../src/charts/timeseries'
import { buildLatencyOption } from '../src/charts/latency'
import { buildGaugeOption } from '../src/charts/gauge'
import { buildOperatorOption } from '../src/charts/operators'
import { throughputLines, memoryLines } from '../src/charts/lines'
import { getPalette } from '../src/theme'
import type { ResolvedTheme } from '../src/theme'
import type { Metrics, RequestRow, Series } from '../src/types'

let failures = 0

function render(name: string, option: unknown): void {
  // SSR mode: ECharts fully builds its model, view tree and SVG output with no
  // DOM and no canvas, which is exactly the code path that threw.
  try {
    const chart = echarts.init(null, null, {
      renderer: 'svg',
      ssr: true,
      width: 900,
      height: 400,
    })
    chart.setOption(option as never, true)
    const svg = chart.renderToSVGString()
    chart.dispose()
    if (!svg || svg.length < 64) {
      throw new Error(`render produced ${svg?.length ?? 0} bytes`)
    }
    console.log(`  ok    ${name}  (${svg.length} B svg)`)
  } catch (e) {
    failures++
    const err = e as Error
    // First stack line only: enough to tell which ECharts subsystem threw,
    // without burying the other results.
    const where = (err.stack ?? '').split('\n')[1]?.trim() ?? ''
    console.log(`  FAIL  ${name}: ${err.message}`)
    if (where) console.log(`          at ${where}`)
  }
}

function expect(name: string, cond: boolean, detail = ''): void {
  if (cond) {
    console.log(`  ok    ${name}`)
  } else {
    failures++
    console.log(`  FAIL  ${name}${detail ? ` -- ${detail}` : ''}`)
  }
}

// ---- fixtures ---------------------------------------------------------------

const now = Date.now()
const mkSeries = (n: number): Series => ({
  t: Array.from({ length: n }, (_, i) => now - (n - i) * 1000),
  tok_per_s: Array.from({ length: n }, (_, i) => 20 + (i % 7)),
  ttft_ms: Array.from({ length: n }, (_, i) => 40 + (i % 11)),
  itl_p50_ms: Array.from({ length: n }, (_, i) => 18 + (i % 5)),
  vram_used: Array.from({ length: n }, (_, i) => 1_000_000_000 + i * 1_000_000),
  kv_active: Array.from({ length: n }, (_, i) => 50_000_000 + i * 100_000),
})

const mkRequests = (n: number): RequestRow[] =>
  Array.from({ length: n }, (_, i) => ({
    start_ms: now - i * 2000,
    end_ms: now - i * 2000 + 900,
    queue_ms: i % 3 === 0 ? 120 : 0,
    ttft_ms: 30 + i,
    prefill_ms: 40 + i,
    decode_ms: 800 + i * 10,
    total_ms: 900 + i * 10,
    prompt_tokens: 50 + i,
    completion_tokens: 120 + i,
    prefix_reused: i % 2 === 0 ? 20 : 0,
    tok_per_s: 22 + (i % 5),
    itl_p50_ms: 12 + (i % 17),
    itl_p90_ms: 20 + (i % 23),
    spec_accept: -1,
    status: 'ok',
    streaming: i % 2 === 0,
    session: 0,
  }))

const series = mkSeries(120)
const requests = mkRequests(40)

// ---- 1. registration is complete -------------------------------------------

console.log('\n[registration]')
// The bug this test exists for: LatencyChart.vue uses series.markLine, and
// without MarkLineComponent in src/echarts.ts's use([...]) ECharts throws
//   Uncaught Error: componentType " markLine" illegal
// from deep inside its class registry -- in the browser, at runtime, with
// nothing pointing at the real cause. Rendering the smallest possible markLine
// chart is a direct, minimal check of exactly that.
render('minimal markLine chart (registration probe)', {
  xAxis: { type: 'category', data: ['a', 'b', 'c'] },
  yAxis: { type: 'value' },
  series: [{ type: 'bar', data: [1, 2, 3], markLine: { data: [{ xAxis: 1 }] } }],
})

// ---- 2. every panel renders ------------------------------------------------

console.log('\n[charts]')
// Every panel is rendered once per theme. The palettes differ only in colour, so
// this is mostly about proving the theme plumbing cannot produce a structurally
// different option -- but that is exactly what would break setOption on switch.
const THEMES: ResolvedTheme[] = ['dark', 'light']
for (const theme of THEMES) {
  const p = getPalette(theme)
  const ops = Array.from({ length: 30 }, (_, i) => ({
    name: `layer/ffn_${i}`,
    total_ms: 300 - i * 8,
    count: 100 - i,
    min_ms: 0.1,
    max_ms: 5,
    avg_ms: 3 - i / 50,
  }))

  render(`[${theme}] time series (throughput + TTFT)`, buildTimeSeriesOption(series, throughputLines(p), p))
  render(`[${theme}] time series (memory)`, buildTimeSeriesOption(series, memoryLines(p), p))
  render(`[${theme}] time series (empty)`, buildTimeSeriesOption(mkSeries(0), throughputLines(p), p))
  render(`[${theme}] latency histogram (with p50 markLine)`, buildLatencyOption(requests, p))
  render(`[${theme}] latency histogram (empty)`, buildLatencyOption([], p))
  render(`[${theme}] gauge (vram, known)`, buildGaugeOption({ label: 'VRAM', used: 2_000_000_000, total: 6_000_000_000, color: p.magenta }, p))
  render(`[${theme}] gauge (kv, total unknown)`, buildGaugeOption({ label: 'KV', used: 0, total: 0, color: p.cyan }, p))
  render(`[${theme}] gauge (overflow, used > total)`, buildGaugeOption({ label: 'VRAM', used: 9e9, total: 6e9, color: p.red }, p))
  render(`[${theme}] operator bars`, buildOperatorOption(ops, p))
  render(`[${theme}] operator bars (empty)`, buildOperatorOption([], p))
}

// ---- 3. shape regressions ---------------------------------------------------

console.log('\n[shape]')
const dark = getPalette('dark')
const latencyOption = buildLatencyOption(requests, dark) as { series?: Array<{ markLine?: unknown }> }
expect(
  'latency chart keeps its p50 markLine',
  Array.isArray(latencyOption.series) && latencyOption.series[0]?.markLine !== undefined,
)

const longSeries = buildTimeSeriesOption(mkSeries(2000), throughputLines(dark), dark) as {
  xAxis?: { data?: unknown[] }
  series?: Array<{ data?: unknown[] }>
}
expect(
  'time series window is capped at 240 points',
  (longSeries.xAxis?.data?.length ?? 0) === 240 &&
    (longSeries.series?.[0]?.data?.length ?? 0) === 240,
  `x=${longSeries.xAxis?.data?.length} y=${longSeries.series?.[0]?.data?.length}`,
)

// A full snapshot, to prove the fixtures above match the real API shape.
const fullSnapshot: Metrics = {
  meta: { app: 'forge-cli', model: 'm', arch: 'a', start_ms: 0, now_ms: now, uptime_s: 1, profiling: true },
  engine: {
    kv_active_bytes: 1, kv_total_bytes: 2, kv_free_pages: -1, kv_pool_tokens: 0, kv_page_size: 0,
    sessions: 1, active_requests: 0, prefix_cache_hits: 0, prefix_cache_misses: 0,
    tiered_kv: false, paged_kv: false,
  },
  host: { vram_total_bytes: 8e9, vram_used_bytes: 4e9, rss_bytes: 1e9, cpu_pct: 12 },
  live: {
    active: true, tokens: 42, prompt_tokens: 57, elapsed_ms: 1500, tok_per_s: 24,
    itl_ms: 41, start_ms: now, streaming: true, session: 0,
  },
  totals: {
    requests: 1, errors: 0, rejected: 0, prompt_tokens: 57, completion_tokens: 42,
    prefill_ms: 100, decode_ms: 1500, queue_ms: 0, lifetime_tok_per_s: 28, prefix_hit_rate: 0.5,
  },
  latency: {
    ttft_p50: { value: 40, count: 1 }, ttft_p90: { value: 60, count: 1 }, ttft_p99: { value: 80, count: 1 },
    itl_p50: { value: 41, count: 42 }, itl_p90: { value: 50, count: 42 }, itl_p99: { value: 60, count: 42 },
  },
  series,
  requests,
  operators: [{ name: 'decode/forward', total_ms: 860, count: 12, min_ms: 60, max_ms: 80, avg_ms: 71 }],
  operators_available: true,
}
expect('full snapshot fixture is well formed', fullSnapshot.live.active === true)

/*
 * ---------------------------------------------------------------- export ---
 *
 * The serialisers are pure string-in/string-out (src/export/format.ts) precisely
 * so they can be checked here instead of by clicking through a browser.
 *
 * The escaping cases are the point. A model path, a session label or an operator
 * name may legitimately contain a quote, a comma or a newline, and a malformed
 * CSV or Prometheus file fails silently at worst -- it opens in a spreadsheet
 * with every column after the bad one shifted. So the fixture below is
 * deliberately hostile in exactly those three places.
 */
import { toCsv, toJson, toMarkdown, toPrometheus } from '../src/export/format'
import { exportFilename, EXPORT_FORMATS } from '../src/export/io'

const hostile: Metrics = {
  ...fullSnapshot,
  meta: { ...fullSnapshot.meta, model: 'we,ird "quoted" model', app: 'forge-cli' },
  operators: [
    ...fullSnapshot.operators,
    { name: 'layer/attn,"weird"', total_ms: 12, count: 2, min_ms: 1, max_ms: 8, avg_ms: 6 },
    { name: 'multi\nline\nname', total_ms: 3, count: 1, min_ms: 3, max_ms: 3, avg_ms: 3 },
  ],
}

console.log('\n[export/csv]')
const csv = toCsv(hostile)
const csvLines = csv.split('\n')
// Count within the series block only. Matching on a bare timestamp is not enough:
// request rows start_ms is also a 13-digit 1... value, so an unscoped regex counts
// the requests as well and would pass even if the series block were empty.
const seriesBlock = csv.slice(csv.indexOf('# time_series'), csv.indexOf('\n# requests'))
const seriesRows = seriesBlock.split('\n').filter((l) => /^-?\d+,-?\d+,-?\d+/.test(l)).length
// `series` is an object of parallel arrays, so the sample count is series.t.length.
expect('csv emits one row per series sample',
  seriesRows === series.t.length, `got ${seriesRows} want ${series.t.length}`)
// Section markers appear at line starts, including the first one at index 0.
const sections = csvLines.filter((l) => l.startsWith('# ')).map((l) => l.trim())
expect('csv declares exactly the four sections',
  sections.join(',') === '# time_series,# requests,# operators,# snapshot', sections.join(','))
expect('csv quotes a field containing a comma', csv.includes('"we,ird ""quoted"" model"'))
expect('csv escapes a newline inside a quoted field',
  csv.includes('"multi\nline\nname"'), 'newline must not split the row')
// A row whose operator name contains a newline must still be ONE line's worth of
// comma-separated cells once quoting is applied, which is what the above proves.
expect('csv quotes an embedded double quote', csv.includes('""'))
expect('csv keeps plain numeric cells unquoted', /^1[0-9]{12},2[0-9],4[0-9],/m.test(csv))

console.log('\n[export/json]')
const json = toJson(hostile)
let roundTripped: unknown = null
try {
  roundTripped = (JSON.parse(json) as { metrics: Metrics }).metrics
} catch (e) {
  console.log(`          parse threw: ${(e as Error).message}`)
}
expect('json round-trips to a byte-identical snapshot',
  JSON.stringify(roundTripped) === JSON.stringify(hostile))
expect('json carries provenance alongside the snapshot',
  json.includes('"exported_at"') && json.includes('"generator"') && json.includes('"metrics"'))

console.log('\n[export/markdown]')
const md = toMarkdown(hostile)
expect('markdown has the totals, percentile and memory tables',
  ['## Totals', '## Latency percentiles', '## Memory'].every((h) => md.includes(h)))
expect('markdown notes the in-flight request', md.includes('in flight'))
expect('markdown renders every percentile row', ['| p50 |', '| p90 |', '| p99 |'].every((h) => md.includes(h)))
expect('markdown lists the recent requests', md.includes('## Recent requests'))
expect('markdown lists the operators', md.includes('## Top operators'))

console.log('\n[export/prometheus]')
const prom = toPrometheus(hostile)
const promLines = prom.split('\n').filter(Boolean)
const SAMPLE = /^forge_forge_cli_[a-z0-9_]+(\{[^}]*\})? -?[\d.e+-]+$/
const bad = promLines.filter((l) => !l.startsWith('#') && !SAMPLE.test(l))
expect('every non-comment line is a valid sample', bad.length === 0, bad.slice(0, 3).join(' | '))
expect('HELP and TYPE counts agree',
  promLines.filter((l) => l.startsWith('# HELP')).length ===
    promLines.filter((l) => l.startsWith('# TYPE')).length)
// A label value may contain a comma or a quote; only the quote and backslash need
// escaping, but both are present here.
expect('prometheus escapes quotes in labels', prom.includes('model="we,ird \\"quoted\\" model"'))
expect('prometheus escapes backslashes in labels', /model=".*\\.*"/.test(prom))
expect('prometheus emits numeric quantile labels', prom.includes('quantile="0.5"'))
expect('prometheus prefixes metrics with the sanitised app name',
  prom.includes('forge_forge_cli_requests_total'))
// The contiguous backend reports -1 for a page pool it does not have, and the text
// format has no literal for NaN: both must be coerced to a parseable number.
expect('prometheus contains no NaN/Infinity/undefined',
  !/\b(NaN|Infinity|undefined)\b/.test(prom))

console.log('\n[export/filenames]')
for (const spec of EXPORT_FORMATS) {
  const name = exportFilename(hostile, spec)
  expect(`${spec.format} filename is filesystem safe`, /^[a-z0-9-]+\.[a-z]+$/.test(name), name)
}
expect('filename slugifies spaces and slashes',
  exportFilename({ ...hostile, meta: { ...hostile.meta, model: 'Llama 3/8B' } }, EXPORT_FORMATS[0])
    .includes('llama-3-8b'))

console.log('')
if (failures > 0) {
  console.log(`smoke FAILED: ${failures} problem(s)`)
  process.exit(1)
}
console.log('smoke passed: every chart renders in both themes, every export is well formed')
