# Forge Web Dashboard

Frontend for the `--dashboard` flag on `forge-cli` and `forge-server`. Vue 3 +
TypeScript + Vite, charts by ECharts.

## How it ships

`cmake/EmbedDashboardAssets.cmake` walks `dist/` and generates a C++ source file
whose asset table is compiled into the binary. The release tarball ships a bare
`forge-cli` with no asset directory next to it, so serving from disk would need a
`--dashboard-root` escape hatch and would still break for every installed copy.

Node.js is therefore a **build-time only** dependency. With
`-DFORGE_BUILD_DASHBOARD=OFF` (the default) nothing here is touched, and the
dashboard server still answers `/api/metrics` while `/dashboard/` explains how to
rebuild.

## Requirements

Node.js 20.19+ (Vite 7). `npm --version` reporting v9 or older means the toolchain
is too old; `build.sh` detects this and disables the dashboard rather than failing
halfway through a build.

## Commands

```bash
npm ci           # install exactly what package-lock.json pins
npm run dev      # vite dev server, hot reload, proxies /api to 127.0.0.1:9090
npm run build    # production bundle into dist/ (CMake redirects this to the build tree)
npm run typecheck# vue-tsc --noEmit
npm test         # typecheck + smoke + build + bundle test, i.e. what CI runs
```

`npm run build` is deliberately separate from `npm run typecheck`: CMake's
`npm run build` step must not fail on a type error, or a type nit would break the
C++ build. CI runs both.

## Layout

```
index.html
vite.config.ts
src/
  main.ts               app bootstrap + global stylesheet
  App.vue               shell; one poll of /api/metrics feeds every panel
  api.ts                fetch client, formatters, and the comments explaining
                        why this polls instead of using SSE
  types.ts              hand-written mirror of the /api/metrics response
  theme.ts              palettes for both themes, shared by every ECharts instance
  charts/               pure option builders, one per panel (see below)
  composables/
    theme.ts            the theme MODE and its localStorage persistence
    palette.ts          provides the active palette to the chart components
  export/
    format.ts           the four serialisers; pure string-in/string-out
    io.ts               download / clipboard plumbing, and the format table
  components/
    EChart.vue          the single place an ECharts instance is created/destroyed
    StatTile.vue        KPI tile
    TimeSeriesChart.vue tok/s and TTFT over time
    LatencyChart.vue    inter-token latency histogram
    GaugePanel.vue      VRAM / KV donuts
    OperatorChart.vue   PerfProfiler top-operator bars
    RequestTable.vue    per-request table
    ThemeSlider.vue     the light/dark/system slider
    ExportMenu.vue      the export button and its format menu
scripts/
  smoke.ts              renders every option headlessly, in both themes
  test-bundle.mjs       mounts the built dist/ in jsdom and drives the UI
```

## Adding a chart

Write the option builder in `src/charts/` as a plain function taking the palette
as its last argument, rather than building the option inline in a `.vue` file.

That is what makes `scripts/smoke.ts` possible: it feeds each builder's output
into a real headless ECharts instance, in **both** themes, so a missing component
registration or a malformed axis fails in CI instead of in someone's browser.
Keep it that way — the `markLine` crash that motivated that test passed
`vue-tsc` and built without a warning, and only failed on page load.

Then render it with `<EChart :option="..." />`. Use `usePalette()` for the
colours rather than importing one; a canvas cannot inherit CSS custom
properties, so every colour has to be handed to each option.

## Themes

`composables/theme.ts` owns the mode (`light` | `dark` | `system`) and persists it
under `forge.dashboard.theme`. `theme.ts` owns the two ECharts palettes;
`style.css` owns the two CSS variable blocks.

Those are duplicated **on purpose** and must be kept in sync: ECharts draws to a
canvas and cannot read a CSS custom property, so there is no way to have one
source for both. `scripts/test-bundle.mjs` asserts the CSS has a
`[data-theme=light]` block and no stray hardcoded dark literal.

Adding a colour means touching all three: the CSS variable in both theme blocks,
the palette field in `theme.ts`, and wherever it is used.

`applyTheme()` runs in `main.ts` before `createApp` so a light-preference machine
never renders a dark frame first.

## Adding an export format

Add a pure serialiser to `src/export/format.ts` (`(m: Metrics) => string`), then
register it in `EXPORT_FORMATS` in `export/io.ts`. The menu, the filename and the
download plumbing follow automatically.

The serialisers take no DOM dependencies on purpose: `scripts/smoke.ts` asserts on
their output directly. Two things to get right there, both covered by tests:

- **Escaping.** A model path or operator name may contain a quote, a comma or a
  newline. Malformed CSV or Prometheus fails *silently* — it opens in a
  spreadsheet with every column after the bad one shifted.
- **Non-finite numbers.** The Prometheus text format has no literal for `NaN` or
  `Infinity`, and the contiguous KV backend reports `-1` for a page pool it does
  not have. Coerce to something parseable.

## Adding a metric

1. Add the field to `DashboardServer::snapshot_to_json()` in
   `src/dashboard/dashboard_server.cpp`.
2. Mirror it in `src/types.ts`.
3. Populate it from `MetricsRegistry` (see `include/forge/metrics.h` for the
   threading contract -- engine-side numbers must be published by the inference
   thread, never polled from the dashboard thread).
4. Render it. Add a unit test to `tests/test_metrics.cpp` if it is accounting
   rather than a straight copy.
5. It appears in all four export formats automatically, since they all serialise
   the whole snapshot.

## No CDN

There are no external fonts, scripts or stylesheets. The dashboard is served from
a loopback port by a process that may well be running on an air-gapped machine,
so anything fetched from the internet would render as a broken page.
