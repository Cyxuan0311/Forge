/**
 * echarts.ts - the ONE place ECharts components are registered.
 *
 * Why this is explicit rather than `import * as echarts from 'echarts'`:
 *
 *  1. The full build is ~1.1 MB and, worse, its registrations are invisible.
 *     A component that is not registered fails at RUNTIME, in the browser, with
 *     an error like `componentType "markLine" illegal` -- thrown from deep
 *     inside ECharts' class registry while it installs extensions, with no hint
 *     that the cause was a missing `use()` entry. Listing every chart and
 *     component here makes the set auditable, and the headless SSR smoke test
 *     (scripts/smoke.ts) fails the build if one is missing.
 *
 *  2. The modular build tree-shakes properly. The same page drops from ~1.1 MB
 *     to a few hundred KB, which matters because this bundle is embedded into
 *     the forge-cli / forge-server binaries.
 *
 * If you add a chart type or a markLine/markPoint/markArea to an option, add
 * the matching import to use() below. The smoke test will tell you if you
 * forget.
 */
import * as echarts from 'echarts/core'

import { BarChart, LineChart, PieChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components'
import { LabelLayout } from 'echarts/features'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([
  // Chart types used by the dashboard: time-series lines, the latency histogram
  // bars (with a markLine median marker), the VRAM/KV donuts, and the operator
  // horizontal bars.
  LineChart,
  BarChart,
  PieChart,
  // Coordinate system + the chrome every chart draws.
  GridComponent,
  TooltipComponent,
  LegendComponent,
  // LatencyChart.vue draws a p50 marker with series.markLine. Without this
  // registration ECharts throws `componentType "markLine" illegal` at runtime.
  MarkLineComponent,
  // Needed so labels inside pie/bar/line series are laid out correctly.
  LabelLayout,
  // We render to a <canvas>.
  CanvasRenderer,
])

export { echarts }
export type { EChartsOption } from 'echarts'
