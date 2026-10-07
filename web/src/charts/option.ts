/**
 * The composed option type for the dashboard's charts.
 *
 * `echarts/core` deliberately does not export a ready-made `EChartsOption`: the
 * full-build type describes every chart and component in the library, most of
 * which this page never uses, so it would accept options that fail at runtime
 * for want of a registration. `ComposeOption` builds the precise union from the
 * pieces actually registered in src/echarts.ts, which means:
 *
 *   - a chart type we forgot to register is a TYPE error, not a runtime crash;
 *   - an option key that belongs to an unregistered component is rejected too.
 *
 * Keep this list and the use([...]) list in src/echarts.ts in sync. They are the
 * same set expressed twice, and scripts/smoke.ts is what catches a drift.
 */
import type { ComposeOption } from 'echarts/core'
import type { BarSeriesOption, LineSeriesOption, PieSeriesOption } from 'echarts/charts'
import type {
  GridComponentOption,
  LegendComponentOption,
  MarkLineComponentOption,
  TooltipComponentOption,
} from 'echarts/components'

export type DashboardOption = ComposeOption<
  | LineSeriesOption
  | BarSeriesOption
  | PieSeriesOption
  | GridComponentOption
  | TooltipComponentOption
  | LegendComponentOption
  | MarkLineComponentOption
>

/** Options are passed to setOption, which takes the composed type or a subset. */
export type { ComposeOption }
export type { ECharts } from 'echarts/core'