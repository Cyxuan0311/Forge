/**
 * OperatorChart option - PerfProfiler top-operator breakdown, one request window.
 *
 * Empty on any build without -DFORGE_PROFILING=ON, because PerfProfiler compiles
 * to nothing there. The Vue wrapper reports that state explicitly rather than
 * showing an empty chart, which would read as "no operators ran".
 */
import type { DashboardOption } from './option'
import { axisStyle, baseOption, seriesColors, tooltipStyle, type Palette } from '../theme'
import type { OperatorRow } from '../types'

const TOP = 12

export function buildOperatorOption(operators: OperatorRow[], p: Palette): DashboardOption {
  const rows = operators.slice(0, TOP)
  // ECharts draws the first category at the bottom of a horizontal bar chart,
  // so reversing puts the most expensive operator at the top where it belongs.
  const data = [...rows].reverse()
  return {
    ...baseOption(p),
    // Explicit margins rather than `grid.containLabel`, which ECharts 6 warns
    // about unless LegacyGridContainLabel is registered (it flagged this on the
    // first headless smoke run). The y-axis labels are truncated to 150px by the
    // axisLabel below, so the left margin is deterministic and needs no
    // measuring pass.
    grid: { left: 160, right: 80, top: 8, bottom: 8 },
    tooltip: {
      ...tooltipStyle(p),
      trigger: 'item',
      formatter: (p: unknown) => {
        const e = p as { dataIndex: number }
        const r = data[e.dataIndex]
        if (!r) return ''
        return `${r.name}<br/>total ${r.total_ms.toFixed(2)} ms<br/>n=${r.count}<br/>avg ${r.avg_ms.toFixed(3)} ms`
      },
    },
    xAxis: { type: 'value', ...axisStyle(p) },
    yAxis: {
      type: 'category',
      data: data.map((r) => r.name),
      ...axisStyle(p),
      axisLabel: { ...axisStyle(p).axisLabel, width: 150, overflow: 'truncate' },
    },
    series: [
      {
        type: 'bar',
        data: data.map((r, i) => ({
          value: r.total_ms,
          itemStyle: { color: seriesColors(p)[(TOP - 1 - i) % seriesColors(p).length] },
        })),
        barMaxWidth: 14,
        label: {
          show: true,
          position: 'right',
          color: p.dim,
          fontSize: 10,
          formatter: (p: { value: unknown }) => `${Number(p.value ?? 0).toFixed(1)} ms`,
        },
      },
    ],
  }
}