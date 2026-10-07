/**
 * Option builders for the time-series panels.
 *
 * These are plain functions rather than inline `computed()` bodies in the SFCs
 * for one reason: they are the only part of the UI that can be executed without
 * a browser. scripts/smoke.ts feeds every option they produce into a real
 * headless ECharts instance, which is what turned "componentType \"markLine\"
 * illegal" from a browser-only crash into a test failure.
 */
import type { DashboardOption } from './option'
import type { Palette } from '../theme'
import { axisStyle, baseOption, tooltipStyle } from '../theme'
import type { Series } from '../types'

/** How many trailing points to draw. The server keeps ~900. */
const WINDOW = 240

export type SeriesLine = [keyof Series, string, string, string]

export function buildTimeSeriesOption(
  series: Series,
  lines: SeriesLine[],
  p: Palette,
): DashboardOption {
  const from = Math.max(0, series.t.length - WINDOW)
  const labels = series.t.slice(-WINDOW)
  return {
    ...baseOption(p),
    grid: { left: 56, right: 16, top: 28, bottom: 24 },
    tooltip: {
      ...tooltipStyle(p),
      trigger: 'axis',
      axisPointer: { type: 'line' },
    },
    legend: {
      top: 0,
      textStyle: { color: '#8b93a3', fontSize: 11 },
      itemWidth: 12,
      itemHeight: 8,
    },
    xAxis: {
      type: 'category',
      data: labels,
      boundaryGap: false,
      ...axisStyle(p),
      axisLabel: { ...axisStyle(p).axisLabel, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      scale: true,
      ...axisStyle(p),
      axisLabel: {
        ...axisStyle(p).axisLabel,
        formatter: (v: number) => String(Math.round(v)),
      },
    },
    series: lines.map(([key, name, color, unit]) => ({
      name,
      type: 'line',
      showSymbol: false,
      smooth: false,
      connectNulls: false,
      lineStyle: { width: 1.6, color },
      itemStyle: { color },
      data: (series[key] as number[]).slice(from),
      // ECharts types this as (value: OptionDataValue | OptionDataValue[], ...)
      // even though a line series only ever receives numbers here; coerce
      // rather than narrow so the callback satisfies the declared signature.
      tooltip: {
        valueFormatter: (v: unknown) => `${Number(v ?? 0).toFixed(1)} ${unit}`,
      },
    })),
  }
}