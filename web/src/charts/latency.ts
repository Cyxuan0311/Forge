/**
 * LatencyChart option - histogram of inter-token latency.
 *
 * Built from the recent requests' itl_p50 rather than from raw per-token
 * samples: the JSON API ships the percentile reservoirs pre-sorted and capped
 * at 512 entries, and a 512-sample histogram of per-token deltas is not what an
 * operator reads anyway. The bars are p50-per-request, which is what the "is
 * this request slow" question is actually about.
 */
import type { DashboardOption } from './option'
import { axisStyle, baseOption, tooltipStyle, type Palette } from '../theme'
import type { RequestRow } from '../types'

const BUCKETS = 16
/** ms. Anything above this is folded into the last bucket. */
const CEILING = 400

export function buildLatencyOption(requests: RequestRow[], p: Palette): DashboardOption {
  const vals = requests.map((r) => r.itl_p50_ms).filter((v) => isFinite(v) && v > 0)

  // A freshly loaded page has no requests yet. Returning an empty option rather
  // than a malformed one keeps ECharts from warning about an empty category
  // axis; the Vue wrapper shows a placeholder instead.
  if (vals.length === 0) {
    return {
      ...baseOption(p),
      grid: { left: 44, right: 16, top: 26, bottom: 26 },
      xAxis: { type: 'category', data: [], ...axisStyle(p) },
      yAxis: { type: 'value', ...axisStyle(p) },
      series: [],
    }
  }

  const width = CEILING / BUCKETS
  const counts = new Array<number>(BUCKETS).fill(0)
  for (const v of vals) counts[Math.min(BUCKETS - 1, Math.floor(v / width))] += 1
  const labels = counts.map((_, i) => `${(i * width).toFixed(0)}`)
  const p50 = [...vals].sort((a, b) => a - b)[Math.floor(vals.length / 2)]

  return {
    ...baseOption(p),
    grid: { left: 44, right: 16, top: 26, bottom: 26 },
    tooltip: {
      ...tooltipStyle(p),
      trigger: 'axis',
      formatter: (params: unknown) => {
        const arr = params as Array<{ dataIndex: number; value: unknown }>
        if (!Array.isArray(arr) || arr.length === 0) return ''
        const i = arr[0].dataIndex
        return `${labels[i]}-${((i + 1) * width).toFixed(0)} ms<br/>${Number(arr[0].value)} requests`
      },
    },
    xAxis: {
      type: 'category',
      data: labels,
      ...axisStyle(p),
      axisLabel: { ...axisStyle(p).axisLabel, interval: 1 },
    },
    yAxis: { type: 'value', ...axisStyle(p), minInterval: 1 },
    series: [
      {
        type: 'bar',
        data: counts,
        // Mark the median so the eye has an anchor: without it a wide flat
        // histogram is impossible to compare against a narrow tall one.
        //
        // This markLine is what needs MarkLineComponent registered in
        // src/echarts.ts -- without it ECharts throws
        // `componentType "markLine" illegal` at runtime.
        markLine: {
          symbol: 'none',
          silent: true,
          label: {
            formatter: `p50 ${p50.toFixed(1)} ms`,
            color: p.amber,
            fontSize: 11,
            position: 'insideEndTop',
          },
          lineStyle: { color: p.amber, type: 'dashed', width: 1 },
          data: [{ xAxis: Math.min(BUCKETS - 1, Math.floor(p50 / width)) }],
        },
        itemStyle: { color: p.cyan, borderRadius: [2, 2, 0, 0] },
        barMaxWidth: 26,
      },
    ],
  }
}