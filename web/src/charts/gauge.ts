/**
 * GaugePanel option - a used/total meter.
 *
 * A donut rather than a speedometer: the only thing that matters here is the
 * filled arc and the numeric label, and a donut reads correctly at a glance
 * where a 270-degree gauge wastes half its canvas on an empty quadrant.
 *
 * Renders an explicit "—" when the total is unknown rather than showing a full
 * or empty ring: `kv_total_bytes` is 0 before the first request publishes, and
 * a 0/0 gauge is indistinguishable from "full" without special-casing it.
 */
import type { DashboardOption } from './option'
import { baseOption, type Palette } from '../theme'
import { fmtBytes } from '../api'

export interface GaugeInput {
  label: string
  used: number
  total: number
  color: string
}

export function buildGaugeOption(input: GaugeInput, p: Palette): DashboardOption {
  const known = input.total > 0
  const pct = known ? Math.max(0, Math.min(1, input.used / input.total)) : 0
  const center = known ? fmtBytes(input.used) : '—'

  return {
    ...baseOption(p),
    series: [
      {
        type: 'pie',
        radius: ['66%', '86%'],
        center: ['50%', '48%'],
        silent: true,
        label: {
          show: true,
          position: 'center',
          formatter: center,
          color: p.text,
          fontSize: 17,
          fontWeight: 600,
        },
        labelLine: { show: false },
        data: [
          { value: known ? pct : 1, itemStyle: { color: input.color } },
          { value: known ? 1 - pct : 0, itemStyle: { color: p.border } },
        ],
      },
    ],
  }
}