/**
 * The standard line sets the dashboard draws, so their colours stay consistent
 * between the two time-series panels and follow the active theme.
 *
 * Functions rather than constants because the colours come from the palette,
 * which is mode-dependent (see ../theme.ts).
 */
import type { Palette } from '../theme'
import type { SeriesLine } from './timeseries'

/** Throughput & TTFT panel. */
export function throughputLines(p: Palette): SeriesLine[] {
  return [
    ['tok_per_s', 'tok/s', p.green, 'tok/s'],
    ['ttft_ms', 'TTFT', p.accent, 'ms'],
  ]
}

/**
 * Memory panel. The unit label is decorative: the series are byte counts and
 * the axis renders them as raw numbers, matching the VRAM/KV readouts which do
 * the byte formatting.
 *
 * These two used to be defined inline in App.vue while this module sat unused;
 * App.vue now imports them so the palette has a single source.
 */
export function memoryLines(p: Palette): SeriesLine[] {
  return [
    ['vram_used', 'VRAM used', p.magenta, 'MiB'],
    ['kv_active', 'KV active', p.cyan, 'MiB'],
  ]
}