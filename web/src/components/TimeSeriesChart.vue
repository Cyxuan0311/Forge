<script setup lang="ts">
/**
 * TimeSeriesChart - a line chart over the /api/metrics `series` arrays.
 *
 * The x axis is wall-clock derived from series.t rather than an index, because
 * the series is SAMPLED on the server's 1 Hz tick. An index axis would mislabel
 * the window whenever a poll was slow or a request was in flight.
 *
 * The option itself lives in ../charts/timeseries.ts so it can be executed
 * headlessly by scripts/smoke.ts, in both themes.
 */
import { computed } from 'vue'
import EChart from './EChart.vue'
import { buildTimeSeriesOption, type SeriesLine } from '../charts/timeseries'
import { usePalette } from '../composables/palette'
import type { Palette } from '../theme'
import type { Series } from '../types'

const props = defineProps<{
  series: Series
  /** [field key, display name, colour, unit] -- colours come from the palette. */
  lines: (p: Palette) => SeriesLine[]
  height?: string
}>()

const palette = usePalette()
const option = computed(() => buildTimeSeriesOption(props.series, props.lines(palette.value), palette.value))
</script>

<template>
  <EChart :option="option" :height="height" />
</template>
