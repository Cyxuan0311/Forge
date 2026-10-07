<script setup lang="ts">
/**
 * GaugePanel - a used/total meter.
 *
 * The option builder lives in ../charts/gauge.ts so scripts/smoke.ts can render
 * it headlessly.
 */
import { computed } from 'vue'
import EChart from './EChart.vue'
import { buildGaugeOption } from '../charts/gauge'
import { usePalette } from '../composables/palette'

const props = defineProps<{
  label: string
  used: number
  total: number
  color: string
  /** Shown under the ring, e.g. "3.2 GiB / 8.0 GiB". */
  detail?: string
}>()

const palette = usePalette()
const option = computed(() =>
  buildGaugeOption(
    { label: props.label, used: props.used, total: props.total, color: props.color },
    palette.value,
  ),
)
const known = computed(() => props.total > 0)
const pct = computed(() =>
  known.value ? Math.max(0, Math.min(1, props.used / props.total)) : 0,
)
</script>

<template>
  <div class="gauge">
    <EChart :option="option" height="150px" />
    <div class="g-label">{{ label }}</div>
    <div class="g-detail">
      <template v-if="detail">{{ detail }}</template>
      <template v-else-if="known">{{ (pct * 100).toFixed(1) }}%</template>
      <template v-else>—</template>
    </div>
  </div>
</template>

<style scoped>
.gauge {
  text-align: center;
}
.g-label {
  color: var(--text);
  font-size: 12px;
  margin-top: -6px;
}
.g-detail {
  color: var(--dim);
  font-size: 11px;
  margin-top: 2px;
}
</style>
