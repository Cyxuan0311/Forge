<script setup lang="ts">
/**
 * LatencyChart - histogram of inter-token latency.
 *
 * The option builder lives in ../charts/latency.ts so it can be executed
 * headlessly by scripts/smoke.ts; that test is what guarantees the markLine
 * marker does not crash the chart at runtime.
 */
import { computed } from 'vue'
import EChart from './EChart.vue'
import { buildLatencyOption } from '../charts/latency'
import { usePalette } from '../composables/palette'
import type { RequestRow } from '../types'

const props = defineProps<{
  requests: RequestRow[]
}>()

const palette = usePalette()
const option = computed(() => buildLatencyOption(props.requests, palette.value))
const empty = computed(
  () => !props.requests.some((r) => isFinite(r.itl_p50_ms) && r.itl_p50_ms > 0),
)
</script>

<template>
  <div v-if="empty" class="note">No completed requests with measured latency yet.</div>
  <EChart v-else :option="option" />
</template>
