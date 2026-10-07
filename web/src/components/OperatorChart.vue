<script setup lang="ts">
/**
 * OperatorChart - PerfProfiler top-operator breakdown, one request window.
 *
 * Empty on any build without -DFORGE_PROFILING=ON, because PerfProfiler
 * compiles to nothing there. The empty state says exactly that rather than
 * showing an empty chart, which would read as "no operators ran".
 *
 * The option builder lives in ../charts/operators.ts for the same headless-test
 * reason as the other panels.
 */
import { computed } from 'vue'
import EChart from './EChart.vue'
import { buildOperatorOption } from '../charts/operators'
import { usePalette } from '../composables/palette'
import type { OperatorRow } from '../types'

const props = defineProps<{
  operators: OperatorRow[]
  available: boolean
  profiling: boolean
}>()

const palette = usePalette()
const rows = computed(() => props.operators.slice(0, 12))
const option = computed(() => buildOperatorOption(props.operators, palette.value))
</script>

<template>
  <div v-if="!available" class="note">
    {{
      profiling
        ? 'No operator data in this request window (collection was not armed).'
        : 'PerfProfiler is compiled out on this build. Rebuild with -DFORGE_PROFILING=ON to populate this panel.'
    }}
  </div>
  <div v-else-if="rows.length === 0" class="note">No operator data.</div>
  <EChart v-else :option="option" height="tall" />
</template>
