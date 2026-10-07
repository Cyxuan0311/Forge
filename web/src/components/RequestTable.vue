<script setup lang="ts">
/**
 * RequestTable - the most recent requests, newest first.
 *
 * Capped at 200 rows even though the server keeps 512: a DOM table of 512 rows
 * re-rendered once a second is the one thing in this dashboard that could make
 * the page feel slower than the engine it is measuring.
 */
import { computed } from 'vue'
import { fmtClock, fmtMs, fmtPercent, fmtTok } from '../api'
import type { RequestRow } from '../types'

const props = defineProps<{
  requests: RequestRow[]
}>()

const MAX_ROWS = 200

const rows = computed(() => props.requests.slice(0, MAX_ROWS))

/** Colour the rate by absolute magnitude; the thresholds are decode-shaped. */
function tokTone(v: number): string {
  if (!v || v <= 0) return 'bad'
  if (v >= 40) return 'good'
  if (v >= 15) return 'warn'
  return 'bad'
}
</script>

<template>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th class="l">Time</th>
          <th>Prompt</th>
          <th>Gen</th>
          <th>Reused</th>
          <th>Queue</th>
          <th>TTFT</th>
          <th>Prefill</th>
          <th>Decode</th>
          <th>ITL p50</th>
          <th>tok/s</th>
          <th>Spec</th>
          <th class="l">Status</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(r, i) in rows" :key="`${r.start_ms}-${i}`">
          <td class="l">{{ fmtClock(r.start_ms) }}</td>
          <td>{{ r.prompt_tokens }}</td>
          <td>{{ r.completion_tokens }}</td>
          <td>{{ r.prefix_reused > 0 ? r.prefix_reused : '—' }}</td>
          <td>{{ fmtMs(r.queue_ms) }}</td>
          <td>{{ fmtMs(r.ttft_ms) }}</td>
          <td>{{ fmtMs(r.prefill_ms) }}</td>
          <td>{{ fmtMs(r.decode_ms) }}</td>
          <td>{{ fmtMs(r.itl_p50_ms) }}</td>
          <td :class="tokTone(r.tok_per_s)">{{ fmtTok(r.tok_per_s) }}</td>
          <td>{{ r.spec_accept >= 0 ? fmtPercent(r.spec_accept, 0) : '—' }}</td>
          <td class="l"><span class="st" :class="r.status">{{ r.status }}</span></td>
        </tr>
        <tr v-if="rows.length === 0">
          <td class="l note" colspan="12">No requests recorded yet.</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.good {
  color: var(--green);
}
.warn {
  color: var(--amber);
}
.bad {
  color: var(--red);
}
</style>