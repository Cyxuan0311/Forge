<script setup lang="ts">
/**
 * App - the dashboard shell.
 *
 * One poll of /api/metrics feeds every panel; nothing here fetches on its own.
 * Polling (rather than SSE) is a hard constraint, not a preference: see the
 * comment at the top of api.ts.
 *
 * While a tab is hidden the poll is paused and the canvas keeps its last frame,
 * so a backgrounded tab costs the engine nothing and does not accumulate a
 * multi-hour backlog of ECharts updates.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { pollMetrics, fmtBytes, fmtCount, fmtMs, fmtPercent, fmtTok, fmtUptime } from './api'
import { providePalette } from './composables/palette'
import { useTheme } from './composables/theme'
import { memoryLines, throughputLines } from './charts/lines'
import type { Metrics } from './types'
import StatTile from './components/StatTile.vue'
import TimeSeriesChart from './components/TimeSeriesChart.vue'
import GaugePanel from './components/GaugePanel.vue'
import LatencyChart from './components/LatencyChart.vue'
import OperatorChart from './components/OperatorChart.vue'
import RequestTable from './components/RequestTable.vue'
import ThemeSlider from './components/ThemeSlider.vue'
import ExportMenu from './components/ExportMenu.vue'

const data = ref<Metrics | null>(null)
const error = ref<string | null>(null)
// Two independent reasons the poll can be paused: the user froze it on purpose,
// or the tab is hidden. Either one stops the poller; the watch below restarts it
// the moment both clear, so resuming always does an immediate fetch.
const manualPaused = ref(false)
const tabHidden = ref(typeof document !== 'undefined' && document.visibilityState === 'hidden')
const effectivePaused = computed(() => manualPaused.value || tabHidden.value)

let poller: ReturnType<typeof pollMetrics> | null = null
let genTimer: ReturnType<typeof setInterval> | undefined

function startPoller() {
  poller?.stop()
  poller = pollMetrics(apply, fail)
}
function stopPoller() {
  poller?.stop()
  poller = null
}

function onVisibility() {
  tabHidden.value = document.visibilityState === 'hidden'
}

function apply(m: Metrics) {
  data.value = m
  error.value = null
}

function fail(e: Error) {
  error.value = e.message
}

function togglePause() {
  manualPaused.value = !manualPaused.value
}

onMounted(() => {
  if (!effectivePaused.value) startPoller()
  document.addEventListener('visibilitychange', onVisibility)
})

onBeforeUnmount(() => {
  stopPoller()
  document.removeEventListener('visibilitychange', onVisibility)
  if (genTimer !== undefined) {
    clearInterval(genTimer)
    genTimer = undefined
  }
})

// Restarting the poller is purely a function of the two pause reasons, so drive
// it from a single computed instead of scattering start/stop calls.
watch(effectivePaused, (paused) => {
  if (paused) stopPoller()
  else startPoller()
})

// ---- derived numbers -------------------------------------------------------

/**
 * Decode rate to display: the live trailing-window rate while a request is in
 * flight, otherwise the last completed request's rate.
 *
 * Both existed as separate concepts and the page showed neither correctly. The
 * old version read the last point of an event-per-completion series, so during a
 * 30 s generation it displayed the PREVIOUS request's speed for the whole
 * generation. Preferring the live value is what makes the tile move.
 */
const live = computed(() => data.value?.live ?? null)
const isGenerating = computed(() => live.value?.active === true)

/**
 * Wall clock that advances ~4 Hz while a request is in flight, so the elapsed /
 * progress readout moves smoothly instead of jumping once per poll (the
 * `liveElapsed` counter is derived from it). It only ticks during generation;
 * otherwise the counter is frozen at the last value.
 */
const now = ref(typeof Date.now === 'function' ? Date.now() : 0)

watch(isGenerating, (active) => {
  if (active) {
    now.value = Date.now()
    if (genTimer === undefined) genTimer = setInterval(() => { now.value = Date.now() }, 250)
  } else if (genTimer !== undefined) {
    clearInterval(genTimer)
    genTimer = undefined
  }
})

const lastTokPerS = computed(() => {
  const l = live.value
  if (l?.active) return l.tok_per_s
  const r = data.value?.requests
  if (r && r.length > 0) return r[0].tok_per_s
  return 0
})

/** Elapsed wall time of the in-flight request, for the progress line. */
const liveElapsed = computed(() => {
  const l = live.value
  if (!l?.active) return 0
  // Re-derive from the live wall clock so the counter ticks at 4 Hz between polls.
  return Math.max(l.elapsed_ms, now.value - l.start_ms)
})

const kvKnown = computed(() => (data.value?.engine.kv_total_bytes ?? 0) > 0)

/*
 * Theme. providePalette() is what makes the resolved theme reach the four chart
 * components (see composables/palette.ts); it returns the palette ref too, for
 * the two inline gauge colours below.
 */
const { resolved } = useTheme()
const palette = providePalette(resolved)

/**
 * Pick a gauge colour by utilisation: red past 90%, amber past 75%, else the
 * baseline hue. Mirrors the tone logic the StatTiles already use, so a GPU or
 * KV cache that is nearly full reads as an alarm instead of its calm baseline.
 */
function gaugeColor(pct: number, base: string): string {
  if (pct >= 0.9) return palette.value.red
  if (pct >= 0.75) return palette.value.amber
  return base
}

const vramColor = computed(() => {
  const h = data.value?.host
  const pct = h && h.vram_total_bytes > 0 ? h.vram_used_bytes / h.vram_total_bytes : 0
  return gaugeColor(pct, palette.value.magenta)
})

const kvColor = computed(() => {
  const e = data.value?.engine
  if (!e || e.kv_total_bytes <= 0) return palette.value.cyan
  return gaugeColor(e.kv_active_bytes / e.kv_total_bytes, palette.value.cyan)
})

/** Rate at which the engine is turning VRAM into output, in bytes per second. */
const memSeriesNote = computed(() => {
  const s = data.value?.series
  if (!s || s.vram_used.length < 2) return ''
  const a = s.vram_used[s.vram_used.length - 2]
  const b = s.vram_used[s.vram_used.length - 1]
  const dt = (s.t[s.t.length - 1] - s.t[s.t.length - 2]) / 1000
  if (dt <= 0) return ''
  const d = b - a
  return `${d >= 0 ? '+' : ''}${fmtBytes(Math.abs(d))} / ${dt.toFixed(1)}s`
})
</script>

<template>
  <div>
    <!-- ---- header ---- -->
    <div class="hdr">
      <h1>FORGE DASHBOARD</h1>
      <span class="meta">
        <span class="dot" :class="error ? 'err' : 'ok'" />{{ error ? 'disconnected' : 'live' }}
      </span>
      <template v-if="data">
        <span class="meta">app <b>{{ data.meta.app }}</b></span>
        <span class="meta">model <b>{{ data.meta.model || '—' }}</b></span>
        <span class="meta">arch <b>{{ data.meta.arch || '—' }}</b></span>
        <span class="meta">uptime <b>{{ fmtUptime(data.meta.uptime_s) }}</b></span>
        <span v-if="isGenerating" class="meta gen">
        <span class="dot live" />generating — {{ live?.tokens ?? 0 }} tokens ·
        {{ (liveElapsed / 1000).toFixed(1) }}s
      </span>
      <span v-if="data.engine.tiered_kv" class="meta">kv <b>tiered (kvmem)</b></span>
        <span v-else-if="data.engine.paged_kv" class="meta">kv <b>paged</b></span>
        <span v-else class="meta">kv <b>contiguous</b></span>
        <span v-if="data.engine.paged_kv && data.engine.kv_page_size" class="meta"
          >page <b>{{ data.engine.kv_page_size }}</b></span
        >
      </template>
      <span class="spacer" />
      <span v-if="effectivePaused" class="meta">{{ manualPaused ? 'paused (manual)' : 'paused (tab hidden)' }}</span>
      <button class="pause-btn" type="button" data-testid="pause-toggle" @click="togglePause">{{ manualPaused ? 'Resume' : 'Pause' }}</button>
      <ExportMenu :data="data" />
      <ThemeSlider />
    </div>

    <!-- Live progress: the one element that makes an in-flight generation
         obviously alive instead of a set of frozen charts. -->
    <div v-if="isGenerating" class="gen-bar">
      <div class="gen-bar-fill" :style="{ animationDuration: '1.1s' }" />
      <span class="gen-bar-text">
        generating — {{ live?.tokens ?? 0 }} tokens in
        {{ (liveElapsed / 1000).toFixed(1) }}s ·
        {{ fmtTok(live?.tok_per_s ?? 0) }} tok/s
        <template v-if="live && live.itl_ms > 0">
          · {{ live.itl_ms.toFixed(1) }} ms/tok
        </template>
      </span>
    </div>

    <div v-if="error" class="err-box">
      /api/metrics unreachable: {{ error }} — retrying with backoff.
    </div>

    <div v-if="!data" class="panel note">Waiting for the first sample…</div>

    <template v-else>
      <!-- ---- latency + throughput ---- -->
      <div class="tiles">
        <StatTile
          :label="isGenerating ? 'Decode rate (live)' : 'Decode rate'"
          :value="fmtTok(lastTokPerS)"
          unit="tok/s"
          :sub="
            isGenerating
              ? `elapsed ${(liveElapsed / 1000).toFixed(1)}s · ${live?.itl_ms ? live.itl_ms.toFixed(1) : '—'} ms/tok`
              : `lifetime ${fmtTok(data.totals.lifetime_tok_per_s)} tok/s`
          "
          :tone="lastTokPerS >= 15 ? 'good' : lastTokPerS > 0 ? 'warn' : 'default'"
        />
        <StatTile
          :label="isGenerating ? 'Tokens emitted' : 'Tokens generated'"
          :value="fmtCount(isGenerating ? (live?.tokens ?? 0) : data.totals.completion_tokens)"
          :sub="isGenerating ? 'this request' : `${fmtCount(data.totals.prompt_tokens)} prompt (lifetime)`"
        />
        <StatTile
          label="TTFT p50"
          :value="fmtMs(data.latency.ttft_p50.value, 0)"
          :sub="`p90 ${fmtMs(data.latency.ttft_p90.value, 0)} · p99 ${fmtMs(data.latency.ttft_p99.value, 0)}`"
          :tone="data.latency.ttft_p50.value > 500 ? 'bad' : data.latency.ttft_p50.value > 150 ? 'warn' : 'good'"
        />
        <StatTile
          label="ITL p50"
          :value="fmtMs(data.latency.itl_p50.value)"
          :sub="`p90 ${fmtMs(data.latency.itl_p90.value)} · p99 ${fmtMs(data.latency.itl_p99.value)}`"
          :tone="data.latency.itl_p50.value > 60 ? 'bad' : data.latency.itl_p50.value > 25 ? 'warn' : 'good'"
        />
        <StatTile
          label="Tokens"
          :value="fmtCount(data.totals.completion_tokens)"
          unit="gen"
          :sub="`${fmtCount(data.totals.prompt_tokens)} prompt`"
        />
        <StatTile
          label="Requests"
          :value="fmtCount(data.totals.requests)"
          :sub="`${data.totals.errors} err · ${data.totals.rejected} 429`"
          :tone="data.totals.errors > 0 ? 'warn' : 'default'"
        />
        <StatTile
          label="Prefill / decode"
          :value="fmtMs(data.totals.prefill_ms, 0)"
          :sub="`decode ${fmtMs(data.totals.decode_ms, 0)} total`"
        />
      </div>

      <!-- ---- throughput over time ---- -->
      <div class="grid2">
        <div class="panel">
          <h2>Throughput &amp; TTFT</h2>
          <TimeSeriesChart :series="data.series" :lines="throughputLines" />
        </div>
        <div class="panel">
          <h2>Inter-token latency distribution</h2>
          <LatencyChart :requests="data.requests" />
        </div>
      </div>

      <!-- ---- queue / prefix cache / sessions ---- -->
      <div class="tiles">
        <StatTile
          label="Active requests"
          :value="String(data.engine.active_requests)"
          :sub="
            isGenerating
              ? `in flight ${(liveElapsed / 1000).toFixed(1)}s`
              : `total ${data.totals.requests}`
          "
          :tone="isGenerating ? 'good' : 'default'"
        />
        <StatTile
          label="Sessions"
          :value="String(data.engine.sessions)"
          :sub="data.engine.tiered_kv ? 'kvmem: 1 at a time' : 'per-session KV reuse'"
        />
        <StatTile
          label="Queue wait avg"
          :value="fmtMs(data.totals.queue_ms / Math.max(1, data.totals.requests), 0)"
          :sub="`total ${fmtMs(data.totals.queue_ms, 0)}`"
          :tone="data.totals.queue_ms / Math.max(1, data.totals.requests) > 50 ? 'warn' : 'default'"
        />
        <StatTile
          label="Prefix cache hit"
          :value="fmtPercent(data.totals.prefix_hit_rate, 0)"
          :sub="`${data.engine.prefix_cache_hits} hit / ${data.engine.prefix_cache_misses} miss`"
          :tone="data.totals.prefix_hit_rate >= 0.5 ? 'good' : 'default'"
        />
        <StatTile
          label="Rejected (429)"
          :value="String(data.totals.rejected)"
          :tone="data.totals.rejected > 0 ? 'warn' : 'default'"
        />
        <StatTile label="CPU" :value="data.host.cpu_pct.toFixed(1)" unit="%" :sub="`RSS ${fmtBytes(data.host.rss_bytes)}`" />
      </div>

      <!-- ---- gauges ---- -->
      <div class="grid2">
        <div class="panel">
          <h2>VRAM &amp; KV cache</h2>
          <div class="gauge-row">
            <GaugePanel
              label="VRAM"
              :used="data.host.vram_used_bytes"
              :total="data.host.vram_total_bytes"
              :color="vramColor"
              :detail="`${fmtBytes(data.host.vram_used_bytes)} / ${fmtBytes(data.host.vram_total_bytes)}`"
            />
            <GaugePanel
              label="KV cache"
              :used="data.engine.kv_active_bytes"
              :total="data.engine.kv_total_bytes"
              :color="kvColor"
              :detail="
                kvKnown
                  ? `${fmtBytes(data.engine.kv_active_bytes)} / ${fmtBytes(data.engine.kv_total_bytes)}`
                  : 'n/a'
              "
            />
          </div>
          <p class="note" v-if="memSeriesNote">{{ memSeriesNote }}</p>
        </div>
        <div class="panel">
          <h2>Memory over time</h2>
          <TimeSeriesChart :series="data.series" :lines="memoryLines" />
        </div>
      </div>

      <!-- ---- operators ---- -->
      <div class="grid2">
        <div class="panel">
          <h2>Operator breakdown (last request window)</h2>
          <OperatorChart
            :operators="data.operators"
            :available="data.operators_available"
            :profiling="data.meta.profiling"
          />
        </div>
        <div class="panel">
          <h2>Recent requests</h2>
          <RequestTable :requests="data.requests" />
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.gauge-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.pause-btn {
  background: var(--panel);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 10px;
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.pause-btn:hover {
  border-color: var(--accent);
}
</style>