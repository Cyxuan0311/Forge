<script setup lang="ts">
/**
 * EChart - the single place an ECharts instance is created and torn down.
 *
 * Every chart panel wraps itself in this. Two things it centralises:
 *
 *  - ResizeObserver. Without it an ECharts canvas keeps the pixel width it had
 *    at init() and overflows its panel as soon as the window narrows, which is
 *    exactly what happens when a user drags the browser edge.
 *
 *  - setOption(..., true) on every update. The default merge mode keeps stale
 *    series around, so a chart that switched from a 300-point series to a
 *    900-point one would silently keep drawing the old points underneath. For
 *    a live-updating dashboard replace is the correct semantic.
 */
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import type { ECharts } from 'echarts/core'
import { echarts } from '../echarts'
import type { DashboardOption } from '../charts/option'

const props = defineProps<{
  option: DashboardOption
  height?: string
}>()

const el = ref<HTMLDivElement | null>(null)
// shallowRef: ECharts instances hold a large mutable object graph; Vue's deep
// reactivity proxying it would be pure overhead.
const chart = shallowRef<ECharts | null>(null)
let ro: ResizeObserver | null = null

onMounted(() => {
  if (!el.value) return
  chart.value = echarts.init(el.value, undefined, { renderer: 'canvas' })
  chart.value.setOption(props.option, true, true)
  ro = new ResizeObserver(() => chart.value?.resize())
  ro.observe(el.value)
})

watch(
  () => props.option,
  (o) => chart.value?.setOption(o, true, true),
  { deep: false },
)

onBeforeUnmount(() => {
  ro?.disconnect()
  ro = null
  chart.value?.dispose()
  chart.value = null
})
</script>

<template>
  <div
    ref="el"
    class="chart"
    :class="{ tall: height === 'tall' }"
    :style="height ? `height:${height}` : ''"
  />
</template>