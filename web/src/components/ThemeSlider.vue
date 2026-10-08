<script setup lang="ts">
/**
 * ThemeSlider - a three-position slider that picks light / dark / system.
 *
 * Replaces the old cycling button. The slider exposes the current MODE on
 * data-mode (not the resolved colour), so `system` is visually distinct from an
 * explicit pick. A thumb that slides behind the active stop is the affordance;
 * the stops themselves are buttons so they are reachable by mouse, touch and Tab.
 *
 * Accessibility: the track is role="slider" with aria-valuenow = the stop index,
 * so a screen reader announces "Light/Dark/System, 1 of 3". Arrow keys move
 * between stops, Home/End jump to the ends. Each stop is also a real <button>
 * labelled "Theme: <name>", which is the fallback path for AT that ignores the
 * slider role.
 */
import { computed } from 'vue'
import { useTheme, type ThemeMode } from '../composables/theme'

const { mode, set } = useTheme()

const STOPS: { mode: ThemeMode; glyph: string; label: string }[] = [
  { mode: 'light', glyph: '☀', label: 'Light' },
  { mode: 'dark', glyph: '☾', label: 'Dark' },
  { mode: 'system', glyph: '⧉', label: 'System' },
]

const index = computed(() => {
  const i = STOPS.findIndex((s) => s.mode === mode.value)
  return i < 0 ? 0 : i
})

function move(next: number): void {
  if (next < 0 || next >= STOPS.length) return
  set(STOPS[next].mode)
}

function onKey(e: KeyboardEvent): void {
  switch (e.key) {
    case 'ArrowRight':
    case 'ArrowDown':
      e.preventDefault()
      move(index.value + 1)
      break
    case 'ArrowLeft':
    case 'ArrowUp':
      e.preventDefault()
      move(index.value - 1)
      break
    case 'Home':
      e.preventDefault()
      move(0)
      break
    case 'End':
      e.preventDefault()
      move(STOPS.length - 1)
      break
  }
}
</script>

<template>
  <div
    class="theme-slider"
    role="slider"
    tabindex="0"
    aria-label="Theme"
    :aria-valuemin="0"
    :aria-valuemax="STOPS.length - 1"
    :aria-valuenow="index"
    :aria-valuetext="STOPS[index].label"
    data-testid="theme-toggle"
    :data-mode="mode"
    :title="`Theme: ${STOPS[index].label}`"
    @keydown="onKey"
  >
    <span
      class="thumb"
      aria-hidden="true"
      :style="{ transform: `translateX(${index * 100}%)` }"
    />
    <button
      v-for="s in STOPS"
      :key="s.mode"
      type="button"
      class="stop"
      :class="{ active: s.mode === mode }"
      :data-testid="`theme-stop-${s.mode}`"
      :aria-label="`Theme: ${s.label}`"
      :aria-pressed="s.mode === mode"
      @click="set(s.mode)"
    >
      <span aria-hidden="true">{{ s.glyph }}</span>
    </button>
  </div>
</template>

<style scoped>
.theme-slider {
  position: relative;
  display: inline-flex;
  align-items: stretch;
  height: 26px;
  padding: 2px;
  border: 1px solid var(--border);
  border-radius: 13px;
  background: var(--panel);
  color: var(--dim);
  cursor: pointer;
  /* Stop the thumb's slide from being clipped by the rounded corners. */
  overflow: hidden;
}
.theme-slider:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
}

.thumb {
  position: absolute;
  top: 2px;
  left: 2px;
  /* One third of the inner width (container minus the 2px padding each side). */
  width: calc((100% - 4px) / 3);
  height: calc(100% - 4px);
  border-radius: 11px;
  background: var(--hover);
  border: 1px solid var(--dim);
  transition: transform 0.16s ease;
  pointer-events: none;
}

.stop {
  position: relative;
  z-index: 1;
  flex: 1 1 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
  border: 0;
  border-radius: 11px;
  background: transparent;
  color: var(--dim);
  font-size: 13px;
  line-height: 1;
  cursor: pointer;
}
.stop:hover {
  color: var(--text);
}
.stop.active {
  color: var(--text);
}
.stop:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: -2px;
}
</style>
