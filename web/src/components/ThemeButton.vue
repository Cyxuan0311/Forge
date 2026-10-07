<script setup lang="ts">
/**
 * ThemeButton - a single button that cycles light -> dark -> system.
 *
 * A cycling button rather than a three-way segmented control: at this width the
 * header has no room for three labelled options, and the icons alone stay
 * legible. The cost is that the current mode is not readable at a glance, which
 * the title and aria-label recover for hover and for screen readers.
 *
 * The icon reflects the stored MODE, not the resolved colour, so `system` never
 * looks like an explicit choice. A user who never touches this button sees the
 * sun or moon for `system` only after they have cycled past it once.
 */
import { computed } from 'vue'
import { THEME_CYCLE, useTheme, type ThemeMode } from '../composables/theme'

const { mode, cycle } = useTheme()

const GLYPH: Record<ThemeMode, string> = {
  light: '☀',
  dark: '☾',
  system: '⧉',
}
const NAME: Record<ThemeMode, string> = {
  light: 'Light',
  dark: 'Dark',
  system: 'System',
}

/** e.g. "Theme: Dark (click for System)". */
const label = computed(() => {
  const next = THEME_CYCLE[(THEME_CYCLE.indexOf(mode.value) + 1) % THEME_CYCLE.length]
  return `Theme: ${NAME[mode.value]} (click for ${NAME[next]})`
})
</script>

<template>
  <button
    type="button"
    class="icon-btn"
    :title="label"
    :aria-label="label"
    data-testid="theme-toggle"
    :data-mode="mode"
    @click="cycle"
  >
    <span aria-hidden="true">{{ GLYPH[mode] }}</span>
  </button>
</template>

<style scoped>
.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 26px;
  padding: 0;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--panel);
  color: var(--dim);
  font-size: 14px;
  line-height: 1;
  cursor: pointer;
}
.icon-btn:hover {
  color: var(--text);
  background: var(--hover);
  border-color: var(--dim);
}
.icon-btn:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
}
</style>