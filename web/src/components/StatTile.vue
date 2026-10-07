<script setup lang="ts">
/**
 * StatTile - one KPI number with an optional sub-line.
 *
 * `value` is pre-formatted by the caller rather than by this component: the
 * units differ per tile (tok/s, ms, MiB, %) and a generic formatter here would
 * have to guess wrong for half of them.
 */
defineProps<{
  label: string
  value: string
  unit?: string
  sub?: string
  /** Tints the value. Used for ok/warn/bad on error rates. */
  tone?: 'default' | 'good' | 'warn' | 'bad'
}>()
</script>

<template>
  <div class="panel tile">
    <div class="label">{{ label }}</div>
    <div class="value" :class="tone !== 'default' ? tone : ''">
      {{ value }}<span v-if="unit" class="unit">{{ unit }}</span>
    </div>
    <div v-if="sub" class="sub">{{ sub }}</div>
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