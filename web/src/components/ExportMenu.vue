<script setup lang="ts">
/**
 * ExportMenu - the data-export control.
 *
 * Exports whatever /api/metrics last returned, i.e. the exact snapshot the
 * charts are drawing, so a saved file always matches what was on screen.
 *
 * Purely client-side by design: there is no /api/export route, so nothing here
 * can be scripted with curl. The Prometheus entry is a textfile for the
 * node_exporter collector, not a scrape target, and its hint says so.
 */
import { computed, onBeforeUnmount, ref } from 'vue'
import { EXPORT_FORMATS, runExport, type ExportSpec } from '../export/io'
import type { Metrics } from '../types'

const props = defineProps<{
  /** null until the first successful poll; the button is disabled without it. */
  data: Metrics | null
}>()

const open = ref(false)
const status = ref<string | null>(null)
const root = ref<HTMLElement | null>(null)

const ready = computed(() => props.data !== null)

function close() {
  open.value = false
}

function toggle() {
  open.value = !open.value
}

/** Click outside and Escape both dismiss; the menu is not a modal. */
function onDocClick(e: MouseEvent) {
  if (!open.value) return
  if (root.value && !root.value.contains(e.target as Node)) close()
}

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape' && open.value) close()
}

document.addEventListener('click', onDocClick)
document.addEventListener('keydown', onKey)
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocClick)
  document.removeEventListener('keydown', onKey)
})

async function pick(spec: ExportSpec) {
  close()
  if (!props.data) return
  status.value = null
  try {
    status.value = await runExport(props.data, spec)
  } catch (err) {
    status.value = 'export failed: ' + (err instanceof Error ? err.message : String(err))
  }
}
</script>

<template>
  <span ref="root" class="export">
    <button
      type="button"
      class="btn"
      :disabled="!ready"
      :title="ready ? 'Export the current snapshot' : 'No data to export yet'"
      data-testid="export-toggle"
      @click="toggle"
    >
      <span aria-hidden="true">⇩</span> Export
    </button>

    <span v-if="open" class="menu" role="menu">
      <button
        v-for="spec in EXPORT_FORMATS"
        :key="spec.format"
        type="button"
        class="item"
        role="menuitem"
        :data-testid="`export-${spec.format}`"
        @click="pick(spec)"
      >
        <span class="item-label">{{ spec.label }}</span>
        <span class="item-hint">{{ spec.hint }}</span>
      </button>
    </span>

    <span v-if="status" class="status" role="status" data-testid="export-status">{{
      status
    }}</span>
  </span>
</template>

<style scoped>
.export {
  position: relative;
  display: inline-flex;
  align-items: center;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 26px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--panel);
  color: var(--text);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.btn:hover:not(:disabled) {
  background: var(--hover);
  border-color: var(--dim);
}
.btn:disabled {
  color: var(--dim);
  cursor: not-allowed;
  opacity: 0.55;
}
.btn:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
}

.menu {
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  z-index: 20;
  display: flex;
  flex-direction: column;
  min-width: 216px;
  padding: 3px;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--panel);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}
.item {
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding: 6px 8px;
  border: 0;
  border-radius: 3px;
  background: transparent;
  color: var(--text);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.item:hover {
  background: var(--hover);
}
.item:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: -2px;
}
.item-label {
  font-size: 12px;
  font-weight: 600;
}
.item-hint {
  color: var(--dim);
  font-size: 11px;
}

.status {
  position: absolute;
  top: calc(100% + 6px);
  right: 0;
  z-index: 21;
  max-width: 300px;
  padding: 4px 8px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--panel);
  color: var(--dim);
  font-size: 11px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>