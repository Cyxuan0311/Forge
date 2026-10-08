<script setup lang="ts">
/**
 * ModalShell - frosted-glass overlay host for the Connection / Settings panels.
 *
 * One component for both so the dismiss behaviour cannot drift apart: click on
 * the backdrop, press Escape, or use the header button. The panel itself keeps
 * the same `.panel` class the panels' own layout rules already target, so this
 * only owns the overlay, the glass surface and the header.
 */
import { onBeforeUnmount, onMounted } from 'vue'

defineProps<{ title: string }>()
const emit = defineEmits<{ (e: 'close'): void }>()

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('close')
}

onMounted(() => window.addEventListener('keydown', onKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="modal-backdrop" @click.self="emit('close')">
    <div class="modal glass" role="dialog" aria-modal="true" :aria-label="title">
      <div class="modal-head">
        <h3>{{ title }}</h3>
        <button class="icon-btn modal-x" aria-label="Close" @click="emit('close')">&times;</button>
      </div>
      <div class="modal-body">
        <slot />
      </div>
    </div>
  </div>
</template>
