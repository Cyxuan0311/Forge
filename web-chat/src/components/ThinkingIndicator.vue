<script setup lang="ts">
/**
 * ThinkingIndicator - the "model is working" state.
 *
 * Two things can be happening at once and the user needs to tell them apart:
 * the request may not have produced anything yet (no reasoning, no content), or
 * reasoning may already be streaming into the collapsible Thinking block. So
 * the label switches, and the bars animate either way.
 *
 * Pure CSS animation on a fixed skeleton -- no timers, no per-frame JS -- so a
 * long turn costs nothing while it waits. prefers-reduced-motion replaces the
 * motion with a static indicator (see style.css).
 */
defineProps<{ mode: 'waiting' | 'thinking' | 'writing' }>()
</script>

<template>
  <div class="thinking" :class="mode" role="status" aria-live="polite">
    <span class="bars" aria-hidden="true"><i /><i /><i /></span>
    <span class="thinking-label">
      {{
        mode === 'thinking' ? 'Thinking' : mode === 'writing' ? 'Writing' : 'Waiting for the model'
      }}
    </span>
  </div>
</template>
