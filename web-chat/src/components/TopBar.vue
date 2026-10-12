<script setup lang="ts">
/**
 * TopBar - the active mode readout, connection status pill, and the panel
 * toggles. The brand now lives in the sidebar header, so the bar stays as quiet
 * as ChatGPT's: a centred mode label and a few icon controls on the right.
 *
 * Connection and Settings share one slot in the overlay (store.panel is a single
 * enum, not two booleans), so opening one closes the other.
 *
 * Every control is icon-only with a `title` + `aria-label`: the glyph already
 * says what the control does (plug = connection, sliders = settings, sun/moon =
 * theme), and spelled-out labels would compete with the thread for attention.
 */
import { computed } from 'vue'
import { currentMode, store, toggleTheme } from '../store'
import { Icon } from '../icons'

const emit = defineEmits<{ (e: 'toggle-drawer'): void }>()

/** Centred readout of the active mode, the way ChatGPT shows the model here. */
const modeLabel = computed(() => currentMode().label)

/** Night-arm of the theme toggle says which theme clicking would switch to. */
const themeIcon = computed(() => (store.theme === 'dark' ? 'sun' : 'moon'))
const themeLabel = computed(
  () => 'Switch to ' + (store.theme === 'dark' ? 'light' : 'dark') + ' theme',
)

const connLabel = computed(() =>
  store.connState === 'ok' ? 'Connected' : store.connState === 'bad' ? 'Disconnected' : 'Idle',
)

function setPanel(p: 'connection' | 'settings') {
  store.panel = store.panel === p ? 'none' : p
}
</script>

<template>
  <div class="topbar">
    <button class="icon-btn ghost drawer-toggle" title="Toggle sidebar" @click="emit('toggle-drawer')">
      <Icon name="menu" :size="18" />
    </button>
    <span class="topbar-mode">{{ modeLabel }}</span>
    <span class="spacer" />
    <span class="pill" :class="{ ok: store.connState === 'ok' }">
      <span class="dot" />
      <span>{{ connLabel }}</span>
    </span>
    <button class="icon-btn" title="Connection" aria-label="Connection" @click="setPanel('connection')">
      <Icon name="plug" :size="17" />
    </button>
    <button class="icon-btn" title="Settings" aria-label="Settings" @click="setPanel('settings')">
      <Icon name="settings" :size="17" />
    </button>
    <button class="icon-btn" :title="themeLabel" :aria-label="themeLabel" @click="toggleTheme()">
      <Icon :name="themeIcon" :size="17" />
    </button>
  </div>
</template>
