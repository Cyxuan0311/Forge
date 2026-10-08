<script setup lang="ts">
/**
 * TopBar - brand, connection status pill, and the panel toggle buttons.
 * The Connection and Settings panels share one slot in the overlay, so
 * activating one closes the other (store.panel is a single enum, not two
 * booleans).
 */
import { store, toggleTheme } from '../store'

function setPanel(p: 'connection' | 'settings') {
  store.panel = store.panel === p ? 'none' : p
}
</script>

<template>
  <div class="topbar">
    <span class="brand">Forge Chat</span>
    <span class="spacer" />
    <span class="pill" :class="{ ok: store.connState === 'ok' }">
      <span class="dot" />
      <span>{{
        store.connState === 'ok' ? 'Connected' : store.connState === 'bad' ? 'Disconnected' : 'Idle'
      }}</span>
    </span>
    <button class="icon-btn" @click="setPanel('connection')">Connection</button>
    <button class="icon-btn" @click="setPanel('settings')">Settings</button>
    <button class="icon-btn" @click="toggleTheme()">Theme</button>
  </div>
</template>
