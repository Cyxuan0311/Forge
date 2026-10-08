<script setup lang="ts">
/**
 * App - the standalone chat shell: sidebar + top bar + thread + composer,
 * plus the frosted-glass overlay that hosts the Connection / Settings panels.
 *
 * State lives in store.ts; this component only wires the layout together and
 * keeps the thread scrolled to the newest token. The one nicety worth noting:
 * scrolling watches `messages` deeply because streaming mutates the assistant
 * message in place -- a shallow watch would never fire during a generation.
 */
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { activeConv, applyTheme, store, testConnection } from './store'
import Sidebar from './components/Sidebar.vue'
import TopBar from './components/TopBar.vue'
import MessageItem from './components/MessageItem.vue'
import Composer from './components/Composer.vue'
import ModalShell from './components/ModalShell.vue'
import ConnectionPanel from './components/ConnectionPanel.vue'
import SettingsPanel from './components/SettingsPanel.vue'

const thread = ref<HTMLElement | null>(null)
const messages = computed(() => activeConv.value?.messages ?? [])

/** The two panels share one slot: opening one closes the other. */
const panelTitle = computed(() =>
  store.panel === 'connection' ? 'Connection' : 'Generation settings',
)

function closePanel() {
  store.panel = 'none'
}

function scrollBottom() {
  nextTick(() => {
    const el = thread.value
    if (el) el.scrollTop = el.scrollHeight
  })
}

watch(messages, scrollBottom, { deep: true })

onMounted(() => {
  applyTheme()
  testConnection()
})
</script>

<template>
  <div class="app">
    <Sidebar />
    <main class="main">
      <TopBar />
      <div ref="thread" class="thread">
        <div v-if="!messages.length" class="center-wrap">
          <div class="greeting">How can I help you today?</div>
        </div>
        <MessageItem v-for="m in messages" :key="m.id" :message="m" />
      </div>
      <Composer />
    </main>

    <ModalShell v-if="store.panel === 'connection'" title="Connection" @close="closePanel">
      <ConnectionPanel />
    </ModalShell>
    <ModalShell v-else-if="store.panel === 'settings'" :title="panelTitle" @close="closePanel">
      <SettingsPanel />
    </ModalShell>
  </div>
</template>
