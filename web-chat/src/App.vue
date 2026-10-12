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
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { activeConv, applyTheme, refreshToolAvailability, save, send, store, testConnection } from './store'
import { Icon } from './icons'
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

/** Whether the sidebar counts as an overlay rather than a column right now. */
function isNarrow(): boolean {
  return !!window.matchMedia && window.matchMedia('(max-width: 900px)').matches
}

// Sidebar visibility is a preference, not session state: someone who opened it
// once expects it open next visit, exactly like the theme and the model.
const drawerOpen = ref(store.sidebarOpen)
watch(drawerOpen, (v) => {
  store.sidebarOpen = v
  save()
})

function closePanel() {
  store.panel = 'none'
}

/**
 * Whether the viewport is parked at the newest message. Following the stream
 * unconditionally is the one thing a chat thread must not do: reading an older
 * answer while tokens arrive would yank the page down on every chunk. So the
 * thread only follows while the reader is already at the bottom, and offers a
 * jump-to-latest button the moment they are not.
 */
const atBottom = ref(true)
const showJump = computed(() => !atBottom.value && messages.value.length > 0)

function onScroll(): void {
  const el = thread.value
  if (!el) return
  atBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 48
}

/** `force` is the explicit jump (button, switching threads): always go down. */
function scrollBottom(force = false): void {
  nextTick(() => {
    const el = thread.value
    if (!el) return
    if (force || atBottom.value) el.scrollTop = el.scrollHeight
    onScroll()
  })
}

watch(messages, () => scrollBottom(), { deep: true })

// Switching threads is a jump, not a stream: land at the newest message even
// if the previous thread was scrolled up.
watch(() => store.activeId, () => {
  // On narrow screens the list covers the answer, so picking a conversation
  // dismisses it — with a persistent column there is nothing to dismiss.
  if (isNarrow()) drawerOpen.value = false
  scrollBottom(true)
})

// Escape closes the drawer, matching the panels and the tools menu.
function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') drawerOpen.value = false
}

/**
 * Empty-state prompts, laid out like ChatGPT's suggestion rows. Clicking one
 * fills the draft and sends it -- the same path a typed message takes -- so the
 * first turn is one tap, and each row doubles as a demo of a mode.
 */
const suggestions = [
  { icon: 'writing', text: 'Help me draft a short professional email' },
  { icon: 'search', text: "Search for today's top headlines" },
  { icon: 'code', text: 'Write a small Python script to rename files' },
]
function useSuggestion(text: string): void {
  store.draft = text
  void send()
}

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
  applyTheme()
  testConnection()
  // Land at the newest message of the restored thread rather than at its top.
  scrollBottom(true)
  // Probe which server tools are usable so the mode menu can hide (or annotate)
  // modes whose backing tool is disabled. Non-blocking: the UI is usable before
  // this resolves, it just offers every mode until then.
  void refreshToolAvailability()
})

onBeforeUnmount(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <div class="app" :class="{ 'drawer-open': drawerOpen }">
    <Sidebar @close="drawerOpen = false" />
    <div class="scrim" @click="drawerOpen = false" />
    <main class="main" :class="{ empty: !messages.length }">
      <TopBar @toggle-drawer="drawerOpen = !drawerOpen" />
      <div ref="thread" class="thread" @scroll.passive="onScroll">
        <div class="thread-inner">
          <div v-if="!messages.length" class="center-wrap">
            <div class="greeting">How can I help you today?</div>
          </div>
          <MessageItem v-for="m in messages" :key="m.id" :message="m" />
        </div>
      </div>
      <div class="composer-dock">
        <button v-if="showJump" class="jump" title="Scroll to latest" @click="scrollBottom(true)">
          <Icon name="chevron-down" :size="18" />
        </button>
        <Composer />
        <div v-if="!messages.length" class="suggestions">
          <button
            v-for="s in suggestions"
            :key="s.text"
            class="sugg"
            type="button"
            @click="useSuggestion(s.text)"
          >
            <Icon :name="s.icon" :size="16" class="sugg-ico" />
            <span>{{ s.text }}</span>
          </button>
        </div>
      </div>
    </main>

    <ModalShell v-if="store.panel === 'connection'" title="Connection" @close="closePanel">
      <ConnectionPanel />
    </ModalShell>
    <ModalShell v-else-if="store.panel === 'settings'" :title="panelTitle" @close="closePanel">
      <SettingsPanel />
    </ModalShell>
  </div>
</template>
