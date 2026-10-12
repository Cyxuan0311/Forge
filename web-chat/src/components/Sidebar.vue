<script setup lang="ts">
/**
 * Sidebar - the conversation list.
 *
 * Layout follows the desktop ChatGPT sidebar: brand and collapse control at the
 * top, a search field, then "New chat" as the first row, then the conversations
 * grouped by age (Today / Yesterday / Previous 7 days / Previous 30 days /
 * Older). Grouping is what makes a long list navigable -- the newest turn is the
 * one wanted almost every time, and a date header is faster to scan than a
 * timestamp on every row.
 *
 * Search filters by title only: there is no message index to search, and a box
 * that silently ignores the body would be worse than an honestly-scoped one.
 * All mutations go through the store.
 */
import { computed, ref } from 'vue'
import { deleteChat, newChat, selectChat, store } from '../store'
import { Icon } from '../icons'
import type { Conversation } from '../types'

/**
 * The parent owns `drawerOpen` because the scrim, the top-bar hamburger and
 * Escape all have to agree on it; the X here just asks the parent to close.
 */
const emit = defineEmits<{ (e: 'close'): void }>()

const query = ref('')

function startOfDay(t: number): number {
  const d = new Date(t)
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}
const DAY = 86_400_000

/** Which age bucket a conversation falls into. */
function bucket(ts: number): string {
  const today = startOfDay(Date.now())
  if (ts >= today) return 'Today'
  if (ts >= today - DAY) return 'Yesterday'
  if (ts >= today - 7 * DAY) return 'Previous 7 days'
  if (ts >= today - 30 * DAY) return 'Previous 30 days'
  return 'Older'
}

const ORDER = ['Today', 'Yesterday', 'Previous 7 days', 'Previous 30 days', 'Older']

interface Group {
  label: string
  items: Conversation[]
}

const groups = computed<Group[]>(() => {
  const q = query.value.trim().toLowerCase()
  const matched = store.conversations
    .filter((c) => !q || c.title.toLowerCase().includes(q))
    .slice()
    .sort((a, b) => b.ts - a.ts)
  return ORDER.map((label) => ({
    label,
    items: matched.filter((c) => bucket(c.ts) === label),
  })).filter((g) => g.items.length > 0)
})
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-head">
      <span class="brand">Forge Chat</span>
      <button class="icon-btn ghost drawer-x" title="Close sidebar" @click="emit('close')">
        <Icon name="close" :size="18" />
      </button>
    </div>

    <div class="sidebar-search">
      <Icon name="search" :size="15" />
      <input v-model="query" type="text" placeholder="Search chats" aria-label="Search chats" />
    </div>

    <button class="new-chat" @click="newChat">
      <Icon name="plus" :size="17" />
      <span>New chat</span>
    </button>

    <div class="conv-list">
      <template v-for="g in groups" :key="g.label">
        <div class="conv-group">{{ g.label }}</div>
        <div
          v-for="c in g.items"
          :key="c.id"
          class="conv"
          :class="{ active: c.id === store.activeId }"
          @click="selectChat(c.id)"
        >
          <span class="conv-title">{{ c.title }}</span>
          <button class="conv-del" title="Delete conversation" @click.stop="deleteChat(c.id)">
            <Icon name="trash" :size="14" />
          </button>
        </div>
      </template>
      <div v-if="!groups.length" class="conv-empty">
        {{ query ? 'No matching chats' : 'No chats yet' }}
      </div>
    </div>

    <div class="sidebar-foot">Forge Chat &middot; local inference</div>
  </aside>
</template>
