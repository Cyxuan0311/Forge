<script setup lang="ts">
/**
 * Sidebar - conversation list. New chat on top, click to switch, hover to
 * delete. All actions come straight from the store.
 */
import { deleteChat, newChat, selectChat, store } from '../store'
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-head">
      <button class="new-chat" @click="newChat">+ New chat</button>
    </div>
    <div class="conv-list">
      <div
        v-for="c in store.conversations"
        :key="c.id"
        class="conv"
        :class="{ active: c.id === store.activeId }"
        @click="selectChat(c.id)"
      >
        <span class="conv-title">{{ c.title }}</span>
        <button class="conv-del" title="Delete conversation" @click.stop="deleteChat(c.id)">
          &times;
        </button>
      </div>
    </div>
    <div class="sidebar-foot">Forge Chat &middot; local inference</div>
  </aside>
</template>
