<script setup lang="ts">
/**
 * Composer - draft input, attachments, Send/Stop.
 *
 * Enter sends, Shift+Enter inserts a newline, the textarea auto-grows to its
 * content (capped at 200px). While a response is streaming the button turns
 * into Stop, which aborts the fetch and keeps the partial answer.
 */
import { onMounted, ref, watch } from 'vue'
import { onDrop, removePending, send, stop, store } from '../store'
import ComposerMenu from './ComposerMenu.vue'
import { Icon } from '../icons'

const draftEl = ref<HTMLTextAreaElement | null>(null)

// The page opens ready to type, and hands the caret back once a reply lands:
// the next thing after reading an answer is nearly always asking again. Not
// while a panel is open — those have their own inputs that own focus.
onMounted(() => draftEl.value?.focus())
watch(
  () => store.streaming,
  (streaming) => {
    if (!streaming && store.panel === 'none') draftEl.value?.focus()
  },
)

function autoGrow() {
  const el = draftEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 200) + 'px'
}

function onInput(e: Event) {
  store.draft = (e.target as HTMLTextAreaElement).value
  autoGrow()
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
    autoGrow()
  }
}
</script>

<template>
  <div class="composer-wrap">
    <form class="composer" @submit.prevent="send()" @dragover.prevent @drop="onDrop($event)">
      <div v-if="store.pending.length" class="attached">
        <div v-for="a in store.pending" :key="a.id" class="att">
          <img v-if="a.kind === 'image' && a.dataUrl" :src="a.dataUrl" :alt="a.name" />
          <Icon v-else name="file" :size="14" class="att-ico" />
          <span class="att-name">{{ a.name }}</span>
          <button class="att-x" title="Remove attachment" @click="removePending(a.id)">
            <Icon name="close" :size="12" />
          </button>
        </div>
      </div>
      <div class="composer-row">
        <ComposerMenu />
        <textarea
          ref="draftEl"
          class="draft"
          rows="1"
          placeholder="Message Forge Chat"
          :value="store.draft"
          @input="onInput"
          @keydown="onKeydown"
        />
        <button
          v-if="!store.streaming"
          class="round"
          type="submit"
          title="Send message"
          :disabled="!store.draft.trim() && !store.pending.length"
          data-testid="send"
        >
          <Icon name="send" :size="18" />
        </button>
        <button v-else class="round stop" type="button" title="Stop generating" @click="stop()">
          <Icon name="stop" :size="18" />
        </button>
      </div>
    </form>
  </div>
</template>
