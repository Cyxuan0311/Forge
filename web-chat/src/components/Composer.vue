<script setup lang="ts">
/**
 * Composer - draft input, attachments, Send/Stop.
 *
 * Enter sends, Shift+Enter inserts a newline, the textarea auto-grows to its
 * content (capped at 200px). While a response is streaming the button turns
 * into Stop, which aborts the fetch and keeps the partial answer.
 */
import { ref } from 'vue'
import { addFiles, onDrop, removePending, send, stop, store } from '../store'

const draftEl = ref<HTMLTextAreaElement | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)

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

function onFileChange(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files) addFiles(input.files)
  input.value = ''
}
</script>

<template>
  <div class="composer-wrap">
    <div class="composer" @dragover.prevent @drop="onDrop($event)">
      <div v-if="store.pending.length" class="attached">
        <div v-for="a in store.pending" :key="a.id" class="att">
          <img v-if="a.kind === 'image' && a.dataUrl" :src="a.dataUrl" :alt="a.name" />
          <span v-else>[file]</span>
          <span class="att-name">{{ a.name }}</span>
          <button class="att-x" title="Remove attachment" @click="removePending(a.id)">
            &times;
          </button>
        </div>
      </div>
      <div class="composer-row">
        <button class="attach-btn" title="Attach image or file" @click="fileInput?.click()">
          +
        </button>
        <input ref="fileInput" type="file" multiple hidden @change="onFileChange" />
        <textarea
          ref="draftEl"
          class="draft"
          rows="1"
          placeholder="Send a message..."
          :value="store.draft"
          @input="onInput"
          @keydown="onKeydown"
        />
        <button v-if="!store.streaming" class="send" @click="send()">Send</button>
        <button v-else class="send danger" @click="stop()">Stop</button>
      </div>
    </div>
    <div class="hint">
      Enter to send &middot; Shift+Enter for a new line &middot; attach images or text files by
      clicking + or dropping them.
    </div>
  </div>
</template>
