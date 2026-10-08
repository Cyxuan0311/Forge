<script setup lang="ts">
/**
 * MessageItem - one chat turn.
 *
 * Assistant output is rendered straight onto the page: no bubble, no border,
 * no role label -- just the markdown, the way the model wrote it. The user turn
 * keeps a filled bubble so the two sides stay distinguishable at a glance.
 *
 * The body comes from renderMd(), which HTML-escapes every input fragment
 * before adding any markup, so v-html here is safe against model output (and
 * pasted user input) containing HTML.
 */
import { computed, ref } from 'vue'
import { renderMd } from '../markdown'
import { activeConv, copyText, deleteMessage, regenerate, store } from '../store'
import ThinkingIndicator from './ThinkingIndicator.vue'
import type { ChatMessage, MessageStats } from '../types'

const props = defineProps<{ message: ChatMessage }>()

const copied = ref(false)
const canRegenerate = computed(() => props.message.role !== 'user')
const body = computed(() => renderMd(props.message.text || ''))

/** True only for the message currently being written into. */
const isStreaming = computed(
  () => store.streaming && store.activeId !== '' && props.message.role !== 'user',
)
/** Reasoning has started but no visible content yet -> "Thinking". */
const thinkingMode = computed<'waiting' | 'thinking' | 'writing'>(() => {
  if (!isStreaming.value) return 'waiting'
  return props.message.reasoning ? 'thinking' : 'waiting'
})

/**
 * The per-turn perf line. Server numbers (tokens, prefill, decode, tok/s) are
 * preferred; the client's own TTFT and wall clock fill in whatever the server
 * did not report, so the line is never half-empty.
 */
const perfBits = computed<string[]>(() => {
  const s: MessageStats | undefined = props.message.stats
  if (!s) return []
  const bits: string[] = []
  const ms = (v: number) => (v >= 1000 ? (v / 1000).toFixed(2) + 's' : Math.round(v) + 'ms')
  if (s.tokens_per_sec !== undefined && s.tokens_per_sec > 0)
    bits.push(s.tokens_per_sec.toFixed(1) + ' tok/s')
  if (s.prefill_ms !== undefined && s.prefill_ms > 0) bits.push('prefill ' + ms(s.prefill_ms))
  if (s.decode_ms !== undefined && s.decode_ms > 0) bits.push('decode ' + ms(s.decode_ms))
  if (s.ttft_ms !== undefined) bits.push('ttft ' + ms(s.ttft_ms))
  if (s.prompt_tokens !== undefined && s.completion_tokens !== undefined)
    bits.push(s.prompt_tokens + ' → ' + s.completion_tokens + ' tok')
  else if (s.completion_tokens !== undefined) bits.push(s.completion_tokens + ' tok')
  if (s.wall_ms !== undefined) bits.push('total ' + ms(s.wall_ms))
  return bits
})

async function copy(msg: string, done: () => void): Promise<void> {
  if (await copyText(msg)) {
    done()
  }
}

async function onCopy(): Promise<void> {
  if (!props.message.text) return
  await copy(props.message.text, () => {
    copied.value = true
    window.setTimeout(() => (copied.value = false), 1400)
  })
}

function onDelete(): void {
  if (activeConv.value) deleteMessage(activeConv.value.id, props.message.id)
}

function onRegenerate(): void {
  void regenerate(props.message.id)
}

/**
 * Code-block copy, wired by delegation because the blocks come from v-html.
 * The text is read straight off the DOM rather than smuggled through a data
 * attribute: nothing the model wrote ever has to survive a round trip through
 * an attribute to be recovered.
 */
async function onBodyClick(e: MouseEvent): Promise<void> {
  const btn = (e.target as HTMLElement).closest('.code-copy') as HTMLElement | null
  if (!btn) return
  const code = btn.closest('.codeblock')?.querySelector('code')?.textContent ?? ''
  const label = btn.textContent
  await copy(code, () => {
    btn.textContent = 'Copied'
    window.setTimeout(() => (btn.textContent = label), 1400)
  })
}
</script>

<template>
  <div class="msg" :class="message.role">
    <div class="avatar" aria-hidden="true">{{ message.role === 'user' ? 'You' : 'AI' }}</div>
    <div class="content">
      <details v-if="message.reasoning" class="reasoning">
        <summary>Thinking</summary>
        <div>{{ message.reasoning }}</div>
      </details>
      <ThinkingIndicator v-if="isStreaming && !message.text" :mode="thinkingMode" />
      <div class="body" v-html="body" @click="onBodyClick" />
      <div v-if="message.images && message.images.length" class="imgs">
        <img v-for="im in message.images" :key="im.name" :src="im.dataUrl" :alt="im.name" />
      </div>
      <div v-if="message.files && message.files.length" class="filechips">
        <span v-for="f in message.files" :key="f.name" class="chip">{{ f.name }}</span>
      </div>
      <div v-if="perfBits.length" class="perf">
        <span v-for="b in perfBits" :key="b" class="perf-bit">{{ b }}</span>
      </div>
      <div v-if="message.text" class="actions">
        <button class="act" :title="copied ? 'Copied' : 'Copy message'" @click="onCopy">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M9 9V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-4" />
            <rect x="5" y="9" width="12" height="12" rx="2" />
          </svg>
          <span>{{ copied ? 'Copied' : 'Copy' }}</span>
        </button>
        <button
          v-if="canRegenerate"
          class="act"
          title="Run this turn again"
          :disabled="store.streaming"
          @click="onRegenerate"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M21 12a9 9 0 1 1-3-6.7" />
            <path d="M21 4v5h-5" />
          </svg>
          <span>Retry</span>
        </button>
        <button class="act danger" title="Delete message" @click="onDelete">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 7h16" />
            <path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
            <path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12" />
          </svg>
          <span>Delete</span>
        </button>
      </div>
    </div>
  </div>
</template>
