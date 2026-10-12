<script setup lang="ts">
/**
 * ComposerMenu - the single button on the left of the composer.
 *
 * One "+" opens one menu holding both things a message can need before it is
 * sent: the chat mode (which decides the system prompt and the tools the model
 * may call) and attachments. They used to be two adjacent buttons — a mode
 * dropdown and a bare "+" — which put two hit targets and two visual weights
 * on a row that already has Send.
 *
 * The file inputs live here rather than in Composer.vue because choosing
 * "Upload image" has to know the accept filter, and the parent only cares that
 * files were chosen.
 *
 * Interaction follows the dashboard's ExportMenu: click-outside and Escape
 * dismiss, the panel is not a modal, listeners are bound on mount and torn
 * down on unmount. Icons are inline SVG (src/icons.ts), not emoji.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { availableModes, currentMode, setMode, store } from '../store'
import { addFiles } from '../store'
import { Icon } from '../icons'

const open = ref(false)
const root = ref<HTMLElement | null>(null)
const imageInput = ref<HTMLInputElement | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)

const modes = computed(() => availableModes())
const active = computed(() => currentMode())

/** Anything but plain chat is worth surfacing on the trigger. */
const isSpecial = computed(() => active.value.id !== 'chat')

/** The trigger doubles as the mode readout, so it says which mode is on. */
const triggerLabel = computed(() =>
  isSpecial.value
    ? 'Mode: ' + active.value.label + ' — change mode or attach a file'
    : 'Add a mode or attachment',
)

function onDocClick(e: MouseEvent) {
  if (root.value && !root.value.contains(e.target as Node)) open.value = false
}
function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') open.value = false
}

onMounted(() => {
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKeydown)
})
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocClick)
  document.removeEventListener('keydown', onKeydown)
})

function pickMode(id: string) {
  setMode(id)
  open.value = false
}

function pickImages() {
  imageInput.value?.click()
}
function pickFiles() {
  fileInput.value?.click()
}

function onFileChange(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files) addFiles(input.files)
  input.value = '' // let the same file be re-picked
  open.value = false
}

/** Why a mode is offered at all: only 'search' and 'code' can be degraded. */
function hintFor(id: string): string {
  if (id === 'search' && store.toolAvail.ready && !store.toolAvail.searchConfigured) {
    return 'No search key configured'
  }
  if (id === 'code' && store.toolAvail.ready && !store.toolAvail.codeExecEnabled) {
    return 'Python execution disabled'
  }
  return ''
}
</script>

<template>
  <span ref="root" class="cmenu">
    <button
      class="cmenu-btn"
      type="button"
      :class="{ active: isSpecial, open }"
      :aria-expanded="open"
      aria-haspopup="menu"
      :aria-label="triggerLabel"
      :title="triggerLabel"
      data-testid="composer-menu"
      @click="open = !open"
    >
      <Icon :name="isSpecial ? active.icon : 'plus'" :size="19" />
      <span class="sr-only">{{ triggerLabel }}</span>
    </button>

    <span v-if="open" class="cmenu-panel" role="menu">
      <div class="cmenu-head">
        <span>Mode</span>
        <span class="cmenu-current">{{ active.label }}</span>
      </div>
      <button
        v-for="m in modes"
        :key="m.id"
        class="cmenu-item"
        :class="{ on: m.id === active.id }"
        role="menuitemradio"
        :aria-checked="m.id === active.id"
        :data-testid="'mode-' + m.id"
        @click="pickMode(m.id)"
      >
        <Icon :name="m.icon" :size="17" class="cmenu-ico" />
        <span class="cmenu-text">
          <span class="cmenu-label">{{ m.label }}</span>
          <span v-if="hintFor(m.id)" class="cmenu-hint">{{ hintFor(m.id) }}</span>
        </span>
        <Icon v-if="m.id === active.id" name="check" :size="16" class="cmenu-check" />
      </button>

      <div class="cmenu-sep" role="separator" />
      <div class="cmenu-head">Add</div>
      <button
        class="cmenu-item"
        role="menuitem"
        data-testid="attach-image"
        @click="pickImages"
      >
        <Icon name="image" :size="17" class="cmenu-ico" />
        <span class="cmenu-text"><span class="cmenu-label">Upload image</span></span>
      </button>
      <button class="cmenu-item" role="menuitem" data-testid="attach-file" @click="pickFiles">
        <Icon name="file" :size="17" class="cmenu-ico" />
        <span class="cmenu-text"><span class="cmenu-label">Upload file</span></span>
      </button>
    </span>

    <input
      ref="imageInput"
      type="file"
      accept="image/*"
      multiple
      hidden
      @change="onFileChange"
    />
    <input ref="fileInput" type="file" multiple hidden @change="onFileChange" />
  </span>
</template>