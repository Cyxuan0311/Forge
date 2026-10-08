<script setup lang="ts">
/**
 * SettingsPanel - generation parameters and the system prompt. Values are
 * bound to the store on change and persisted immediately; they map 1:1 onto
 * the forge-server chat-completion request fields.
 */
import { save, store } from '../store'
import type { GenParams } from '../types'

const NUM_FIELDS: Array<{ key: keyof GenParams; label: string; step: string }> = [
  { key: 'temperature', label: 'Temperature', step: '0.1' },
  { key: 'max_tokens', label: 'Max tokens', step: '1' },
  { key: 'top_p', label: 'Top P', step: '0.05' },
  { key: 'top_k', label: 'Top K', step: '1' },
  { key: 'repeat_penalty', label: 'Repeat penalty', step: '0.05' },
]

function onNum(key: keyof GenParams, e: Event) {
  ;(store.params[key] as number) = parseFloat((e.target as HTMLInputElement).value)
  save()
}

function onSession(e: Event) {
  store.params.session_id = (e.target as HTMLInputElement).value
  save()
}

function onSystem(e: Event) {
  store.params.system = (e.target as HTMLTextAreaElement).value
  save()
}

function onStream(e: Event) {
  store.params.stream = (e.target as HTMLInputElement).checked
  save()
}
</script>

<template>
  <div class="panel-body">
    <div class="grid3">
      <label v-for="f in NUM_FIELDS" :key="f.key" class="field">
        <span>{{ f.label }}</span>
        <input
          type="number"
          :step="f.step"
          :value="store.params[f.key]"
          @change="onNum(f.key, $event)"
        />
      </label>
      <label class="field">
        <span>Session ID</span>
        <input type="text" :value="store.params.session_id" @change="onSession" />
      </label>
    </div>
    <label class="field" style="margin-top: 8px">
      <span>System prompt</span>
      <textarea rows="2" :value="store.params.system" @change="onSystem" />
    </label>
    <label class="row-inline">
      <input type="checkbox" :checked="store.params.stream" @change="onStream" /> Stream response
    </label>
  </div>
</template>
