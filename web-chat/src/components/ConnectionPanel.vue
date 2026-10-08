<script setup lang="ts">
/**
 * ConnectionPanel - base URL / API key / model selection.
 *
 * The base URL is usually left empty: the UI is embedded and served from
 * /chat/ on the same origin as /v1. Test connection lists the models the
 * server exposes and auto-selects the first one when nothing is picked.
 */
import { ref } from 'vue'
import { save, store, testConnection } from '../store'

const testing = ref(false)
const status = ref('')

function onBase(e: Event) {
  store.connection.baseUrl = (e.target as HTMLInputElement).value
  save()
}

function onKey(e: Event) {
  store.connection.apiKey = (e.target as HTMLInputElement).value
  save()
}

function onModel(e: Event) {
  store.connection.model = (e.target as HTMLSelectElement).value
  save()
}

async function test() {
  testing.value = true
  status.value = 'Testing...'
  await testConnection()
  testing.value = false
  status.value =
    store.connState === 'ok'
      ? 'Connected - ' + store.models.length + ' model(s) available'
      : 'Connection failed - check the base URL and API key'
}
</script>

<template>
  <div class="panel-body">
    <div class="grid2">
      <label class="field">
        <span>Base URL (leave empty for same origin)</span>
        <input
          type="text"
          placeholder="e.g. http://127.0.0.1:8080"
          :value="store.connection.baseUrl"
          @change="onBase"
        />
      </label>
      <label class="field">
        <span>Model</span>
        <select :value="store.connection.model" @change="onModel">
          <option value="">model (auto)</option>
          <option v-for="m in store.models" :key="m" :value="m">{{ m }}</option>
        </select>
      </label>
      <label class="field">
        <span>API key (optional)</span>
        <input
          type="text"
          placeholder="Bearer token"
          :value="store.connection.apiKey"
          @change="onKey"
        />
      </label>
    </div>
    <div class="models-status">{{ status }}</div>
    <div class="row-inline">
      <button class="icon-btn" :disabled="testing" @click="test">Test connection</button>
      <span class="hint">The UI calls the OpenAI-compatible API at &lt;base&gt;/v1.</span>
    </div>
  </div>
</template>
