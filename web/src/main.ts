/**
 * main.ts - app bootstrap + global styles.
 */
import { createApp } from 'vue'
import App from './App.vue'
import './style.css'
import { applyTheme } from './composables/theme'

// Before createApp, so a light-preference machine never renders a dark frame
// first. Reads localStorage and matchMedia, both of which are available here.
applyTheme()

createApp(App).mount('#app')