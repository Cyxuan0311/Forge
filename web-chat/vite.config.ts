/**
 * Vite config for the Forge chat UI.
 *
 * `base: './'` is load-bearing, not cosmetic: the built bundle is embedded and
 * served from /chat/index.html, so every asset URL has to be relative to the
 * document. An absolute "/assets/..." base would 404 against the mount prefix
 * (same reasoning as web/vite.config.ts for /dashboard/).
 *
 * No CDN, no external fonts, no runtime network access: the UI is served from
 * a process that may be running on an air-gapped machine.
 *
 * sourcemap stays off: the .map files would be embedded into the C++ binary as
 * dead weight (EmbedDashboardAssets.cmake skips them anyway).
 */
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  base: './',
  plugins: [vue()],
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    emptyOutDir: true,
    target: 'es2020',
    sourcemap: false,
  },
  server: {
    port: 5174,
    proxy: {
      // `npm run dev` against a locally running forge-server.
      '/v1': 'http://127.0.0.1:8080',
    },
  },
})
