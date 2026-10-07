/**
 * Vite config for the Forge dashboard.
 *
 * `base: './'` is load-bearing, not cosmetic: the built bundle is embedded and
 * served from /dashboard/index.html, so every asset URL has to be relative to
 * the document. An absolute "/assets/..." base would 404 against the mount
 * prefix.
 *
 * No CDN, no external fonts, no runtime network access. The dashboard is served
 * from a loopback port by a process that may well be running on an air-gapped
 * machine, so anything fetched from the internet would render as a broken page.
 *
 * sourcemap stays off: the .map files would be embedded into the C++ binary as
 * dead weight (EmbedDashboardAssets.cmake skips them anyway) and nothing is
 * going to attach a debugger to a browser tab pointed at a local inference
 * engine.
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
    chunkSizeWarningLimit: 2048,
    // No manualChunks. Forcing `echarts` into a named chunk made the namespace
    // import (import * as echarts from 'echarts') collapse to a single mangled
    // export, and it broke `vite build --ssr` outright with:
    //   "echarts" cannot be included in manualChunks because it is resolved as
    //   an external module
    // Rollup's own chunking is better here anyway: src/echarts.ts uses the
    // modular entry points, so only the charts and components the dashboard
    // actually draws end up in the bundle at all.
  },
  server: {
    port: 5173,
    proxy: {
      // `npm run dev` against a locally running forge-server / forge-cli.
      '/api': 'http://127.0.0.1:9090',
    },
  },
})