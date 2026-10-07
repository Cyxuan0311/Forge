/**
 * palette.ts - provides the active ECharts palette to every chart component.
 *
 * provide/inject rather than a prop on each panel: there are four chart
 * components, each of which would otherwise need a `palette` prop threaded
 * through App.vue and re-emitted, and adding a fifth panel would mean touching
 * every call site again. Injection keeps the dependency implicit and
 * automatic.
 *
 * The palette is a computed, so a theme switch re-renders the consumers, each
 * of which rebuilds its option object, which EChart.vue pushes into the existing
 * instance with setOption(option, true). No chart is disposed and re-initialised
 * on a theme change -- that would drop the animation state and flicker.
 */
import { computed, inject, provide, type InjectionKey, type Ref } from 'vue'
import { getPalette, type Palette } from '../theme'
import type { ResolvedTheme } from './theme'

export const PALETTE_KEY: InjectionKey<Ref<Palette>> = Symbol('forge:palette')

/** Called once from App.vue. */
export function providePalette(theme: Ref<ResolvedTheme>): Ref<Palette> {
  const palette = computed(() => getPalette(theme.value))
  provide(PALETTE_KEY, palette)
  return palette
}

/** Called from each chart panel. */
export function usePalette(): Ref<Palette> {
  const p = inject(PALETTE_KEY)
  if (!p) {
    // A chart rendered outside App.vue would silently fall back to dark, which
    // is exactly the kind of bug that only shows up on one theme. Fail loudly.
    throw new Error(
      'usePalette() called outside of App.vue -- did you forget providePalette()?',
    )
  }
  return p
}