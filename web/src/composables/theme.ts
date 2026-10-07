/**
 * theme.ts (composables) - the theme MODE, and how it resolves to a colour
 * scheme.
 *
 * Three states, deliberately not two:
 *
 *   light   the user picked light
 *   dark    the user picked dark
 *   system  follow the OS, re-resolving live when the OS setting changes
 *
 * `system` is the default so a fresh visitor gets whatever their machine is
 * already using, and an explicit choice always wins over it.
 */
import { computed, ref } from 'vue'

export type ThemeMode = 'light' | 'dark' | 'system'
export type ResolvedTheme = 'light' | 'dark'

/** Cycle order for the single button: light -> dark -> system -> light. */
export const THEME_CYCLE: ThemeMode[] = ['light', 'dark', 'system']

/**
 * localStorage key. Note this is scoped to the ORIGIN INCLUDING PORT, so if
 * the dashboard falls back to a different port because 9090 was busy, the
 * stored choice does not come with it and the theme resets to `system`.
 */
export const THEME_STORAGE_KEY = 'forge.dashboard.theme'

function isThemeMode(v: unknown): v is ThemeMode {
  return v === 'light' || v === 'dark' || v === 'system'
}

function readStoredMode(): ThemeMode {
  try {
    const v = localStorage.getItem(THEME_STORAGE_KEY)
    return isThemeMode(v) ? v : 'system'
  } catch {
    // localStorage throws in a sandboxed iframe / disabled-storage context.
    // Falling back to `system` is correct: the page still themes, it just
    // forgets the choice on reload.
    return 'system'
  }
}

const mode = ref<ThemeMode>(readStoredMode())

const systemPrefersDark = ref(true)
if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
  const mq = window.matchMedia('(prefers-color-scheme: dark)')
  systemPrefersDark.value = mq.matches
  // Fires when the user flips the OS theme while the page is open. Only
  // meaningful while the mode is `system`, but the listener stays attached so
  // switching from `system` to `dark` and back picks up the current value.
  const onChange = (e: MediaQueryListEvent) => {
    systemPrefersDark.value = e.matches
  }
  if (typeof mq.addEventListener === 'function') mq.addEventListener('change', onChange)
  else if (typeof mq.addListener === 'function') mq.addListener(onChange)
}

const resolved = computed<ResolvedTheme>(() =>
  mode.value === 'system' ? (systemPrefersDark.value ? 'dark' : 'light') : mode.value,
)

/**
 * Reflects the resolved theme onto <html>, which is what the CSS in
 * style.css keys off ([data-theme="light"]). Also sets color-scheme so form
 * controls, scrollbars and the canvas clear colour follow.
 *
 * Called from main.ts BEFORE createApp so a light-preference machine never sees
 * a dark flash.
 */
export function applyTheme(): void {
  const t = resolved.value
  const root = document.documentElement
  root.dataset.theme = t
  root.style.colorScheme = t
}

export function useTheme() {
  function set(next: ThemeMode): void {
    mode.value = next
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next)
    } catch {
      // Not being able to persist is not worth failing the click over.
    }
    applyTheme()
  }

  /** Advance to the next state in THEME_CYCLE. Used by the header button. */
  function cycle(): ThemeMode {
    const i = THEME_CYCLE.indexOf(mode.value)
    set(THEME_CYCLE[(i + 1) % THEME_CYCLE.length])
    return mode.value
  }

  return { mode, resolved, set, cycle, applyTheme }
}