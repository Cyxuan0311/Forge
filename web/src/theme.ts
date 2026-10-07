/**
 * theme.ts - the palette, and the ECharts option fragments derived from it.
 *
 * Mode-dependent rather than a module constant, because the dashboard has a
 * light and a dark theme and ECharts has no notion of cascading a CSS custom
 * property into a canvas: every colour has to be handed to each option that
 * calls init(). Centralising them here is what keeps the six charts looking like
 * one page in either theme.
 *
 * The light palette is NOT the dark palette inverted. `--dim`, `--border` and
 * `--text` have to be chosen for contrast against a white panel; the dark
 * mid-tones fail that badly. Text colours here are checked against WCAG AA for
 * their normal size (4.5:1) against --panel, not eyeballed.
 */
import type { ResolvedTheme, ThemeMode } from './composables/theme'

export type { ResolvedTheme, ThemeMode }

export interface Palette {
  /** Page background; mirrors --bg in style.css. */
  bg: string
  /** Panel background; mirrors --panel. */
  panel: string
  /** Hairlines and axes; mirrors --border. */
  border: string
  /** Primary text; mirrors --text. */
  text: string
  /** Secondary text; mirrors --dim. */
  dim: string
  accent: string
  green: string
  amber: string
  red: string
  magenta: string
  cyan: string
  /** Tooltip surface, slightly lifted off --panel. */
  tipBg: string
}

/**
 * Dark is the original palette, unchanged.
 *
 * Text #e6e9ef on panel #161a21 is ~13:1. dim #8b93a3 on #161a21 is ~5.4:1,
 * which clears AA for normal text.
 */
const DARK: Palette = {
  bg: '#0f1115',
  panel: '#161a21',
  border: '#232935',
  text: '#e6e9ef',
  dim: '#8b93a3',
  accent: '#4da3ff',
  green: '#3ddc97',
  amber: '#ffb454',
  red: '#ff6b6b',
  magenta: '#c792ea',
  cyan: '#4dd0e1',
  tipBg: '#1b1f27',
}

/**
 * Light: text #1b1f24 on panel #ffffff is ~16:1; dim #5a6472 on white is
 * ~5.6:1. The series hues are darkened from the dark set so they stay legible on
 * white -- the originals are mid-tone and wash out against it, which shows up
 * most on `cyan` and `green`.
 */
const LIGHT: Palette = {
  bg: '#f4f6f9',
  panel: '#ffffff',
  border: '#d9dee6',
  text: '#1b1f24',
  dim: '#5a6472',
  accent: '#1668c4',
  green: '#0e8a5f',
  amber: '#a86308',
  red: '#c2321f',
  magenta: '#8144b0',
  cyan: '#0a7285',
  tipBg: '#ffffff',
}

const PALETTES: Record<ResolvedTheme, Palette> = { dark: DARK, light: LIGHT }

export function getPalette(theme: ResolvedTheme): Palette {
  return PALETTES[theme]
}

/** Per-series colours, assigned in order so a chart's series keep their hue. */
export function seriesColors(p: Palette): string[] {
  return [p.accent, p.green, p.amber, p.magenta, p.cyan, p.red]
}

export function axisStyle(p: Palette) {
  return {
    axisLine: { lineStyle: { color: p.border } },
    axisLabel: { color: p.dim, fontSize: 11 },
    splitLine: { lineStyle: { color: p.border, opacity: 0.5 } },
  }
}

export function tooltipStyle(p: Palette) {
  return {
    backgroundColor: p.tipBg,
    borderColor: p.border,
    textStyle: { color: p.text, fontSize: 12 },
  }
}

export function baseOption(p: Palette) {
  return {
    backgroundColor: 'transparent',
    textStyle: {
      color: p.text,
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    },
    animation: false,
  }
}