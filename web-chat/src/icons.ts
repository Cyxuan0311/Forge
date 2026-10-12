/**
 * Inline SVG icons.
 *
 * These were emoji before. Emoji render differently per platform (and often as
 * full-colour glyphs that fight the monochrome UI), cannot be sized or coloured
 * with CSS, and break the flat line-art look of the rest of the composer. Each
 * icon is 24x24, stroked with `currentColor` so it inherits the button's text
 * colour in both themes.
 *
 * Exposed as a functional component so callers write `<Icon name="search" />`
 * in templates with no import gymnastics beyond the one at the top.
 */
import { h } from 'vue'

/** Inner SVG markup per name (paths only; the wrapper supplies the attributes). */
const PATHS: Record<string, string> = {
  // Speech bubble — plain chat.
  chat: '<path d="M21 14a2 2 0 0 1-2 2H8l-4 4V6a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2z"/>',

  // Magnifier — web search.
  search:
    '<circle cx="11" cy="11" r="7"/><path d="M20.5 20.5l-4.6-4.6"/>',

  // Pen nib — writing.
  writing:
    '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z"/>',

  // Angle brackets — coding.
  code: '<path d="M16 18l6-6-6-6"/><path d="M8 6l-6 6 6 6"/>',

  // Landscape — attach image.
  image:
    '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>' +
    '<path d="M21 15l-5-5L5 21"/>',

  // Document — attach a file.
  file:
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>' +
    '<path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h5"/>',

  // Plus — the single composer-menu trigger.
  plus: '<path d="M12 5v14"/><path d="M5 12h14"/>',

  // Tick — marks the active mode.
  check: '<path d="M20 6L9 17l-5-5"/>',

  // Cross — dismiss a staged attachment / close the sidebar.
  close: '<path d="M18 6L6 18"/><path d="M6 6l12 12"/>',

  // Hamburger — toggles the conversation drawer.
  menu: '<path d="M3 6h18"/><path d="M3 12h18"/><path d="M3 18h18"/>',

  // Arrow up — the send affordance, drawn as a chevron on a stem.
  send: '<path d="M12 20V6"/><path d="M5 13l7-7 7 7"/>',

  // Square — the stop affordance while a reply streams.
  stop: '<rect x="7" y="7" width="10" height="10" rx="2"/>',

  // Sliders — generation settings.
  settings:
    '<path d="M4 6h10"/><path d="M18 6h2"/><circle cx="16" cy="6" r="2"/>' +
    '<path d="M4 12h4"/><path d="M12 12h8"/><circle cx="10" cy="12" r="2"/>' +
    '<path d="M4 18h10"/><path d="M18 18h2"/><circle cx="16" cy="18" r="2"/>',

  // Plug — server connection.
  plug:
    '<path d="M9 3v6"/><path d="M15 3v6"/><path d="M6 9h12v3a6 6 0 0 1-6 6 6 6 0 0 1-6-6z"/>' +
    '<path d="M12 18v3"/>',

  // Sun / moon — theme toggle.
  sun:
    '<circle cx="12" cy="12" r="4"/>' +
    '<path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.9 4.9l1.4 1.4"/>' +
    '<path d="M17.7 17.7l1.4 1.4"/><path d="M2 12h2"/><path d="M20 12h2"/>' +
    '<path d="M4.9 19.1l1.4-1.4"/><path d="M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',

  // Copy — duplicate a message.
  copy:
    '<path d="M9 9V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-4"/>' +
    '<rect x="5" y="9" width="12" height="12" rx="2"/>',

  // Circular arrow — regenerate an answer.
  retry:
    '<path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 4v5h-5"/>',

  // Chevron — the "jump to the newest message" control over the composer.
  'chevron-down': '<path d="M6 9l6 6 6-6"/>',

  // Bin — delete a conversation / a turn.
  trash:
    '<path d="M4 7h16"/><path d="M9 7V5h6v2"/>' +
    '<path d="M6 7l1 13h10l1-13"/>',
}

export const ICON_NAMES = Object.keys(PATHS)

export function Icon(props: { name: string; size?: number; class?: string }): unknown {
  const size = props.size ?? 18
  return h('svg', {
    class: props.class,
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    'stroke-width': 1.7,
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
    'aria-hidden': 'true',
    innerHTML: PATHS[props.name] ?? '',
  })
}