/**
 * Theme override test: the CSS custom properties must resolve per theme.
 *
 * This exists because of a real bug. The dashboard's chrome tokens used to live
 * in a THIRD `:root, [data-theme='dark']` block that had been appended after the
 * `[data-theme='light']` block. `:root` and `[data-theme='light']` have equal
 * specificity, so source order decides -- the appended block won, and the
 * header kept its dark `--header-bg` in light theme. The header looked fine in
 * dark theme and wrong in light, which is exactly the kind of thing that takes
 * an hour to spot by eye and one test to pin down.
 *
 * The assertion is on the *resolved* value the browser would use, computed by
 * walking the stylesheet rules the same way CSS does: for each declaration of
 * the property, keep the last rule whose selector matches `<html>`.
 *
 * Run with:  npm run test:theme   (requires `npm run build` first)
 */
import { readdirSync, readFileSync } from 'node:fs'
import { join, basename } from 'node:path'
import { JSDOM } from 'jsdom'

const ASSETS = 'dist/assets'
let failures = 0
const fail = (m) => { failures++; console.log(`  FAIL  ${m}`) }
const ok = (m) => console.log(`  ok    ${m}`)
const expect = (name, cond, detail = '') =>
  cond ? ok(name) : fail(`${name}${detail ? ` -- ${detail}` : ''}`)

// ---- load the built stylesheet -------------------------------------------

const cssFile = readdirSync(ASSETS).find((f) => /^index-.*\.css$/.test(f))
if (!cssFile) {
  console.error('FAIL no built css in dist/assets -- run npm run build first')
  process.exit(1)
}
const css = readFileSync(join(ASSETS, cssFile), 'utf8')

const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>')
const { document } = dom.window

const style = document.createElement('style')
style.textContent = css
document.head.appendChild(style)

// ---- resolve a custom property the way CSS does ---------------------------
/**
 * Last matching declaration wins (equal specificity here: every selector in the
 * theme blocks is a single pseudo-class or attribute selector).
 */
function resolveVar(prop) {
  const out = []
  const walk = (rules) => {
    for (const rule of rules) {
      if (rule instanceof dom.window.CSSStyleRule) {
        if (rule.selectorText.split(',').some((sel) => {
          try {
            return document.querySelector(sel.trim())
          } catch {
            return false
          }
        })) {
          const v = rule.style.getPropertyValue(prop)
          if (v) out.push({ sel: rule.selectorText.trim(), v: v.trim() })
        }
      } else if (rule.cssRules) {
        walk(rule.cssRules)  // @media / @supports
      }
    }
  }
  walk(document.styleSheets[0].cssRules)
  return out
}

function resolvedVar(prop) {
  const all = resolveVar(prop)
  return all.length ? all[all.length - 1].v : ''
}

// ---- assertions -----------------------------------------------------------

console.log(`--- ${basename(cssFile)} ---`)

for (const [theme, expectDark] of [['dark', true], ['light', false]]) {
  document.documentElement.dataset.theme = theme
  const hdr = resolvedVar('--header-bg')
  const panel = resolvedVar('--radius-panel')
  const shadow = resolvedVar('--shadow-panel')

  // "dark" luminance < 0.35, "light" > 0.6, from the 8-digit hex.
  const hex = (hdr.match(/^#([0-9a-f]{6})/i) || [])[1]
  if (!hex) {
    fail(`[${theme}] --header-bg is not a hex colour: ${hdr}`)
    continue
  }
  const r = parseInt(hex.slice(0, 2), 16) / 255
  const g = parseInt(hex.slice(2, 4), 16) / 255
  const b = parseInt(hex.slice(4, 6), 16) / 255
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

  expect(
    `[${theme}] header bg is ${expectDark ? 'dark' : 'light'}`,
    expectDark ? lum < 0.35 : lum > 0.6,
    `lum=${lum.toFixed(3)} value=${hdr}`,
  )
  expect(`[${theme}] panel radius resolved`, panel === '12px', panel)
  expect(`[${theme}] shadow resolved`, shadow.length > 0, shadow)

  console.log(`       ${theme}: --header-bg=${hdr}  --shadow-panel=${shadow}`)
}

// The background itself has to follow the theme too, or the header is the only
// thing that changes and the page reads as broken.
for (const [theme, want] of [['dark', '#0f1115'], ['light', '#f4f6f9']]) {
  document.documentElement.dataset.theme = theme
  expect(`[${theme}] --bg is ${want}`, resolvedVar('--bg') === want, resolvedVar('--bg'))
}

// #app .hdr must be the sticky, full-bleed header -- the descendant selector is
// load-bearing, since the real DOM is #app > div > .hdr.
const hdrRule = Array.from(document.styleSheets[0].cssRules).find(
  (r) => r instanceof dom.window.CSSStyleRule && /^#app\s+\.hdr$/.test(r.selectorText),
)
expect('sticky header rule exists with #app prefix', !!hdrRule)
if (hdrRule) expect('header is sticky', hdrRule.style.position === 'sticky', hdrRule.style.position)

// Both theme blocks must exist and in the right order.
const order = Array.from(document.styleSheets[0].cssRules)
  .filter((r) => r instanceof dom.window.CSSStyleRule)
  .map((r) => r.selectorText.trim())
expect('one :root block (no shadowing duplicate)', order.filter((s) => s.startsWith(':root')).length === 1,
  order.filter((s) => s.startsWith(':root')).join(' | '))

console.log()
if (failures) {
  console.log(`theme: FAIL (${failures} check(s))`)
  process.exit(1)
}
console.log('theme: PASS')