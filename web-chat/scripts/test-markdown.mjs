// Render/safety checks for src/markdown.ts.
//
// The renderer feeds v-html in MessageItem, so the interesting cases are the
// adversarial ones: markup in the model's answer must come out escaped, and a
// hostile href must never survive. Run with: node scripts/test-markdown.mjs
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { build } from 'esbuild'

const here = dirname(fileURLToPath(import.meta.url))
// markdown.ts now pulls in highlight.js + katex, so transformSync (which leaves
// bare imports unresolved) is not enough -- bundle them into the test module.
const result = await build({
  entryPoints: [join(here, '..', 'src', 'markdown.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  write: false,
  logLevel: 'silent',
})
const js = result.outputFiles[0].text
const mod = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))
const renderMd = mod.renderMd

let failed = 0
function check(name, cond, got) {
  if (cond) {
    console.log('  ok   ' + name)
  } else {
    failed++
    console.log('  FAIL ' + name + (got === undefined ? '' : '\n       got: ' + got))
  }
}
function has(name, needle, src2) {
  check(name, (src2 ?? '').indexOf(needle) >= 0, src2)
}
function hasNot(name, needle, src2) {
  check(name, (src2 ?? '').indexOf(needle) < 0, src2)
}

console.log('=== injection / escaping ===')
const xss = renderMd('<img src=x onerror=alert(1)> and <script>alert(2)</script>')
hasNot('raw <img> tag is escaped', '<img', xss)
hasNot('raw <script> tag is escaped', '<script', xss)
has('escaped text survives', '&lt;script&gt;', xss)

hasNot('javascript: href dropped', 'href="javascript:', renderMd('[click](javascript:alert(1))'))
hasNot('data: href dropped', 'href="data:', renderMd('[x](data:text/html,<script>)'))
has('http href kept', 'href="https://example.com/a"', renderMd('[a](https://example.com/a)'))
hasNot('safe link carries rel=noopener', 'rel="noopener"', renderMd('[a](https://x.test)'))
has('autolink bare url', 'href="https://x.test/p"', renderMd('see https://x.test/p now'))

console.log('=== inline ===')
has('bold', '<strong>hi</strong>', renderMd('**hi**'))
has('italic', '<em>hi</em>', renderMd('*hi*'))
has('strike', '<del>hi</del>', renderMd('~~hi~~'))
has('inline code', '<code class="inline">a*b</code>', renderMd('`a*b`'))
has('code span content is not emphasised', '<code class="inline">*raw*</code>', renderMd('`*raw*`'))
has('single newline becomes <br>', 'one<br />two', renderMd('one\ntwo'))
hasNot('placeholder sentinel cannot leak', '\\u0000', renderMd('plain text  C0  here'))

console.log('=== blocks ===')
has(
  'fenced code with language',
  '<span class="codelang">python</span>',
  renderMd('```python\nx=1\n```'),
)
has('code body escaped', '&lt;', renderMd('```html\n<div>\n```'))
has('copy button emitted', 'class="code-copy"', renderMd('```\nx\n```'))
has('unclosed fence still closes', '</code></pre>', renderMd('```\nunfinished'))
has('atx heading', '<h2>Title</h2>', renderMd('## Title'))
has('heading level 6', '<h6>deep</h6>', renderMd('###### deep'))
has('unordered list', '<ul><li>one</li><li>two</li></ul>', renderMd('- one\n- two'))
has('ordered list', '<ol><li>first</li></ol>', renderMd('1. first'))
has('nested list', '<ul><li>a<ul><li>b</li></ul></li></ul>', renderMd('- a\n  - b'))
has('blockquote', '<blockquote><p>quoted</p></blockquote>', renderMd('> quoted'))
has('thematic break', '<hr />', renderMd('---'))
has('table header + body', '<th>A</th>', renderMd('| A | B |\n| --- | --- |\n| 1 | 2 |'))
has('table body cell', '<td>1</td>', renderMd('| A | B |\n| --- | --- |\n| 1 | 2 |'))
has('escaped pipe inside cell', '<td>a | b</td>', renderMd('| A |\n| --- |\n| a \\| b |'))
hasNot('table delimiter row is not content', '---', renderMd('| A |\n| --- |\n| 1 |'))
has(
  'blank line between items keeps one loose list',
  '<li><p>a</p></li><li><p>b</p></li>',
  renderMd('- a\n\n- b'),
)
has('tight item has no <p>', '<li>plain</li>', renderMd('- plain'))

console.log('=== code highlight ===')
const hi = renderMd('```python\nfor x in range(3):\n    print(x)\n```')
has('python keyword is highlighted', 'hljs-keyword', hi)
has('highlighted code still escapes', '&lt;', renderMd('```html\n<div>\n```'))
hasNot('highlighted output has no raw script tag', '<script', renderMd('```js\nx=1\n```'))

console.log('=== math ===')
const inl = renderMd('energy is $E = mc^2$ here')
has('inline math rendered', 'katex', inl)
hasNot('inline math not left as literal dollar', inl.indexOf('$E = mc^2$') >= 0, inl)
const disp = renderMd('$$\n\\frac{a}{b}\n$$')
has('display math rendered', 'katex-display', disp)
hasNot('display math does not leak delimiters', '$$', disp)
const disp1 = renderMd('the answer is $$x^2$$ done')
has('single-line display math rendered', 'katex-display', disp1)
hasNot('single-line display math does not leak delimiters', '$$', disp1)
hasNot('currency is not mistaken for math', 'katex', renderMd('the cost is $5 and $10 total'))
hasNot('escaped dollar stays literal', 'katex', renderMd('escape a \\$ sign'))
hasNot('math cannot inject a <script> tag', '<script', renderMd('mix $<script>alert(1)</script>$ with text'))

// LaTeX \( ... \) / \[ ... \] delimiters (what many models emit instead of $).
const inlParen = renderMd('energy is \\(E = mc^2\\) here')
has('inline \\(...\\) rendered', 'katex', inlParen)
hasNot('inline \\(...\\) not left literal', inlParen.indexOf('\\(') >= 0, inlParen)
const dispBracket = renderMd('\\[\n\\frac{a}{b}\n\\]')
has('display \\[...\\] rendered', 'katex-display', dispBracket)
hasNot('display \\[...\\] does not leak delimiters', dispBracket.indexOf('\\[') >= 0, dispBracket)
const dispBracket1 = renderMd('the answer is \\[x^2\\] done')
has('single-line \\[...\\] rendered', 'katex-display', dispBracket1)
hasNot('single-line \\[...\\] does not leak delimiters', dispBracket1.indexOf('\\[') >= 0, dispBracket1)

console.log('=== structure ===')
hasNot('no empty paragraph for blank input', '<p>', renderMd(''))
hasNot('empty input renders nothing', '<', renderMd(''))
has('multi-paragraph split', '</p><p>', renderMd('one\n\ntwo'))

if (failed) {
  console.log('\nmarkdown: FAIL (' + failed + ' check(s))')
  process.exit(1)
}
console.log('\nmarkdown: PASS')
