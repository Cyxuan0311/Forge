/**
 * Dependency-free markdown renderer for chat output.
 *
 * Safety model: every character of the model's answer is HTML-escaped BEFORE
 * any tag is produced, and the only tags in the result are ones this file
 * writes. Link hrefs are additionally scheme-checked (http/https/mailto), so a
 * `javascript:` or `data:` URL can never reach an attribute. That is what makes
 * v-html in MessageItem safe against model output (and pasted user input)
 * containing markup.
 *
 * Supported: ATX headings, fenced code blocks (each with a copy button),
 * GFM tables, blockquotes, thematic breaks, ordered/unordered lists (nested via
 * the recursive block parser), paragraphs, and inline bold / italic /
 * strikethrough / code / links / autolinks.
 *
 * Deliberate deviation from strict CommonMark: a single newline inside a
 * paragraph renders as <br>. Assistants use single newlines as line structure
 * (one sentence per line), and collapsing them loses that.
 */

import hljs from 'highlight.js/lib/core'
import katex from 'katex'

import bash from 'highlight.js/lib/languages/bash'
import c from 'highlight.js/lib/languages/c'
import cpp from 'highlight.js/lib/languages/cpp'
import csharp from 'highlight.js/lib/languages/csharp'
import cssLang from 'highlight.js/lib/languages/css'
import go from 'highlight.js/lib/languages/go'
import htmlXml from 'highlight.js/lib/languages/xml'
import java from 'highlight.js/lib/languages/java'
import javascript from 'highlight.js/lib/languages/javascript'
import json from 'highlight.js/lib/languages/json'
import markdownLang from 'highlight.js/lib/languages/markdown'
import python from 'highlight.js/lib/languages/python'
import rust from 'highlight.js/lib/languages/rust'
import sql from 'highlight.js/lib/languages/sql'
import typescript from 'highlight.js/lib/languages/typescript'
import yaml from 'highlight.js/lib/languages/yaml'

// A curated set covering what code models actually emit. Registering a subset of
// highlight.js keeps the bundle a few hundred KB smaller than the all-languages
// build, and `highlightAuto` falls back to this set for unregistered langs.
hljs.registerLanguage('bash', bash)
hljs.registerLanguage('sh', bash)
hljs.registerLanguage('shell', bash)
hljs.registerLanguage('c', c)
hljs.registerLanguage('cpp', cpp)
hljs.registerLanguage('c++', cpp)
hljs.registerLanguage('c#', csharp)
hljs.registerLanguage('csharp', csharp)
hljs.registerLanguage('cs', csharp)
hljs.registerLanguage('css', cssLang)
hljs.registerLanguage('go', go)
hljs.registerLanguage('html', htmlXml)
hljs.registerLanguage('xml', htmlXml)
hljs.registerLanguage('java', java)
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('js', javascript)
hljs.registerLanguage('json', json)
hljs.registerLanguage('markdown', markdownLang)
hljs.registerLanguage('md', markdownLang)
hljs.registerLanguage('python', python)
hljs.registerLanguage('py', python)
hljs.registerLanguage('rust', rust)
hljs.registerLanguage('rs', rust)
hljs.registerLanguage('sql', sql)
hljs.registerLanguage('typescript', typescript)
hljs.registerLanguage('ts', typescript)
hljs.registerLanguage('yaml', yaml)
hljs.registerLanguage('yml', yaml)

function escapeHtml(s: string): string {
  return String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      })[c] as string,
  )
}

function unescapeHtml(s: string): string {
  return String(s)
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
}

/** Only these schemes may end up in an href. */
function isSafeUrl(url: string): boolean {
  return /^(https?:\/\/|mailto:)/i.test(url)
}

const LINK_ATTRS = ' target="_blank" rel="noopener noreferrer nofollow"'

/**
 * Inline formatting on already-escaped text. Code spans are lifted out first so
 * their contents can never be read as emphasis or links.
 */
function renderInline(raw: string): string {
  const codes: string[] = []
  let s = raw.replace(/`([^`]+)`/g, (_m, code: string) => {
    codes.push(code)
    return '\u0000C' + (codes.length - 1) + '\u0000'
  })

  // Lift math before escaping so its contents are never HTML-escaped or read as
  // emphasis/links, exactly like code spans. Display ($$...$$) and inline ($...$)
  // are handled here (not in the block parser) so they work inside any inline
  // context; block-level $$ is handled separately in renderBlocks.
  const maths: string[] = []
  // A backslash-escaped dollar is a literal dollar, not a math delimiter.
  s = s.replace(/\\\$/g, '\u0001')
  // LaTeX \[ ... \] (display) and \( ... \) (inline) delimiters -- many models
  // emit these instead of the dollar forms. Lifted before the $ rules so a $
  // inside a \(..\) span is not mistaken for a delimiter.
  s = s.replace(/\\\[([\s\S]+?)\\\]/g, (_m, tex: string) => {
    maths.push(renderMath(tex, true))
    return '\u0000M' + (maths.length - 1) + '\u0000'
  })
  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (_m, tex: string) => {
    maths.push(renderMath(tex, false))
    return '\u0000M' + (maths.length - 1) + '\u0000'
  })
  s = s.replace(/\$\$([\s\S]+?)\$\$/g, (_m, tex: string) => {
    maths.push(renderMath(tex, true))
    return '\u0000M' + (maths.length - 1) + '\u0000'
  })
  // Inline $...$: opening $ not preceded by alnum/$, and not followed by
  // whitespace or a digit; closing $ not preceded by whitespace and not followed
  // by alnum/$. This keeps common currency ("$5 and $10") out of math while
  // accepting normal formulas.
  s = s.replace(
    /(?<![A-Za-z0-9$])\$(?![\s\d])([^$\n]+?)(?<!\s)\$(?![A-Za-z0-9$])/g,
    (_m, tex: string) => {
      maths.push(renderMath(tex, false))
      return '\u0000M' + (maths.length - 1) + '\u0000'
    },
  )

  s = escapeHtml(s)

  // [text](url) -- the href arrives escaped, so quotes inside it cannot break
  // out of the attribute; validate the real URL before emitting the anchor.
  s = s.replace(/\[([^\]\n]*)\]\(([^)\s]+)\)/g, (_m, text: string, href: string) => {
    if (!isSafeUrl(unescapeHtml(href))) return text
    return '<a href="' + href + '"' + LINK_ATTRS + '>' + text + '</a>'
  })

  // Bare URLs, but not ones already inside an href we just emitted.
  s = s.replace(
    /(^|[\s(])((?:https?:\/\/)[^\s<>()]+[^\s<>().,;:!?'"])/g,
    (_m, pre: string, url: string) =>
      pre + '<a href="' + url + '"' + LINK_ATTRS + '>' + url + '</a>',
  )

  s = s.replace(/\*\*\*([^*\n]+)\*\*\*/g, '<strong><em>$1</em></strong>')
  s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
  s = s.replace(/(^|[^\w*])\*([^*\n]+)\*(?!\w)/g, '$1<em>$2</em>')
  s = s.replace(/(^|[^\w_])_([^_\n]+)_(?!\w)/g, '$1<em>$2</em>')
  s = s.replace(/~~([^~]+)~~/g, '<del>$1</del>')

  // Restore code spans (their contents still need escaping).
  s = s.replace(
    /\u0000C(\d+)\u0000/g,
    (_m, i: string) => '<code class="inline">' + escapeHtml(codes[Number(i)]) + '</code>',
  )
  // Restore math (trusted KaTeX HTML) and literal backslash-dollars.
  s = s.replace(/\u0000M(\d+)\u0000/g, (_m, i: string) => maths[Number(i)])
  s = s.replace(/\u0001/g, '$')
  return s
}

/**
 * Render a LaTeX fragment to HTML with KaTeX. `throwOnError: false` turns a bad
 * formula into a red error span instead of throwing, so a typo in the model's
 * output never blanks the message. The output is trusted HTML (KaTeX escapes its
 * own input), so it is inserted verbatim after the rest of the inline/block
 * passes have run.
 */
function renderMath(tex: string, display: boolean): string {
  try {
    return katex.renderToString(tex.trim(), {
      displayMode: display,
      throwOnError: false,
      output: 'htmlAndMathml',
    })
  } catch {
    // renderToString only throws if throwOnError is true; keep this belt-and-braces.
    return escapeHtml(tex)
  }
}

function codeBlock(lang: string, code: string): string {
  const key = (lang || '').toLowerCase()
  let body: string
  // highlight.js escapes its own output, so we must NOT escapeHtml() the code
  // again -- doing so would double-encode and mangle the token spans.
  try {
    if (key && hljs.getLanguage(key)) {
      body = hljs.highlight(code, { language: key, ignoreIllegals: true }).value
    } else {
      body = hljs.highlightAuto(code).value
    }
  } catch {
    body = escapeHtml(code)
  }
  return (
    '<div class="codeblock">' +
    '<div class="codebar"><span class="codelang">' +
    escapeHtml(lang || 'code') +
    '</span><button type="button" class="code-copy">Copy</button></div>' +
    '<pre class="code"><code class="hljs' +
    (key ? ' language-' + escapeHtml(key) : '') +
    '">' +
    body +
    '</code></pre></div>'
  )
}

function indentOf(line: string): number {
  const m = /^\s*/.exec(line)
  return m ? m[0].length : 0
}

const FENCE_RE = /^\s*```+\s*([\w+#.-]*)\s*$/
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/
const HR_RE = /^\s*([-*_])\s*(\1\s*){2,}$/
const BULLET_RE = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/
const TABLE_DELIM_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/
// A line that opens a display-math block ($$\n.. or \[.. , incl. single-line).
const MATH_OPEN_RE = /^\s*(\$\$|\\\[)/

/** True for lines that must not be swallowed as paragraph continuation. */
function isBlockStart(line: string): boolean {
  return (
    FENCE_RE.test(line) ||
    HEADING_RE.test(line) ||
    HR_RE.test(line) ||
    /^\s*>/.test(line) ||
    BULLET_RE.test(line) ||
    MATH_OPEN_RE.test(line)
  )
}

function splitRow(line: string): string[] {
  let s = line.trim()
  if (s.startsWith('|')) s = s.slice(1)
  if (s.endsWith('|')) s = s.slice(0, -1)
  const cells: string[] = []
  let cur = ''
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '\\' && s[i + 1] === '|') {
      cur += '|'
      i++
      continue
    }
    if (s[i] === '|') {
      cells.push(cur)
      cur = ''
      continue
    }
    cur += s[i]
  }
  cells.push(cur)
  return cells.map((c) => c.trim())
}

function renderBlocks(lines: string[]): string[] {
  const out: string[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // ---- fenced code ------------------------------------------------------
    const fence = FENCE_RE.exec(line)
    if (fence) {
      const lang = fence[1] || ''
      const body: string[] = []
      i++
      while (i < lines.length && !/^\s*```+\s*$/.test(lines[i])) body.push(lines[i++])
      i++ // closing fence (or EOF)
      out.push(codeBlock(lang, body.join('\n')))
      continue
    }

    // ---- display math block ($$ ... $$) -----------------------------------
    // A line that opens with $$ (alone, or with content on the same line) starts
    // a display-math block, collected until a closing $$ line. Runs before the
    // paragraph pass so $$ fenced math is never wrapped in <p>.
    if (/^\s*\$\$\s*$/.test(line) || /^\s*\$\$[^\s]/.test(line)) {
      const body: string[] = []
      const openInline = line.replace(/^\s*\$\$/, '').replace(/\$\$\s*$/, '').trim()
      // A single `$$ ... $$` on one line is self-contained; only fall through to
      // collecting further lines when the opening line has no closing $$ (i.e. a
      // standalone `$$` fence followed by content on later lines).
      const selfContained = openInline.length > 0 && /\$\$\s*$/.test(line)
      if (openInline) body.push(openInline)
      i++
      if (!selfContained) {
        while (i < lines.length && !/^\s*\$\$\s*$/.test(lines[i])) body.push(lines[i++])
        if (i < lines.length) i++ // closing $$
      }
      out.push('<div class="math-display">' + renderMath(body.join('\n'), true) + '</div>')
      continue
    }

    // ---- display math block (\[ ... \]) -----------------------------------
    // Same idea as the $$ block above, for the \[ ... \] delimiters that many
    // models emit. The closing \] may sit on its own line or end a content line.
    if (/^\s*\\\[/.test(line)) {
      const body: string[] = []
      let rest = line.replace(/^\s*\\\[/, '')
      let closed = false
      for (;;) {
        const m = /\\\]\s*$/.exec(rest)
        if (m) {
          body.push(rest.slice(0, m.index))
          closed = true
          break
        }
        body.push(rest)
        i++
        if (i >= lines.length) break
        rest = lines[i]
      }
      if (closed) i++
      out.push('<div class="math-display">' + renderMath(body.join('\n'), true) + '</div>')
      continue
    }

    if (!line.trim()) {
      i++
      continue
    }

    // ---- thematic break ---------------------------------------------------
    if (HR_RE.test(line)) {
      out.push('<hr />')
      i++
      continue
    }

    // ---- heading ----------------------------------------------------------
    const heading = HEADING_RE.exec(line)
    if (heading) {
      const level = heading[1].length
      out.push('<h' + level + '>' + renderInline(heading[2]) + '</h' + level + '>')
      i++
      continue
    }

    // ---- table ------------------------------------------------------------
    if (line.indexOf('|') >= 0 && i + 1 < lines.length && TABLE_DELIM_RE.test(lines[i + 1])) {
      const head = splitRow(line)
      i += 2
      const rows: string[][] = []
      while (i < lines.length && lines[i].indexOf('|') >= 0 && lines[i].trim()) {
        rows.push(splitRow(lines[i++]))
      }
      const th = head.map((c) => '<th>' + renderInline(c) + '</th>').join('')
      const tb = rows
        .map((r) => {
          // Pad short rows so every <tr> has the header's column count.
          const cells = head.map((_, k) => '<td>' + renderInline(r[k] ?? '') + '</td>')
          return '<tr>' + cells.join('') + '</tr>'
        })
        .join('')
      out.push(
        '<div class="tablewrap"><table><thead><tr>' +
          th +
          '</tr></thead><tbody>' +
          tb +
          '</tbody></table></div>',
      )
      continue
    }

    // ---- blockquote -------------------------------------------------------
    if (/^\s*>/.test(line)) {
      const body: string[] = []
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        body.push(lines[i++].replace(/^\s*>\s?/, ''))
      }
      out.push('<blockquote>' + renderBlocks(body).join('') + '</blockquote>')
      continue
    }

    // ---- list -------------------------------------------------------------
    const first = BULLET_RE.exec(line)
    if (first) {
      const ordered = /\d/.test(first[2])
      const base = first[1].length
      const items: string[] = []
      // A blank line between top-level items makes the whole list "loose": items
      // then keep their paragraph wrapper, exactly like GitHub.
      let loose = false
      while (i < lines.length) {
        const m = BULLET_RE.exec(lines[i])
        const ind = indentOf(lines[i])
        if (m && ind === base) {
          items.push(m[3])
          i++
          continue
        }
        if (m && ind > base) {
          // Nested item: hand it to the recursive parser with the item's text.
          if (items.length) items[items.length - 1] += '\n' + lines[i].slice(ind)
          i++
          continue
        }
        if (!lines[i].trim()) {
          const next = lines[i + 1]
          if (next === undefined) break
          const nind = indentOf(next)
          const nbull = BULLET_RE.test(next)
          if (nbull && nind > base) {
            // Blank line inside the current item.
            if (items.length) items[items.length - 1] += '\n'
            i++
            continue
          }
          if (nbull && nind === base) {
            // Blank line between items: same list, loose rendering.
            loose = true
            i++
            continue
          }
          // Indented lazy continuation of the current item.
          if (nind > base && !isBlockStart(next)) {
            if (items.length) items[items.length - 1] += '\n' + next.trim()
            i += 2
            loose = true
            continue
          }
          break
        }
        // Lazy continuation line of the current item.
        if (items.length && !isBlockStart(lines[i])) {
          items[items.length - 1] += '\n' + lines[i].trim()
          i++
          continue
        }
        break
      }
      const tag = ordered ? 'ol' : 'ul'
      const body = items
        .map((t) => {
          const inner = renderBlocks(t.split('\n')).join('')
          if (!loose && inner.startsWith('<p>')) {
            // Tight list: drop the leading paragraph wrapper so a one-line item
            // does not gain vertical space the source never had.
            const close = inner.indexOf('</p>')
            return '<li>' + inner.slice(3, close) + inner.slice(close + 4) + '</li>'
          }
          return '<li>' + inner + '</li>'
        })
        .join('')
      out.push('<' + tag + '>' + body + '</' + tag + '>')
      continue
    }

    // ---- paragraph --------------------------------------------------------
    const para: string[] = []
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) {
      para.push(lines[i++])
    }
    if (para.length) out.push('<p>' + para.map((l) => renderInline(l)).join('<br />') + '</p>')
  }

  return out
}

export function renderMd(src: string): string {
  if (!src) return ''
  return renderBlocks(src.replace(/\r\n?/g, '\n').split('\n')).join('')
}
