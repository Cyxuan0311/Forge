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
  return s
}

function codeBlock(lang: string, code: string): string {
  return (
    '<div class="codeblock">' +
    '<div class="codebar"><span class="codelang">' +
    escapeHtml(lang || 'code') +
    '</span><button type="button" class="code-copy">Copy</button></div>' +
    '<pre class="code"><code>' +
    escapeHtml(code) +
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

/** True for lines that must not be swallowed as paragraph continuation. */
function isBlockStart(line: string): boolean {
  return (
    FENCE_RE.test(line) ||
    HEADING_RE.test(line) ||
    HR_RE.test(line) ||
    /^\s*>/.test(line) ||
    BULLET_RE.test(line)
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
