/**
 * Chat modes.
 *
 * A mode is a named bundle of (system prompt, tool set). "Chat" is the plain
 * one — no tools, no extra instructions — and stays the default so existing
 * conversations behave exactly as before. The others declare which of the
 * server's tools they need, so `available()` can hide a mode whose backing tool
 * the server has disabled (no search key, code exec off) instead of letting the
 * user pick something that will fail on the first call.
 *
 * The tool NAMES must match the server registry (see
 * include/forge/agent/tool_registry.h). Availability is not hardcoded: it is
 * read from GET /v1/tools at startup, so a mode opens up the moment the
 * operator configures a key.
 */
import type { Connection } from './types'
import { listTools } from './tools'
import type { ToolSpecInfo } from './tools'

export interface ModeDef {
  id: string
  label: string
  /** Key into src/icons.ts (SVG), not an emoji. */
  icon: string
  /** Prepended to the user's system prompt while this mode is active. */
  systemPrompt: string
  /** Server tool names this mode needs. Empty = plain chat, no tools sent. */
  tools: string[]
  /**
   * When true the mode still works with no tools configured — the instructions
   * alone shape the answer (writing), so the mode stays selectable.
   */
  optionalTools?: boolean
}

export const MODES: ModeDef[] = [
  {
    id: 'chat',
    label: 'Chat',
    icon: 'chat',
    systemPrompt: '',
    tools: [],
  },
  {
    id: 'search',
    label: 'Web search',
    icon: 'search',
    systemPrompt:
      'Answer using web search. Search when a fact may have changed or you are unsure; ' +
      'rely on your own knowledge for stable topics. Cite sources as [1], [2] matching ' +
      'the numbered results. If a search fails or returns nothing, say so plainly ' +
      'instead of inventing results, and answer from your own knowledge instead.',
    tools: ['web_search'],
    optionalTools: true,
  },
  {
    id: 'writing',
    label: 'Writing',
    icon: 'writing',
    systemPrompt:
      'You are a writing assistant. Help draft, edit and polish prose. Match the ' +
      'requested tone and audience, prefer concrete language over filler, and keep ' +
      'the user\'s own voice when editing rather than replacing it with your own. ' +
      'When you revise text, show the improved version and briefly say what changed.',
    tools: [],
  },
  {
    id: 'code',
    label: 'Coding',
    icon: 'code',
    systemPrompt:
      'You are a coding assistant. Give complete, runnable answers: state the language, ' +
      'include imports and error handling, and use the Python tool to actually run code ' +
      'when a calculation, a repro, or verifying a fix would help. Never present code ' +
      'as tested unless you ran it. Explain non-obvious choices briefly.',
    tools: ['run_python'],
    optionalTools: true,
  },
]

export const DEFAULT_MODE = 'chat'

export function modeById(id: string): ModeDef {
  return MODES.find((m) => m.id === id) || MODES[0]
}

/**
 * Tool availability, as reported by GET /v1/tools. Defaults to "all unknown"
 * so the first paint is not blocked on the probe and no mode is wrongly hidden.
 */
export interface ToolAvailability {
  ready: boolean
  /** tool name -> can the server run it right now? */
  available: Record<string, boolean>
  /** tool name -> why not (shown as a hint on a disabled mode). */
  reason: Record<string, string>
  /** Full spec per tool, so the request's `tools` array matches the server. */
  specs: Record<string, ToolSpecInfo>
  codeExecEnabled: boolean
  searchConfigured: boolean
}

export const UNKNOWN_AVAILABILITY: ToolAvailability = {
  ready: false,
  available: {},
  reason: {},
  specs: {},
  codeExecEnabled: false,
  searchConfigured: false,
}

/** Fetch the tool catalogue. Never throws: a failure means "unknown". */
export async function fetchAvailability(conn: Connection): Promise<ToolAvailability> {
  try {
    const info = await listTools(conn)
    const available: Record<string, boolean> = {}
    const reason: Record<string, string> = {}
    const specs: Record<string, ToolSpecInfo> = {}
    for (const t of info.tools) {
      available[t.name] = t.available
      if (t.unavailable_reason) reason[t.name] = t.unavailable_reason
      specs[t.name] = t
    }
    return {
      ready: true,
      available,
      reason,
      specs,
      codeExecEnabled: info.code_exec_enabled,
      searchConfigured: info.search_configured,
    }
  } catch {
    return UNKNOWN_AVAILABILITY
  }
}

/** Can this mode be offered given what the server reports it supports? */
export function modeAvailable(mode: ModeDef, avail: ToolAvailability): boolean {
  if (!mode.tools.length || mode.optionalTools) return true
  // While the probe is in flight everything is offered; a wrong offer is
  // corrected on the next turn, whereas hiding a working mode is not.
  if (!avail.ready) return true
  return mode.tools.every((n) => avail.available[n] !== false)
}