#!/usr/bin/env python3
"""Phi-4 (15B, llama arch) interactive chat using Forge.

Text-only inference for llama-family GGUF models such as Phi-4 Q2_K_L.
Uses Forge's built-in Tokenizer and the model's GGUF jinja chat template,
so no external tokenizer files or transformers dependency is required.

Usage:
  python examples/legacy/phi4_inference.py
  python examples/legacy/phi4_inference.py --device cpu --gpu-layers 0
  python examples/legacy/phi4_inference.py --device cuda
  python examples/legacy/phi4_inference.py --model-path /mnt/g/AI/phi-4-GGUF/phi-4-Q2_K_L.gguf
  echo "The capital of France is" | python examples/legacy/phi4_inference.py --prompt -

Default device is cuda (all layers offloaded to GPU). Use --device cpu for
CPU-only inference, or --device cuda --gpu-layers N for hybrid offload.

Interactive commands:
  /quit   - Exit the chat
  /clear  - Clear conversation history
  /help   - Show help message
"""

import os
import sys
import time

# Allow running from within examples/legacy/ — add examples/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse

import chat_utils as chat_utils_mod
from chat_utils import (
    add_common_args,
    interactive_chat,
    load_model_and_tokenize,
    resolve_model_path,
)

MODEL_PATH = "/mnt/g/AI/phi-4-GGUF/phi-4-Q2_K_L.gguf"

# <|im_end|> (EOS) for the Phi-4 ChatML-style template
PHI4_EOS_ID = 100265

_engine_cache = {}


def _template_engine(tokenizer):
    """Return a cached ChatTemplateEngine (GGUF jinja template) for the tokenizer."""
    import forge

    engine = _engine_cache.get(id(tokenizer))
    if engine is None:
        engine = forge.ChatTemplateEngine.from_tokenizer(tokenizer)
        _engine_cache[id(tokenizer)] = engine
    return engine


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply the model's GGUF jinja chat template (Phi-4 ChatML variant)."""
    engine = _template_engine(tokenizer)

    inp = _template_input(messages, add_generation_prompt)
    return engine.apply(inp)


def _template_input(messages, add_generation_prompt):
    import forge

    inp = forge.ChatTemplateInput()
    inp.add_generation_prompt = add_generation_prompt
    msgs = []
    for m in messages:
        tm = forge.ChatTemplateMessage()
        tm.role = m["role"]
        tm.content = m["content"]
        msgs.append(tm)
    inp.messages = msgs
    return inp


def run_once(model, tokenizer, args, prompt_text):
    """Non-interactive single-shot generation (stdin pipe / --prompt)."""
    import forge

    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = apply_chat_template(tokenizer, messages)

    ctx = model.create_context(
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_layers=args.gpu_layers,
    )

    gen_cfg = forge.GenerationConfig()
    gen_cfg.max_new_tokens = args.max_new_tokens
    gen_cfg.temperature = args.temperature
    gen_cfg.top_k = args.top_k
    gen_cfg.top_p = args.top_p
    gen_cfg.repeat_penalty = args.repeat_penalty
    gen_cfg.do_sample = args.temperature > 0
    gen_cfg.eos_token_id = PHI4_EOS_ID

    print("Assistant: ", end="", flush=True)
    t0 = time.time()
    result = ctx.generate(prompt_ids, gen_cfg)
    elapsed = time.time() - t0
    response_ids = list(result.token_ids)
    print()
    text = tokenizer.decode(response_ids, skip_special=True)
    print(text)

    n = len(response_ids)
    print(f"\n[{n} tokens in {elapsed:.2f}s = {n / elapsed if elapsed > 0 else 0:.2f} tok/s]")

    if chat_utils_mod.profiling_enabled:
        stats = ctx.memory_stats()
        print(
            f"KV Cache: dtype={stats.get('kv_cache_dtype', 'unknown')}, "
            f"size: {stats.get('kv_cache_nbytes', 0) / 1024 / 1024:.1f} MB"
        )
        chat_utils_mod.print_cpp_profiler_summary()
    return text


def main():
    parser = argparse.ArgumentParser(description="Phi-4 inference with Forge")
    add_common_args(parser, gpu_layers_default=-1, temperature_default=0.7)
    parser.add_argument("--prompt", default=None,
                        help="Single-shot prompt. Use '-' to read from stdin. "
                             "Omit for interactive chat.")
    args = parser.parse_args()

    if args.profile:
        chat_utils_mod.profiling_enabled = True
        print("[Profiling enabled - Python timing + C++ PerfProfiler]")

    model_path = resolve_model_path(args, [MODEL_PATH])
    model, tokenizer = load_model_and_tokenize(args, model_path)

    if model is None:
        return

    if args.prompt is not None:
        if args.prompt == "-":
            prompt_text = sys.stdin.read()
        else:
            prompt_text = args.prompt
        if not prompt_text.strip():
            print("ERROR: empty prompt")
            return
        run_once(model, tokenizer, args, prompt_text)
    else:
        interactive_chat(
            model, tokenizer, args,
            apply_chat_template_fn=apply_chat_template,
            model_name="Phi-4",
            eos_token_id=PHI4_EOS_ID,
            system_msg={"role": "system", "content": "You are a helpful assistant."},
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
