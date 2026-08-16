#!/usr/bin/env python3
"""Qwen3.8-27B (qwen35 arch) interactive chat using Forge.

Text-only inference for qwen35-family GGUF models (e.g. Qwen3.8-27B IQ4_XS).
Uses Forge's built-in Tokenizer loaded directly from the GGUF file.

Usage:
  python examples/legacy/qwen38_inference.py --model-path /mnt/g/AI/Qwen3.8-27B/Qwen3.8-27B-IQ4_XS.gguf
  python examples/legacy/qwen38_inference.py --device cpu --gpu-layers 0
  echo "The capital of France is" | python examples/legacy/qwen38_inference.py --prompt -

Interactive commands:
  /quit   - Exit the chat
  /clear  - Clear conversation history
  /help   - Show help message
"""

import os
import sys

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

MODEL_PATH = "/mnt/g/AI/Qwen3.8-27B/Qwen3.8-27B-IQ4_XS.gguf"


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply ChatML template for Qwen3.8."""
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")

    ids = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        ids.append(im_start_id)
        ids.extend(tokenizer.encode(role + "\n", add_bos=False))
        ids.extend(tokenizer.encode(content, add_bos=False))
        ids.append(im_end_id)
        ids.extend(tokenizer.encode("\n", add_bos=False))

    if add_generation_prompt:
        ids.append(im_start_id)
        ids.extend(tokenizer.encode("assistant\n", add_bos=False))

    return ids


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

    print("Assistant: ", end="", flush=True)
    result = ctx.generate(prompt_ids, gen_cfg)
    response_ids = list(result.token_ids)
    print()
    text = tokenizer.decode(response_ids, skip_special=True)
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Qwen3.8-27B inference with Forge")
    add_common_args(parser, gpu_layers_default=0, temperature_default=0.7)
    parser.add_argument("--prompt", default=None,
                        help="Single-shot prompt. Use '-' to read from stdin. "
                             "Omit for interactive chat.")
    # 27B IQ4_XS (~15GB) does not fit in the 6GB RTX 4050 VRAM -> force CPU.
    parser.set_defaults(device="cpu")
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
            model_name="Qwen3.8-27B",
            system_msg={"role": "system", "content": "You are a helpful assistant."},
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
