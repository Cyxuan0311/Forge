"""Gemma-4-12B interactive chat using Forge.

Supports Gemma 4 MoE models in GGUF format, including per-layer embeddings,
QK-Norm, sliding window attention, and final logit softcapping.

Usage:
  python examples/gemma4_inference.py --model-path /mnt/g/AI/Gemma-4-12B/gemma-4-E2B-it-Q4_K_M.gguf
  python examples/gemma4_inference.py --device cpu --gpu-layers 0
  python examples/gemma4_inference.py --think

Interactive commands:
  /quit    - Exit the chat
  /clear   - Clear conversation history
  /help    - Show help message
"""

import os
import sys
import time
import gc
import argparse

from chat_utils import (
    PerfTimer,
    add_common_args,
    generate_streaming,
    generate_batch,
    load_model_and_tokenize,
    profiling_enabled,
    perf,
    resolve_model_path,
)

import forge
import numpy as np

GGUF_MODEL_PATH = "/mnt/g/AI/Gemma-4-12B/gemma-4-E2B-it-Q4_K_M.gguf"
GEMMA4_STOP_TOKEN_IDS = [106]


def apply_chat_template(tokenizer, messages, add_generation_prompt=True, enable_thinking=False):
    """Apply Gemma 4 chat template."""
    bos_id = tokenizer.bos_token_id
    turn_start_id = tokenizer.token_to_id("<|turn>")
    turn_end_id = tokenizer.token_to_id("<turn|>")
    think_id = tokenizer.token_to_id("<|think|>")
    newline_id = 107

    user_id = tokenizer.token_to_id("user")
    model_id = tokenizer.token_to_id("model")
    system_id = tokenizer.token_to_id("system")

    ids = []

    if bos_id >= 0:
        ids.append(bos_id)

    if enable_thinking and think_id is not None and think_id >= 0:
        ids.append(turn_start_id)
        ids.append(system_id)
        ids.append(newline_id)
        ids.append(think_id)
        ids.append(newline_id)
        ids.append(turn_end_id)
        ids.append(newline_id)

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            role_id = system_id
        elif role == "user":
            role_id = user_id
        elif role == "assistant":
            role_id = model_id
        else:
            role_id = user_id

        ids.append(turn_start_id)
        ids.append(role_id)
        ids.append(newline_id)
        ids.extend(tokenizer.encode(content, add_bos=False))
        ids.append(turn_end_id)
        ids.append(newline_id)

    if add_generation_prompt:
        ids.append(turn_start_id)
        ids.append(model_id)
        ids.append(newline_id)

    return ids


def interactive_chat_gemma4(model, tokenizer, args):
    """Gemma4-specific interactive chat with thinking mode support."""
    conversation = []
    ctx = None

    print("\n" + "=" * 60)
    print(f"  Gemma-4-12B Interactive Chat (Forge)")
    print(f"  Device: {args.device}")
    if args.think:
        print("  Thinking mode: ON")
    print("  Commands: /quit, /clear, /help")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Bye!")
            break
        elif user_input == "/clear":
            conversation = []
            print("[Conversation cleared]\n")
            continue
        elif user_input == "/help":
            print("  /quit    - Exit the chat")
            print("  /clear   - Clear conversation history")
            print("  /help    - Show this help message\n")
            continue

        conversation.append({"role": "user", "content": user_input})
        input_ids = apply_chat_template(
            tokenizer, conversation, add_generation_prompt=True,
            enable_thinking=args.think,
        )

        print("Assistant: ", end="", flush=True)

        if args.no_stream:
            num_generated, elapsed = generate_batch(
                model, tokenizer, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                stop_token_ids=GEMMA4_STOP_TOKEN_IDS,
                kv_cache_dtype=args.kv_cache_dtype,
                gpu_layers=args.gpu_layers,
            )
            assistant_text = ""
        else:
            if ctx is not None:
                del ctx
                gc.collect()
            ctx = model.create_context(
                kv_cache_dtype=args.kv_cache_dtype,
                gpu_layers=args.gpu_layers,
            )
            generated_tokens, elapsed = generate_streaming(
                model, ctx, tokenizer, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                stop_token_ids=GEMMA4_STOP_TOKEN_IDS,
                kv_cache_dtype=args.kv_cache_dtype,
                gpu_layers=args.gpu_layers,
            )
            num_generated = len(generated_tokens)
            assistant_text = tokenizer.decode(
                generated_tokens, skip_special=True, strip_leading_space=False
            )

        print()
        if num_generated > 0 and elapsed > 0:
            speed = num_generated / elapsed
            prompt_len = len(input_ids)
            print(f"[{num_generated} tokens, {elapsed:.2f}s, {speed:.1f} tok/s, prompt={prompt_len} tokens]")

        conversation.append({"role": "assistant", "content": assistant_text})
        print()


def main():
    parser = argparse.ArgumentParser(description="Gemma-4-12B inference with Forge")
    add_common_args(parser, gpu_layers_default=-1, temperature_default=0.7)
    parser.add_argument("--think", action="store_true", help="Enable thinking mode for Gemma4")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single-turn prompt (if omitted, enters interactive chat mode)",
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args, [GGUF_MODEL_PATH])
    model, tokenizer = load_model_and_tokenize(args, model_path)

    if model is None:
        return

    # Print MoE-specific config
    cfg = model.config
    if cfg.n_expert > 0:
        print(f"MoE config: experts={cfg.n_expert}, top_k={cfg.n_expert_used}, ff_exp={cfg.n_ff_exp}")
    if cfg.n_embd_per_layer > 0:
        print(f"Per-layer embeddings: dim={cfg.n_embd_per_layer}")
    if cfg.n_swa > 0:
        print(f"Sliding window attention: size={cfg.n_swa}")

    interactive_chat_gemma4(model, tokenizer, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
