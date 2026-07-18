"""
Gemma-4-12B interactive chat using Forge.

Supports Gemma 4 MoE models in GGUF format, including per-layer embeddings,
QK-Norm, sliding window attention, and final logit softcapping.

Prerequisites:
  1. Download GGUF model (e.g., gemma-4-E2B-it-Q4_K_M.gguf) to /mnt/g/AI/Gemma-4-12B/

Usage:
  python examples/gemma4_inference.py --model-path /mnt/g/AI/Gemma-4-12B/gemma-4-E2B-it-Q4_K_M.gguf

  # Other options:
  python examples/gemma4_inference.py --device cpu --gpu-layers 0
  python examples/gemma4_inference.py --no-stream
  python examples/gemma4_inference.py --max-new-tokens 512 --temperature 0.7

Interactive commands:
  /quit    - Exit the chat
  /clear   - Clear conversation history
  /help    - Show help message
"""

import sys
import os
import time
import gc
import argparse

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge
import numpy as np


GGUF_MODEL_PATH = "/mnt/g/AI/Gemma-4-12B/gemma-4-E2B-it-Q4_K_M.gguf"

# Gemma4 stop tokens: <turn|> (ID=106) marks end of model's turn
GEMMA4_STOP_TOKEN_IDS = [106]


def load_tokenizer(model_path):
    """Load tokenizer from GGUF file using Forge's built-in Tokenizer."""
    tok = forge.Tokenizer()
    tok.load_from_gguf(model_path)
    return tok


def apply_chat_template(tokenizer, messages, add_generation_prompt=True, enable_thinking=False):
    """Apply Gemma 4 chat template to produce token IDs.

    Gemma 4 GGUF uses these special tokens:
       <|turn>  (ID=105)  - start of a turn (replaces <start_of_turn>)
       <turn|>  (ID=106)  - end of a turn (replaces <end_of_turn>)
       <|think|> (ID=98)  - triggers thinking mode

    Chat template format:
       <bos><|turn>user\n{content}<turn|>\n<|turn>model\n

    With thinking enabled:
       <bos><|turn>system\n<|think|><turn|>\n<|turn>user\n{content}<turn|>\n<|turn>model\n
    """
    bos_id = tokenizer.bos_token_id

    # Gemma4 special tokens
    turn_start_id = tokenizer.token_to_id("<|turn>")   # 105
    turn_end_id = tokenizer.token_to_id("<turn|>")      # 106
    think_id = tokenizer.token_to_id("<|think|>")       # 98
    newline_id = 107  # '\n' token

    # Role token IDs (without SentencePiece space prefix ▁)
    user_id = tokenizer.token_to_id("user")     # no-space version
    model_id = tokenizer.token_to_id("model")   # no-space version
    system_id = tokenizer.token_to_id("system") # no-space version

    ids = []

    # Add BOS token at the start
    if bos_id >= 0:
        ids.append(bos_id)

    # When thinking is enabled, inject a system turn with <|think|> token
    # Template: <bos><|turn>system\n<|think|>\n<turn|>\n
    if enable_thinking and think_id is not None and think_id >= 0:
        ids.append(turn_start_id)
        ids.append(system_id)
        ids.append(newline_id)
        ids.append(think_id)
        ids.append(newline_id)  # <|think|>\n
        ids.append(turn_end_id)
        ids.append(newline_id)

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Map role to role token
        if role == "system":
            role_id = system_id
        elif role == "user":
            role_id = user_id
        elif role == "assistant":
            role_id = model_id
        else:
            role_id = user_id  # fallback

        # <|turn>{role}\n{content}<turn|>\n
        ids.append(turn_start_id)
        ids.append(role_id)
        ids.append(newline_id)
        ids.extend(tokenizer.encode(content, add_bos=False))
        ids.append(turn_end_id)
        ids.append(newline_id)

    if add_generation_prompt:
        # <|turn>model\n
        ids.append(turn_start_id)
        ids.append(model_id)
        ids.append(newline_id)

    return ids


def generate_streaming(
    model, tokenizer, input_ids, max_new_tokens=256, temperature=0.7,
    top_k=40, top_p=0.9, repeat_penalty=1.1, eos_token_id=None,
    enable_thinking=False,
):
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    generated_tokens = []
    token_buffer = []
    decode_start = time.time()

    # Thinking mode state
    channel_start_id = tokenizer.token_to_id("<channel|>") if enable_thinking else None
    in_thinking = enable_thinking  # starts in thinking mode if enabled
    thinking_text = ""
    response_started = False

    def on_token(token_id, step):
        nonlocal in_thinking, thinking_text, response_started
        generated_tokens.append(token_id)
        token_buffer.append(token_id)

        # Check for channel transition (thinking → response)
        if enable_thinking and channel_start_id is not None and token_id == channel_start_id:
            in_thinking = False
            response_started = True
            token_buffer.clear()
            if thinking_text.strip():
                print(f"\n[Thinking: {thinking_text.strip()}]\n", end="", flush=True)
            thinking_text = ""
            return

        if len(token_buffer) >= 4 or token_id == eos_token_id:
            try:
                text = tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
                if in_thinking:
                    thinking_text += text
                else:
                    if not response_started:
                        response_started = True
                    print(text, end="", flush=True)
                token_buffer.clear()
            except UnicodeDecodeError:
                if token_id == eos_token_id:
                    text = tokenizer.decode(
                        token_buffer, skip_special=True, strip_leading_space=False
                    )
                    if in_thinking:
                        thinking_text += text
                    else:
                        print(text, end="", flush=True)
                    token_buffer.clear()

    np_ids = np.array(input_ids, dtype=np.int32)
    model.generate_stream(
        prompt_ids=np_ids,
        callback=on_token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        do_sample=temperature > 0,
        eos_token_id=eos_token_id,
        stop_token_ids=GEMMA4_STOP_TOKEN_IDS,
        kv_cache_dtype="fp32",
        gpu_layers=-1,
    )

    if token_buffer:
        text = tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
        if in_thinking:
            thinking_text += text
            if thinking_text.strip():
                print(f"\n[Thinking: {thinking_text.strip()}]\n", end="", flush=True)
        else:
            print(text, end="", flush=True)
        token_buffer.clear()

    elapsed = time.time() - decode_start
    return generated_tokens, elapsed


def generate_batch(
    model, tokenizer, input_ids, max_new_tokens=256, temperature=0.7,
    top_k=40, top_p=0.9, repeat_penalty=1.1, eos_token_id=None,
    enable_thinking=False,
):
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    start_time = time.time()
    np_ids = np.array(input_ids, dtype=np.int32)
    result = model.generate(
        prompt_ids=np_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        do_sample=temperature > 0,
        eos_token_id=eos_token_id,
        stop_token_ids=GEMMA4_STOP_TOKEN_IDS,
        kv_cache_dtype="fp32",
        gpu_layers=-1,
    )
    elapsed = time.time() - start_time

    num_generated = result["num_generated_tokens"]
    new_tokens = list(result["token_ids"])

    if enable_thinking:
        channel_start_id = tokenizer.token_to_id("<channel|>")
        # Split tokens at channel boundary: thinking | response
        thinking_tokens = []
        response_tokens = []
        found_channel = False
        for tid in new_tokens:
            if not found_channel:
                if tid == channel_start_id:
                    found_channel = True
                else:
                    thinking_tokens.append(tid)
            else:
                response_tokens.append(tid)
        thinking_text = tokenizer.decode(thinking_tokens, skip_special=True, strip_leading_space=False)
        if thinking_text.strip():
            print(f"[Thinking: {thinking_text.strip()}]\n", end="", flush=True)
        response = tokenizer.decode(response_tokens, skip_special=True, strip_leading_space=False)
        print(response, end="", flush=True)
    else:
        response = tokenizer.decode(new_tokens, skip_special=True, strip_leading_space=False)
        print(response, end="", flush=True)

    return num_generated, elapsed


def interactive_chat(model, tokenizer, args):
    """Interactive multi-turn chat loop."""
    conversation = []
    ctx = None

    print("\n" + "=" * 60)
    print("  Gemma-4-12B Interactive Chat (Forge)")
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
                enable_thinking=args.think,
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
                model, tokenizer, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                enable_thinking=args.think,
            )
            num_generated = len(generated_tokens)
            assistant_text = tokenizer.decode(
                generated_tokens, skip_special=True, strip_leading_space=False
            )

        print()
        if num_generated > 0 and elapsed > 0:
            speed = num_generated / elapsed
            prompt_len = len(input_ids)
            print(
                f"[{num_generated} tokens, {elapsed:.2f}s, {speed:.1f} tok/s, prompt={prompt_len} tokens]"
            )

        conversation.append({"role": "assistant", "content": assistant_text})
        print()


def single_turn(model, tokenizer, args):
    """Single-turn generation from a prompt."""
    prompt = args.prompt
    input_ids = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=args.think,
    )

    print(f"User: {prompt}")
    print("Assistant: ", end="", flush=True)

    if args.no_stream:
        num_generated, elapsed = generate_batch(
            model, tokenizer, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            enable_thinking=args.think,
        )
    else:
        model.create_context(
            kv_cache_dtype=args.kv_cache_dtype,
            gpu_layers=args.gpu_layers,
        )
        generated_tokens, elapsed = generate_streaming(
            model, tokenizer, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            enable_thinking=args.think,
        )
        num_generated = len(generated_tokens)

    print()
    if num_generated > 0 and elapsed > 0:
        speed = num_generated / elapsed
        print(f"[{num_generated} tokens, {elapsed:.2f}s, {speed:.1f} tok/s]")


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma-4-12B inference with Forge")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .gguf model file")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=-1,
        help="Number of layers to place on GPU (-1 for all)",
    )
    parser.add_argument(
        "--kv-cache-dtype", type=str, default="fp32", choices=["fp32", "q4_0"],
        help="KV cache data type",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="Top-k sampling parameter (0 to disable)",
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode for Gemma4")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single-turn prompt (if omitted, enters interactive chat mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path
    if model_path is None:
        if os.path.exists(GGUF_MODEL_PATH):
            model_path = GGUF_MODEL_PATH
        else:
            print("No model file found. Searched:")
            print(f"  {GGUF_MODEL_PATH}")
            print("Please specify --model-path")
            sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    print("Loading tokenizer from GGUF...")
    tokenizer = load_tokenizer(model_path)
    print(
        f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
        f"model_type={tokenizer.model_type}, "
        f"bos_id={tokenizer.bos_token_id}, eos_id={tokenizer.eos_token_id}"
    )

    print(f"Loading GGUF model on {args.device}...")
    forge.Logger.set_level(2 if args.verbose else 1)

    model = forge.Model()
    model.load_gguf(model_path, device=args.device)

    cfg = model.config
    print(
        f"Model loaded! arch={cfg.arch_type}, layers={cfg.num_layers}, "
        f"hidden={cfg.hidden_dim}, heads={cfg.num_heads}, "
        f"kv_heads={cfg.num_kv_heads}, vocab={cfg.vocab_size}"
    )

    if cfg.n_expert > 0:
        print(
            f"MoE config: experts={cfg.n_expert}, top_k={cfg.n_expert_used}, "
            f"ff_exp={cfg.n_ff_exp}"
        )
    if cfg.n_embd_per_layer > 0:
        print(f"Per-layer embeddings: dim={cfg.n_embd_per_layer}")
    if cfg.n_swa > 0:
        print(f"Sliding window attention: size={cfg.n_swa}")

    if args.device == "cpu":
        cpu_threads = (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        )
        forge.set_num_threads(cpu_threads)
        print(f"CPU threads set to: {cpu_threads}")

    ctx = model.create_context(
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_layers=args.gpu_layers,
    )
    stats = ctx.memory_stats()
    print(
        f"KV Cache: dtype={stats.get('kv_cache_dtype', 'unknown')}, "
        f"size: {stats.get('kv_cache_nbytes', 0) / 1024 / 1024:.1f} MB"
    )

    # Warmup CUDA kernels
    if args.device == "cuda":
        print("Warming up CUDA kernels...")
        ctx.warmup()
        del ctx
        gc.collect()
        print("Warmup done!")
    else:
        del ctx

    if args.prompt:
        single_turn(model, tokenizer, args)
    else:
        interactive_chat(model, tokenizer, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
