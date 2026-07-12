#!/usr/bin/env python3
"""MiniCPM-V 4.6 Interactive Chat using Forge.

Supports multi-turn conversation with optional image input.

Prerequisites:
  1. Download model files:
     - MiniCPM-V-4_6-Q4_K_M.gguf (LLM)
     - mmproj-model-f16.gguf (vision encoder)

Usage:
  python examples/minicpmv_cli_inference.py
  python examples/minicpmv_cli_inference.py --model-path /path/to/model
  python examples/minicpmv_cli_inference.py --device cpu --gpu-layers 0

Interactive commands:
  /image <path>  - Load an image for the next query
  /video <path>  - Load a video for the next query (requires decord)
  /clear        - Clear conversation history
  /quit         - Exit the chat
  /help         - Show help message
"""

import sys
import os
import time
import gc
import argparse
import numpy as np

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge

# Constants
MODEL_PATH = "/mnt/g/AI/MiniCPM-V-4.6.F16"
LLM_FILE = "MiniCPM-V-4_6-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-model-f16.gguf"


def _sanitize_utf8(text: str) -> str:
    """Remove lone surrogate characters that cannot be encoded as UTF-8."""
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


def _extract_frames(video_path, num_frames=8):
    """Uniformly sample N frames from a video using decord."""
    import decord

    vr = decord.VideoReader(video_path)
    total = len(vr)
    indices = [int(total * i / num_frames) for i in range(num_frames)]
    frames = vr.get_batch(indices).asnumpy()  # [N, H, W, 3] uint8
    return frames


def build_multimodal_prompt(tokenizer, prompt_text, num_img_tokens_list, enable_thinking=False):
    """Build prompt with image placeholder tokens and return token IDs + insertion info.

    MiniCPM-V 4.6 uses Qwen-style chat template with optional image blocks:
      <|im_start|>user\n<image>[img]</image>...{prompt}<|im_end|>\n<|im_start|>assistant\n

    Supports multiple image blocks for video: pass a list of token counts,
    e.g. [64] for single image, [64, 64, 64] for 3-frame video.

    Returns:
        full_ids: list of token IDs (with dummy tokens at image positions)
        image_insert_spans: list of (start, length) for each image block
        total_img_tokens: sum of all image token counts
    """
    # Look up special token IDs directly from vocabulary
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    img_start_id = tokenizer.token_to_id("<image>")
    img_end_id = tokenizer.token_to_id("</image>")

    # Newline: token_to_id("\n") returns unk for Qwen tokenizer,
    # must use encode() to get the correct BPE token (id 198)
    newline_ids = list(tokenizer.encode("\n", add_bos=False))

    # Encode "user" and "assistant" as regular text (no BOS/EOS)
    user_text_ids = list(tokenizer.encode("user", add_bos=False))
    assistant_text_ids = list(tokenizer.encode("assistant", add_bos=False))

    # Encode the prompt text (no BOS/EOS - we add <|im_start|> manually)
    prompt_text = _sanitize_utf8(prompt_text)
    prompt_ids = list(tokenizer.encode(prompt_text, add_bos=False))

    total_img = sum(num_img_tokens_list)
    if total_img > 0:
        # Build multiple <image>[dummy]</image> blocks
        frame_blocks = []
        for nt in num_img_tokens_list:
            frame_blocks += [img_start_id] + [0] * nt + [img_end_id]
        full_ids = (
            [im_start_id]
            + user_text_ids
            + newline_ids
            + frame_blocks
            + prompt_ids
            + [im_end_id]
            + newline_ids
            + [im_start_id]
            + assistant_text_ids
            + newline_ids
        )
        # Compute insert spans: position of first dummy token in each block
        insert_spans = []
        pos = 1 + len(user_text_ids) + len(newline_ids) + 1  # after <|im_start|>user\n<image>
        for nt in num_img_tokens_list:
            insert_spans.append((pos, nt))
            pos += nt + 2  # +2 for <image> and </image>
    else:
        # Text-only: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        full_ids = (
            [im_start_id]
            + user_text_ids
            + newline_ids
            + prompt_ids
            + [im_end_id]
            + newline_ids
            + [im_start_id]
            + assistant_text_ids
            + newline_ids
        )
        insert_spans = []

    return full_ids, insert_spans, total_img


def build_multimodal_embeddings(ctx, prompt_ids, image_embeddings, image_insert_spans):
    """Build full embeddings with image embeddings injected.

    Supports multiple image blocks (video): iterates through spans and replaces
    each block's dummy embeddings with the corresponding vision encoder outputs.
    """
    prompt_array = np.array(prompt_ids, dtype=np.int32)
    text_embeddings = np.array(ctx.get_embeddings(prompt_array))

    if image_insert_spans and image_embeddings is not None:
        offset = 0
        for start, length in image_insert_spans:
            end = start + length
            text_embeddings[start:end] = image_embeddings[offset : offset + length]
            offset += length

    return text_embeddings


def generate_response(
    ctx,
    tokenizer,
    prompt_ids,
    image_embeddings,
    image_insert_spans,
    max_new_tokens=256,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.1,
):
    """Generate a response for a single turn.

    Handles the full forward + decode loop, returns generated text and timing.
    Supports thinking mode: content between <think\\n and \\n</think\\n is
    treated as reasoning and displayed separately.
    """
    # Forward the prompt
    if image_insert_spans and image_embeddings is not None:
        combined_emb = build_multimodal_embeddings(
            ctx, prompt_ids, image_embeddings, image_insert_spans
        )
        logits = ctx.forward_with_embeddings(combined_emb, 0)
    else:
        prompt_array = np.array(prompt_ids, dtype=np.int32)
        logits = ctx.forward(prompt_array, 0)

    # Generate tokens
    start_pos = len(prompt_ids)
    generated_ids = []
    generated_text = ""
    token_buffer = []
    gen_start = time.time()

    # Thinking mode state
    # <think and </think are single special tokens in Qwen3 vocabulary
    # token_to_id may not find them, so we detect by trying to decode candidate IDs
    think_start_id = None
    think_end_id = None
    # Try to find <think and </think tokens by checking near the end of vocabulary
    # (special tokens are typically added at the end)
    for tid in range(tokenizer.vocab_size - 1, max(0, tokenizer.vocab_size - 200), -1):
        try:
            decoded = tokenizer.decode([tid], skip_special=False)
            if decoded == "<think":
                think_start_id = tid
            elif decoded == "</think":
                think_end_id = tid
            if think_start_id is not None and think_end_id is not None:
                break
        except Exception:
            pass
    in_thinking = False
    thinking_text = ""

    for step in range(max_new_tokens):
        logits_np = np.array(logits)
        last_logits = logits_np[-1] if logits_np.ndim > 1 else logits_np

        # Sample next token
        if temperature > 0:
            # Apply repeat penalty
            if repeat_penalty != 1.0 and len(generated_ids) > 0:
                recent_ids = generated_ids[-64:]
                for tid in set(recent_ids):
                    if last_logits[tid] > 0:
                        last_logits[tid] /= repeat_penalty
                    else:
                        last_logits[tid] *= repeat_penalty

            # Numerically stable softmax with top-k
            shifted = last_logits - np.max(last_logits)
            probs = np.exp(shifted / temperature)
            probs = probs / probs.sum()
            top_k_actual = min(top_k, len(probs))
            top_indices = np.argsort(probs)[-top_k_actual:]
            top_probs = probs[top_indices]
            top_probs = top_probs / top_probs.sum()
            next_token = int(np.random.choice(top_indices, p=top_probs))
        else:
            next_token = int(np.argmax(last_logits))

        generated_ids.append(next_token)

        # Check EOS
        if next_token == tokenizer.eos_token_id:
            break

        # Check thinking mode transitions
        if think_start_id is not None and next_token == think_start_id:
            in_thinking = True
            continue
        if think_end_id is not None and next_token == think_end_id:
            if in_thinking and thinking_text:
                print(f"\n[Thinking: {thinking_text.strip()}]\n", end="", flush=True)
                thinking_text = ""
            in_thinking = False
            continue

        # Decode and print (buffered for smoother output)
        token_buffer.append(next_token)
        if len(token_buffer) >= 4:
            text = tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
            if in_thinking:
                thinking_text += text
            else:
                print(text, end="", flush=True)
                generated_text += text
            token_buffer.clear()

        # Forward next token
        next_ids = np.array([next_token], dtype=np.int32)
        try:
            logits = ctx.forward(next_ids, start_pos)
        except Exception as e:
            print(f"\nERROR at step {step}: {e}")
            break
        start_pos += 1

    # Flush remaining buffer
    if token_buffer:
        text = tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
        if in_thinking:
            thinking_text += text
            if thinking_text:
                print(f"[Thinking: {thinking_text.strip()}]", end="", flush=True)
        else:
            print(text, end="", flush=True)
            generated_text += text
        token_buffer.clear()

    elapsed = time.time() - gen_start
    return generated_text, len(generated_ids), elapsed


def interactive_chat(mm_model, tokenizer, args):
    """Interactive multi-turn chat loop with optional image/video support."""
    conversation = []
    image_embeddings = None
    num_img_tokens_list = []
    ctx = None

    print("\n" + "=" * 60)
    print("  MiniCPM-V 4.6 Interactive Chat (Forge)")
    print(f"  Device: {args.device}")
    print("  Commands: /image <path>, /video <path>, /clear, /quit, /help")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Defensive: sanitize input before any encode() call
        user_input = _sanitize_utf8(user_input)

        # Handle commands
        if user_input == "/quit":
            print("Bye!")
            break
        elif user_input == "/clear":
            conversation = []
            image_embeddings = None
            num_img_tokens_list = []
            print("[Conversation cleared]\n")
            continue
        elif user_input == "/help":
            print("  /image <path> - Load an image for the next query")
            print("  /video <path> - Load a video for the next query")
            print("  /clear        - Clear conversation history and media")
            print("  /quit         - Exit the chat")
            print("  /help         - Show this help message\n")
            continue
        elif user_input.startswith("/image "):
            img_path = user_input[7:].strip()
            if not os.path.exists(img_path):
                print(f"  Image not found: {img_path}\n")
                continue
            try:
                from PIL import Image

                print(f"  Encoding image: {img_path}...")
                t0 = time.time()
                img = Image.open(img_path).convert("RGB")
                pixels = np.array(img, dtype=np.uint8)
                emb = mm_model.encode_image(pixels)
                image_embeddings = emb
                num_img_tokens_list = [emb.shape[0]]
                print(
                    f"  Image loaded: {img.size}, {emb.shape[0]} tokens, took {time.time() - t0:.2f}s"
                )
                if forge.profiler_enabled():
                    forge.profiler_print()
                    forge.profiler_reset()
                print()
            except Exception as e:
                print(f"  ERROR encoding image: {e}\n")
                image_embeddings = None
                num_img_tokens_list = []
            continue
        elif user_input.startswith("/video "):
            video_path = user_input[7:].strip()
            if not os.path.exists(video_path):
                print(f"  Video not found: {video_path}\n")
                continue
            try:
                n_frames = args.video_frames
                print(f"  Extracting {n_frames} frames from: {video_path}...")
                t0 = time.time()
                frames = _extract_frames(video_path, num_frames=n_frames)
                print(f"  Encoding {len(frames)} frames...")
                frame_embs = []
                for i, frame in enumerate(frames):
                    t1 = time.time()
                    emb = mm_model.encode_image(frame)
                    frame_embs.append(emb)
                    dt = time.time() - t1
                    print(f"    frame {i + 1}/{len(frames)} encoded ({dt:.2f}s)")
                image_embeddings = np.concatenate(frame_embs, axis=0)
                num_img_tokens_list = [emb.shape[0] for emb in frame_embs]
                total_tok = image_embeddings.shape[0]
                print(
                    f"  Video loaded: {len(frames)} frames, {total_tok} tokens, took {time.time() - t0:.2f}s\n"
                )
                if forge.profiler_enabled():
                    forge.profiler_print()
                    forge.profiler_reset()
            except ImportError:
                print("  ERROR: decord not installed. Run: pip install decord\n")
                image_embeddings = None
                num_img_tokens_list = []
            except Exception as e:
                print(f"  ERROR processing video: {e}\n")
                image_embeddings = None
                num_img_tokens_list = []
            continue

        # Build prompt for this turn
        conversation.append({"role": "user", "content": user_input})
        prompt_ids, insert_spans, total_img = build_multimodal_prompt(
            tokenizer, user_input, num_img_tokens_list
        )

        # Free old context before creating a new one to release GPU memory
        if ctx is not None:
            del ctx
            gc.collect()
        ctx = mm_model.create_context(args.kv_cache_dtype, args.gpu_layers)

        print("Assistant: ", end="", flush=True)

        try:
            response_text, num_generated, elapsed = generate_response(
                ctx,
                tokenizer,
                prompt_ids,
                image_embeddings,
                insert_spans,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
            )
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback

            traceback.print_exc()
            # Remove the failed user message from history
            conversation.pop()
            continue

        print()
        if num_generated > 0 and elapsed > 0:
            speed = num_generated / elapsed
            print(f"[{num_generated} tokens, {elapsed:.2f}s, {speed:.1f} tok/s]")

        conversation.append({"role": "assistant", "content": response_text})

        # Clear media after use (next turn is text-only unless /image or /video is used again)
        image_embeddings = None
        num_img_tokens_list = []
        print()


def main():
    parser = argparse.ArgumentParser(description="MiniCPM-V 4.6 Interactive Chat")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Model directory path")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference"
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=99, help="Number of layers to place on GPU (-1 for all)"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "q4_0"],
        help="KV cache data type",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy)"
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="Top-k sampling parameter (0 to disable)"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiler (uses chrono for CPU stages)",
    )
    parser.add_argument(
        "--profile-cuda",
        action="store_true",
        help="Enable profiler with CUDA events (for GPU stages only)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--trace", action="store_true", help="Enable TRACE logging (most verbose)")
    parser.add_argument(
        "--video-frames", type=int, default=8, help="Number of frames to sample for video input"
    )
    args = parser.parse_args()

    print("Loading tokenizer from GGUF...")
    llm_path = os.path.join(args.model_path, LLM_FILE)
    mmproj_path = os.path.join(args.model_path, MMPROJ_FILE)

    tokenizer = forge.Tokenizer()
    tokenizer.load_from_gguf(llm_path)
    print(
        f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
        f"bos_id={tokenizer.bos_token_id}, eos_id={tokenizer.eos_token_id}"
    )

    if args.trace:
        forge.Logger.set_level(5)  # TRACE
    elif args.debug:
        forge.Logger.set_level(4)  # DEBUG
    else:
        forge.Logger.set_level(2 if args.verbose else 1)  # existing: WARN/VERBOSE

    if args.profile or args.profile_cuda:
        forge.profiler_enable()
        forge.profiler_set_cuda_events(args.profile_cuda)
        mode = "CUDA events" if args.profile_cuda else "chrono"
        print(f"Performance profiler enabled ({mode})")

    print(f"Loading multimodal model on {args.device}...")

    mm_model = forge.MultimodalModel()
    try:
        mm_model.load_with_mmproj(llm_path, mmproj_path, args.device)
    except RuntimeError as e:
        print(f"  CUDA failed ({e}), falling back to CPU")
        args.device = "cpu"
        args.gpu_layers = 0
        mm_model.load_with_mmproj(llm_path, mmproj_path, args.device)

    cfg = mm_model.config
    print(
        f"Model loaded! arch={cfg.arch_type}, hidden_dim={cfg.hidden_dim}, "
        f"num_layers={cfg.num_layers}, num_heads={cfg.num_heads}, use_ssm={cfg.use_ssm}"
    )
    if cfg.use_ssm:
        print(
            f"  SSM: inner_size={cfg.ssm_inner_size}, state_size={cfg.ssm_state_size}, "
            f"group_count={cfg.ssm_group_count}, dt_rank={cfg.ssm_time_step_rank}, "
            f"conv_kernel={cfg.ssm_conv_kernel}, full_attn_interval={cfg.full_attention_interval}"
        )

    # Print context info
    cuda_ok = True
    try:
        ctx = mm_model.create_context(args.kv_cache_dtype, args.gpu_layers)
        stats = ctx.memory_stats()
        print(
            f"KV Cache: dtype={stats.get('kv_cache_dtype', 'unknown')}, "
            f"size: {stats.get('kv_cache_nbytes', 0) / 1024 / 1024:.1f} MB"
        )

        # Warmup to trigger CUDA kernel JIT compilation
        if args.device == "cuda":
            print("Warming up CUDA kernels...")
            try:
                ctx.warmup()
                print("Warmup done!")
            except RuntimeError as e:
                print(f"Warmup skipped ({e})")
                cuda_ok = False

        del ctx
        gc.collect()
    except RuntimeError as e:
        print(f"Context creation failed: {e}")
        cuda_ok = False

    if not cuda_ok and args.device == "cuda":
        print("CUDA out of memory, falling back to CPU")
        args.device = "cpu"
        args.gpu_layers = 0

    interactive_chat(mm_model, tokenizer, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
