#!/usr/bin/env python3
"""MiniCPM-V 4.6 Interactive Chat using Forge.

Supports multi-turn conversation with optional image/video input.

Usage:
  python examples/minicpmv_cli_inference.py
  python examples/minicpmv_cli_inference.py --model-path /path/to/model
  python examples/minicpmv_cli_inference.py --device cpu --gpu-layers 0

Interactive commands:
  /image <path>  - Load an image for the next query
  /video <path>  - Load a video for the next query (requires decord)
  /clear        - Clear conversation history
  /quit         - Exit the chat
"""

import os
import sys
import time
import gc
import argparse
import numpy as np

from chat_utils import (
    add_common_args,
    load_tokenizer,
)

import forge

MODEL_PATH = "/mnt/g/AI/MiniCPM-V-4.6.F16"
LLM_FILE = "MiniCPM-V-4_6-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-model-f16.gguf"


def _sanitize_utf8(text: str) -> str:
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


def _extract_frames(video_path, num_frames=8):
    import decord
    vr = decord.VideoReader(video_path)
    total = len(vr)
    indices = [int(total * i / num_frames) for i in range(num_frames)]
    return vr.get_batch(indices).asnumpy()


def build_multimodal_prompt(tokenizer, prompt_text, num_img_tokens_list):
    """Build prompt with image placeholder tokens."""
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    img_start_id = tokenizer.token_to_id("<image>")
    img_end_id = tokenizer.token_to_id("</image>")
    newline_ids = list(tokenizer.encode("\n", add_bos=False))
    user_text_ids = list(tokenizer.encode("user", add_bos=False))
    assistant_text_ids = list(tokenizer.encode("assistant", add_bos=False))

    prompt_text = _sanitize_utf8(prompt_text)
    prompt_ids = list(tokenizer.encode(prompt_text, add_bos=False))

    total_img = sum(num_img_tokens_list)
    if total_img > 0:
        frame_blocks = []
        for nt in num_img_tokens_list:
            frame_blocks += [img_start_id] + [0] * nt + [img_end_id]
        full_ids = (
            [im_start_id] + user_text_ids + newline_ids
            + frame_blocks + prompt_ids
            + [im_end_id] + newline_ids
            + [im_start_id] + assistant_text_ids + newline_ids
        )
        insert_spans = []
        pos = 1 + len(user_text_ids) + len(newline_ids) + 1
        for nt in num_img_tokens_list:
            insert_spans.append((pos, nt))
            pos += nt + 2
    else:
        full_ids = (
            [im_start_id] + user_text_ids + newline_ids + prompt_ids
            + [im_end_id] + newline_ids
            + [im_start_id] + assistant_text_ids + newline_ids
        )
        insert_spans = []

    return full_ids, insert_spans, total_img


def build_multimodal_embeddings(ctx, prompt_ids, image_embeddings, image_insert_spans):
    prompt_array = np.array(prompt_ids, dtype=np.int32)
    text_embeddings = np.array(ctx.get_embeddings(prompt_array))

    if image_insert_spans and image_embeddings is not None:
        offset = 0
        for start, length in image_insert_spans:
            end = start + length
            text_embeddings[start:end] = image_embeddings[offset:offset + length]
            offset += length

    return text_embeddings


def find_thinking_tokens(tokenizer):
    think_start_id = None
    think_end_id = None
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
    return think_start_id, think_end_id


def generate_response(ctx, tokenizer, prompt_ids, image_embeddings, image_insert_spans,
                      max_new_tokens=256, temperature=0.7, top_k=40, top_p=0.9, repeat_penalty=1.1):
    if image_insert_spans and image_embeddings is not None:
        combined_emb = build_multimodal_embeddings(ctx, prompt_ids, image_embeddings, image_insert_spans)
        logits = ctx.forward_with_embeddings(combined_emb, 0)
    else:
        prompt_array = np.array(prompt_ids, dtype=np.int32)
        logits = ctx.forward(prompt_array, 0)

    think_start_id, think_end_id = find_thinking_tokens(tokenizer)
    start_pos = len(prompt_ids)
    generated_ids = []
    generated_text = ""
    token_buffer = []
    gen_start = time.time()
    in_thinking = False
    thinking_text = ""

    for step in range(max_new_tokens):
        logits_np = np.array(logits)
        last_logits = logits_np[-1] if logits_np.ndim > 1 else logits_np

        if temperature > 0:
            if repeat_penalty != 1.0 and len(generated_ids) > 0:
                for tid in set(generated_ids[-64:]):
                    if last_logits[tid] > 0:
                        last_logits[tid] /= repeat_penalty
                    else:
                        last_logits[tid] *= repeat_penalty
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

        if next_token == tokenizer.eos_token_id:
            break

        if think_start_id is not None and next_token == think_start_id:
            in_thinking = True
            continue
        if think_end_id is not None and next_token == think_end_id:
            if in_thinking and thinking_text:
                print(f"\n[Thinking: {thinking_text.strip()}]\n", end="", flush=True)
                thinking_text = ""
            in_thinking = False
            continue

        token_buffer.append(next_token)
        if len(token_buffer) >= 4:
            text = tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
            if in_thinking:
                thinking_text += text
            else:
                print(text, end="", flush=True)
                generated_text += text
            token_buffer.clear()

        next_ids = np.array([next_token], dtype=np.int32)
        try:
            logits = ctx.forward(next_ids, start_pos)
        except Exception as e:
            print(f"\nERROR at step {step}: {e}")
            break
        start_pos += 1

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
        user_input = _sanitize_utf8(user_input)

        if user_input == "/quit":
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
            print("  /clear, /quit, /help\n")
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
                print(f"  Image loaded: {img.size}, {emb.shape[0]} tokens, took {time.time() - t0:.2f}s\n")
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
                frame_embs = []
                for i, frame in enumerate(frames):
                    t1 = time.time()
                    emb = mm_model.encode_image(frame)
                    frame_embs.append(emb)
                    print(f"    frame {i+1}/{len(frames)} encoded ({time.time()-t1:.2f}s)")
                image_embeddings = np.concatenate(frame_embs, axis=0)
                num_img_tokens_list = [emb.shape[0] for emb in frame_embs]
                print(f"  Video loaded: {len(frames)} frames, {image_embeddings.shape[0]} tokens, took {time.time()-t0:.2f}s\n")
            except ImportError:
                print("  ERROR: decord not installed. Run: pip install decord\n")
                image_embeddings = None
                num_img_tokens_list = []
            except Exception as e:
                print(f"  ERROR processing video: {e}\n")
                image_embeddings = None
                num_img_tokens_list = []
            continue

        conversation.append({"role": "user", "content": user_input})
        prompt_ids, insert_spans, total_img = build_multimodal_prompt(tokenizer, user_input, num_img_tokens_list)

        if ctx is not None:
            del ctx
            gc.collect()
        ctx = mm_model.create_context(args.kv_cache_dtype, args.gpu_layers)

        print("Assistant: ", end="", flush=True)

        try:
            response_text, num_generated, elapsed = generate_response(
                ctx, tokenizer, prompt_ids, image_embeddings, insert_spans,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
            )
        except Exception as e:
            print(f"\nERROR: {e}")
            conversation.pop()
            continue

        print()
        if num_generated > 0 and elapsed > 0:
            speed = num_generated / elapsed
            print(f"[{num_generated} tokens, {elapsed:.2f}s, {speed:.1f} tok/s]")

        conversation.append({"role": "assistant", "content": response_text})
        image_embeddings = None
        num_img_tokens_list = []
        print()


def main():
    parser = argparse.ArgumentParser(description="MiniCPM-V 4.6 Interactive Chat")
    add_common_args(parser, gpu_layers_default=99, temperature_default=0.7)
    parser.add_argument("--video-frames", type=int, default=8, help="Number of frames to sample for video input")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--trace", action="store_true", help="Enable TRACE logging")
    parser.add_argument("--profile-cuda", action="store_true", help="Enable profiler with CUDA events")
    args = parser.parse_args()

    llm_path = os.path.join(args.model_path or MODEL_PATH, LLM_FILE)
    mmproj_path = os.path.join(args.model_path or MODEL_PATH, MMPROJ_FILE)

    # Validate model path
    if not os.path.exists(llm_path):
        print(f"LLM file not found: {llm_path}")
        print("Please specify --model-path")
        sys.exit(1)

    # Tokenizer
    print("Loading tokenizer from GGUF...")
    tokenizer = load_tokenizer(llm_path)
    print(
        f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
        f"bos_id={tokenizer.bos_token_id}, eos_id={tokenizer.eos_token_id}"
    )

    # Logging
    if getattr(args, "trace", False):
        forge.Logger.set_level(5)
    elif getattr(args, "debug", False):
        forge.Logger.set_level(4)
    else:
        forge.Logger.set_level(2 if getattr(args, "verbose", False) else 1)

    # Profiling
    if getattr(args, "profile", False) or getattr(args, "profile_cuda", False):
        forge.profiler_enable()
        forge.profiler_set_cuda_events(getattr(args, "profile_cuda", False))
        print(f"Performance profiler enabled ({'CUDA events' if args.profile_cuda else 'chrono'})")

    # Load multimodal model
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

    # Context + warmup
    cuda_ok = True
    try:
        ctx = mm_model.create_context(args.kv_cache_dtype, args.gpu_layers)
        stats = ctx.memory_stats()
        print(f"KV Cache: dtype={stats.get('kv_cache_dtype', 'unknown')}, size: {stats.get('kv_cache_nbytes', 0) / 1024 / 1024:.1f} MB")
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
