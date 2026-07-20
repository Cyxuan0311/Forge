"""Qwen3-VL-4B-Instruct interactive chat using Forge.

Supports text-only and multimodal (image/video) conversation.
Uses Forge's built-in ViT for vision encoding (no HuggingFace transformers needed).

Usage:
  python examples/qwen3vl_inference.py --model-path /path/to/Qwen3-VL-4B-Instruct-Q3_K_M.gguf
  python examples/qwen3vl_inference.py --multimodal
  python examples/qwen3vl_inference.py --multimodal --device cpu

Interactive commands:
  /image <path>  - Load an image for the next query (multimodal mode only)
  /video <path>  - Load a video for the next query (multimodal mode only)
  /clear         - Clear conversation history
  /quit          - Exit the chat
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

MODEL_DIR = "/mnt/g/AI/Qwen3-VL-4B-Instruct-GGUF"
GGUF_MODEL_PATH = os.path.join(MODEL_DIR, "Qwen3-VL-4B-Instruct-Q3_K_M.gguf")
MMPROJ_PATH = os.path.join(MODEL_DIR, "mmproj-F16.gguf")


def _sanitize_utf8(text: str) -> str:
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


def _extract_frames(video_path, num_frames=8):
    """Extract frames from video using decord."""
    import decord
    vr = decord.VideoReader(video_path)
    total = len(vr)
    indices = [int(total * i / num_frames) for i in range(num_frames)]
    return vr.get_batch(indices).asnumpy()


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply ChatML-style chat template for Qwen3-VL."""
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")

    ids = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        ids.append(im_start_id)
        ids.extend(tokenizer.encode(role + "\n", add_bos=False))

        if isinstance(content, list):
            for segment in content:
                if segment["type"] == "image":
                    vision_start_id = tokenizer.token_to_id("<|vision_start|>")
                    vision_end_id = tokenizer.token_to_id("<|vision_end|>")
                    image_pad_id = tokenizer.token_to_id("<|image_pad|>")
                    num_pad = segment.get("num_tokens", 256)
                    ids.append(vision_start_id)
                    ids.extend([image_pad_id] * num_pad)
                    ids.append(vision_end_id)
                elif segment["type"] == "text":
                    ids.extend(tokenizer.encode(segment["text"], add_bos=False))
        else:
            ids.extend(tokenizer.encode(content, add_bos=False))

        ids.append(im_end_id)
        ids.extend(tokenizer.encode("\n", add_bos=False))

    if add_generation_prompt:
        ids.append(im_start_id)
        ids.extend(tokenizer.encode("assistant\n", add_bos=False))

    return ids


def build_multimodal_embeddings(ctx, tokenizer, prompt_ids, image_embeddings_list):
    """Build full embeddings with image embeddings injected at vision placeholder positions."""
    prompt_array = np.array(prompt_ids, dtype=np.int32)
    text_embeddings = np.array(ctx.get_embeddings(prompt_array))
    image_pad_id = tokenizer.token_to_id("<|image_pad|>")

    for img_emb in image_embeddings_list:
        emb_len = img_emb.shape[0]
        pad_positions = [i for i, tid in enumerate(prompt_ids) if tid == image_pad_id]
        if len(pad_positions) >= emb_len:
            for j, pos in enumerate(pad_positions[:emb_len]):
                text_embeddings[pos] = img_emb[j]

    return text_embeddings


def find_think_tokens(tokenizer):
    """Find <think and </think token IDs in the vocabulary."""
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


def generate_response(ctx, tokenizer, prompt_ids, image_embeddings_list=None,
                      max_new_tokens=256, temperature=0.7, top_k=40, top_p=0.9,
                      repeat_penalty=1.1):
    """Generate a response for a single turn with thinking mode support."""
    if image_embeddings_list:
        combined_emb = build_multimodal_embeddings(ctx, tokenizer, prompt_ids, image_embeddings_list)
        logits = ctx.forward_with_embeddings(combined_emb, 0)
    else:
        prompt_array = np.array(prompt_ids, dtype=np.int32)
        logits = ctx.forward(prompt_array, 0)

    think_start_id, think_end_id = find_think_tokens(tokenizer)

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
                recent_ids = generated_ids[-64:]
                for tid in set(recent_ids):
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
        elif think_end_id is not None and next_token == think_end_id:
            if in_thinking and thinking_text:
                print(f"\n[Thinking: {thinking_text.strip()}]\n", end="", flush=True)
                thinking_text = ""
            in_thinking = False
        else:
            token_buffer.append(next_token)

        # Try to decode buffered tokens (handles split multi-byte UTF-8)
        if len(token_buffer) >= 4:
            try:
                text = tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
                if in_thinking:
                    thinking_text += text
                else:
                    print(text, end="", flush=True)
                    generated_text += text
                token_buffer.clear()
            except (UnicodeDecodeError, UnicodeError):
                # Multi-byte char split across token boundary — keep accumulating.
                # If buffer grows too large, fall back to full-sequence decode.
                if len(token_buffer) > 16:
                    full_text = tokenizer.decode(generated_ids, skip_special=True,
                                                 strip_leading_space=False)
                    text = full_text[len(generated_text):]
                    if in_thinking:
                        thinking_text += text
                    else:
                        print(text, end="", flush=True)
                        generated_text += text
                    token_buffer.clear()
                # else: wait for more tokens to complete the multi-byte char

        # Always advance the model — forward pass must not be skipped
        next_ids = np.array([next_token], dtype=np.int32)
        try:
            logits = ctx.forward(next_ids, start_pos)
        except Exception as e:
            print(f"\nERROR at step {step}: {e}")
            break
        start_pos += 1

    if token_buffer:
        try:
            text = tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
        except (UnicodeDecodeError, UnicodeError):
            # Decode all generated IDs to get the tail portion
            full_text = tokenizer.decode(generated_ids, skip_special=True,
                                         strip_leading_space=False)
            text = full_text[len(generated_text):]
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


def interactive_chat_vl(mm_model, tokenizer, args):
    """Interactive chat with optional image/video support using Forge's built-in ViT."""
    conversation = []
    image_embeddings_list = []
    pending_image_content = None
    ctx = None
    multimodal = args.multimodal

    print("\n" + "=" * 60)
    print("  Qwen3-VL-4B-Instruct Interactive Chat (Forge)")
    print(f"  Device: {args.device}")
    print("  Vision: Forge built-in ViT (no HuggingFace transformers)")
    print(f"  Mode: {'Multimodal' if multimodal else 'Text-only'}")
    print("  Commands: /clear, /quit, /help" + (", /image <path>, /video <path>" if multimodal else ""))
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
            image_embeddings_list = []
            pending_image_content = None
            print("[Conversation cleared]\n")
            continue
        elif user_input == "/help":
            print("  /clear, /quit, /help")
            if multimodal:
                print("  /image <path>  - Load an image for the next query")
                print("  /video <path>  - Load a video for the next query")
            print()
            continue
        elif multimodal and user_input.startswith("/image "):
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
                num_tokens = emb.shape[0]
                image_embeddings_list.append(emb)
                pending_image_content = [
                    {"type": "image", "num_tokens": num_tokens},
                    {"type": "text", "text": ""},
                ]
                print(f"  Image loaded: {img.size}, {num_tokens} tokens, took {time.time() - t0:.2f}s\n")
            except Exception as e:
                print(f"  ERROR encoding image: {e}\n")
                image_embeddings_list = []
                pending_image_content = None
            continue
        elif multimodal and user_input.startswith("/video "):
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
                all_embs = np.concatenate(frame_embs, axis=0)
                total_tokens = all_embs.shape[0]
                # Each frame's tokens need separate image segments
                for emb in frame_embs:
                    image_embeddings_list.append(emb)
                pending_image_content = [
                    {"type": "image", "num_tokens": frame_embs[0].shape[0]},
                    {"type": "text", "text": ""},
                ]
                # For multiple frames, we need multiple image segments
                if len(frame_embs) > 1:
                    pending_image_content = []
                    for emb in frame_embs:
                        pending_image_content.append({"type": "image", "num_tokens": emb.shape[0]})
                    pending_image_content.append({"type": "text", "text": ""})
                print(f"  Video loaded: {len(frames)} frames, {total_tokens} tokens, took {time.time()-t0:.2f}s\n")
            except ImportError:
                print("  ERROR: decord not installed. Run: pip install decord\n")
                image_embeddings_list = []
                pending_image_content = None
            except Exception as e:
                print(f"  ERROR processing video: {e}\n")
                image_embeddings_list = []
                pending_image_content = None
            continue

        if multimodal and pending_image_content is not None:
            for seg in pending_image_content:
                if seg["type"] == "text":
                    seg["text"] = user_input
            conversation.append({"role": "user", "content": pending_image_content})
            pending_image_content = None
        else:
            conversation.append({"role": "user", "content": user_input})

        input_ids = apply_chat_template(tokenizer, conversation, add_generation_prompt=True)

        if ctx is not None:
            del ctx
            gc.collect()
        ctx = mm_model.create_context(args.kv_cache_dtype, args.gpu_layers)

        print("Assistant: ", end="", flush=True)

        try:
            response_text, num_generated, elapsed = generate_response(
                ctx, tokenizer, input_ids,
                image_embeddings_list if multimodal else None,
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
        image_embeddings_list = []
        pending_image_content = None
        print()


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B-Instruct inference with Forge")
    add_common_args(parser, gpu_layers_default=36, temperature_default=0.7)
    parser.add_argument("--multimodal", action="store_true", help="Enable multimodal (image/video) support")
    parser.add_argument("--video-frames", type=int, default=8, help="Number of frames to sample for video input")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--trace", action="store_true", help="Enable TRACE logging")
    args = parser.parse_args()

    # Resolve model paths
    model_path = args.model_path if args.model_path else GGUF_MODEL_PATH
    if not os.path.exists(model_path):
        # Try as directory containing GGUF
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "Qwen3-VL-4B-Instruct-Q3_K_M.gguf")

    # mmproj is in the same directory as the GGUF model
    model_dir = os.path.dirname(model_path)
    mmproj_path = os.path.join(model_dir, "mmproj-F16.gguf")

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please specify --model-path")
        sys.exit(1)

    # Tokenizer
    print("Loading tokenizer from GGUF...")
    tokenizer = load_tokenizer(model_path)
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

    # Load multimodal model with Forge's built-in ViT
    print(f"Loading multimodal model on {args.device}...")
    if not os.path.exists(mmproj_path):
        print(f"WARNING: mmproj not found at {mmproj_path}")
        print("  Multimodal support will not be available.")
        mmproj_path = ""

    mm_model = forge.MultimodalModel()
    try:
        mm_model.load_with_mmproj(model_path, mmproj_path, args.device)
    except RuntimeError as e:
        print(f"  CUDA failed ({e}), falling back to CPU")
        args.device = "cpu"
        args.gpu_layers = 0
        mm_model.load_with_mmproj(model_path, mmproj_path, args.device)

    cfg = mm_model.config
    print(
        f"Model loaded! arch={cfg.arch_type}, hidden_dim={cfg.hidden_dim}, "
        f"num_layers={cfg.num_layers}, num_heads={cfg.num_heads}"
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

    interactive_chat_vl(mm_model, tokenizer, args)
    print("\nDone!")


if __name__ == "__main__":
    main()
