"""Qwen3-VL-4B-Instruct interactive chat using Forge.

Supports text-only and multimodal (image/video) conversation.

Usage:
  python examples/qwen3vl_inference.py --model-path /path/to/Qwen3-VL-4B-Instruct-Q3_K_M.gguf
  python examples/qwen3vl_inference.py --multimodal

Interactive commands:
  /image <path>  - Load an image for the next query (multimodal mode only)
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
    PerfTimer,
    add_common_args,
    load_model_and_tokenize,
    load_tokenizer,
    profiling_enabled,
    resolve_model_path,
)

import forge

MODEL_DIR = "/mnt/g/AI/Qwen3-VL-4B-Instruct-GGUF"
GGUF_MODEL_PATH = os.path.join(MODEL_DIR, "Qwen3-VL-4B-Instruct-Q3_K_M.gguf")
MMPROJ_PATH = os.path.join(MODEL_DIR, "mmproj-F16.gguf")


def _sanitize_utf8(text: str) -> str:
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


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


def encode_image_with_transformers(image_path, model_name_or_path=None):
    """Encode an image using transformers Qwen3VL processor."""
    from PIL import Image
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    if model_name_or_path is None:
        candidate_paths = [
            "/mnt/g/AI/Qwen3-VL-4B-Instruct",
            os.path.join(MODEL_DIR, "original"),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                model_name_or_path = p
                break

    if model_name_or_path is None:
        raise FileNotFoundError("Cannot find original HuggingFace model for vision encoding.")

    image = Image.open(image_path).convert("RGB")
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "placeholder"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        image_embeds = model.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw)

    return image_embeds.cpu().float().numpy(), image_embeds.shape[1]


def interactive_chat_vl(model, tokenizer, args):
    """Interactive chat with optional image support."""
    conversation = []
    image_embeddings_list = []
    pending_image_content = None
    ctx = None
    multimodal = args.multimodal

    print("\n" + "=" * 60)
    print(f"  Qwen3-VL-4B-Instruct Interactive Chat (Forge)")
    print(f"  Device: {args.device}")
    print(f"  Mode: {'Multimodal' if multimodal else 'Text-only'}")
    print("  Commands: /clear, /quit, /help" + (", /image <path>" if multimodal else ""))
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
                print("  /image <path>")
            print()
            continue
        elif multimodal and user_input.startswith("/image "):
            img_path = user_input[7:].strip()
            if not os.path.exists(img_path):
                print(f"  Image not found: {img_path}\n")
                continue
            try:
                print(f"  Encoding image: {img_path}...")
                t0 = time.time()
                img_emb, num_tokens = encode_image_with_transformers(img_path, args.hf_model_path)
                image_embeddings_list.append(img_emb)
                pending_image_content = [
                    {"type": "image", "num_tokens": num_tokens},
                    {"type": "text", "text": ""},
                ]
                print(f"  Image loaded: {num_tokens} tokens, took {time.time() - t0:.2f}s\n")
            except Exception as e:
                print(f"  ERROR encoding image: {e}\n")
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
        ctx = model.create_context(kv_cache_dtype=args.kv_cache_dtype, gpu_layers=args.gpu_layers)

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
    parser.add_argument("--multimodal", action="store_true", help="Enable multimodal (image) support")
    parser.add_argument("--hf-model-path", type=str, default=None, help="Path to HuggingFace model for vision encoding")
    args = parser.parse_args()

    model_path = resolve_model_path(args, [GGUF_MODEL_PATH])
    model, tokenizer = load_model_and_tokenize(args, model_path)

    if model is None:
        return

    interactive_chat_vl(model, tokenizer, args)
    print("\nDone!")


if __name__ == "__main__":
    main()
