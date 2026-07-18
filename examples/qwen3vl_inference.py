"""Qwen3-VL-4B-Instruct interactive chat using Forge.

Supports text-only and multimodal (image/video) conversation.
For multimodal mode, requires transformers + PIL for image preprocessing.

Prerequisites:
  1. Download GGUF model files:
     - Qwen3-VL-4B-Instruct-Q3_K_M.gguf (LLM)
     - mmproj-F16.gguf (vision encoder, optional for multimodal)

Usage:
  # Text-only mode:
  python examples/qwen3vl_inference.py --model-path /path/to/Qwen3-VL-4B-Instruct-Q3_K_M.gguf

  # With CUDA:
  python examples/qwen3vl_inference.py --device cuda --gpu-layers 36

  # No streaming:
  python examples/qwen3vl_inference.py --no-stream

  # Multimodal mode (requires transformers):
  python examples/qwen3vl_inference.py --multimodal

Interactive commands:
  /image <path>  - Load an image for the next query (multimodal mode only)
  /clear         - Clear conversation history
  /quit          - Exit the chat
  /help          - Show help message
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

# Default model paths
MODEL_DIR = "/mnt/g/AI/Qwen3-VL-4B-Instruct-GGUF"
GGUF_MODEL_PATH = os.path.join(MODEL_DIR, "Qwen3-VL-4B-Instruct-Q3_K_M.gguf")
MMPROJ_PATH = os.path.join(MODEL_DIR, "mmproj-F16.gguf")


def _sanitize_utf8(text: str) -> str:
    """Remove lone surrogate characters that cannot be encoded as UTF-8."""
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply ChatML-style chat template for Qwen3-VL.

    Format:
      <|im_start|>system\n{content}<|im_end|>\n
      <|im_start|>user\n{content}<|im_end|>\n
      <|im_start|>assistant\n
    """
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")

    ids = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # <|im_start|>role\n
        ids.append(im_start_id)
        ids.extend(tokenizer.encode(role + "\n", add_bos=False))

        # content (with optional vision placeholder tokens)
        if isinstance(content, list):
            # Multimodal content: list of text/image segments
            for segment in content:
                if segment["type"] == "image":
                    # Insert vision tokens: <|vision_start|><|image_pad|>...<|vision_end|>
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

        # <|im_end|>\n
        ids.append(im_end_id)
        ids.extend(tokenizer.encode("\n", add_bos=False))

    if add_generation_prompt:
        # <|im_start|>assistant\n
        ids.append(im_start_id)
        ids.extend(tokenizer.encode("assistant\n", add_bos=False))

    return ids


def build_multimodal_embeddings(ctx, tokenizer, prompt_ids, image_embeddings_list):
    """Build full embeddings with image embeddings injected at vision placeholder positions.

    Replaces <|image_pad|> token embeddings with actual vision encoder output.
    """
    prompt_array = np.array(prompt_ids, dtype=np.int32)
    text_embeddings = np.array(ctx.get_embeddings(prompt_array))

    image_pad_id = tokenizer.token_to_id("<|image_pad|>")

    for img_emb in image_embeddings_list:
        emb_len = img_emb.shape[0]
        # Find contiguous <|image_pad|> spans and replace
        pad_positions = [i for i, tid in enumerate(prompt_ids) if tid == image_pad_id]
        if len(pad_positions) >= emb_len:
            for j, pos in enumerate(pad_positions[:emb_len]):
                text_embeddings[pos] = img_emb[j]

    return text_embeddings


def find_think_tokens(tokenizer):
    """Find <think and </think token IDs in the vocabulary."""
    think_start_id = None
    think_end_id = None
    # Check near the end of vocabulary (special tokens are typically added there)
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


def encode_image_with_transformers(image_path, model_name_or_path=None):
    """Encode an image using transformers Qwen3VL processor.

    Returns: (image_embeddings, num_image_tokens) as numpy array and int.
    """
    from PIL import Image
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Use the original HuggingFace model for vision encoding
    if model_name_or_path is None:
        # Try common paths
        candidate_paths = [
            "/mnt/g/AI/Qwen3-VL-4B-Instruct",
            os.path.join(MODEL_DIR, "original"),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                model_name_or_path = p
                break

    if model_name_or_path is None:
        raise FileNotFoundError(
            "Cannot find original HuggingFace model for vision encoding. "
            "Please specify --hf-model-path."
        )

    image = Image.open(image_path).convert("RGB")

    # Load processor and model (vision encoder only)
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )

    # Process image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "placeholder"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    # Extract image embeddings from vision encoder
    with torch.no_grad():
        pixel_values = inputs.pixel_values
        image_embeds = model.visual(pixel_values, grid_thw=inputs.image_grid_thw)
        # image_embeds: [num_image_tokens, hidden_dim]

    return image_embeds.cpu().float().numpy(), image_embeds.shape[1]


def generate_response(
    ctx,
    tokenizer,
    prompt_ids,
    image_embeddings_list=None,
    max_new_tokens=256,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.1,
):
    """Generate a response for a single turn with thinking mode support."""
    # Forward the prompt
    if image_embeddings_list:
        combined_emb = build_multimodal_embeddings(ctx, tokenizer, prompt_ids, image_embeddings_list)
        logits = ctx.forward_with_embeddings(combined_emb, 0)
    else:
        prompt_array = np.array(prompt_ids, dtype=np.int32)
        logits = ctx.forward(prompt_array, 0)

    # Find think tokens
    think_start_id, think_end_id = find_think_tokens(tokenizer)

    start_pos = len(prompt_ids)
    generated_ids = []
    generated_text = ""
    token_buffer = []
    gen_start = time.time()

    # Thinking mode state
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

        # Decode and print (buffered)
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


def interactive_chat(model, tokenizer, args):
    """Interactive multi-turn chat loop with optional image support."""
    conversation = []
    image_embeddings_list = []
    pending_image_content = None  # Multimodal content segments for next turn
    ctx = None

    multimodal = args.multimodal

    print("\n" + "=" * 60)
    print("  Qwen3-VL-4B-Instruct Interactive Chat (Forge)")
    print(f"  Device: {args.device}")
    if multimodal:
        print("  Mode: Multimodal (image support enabled)")
    else:
        print("  Mode: Text-only")
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

        # Handle commands
        if user_input == "/quit":
            print("Bye!")
            break
        elif user_input == "/clear":
            conversation = []
            image_embeddings_list = []
            pending_image_content = None
            print("[Conversation cleared]\n")
            continue
        elif user_input == "/help":
            print("  /clear        - Clear conversation history")
            print("  /quit         - Exit the chat")
            print("  /help         - Show this help message")
            if multimodal:
                print("  /image <path> - Load an image for the next query")
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
                img_emb, num_tokens = encode_image_with_transformers(
                    img_path, args.hf_model_path
                )
                image_embeddings_list.append(img_emb)
                # Build content segments for multimodal prompt
                pending_image_content = [
                    {"type": "image", "num_tokens": num_tokens},
                    {"type": "text", "text": ""},  # placeholder, will be filled with user text
                ]
                print(f"  Image loaded: {num_tokens} tokens, took {time.time() - t0:.2f}s\n")
            except Exception as e:
                print(f"  ERROR encoding image: {e}\n")
                image_embeddings_list = []
                pending_image_content = None
            continue

        # Build message for this turn
        if multimodal and pending_image_content is not None:
            # Update the text segment with user input
            for seg in pending_image_content:
                if seg["type"] == "text":
                    seg["text"] = user_input
            conversation.append({"role": "user", "content": pending_image_content})
            pending_image_content = None
        else:
            conversation.append({"role": "user", "content": user_input})

        # Build prompt
        input_ids = apply_chat_template(tokenizer, conversation, add_generation_prompt=True)

        # Free old context
        if ctx is not None:
            del ctx
            gc.collect()
        ctx = model.create_context(kv_cache_dtype=args.kv_cache_dtype, gpu_layers=args.gpu_layers)

        print("Assistant: ", end="", flush=True)

        try:
            response_text, num_generated, elapsed = generate_response(
                ctx,
                tokenizer,
                input_ids,
                image_embeddings_list if multimodal else None,
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
            conversation.pop()
            continue

        print()
        if num_generated > 0 and elapsed > 0:
            speed = num_generated / elapsed
            print(f"[{num_generated} tokens, {elapsed:.2f}s, {speed:.1f} tok/s]")

        conversation.append({"role": "assistant", "content": response_text})

        # Clear image data after use
        image_embeddings_list = []
        pending_image_content = None
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B-Instruct inference with Forge")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to .gguf model file"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference"
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=36, help="Number of layers to place on GPU (-1 for all)"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="fp32",
        choices=["fp32", "q4_0"],
        help="KV cache data type",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Maximum number of tokens to generate"
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
        "--multimodal", action="store_true", help="Enable multimodal (image) support"
    )
    parser.add_argument(
        "--hf-model-path",
        type=str,
        default=None,
        help="Path to original HuggingFace model (for vision encoding in multimodal mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path
    if model_path is None:
        if os.path.exists(GGUF_MODEL_PATH):
            model_path = GGUF_MODEL_PATH
        else:
            print(f"Model file not found at {GGUF_MODEL_PATH}")
            print("Please specify --model-path")
            sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    # Load tokenizer
    t0 = time.time()
    print("Loading tokenizer from GGUF...")
    tokenizer = forge.Tokenizer()
    tokenizer.load_from_gguf(model_path)
    t1 = time.time()
    print(
        f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
        f"model_type={tokenizer.model_type}, "
        f"bos_id={tokenizer.bos_token_id}, eos_id={tokenizer.eos_token_id} "
        f"[{t1 - t0:.2f}s]"
    )

    # Load model
    print(f"Loading model on {args.device}...")
    forge.Logger.set_level(2 if args.verbose else 1)

    t2 = time.time()
    model = forge.Model()
    model.load_gguf(model_path, device=args.device)
    t3 = time.time()

    cfg = model.config
    print(
        f"Model loaded! arch={cfg.arch_type}, hidden_dim={cfg.hidden_dim}, "
        f"num_layers={cfg.num_layers}, num_heads={cfg.num_heads}, "
        f"num_kv_heads={cfg.num_kv_heads}, use_gqa={cfg.use_gqa}, "
        f"use_mrope={cfg.use_mrope} [{t3 - t2:.2f}s]"
    )

    # Create context
    t4 = time.time()
    ctx = model.create_context(kv_cache_dtype=args.kv_cache_dtype, gpu_layers=args.gpu_layers)
    stats = ctx.memory_stats()
    print(
        f"KV Cache: dtype={stats.get('kv_cache_dtype', 'unknown')}, "
        f"size: {stats.get('kv_cache_nbytes', 0) / 1024 / 1024:.1f} MB"
    )

    # Warmup CUDA kernels
    if args.device == "cuda":
        print("Warming up CUDA kernels...")
        try:
            ctx.warmup()
            print("Warmup done!")
        except RuntimeError as e:
            print(f"Warmup skipped ({e})")

    del ctx
    gc.collect()
    t5 = time.time()
    print(f"Context + warmup: [{t5 - t4:.2f}s]")
    print(f"Total startup: [{t5 - t0:.2f}s]")

    interactive_chat(model, tokenizer, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
