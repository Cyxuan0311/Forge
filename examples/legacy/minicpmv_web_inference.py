#!/usr/bin/env python3
"""MiniCPM-V 4.6 Multimodal Inference - Web Interface

Web-based inference interface for MiniCPM-V-4.6.F16 multimodal model.
Supports image upload and multi-turn conversation.

Usage:
    python3 examples/minicpmv_web_inference.py

Requirements:
    pip install gradio Pillow
"""

import sys
import os
import gc
import numpy as np
from pathlib import Path

from chat_utils import load_tokenizer

import forge

# Configuration
MODEL_DIR = "/mnt/g/AI/MiniCPM-V-4.6.F16"
DEVICE = "cuda"
GPU_LAYERS = -1
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.9
REPEAT_PENALTY = 1.1
KV_CACHE_DTYPE = "fp32"


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply MiniCPM-V chat template (Qwen-style ChatML)."""
    token_ids = []
    image_insert_start = 0
    num_img_tokens = 0

    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    img_start_id = tokenizer.token_to_id("<image>")
    img_end_id = tokenizer.token_to_id("</image>")

    newline_ids = list(tokenizer.encode("\n", add_bos=False))
    user_text_ids = list(tokenizer.encode("user", add_bos=False))
    assistant_text_ids = list(tokenizer.encode("assistant", add_bos=False))

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        has_image = msg.get("has_image", False)

        if role == "user":
            n_img = msg.get("num_image_tokens", 0) if has_image else 0
            if n_img > 0:
                num_img_tokens = n_img
                prompt_ids = list(tokenizer.encode(content, add_bos=False))
                user_ids = (
                    [im_start_id] + user_text_ids + newline_ids
                    + [img_start_id] + [0] * n_img + [img_end_id]
                    + prompt_ids + [im_end_id] + newline_ids
                )
                image_insert_start = len(token_ids) + 1 + len(user_text_ids) + len(newline_ids) + 1
                token_ids.extend(user_ids)
            else:
                prompt_ids = list(tokenizer.encode(content, add_bos=False))
                user_ids = (
                    [im_start_id] + user_text_ids + newline_ids
                    + prompt_ids + [im_end_id] + newline_ids
                )
                token_ids.extend(user_ids)

        elif role == "assistant":
            asst_ids = list(tokenizer.encode(content, add_bos=False))
            asst_full = [im_start_id] + assistant_text_ids + newline_ids + asst_ids + [im_end_id] + newline_ids
            token_ids.extend(asst_full)

    if add_generation_prompt:
        token_ids.extend([im_start_id] + assistant_text_ids + newline_ids)

    return token_ids, image_insert_start, num_img_tokens


def sample_token(logits, temperature, top_k, top_p, repeat_penalty, generated_ids):
    if isinstance(logits, np.ndarray):
        logits_1d = logits[-1].copy()
    else:
        logits_1d = np.array(logits)[-1].copy()

    logits_1d = np.where(np.isfinite(logits_1d), logits_1d, -1e9)

    if repeat_penalty != 1.0 and len(generated_ids) > 0:
        for tid in set(generated_ids[-64:]):
            if 0 <= tid < len(logits_1d):
                if logits_1d[tid] > 0:
                    logits_1d[tid] /= repeat_penalty
                else:
                    logits_1d[tid] *= repeat_penalty

    if temperature > 0:
        shifted = logits_1d - np.max(logits_1d)
        probs = np.exp(shifted / temperature)
        probs = probs / probs.sum()
        k = min(top_k, len(probs))
        top_indices = np.argsort(probs)[-k:]
        top_probs = probs[top_indices]
        top_probs = top_probs / top_probs.sum()
        token_id = int(np.random.choice(top_indices, p=top_probs))
    else:
        token_id = int(np.argmax(logits_1d))

    return token_id


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


class MiniCPMVModel:
    def __init__(self, model_dir, device="cuda", gpu_layers=-1):
        self.model_dir = model_dir
        self.device = device
        self.gpu_layers = gpu_layers
        self.tokenizer = None
        self.mm_model = None
        self.conversation_history = []
        self.think_start_id = None
        self.think_end_id = None

    def load(self):
        print(f"Loading model from {self.model_dir}...")
        llm_path, mmproj_path = self._find_gguf_files()
        if not llm_path:
            raise FileNotFoundError(f"No LLM GGUF file found in {self.model_dir}")

        print(f"  LLM: {os.path.basename(llm_path)}")
        print(f"  mmproj: {os.path.basename(mmproj_path) if mmproj_path else 'N/A'}")

        print("Loading tokenizer...")
        self.tokenizer = load_tokenizer(llm_path)
        print(f"  vocab_size={self.tokenizer.vocab_size}, bos_id={self.tokenizer.bos_token_id}, eos_id={self.tokenizer.eos_token_id}")

        print("Loading multimodal model...")
        self.mm_model = forge.MultimodalModel()
        try:
            self.mm_model.load_with_mmproj(llm_path, mmproj_path, self.device)
        except RuntimeError as e:
            print(f"  CUDA failed ({e}), falling back to CPU")
            self.device = "cpu"
            self.gpu_layers = 0
            self.mm_model.load_with_mmproj(llm_path, mmproj_path, self.device)

        cfg = self.mm_model.config
        print(f"  arch={cfg.arch_type}, hidden_dim={cfg.hidden_dim}, num_layers={cfg.num_layers}, num_heads={cfg.num_heads}, use_ssm={cfg.use_ssm}")

        self.think_start_id, self.think_end_id = find_thinking_tokens(self.tokenizer)
        if self.think_start_id is not None:
            print(f"  Thinking mode: <think={self.think_start_id}, </think={self.think_end_id}")

        if self.device == "cuda":
            print("Warming up CUDA kernels...")
            try:
                ctx = self.mm_model.create_context(KV_CACHE_DTYPE, self.gpu_layers)
                ctx.warmup()
                del ctx
                gc.collect()
                print("Warmup done!")
            except RuntimeError as e:
                print(f"Warmup skipped ({e})")

        print("Model loaded successfully!")
        return True

    def _find_gguf_files(self):
        model_dir = Path(self.model_dir)
        if model_dir.is_file() and model_dir.suffix == ".gguf":
            return str(model_dir), None
        if not model_dir.is_dir():
            return None, None

        gguf_files = list(model_dir.glob("*.gguf"))
        llm_path = None
        mmproj_path = None
        for f in gguf_files:
            name_lower = f.name.lower()
            if "mmproj" in name_lower or "vision" in name_lower or "clip" in name_lower:
                mmproj_path = str(f)
            else:
                llm_path = str(f)
        if llm_path is None and len(gguf_files) == 1:
            llm_path = str(gguf_files[0])
        return llm_path, mmproj_path

    def encode_image(self, image):
        if isinstance(image, np.ndarray):
            img_array = image
        else:
            img_array = np.array(image.convert("RGB"))
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)
        img_array = img_array.astype(np.uint8)
        return self.mm_model.encode_image(img_array)

    def chat(self, user_message, image=None, max_new_tokens=MAX_NEW_TOKENS,
             temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, reset_conversation=False):
        if reset_conversation:
            self.conversation_history = []

        image_embeddings = None
        num_img_tokens = 0
        has_image = image is not None

        if has_image:
            image_embeddings = self.encode_image(image)
            num_img_tokens = image_embeddings.shape[0]

        self.conversation_history.append({"role": "user", "content": user_message, "has_image": has_image, "num_image_tokens": num_img_tokens})

        prompt_ids, img_insert_start, n_img = apply_chat_template(self.tokenizer, self.conversation_history)
        ctx = self.mm_model.create_context(KV_CACHE_DTYPE, self.gpu_layers)

        if has_image and image_embeddings is not None:
            prompt_array = np.array(prompt_ids, dtype=np.int32)
            text_embeddings = np.array(ctx.get_embeddings(prompt_array))
            if n_img > 0 and image_embeddings is not None:
                text_embeddings[img_insert_start:img_insert_start + num_img_tokens] = image_embeddings
            logits = ctx.forward_with_embeddings(text_embeddings, 0)
        else:
            prompt_array = np.array(prompt_ids, dtype=np.int32)
            logits = ctx.forward(prompt_array, 0)

        start_pos = len(prompt_ids)
        generated_ids = []
        generated_text = ""
        thinking_text = ""
        in_thinking = False
        token_buffer = []

        for step in range(max_new_tokens):
            next_token = sample_token(logits, temperature, top_k, top_p, REPEAT_PENALTY, generated_ids)
            generated_ids.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break
            if self.think_start_id is not None and next_token == self.think_start_id:
                in_thinking = True
                continue
            if self.think_end_id is not None and next_token == self.think_end_id:
                in_thinking = False
                continue

            token_buffer.append(next_token)
            if len(token_buffer) >= 4:
                text = self.tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
                if in_thinking:
                    thinking_text += text
                else:
                    generated_text += text
                token_buffer.clear()

            next_ids = np.array([next_token], dtype=np.int32)
            try:
                logits = ctx.forward(next_ids, start_pos)
            except Exception:
                break
            start_pos += 1

        if token_buffer:
            text = self.tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
            if in_thinking:
                thinking_text += text
            else:
                generated_text += text

        del ctx
        gc.collect()

        self.conversation_history.append({"role": "assistant", "content": generated_text.strip(), "has_image": False})

        result = generated_text.strip()
        if thinking_text.strip():
            result = f"[Thinking: {thinking_text.strip()}]\n\n{result}"
        return result

    def reset(self):
        self.conversation_history = []


def create_web_interface(model):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        sys.exit(1)

    from PIL import Image

    def chat_fn(message, image, history, temperature, top_k, top_p, max_tokens):
        if not message.strip() and image is None:
            return history, ""

        img_array = None
        if image is not None:
            if isinstance(image, dict):
                img = image.get("image", image.get("path", None))
                if isinstance(img, str):
                    img = Image.open(img)
                img_array = np.array(img.convert("RGB"))
            elif isinstance(image, str):
                img = Image.open(image)
                img_array = np.array(img.convert("RGB"))
            else:
                img_array = np.array(image.convert("RGB") if hasattr(image, "convert") else image)

        try:
            response = model.chat(
                user_message=message, image=img_array if img_array is not None else None,
                max_new_tokens=int(max_tokens), temperature=temperature,
                top_k=int(top_k), top_p=top_p,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            response = f"Error: {str(e)}"

        history = history or []
        if image is not None:
            user_msg = f"[Image uploaded] {message}"
        else:
            user_msg = message
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": response})
        return history, ""

    def reset_fn():
        model.reset()
        return [], ""

    with gr.Blocks(title="MiniCPM-V 4.6 - Forge") as demo:
        gr.Markdown("# MiniCPM-V 4.6 Multimodal Chat\nPowered by Forge")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                with gr.Row():
                    msg_input = gr.Textbox(placeholder="Type your message here...", show_label=False, scale=4)
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Upload Image (optional)", height=200)
                    clear_btn = gr.Button("Clear Conversation")
            with gr.Column(scale=1):
                gr.Markdown("### Generation Parameters")
                temperature = gr.Slider(minimum=0.0, maximum=2.0, value=TEMPERATURE, step=0.1, label="Temperature")
                top_k = gr.Slider(minimum=0, maximum=200, value=TOP_K, step=1, label="Top-K")
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=TOP_P, step=0.05, label="Top-P")
                max_tokens = gr.Slider(minimum=32, maximum=2048, value=MAX_NEW_TOKENS, step=32, label="Max Tokens")

        submit_btn.click(chat_fn, inputs=[msg_input, image_input, chatbot, temperature, top_k, top_p, max_tokens], outputs=[chatbot, msg_input])
        msg_input.submit(chat_fn, inputs=[msg_input, image_input, chatbot, temperature, top_k, top_p, max_tokens], outputs=[chatbot, msg_input])
        clear_btn.click(reset_fn, outputs=[chatbot, msg_input])

    return demo


def main():
    print("=" * 60)
    print("MiniCPM-V 4.6 Multimodal Inference - Forge")
    print("=" * 60)

    model = MiniCPMVModel(MODEL_DIR, DEVICE, GPU_LAYERS)

    try:
        model.load()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nFailed to load model: {e}")
        try:
            import gradio as gr
        except ImportError:
            print("Gradio not installed. Install with: pip install gradio")
            return
        with gr.Blocks(title="MiniCPM-V 4.6 - Error") as demo:
            gr.Markdown(f"# Error Loading Model\n\n{e}")
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
        return

    demo = create_web_interface(model)
    print("\nLaunching web interface at http://0.0.0.0:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
