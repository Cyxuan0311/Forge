"""TinyLlama-1.1B-Chat-v1.0 Q4_0 interactive chat using Forge.

Uses Forge's built-in Tokenizer loaded directly from GGUF files.
No external tokenizer files or transformers dependency required.

Prerequisites:
  1. Download GGUF model (if not already present):
     wget https://hf-mirror.com/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

Usage:
  # Interactive chat mode (default):
  python3 examples/tinyllama_inference.py

  # Specify model path:
  python3 examples/tinyllama_inference.py --model-path models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

  # CPU-only mode:
  python3 examples/tinyllama_inference.py --device cpu --gpu-layers 0

  # Other options:
  python3 examples/tinyllama_inference.py --no-stream
  python3 examples/tinyllama_inference.py --max-new-tokens 256 --temperature 0.8
  python3 examples/tinyllama_inference.py --verify-tokenizer

Interactive commands:
  /quit   - Exit the chat
  /clear  - Clear conversation history
  /help   - Show help message
"""

import os
import sys

from chat_utils import (
    add_common_args,
    interactive_chat,
    load_model_and_tokenize,
    profiling_enabled,
    resolve_model_path,
)

import forge

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
GGUF_MODEL_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
TOKENIZER_DIR = os.path.join(MODELS_DIR, "tinyllama-tokenizer")

TINYLLAMA_EOS_ID = 2


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply Zephyr/Llama-2 chat template for TinyLlama."""
    template = tokenizer.chat_template

    if "user|" in template and "assistant|" in template:
        # Zephyr format
        ids = []
        for msg in messages:
            if msg["role"] == "system":
                ids.extend(tokenizer.encode("<|system|>\n", add_bos=False))
                ids.extend(tokenizer.encode(msg["content"], add_bos=False, add_dummy_prefix=False))
                ids.append(tokenizer.eos_token_id)
                ids.extend(tokenizer.encode("\n", add_bos=False, add_dummy_prefix=False))
            elif msg["role"] == "user":
                ids.extend(tokenizer.encode("<|user|>\n", add_bos=False))
                ids.extend(tokenizer.encode(msg["content"], add_bos=False, add_dummy_prefix=False))
                ids.append(tokenizer.eos_token_id)
                ids.extend(tokenizer.encode("\n", add_bos=False, add_dummy_prefix=False))
            elif msg["role"] == "assistant":
                ids.extend(tokenizer.encode("<|assistant|>\n", add_bos=False, add_dummy_prefix=False))
                ids.extend(tokenizer.encode(msg["content"], add_bos=False, add_dummy_prefix=False))
                ids.append(tokenizer.eos_token_id)
                ids.extend(tokenizer.encode("\n", add_bos=False, add_dummy_prefix=False))
        if add_generation_prompt:
            ids.extend(tokenizer.encode("<|assistant|>\n", add_bos=False, add_dummy_prefix=False))
        return ids

    elif "[INST]" in template:
        # Llama-2 format
        ids = []
        for msg in messages:
            if msg["role"] == "user":
                ids.append(tokenizer.bos_token_id)
                ids.extend(tokenizer.encode("[INST] ", add_bos=False))
                ids.extend(tokenizer.encode(msg["content"], add_bos=False, add_dummy_prefix=False))
                ids.extend(tokenizer.encode(" [/INST]", add_bos=False))
            elif msg["role"] == "assistant":
                ids.extend(tokenizer.encode(" ", add_bos=False, add_dummy_prefix=False))
                ids.extend(tokenizer.encode(msg["content"], add_bos=False, add_dummy_prefix=False))
                ids.append(tokenizer.eos_token_id)
        return ids

    else:
        # Fallback: simple format
        ids = [tokenizer.bos_token_id]
        for msg in messages:
            ids.extend(tokenizer.encode(f"{msg['role']}: {msg['content']}\n", add_bos=False))
        return ids


def download_model(model_path):
    if os.path.exists(model_path):
        return True
    print(f"Model not found at {model_path}")
    print("Downloading model from hf-mirror.com ...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    ret = os.system(
        f"wget -c https://hf-mirror.com/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf "
        f"-O {model_path}"
    )
    if ret != 0:
        print("Failed to download model. Please download manually:")
        print(
            "  wget https://hf-mirror.com/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
        )
        return False
    print("Model downloaded successfully!")
    return True


def main():
    global profiling_enabled

    parser = __import__("argparse").ArgumentParser(description="TinyLlama-1.1B-Chat inference with Forge")
    add_common_args(parser, gpu_layers_default=22, temperature_default=0.7)
    args = parser.parse_args()

    if args.profile:
        profiling_enabled = True
        print("[Profiling enabled - Python timing + C++ PerfProfiler]")

    model_path = args.model_path
    if model_path is None:
        model_path = GGUF_MODEL_PATH
    if not os.path.exists(model_path):
        if not download_model(model_path):
            sys.exit(1)

    args.tokenizer_dir = TOKENIZER_DIR
    model, tokenizer = load_model_and_tokenize(args, model_path)

    if model is None:
        return  # verify-tokenizer mode

    interactive_chat(
        model, tokenizer, args,
        apply_chat_template_fn=apply_chat_template,
        model_name="TinyLlama-1.1B-Chat",
        eos_token_id=TINYLLAMA_EOS_ID,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
