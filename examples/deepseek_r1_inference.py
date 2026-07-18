"""DeepSeek-R1-Distill-Qwen-7B Q4_K_M interactive chat using Forge.

Uses Forge's built-in Tokenizer loaded directly from GGUF files.

Usage:
  python examples/deepseek_r1_inference.py --model-path /mnt/g/AI/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
  python examples/deepseek_r1_inference.py --device cpu --gpu-layers 0
  python examples/deepseek_r1_inference.py --profile

Interactive commands:
  /quit    - Exit the chat
  /clear   - Clear conversation history
  /help    - Show help message
  /profile - Toggle profiling on/off
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

GGUF_MODEL_PATH = "/mnt/g/AI/DeepSeek-R1-Distill-Qwen-7B/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
TOKENIZER_DIR = "/mnt/g/AI/DeepSeek-R1-Distill-Qwen-7B"


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply DeepSeek-R1 chat template."""
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    user_id = tokenizer.token_to_id("<｜User｜>")
    asst_id = tokenizer.token_to_id("<｜Assistant｜>")

    ids = []

    if bos_id >= 0:
        ids.append(bos_id)

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            ids.extend(tokenizer.encode(content, add_bos=False))
        elif role == "user":
            ids.append(user_id)
            ids.extend(tokenizer.encode(content, add_bos=False))
        elif role == "assistant":
            ids.append(asst_id)
            ids.extend(tokenizer.encode(content, add_bos=False))
            ids.append(eos_id)

    if add_generation_prompt:
        ids.append(asst_id)

    return ids


def main():
    global profiling_enabled

    parser = __import__("argparse").ArgumentParser(
        description="DeepSeek-R1-Distill-Qwen-7B inference with Forge"
    )
    add_common_args(parser, gpu_layers_default=28, temperature_default=0.6)
    args = parser.parse_args()

    if args.profile:
        profiling_enabled = True
        print("[Profiling enabled - Python timing + C++ PerfProfiler]")

    model_path = resolve_model_path(args, [GGUF_MODEL_PATH])
    args.tokenizer_dir = TOKENIZER_DIR
    model, tokenizer = load_model_and_tokenize(args, model_path)

    if model is None:
        return

    interactive_chat(
        model, tokenizer, args,
        apply_chat_template_fn=apply_chat_template,
        model_name="DeepSeek-R1-Distill-Qwen-7B",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
