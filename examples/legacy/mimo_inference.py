"""MiMo-7B-RL-Q4_K_M interactive chat using Forge.

Uses Forge's built-in Tokenizer loaded directly from GGUF files.
No external tokenizer files or transformers dependency required.

Usage:
  python examples/mimo_inference.py
  python examples/mimo_inference.py --device cpu --gpu-layers 0
  python examples/mimo_inference.py --no-stream
  python examples/mimo_inference.py --profile
  python examples/mimo_inference.py --model-path /path/to/model.gguf

Interactive commands:
  /quit    - Exit the chat
  /clear   - Clear conversation history
  /help    - Show help message
  /profile - Toggle profiling on/off
"""

import os

import chat_utils as chat_utils_mod
from chat_utils import (
    add_common_args,
    interactive_chat,
    load_model_and_tokenize,
    resolve_model_path,
)

MODEL_DIR = "/mnt/g/AI/MiMo-7B-RL-Q4_K_M"
GGUF_MODEL_PATH = os.path.join(MODEL_DIR, "MiMo-7B-RL-Q4_K_M.gguf")

MIMO_EOS_ID = 151645


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply ChatML template for MiMo."""
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")

    ids = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        ids.append(im_start_id)
        ids.extend(tokenizer.encode(role + "\n", add_bos=False))
        ids.extend(tokenizer.encode(content, add_bos=False))
        ids.append(im_end_id)
        ids.extend(tokenizer.encode("\n", add_bos=False))

    if add_generation_prompt:
        ids.append(im_start_id)
        ids.extend(tokenizer.encode("assistant\n", add_bos=False))

    return ids


def main():
    parser = __import__("argparse").ArgumentParser(description="MiMo-7B-RL inference with Forge")
    add_common_args(parser, gpu_layers_default=28, temperature_default=0.7)
    args = parser.parse_args()

    if args.profile:
        chat_utils_mod.profiling_enabled = True
        print("[Profiling enabled - Python timing + C++ PerfProfiler]")

    model_path = resolve_model_path(args, [GGUF_MODEL_PATH])
    args.tokenizer_dir = MODEL_DIR
    model, tokenizer = load_model_and_tokenize(args, model_path)

    if model is None:
        return

    interactive_chat(
        model,
        tokenizer,
        args,
        apply_chat_template_fn=apply_chat_template,
        model_name="MiMo-7B-RL",
        eos_token_id=MIMO_EOS_ID,
        system_msg={"role": "system", "content": "You are a helpful assistant."},
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
