"""Meta-Llama-3.1-8B-Instruct interactive chat using Forge.

Uses Forge's built-in Tokenizer loaded directly from GGUF files.
No external tokenizer files or transformers dependency required.

Usage:
  python examples/llama3_inference.py
  python examples/llama3_inference.py --device cpu --gpu-layers 0
  python examples/llama3_inference.py --no-stream
  python examples/llama3_inference.py --profile
  python examples/llama3_inference.py --model-path /mnt/g/AI/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf

Interactive commands:
  /quit    - Exit the chat
  /clear   - Clear conversation history
  /help    - Show help message
  /profile - Toggle profiling on/off
"""

import os
import sys

# Allow running from within examples/legacy/ — add examples/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import chat_utils as chat_utils_mod
from chat_utils import (
    add_common_args,
    interactive_chat,
    load_model_and_tokenize,
    resolve_model_path,
)

MODEL_DIR = "/mnt/g/AI/Meta-Llama-3.1-8B-Instruct"
GGUF_MODEL_PATH = os.path.join(MODEL_DIR, "Meta-Llama-3.1-8B-Instruct-Q2_K.gguf")

LLAMA31_EOS_ID = 128009


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply LLaMA 3.1 chat template."""
    start_header_id = tokenizer.token_to_id("<|start_header_id|>")
    end_header_id = tokenizer.token_to_id("<|end_header_id|>")
    eot_id = tokenizer.token_to_id("<|eot_id|>")

    ids = [tokenizer.bos_token_id]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        ids.append(start_header_id)
        ids.extend(tokenizer.encode(role, add_bos=False))
        ids.append(end_header_id)
        ids.extend(tokenizer.encode("\n\n", add_bos=False))
        ids.extend(tokenizer.encode(content, add_bos=False))
        ids.append(eot_id)

    if add_generation_prompt:
        ids.append(start_header_id)
        ids.extend(tokenizer.encode("assistant", add_bos=False))
        ids.append(end_header_id)
        ids.extend(tokenizer.encode("\n\n", add_bos=False))

    return ids


def main():
    parser = __import__("argparse").ArgumentParser(
        description="Meta-Llama-3.1-8B-Instruct inference with Forge"
    )
    add_common_args(parser, gpu_layers_default=32, temperature_default=0.7)
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
        model_name="Meta-Llama-3.1-8B-Instruct",
        eos_token_id=LLAMA31_EOS_ID,
        system_msg={"role": "system", "content": "You are a helpful assistant."},
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
