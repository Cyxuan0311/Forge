"""Phi-mini-MoE-instruct interactive chat using Forge.

Supports Phi-mini-MoE models in GGUF format, including MoE FFN (16 experts,
top-2), NeoX RoPE, and norm biases.

Usage:
  python examples/phi_mini_moe_inference.py
  python examples/phi_mini_moe_inference.py --device cpu --gpu-layers 0
  python examples/phi_mini_moe_inference.py --no-stream
  python examples/phi_mini_moe_inference.py --profile

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

MODEL_DIR = "/mnt/g/AI/Phi-mini-MoE-instruct-IQ2_S"
GGUF_MODEL_PATH = os.path.join(MODEL_DIR, "Phi-mini-MoE-instruct-IQ2_S.gguf")

# Phi-3 style: <|end|> is the end-of-turn token (ID 32000)
PHIMOE_END_TOKEN_ID = 32000


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply Phi-3-style chat template for phimoe."""
    user_id = tokenizer.token_to_id("<|user|>")
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    end_id = tokenizer.token_to_id("<|end|>")

    ids = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            ids.append(user_id)
            ids.extend(tokenizer.encode("\n", add_bos=False))
            ids.extend(tokenizer.encode(content, add_bos=False))
            ids.append(end_id)
            ids.extend(tokenizer.encode("\n", add_bos=False))
        elif role == "assistant":
            ids.append(assistant_id)
            ids.extend(tokenizer.encode("\n", add_bos=False))
            ids.extend(tokenizer.encode(content, add_bos=False))
            ids.append(end_id)
            ids.extend(tokenizer.encode("\n", add_bos=False))

    if add_generation_prompt:
        ids.append(assistant_id)
        ids.extend(tokenizer.encode("\n", add_bos=False))

    return ids


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phi-mini-MoE-instruct inference with Forge"
    )
    add_common_args(parser, gpu_layers_default=-1, temperature_default=0.7)
    args = parser.parse_args()

    if args.profile:
        chat_utils_mod.profiling_enabled = True
        print("[Profiling enabled - Python timing + C++ PerfProfiler]")

    model_path = resolve_model_path(args, [GGUF_MODEL_PATH])
    model, tokenizer = load_model_and_tokenize(args, model_path)

    if model is None:
        return

    # Print MoE-specific config
    cfg = model.config
    if cfg.n_expert > 0:
        print(f"MoE config: experts={cfg.n_expert}, top_k={cfg.n_expert_used}, ff_exp={cfg.n_ff_exp}")
    print(f"Architecture: {cfg.arch_type}, layers={cfg.num_layers}, hidden={cfg.hidden_dim}")
    print(f"Heads: q={cfg.num_heads}, kv={cfg.num_kv_heads}, head_dim={cfg.head_dim}")

    interactive_chat(
        model,
        tokenizer,
        args,
        apply_chat_template_fn=apply_chat_template,
        model_name="Phi-mini-MoE-instruct",
        stop_token_ids=[PHIMOE_END_TOKEN_ID],
        system_msg=None,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
