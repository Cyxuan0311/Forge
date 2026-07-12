"""
Generate a synthetic .ninf test model with random FP32 weights.

Usage:
    from tools.create_test_model import create_test_model
    create_test_model('tests/fixtures/test_model.ninf')
"""

import numpy as np

from tools.format import NINF_DTYPE_FP32, build_ninf


def create_test_model(
    output_path,
    vocab_size=3200,
    hidden_dim=256,
    intermediate_dim=512,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=64,
):
    tensors = []

    npy = np.random.default_rng(42)

    def add(name, shape):
        data = npy.normal(0, 0.02, size=shape).astype(np.float32).tobytes()
        tensors.append((name, NINF_DTYPE_FP32, list(shape), data))

    # Global tensors
    add("model.embed_tokens.weight", (vocab_size, hidden_dim))
    add("model.norm.weight", (hidden_dim,))
    add("lm_head.weight", (vocab_size, hidden_dim))

    # Per-layer tensors
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        add(f"{prefix}.input_layernorm.weight", (hidden_dim,))
        add(f"{prefix}.self_attn.q_proj.weight", (hidden_dim, num_heads * head_dim))
        add(f"{prefix}.self_attn.k_proj.weight", (hidden_dim, num_kv_heads * head_dim))
        add(f"{prefix}.self_attn.v_proj.weight", (hidden_dim, num_kv_heads * head_dim))
        add(f"{prefix}.self_attn.o_proj.weight", (num_heads * head_dim, hidden_dim))
        add(f"{prefix}.post_attention_layernorm.weight", (hidden_dim,))
        add(f"{prefix}.mlp.gate_proj.weight", (hidden_dim, intermediate_dim))
        add(f"{prefix}.mlp.up_proj.weight", (hidden_dim, intermediate_dim))
        add(f"{prefix}.mlp.down_proj.weight", (intermediate_dim, hidden_dim))

    meta = {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "model_type": "llama",
    }

    size = build_ninf(meta, tensors, output_path)
    print(f"Test model created: {output_path} ({size / 1024:.1f} KB)")
    return output_path
