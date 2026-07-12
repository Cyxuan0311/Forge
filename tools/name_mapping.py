"""
GGUF → Forge canonical tensor name mapping.

Supports LLaMA-family models (blk.N.xxx naming).
Extend SUFFIX_MAP for new architectures.
"""

# Global name remapping (no layer index)
GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

# Layer suffix remapping for blk.N.suffix names
SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
}


def map_gguf_name(gguf_name):
    """Convert a GGUF tensor name to Forge canonical form."""
    if gguf_name in GLOBAL_MAP:
        return GLOBAL_MAP[gguf_name]

    if gguf_name.startswith("blk."):
        rest = gguf_name[4:]
        parts = rest.split(".", 1)
        if len(parts) == 2:
            layer_idx = parts[0]
            suffix = parts[1]
            mapped_suffix = SUFFIX_MAP.get(suffix, suffix)
            return f"model.layers.{layer_idx}.{mapped_suffix}"

    return gguf_name
