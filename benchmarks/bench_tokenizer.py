"""
NanoInfer Tokenizer Performance Benchmarks.

Benchmarks tokenizer encode/decode speed, special token handling,
and multi-language tokenization performance.

Usage:
    python3 benchmarks/bench_tokenizer.py
"""
import os
import sys
import time

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import nanoinfer
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
TINYLLAMA_Q4_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")


def bench_encode(tok, text, iters=100, warmup=10):
    for _ in range(warmup):
        tok.encode(text, add_bos=False)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        tok.encode(text, add_bos=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    ids = tok.encode(text, add_bos=False)
    return {
        "mean_us": times.mean() * 1e6,
        "p50_us": np.percentile(times, 50) * 1e6,
        "p99_us": np.percentile(times, 99) * 1e6,
        "num_tokens": len(ids),
        "chars_per_token": len(text) / len(ids) if len(ids) > 0 else 0,
    }


def bench_decode(tok, ids, iters=100, warmup=10):
    for _ in range(warmup):
        tok.decode(ids, skip_special=True)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        tok.decode(ids, skip_special=True)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean_us": times.mean() * 1e6,
        "p50_us": np.percentile(times, 50) * 1e6,
        "p99_us": np.percentile(times, 99) * 1e6,
        "num_tokens": len(ids),
    }


def bench_token_to_id(tok, token_str, iters=1000, warmup=100):
    for _ in range(warmup):
        tok.token_to_id(token_str)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        tok.token_to_id(token_str)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {"mean_ns": times.mean() * 1e9, "p50_ns": np.percentile(times, 50) * 1e9}


def bench_id_to_token(tok, token_id, iters=1000, warmup=100):
    for _ in range(warmup):
        tok.id_to_token(token_id)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        tok.id_to_token(token_id)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {"mean_ns": times.mean() * 1e9, "p50_ns": np.percentile(times, 50) * 1e9}


def main():
    if not os.path.exists(TINYLLAMA_Q4_PATH):
        print("TinyLlama GGUF model not found at:", TINYLLAMA_Q4_PATH)
        return

    print("=" * 70)
    print("  NanoInfer Tokenizer Performance Benchmarks")
    print("  Model: TinyLlama-1.1B (SPM Tokenizer)")
    print("=" * 70)

    tok = nanoinfer.Tokenizer()
    tok.load_from_gguf(TINYLLAMA_Q4_PATH)

    # --- Encode Speed ---
    print("\n--- Encode Speed ---")
    texts = [
        ("1 word", "Hello"),
        ("1 sentence", "The quick brown fox jumps over the lazy dog."),
        ("2 sentences", "Hello world. This is a test of the tokenizer."),
        ("1 paragraph", "Artificial intelligence is the simulation of human intelligence "
                        "by machines. It includes learning, reasoning, and self-correction. "
                        "AI has become an essential part of the technology industry."),
        ("Chinese", "人工智能是计算机科学的一个分支，它企图了解智能的实质。"),
        ("Japanese", "人工知能は、人間の知能を模倣するコンピュータシステムです。"),
        ("Mixed", "Hello 你好 こんにちは! This is a mixed-language text."),
        ("Code", "def foo(x: int) -> int:\n    return x * 2 + 1"),
        ("JSON", '{"name": "test", "value": 42, "items": [1, 2, 3]}'),
        ("Long (500 chars)", "The " * 125),
    ]

    for name, text in texts:
        stats = bench_encode(tok, text)
        print(f"  {name:20s}: {stats['mean_us']:8.1f} us  "
              f"({stats['num_tokens']:4d} tokens,  "
              f"{stats['chars_per_token']:.1f} chars/tok)")

    # --- Decode Speed ---
    print("\n--- Decode Speed ---")
    for name, text in texts:
        ids = tok.encode(text, add_bos=False)
        if len(ids) == 0:
            continue
        stats = bench_decode(tok, ids)
        print(f"  {name:20s}: {stats['mean_us']:8.1f} us  "
              f"({stats['num_tokens']:4d} tokens)")

    # --- Token Lookup Speed ---
    print("\n--- Token Lookup Speed ---")
    for token_str in ["<s>", "</s>", "the", "a", "hello", "▁Hello"]:
        stats = bench_token_to_id(tok, token_str)
        print(f"  token_to_id('{token_str:12s}'): {stats['mean_ns']:8.1f} ns")

    for token_id in [0, 1, 2, 100, 1000, 31999]:
        stats = bench_id_to_token(tok, token_id)
        print(f"  id_to_token({token_id:6d}):       {stats['mean_ns']:8.1f} ns")

    # --- Encode with Options ---
    print("\n--- Encode Options Overhead ---")
    text = "Hello world, this is a test."
    base = bench_encode(tok, text)

    ids_bos = tok.encode(text, add_bos=True)
    t0 = time.perf_counter()
    for _ in range(100):
        tok.encode(text, add_bos=True)
    t1 = time.perf_counter()
    bos_us = (t1 - t0) / 100 * 1e6

    ids_eos = tok.encode(text, add_bos=False, add_eos=True)
    t0 = time.perf_counter()
    for _ in range(100):
        tok.encode(text, add_bos=False, add_eos=True)
    t1 = time.perf_counter()
    eos_us = (t1 - t0) / 100 * 1e6

    print(f"  base (no bos/eos): {base['mean_us']:8.1f} us  ({base['num_tokens']} tokens)")
    print(f"  with bos:          {bos_us:8.1f} us  ({len(ids_bos)} tokens)")
    print(f"  with eos:          {eos_us:8.1f} us  ({len(ids_eos)} tokens)")

    # --- Throughput ---
    print("\n--- Throughput (tokens/sec) ---")
    long_text = "The quick brown fox jumps over the lazy dog. " * 100
    stats = bench_encode(tok, long_text, iters=20, warmup=5)
    encode_tps = stats["num_tokens"] / (stats["mean_us"] * 1e-6) if stats["mean_us"] > 0 else 0
    print(f"  encode: {encode_tps:,.0f} tokens/sec  ({stats['num_tokens']} tokens in {stats['mean_us']:.1f} us)")

    ids_long = tok.encode(long_text, add_bos=False)
    stats = bench_decode(tok, ids_long, iters=20, warmup=5)
    decode_tps = stats["num_tokens"] / (stats["mean_us"] * 1e-6) if stats["mean_us"] > 0 else 0
    print(f"  decode: {decode_tps:,.0f} tokens/sec  ({stats['num_tokens']} tokens in {stats['mean_us']:.1f} us)")

    print("\n" + "=" * 70)
    print("  Tokenizer benchmarks complete!")


if __name__ == "__main__":
    main()
