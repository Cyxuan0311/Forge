"""
Forge End-to-End Inference Benchmarks.

Benchmarks full inference pipeline: text generation, streaming, batched prompts,
and throughput measurements (tokens/sec).

Usage:
    python3 benchmarks/bench_inference.py
    python3 benchmarks/bench_inference.py --device cpu --max-tokens 64
    python3 benchmarks/bench_inference.py --tinyllama
"""
import os
import sys
import time
import argparse

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
TINYLLAMA_Q4_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")


def bench_generate(model, prompt_ids, max_new_tokens, do_sample=False, iters=5, warmup=2):
    """Benchmark full generate() call."""
    def run():
        model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=do_sample)

    for _ in range(warmup):
        run()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=do_sample)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean_ms": times.mean() * 1000,
        "p50_ms": np.percentile(times, 50) * 1000,
        "p99_ms": np.percentile(times, 99) * 1000,
        "tokens_per_sec": result["num_generated_tokens"] / times.mean() if times.mean() > 0 else 0,
        "num_tokens": result["num_generated_tokens"],
    }


def bench_generate_stream(model, prompt_ids, max_new_tokens, do_sample=False, iters=3, warmup=1):
    """Benchmark streaming generation."""
    def run():
        tokens = []
        def callback(token_id, logits):
            tokens.append(token_id)
            return True
        model.generate_stream(prompt_ids, callback, max_new_tokens=max_new_tokens, do_sample=do_sample)
        return len(tokens)

    for _ in range(warmup):
        run()

    times = []
    token_counts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        n_tokens = run()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        token_counts.append(n_tokens)

    times = np.array(times)
    avg_tokens = np.mean(token_counts)
    return {
        "mean_ms": times.mean() * 1000,
        "p50_ms": np.percentile(times, 50) * 1000,
        "tokens_per_sec": avg_tokens / times.mean() if times.mean() > 0 else 0,
        "num_tokens": int(avg_tokens),
    }


def bench_prompt_processing(ctx, prompt_len, iters=10, warmup=3):
    """Benchmark prompt (prefill) processing speed."""
    prompt_ids = np.arange(prompt_len, dtype=np.int32) % 32000

    def run():
        ctx.reset_kv()
        ctx.forward(prompt_ids)

    for _ in range(warmup):
        run()

    times = []
    for _ in range(iters):
        ctx.reset_kv()
        t0 = time.perf_counter()
        ctx.forward(prompt_ids)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "prompt_len": prompt_len,
        "mean_ms": times.mean() * 1000,
        "tokens_per_sec": prompt_len / times.mean() if times.mean() > 0 else 0,
    }


def bench_decode_speed(ctx, prompt_len, gen_len, iters=5, warmup=2):
    """Benchmark token-by-token decode speed (tokens/sec)."""
    prompt_ids = np.arange(prompt_len, dtype=np.int32) % 32000

    def run():
        ctx.reset_kv()
        ctx.forward(prompt_ids)
        pos = prompt_len
        for _ in range(gen_len):
            next_id = np.array([1], dtype=np.int32)
            ctx.forward(next_id, start_pos=pos)
            pos += 1

    for _ in range(warmup):
        run()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    # Only count decode tokens (not prompt)
    return {
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "mean_ms": times.mean() * 1000,
        "decode_tokens_per_sec": gen_len / times.mean() if times.mean() > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Forge Inference Benchmarks")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument("--iters", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    args = parser.parse_args()

    if not os.path.exists(TINYLLAMA_Q4_PATH):
        print("TinyLlama GGUF model not found at:", TINYLLAMA_Q4_PATH)
        print("Please download it first.")
        return

    print("=" * 70)
    print("  Forge End-to-End Inference Benchmarks")
    print(f"  Model: TinyLlama-1.1B Q4_0  |  Device: {args.device}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = forge.Model()
    model.load_auto(TINYLLAMA_Q4_PATH, device=args.device)

    gpu_layers = 0 if args.device == "cpu" else -1
    ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=gpu_layers)

    # Load tokenizer
    tok = forge.Tokenizer()
    tok.load_from_gguf(TINYLLAMA_Q4_PATH)

    # --- Prompt Processing Speed ---
    print("\n--- Prompt Processing (Prefill) Speed ---")
    for prompt_len in [8, 32, 64, 128, 256]:
        stats = bench_prompt_processing(ctx, prompt_len, iters=args.iters, warmup=args.warmup)
        print(f"  prompt_len={prompt_len:4d}: {stats['mean_ms']:8.2f} ms  "
              f"({stats['tokens_per_sec']:8.1f} tokens/sec)")

    # --- Decode Speed ---
    print("\n--- Decode Speed (Token-by-Token Generation) ---")
    for gen_len in [8, 16, 32, 64]:
        stats = bench_decode_speed(ctx, prompt_len=8, gen_len=gen_len,
                                    iters=args.iters, warmup=args.warmup)
        print(f"  gen_len={gen_len:3d}: {stats['mean_ms']:8.2f} ms  "
              f"({stats['decode_tokens_per_sec']:8.1f} decode tokens/sec)")

    # --- Full Generate ---
    print("\n--- Full generate() Benchmark ---")
    prompts = [
        ("short", "Hello"),
        ("medium", "The meaning of life is"),
        ("long", "In the year 2050, artificial intelligence will"),
    ]
    for name, text in prompts:
        prompt_ids = np.array(tok.encode(text, add_bos=True), dtype=np.int32)
        stats = bench_generate(model, prompt_ids, args.max_tokens,
                               do_sample=False, iters=args.iters, warmup=args.warmup)
        print(f"  {name:10s}: {stats['mean_ms']:8.1f} ms  "
              f"({stats['tokens_per_sec']:6.1f} tok/s,  "
              f"{stats['num_tokens']} tokens generated)")

    # --- Streaming Generate ---
    print("\n--- Streaming generate_stream() Benchmark ---")
    prompt_ids = np.array(tok.encode("Hello, how are you?", add_bos=True), dtype=np.int32)
    stats = bench_generate_stream(model, prompt_ids, args.max_tokens,
                                   do_sample=False, iters=args.iters, warmup=args.warmup)
    print(f"  stream:    {stats['mean_ms']:8.1f} ms  "
          f"({stats['tokens_per_sec']:6.1f} tok/s,  "
          f"{stats['num_tokens']} tokens)")

    # --- Graph Mode Comparison ---
    print("\n--- Graph Mode vs Imperative (Decode Speed) ---")
    for mode_name, use_graph in [("imperative", False), ("graph", True)]:
        ctx.set_use_graph(use_graph)
        stats = bench_decode_speed(ctx, prompt_len=8, gen_len=32,
                                    iters=args.iters, warmup=args.warmup)
        print(f"  {mode_name:12s}: {stats['mean_ms']:8.2f} ms  "
              f"({stats['decode_tokens_per_sec']:8.1f} tok/s)")
    ctx.set_use_graph(False)

    # --- KV Cache Memory ---
    print("\n--- KV Cache Memory Usage ---")
    for seq_len in [32, 64, 128, 256]:
        ctx.reset_kv()
        ids = np.arange(seq_len, dtype=np.int32) % 32000
        ctx.forward(ids)
        mem = ctx.memory_stats()
        print(f"  seq_len={seq_len:4d}: {mem}")

    print("\n" + "=" * 70)
    print("  Inference benchmarks complete!")


if __name__ == "__main__":
    main()
