"""
NanoInfer Core Micro-Benchmarks.

Benchmarks low-level operations: forward pass, tensor ops, model loading,
operator performance, and graph mode overhead.

Usage:
    python3 benchmarks/bench_core.py
    python3 benchmarks/bench_core.py --device cpu
    python3 benchmarks/bench_core.py --seq-lengths 1 8 32 128
"""
import os
import sys
import time
import argparse

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import nanoinfer
import numpy as np

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
TINYLLAMA_Q4_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")


# ============================================================================
# Utility
# ============================================================================

def fmt_time(seconds):
    """Format time with appropriate unit."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def bench(fn, warmup=5, iters=50):
    """Run a benchmark function multiple times and return stats."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean": times.mean(),
        "std": times.std(),
        "min": times.min(),
        "p50": np.percentile(times, 50),
        "p99": np.percentile(times, 99),
        "iters": iters,
    }


def print_result(name, stats, unit="ms", extra=""):
    scale = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1}[unit]
    mean = stats["mean"] * scale
    p50 = stats["p50"] * scale
    p99 = stats["p99"] * scale
    std = stats["std"] * scale
    print(f"  {name:40s}  mean={mean:8.2f} {unit}  p50={p50:8.2f}  p99={p99:8.2f}  std={std:6.2f}  {extra}")


# ============================================================================
# Model & Context Setup
# ============================================================================

def get_small_model_and_context(device="cpu"):
    """Load the small test model."""
    model_path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
    if not os.path.exists(model_path):
        return None, None
    model = nanoinfer.Model()
    model.load(model_path, vocab_size=100, hidden_dim=32, intermediate_dim=64,
               num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16, device=device)
    ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1 if device != "cpu" else 0)
    return model, ctx


def get_tinyllama_model_and_context(device="cpu", gpu_layers=0):
    """Load TinyLlama GGUF model."""
    if not os.path.exists(TINYLLAMA_Q4_PATH):
        return None, None
    model = nanoinfer.Model()
    model.load_auto(TINYLLAMA_Q4_PATH, device=device)
    ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=gpu_layers)
    return model, ctx


# ============================================================================
# Benchmark: Forward Pass
# ============================================================================

def bench_forward(ctx, seq_len, warmup=5, iters=50):
    ids = np.arange(seq_len, dtype=np.int32) % 100

    def run():
        ctx.reset_kv()
        ctx.forward(ids)

    return bench(run, warmup, iters)


def bench_forward_incremental(ctx, prompt_len, gen_len, warmup=3, iters=20):
    """Benchmark incremental (token-by-token) generation."""
    prompt_ids = np.arange(prompt_len, dtype=np.int32) % 100

    def run():
        ctx.reset_kv()
        ctx.forward(prompt_ids)
        pos = prompt_len
        for _ in range(gen_len):
            next_id = np.array([1], dtype=np.int32)
            ctx.forward(next_id, start_pos=pos)
            pos += 1

    return bench(run, warmup, iters)


# ============================================================================
# Benchmark: Tensor Operations
# ============================================================================

def bench_tensor_creation(shape, warmup=10, iters=200):
    def run():
        nanoinfer.Tensor(nanoinfer.DataType.FP32, shape, nanoinfer.DeviceType.CPU)
    return bench(run, warmup, iters)


def bench_tensor_copy(size, warmup=5, iters=50):
    src = nanoinfer.Tensor(nanoinfer.DataType.FP32, [size], nanoinfer.DeviceType.CPU)
    dst = nanoinfer.Tensor(nanoinfer.DataType.FP32, [size], nanoinfer.DeviceType.CPU)

    def run():
        dst.copy_from(src)
    return bench(run, warmup, iters)


# ============================================================================
# Benchmark: Model Loading
# ============================================================================

def bench_model_loading_ninf(warmup=2, iters=10):
    model_path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
    if not os.path.exists(model_path):
        return None

    def run():
        model = nanoinfer.Model()
        model.load(model_path, vocab_size=100, hidden_dim=32, intermediate_dim=64,
                   num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16, device="cpu")
    return bench(run, warmup, iters)


def bench_model_loading_gguf(warmup=1, iters=5):
    if not os.path.exists(TINYLLAMA_Q4_PATH):
        return None

    def run():
        model = nanoinfer.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
    return bench(run, warmup, iters)


# ============================================================================
# Benchmark: Graph Mode Overhead
# ============================================================================

def bench_graph_vs_imperative(ctx, seq_len, warmup=5, iters=50):
    ids = np.arange(seq_len, dtype=np.int32) % 100

    # Imperative
    ctx.set_use_graph(False)
    def run_imperative():
        ctx.reset_kv()
        ctx.forward(ids)
    stats_imp = bench(run_imperative, warmup, iters)

    # Graph
    ctx.set_use_graph(True)
    def run_graph():
        ctx.reset_kv()
        ctx.forward(ids)
    stats_graph = bench(run_graph, warmup, iters)

    ctx.set_use_graph(False)
    return stats_imp, stats_graph


# ============================================================================
# Benchmark: KV Cache
# ============================================================================

def bench_kv_cache_reset(ctx, warmup=10, iters=100):
    def run():
        ctx.reset_kv()
    return bench(run, warmup, iters)


def bench_kv_cache_memory_stats(ctx, warmup=10, iters=200):
    def run():
        ctx.memory_stats()
    return bench(run, warmup, iters)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NanoInfer Core Micro-Benchmarks")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for benchmarks")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[1, 4, 16, 32, 64],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--tinyllama", action="store_true", help="Include TinyLlama benchmarks")
    args = parser.parse_args()

    print("=" * 70)
    print("  NanoInfer Core Micro-Benchmarks")
    print(f"  Device: {args.device}  |  Iters: {args.iters}  |  Warmup: {args.warmup}")
    print("=" * 70)

    # --- Small model benchmarks ---
    model, ctx = get_small_model_and_context(args.device)
    if ctx is None:
        print("Small test model not found, skipping")
        return

    # Forward pass
    print("\n--- Forward Pass (small model, 1 layer) ---")
    for sl in args.seq_lengths:
        stats = bench_forward(ctx, sl, args.warmup, args.iters)
        print_result(f"forward(seq_len={sl})", stats, "ms")

    # Incremental generation
    print("\n--- Incremental Generation (small model) ---")
    for prompt_len in [4, 16]:
        for gen_len in [8, 32]:
            stats = bench_forward_incremental(ctx, prompt_len, gen_len, 3, 20)
            total = prompt_len + gen_len
            print_result(f"incremental(prompt={prompt_len}, gen={gen_len}, total={total})", stats, "ms")

    # Graph mode comparison
    print("\n--- Graph Mode vs Imperative (small model) ---")
    for sl in [4, 16, 32]:
        stats_imp, stats_graph = bench_graph_vs_imperative(ctx, sl, args.warmup, args.iters)
        speedup = stats_imp["mean"] / stats_graph["mean"] if stats_graph["mean"] > 0 else 0
        print_result(f"imperative(seq_len={sl})", stats_imp, "ms")
        print_result(f"graph(seq_len={sl})", stats_graph, "ms", f"speedup={speedup:.2f}x")

    # KV Cache operations
    print("\n--- KV Cache Operations ---")
    stats = bench_kv_cache_reset(ctx, 10, 100)
    print_result("reset_kv()", stats, "us")

    stats = bench_kv_cache_memory_stats(ctx, 10, 200)
    print_result("memory_stats()", stats, "us")

    # Tensor operations
    print("\n--- Tensor Operations ---")
    for shape in [[1], [10], [100], [32, 32], [128, 128]]:
        stats = bench_tensor_creation(shape, 10, 200)
        print_result(f"create(shape={shape})", stats, "us")

    for size in [100, 1024, 4096, 16384]:
        stats = bench_tensor_copy(size, 5, 50)
        print_result(f"copy(size={size})", stats, "us")

    # Model loading
    print("\n--- Model Loading ---")
    stats = bench_model_loading_ninf(2, 10)
    if stats:
        print_result("load_ninf(small)", stats, "ms")

    # --- TinyLlama benchmarks ---
    if args.tinyllama:
        print("\n" + "=" * 70)
        print("  TinyLlama-1.1B Benchmarks")
        print("=" * 70)

        gpu_layers = 0 if args.device == "cpu" else -1
        model_tl, ctx_tl = get_tinyllama_model_and_context(args.device, gpu_layers)
        if ctx_tl is None:
            print("TinyLlama model not found, skipping")
        else:
            print("\n--- Forward Pass (TinyLlama-1.1B Q4_0) ---")
            for sl in [1, 8, 32, 64, 128]:
                stats = bench_forward(ctx_tl, sl, 3, 10)
                print_result(f"forward(seq_len={sl})", stats, "ms")

            print("\n--- Incremental Generation (TinyLlama) ---")
            for gen_len in [8, 32, 64]:
                stats = bench_forward_incremental(ctx_tl, 4, gen_len, 2, 5)
                print_result(f"incremental(prompt=4, gen={gen_len})", stats, "ms")

            print("\n--- GGUF Model Loading (TinyLlama) ---")
            stats = bench_model_loading_gguf(1, 3)
            if stats:
                print_result("load_gguf(tinyllama)", stats, "ms")

    print("\n" + "=" * 70)
    print("  Benchmarks complete!")


if __name__ == "__main__":
    main()
