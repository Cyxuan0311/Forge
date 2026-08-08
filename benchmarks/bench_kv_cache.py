"""
Forge KV Cache Comprehensive Benchmarks (Phase 0: Baseline).

Measures key KV cache metrics without changing the Python API:
  - Prompt prefill speed (tokens/sec)
  - Decode speed (tokens/sec)
  - KV update latency
  - KV read/materialize latency
  - Attention latency (decode step)
  - Allocated bytes, active bytes, free slots
  - Prefix hit rate (via RequestScheduler)

Usage:
    python3 benchmarks/bench_kv_cache.py
    python3 benchmarks/bench_kv_cache.py --device cpu --dtype fp32
    python3 benchmarks/bench_kv_cache.py --device cuda --dtype fp32 f16 q8_0
    python3 benchmarks/bench_kv_cache.py --tinyllama
"""

import os
import sys
import time
import argparse
import json

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge
import numpy as np

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures"
)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
TINYLLAMA_Q4_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")

# ============================================================================
# Utility
# ============================================================================


def bench(fn, warmup=5, iters=50):
    """Run a benchmark function and return stats dict."""
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
        "mean_s": float(times.mean()),
        "std_s": float(times.std()),
        "p50_s": float(np.percentile(times, 50)),
        "p99_s": float(np.percentile(times, 99)),
        "min_s": float(times.min()),
        "iters": iters,
    }


def fmt_ms(seconds):
    return f"{seconds * 1000:.3f} ms"


def fmt_rate(count, seconds):
    if seconds <= 0:
        return "0.0"
    return f"{count / seconds:.1f}"


# ============================================================================
# Model Loading
# ============================================================================


def load_small_model(device="cpu"):
    """Load the small NINF test model (1 layer, tiny)."""
    model_path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
    if not os.path.exists(model_path):
        return None, None
    model = forge.Model()
    model.load(
        model_path,
        vocab_size=100,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        device=device,
    )
    return model, model_path


def load_tinyllama(device="cpu"):
    """Load TinyLlama GGUF model."""
    if not os.path.exists(TINYLLAMA_Q4_PATH):
        return None, None
    model = forge.Model()
    model.load_auto(TINYLLAMA_Q4_PATH, device=device)
    return model, TINYLLAMA_Q4_PATH


def create_context_for_model(model, kv_dtype="fp32", gpu_layers=0):
    return model.create_context(kv_cache_dtype=kv_dtype, gpu_layers=gpu_layers)


# ============================================================================
# Benchmarks: Prefill / Decode / KV Update
# ============================================================================


def _print_mem(stats):
    """Print memory_stats dict in compact form."""
    items = []
    for k, v in stats.items():
        if isinstance(v, int) and v > 1024 * 1024:
            items.append(f"{k}={v / (1024 * 1024):.1f}MB")
        elif isinstance(v, int) and v > 1024:
            items.append(f"{k}={v / 1024:.1f}KB")
        else:
            items.append(f"{k}={v}")
    return ", ".join(items)


def bench_prefill(ctx, seq_len, warmup=3, iters=10):
    """Benchmark prompt prefill speed."""
    ids = np.arange(seq_len, dtype=np.int32) % 32000

    def run():
        ctx.reset_kv()
        ctx.forward(ids)

    stats = bench(run, warmup, iters)
    stats["seq_len"] = seq_len
    stats["tokens_per_sec"] = seq_len / stats["mean_s"] if stats["mean_s"] > 0 else 0
    return stats


def bench_decode_step(ctx, prompt_len, gen_len, warmup=2, iters=5):
    """Benchmark per-step decode latency (token-by-token)."""
    prompt_ids = np.arange(prompt_len, dtype=np.int32) % 32000

    def run():
        ctx.reset_kv()
        ctx.forward(prompt_ids)
        pos = prompt_len
        for _ in range(gen_len):
            next_id = np.array([1], dtype=np.int32)
            ctx.forward(next_id, start_pos=pos)
            pos += 1

    stats = bench(run, warmup, iters)
    stats["gen_len"] = gen_len
    stats["prompt_len"] = prompt_len
    # Average per-decode-step time
    stats["decode_step_mean_s"] = stats["mean_s"] / gen_len if gen_len > 0 else 0
    stats["decode_tokens_per_sec"] = gen_len / stats["mean_s"] if stats["mean_s"] > 0 else 0
    return stats


def bench_kv_update_overhead(ctx, prompt_len, warmup=3, iters=10):
    """Estimate KV update overhead by measuring forward w/ and w/o reset.

    The difference between forward with reset_kv (cold cache) and forward
    without reset (warm cache, metadata only) gives KV update overhead.
    """
    ids = np.arange(prompt_len, dtype=np.int32) % 32000

    # Cold: full KV write
    def run_cold():
        ctx.reset_kv()
        ctx.forward(ids)

    cold = bench(run_cold, warmup, iters)

    # Warm: cache already filled, this is KV read + minimal update
    ctx.reset_kv()
    ctx.forward(ids)  # fill once

    def run_warm():
        ctx.forward(ids)

    warm = bench(run_warm, warmup, iters)

    return {
        "cold_mean_s": cold["mean_s"],
        "warm_mean_s": warm["mean_s"],
        "kv_update_overhead_s": max(0, cold["mean_s"] - warm["mean_s"]),
        "prompt_len": prompt_len,
    }


def bench_kv_read(ctx, prompt_len, warmup=3, iters=20):
    """Benchmark memory_stats() latency as proxy for KV read overhead."""
    ids = np.arange(prompt_len, dtype=np.int32) % 32000

    def run():
        ctx.reset_kv()
        ctx.forward(ids)
        ctx.memory_stats()

    stats = bench(run, warmup, iters)
    stats["prompt_len"] = prompt_len
    return stats


# ============================================================================
# Benchmark: Memory Stats Over Sequence Length
# ============================================================================


def bench_memory_across_lengths(ctx, seq_lengths, dtype_str):
    """Collect memory stats at various sequence lengths."""
    results = []
    for sl in seq_lengths:
        ctx.reset_kv()
        ids = np.arange(sl, dtype=np.int32) % 32000
        ctx.forward(ids)
        mem = ctx.memory_stats()
        results.append(
            {
                "seq_len": sl,
                "kv_cache_nbytes": mem.get("kv_cache_nbytes", 0),
                "kv_cache_active_bytes": mem.get("kv_cache_active_bytes", 0),
                "kv_cache_free_slots": mem.get("kv_cache_free_slots", 0),
                "kv_cache_filled": mem.get("kv_cache_filled", 0),
                "kv_cache_dtype": mem.get("kv_cache_dtype", dtype_str),
                "kv_cache_type_k": mem.get("kv_cache_type_k", dtype_str),
                "kv_cache_type_v": mem.get("kv_cache_type_v", dtype_str),
            }
        )
    return results


# ============================================================================
# Benchmark: Multi-DType (TinyLlama only)
# ============================================================================


def bench_all_dtypes(model, device, gpu_layers, dtypes, prompt_len=32, gen_len=16):
    """Run prefill + decode benchmarks across multiple KV cache dtypes."""
    results = []
    for dt in dtypes:
        try:
            ctx = model.create_context(kv_cache_dtype=dt, gpu_layers=gpu_layers)
        except Exception as e:
            print(f"    [SKIP] dtype={dt}: {e}")
            results.append({"dtype": dt, "error": str(e)})
            continue

        prefill = bench_prefill(ctx, prompt_len, warmup=2, iters=5)
        decode = bench_decode_step(ctx, prompt_len, gen_len, warmup=1, iters=3)
        ctx.reset_kv()
        ids = np.arange(prompt_len, dtype=np.int32) % 32000
        ctx.forward(ids)
        mem = ctx.memory_stats()

        results.append(
            {
                "dtype": dt,
                "prefill_tokens_per_sec": prefill["tokens_per_sec"],
                "prefill_mean_ms": prefill["mean_s"] * 1000,
                "decode_tokens_per_sec": decode["decode_tokens_per_sec"],
                "decode_step_mean_ms": decode["decode_step_mean_s"] * 1000,
                "kv_cache_nbytes": mem.get("kv_cache_nbytes", 0),
                "kv_cache_active_bytes": mem.get("kv_cache_active_bytes", 0),
                "kv_cache_free_slots": mem.get("kv_cache_free_slots", 0),
                "kv_cache_filled": mem.get("kv_cache_filled", 0),
            }
        )
    return results


# ============================================================================
# Prefix Cache Benchmarks (via RequestScheduler)
# ============================================================================


def bench_prefix_cache(model, device, prompt_tokens, prefix_len, max_new_tokens=8):
    """Benchmark prefix cache using RequestScheduler.

    Submits two identical requests sequentially. The second should benefit from
    prefix caching (via seq_cp zero-copy sharing).
    """
    scheduler = forge.RequestScheduler(model, block_size=16, max_num_seqs=1)

    # Submit first request (populates cache)
    scheduler.submit(prompt_tokens, max_new_tokens=max_new_tokens)
    while scheduler.has_pending() or scheduler.num_active() > 0:
        scheduler.step()
    finished = scheduler.get_finished()
    req1 = finished[0] if finished else None
    prefix_hits_before = scheduler.prefix_cache_hits()

    # Submit second request (identical prompt — should hit prefix cache)
    scheduler.submit(prompt_tokens, max_new_tokens=max_new_tokens)
    while scheduler.has_pending() or scheduler.num_active() > 0:
        scheduler.step()
    finished2 = scheduler.get_finished()
    req2 = finished2[0] if finished2 else None
    prefix_hits_after = scheduler.prefix_cache_hits()

    return {
        "prefix_hits_before": prefix_hits_before,
        "prefix_hits_after": prefix_hits_after,
        "prefix_hit_delta": prefix_hits_after - prefix_hits_before,
        "prompt_len": len(prompt_tokens),
        "prefix_len": prefix_len,
        "req1_from_cache": req1.from_cache if req1 else None,
        "req2_from_cache": req2.from_cache if req2 else None,
    }


# ============================================================================
# Print Helpers
# ============================================================================


def print_separator(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    print(f"\n--- {title} ---")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Forge KV Cache Benchmarks (Phase 0)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="+",
        default=["fp32"],
        help="KV cache dtype(s) to benchmark (fp32, f16, q8_0, q4_0, q4_k)",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--iters", type=int, default=10, help="Default benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Default warmup iterations")
    parser.add_argument("--tinyllama", action="store_true", help="Include TinyLlama benchmarks")
    parser.add_argument("--json", type=str, default="", help="Save results to JSON file")
    args = parser.parse_args()

    all_results = {
        "device": args.device,
        "dtype": args.dtype,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print_separator("Forge KV Cache Benchmarks — Phase 0: Baseline")
    print(f"  Device: {args.device}  |  DType(s): {args.dtype}")
    print(f"  Iters: {args.iters}  |  Warmup: {args.warmup}")

    # ================================================================
    # Small model benchmarks
    # ================================================================
    print_separator("Small Test Model (1 layer, NINF)")
    model, model_path = load_small_model(args.device)
    if model is None:
        print("  Small test model not found, skipping small model tests.")
    else:
        gpu_layers = -1 if args.device != "cpu" else 0
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=gpu_layers)

        # --- Prefill speed ---
        print_subsection("Prompt Prefill Speed")
        prefill_results = []
        for sl in args.seq_lengths:
            stats = bench_prefill(ctx, sl, args.warmup, min(5, args.iters))
            prefill_results.append(stats)
            print(
                f"  seq_len={sl:4d}: {fmt_ms(stats['mean_s']):>10s}  "
                f"({fmt_rate(sl, stats['mean_s']):>8s} tok/s)"
            )

        # --- Decode speed ---
        print_subsection("Decode Speed (token-by-token)")
        decode_results = []
        for gl in [4, 8, 16]:
            stats = bench_decode_step(
                ctx, prompt_len=4, gen_len=gl, warmup=2, iters=min(3, args.iters)
            )
            decode_results.append(stats)
            print(
                f"  gen_len={gl:3d}: total={fmt_ms(stats['mean_s']):>10s}  "
                f"per_step={fmt_ms(stats['decode_step_mean_s']):>10s}  "
                f"({fmt_rate(gl, stats['mean_s']):>8s} tok/s)"
            )

        # --- KV update overhead ---
        print_subsection("KV Update Overhead")
        for sl in [8, 16, 32]:
            stats = bench_kv_update_overhead(ctx, sl, args.warmup, min(5, args.iters))
            print(
                f"  seq_len={sl:4d}: cold={fmt_ms(stats['cold_mean_s']):>10s}  "
                f"warm={fmt_ms(stats['warm_mean_s']):>10s}  "
                f"kv_overhead={fmt_ms(stats['kv_update_overhead_s']):>10s}"
            )

        # --- KV read (memory_stats) ---
        print_subsection("KV Read / Memory Stats")
        for sl in [8, 32, 64]:
            stats = bench_kv_read(ctx, sl, 3, min(10, args.iters))
            print(f"  seq_len={sl:4d}: forward+stats={fmt_ms(stats['mean_s']):>10s}")

        # --- Memory across lengths ---
        print_subsection("KV Cache Memory Usage")
        mem_results = bench_memory_across_lengths(ctx, args.seq_lengths, "fp32")
        for r in mem_results:
            print(
                f"  seq_len={r['seq_len']:4d}: allocated={r['kv_cache_nbytes'] / 1024:.1f}KB  "
                f"active={r['kv_cache_active_bytes'] / 1024:.1f}KB  "
                f"free_slots={r['kv_cache_free_slots']}  filled={r['kv_cache_filled']}"
            )

        all_results["small_model"] = {
            "model_path": model_path,
            "prefill": prefill_results,
            "decode": decode_results,
            "memory": mem_results,
        }

    # ================================================================
    # TinyLlama benchmarks
    # ================================================================
    if args.tinyllama:
        print_separator("TinyLlama-1.1B Benchmarks")
        model_tl, tl_path = load_tinyllama(args.device)
        if model_tl is None:
            print("  TinyLlama model not found, skipping.")
        else:
            gpu_layers = -1 if args.device != "cpu" else 0

            # --- Multi-DType ---
            print_subsection("Multi-DType Comparison")
            dtype_results = bench_all_dtypes(
                model_tl, args.device, gpu_layers, args.dtype, prompt_len=32, gen_len=16
            )
            for r in dtype_results:
                if "error" in r:
                    print(f"  dtype={r['dtype']:6s}: ERROR - {r['error']}")
                else:
                    print(
                        f"  dtype={r['dtype']:6s}: "
                        f"prefill={r['prefill_tokens_per_sec']:8.1f} tok/s  "
                        f"decode={r['decode_tokens_per_sec']:8.1f} tok/s  "
                        f"step={r['decode_step_mean_ms']:6.1f} ms  "
                        f"active={r['kv_cache_active_bytes'] / 1024:.1f}KB"
                    )

            # --- Prefill across lengths (fp32) ---
            ctx_tl = model_tl.create_context(kv_cache_dtype="fp32", gpu_layers=gpu_layers)
            print_subsection("Prompt Prefill Speed (fp32)")
            tl_prefill = []
            for sl in [8, 32, 64, 128]:
                stats = bench_prefill(ctx_tl, sl, warmup=2, iters=min(5, args.iters))
                tl_prefill.append(stats)
                print(
                    f"  seq_len={sl:4d}: {fmt_ms(stats['mean_s']):>10s}  "
                    f"({fmt_rate(sl, stats['mean_s']):>8s} tok/s)"
                )

            # --- Decode speed (fp32) ---
            print_subsection("Decode Speed (fp32)")
            tl_decode = []
            for gl in [8, 16, 32]:
                stats = bench_decode_step(ctx_tl, prompt_len=8, gen_len=gl, warmup=1, iters=3)
                tl_decode.append(stats)
                print(
                    f"  gen_len={gl:3d}: step={fmt_ms(stats['decode_step_mean_s']):>10s}  "
                    f"({fmt_rate(gl, stats['mean_s']):>8s} tok/s)"
                )

            # --- Memory across lengths ---
            print_subsection("KV Cache Memory Usage Across Lengths")
            tl_mem = bench_memory_across_lengths(ctx_tl, [32, 64, 128, 256], "fp32")
            for r in tl_mem:
                alloc_mb = r["kv_cache_nbytes"] / (1024 * 1024)
                active_kb = r["kv_cache_active_bytes"] / 1024
                print(
                    f"  seq_len={r['seq_len']:4d}: allocated={alloc_mb:.1f}MB  "
                    f"active={active_kb:.1f}KB  free_slots={r['kv_cache_free_slots']}"
                )

            # --- Prefix cache ---
            print_subsection("Prefix Cache Behavior")
            tok = forge.Tokenizer()
            tok.load_from_gguf(tl_path)
            prompt_text = "The capital of France is"
            prompt_ids = tok.encode(prompt_text, add_bos=True)
            pc = bench_prefix_cache(
                model_tl, args.device, prompt_ids, len(prompt_ids), max_new_tokens=4
            )
            print(f"  prompt='{prompt_text}' ({len(prompt_ids)} tokens)")
            print(
                f"  prefix_hits: before={pc['prefix_hits_before']} -> after={pc['prefix_hits_after']} "
                f"(delta={pc['prefix_hit_delta']})"
            )
            print(
                f"  req1 from_cache={pc['req1_from_cache']}  req2 from_cache={pc['req2_from_cache']}"
            )

            all_results["tinyllama"] = {
                "model_path": tl_path,
                "dtype_comparison": dtype_results,
                "prefill_fp32": tl_prefill,
                "decode_fp32": tl_decode,
                "memory": tl_mem,
                "prefix_cache": pc,
            }

    # ================================================================
    # Summary
    # ================================================================
    print_separator("Phase 0 Baseline Complete")

    if args.json:
        with open(args.json, "w") as f:
            # Convert numpy values
            clean = json.loads(json.dumps(all_results, default=str))
            json.dump(clean, f, indent=2)
        print(f"  Results saved to: {args.json}")


if __name__ == "__main__":
    main()
