"""Shared utilities for Forge inference scripts.

Provides common infrastructure for interactive chat, streaming generation,
profiling, and model loading, so individual inference scripts only need
to specialize:
  - apply_chat_template()  — model-specific chat format
  - default arguments      — model path, gpu_layers, temperature, etc.
  - extra CLI flags        — model-specific options (e.g. --think)
"""

import sys
import os
import time
import gc
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup — ensure the build directory is importable
# ---------------------------------------------------------------------------
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge
import numpy as np


# ===========================================================================
# Performance Profiling
# ===========================================================================


class PerfTimer:
    """Accumulative performance timer for profiling inference stages."""

    def __init__(self):
        self._timings = defaultdict(
            lambda: {"total_ms": 0.0, "count": 0, "min_ms": float("inf"), "max_ms": 0.0}
        )
        self._starts = {}

    def start(self, name):
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        if name not in self._starts:
            return
        elapsed_ms = (time.perf_counter() - self._starts[name]) * 1000.0
        rec = self._timings[name]
        rec["total_ms"] += elapsed_ms
        rec["count"] += 1
        rec["min_ms"] = min(rec["min_ms"], elapsed_ms)
        rec["max_ms"] = max(rec["max_ms"], elapsed_ms)
        del self._starts[name]

    def record(self, name, ms):
        rec = self._timings[name]
        rec["total_ms"] += ms
        rec["count"] += 1
        rec["min_ms"] = min(rec["min_ms"], ms)
        rec["max_ms"] = max(rec["max_ms"], ms)

    def reset(self):
        self._timings.clear()
        self._starts.clear()

    def print_summary(self):
        if not self._timings:
            return
        print("\n" + "=" * 90)
        print("  Python-Level Performance Profile")
        print("=" * 90)
        print(
            f"{'Stage':<40} {'Count':>6} {'Total(ms)':>10} {'Avg(ms)':>10} "
            f"{'Min(ms)':>10} {'Max(ms)':>10} {'%Total':>7}"
        )
        print("-" * 90)

        grand_total = sum(r["total_ms"] for r in self._timings.values())
        sorted_items = sorted(self._timings.items(), key=lambda x: x[1]["total_ms"], reverse=True)

        for name, rec in sorted_items:
            avg = rec["total_ms"] / rec["count"] if rec["count"] > 0 else 0
            pct = (rec["total_ms"] / grand_total * 100) if grand_total > 0 else 0
            print(
                f"{name:<40} {rec['count']:>6} {rec['total_ms']:>10.2f} "
                f"{avg:>10.3f} {rec['min_ms']:>10.3f} {rec['max_ms']:>10.3f} {pct:>6.1f}%"
            )

        print("-" * 90)
        print(
            f"{'TOTAL':<40} {'':>6} {grand_total:>10.2f} {'':>10} "
            f"{'':>10} {'':>10} {'100.0%':>7}"
        )
        print("=" * 90)


# Global profiler instance
perf = PerfTimer()
profiling_enabled = False


def print_cpp_profiler_summary():
    """Print the C++ PerfProfiler summary (operator-level timing)."""
    try:
        summary = forge.profiler_summary()
        if not summary:
            return
        print("\n" + "=" * 90)
        print("  C++ Operator-Level Performance Profile (from PerfProfiler)")
        print("=" * 90)
        print(
            f"{'Operation':<45} {'Count':>6} {'Total(ms)':>10} {'Avg(ms)':>10} "
            f"{'Min(ms)':>10} {'Max(ms)':>10} {'%Total':>7}"
        )
        print("-" * 90)

        sorted_ops = sorted(summary.items(), key=lambda x: x[1]["total_ms"], reverse=True)
        grand_total = sum(v["total_ms"] for v in summary.values())

        for name, rec in sorted_ops:
            pct = (rec["total_ms"] / grand_total * 100) if grand_total > 0 else 0
            print(
                f"{name:<45} {rec['count']:>6} {rec['total_ms']:>10.2f} "
                f"{rec['avg_ms']:>10.3f} {rec['min_ms']:>10.3f} {rec['max_ms']:>10.3f} "
                f"{pct:>6.1f}%"
            )

        print("-" * 90)
        print(
            f"{'TOTAL':<45} {'':>6} {grand_total:>10.2f} {'':>10} "
            f"{'':>10} {'':>10} {'100.0%':>7}"
        )
        print("=" * 90)
    except Exception as e:
        print(f"[Profiler] Could not retrieve C++ profile: {e}")


def print_full_profile():
    """Print both Python-level and C++ operator-level profiles."""
    perf.print_summary()
    print_cpp_profiler_summary()


# ===========================================================================
# Tokenizer
# ===========================================================================


def load_tokenizer(model_path):
    """Load tokenizer from GGUF file using Forge's built-in Tokenizer."""
    tok = forge.Tokenizer()
    tok.load_from_gguf(model_path)
    return tok


def verify_tokenizer(tokenizer, tokenizer_dir):
    """Verify Forge tokenizer against transformers reference."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not installed, skipping verification")
        return

    ref_file = "tokenizer.json" if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")) else None
    if ref_file is None and not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.model")):
        print(f"Reference tokenizer not found at {tokenizer_dir}, skipping verification")
        return

    ref = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    test_texts = [
        "Hello world",
        "What is the capital of France?",
        "I love programming in Python!",
        "The quick brown fox jumps over the lazy dog.",
        "你好世界",
        "  multiple   spaces  ",
        "Line1\nLine2",
        "a",
        " a",
        "  a",
        "a ",
        "a  ",
        "a b",
        "a  b",
        "你好 世界",
        "Hello\nWorld",
        "tab\there",
        "\n\n",
        "\n",
        "test\r\n",
    ]

    print("\n" + "=" * 60)
    print("  Tokenizer Verification: Forge vs transformers")
    print("=" * 60)

    encode_pass = 0
    decode_pass = 0
    total = len(test_texts)

    for text in test_texts:
        our_ids = tokenizer.encode(text, add_bos=False)
        ref_ids = ref.encode(text, add_special_tokens=False)
        our_dec = tokenizer.decode(our_ids, skip_special=True)
        ref_dec = ref.decode(ref_ids, skip_special_tokens=True)

        encode_ok = our_ids == ref_ids
        decode_ok = our_dec == ref_dec

        if encode_ok:
            encode_pass += 1
        if decode_ok:
            decode_pass += 1

        status = "PASS" if (encode_ok and decode_ok) else "FAIL"
        print(f"[{status}] {text!r}")
        if not encode_ok:
            our_tokens = [tokenizer.id_to_token(i) for i in our_ids]
            ref_tokens = [ref.convert_ids_to_tokens(i) for i in ref_ids]
            print(f"  ours: {our_tokens}")
            print(f"  ref:  {ref_tokens}")
        if not decode_ok:
            print(f"  decode: ours={our_dec!r} ref={ref_dec!r}")

    print(f"\nEncode: {encode_pass}/{total} | Decode: {decode_pass}/{total}")
    if encode_pass == total and decode_pass == total:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)


# ===========================================================================
# Generation
# ===========================================================================


def generate_streaming(
    model,
    ctx,
    tokenizer,
    input_ids,
    max_new_tokens=256,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.1,
    eos_token_id=None,
    kv_cache_dtype="fp32",
    gpu_layers=-1,
    stop_token_ids=None,
):
    """Stream-generate tokens with buffered decode and profiling support.

    Args:
        model: Forge Model instance.
        ctx: Forge Context instance (not used directly, kept for API compat).
        tokenizer: Forge Tokenizer instance.
        input_ids: List of prompt token IDs.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-K sampling parameter.
        top_p: Top-P sampling parameter.
        repeat_penalty: Repetition penalty.
        eos_token_id: End-of-sequence token ID.
        kv_cache_dtype: KV cache data type.
        gpu_layers: Number of GPU layers.
        stop_token_ids: Additional stop token IDs (e.g., Gemma4 turn-end).

    Returns:
        (generated_tokens, elapsed_seconds)
    """
    global profiling_enabled

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    generated_tokens = []
    token_buffer = []
    decode_start = time.time()

    token_times = [] if profiling_enabled else None

    def on_token(token_id, step):
        generated_tokens.append(token_id)
        token_buffer.append(token_id)

        if profiling_enabled and step > 0:
            # timing is approximate since we can't measure per-token precisely in callback
            pass

        if len(token_buffer) >= 4 or token_id == eos_token_id:
            try:
                if profiling_enabled:
                    perf.start("decode/tokenize")
                text = tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
                if profiling_enabled:
                    perf.stop("decode/tokenize")
                print(text, end="", flush=True)
                token_buffer.clear()
            except UnicodeDecodeError:
                if token_id == eos_token_id:
                    text = tokenizer.decode(
                        token_buffer, skip_special=True, strip_leading_space=False
                    )
                    print(text, end="", flush=True)
                    token_buffer.clear()

    if profiling_enabled:
        perf.start("generate_stream/total")

    np_ids = np.array(input_ids, dtype=np.int32)
    gen_kwargs = dict(
        prompt_ids=np_ids,
        callback=on_token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        do_sample=temperature > 0,
        eos_token_id=eos_token_id,
        kv_cache_dtype=kv_cache_dtype,
        gpu_layers=gpu_layers,
    )
    if stop_token_ids:
        gen_kwargs["stop_token_ids"] = stop_token_ids
    model.generate_stream(**gen_kwargs)

    if profiling_enabled:
        perf.stop("generate_stream/total")

    if token_buffer:
        text = tokenizer.decode(
            token_buffer, skip_special=True, strip_leading_space=False
        )
        print(text, end="", flush=True)
        token_buffer.clear()

    elapsed = time.time() - decode_start

    if profiling_enabled and token_times and len(token_times) > 1:
        decode_times = token_times[1:]
        avg_ms = sum(decode_times) / len(decode_times) if decode_times else 0
        perf.record("decode/per_token_avg_ms", avg_ms)

    return generated_tokens, elapsed


def generate_batch(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=256,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.1,
    eos_token_id=None,
    kv_cache_dtype="fp32",
    gpu_layers=-1,
    stop_token_ids=None,
):
    """Batch-generate tokens (non-streaming).

    Returns:
        (num_generated, elapsed_seconds)
    """
    global profiling_enabled

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    if profiling_enabled:
        perf.start("generate/total")

    start_time = time.time()
    np_ids = np.array(input_ids, dtype=np.int32)
    gen_kwargs = dict(
        prompt_ids=np_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        do_sample=temperature > 0,
        eos_token_id=eos_token_id,
        kv_cache_dtype=kv_cache_dtype,
        gpu_layers=gpu_layers,
    )
    if stop_token_ids:
        gen_kwargs["stop_token_ids"] = stop_token_ids
    result = model.generate(**gen_kwargs)
    elapsed = time.time() - start_time

    if profiling_enabled:
        perf.stop("generate/total")

    num_generated = result["num_generated_tokens"]
    new_tokens = list(result["token_ids"])

    if profiling_enabled:
        perf.start("decode/tokenize")
    response = tokenizer.decode(new_tokens, skip_special=True, strip_leading_space=False)
    if profiling_enabled:
        perf.stop("decode/tokenize")
    print(response, end="", flush=True)

    return num_generated, elapsed


# ===========================================================================
# Interactive Chat
# ===========================================================================


def interactive_chat(
    model,
    tokenizer,
    args,
    apply_chat_template_fn,
    model_name="Forge",
    eos_token_id=None,
    stop_token_ids=None,
    system_msg=None,
    extra_commands=None,
):
    """Interactive multi-turn chat loop.

    Args:
        model: Forge Model instance.
        tokenizer: Forge Tokenizer instance.
        args: Parsed argparse namespace with standard fields.
        apply_chat_template_fn: Callable(tokenizer, messages, add_generation_prompt) -> list[int].
        model_name: Display name for the banner.
        eos_token_id: Override EOS token ID.
        stop_token_ids: Additional stop token IDs.
        system_msg: Optional system message dict, e.g. {"role": "system", "content": "..."}.
        extra_commands: Optional dict mapping command string to handler function.
    """
    global profiling_enabled

    conversation = [system_msg] if system_msg else []
    ctx = None

    print("\n" + "=" * 60)
    print(f"  {model_name} Interactive Chat (Forge)")
    print(f"  Device: {args.device}")
    if profiling_enabled:
        print("  Profiling: ON (Python + C++ PerfProfiler)")
    cmd_list = "/quit, /clear, /help, /profile"
    if extra_commands:
        cmd_list += ", " + ", ".join(f"/{k}" for k in extra_commands)
    print(f"  Commands: {cmd_list}")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            break
        elif user_input == "/clear":
            conversation = [system_msg] if system_msg else []
            perf.reset()
            print("[Conversation cleared]\n")
            continue
        elif user_input == "/help":
            print("  /quit    - Exit the chat")
            print("  /clear   - Clear conversation history")
            print("  /help    - Show this help message")
            print("  /profile - Toggle profiling on/off")
            if extra_commands:
                for cmd, handler in extra_commands.items():
                    doc = handler.__doc__ or f"/{cmd}"
                    print(f"  /{cmd} - {doc}")
            print()
            continue
        elif user_input == "/profile":
            profiling_enabled = not profiling_enabled
            if profiling_enabled:
                forge.profiler_enable()
                forge.profiler_reset()
                perf.reset()
            else:
                forge.profiler_disable()
            print(f"[Profiling {'enabled' if profiling_enabled else 'disabled'}]\n")
            continue
        elif extra_commands:
            handled = False
            for cmd, handler in extra_commands.items():
                prefix = f"/{cmd}"
                if user_input == prefix or user_input.startswith(prefix + " "):
                    handler(user_input)
                    handled = True
                    break
            if handled:
                continue

        conversation.append({"role": "user", "content": user_input})

        if profiling_enabled:
            perf.start("template/encode")
        input_ids = apply_chat_template_fn(tokenizer, conversation, add_generation_prompt=True)
        if profiling_enabled:
            perf.stop("template/encode")

        print("Assistant: ", end="", flush=True)

        if getattr(args, "no_stream", False):
            num_generated, elapsed = generate_batch(
                model, tokenizer, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                eos_token_id=eos_token_id,
                kv_cache_dtype=args.kv_cache_dtype,
                gpu_layers=args.gpu_layers,
                stop_token_ids=stop_token_ids,
            )
            assistant_text = tokenizer.decode(
                list(range(num_generated)), skip_special=True, strip_leading_space=False
            ) if num_generated == 0 else ""
        else:
            if ctx is not None:
                del ctx
                gc.collect()
            ctx = model.create_context(
                kv_cache_dtype=args.kv_cache_dtype,
                gpu_layers=args.gpu_layers,
            )
            generated_tokens, elapsed = generate_streaming(
                model, ctx, tokenizer, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                eos_token_id=eos_token_id,
                kv_cache_dtype=args.kv_cache_dtype,
                gpu_layers=args.gpu_layers,
                stop_token_ids=stop_token_ids,
            )
            num_generated = len(generated_tokens)
            assistant_text = tokenizer.decode(
                generated_tokens, skip_special=True, strip_leading_space=False
            )

        print()
        if num_generated > 0 and elapsed > 0:
            speed = num_generated / elapsed
            prompt_len = len(input_ids)
            print(f"[{num_generated} tokens, {elapsed:.2f}s, {speed:.1f} tok/s, prompt={prompt_len} tokens]")

        if profiling_enabled:
            print_full_profile()
            perf.reset()
            forge.profiler_reset()

        conversation.append({"role": "assistant", "content": assistant_text})
        print()


# ===========================================================================
# Model Loading & Startup
# ===========================================================================


def load_model_and_tokenize(args, model_path=None):
    """Common startup sequence: load tokenizer + model + create context + warmup.

    Args:
        args: Parsed argparse namespace.
        model_path: Override model path (if None, uses args.model_path).

    Returns:
        (model, tokenizer) tuple.
    """
    global profiling_enabled

    if model_path is None:
        model_path = args.model_path

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    # Tokenizer
    t0 = time.time()
    print("Loading tokenizer from GGUF...")
    tokenizer = load_tokenizer(model_path)
    t1 = time.time()
    print(
        f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
        f"model_type={tokenizer.model_type}, "
        f"bos_id={tokenizer.bos_token_id}, eos_id={tokenizer.eos_token_id} "
        f"[{t1 - t0:.2f}s]"
    )

    # Verify tokenizer if requested
    if getattr(args, "verify_tokenizer", False):
        tokenizer_dir = getattr(args, "tokenizer_dir", os.path.dirname(model_path))
        verify_tokenizer(tokenizer, tokenizer_dir)
        return None, tokenizer

    # Model
    is_gguf = model_path.endswith(".gguf")
    print(f"Loading model ({'GGUF' if is_gguf else 'NINF'}) on {args.device}...")
    forge.Logger.set_level(2 if getattr(args, "verbose", False) else 1)

    if profiling_enabled:
        perf.start("startup.model_load")
    model = forge.Model()
    if is_gguf:
        model.load_gguf(model_path, device=args.device)
    else:
        # NINF loading requires config — caller should handle this
        raise ValueError("NINF loading requires model config; use load_gguf instead")
    if profiling_enabled:
        perf.stop("startup.model_load")

    cfg = model.config
    print(
        f"Model loaded! arch={cfg.arch_type}, layers={cfg.num_layers}, "
        f"hidden={cfg.hidden_dim}, heads={cfg.num_heads}, "
        f"kv_heads={cfg.num_kv_heads}, vocab={cfg.vocab_size}"
    )

    # CPU threads
    if args.device == "cpu":
        cpu_threads = (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        )
        forge.set_num_threads(cpu_threads)
        print(f"CPU threads set to: {cpu_threads}")

    # Profiling setup
    if profiling_enabled:
        forge.profiler_enable()
        forge.profiler_set_cuda_events(args.device == "cuda")

    # Context + warmup
    ctx = model.create_context(
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_layers=args.gpu_layers,
    )
    stats = ctx.memory_stats()
    print(
        f"KV Cache: dtype={stats.get('kv_cache_dtype', 'unknown')}, "
        f"size: {stats.get('kv_cache_nbytes', 0) / 1024 / 1024:.1f} MB"
    )

    if args.device == "cuda":
        print("Warming up CUDA kernels...")
        try:
            if profiling_enabled:
                perf.start("startup.warmup")
            ctx.warmup()
            if profiling_enabled:
                perf.stop("startup.warmup")
            print("Warmup done!")
        except RuntimeError as e:
            print(f"Warmup skipped ({e})")
        del ctx
        gc.collect()
    else:
        del ctx

    t5 = time.time()
    if profiling_enabled:
        print(f"\n--- Startup: [{t5 - t0:.2f}s] ---")
        perf.print_summary()
        perf.reset()
        forge.profiler_reset()

    return model, tokenizer


# ===========================================================================
# Argument Parsing
# ===========================================================================


def add_common_args(parser, gpu_layers_default=-1, temperature_default=0.7):
    """Add standard inference arguments to an ArgumentParser.

    Individual scripts should call this and then add model-specific args.
    """
    parser.add_argument("--model-path", type=str, default=None, help="Path to .gguf model file")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=gpu_layers_default,
        help="Number of layers to place on GPU (-1 for all)",
    )
    parser.add_argument(
        "--kv-cache-dtype", type=str, default="fp32", choices=["fp32", "q4_0"],
        help="KV cache data type",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=temperature_default,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="Top-k sampling parameter (0 to disable)"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable performance profiling (Python + C++ PerfProfiler)",
    )
    parser.add_argument(
        "--verify-tokenizer", action="store_true",
        help="Verify tokenizer against transformers and exit",
    )


def resolve_model_path(args, default_paths):
    """Resolve model path from args or default paths.

    Args:
        args: Parsed argparse namespace.
        default_paths: List of candidate paths to try if args.model_path is None.

    Returns:
        Resolved model path string.
    """
    if args.model_path is not None:
        return args.model_path

    for path in default_paths:
        if os.path.exists(path):
            return path

    print("No model file found. Searched:")
    for path in default_paths:
        print(f"  {path}")
    print("Please specify --model-path")
    sys.exit(1)
