"""
Qwen2.5-7B-Instruct Q4_0 interactive chat using Forge.

Uses Forge's built-in Tokenizer loaded directly from GGUF files.
No external tokenizer files or transformers dependency required.

Prerequisites:
  1. Download GGUF model:
     (e.g., Qwen2.5-7B-Instruct-Q4_0.gguf from HuggingFace)

Usage:
  # Direct GGUF loading (recommended):
  python examples/qwen_inference.py --model-path /path/to/model.gguf

  # Legacy NINF format:
  python examples/qwen_inference.py --model-path /path/to/model.ninf

  # Other options:
  python examples/qwen_inference.py --device cpu --gpu-layers 0
  python examples/qwen_inference.py --no-stream
  python examples/qwen_inference.py --max-new-tokens 512 --temperature 0.8
  python examples/qwen_inference.py --verify-tokenizer

Interactive commands:
  /quit   - Exit the chat
  /clear  - Clear conversation history
  /help   - Show help message
"""

import sys
import os
import time
import gc
import argparse
from collections import defaultdict

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge
import numpy as np

MODEL_DIR = "/mnt/g/AI/Qwen2.5-7B-Instruct-GGUF"
GGUF_MODEL_PATH = os.path.join(MODEL_DIR, "Qwen2.5-7B-Instruct-Q4_0.gguf")
NINF_MODEL_PATH = os.path.join(MODEL_DIR, "qwen2.5-7b-q4_0_v6.ninf")
TOKENIZER_DIR = os.path.join(MODEL_DIR)

QWEN_CONFIG = {
    "vocab_size": 152064,
    "hidden_dim": 3584,
    "intermediate_dim": 18944,
    "num_layers": 28,
    "num_heads": 28,
    "num_kv_heads": 4,
    "head_dim": 128,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "max_seq_len": 4096,
    "arch_type": "qwen2",
    "norm_type": "rmsnorm",
    "activation": "silu_gelu",
    "tie_embeddings": True,
}


class PerfTimer:
    """Lightweight hierarchical performance timer for Python-level profiling."""

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
            f"{'Stage':<40} {'Count':>6} {'Total(ms)':>10} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'%Total':>7}"
        )
        print("-" * 90)

        grand_total = sum(r["total_ms"] for r in self._timings.values())
        sorted_items = sorted(self._timings.items(), key=lambda x: x[1]["total_ms"], reverse=True)

        for name, rec in sorted_items:
            avg = rec["total_ms"] / rec["count"] if rec["count"] > 0 else 0
            pct = (rec["total_ms"] / grand_total * 100) if grand_total > 0 else 0
            print(
                f"{name:<40} {rec['count']:>6} {rec['total_ms']:>10.2f} {avg:>10.3f} {rec['min_ms']:>10.3f} {rec['max_ms']:>10.3f} {pct:>6.1f}%"
            )

        print("-" * 90)
        print(
            f"{'TOTAL':<40} {'':>6} {grand_total:>10.2f} {'':>10} {'':>10} {'':>10} {'100.0%':>7}"
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
            f"{'Operation':<45} {'Count':>6} {'Total(ms)':>10} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'%Total':>7}"
        )
        print("-" * 90)

        sorted_ops = sorted(summary.items(), key=lambda x: x[1]["total_ms"], reverse=True)
        grand_total = sum(v["total_ms"] for v in summary.values())

        for name, rec in sorted_ops:
            pct = (rec["total_ms"] / grand_total * 100) if grand_total > 0 else 0
            print(
                f"{name:<45} {rec['count']:>6} {rec['total_ms']:>10.2f} {rec['avg_ms']:>10.3f} {rec['min_ms']:>10.3f} {rec['max_ms']:>10.3f} {pct:>6.1f}%"
            )

        print("-" * 90)
        print(
            f"{'TOTAL':<45} {'':>6} {grand_total:>10.2f} {'':>10} {'':>10} {'':>10} {'100.0%':>7}"
        )
        print("=" * 90)
    except Exception as e:
        print(f"[Profiler] Could not retrieve C++ profile: {e}")


def print_full_profile():
    """Print both Python-level and C++ operator-level profiles."""
    perf.print_summary()
    print_cpp_profiler_summary()


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

    if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
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


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """Apply chat template to produce token IDs.

    Supports ChatML format (used by Qwen2.5):
       system:    <|im_start|>system\n{content}<|im_end|>\n
       user:      <|im_start|>user\n{content}<|im_end|>\n
       assistant: <|im_start|>assistant\n{content}<|im_end|>\n
       generation prompt: <|im_start|>assistant\n

    Special tokens (<|im_start|>, <|im_end|>) are inserted as single token IDs
    to ensure correct encoding. Text segments are encoded individually.
    """
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")

    ids = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # <|im_start|>role\n
        ids.append(im_start_id)
        ids.extend(tokenizer.encode(role + "\n", add_bos=False))

        # content
        ids.extend(tokenizer.encode(content, add_bos=False))

        # <|im_end|>\n
        ids.append(im_end_id)
        ids.extend(tokenizer.encode("\n", add_bos=False))

    if add_generation_prompt:
        # <|im_start|>assistant\n
        ids.append(im_start_id)
        ids.extend(tokenizer.encode("assistant\n", add_bos=False))

    return ids


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
):
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    generated_tokens = []
    token_buffer = []
    decode_start = time.time()
    debug_tokens = os.environ.get("FORGE_DEBUG_TOKENS", "0") == "1"

    # Per-token timing for profiling
    token_times = [] if profiling_enabled else None

    def on_token(token_id, step):
        t0 = time.perf_counter() if profiling_enabled else 0
        if debug_tokens:
            token_str = (
                tokenizer.id_to_token(token_id) if hasattr(tokenizer, "id_to_token") else "?"
            )
            print(f"\n[DEBUG token_id={token_id} token={token_str!r}]", end="", flush=True)
        generated_tokens.append(token_id)
        token_buffer.append(token_id)

        if len(token_buffer) >= 4 or token_id == eos_token_id:
            try:
                perf.start("decode/tokenize") if profiling_enabled else None
                text = tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
                perf.stop("decode/tokenize") if profiling_enabled else None
                print(text, end="", flush=True)
                token_buffer.clear()
            except UnicodeDecodeError:
                # Partial multi-byte UTF-8 at buffer boundary, wait for more tokens
                if token_id == eos_token_id:
                    text = tokenizer.decode(
                        token_buffer, skip_special=True, strip_leading_space=False, errors="replace"
                    )
                    print(text, end="", flush=True)
                    token_buffer.clear()

        if profiling_enabled:
            t1 = time.perf_counter()
            token_times.append((t1 - t0) * 1000.0)

    if profiling_enabled:
        perf.start("generate_stream/total")

    np_ids = np.array(input_ids, dtype=np.int32)
    model.generate_stream(
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

    if profiling_enabled:
        perf.stop("generate_stream/total")

    if token_buffer:
        text = tokenizer.decode(token_buffer, skip_special=True, strip_leading_space=False)
        print(text, end="", flush=True)
        token_buffer.clear()

    elapsed = time.time() - decode_start

    if profiling_enabled and token_times and len(token_times) > 1:
        # Skip first token (prefill)
        decode_times = token_times[1:]
        avg_ms = sum(decode_times) / len(decode_times) if decode_times else 0
        perf.record("decode/per_token_avg_ms", avg_ms)
        perf.record("decode/per_token_min_ms", min(decode_times) if decode_times else 0)
        perf.record("decode/per_token_max_ms", max(decode_times) if decode_times else 0)

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
):
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    start_time = time.time()
    np_ids = np.array(input_ids, dtype=np.int32)
    result = model.generate(
        prompt_ids=np_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        do_sample=temperature > 0,
        eos_token_id=eos_token_id,
        kv_cache_dtype="fp32",
        gpu_layers=-1,
    )
    elapsed = time.time() - start_time

    num_generated = result["num_generated_tokens"]
    new_tokens = list(result["token_ids"])

    response = tokenizer.decode(new_tokens, skip_special=True, strip_leading_space=False)
    print(response, end="", flush=True)

    return num_generated, elapsed


def interactive_chat(model, tokenizer, args):
    """Interactive multi-turn chat loop."""
    global profiling_enabled
    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    conversation = [system_msg]
    ctx = None

    print("\n" + "=" * 60)
    print("  Qwen2.5-7B-Instruct Interactive Chat (Forge)")
    print(f"  Device: {args.device}")
    if profiling_enabled:
        print("  Profiling: ON (Python + C++ PerfProfiler)")
    print("  Commands: /quit, /clear, /help, /profile")
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
            print("Bye!")
            break
        elif user_input == "/clear":
            conversation = [system_msg]
            perf.reset()
            print("[Conversation cleared]\n")
            continue
        elif user_input == "/help":
            print("  /quit    - Exit the chat")
            print("  /clear   - Clear conversation history")
            print("  /help    - Show this help message")
            print("  /profile - Toggle profiling on/off\n")
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

        conversation.append({"role": "user", "content": user_input})

        if profiling_enabled:
            perf.start("template/encode")
        input_ids = apply_chat_template(tokenizer, conversation, add_generation_prompt=True)
        if profiling_enabled:
            perf.stop("template/encode")

        print("Assistant: ", end="", flush=True)

        if args.no_stream:
            if profiling_enabled:
                perf.start("generate/total")
            start_time = time.time()
            np_ids = np.array(input_ids, dtype=np.int32)
            result = model.generate(
                prompt_ids=np_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                do_sample=args.temperature > 0,
                eos_token_id=tokenizer.eos_token_id,
                kv_cache_dtype=args.kv_cache_dtype,
                gpu_layers=args.gpu_layers,
            )
            elapsed = time.time() - start_time
            if profiling_enabled:
                perf.stop("generate/total")
            num_generated = result["num_generated_tokens"]
            new_tokens = list(result["token_ids"])
            if profiling_enabled:
                perf.start("decode/tokenize")
            assistant_text = tokenizer.decode(
                new_tokens, skip_special=True, strip_leading_space=False
            )
            if profiling_enabled:
                perf.stop("decode/tokenize")
            print(assistant_text, end="", flush=True)
        else:
            if ctx is not None:
                del ctx
                gc.collect()
            ctx = model.create_context(
                kv_cache_dtype=args.kv_cache_dtype,
                gpu_layers=args.gpu_layers,
            )
            generated_tokens, elapsed = generate_streaming(
                model,
                ctx,
                tokenizer,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                kv_cache_dtype=args.kv_cache_dtype,
                gpu_layers=args.gpu_layers,
            )
            num_generated = len(generated_tokens)
            assistant_text = tokenizer.decode(
                generated_tokens, skip_special=True, strip_leading_space=False
            )

        print()
        if num_generated > 0 and elapsed > 0:
            speed = num_generated / elapsed
            prompt_len = len(input_ids)
            print(
                f"[{num_generated} tokens, {elapsed:.2f}s, {speed:.1f} tok/s, prompt={prompt_len} tokens]"
            )

        if profiling_enabled:
            print_full_profile()
            perf.reset()
            forge.profiler_reset()

        conversation.append({"role": "assistant", "content": assistant_text})
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct inference with Forge")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to .gguf or .ninf model file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=28, help="Number of layers to place on GPU (-1 for all)"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="fp32",
        choices=["fp32", "q4_0"],
        help="KV cache data type",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy)"
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="Top-k sampling parameter (0 to disable)"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling (Python + C++ PerfProfiler)",
    )
    parser.add_argument(
        "--verify-tokenizer",
        action="store_true",
        help="Verify tokenizer against transformers and exit",
    )
    return parser.parse_args()


def main():
    global profiling_enabled
    args = parse_args()

    if args.profile:
        profiling_enabled = True
        print("[Profiling enabled - Python timing + C++ PerfProfiler]")

    model_path = args.model_path
    if model_path is None:
        if os.path.exists(GGUF_MODEL_PATH):
            model_path = GGUF_MODEL_PATH
        elif os.path.exists(NINF_MODEL_PATH):
            model_path = NINF_MODEL_PATH
        else:
            print("No model file found. Searched:")
            print(f"  {GGUF_MODEL_PATH}")
            print(f"  {NINF_MODEL_PATH}")
            print("Please specify --model-path")
            sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    is_gguf = model_path.endswith(".gguf")

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

    if args.verify_tokenizer:
        verify_tokenizer(tokenizer, TOKENIZER_DIR)
        return

    print(f"Loading model ({'GGUF' if is_gguf else 'NINF'}) on {args.device}...")
    forge.Logger.set_level(2 if args.verbose else 1)

    t2 = time.time()
    model = forge.Model()
    if is_gguf:
        model.load_gguf(model_path, device=args.device)
    else:
        model.load(model_path, **QWEN_CONFIG, device=args.device)
    t3 = time.time()
    print(f"Model loaded! [{t3 - t2:.2f}s]")

    if profiling_enabled:
        forge.profiler_enable()
        if args.device == "cuda":
            forge.profiler_set_cuda_events(True)
        else:
            forge.profiler_set_cuda_events(False)

    t4 = time.time()
    ctx = model.create_context(
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_layers=args.gpu_layers,
    )
    stats = ctx.memory_stats()
    print(
        f"KV Cache: dtype={stats.get('kv_cache_dtype', 'unknown')}, "
        f"size: {stats.get('kv_cache_nbytes', 0) / 1024 / 1024:.1f} MB"
    )

    # Warmup to trigger CUDA kernel JIT compilation
    if args.device == "cuda":
        print("Warming up CUDA kernels...")
        try:
            ctx.warmup()
            print("Warmup done!")
        except RuntimeError as e:
            print(f"Warmup skipped ({e})")
        del ctx  # Free warmup context GPU memory
        gc.collect()
    else:
        del ctx
    t5 = time.time()
    print(f"Context + warmup: [{t5 - t4:.2f}s]")
    print(
        f"Total startup: [{t5 - t0:.2f}s] (tokenizer={t1 - t0:.2f}s, model={t3 - t2:.2f}s, ctx={t5 - t4:.2f}s)"
    )

    interactive_chat(model, tokenizer, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
