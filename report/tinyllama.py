"""
TinyLlama-1.1B-Chat benchmark: Forge vs llama.cpp comparison.

Usage:
    python -m report.tinyllama
    python -m report.tinyllama --skip-llama-cpp --skip-cpu
    python -m report.tinyllama --num-runs 5
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base import BenchmarkBase


class TinyLlamaBenchmark(BenchmarkBase):
    MODEL_NAME = "TinyLlama-1.1B-Chat Q4_0"
    MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
    BENCH_TOKENS = 100


def main():
    parser = argparse.ArgumentParser(description="TinyLlama-1.1B Benchmark")
    parser.add_argument("--skip-llama-cpp", action="store_true",
                        help="Skip llama.cpp benchmark")
    parser.add_argument("--skip-cpu", action="store_true",
                        help="Skip CPU benchmarks (slow)")
    parser.add_argument("--num-runs", type=int, default=None,
                        help="Number of runs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results + charts")
    args = parser.parse_args()

    bench = TinyLlamaBenchmark(
        output_dir=args.output_dir,
        skip_llama_cpp=args.skip_llama_cpp,
        skip_cpu=args.skip_cpu,
    )
    if args.num_runs:
        bench.NUM_RUNS = args.num_runs

    bench.run()
    bench.save()


if __name__ == "__main__":
    main()
