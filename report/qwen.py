"""
Qwen2.5-7B-Instruct benchmark: Forge vs llama.cpp comparison.

Usage:
    python -m report.qwen
    python -m report.qwen --skip-llama-cpp --skip-cpu
    python -m report.qwen --num-runs 5
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base import BenchmarkBase


class QwenBenchmark(BenchmarkBase):
    MODEL_NAME = "Qwen2.5-7B-Instruct Q4_0"
    MODEL_PATH = "/mnt/g/AI/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_0.gguf"
    BENCH_TOKENS = 100


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct Benchmark")
    parser.add_argument("--skip-llama-cpp", action="store_true", help="Skip llama.cpp benchmark")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU benchmarks (slow)")
    parser.add_argument("--num-runs", type=int, default=None, help="Number of runs")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results + charts"
    )
    args = parser.parse_args()

    bench = QwenBenchmark(
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
