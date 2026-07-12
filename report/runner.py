"""
Benchmark 统一运行器：自动发现并运行所有 BenchmarkBase 子类。

Usage:
    python -m report.runner                          # 全部模型
    python -m report.runner --models tinyllama,qwen   # 指定模型
    python -m report.runner --skip-llama-cpp          # 跳过 llama.cpp
    python -m report.runner --skip-cpu                # 跳过 CPU
    python -m report.runner --num-runs 5
    python -m report.runner --output-dir resource/reports
"""

import sys
import os
import argparse
import importlib
import pkgutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base import BenchmarkBase, get_gpu_info


def discover_benchmarks():
    """自动发现 report/ 下所有 BenchmarkBase 子类。"""
    benchmarks = []
    for importer, modname, ispkg in pkgutil.iter_modules(
        [os.path.dirname(os.path.abspath(__file__))]
    ):
        if modname in ("base", "chart", "runner", "benchmark", "generate_report", "__init__"):
            continue
        if modname.endswith("_benchmark") or modname in ("tinyllama", "qwen"):
            try:
                mod = importlib.import_module(modname)
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, BenchmarkBase)
                        and obj is not BenchmarkBase
                    ):
                        benchmarks.append(obj)
            except Exception:
                pass
    return benchmarks


def main():
    parser = argparse.ArgumentParser(description="Forge Benchmark Runner")
    parser.add_argument(
        "--models", type=str, default=None, help="Comma-separated list of model names to benchmark"
    )
    parser.add_argument("--skip-llama-cpp", action="store_true", help="Skip llama.cpp benchmark")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU benchmarks (slow)")
    parser.add_argument("--num-runs", type=int, default=None, help="Number of benchmark runs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: resource/reports)",
    )
    args = parser.parse_args()

    gpu_name, gpu_mem = get_gpu_info()
    print(f"GPU: {gpu_name} ({gpu_mem} MB)")
    print("=" * 60)
    print("  Forge vs llama.cpp Performance Benchmark")
    print("=" * 60)

    all_benchmarks = discover_benchmarks()

    if args.models:
        selected_names = [n.strip().lower() for n in args.models.split(",")]
        all_benchmarks = [
            b
            for b in all_benchmarks
            if b.MODEL_NAME.lower() in selected_names
            or b.__module__.split(".")[-1].lower() in selected_names
        ]

    if not all_benchmarks:
        print("No benchmarks found.")
        sys.exit(1)

    for bench_cls in all_benchmarks:
        name = getattr(bench_cls, "MODEL_NAME", bench_cls.__name__)
        print(f"\n{'=' * 60}")
        print(f"  Found: {name}")
        print(f"{'=' * 60}")

        bench = bench_cls(
            output_dir=args.output_dir,
            skip_llama_cpp=args.skip_llama_cpp,
            skip_cpu=args.skip_cpu,
        )
        if args.num_runs:
            bench.NUM_RUNS = args.num_runs

        bench.run()
        bench.save()

    print("\nAll benchmarks complete!")


if __name__ == "__main__":
    main()
