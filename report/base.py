"""
BenchmarkBase: 所有模型 benchmark 的抽象基类。

子类只需设置类常量并调用 run() + save() 即可完成完整的 benchmark 流程。
"""

import sys
import os
import time
import json
import subprocess
from abc import ABC

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_DIR, "build")
if os.path.exists(BUILD_DIR):
    sys.path.insert(0, BUILD_DIR)

import numpy as np


def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return parts[0].strip(), int(parts[1].strip().split()[0])
    except Exception:
        pass
    return "Unknown GPU", 0


def resolve_path(path):
    """相对路径相对于项目根目录解析。"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_DIR, path)


class BenchmarkBase(ABC):
    """Benchmark 基类。继承并覆写类常量即可定义新模型 benchmark。"""

    MODEL_NAME = ""
    MODEL_PATH = ""
    BENCH_TOKENS = 100
    PREFILL_LENGTHS = [1, 4, 16, 64, 128, 256]
    NUM_RUNS = 3
    WARMUP_TOKENS = 10
    GPU_LAYERS = -1  # -1 = all layers on GPU

    def __init__(self, output_dir=None, skip_llama_cpp=False, skip_cpu=False):
        self.output_dir = output_dir or self._default_output_dir()
        self.skip_llama_cpp = skip_llama_cpp
        self.skip_cpu = skip_cpu
        self.results = {}

    # ── 子类可覆写 ──────────────────────────────

    @property
    def model_path(self):
        return resolve_path(self.MODEL_PATH)

    def _default_output_dir(self):
        safe = (
            self.MODEL_NAME.replace(" ", "_")
            .replace("/", "_")
            .replace("Q4_0", "")
            .strip("_")
            .lower()
        )
        return os.path.join(PROJECT_DIR, "resource", "reports", safe)

    # ── Benchmark 方法 ───────────────────────────

    @staticmethod
    def bench_forge_decode(model_path, device, gpu_layers, num_tokens, num_runs):
        import forge

        forge.Logger.set_level(0)

        model = forge.Model()
        model.load_gguf(model_path, device=device)

        ids = np.array([1], dtype=np.int32)

        model.generate(
            ids, max_new_tokens=BenchmarkBase.WARMUP_TOKENS, do_sample=False, gpu_layers=gpu_layers
        )

        speeds = []
        for _ in range(num_runs):
            start = time.time()
            result = model.generate(
                ids, max_new_tokens=num_tokens, do_sample=False, gpu_layers=gpu_layers
            )
            elapsed = time.time() - start
            actual = result.get("num_generated_tokens", num_tokens)
            speeds.append(actual / elapsed)

        return speeds

    @staticmethod
    def bench_forge_cpu_decode(model_path, num_tokens, num_runs):
        return BenchmarkBase.bench_forge_decode(model_path, "cpu", 0, num_tokens, num_runs)

    @staticmethod
    def bench_forge_prefill(model_path, device, gpu_layers, prompt_lengths, num_runs):
        import forge

        forge.Logger.set_level(0)

        model = forge.Model()
        model.load_gguf(model_path, device=device)

        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=gpu_layers)

        results = {}
        for plen in prompt_lengths:
            prompt_ids = np.ones(plen, dtype=np.int32)
            speeds = []
            for _ in range(num_runs):
                ctx.reset_kv()
                start = time.time()
                ctx.forward(prompt_ids)
                elapsed = time.time() - start
                speeds.append(plen / elapsed)
            results[plen] = speeds

        return results

    @staticmethod
    def bench_llama_cpp_decode(model_path, n_gpu_layers, num_tokens, num_runs):
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, verbose=False)

        llm.create_completion("test", max_tokens=BenchmarkBase.WARMUP_TOKENS, temperature=0)

        speeds = []
        for _ in range(num_runs):
            start = time.time()
            response = llm.create_completion("test", max_tokens=num_tokens, temperature=0)
            elapsed = time.time() - start
            completion_tokens = response["usage"].get("completion_tokens", num_tokens)
            if completion_tokens > 0:
                speeds.append(completion_tokens / elapsed)
            else:
                speeds.append(0)

        return speeds

    @staticmethod
    def bench_llama_cpp_cpu_decode(model_path, num_tokens, num_runs):
        return BenchmarkBase.bench_llama_cpp_decode(model_path, 0, num_tokens, num_runs)

    @staticmethod
    def bench_llama_cpp_prefill(model_path, n_gpu_layers, prompt_lengths, num_runs):
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, verbose=False)

        results = {}
        for plen in prompt_lengths:
            prompt = "hello " * max(1, plen // 2)
            prompt = prompt[:plen] if len(prompt) > plen else prompt
            speeds = []
            for _ in range(num_runs):
                start = time.time()
                llm.eval(llm.tokenize(prompt.encode("utf-8"), add_bos=True))
                elapsed = time.time() - start
                token_count = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True))
                speeds.append(token_count / elapsed)
            results[plen] = speeds

        return results

    # ── 运行全部 benchmark ──────────────────────

    def run(self):
        """运行所有 benchmark，返回 self.results dict。"""
        model_path = self.model_path
        model_name = self.MODEL_NAME
        skip_cpu = self.skip_cpu
        skip_llama_cpp = self.skip_llama_cpp
        num_runs = self.NUM_RUNS
        num_tokens = self.BENCH_TOKENS
        plens = self.PREFILL_LENGTHS
        gl = self.GPU_LAYERS

        print(f"\n{'=' * 60}")
        print(f"  {model_name}")
        print(f"  {model_path}")
        print(f"{'=' * 60}")

        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            return self.results

        self.results = {"model_name": model_name, "model_path": model_path}

        gpu_name, gpu_mem = get_gpu_info()
        self.results["gpu_name"] = gpu_name
        self.results["gpu_mem_mb"] = gpu_mem

        # Forge GPU Decode
        print(f"\n  [Forge GPU Decode] {num_tokens} tokens x {num_runs} runs...")
        try:
            speeds = self.bench_forge_decode(model_path, "cuda", gl, num_tokens, num_runs)
            self.results["forge_gpu_decode"] = {
                "speeds": speeds,
                "mean": float(np.mean(speeds)),
                "std": float(np.std(speeds)),
                "median": float(np.median(speeds)),
            }
            print(f"    Mean: {np.mean(speeds):.1f} ± {np.std(speeds):.1f} tok/s")
        except Exception as e:
            print(f"    Failed: {e}")
            self.results["forge_gpu_decode"] = None

        # Forge CPU Decode
        if not skip_cpu:
            print(f"\n  [Forge CPU Decode] {num_tokens} tokens x {num_runs} runs...")
            try:
                speeds = self.bench_forge_cpu_decode(model_path, num_tokens, num_runs)
                self.results["forge_cpu_decode"] = {
                    "speeds": speeds,
                    "mean": float(np.mean(speeds)),
                    "std": float(np.std(speeds)),
                    "median": float(np.median(speeds)),
                }
                print(f"    Mean: {np.mean(speeds):.1f} ± {np.std(speeds):.1f} tok/s")
            except Exception as e:
                print(f"    Failed: {e}")
                self.results["forge_cpu_decode"] = None

        # Forge GPU Prefill
        print(f"\n  [Forge GPU Prefill] lengths={plens}...")
        try:
            prefill = self.bench_forge_prefill(model_path, "cuda", gl, plens, num_runs)
            self.results["forge_gpu_prefill"] = {}
            for plen, speeds in prefill.items():
                self.results["forge_gpu_prefill"][plen] = {
                    "speeds": speeds,
                    "mean": float(np.mean(speeds)),
                    "std": float(np.std(speeds)),
                }
                print(f"    len={plen}: {np.mean(speeds):.1f} tok/s")
        except Exception as e:
            print(f"    Failed: {e}")
            self.results["forge_gpu_prefill"] = None

        # llama.cpp GPU Decode
        if not skip_llama_cpp:
            print(f"\n  [llama.cpp GPU Decode] {num_tokens} tokens x {num_runs} runs...")
            try:
                speeds = self.bench_llama_cpp_decode(model_path, gl, num_tokens, num_runs)
                self.results["llama_cpp_gpu_decode"] = {
                    "speeds": speeds,
                    "mean": float(np.mean(speeds)),
                    "std": float(np.std(speeds)),
                    "median": float(np.median(speeds)),
                }
                print(f"    Mean: {np.mean(speeds):.1f} ± {np.std(speeds):.1f} tok/s")
            except Exception as e:
                print(f"    Failed: {e}")
                self.results["llama_cpp_gpu_decode"] = None

            # llama.cpp CPU Decode
            if not skip_cpu:
                print(f"\n  [llama.cpp CPU Decode] {num_tokens} tokens x {num_runs} runs...")
                try:
                    speeds = self.bench_llama_cpp_cpu_decode(model_path, num_tokens, num_runs)
                    self.results["llama_cpp_cpu_decode"] = {
                        "speeds": speeds,
                        "mean": float(np.mean(speeds)),
                        "std": float(np.std(speeds)),
                        "median": float(np.median(speeds)),
                    }
                    print(f"    Mean: {np.mean(speeds):.1f} ± {np.std(speeds):.1f} tok/s")
                except Exception as e:
                    print(f"    Failed: {e}")
                    self.results["llama_cpp_cpu_decode"] = None

            # llama.cpp GPU Prefill
            print(f"\n  [llama.cpp GPU Prefill] lengths={plens}...")
            try:
                prefill = self.bench_llama_cpp_prefill(model_path, gl, plens, num_runs)
                self.results["llama_cpp_gpu_prefill"] = {}
                for plen, speeds in prefill.items():
                    self.results["llama_cpp_gpu_prefill"][plen] = {
                        "speeds": speeds,
                        "mean": float(np.mean(speeds)),
                        "std": float(np.std(speeds)),
                    }
                    print(f"    len={plen}: {np.mean(speeds):.1f} tok/s")
            except Exception as e:
                print(f"    Failed: {e}")
                self.results["llama_cpp_gpu_prefill"] = None

        return self.results

    # ── 保存 ────────────────────────────────────

    def save_results_json(self):
        """保存 results.json 到 output_dir。"""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "results.json")
        with open(path, "w") as f:
            json.dump([self.results], f, indent=2, default=str)
        print(f"\nResults saved to {path}")

    def save_report(self):
        """生成图表并保存到 output_dir。"""
        from chart import generate_report

        generate_report([self.results], self.output_dir)

    def save(self):
        """保存 JSON + 图表。"""
        self.save_results_json()
        self.save_report()
