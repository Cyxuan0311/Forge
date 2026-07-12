"""
Forge Backend & Device Performance Benchmarks.

Benchmarks backend capabilities, memory allocation, device transfer,
and CPU vs CUDA performance comparison.

Usage:
    python3 benchmarks/bench_backend.py
"""

import os
import sys
import time

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge
import numpy as np


def bench_tensor_alloc_dealloc(dtype, shape, device, iters=100, warmup=10):
    """Benchmark tensor creation + deallocation (as a proxy for backend allocation)."""

    def run():
        t = forge.Tensor(dtype, shape, device)
        del t

    for _ in range(warmup):
        run()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        t = forge.Tensor(dtype, shape, device)
        del t
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean_us": times.mean() * 1e6,
        "p50_us": np.percentile(times, 50) * 1e6,
    }


def bench_tensor_device_transfer(size, iters=20, warmup=5):
    """Benchmark CPU->CUDA and CUDA->CPU tensor transfer (if CUDA available)."""
    mgr = forge.BackendManager.instance()
    if not mgr.has_cuda():
        return None

    cpu_tensor = forge.Tensor(forge.DataType.FP32, [size], forge.DeviceType.CPU)
    cuda_tensor = forge.Tensor(forge.DataType.FP32, [size], forge.DeviceType.CUDA)

    # CPU -> CUDA
    for _ in range(warmup):
        cuda_tensor.copy_from(cpu_tensor)

    times_h2d = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cuda_tensor.copy_from(cpu_tensor)
        t1 = time.perf_counter()
        times_h2d.append(t1 - t0)

    # CUDA -> CPU
    for _ in range(warmup):
        cpu_tensor.copy_from(cuda_tensor)

    times_d2h = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cpu_tensor.copy_from(cuda_tensor)
        t1 = time.perf_counter()
        times_d2h.append(t1 - t0)

    times_h2d = np.array(times_h2d)
    times_d2h = np.array(times_d2h)
    mb = size * 4 / (1024 * 1024)  # FP32 = 4 bytes

    return {
        "size_mb": mb,
        "h2d_mean_ms": times_h2d.mean() * 1000,
        "h2d_bandwidth_gbps": mb / (times_h2d.mean() * 1e3) if times_h2d.mean() > 0 else 0,
        "d2h_mean_ms": times_d2h.mean() * 1000,
        "d2h_bandwidth_gbps": mb / (times_d2h.mean() * 1e3) if times_d2h.mean() > 0 else 0,
    }


def main():
    print("=" * 70)
    print("  Forge Backend & Device Performance Benchmarks")
    print("=" * 70)

    mgr = forge.BackendManager.instance()

    # --- Backend Info ---
    print("\n--- Available Backends ---")
    for name in mgr.available_backends():
        backend = mgr.get_backend(name)
        cap_strs = []
        for cap_name, cap_val in [
            ("FP32", forge.BackendCapability.FP32),
            ("FP16", forge.BackendCapability.FP16),
            ("INT8", forge.BackendCapability.INT8),
            ("Quantized", forge.BackendCapability.Quantized),
            ("UnifiedMem", forge.BackendCapability.UnifiedMemory),
            ("StreamAsync", forge.BackendCapability.StreamAsync),
        ]:
            if backend.supports(cap_val):
                cap_strs.append(cap_name)

        total_mem = backend.device_memory_total()
        free_mem = backend.device_memory_free()
        total_gb = total_mem / (1024**3)
        free_gb = free_mem / (1024**3)

        print(
            f"  {name:8s}: caps=[{', '.join(cap_strs)}]  "
            f"mem={free_gb:.1f}/{total_gb:.1f} GB  device_id={backend.device_id()}"
        )

    # --- CPU Backend Allocation ---
    print("\n--- CPU Backend Memory Allocation (via Tensor) ---")
    for size_name, shape in [
        ("64 B", [16]),
        ("4 KB", [1024]),
        ("64 KB", [16384]),
        ("1 MB", [262144]),
        ("16 MB", [4194304]),
    ]:
        stats = bench_tensor_alloc_dealloc(forge.DataType.FP32, shape, forge.DeviceType.CPU)
        print(
            f"  alloc+dealloc({size_name:8s}): {stats['mean_us']:8.1f} us  "
            f"p50={stats['p50_us']:8.1f} us"
        )

    # --- CUDA Backend ---
    if mgr.has_cuda():
        print("\n--- CUDA Backend Memory Allocation (via Tensor) ---")
        for size_name, shape in [
            ("64 B", [16]),
            ("4 KB", [1024]),
            ("64 KB", [16384]),
            ("1 MB", [262144]),
            ("16 MB", [4194304]),
        ]:
            stats = bench_tensor_alloc_dealloc(forge.DataType.FP32, shape, forge.DeviceType.CUDA)
            print(
                f"  alloc+dealloc({size_name:8s}): {stats['mean_us']:8.1f} us  "
                f"p50={stats['p50_us']:8.1f} us"
            )

        # --- Device Transfer ---
        print("\n--- Device Transfer Bandwidth ---")
        for size_name, size in [
            ("1K", 1024),
            ("10K", 10 * 1024),
            ("100K", 100 * 1024),
            ("1M", 1024 * 1024),
            ("10M", 10 * 1024 * 1024),
        ]:
            stats = bench_tensor_device_transfer(size)
            if stats:
                print(
                    f"  {size_name:6s} ({stats['size_mb']:6.2f} MB):  "
                    f"H2D={stats['h2d_mean_ms']:6.2f} ms ({stats['h2d_bandwidth_gbps']:.2f} GB/s)  "
                    f"D2H={stats['d2h_mean_ms']:6.2f} ms ({stats['d2h_bandwidth_gbps']:.2f} GB/s)"
                )
    else:
        print("\n  CUDA not available, skipping CUDA benchmarks")

    # --- Tensor Creation by Device ---
    print("\n--- Tensor Creation by Device ---")
    for device_name, device_type in [("CPU", forge.DeviceType.CPU)]:
        for shape_name, shape in [
            ("[100]", [100]),
            ("[1K]", [1024]),
            ("[32x32]", [32, 32]),
            ("[256x256]", [256, 256]),
        ]:
            times = []
            for _ in range(200):
                t0 = time.perf_counter()
                forge.Tensor(forge.DataType.FP32, shape, device_type)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            times = np.array(times)
            print(f"  {device_name} {shape_name:12s}: {times.mean() * 1e6:8.1f} us")

    if mgr.has_cuda():
        for shape_name, shape in [
            ("[100]", [100]),
            ("[1K]", [1024]),
            ("[32x32]", [32, 32]),
            ("[256x256]", [256, 256]),
        ]:
            times = []
            for _ in range(200):
                t0 = time.perf_counter()
                forge.Tensor(forge.DataType.FP32, shape, forge.DeviceType.CUDA)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            times = np.array(times)
            print(f"  CUDA {shape_name:12s}: {times.mean() * 1e6:8.1f} us")

    print("\n" + "=" * 70)
    print("  Backend benchmarks complete!")


if __name__ == "__main__":
    main()
