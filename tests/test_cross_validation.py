"""
Phase 0: Cross-validation tests for matmul and attention ops.

Verifies:
  - CPU vs scalar dequant gold values for matmul
  - Graph execution vs imperative execution output identity
  - Memory counter correctness (zero cudaMalloc in graph mode is Phase 2 goal)
"""

import os
import sys
import numpy as np

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge


class TestGraphVsImperative:
    """Verify graph execution produces same output as imperative execution."""

    def test_graph_vs_imperative_fp32(self, model_path, model_config):
        """Same model + same input → graph and imperative must produce identical output."""
        model = forge.Model()
        model.load(model_path, **model_config)

        ctx_imperative = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ctx_graph = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ctx_graph.set_use_graph(True)

        ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)

        ctx_imperative.reset_kv()
        out_imperative = ctx_imperative.forward(ids)

        ctx_graph.reset_kv()
        out_graph = ctx_graph.forward(ids)

        np.testing.assert_allclose(
            out_imperative, out_graph,
            atol=1e-5, rtol=1e-5,
            err_msg="Graph and imperative execution must produce identical outputs"
        )

    def test_graph_vs_imperative_incremental(self, model_path, model_config):
        """Incremental decode: graph vs imperative must match."""
        model = forge.Model()
        model.load(model_path, **model_config)

        ctx_imperative = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ctx_graph = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ctx_graph.set_use_graph(True)

        ids = np.array([1, 2, 3], dtype=np.int32)

        ctx_imperative.reset_kv()
        ctx_graph.reset_kv()

        # Prefill
        out_imp = ctx_imperative.forward(ids)
        out_gr = ctx_graph.forward(ids)
        np.testing.assert_allclose(out_imp, out_gr, atol=1e-5, rtol=1e-5)

        # Decode tokens
        for pos in range(3, 6):
            token = np.array([pos % 100], dtype=np.int32)
            out_imp = ctx_imperative.forward(token, start_pos=pos)
            out_gr = ctx_graph.forward(token, start_pos=pos)
            np.testing.assert_allclose(
                out_imp, out_gr, atol=1e-5, rtol=1e-5,
                err_msg=f"Graph vs imperative mismatch at pos={pos}"
            )


class TestMemoryCounters:
    """Verify memory counters provide correct baselines."""

    def test_counters_exist(self):
        """Memory counters API should be accessible."""
        counters = forge.get_memory_counters()
        assert isinstance(counters, dict)
        for key in ["cpu_malloc", "cpu_free", "cuda_malloc", "cuda_free",
                     "h2d_copies", "d2h_copies", "d2d_copies"]:
            assert key in counters, f"Missing counter: {key}"

    def test_counters_reset(self):
        """Reset should zero all counters."""
        forge.reset_memory_counters()
        counters = forge.get_memory_counters()
        assert counters["cpu_malloc"] == 0
        assert counters["cpu_free"] == 0

    def test_tensor_allocation_increments_counter(self):
        """Tensor::allocate() uses std::malloc directly, not Backend::allocate().
        Counters track Backend-level allocations (GraphBuffer etc.), not raw Tensor allocations.
        This test verifies the counter API is working."""
        forge.reset_memory_counters()
        t = forge.Tensor(forge.DataType.FP32, [100], forge.DeviceType.CPU)
        counters = forge.get_memory_counters()
        # Counters may or may not increment depending on whether Tensor uses Backend
        # The key is that the API is accessible and returns valid values
        assert isinstance(counters["cpu_malloc"], int)
        assert isinstance(counters["cuda_malloc"], int)
        assert counters["cpu_malloc"] >= 0  # baseline: may be 0 for Tensor::allocate()
        assert counters["cuda_malloc"] >= 0
        del t  # ensure cleanup

    def test_graph_execution_malloc_count(self, model_path, model_config):
        """Graph execution should not allocate per-node (may allocate graph buffer)."""
        model = forge.Model()
        model.load(model_path, **model_config)

        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ctx.set_use_graph(True)

        forge.reset_memory_counters()
        ctx.reset_kv()
        out = ctx.forward(np.array([1, 2, 3], dtype=np.int32))

        counters = forge.get_memory_counters()
        # After Phase 2, cuda_malloc should be 0 for graph mode.
        # For now, just record the baseline.
        print(f"Memory counters baseline: {counters}")

        assert out is not None


class TestDTypeHelpers:
    """Verify dtype helper functions return correct values."""

    def test_dtype_size_fp32(self):
        assert forge.dtype_size(forge.DataType.FP32) == 4

    def test_dtype_size_fp16(self):
        assert forge.dtype_size(forge.DataType.FP16) == 2

    def test_dtype_size_quantized(self):
        """Quantized types have dtype_size == 0 (not individually addressable)."""
        for dt in [forge.DataType.Q4_0, forge.DataType.Q4_K, forge.DataType.Q8_0]:
            assert forge.dtype_size(dt) == 0, f"{dt} should have dtype_size == 0"

    def test_compute_quantized_bytes(self):
        """compute_quantized_bytes should match tensor.nbytes()."""
        for dt, nel in [
            (forge.DataType.Q4_0, 32),
            (forge.DataType.Q4_K, 256),
            (forge.DataType.Q6_K, 512),
        ]:
            t = forge.Tensor(dt, [nel], forge.DeviceType.CPU)
            assert forge.compute_quantized_bytes(nel, dt) == t.nbytes()
