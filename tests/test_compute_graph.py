"""Tests for the ComputeGraph and graph-based inference mode."""
import os
import sys
import pytest
import numpy as np

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TINYLLAMA_Q4_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")


def tinyllama_available():
    return os.path.exists(TINYLLAMA_Q4_PATH)


@pytest.fixture
def model_path():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "test_model_small.ninf")
    if not os.path.exists(path):
        pytest.skip("test_model_small.ninf not found")
    return path


@pytest.fixture
def model_config():
    return {
        "vocab_size": 100,
        "hidden_dim": 32,
        "intermediate_dim": 64,
        "num_layers": 1,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 16,
        "device": "cpu",
    }


class TestGraphModeAPI:
    """Test the graph mode API on InferenceContext."""

    def test_set_use_graph(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, **model_config)
        ctx = model.create_context()
        # Should not raise
        ctx.set_use_graph(True)

    def test_use_graph_default_false(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, **model_config)
        ctx = model.create_context()
        assert ctx.use_graph() is False

    def test_use_graph_set_and_get(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, **model_config)
        ctx = model.create_context()
        ctx.set_use_graph(True)
        assert ctx.use_graph() is True
        ctx.set_use_graph(False)
        assert ctx.use_graph() is False

    def test_forward_with_graph_mode(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, **model_config)
        ctx = model.create_context()
        ctx.set_use_graph(True)
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = ctx.forward(ids)
        assert logits.shape == (3, model_config["vocab_size"])
        assert np.isfinite(logits).all()

    def test_forward_graph_vs_imperative(self, model_path, model_config):
        """Graph mode and imperative mode should produce the same output."""
        model = forge.Model()
        model.load(model_path, **model_config)

        # Imperative mode
        ctx_imp = model.create_context()
        ctx_imp.set_use_graph(False)
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits_imp = ctx_imp.forward(ids)

        # Graph mode
        ctx_graph = model.create_context()
        ctx_graph.set_use_graph(True)
        logits_graph = ctx_graph.forward(ids)

        # If graph builder is available, outputs should match
        # If not available, it falls back to imperative, so they should still match
        np.testing.assert_allclose(logits_imp, logits_graph, atol=1e-5,
                                   err_msg="Graph and imperative modes should produce same output")

    def test_generate_with_graph_mode(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, **model_config)
        ctx = model.create_context()
        ctx.set_use_graph(True)
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = model.generate(prompt, max_new_tokens=5, do_sample=False)
        assert result["num_generated_tokens"] >= 1

    def test_graph_mode_deterministic(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, **model_config)

        ctx1 = model.create_context()
        ctx1.set_use_graph(True)
        ids = np.array([1, 2, 3], dtype=np.int32)
        out1 = ctx1.forward(ids)

        ctx2 = model.create_context()
        ctx2.set_use_graph(True)
        out2 = ctx2.forward(ids)

        np.testing.assert_array_equal(out1, out2,
                                       err_msg="Graph mode should be deterministic")

    def test_graph_mode_incremental_forward(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, **model_config)
        ctx = model.create_context()
        ctx.set_use_graph(True)

        # Full forward
        ids = np.array([1, 2, 3], dtype=np.int32)
        ctx.reset_kv()
        full_logits = ctx.forward(ids)

        # Incremental forward
        ctx.reset_kv()
        for i, tid in enumerate(ids):
            step_logits = ctx.forward(np.array([tid], dtype=np.int32), start_pos=i)

        # Last token logits should match
        np.testing.assert_allclose(full_logits[-1], step_logits[-1] if step_logits.ndim > 1 else step_logits[0],
                                   atol=1e-3,
                                   err_msg="Incremental forward should match full forward in graph mode")


@pytest.mark.skipif(not tinyllama_available(), reason="TinyLlama GGUF model not found")
class TestGraphModeTinyLlama:
    """Test graph mode with real TinyLlama model."""

    def test_graph_mode_forward(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)
        ctx.set_use_graph(True)

        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = ctx.forward(ids)
        assert logits.shape[1] == 32000
        assert np.isfinite(logits).all()

    def test_graph_vs_imperative_tinyllama(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")

        ids = np.array([1, 2, 3], dtype=np.int32)

        # Imperative
        ctx_imp = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)
        ctx_imp.set_use_graph(False)
        logits_imp = ctx_imp.forward(ids)

        # Graph
        ctx_graph = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)
        ctx_graph.set_use_graph(True)
        logits_graph = ctx_graph.forward(ids)

        np.testing.assert_allclose(logits_imp, logits_graph, atol=1e-3,
                                   err_msg="Graph and imperative should match for TinyLlama")

    def test_graph_mode_generate_tinyllama(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        prompt = np.array([1, 450, 4996, 29901], dtype=np.int32)
        result = model.generate(prompt, max_new_tokens=10, do_sample=False,
                                gpu_layers=0, kv_cache_dtype="fp32")
        assert result["num_generated_tokens"] >= 1
