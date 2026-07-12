import numpy as np
import pytest
import os
import sys
import struct

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestNinfFormatDetection:
    def test_detect_ninf_format(self):
        path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
        if not os.path.exists(path):
            pytest.skip("ninf test model not found")
        fmt = forge.Model.detect_format(path)
        assert fmt == "ninf"

    def test_detect_invalid_format(self):
        fmt = forge.Model.detect_format("/nonexistent/file.bin")
        assert fmt == ""

    def test_detect_non_model_file(self, tmp_path):
        dummy = tmp_path / "dummy.txt"
        dummy.write_text("hello world")
        fmt = forge.Model.detect_format(str(dummy))
        assert fmt == ""


class TestNinfModelLoading:
    def test_load_ninf_with_config(self):
        path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
        if not os.path.exists(path):
            pytest.skip("ninf test model not found")
        model = forge.Model()
        model.load(path,
                   vocab_size=100, hidden_dim=32, intermediate_dim=64,
                   num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                   device="cpu")
        assert model.config.vocab_size == 100
        assert model.config.hidden_dim == 32
        assert model.config.num_layers == 1

    def test_load_ninf_auto(self):
        path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
        if not os.path.exists(path):
            pytest.skip("ninf test model not found")
        model = forge.Model()
        model.load_auto(path, device="cpu")
        cfg = model.config
        assert cfg.vocab_size == 100
        assert cfg.hidden_dim == 32
        assert cfg.num_layers == 1
        assert cfg.num_heads == 2
        assert cfg.num_kv_heads == 1
        assert cfg.head_dim == 16

    def test_ninf_forward_after_load(self):
        path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
        if not os.path.exists(path):
            pytest.skip("ninf test model not found")
        model = forge.Model()
        model.load_auto(path, device="cpu")
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = ctx.forward(ids)
        assert logits.shape == (3, 100)
        assert np.isfinite(logits).all()


class TestGGUFFormatDetection:
    def test_detect_gguf_magic(self, tmp_path):
        gguf_path = tmp_path / "test.gguf"
        with open(str(gguf_path), "wb") as f:
            f.write(struct.pack("<I", 0x46554747))
            f.write(b"\x00" * 100)
        fmt = forge.Model.detect_format(str(gguf_path))
        assert fmt == "gguf"

    def test_load_gguf_nonexistent(self):
        model = forge.Model()
        with pytest.raises(RuntimeError):
            model.load_gguf("/nonexistent/model.gguf", device="cpu")


class TestModelConfig:
    def test_config_defaults(self):
        cfg = forge.ModelConfig()
        assert cfg.vocab_size == 0
        assert cfg.hidden_dim == 0

    def test_config_modify(self):
        cfg = forge.ModelConfig()
        cfg.vocab_size = 32000
        cfg.hidden_dim = 4096
        cfg.num_layers = 32
        cfg.num_heads = 32
        cfg.num_kv_heads = 32
        cfg.head_dim = 128
        cfg.arch_type = "llama"
        assert cfg.vocab_size == 32000
        assert cfg.arch_type == "llama"

    def test_config_from_loaded_model(self):
        path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
        if not os.path.exists(path):
            pytest.skip("ninf test model not found")
        model = forge.Model()
        model.load_auto(path, device="cpu")
        cfg = model.config
        assert cfg.vocab_size == 100
        assert cfg.hidden_dim == 32
        assert cfg.intermediate_dim == 64
        assert cfg.num_layers == 1
        assert cfg.num_heads == 2
        assert cfg.num_kv_heads == 1
        assert cfg.head_dim == 16
        assert cfg.arch_type == "llama"


class TestModelDevice:
    def test_model_cpu_device(self):
        path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
        if not os.path.exists(path):
            pytest.skip("ninf test model not found")
        model = forge.Model()
        model.load_auto(path, device="cpu")
        assert model.device == forge.DeviceType.CPU

    def test_context_cpu_device(self):
        path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
        if not os.path.exists(path):
            pytest.skip("ninf test model not found")
        model = forge.Model()
        model.load_auto(path, device="cpu")
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        assert ctx.device == forge.DeviceType.CPU


class TestModelErrors:
    def test_load_nonexistent_path(self):
        model = forge.Model()
        with pytest.raises(RuntimeError):
            model.load("/nonexistent/model.ninf",
                       vocab_size=100, hidden_dim=32, intermediate_dim=64,
                       num_layers=1, num_heads=2, device="cpu")

    def test_forward_without_load(self):
        model = forge.Model()
        with pytest.raises(Exception):
            ctx = model.create_context()


class TestModelCreateContext:
    def test_create_context_fp32(self):
        path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
        if not os.path.exists(path):
            pytest.skip("ninf test model not found")
        model = forge.Model()
        model.load_auto(path, device="cpu")
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        assert ctx is not None

    def test_create_context_q4_0(self):
        path = os.path.join(FIXTURES_DIR, "test_model_small.ninf")
        if not os.path.exists(path):
            pytest.skip("ninf test model not found")
        model = forge.Model()
        model.load_auto(path, device="cpu")
        ctx = model.create_context(kv_cache_dtype="q4_0", gpu_layers=-1)
        assert ctx is not None
