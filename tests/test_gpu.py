import os
import sys
import pytest
import numpy as np

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge

CUDA_AVAILABLE = False
try:
    import ctypes

    cuda_rt = ctypes.CDLL("libcudart.so")
    device_count = ctypes.c_int(0)
    result = cuda_rt.cudaGetDeviceCount(ctypes.byref(device_count))
    if result == 0 and device_count.value > 0:
        CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False

skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device not available")


@pytest.fixture
def model_path():
    return os.path.join(os.path.dirname(__file__), "fixtures", "test_model_small.ninf")


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
    }


@skip_no_cuda
class TestGPUForward:
    def test_gpu_forward_shape(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = ctx.forward(ids)
        assert logits.shape == (3, model_config["vocab_size"])
        assert np.isfinite(logits).all()

    def test_gpu_forward_single_token(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        ids = np.array([5], dtype=np.int32)
        logits = ctx.forward(ids)
        assert logits.shape == (1, model_config["vocab_size"])
        assert np.isfinite(logits).all()

    def test_gpu_forward_determinism(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        ids = np.array([3, 7, 15], dtype=np.int32)
        ctx.reset_kv()
        out1 = ctx.forward(ids)
        ctx.reset_kv()
        out2 = ctx.forward(ids)
        np.testing.assert_array_equal(out1, out2)

    def test_gpu_forward_logits_range(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        logits = ctx.forward(ids)
        assert logits.max() < 1e6
        assert logits.min() > -1e6

    def test_gpu_forward_different_lengths(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        for sl in [1, 2, 3, 5, 8]:
            ids = np.arange(sl, dtype=np.int32) % 100
            logits = ctx.forward(ids)
            assert logits.shape[0] == sl
            assert logits.shape[1] == model_config["vocab_size"]


@skip_no_cuda
class TestGPUGenerate:
    def test_gpu_generate_basic(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 5
        cfg.do_sample = False
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = ctx.generate(prompt, cfg)
        assert hasattr(result, "token_ids")
        assert result.num_prompt_tokens == 3
        assert result.num_generated_tokens >= 1
        assert result.num_generated_tokens <= 5
        assert result.finished is True

    def test_gpu_generate_greedy_deterministic(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 10
        cfg.do_sample = False
        prompt = np.array([1, 2, 3], dtype=np.int32)
        r1 = ctx.generate(prompt, cfg)
        r2 = ctx.generate(prompt, cfg)
        assert list(r1.token_ids) == list(r2.token_ids)

    def test_gpu_generate_max_tokens_limit(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 3
        cfg.do_sample = False
        prompt = np.array([1, 2], dtype=np.int32)
        result = ctx.generate(prompt, cfg)
        assert result.num_generated_tokens <= 3
        assert result.finish_reason == "length"

    def test_gpu_generate_with_temperature(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 5
        cfg.temperature = 0.8
        cfg.do_sample = True
        cfg.seed = 42
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = ctx.generate(prompt, cfg)
        assert result.num_generated_tokens >= 1

    def test_gpu_generate_with_eos(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 10
        cfg.eos_token_id = 5
        cfg.do_sample = False
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = ctx.generate(prompt, cfg)
        assert result.finished is True
        if result.finish_reason == "eos":
            assert result.token_ids[-1] == 5

    def test_gpu_generate_token_ids_valid(self, model_path, model_config):
        model = forge.Model()
        model.load(model_path, device="cuda", **model_config)
        ctx = model.create_context()
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 5
        cfg.do_sample = False
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = ctx.generate(prompt, cfg)
        for tid in result.token_ids:
            assert 0 <= tid < model_config["vocab_size"]


@skip_no_cuda
class TestCPUGPUConsistency:
    def test_forward_logits_consistency(self, model_path, model_config):
        model_cpu = forge.Model()
        model_cpu.load(model_path, device="cpu", **model_config)
        model_gpu = forge.Model()
        model_gpu.load(model_path, device="cuda", **model_config)

        ctx_cpu = model_cpu.create_context()
        ctx_gpu = model_gpu.create_context()

        ids = np.array([1, 2, 3], dtype=np.int32)
        ctx_cpu.reset_kv()
        logits_cpu = ctx_cpu.forward(ids)
        ctx_gpu.reset_kv()
        logits_gpu = ctx_gpu.forward(ids)

        np.testing.assert_allclose(
            logits_cpu,
            logits_gpu,
            atol=2.0,
            err_msg="CPU and GPU forward logits should match within 2.0",
        )

    def test_forward_single_token_consistency(self, model_path, model_config):
        model_cpu = forge.Model()
        model_cpu.load(model_path, device="cpu", **model_config)
        model_gpu = forge.Model()
        model_gpu.load(model_path, device="cuda", **model_config)

        ctx_cpu = model_cpu.create_context()
        ctx_gpu = model_gpu.create_context()

        ids = np.array([5], dtype=np.int32)
        ctx_cpu.reset_kv()
        logits_cpu = ctx_cpu.forward(ids)
        ctx_gpu.reset_kv()
        logits_gpu = ctx_gpu.forward(ids)

        np.testing.assert_allclose(
            logits_cpu, logits_gpu, atol=0.5, err_msg="CPU and GPU single-token logits should match"
        )

    def test_generate_greedy_consistency(self, model_path, model_config):
        model_cpu = forge.Model()
        model_cpu.load(model_path, device="cpu", **model_config)
        model_gpu = forge.Model()
        model_gpu.load(model_path, device="cuda", **model_config)

        ctx_cpu = model_cpu.create_context()
        ctx_gpu = model_gpu.create_context()
        cfg_cpu = forge.GenerationConfig()
        cfg_cpu.max_new_tokens = 10
        cfg_cpu.do_sample = False
        cfg_gpu = forge.GenerationConfig()
        cfg_gpu.max_new_tokens = 10
        cfg_gpu.do_sample = False
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result_cpu = ctx_cpu.generate(prompt, cfg_cpu)
        result_gpu = ctx_gpu.generate(prompt, cfg_gpu)

        assert list(result_cpu.token_ids) == list(result_gpu.token_ids), (
            f"CPU tokens: {result_cpu.token_ids}, GPU tokens: {result_gpu.token_ids}"
        )

    def test_generate_longer_prompt_consistency(self, model_path, model_config):
        model_cpu = forge.Model()
        model_cpu.load(model_path, device="cpu", **model_config)
        model_gpu = forge.Model()
        model_gpu.load(model_path, device="cuda", **model_config)

        ctx_cpu = model_cpu.create_context()
        ctx_gpu = model_gpu.create_context()
        cfg_cpu = forge.GenerationConfig()
        cfg_cpu.max_new_tokens = 8
        cfg_cpu.do_sample = False
        cfg_gpu = forge.GenerationConfig()
        cfg_gpu.max_new_tokens = 8
        cfg_gpu.do_sample = False
        prompt = np.array([5, 10, 15, 20, 25], dtype=np.int32)
        result_cpu = ctx_cpu.generate(prompt, cfg_cpu)
        result_gpu = ctx_gpu.generate(prompt, cfg_gpu)

        assert list(result_cpu.token_ids) == list(result_gpu.token_ids), (
            f"CPU tokens: {result_cpu.token_ids}, GPU tokens: {result_gpu.token_ids}"
        )

    def test_forward_different_seq_lens_consistency(self, model_path, model_config):
        model_cpu = forge.Model()
        model_cpu.load(model_path, device="cpu", **model_config)
        model_gpu = forge.Model()
        model_gpu.load(model_path, device="cuda", **model_config)

        ctx_cpu = model_cpu.create_context()
        ctx_gpu = model_gpu.create_context()

        for sl in [1, 2, 4, 8]:
            ids = np.arange(sl, dtype=np.int32) % 100
            ctx_cpu.reset_kv()
            ctx_gpu.reset_kv()
            logits_cpu = ctx_cpu.forward(ids)
            logits_gpu = ctx_gpu.forward(ids)
            np.testing.assert_allclose(
                logits_cpu, logits_gpu, atol=0.5, err_msg=f"CPU/GPU mismatch at seq_len={sl}"
            )
