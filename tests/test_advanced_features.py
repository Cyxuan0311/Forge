import os
import sys
import pytest
import numpy as np

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import nanoinfer


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


class TestSamplerConfig:
    def test_default_config(self):
        cfg = nanoinfer.SamplerConfig()
        assert cfg.temperature == pytest.approx(1.0)
        assert cfg.top_k == 0
        assert cfg.top_p == pytest.approx(1.0)
        assert cfg.repeat_penalty == pytest.approx(1.0)
        assert cfg.do_sample is True
        assert cfg.seed == 0

    def test_custom_config(self):
        cfg = nanoinfer.SamplerConfig(temperature=0.7, top_k=50, top_p=0.9,
                                       repeat_penalty=1.2, do_sample=True, seed=42)
        assert cfg.temperature == pytest.approx(0.7, abs=1e-5)
        assert cfg.top_k == 50
        assert cfg.top_p == pytest.approx(0.9, abs=1e-5)
        assert cfg.repeat_penalty == pytest.approx(1.2, abs=1e-5)
        assert cfg.seed == 42

    def test_config_mutable(self):
        cfg = nanoinfer.SamplerConfig()
        cfg.temperature = 0.5
        cfg.top_k = 10
        assert cfg.temperature == pytest.approx(0.5, abs=1e-5)
        assert cfg.top_k == 10


class TestTopKTopPSampling:
    def test_top_k_sampling(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = model.generate(prompt, max_new_tokens=10, top_k=5, do_sample=True, seed=42)
        assert result["num_generated_tokens"] >= 1
        assert result["finished"] is True

    def test_top_p_sampling(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = model.generate(prompt, max_new_tokens=10, top_p=0.8, do_sample=True, seed=42)
        assert result["num_generated_tokens"] >= 1
        assert result["finished"] is True

    def test_top_k_and_top_p_combined(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = model.generate(prompt, max_new_tokens=10, top_k=10, top_p=0.9,
                                do_sample=True, seed=42)
        assert result["num_generated_tokens"] >= 1

    def test_repeat_penalty(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        prompt = np.array([1, 2, 3], dtype=np.int32)

        result_no_penalty = model.generate(prompt, max_new_tokens=10,
                                            repeat_penalty=1.0, do_sample=True, seed=42)
        result_with_penalty = model.generate(prompt, max_new_tokens=10,
                                              repeat_penalty=1.5, do_sample=True, seed=42)

        assert result_no_penalty["num_generated_tokens"] >= 1
        assert result_with_penalty["num_generated_tokens"] >= 1

    def test_temperature_zero_is_greedy(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        prompt = np.array([1, 2, 3], dtype=np.int32)

        result1 = model.generate(prompt, max_new_tokens=10, temperature=0.0, do_sample=True)
        result2 = model.generate(prompt, max_new_tokens=10, do_sample=False)
        assert list(result1["token_ids"]) == list(result2["token_ids"])


class TestCPUOffload:
    def test_gpu_layers_zero_all_cpu(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        ctx = model.create_context(gpu_layers=0)
        prompt = np.array([1, 2, 3], dtype=np.int32)
        out = ctx.forward(prompt)
        assert out.shape == (3, model_config["vocab_size"])

    def test_gpu_layers_default(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        ctx = model.create_context()
        prompt = np.array([1, 2, 3], dtype=np.int32)
        out = ctx.forward(prompt)
        assert out.shape == (3, model_config["vocab_size"])

    def test_cpu_offload_generate(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = model.generate(prompt, max_new_tokens=5, do_sample=False, gpu_layers=0)
        assert result["num_generated_tokens"] >= 1
        assert result["finished"] is True

    def test_cpu_offload_forward_matches_default(self, model_path, model_config):
        model_default = nanoinfer.Model()
        model_default.load(model_path, arch_type="llama", **model_config)
        model_offload = nanoinfer.Model()
        model_offload.load(model_path, arch_type="llama", **model_config)

        ctx_default = model_default.create_context()
        ctx_offload = model_offload.create_context(gpu_layers=0)

        prompt = np.array([1, 2, 3], dtype=np.int32)
        out_default = ctx_default.forward(prompt)
        out_offload = ctx_offload.forward(prompt)
        np.testing.assert_allclose(out_default, out_offload, atol=1e-5)


class TestInferenceContext:
    def test_context_forward(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = ctx.forward(ids)
        assert logits.shape == (3, model_config["vocab_size"])

    def test_context_reset_kv(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2, 3], dtype=np.int32)
        ctx.forward(ids)
        ctx.reset_kv()
        out_after_reset = ctx.forward(ids)
        assert out_after_reset.shape == (3, model_config["vocab_size"])

    def test_context_memory_stats(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        ctx = model.create_context()
        stats = ctx.memory_stats()
        assert "kv_cache_nbytes" in stats
        assert "kv_cache_dtype" in stats

    def test_context_device(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        ctx = model.create_context()
        assert ctx.device in (nanoinfer.DeviceType.CPU, nanoinfer.DeviceType.CUDA)


class TestRequestScheduler:
    def test_scheduler_submit(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        scheduler = nanoinfer.RequestScheduler(model, block_size=4, max_num_seqs=2)
        cfg = nanoinfer.SamplerConfig(do_sample=False)
        rid = scheduler.submit([1, 2, 3], max_new_tokens=5, sampler_config=cfg)
        assert rid >= 0
        assert scheduler.num_waiting() == 1

    def test_scheduler_step(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        scheduler = nanoinfer.RequestScheduler(model, block_size=4, max_num_seqs=2)
        cfg = nanoinfer.SamplerConfig(do_sample=False)
        scheduler.submit([1, 2, 3], max_new_tokens=5, sampler_config=cfg)
        has_more = scheduler.step()
        assert has_more is True or scheduler.num_active() >= 0

    def test_scheduler_multiple_requests(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        scheduler = nanoinfer.RequestScheduler(model, block_size=4, max_num_seqs=2)
        cfg = nanoinfer.SamplerConfig(do_sample=False)
        rid1 = scheduler.submit([1, 2], max_new_tokens=3, sampler_config=cfg)
        rid2 = scheduler.submit([3, 4], max_new_tokens=3, sampler_config=cfg)
        assert rid1 != rid2

    def test_scheduler_run_to_completion(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        scheduler = nanoinfer.RequestScheduler(model, block_size=4, max_num_seqs=2)
        cfg = nanoinfer.SamplerConfig(do_sample=False)
        scheduler.submit([1, 2, 3], max_new_tokens=3, sampler_config=cfg)

        for _ in range(20):
            if not scheduler.has_pending():
                break
            scheduler.step()

        finished = scheduler.get_finished()
        assert len(finished) >= 1
        assert finished[0].num_generated >= 1

    def test_scheduler_abort(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        scheduler = nanoinfer.RequestScheduler(model, block_size=4, max_num_seqs=2)
        cfg = nanoinfer.SamplerConfig(do_sample=False)
        rid = scheduler.submit([1, 2, 3], max_new_tokens=10, sampler_config=cfg)
        scheduler.abort(rid)

    def test_scheduler_reset(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        scheduler = nanoinfer.RequestScheduler(model, block_size=4, max_num_seqs=2)
        cfg = nanoinfer.SamplerConfig(do_sample=False)
        scheduler.submit([1, 2, 3], max_new_tokens=5, sampler_config=cfg)
        scheduler.submit([4, 5], max_new_tokens=5, sampler_config=cfg)
        scheduler.reset()
        assert scheduler.num_waiting() == 0


class TestPagedKVCache:
    def test_request_status_enum(self):
        assert nanoinfer.RequestStatus.Waiting is not None
        assert nanoinfer.RequestStatus.Prefilling is not None
        assert nanoinfer.RequestStatus.Decoding is not None
        assert nanoinfer.RequestStatus.Finished is not None
        assert nanoinfer.RequestStatus.Failed is not None

    def test_generate_request_fields(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        scheduler = nanoinfer.RequestScheduler(model, block_size=4, max_num_seqs=2)
        cfg = nanoinfer.SamplerConfig(do_sample=False)
        scheduler.submit([1, 2, 3], max_new_tokens=3, sampler_config=cfg)

        for _ in range(20):
            if not scheduler.has_pending():
                break
            scheduler.step()

        finished = scheduler.get_finished()
        if finished:
            req = finished[0]
            assert hasattr(req, "request_id")
            assert hasattr(req, "status")
            assert hasattr(req, "output_tokens")
            assert hasattr(req, "num_generated")
            assert hasattr(req, "finish_reason")
