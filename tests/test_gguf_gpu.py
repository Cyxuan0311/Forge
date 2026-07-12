import os
import sys
import gc
import pytest
import numpy as np

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import nanoinfer

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

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TINYLLAMA_Q4_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")


def tinyllama_available():
    return os.path.exists(TINYLLAMA_Q4_PATH)


tinyllama_skip = pytest.mark.skipif(not tinyllama_available(), reason="TinyLlama GGUF model not found")


@pytest.fixture(scope="module")
def gpu_model():
    if not CUDA_AVAILABLE or not tinyllama_available():
        pytest.skip("CUDA or TinyLlama model not available")
    model = nanoinfer.Model()
    model.load_gguf(TINYLLAMA_Q4_PATH, device="cuda")
    yield model


@skip_no_cuda
@tinyllama_skip
class TestGGUFGPUInference:
    def test_gpu_forward_no_nan(self, gpu_model):
        ctx = gpu_model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ids = np.array([1, 450, 4996, 29901], dtype=np.int32)
        ctx.reset_kv()
        logits = ctx.forward(ids)
        assert not np.any(np.isnan(logits)), "GPU logits should not contain NaN"
        assert not np.any(np.isinf(logits)), "GPU logits should not contain Inf"

    def test_gpu_forward_shape(self, gpu_model):
        ctx = gpu_model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ids = np.array([1, 450, 4996, 29901], dtype=np.int32)
        ctx.reset_kv()
        logits = ctx.forward(ids)
        assert logits.shape == (4, 32000)

    def test_gpu_forward_logits_range(self, gpu_model):
        ctx = gpu_model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ids = np.array([1, 450, 4996, 29901], dtype=np.int32)
        ctx.reset_kv()
        logits = ctx.forward(ids)
        assert logits.max() < 100, "Logits should be in reasonable range"
        assert logits.min() > -100, "Logits should be in reasonable range"

    def test_gpu_deterministic(self, gpu_model):
        ctx = gpu_model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ids = np.array([1, 450, 4996, 29901], dtype=np.int32)
        ctx.reset_kv()
        out1 = ctx.forward(ids)
        ctx.reset_kv()
        out2 = ctx.forward(ids)
        np.testing.assert_allclose(out1, out2, atol=1e-2,
                                   err_msg="GPU forward should be deterministic within float precision")


@skip_no_cuda
@tinyllama_skip
class TestGGUFCPUGPUConsistency:
    def test_logits_correlation(self):
        ids = np.array([1, 450, 4996, 29901], dtype=np.int32)

        model_cpu = nanoinfer.Model()
        model_cpu.load_gguf(TINYLLAMA_Q4_PATH, device="cpu")
        ctx_cpu = model_cpu.create_context(kv_cache_dtype="fp32", gpu_layers=0)
        ctx_cpu.reset_kv()
        logits_cpu = ctx_cpu.forward(ids)
        del ctx_cpu
        del model_cpu
        gc.collect()

        model_gpu = nanoinfer.Model()
        model_gpu.load_gguf(TINYLLAMA_Q4_PATH, device="cuda")
        ctx_gpu = model_gpu.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ctx_gpu.reset_kv()
        logits_gpu = ctx_gpu.forward(ids)

        cpu_flat = logits_cpu[-1].flatten()
        gpu_flat = logits_gpu[-1].flatten()
        corr = np.corrcoef(cpu_flat, gpu_flat)[0, 1]
        assert corr > 0.90, f"CPU/GPU logit correlation should be > 0.90, got {corr}"

    def test_top_token_match(self):
        ids = np.array([1, 450, 4996, 29901], dtype=np.int32)

        model_cpu = nanoinfer.Model()
        model_cpu.load_gguf(TINYLLAMA_Q4_PATH, device="cpu")
        ctx_cpu = model_cpu.create_context(kv_cache_dtype="fp32", gpu_layers=0)
        ctx_cpu.reset_kv()
        logits_cpu = ctx_cpu.forward(ids)
        del ctx_cpu
        del model_cpu
        gc.collect()

        model_gpu = nanoinfer.Model()
        model_gpu.load_gguf(TINYLLAMA_Q4_PATH, device="cuda")
        ctx_gpu = model_gpu.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ctx_gpu.reset_kv()
        logits_gpu = ctx_gpu.forward(ids)

        assert np.argmax(logits_cpu[-1]) == np.argmax(logits_gpu[-1]), \
            "CPU and GPU should predict same top token"

    def test_logits_close(self):
        ids = np.array([1, 450, 4996, 29901], dtype=np.int32)

        model_cpu = nanoinfer.Model()
        model_cpu.load_gguf(TINYLLAMA_Q4_PATH, device="cpu")
        ctx_cpu = model_cpu.create_context(kv_cache_dtype="fp32", gpu_layers=0)
        ctx_cpu.reset_kv()
        logits_cpu = ctx_cpu.forward(ids)
        del ctx_cpu
        del model_cpu
        gc.collect()

        model_gpu = nanoinfer.Model()
        model_gpu.load_gguf(TINYLLAMA_Q4_PATH, device="cuda")
        ctx_gpu = model_gpu.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ctx_gpu.reset_kv()
        logits_gpu = ctx_gpu.forward(ids)

        corr = np.corrcoef(logits_cpu.flatten(), logits_gpu.flatten())[0, 1]
        assert corr > 0.90, f"CPU/GPU logit correlation should be > 0.90, got {corr}"

    def test_greedy_generation_match(self):
        prompt = np.array([1, 450, 4996, 29901], dtype=np.int32)

        model_cpu = nanoinfer.Model()
        model_cpu.load_gguf(TINYLLAMA_Q4_PATH, device="cpu")
        result_cpu = model_cpu.generate(prompt, max_new_tokens=10, do_sample=False,
                                        gpu_layers=0, kv_cache_dtype="fp32")
        del model_cpu
        gc.collect()

        model_gpu = nanoinfer.Model()
        model_gpu.load_gguf(TINYLLAMA_Q4_PATH, device="cuda")
        result_gpu = model_gpu.generate(prompt, max_new_tokens=10, do_sample=False,
                                        gpu_layers=-1, kv_cache_dtype="fp32")

        match = sum(1 for a, b in zip(result_cpu["token_ids"], result_gpu["token_ids"]) if a == b)
        assert match >= 2, \
            f"CPU/GPU should match at least 2/10 tokens, got {match}/10. CPU: {result_cpu['token_ids']}, GPU: {result_gpu['token_ids']}"


@skip_no_cuda
@tinyllama_skip
class TestGPULayers:
    def test_gpu_layers_negative_one(self, gpu_model):
        ctx = gpu_model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        ids = np.array([1, 2, 3], dtype=np.int32)
        ctx.reset_kv()
        logits = ctx.forward(ids)
        assert not np.any(np.isnan(logits))

    def test_gpu_layers_all(self, gpu_model):
        ctx = gpu_model.create_context(kv_cache_dtype="fp32", gpu_layers=22)
        ids = np.array([1, 2, 3], dtype=np.int32)
        ctx.reset_kv()
        logits = ctx.forward(ids)
        assert not np.any(np.isnan(logits))


@skip_no_cuda
@tinyllama_skip
class TestGGUFGeneration:
    def test_gpu_generate_no_repeat(self, gpu_model):
        prompt = np.array([1, 450, 4996, 29901], dtype=np.int32)
        result = gpu_model.generate(prompt, max_new_tokens=20, do_sample=False,
                                    gpu_layers=-1, kv_cache_dtype="fp32")
        tokens = result["token_ids"]
        unique_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
        assert unique_ratio > 0.3, \
            f"Generated tokens should not be highly repetitive, unique_ratio={unique_ratio}"

    def test_gpu_generate_streaming(self, gpu_model):
        prompt = np.array([1, 450, 4996, 29901], dtype=np.int32)

        collected_tokens = []

        def on_token(tid, step):
            collected_tokens.append(tid)

        gpu_model.generate_stream(
            prompt_ids=prompt,
            callback=on_token,
            max_new_tokens=10,
            temperature=0.0,
            top_k=1,
            do_sample=False,
            gpu_layers=-1,
            kv_cache_dtype="fp32"
        )
        assert len(collected_tokens) >= 1, "Should generate at least one token"

    def test_gpu_generate_with_sampling(self, gpu_model):
        prompt = np.array([1, 450, 4996, 29901], dtype=np.int32)
        result = gpu_model.generate(prompt, max_new_tokens=10, temperature=0.8,
                                    top_k=40, top_p=0.9, do_sample=True, seed=42,
                                    gpu_layers=-1, kv_cache_dtype="fp32")
        assert result["num_generated_tokens"] >= 1


@skip_no_cuda
@tinyllama_skip
class TestGGUFIncrementalInference:
    def test_incremental_matches_full_gpu(self, gpu_model):
        ctx = gpu_model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)

        ids = np.array([1, 450, 4996], dtype=np.int32)

        ctx.reset_kv()
        full_logits = ctx.forward(ids)

        ctx.reset_kv()
        incremental_outputs = []
        for i, tid in enumerate(ids):
            out = ctx.forward(np.array([tid], dtype=np.int32), start_pos=i)
            incremental_outputs.append(out[0])

        for i in range(len(ids)):
            np.testing.assert_allclose(full_logits[i], incremental_outputs[i], atol=0.05,
                                       err_msg=f"Token {i} mismatch between full and incremental on GPU")

    def test_incremental_deterministic_gpu(self, gpu_model):
        ctx = gpu_model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)

        ids = np.array([1, 450, 4996, 29901], dtype=np.int32)

        ctx.reset_kv()
        out1 = ctx.forward(ids)

        ctx.reset_kv()
        out2 = ctx.forward(ids)

        np.testing.assert_allclose(out1, out2, atol=1e-2,
                                   err_msg="GPU incremental inference should be deterministic within float precision")
