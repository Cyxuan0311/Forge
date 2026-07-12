import os
import sys
import pytest

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


@pytest.fixture
def loaded_model(model_path, model_config):
    model = nanoinfer.Model()
    model.load(model_path, **model_config)
    return model


@pytest.fixture
def loaded_context(loaded_model):
    ctx = loaded_model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
    return ctx
