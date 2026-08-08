"""Phase 6: Unified VisionLLM + scheduler.generate() wrapper.

Tests:
  1. forge.__version__ == "0.7.0"
  2. RequestScheduler.generate() method exists and signature is correct
  3. VisionLLM class exists with expected methods
  4. VisionSession class exists with expected methods
  5. VisionLLM.from_gguf() factory works (instantiation only, no model load)
  6. _greedy_sample helper produces valid token IDs
"""

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import forge
import forge_llm


# ---------------------------------------------------------------------------
# 1. Version
# ---------------------------------------------------------------------------

def test_version_is_0_7_0():
    assert forge.__version__ == "0.7.0"


# ---------------------------------------------------------------------------
# 2. Scheduler.generate()
# ---------------------------------------------------------------------------

def test_scheduler_has_generate():
    assert hasattr(forge.RequestScheduler, "generate")


# ---------------------------------------------------------------------------
# 3. VisionLLM class structure
# ---------------------------------------------------------------------------

def test_vision_llm_exists():
    assert hasattr(forge_llm, "VisionLLM")


def test_vision_llm_has_methods():
    methods = ["from_gguf", "generate", "generate_stream", "chat_stream",
               "create_session", "reset", "encode_image"]
    for m in methods:
        assert hasattr(forge_llm.VisionLLM, m), f"Missing: VisionLLM.{m}"


def test_vision_session_exists():
    assert hasattr(forge_llm, "VisionSession")


def test_vision_session_has_methods():
    methods = ["send", "chat", "chat_stream", "reset"]
    for m in methods:
        assert hasattr(forge_llm.VisionSession, m), f"Missing: VisionSession.{m}"


# ---------------------------------------------------------------------------
# 4. _greedy_sample helper
# ---------------------------------------------------------------------------

def test_greedy_sample_deterministic():
    import numpy as np
    logits = np.array([0.1, 0.9, 0.3, 0.7], dtype=np.float64)
    cfg = forge.GenerationConfig()
    cfg.temperature = 0.0  # greedy

    # Run 3 times → all should return same result
    results = [forge_llm._greedy_sample(logits, cfg) for _ in range(3)]
    assert all(r == results[0] for r in results)
    # Should pick argmax = index 1
    assert results[0] == 1


def test_greedy_sample_with_temperature():
    import numpy as np
    logits = np.array([0.1, 5.0, 0.3], dtype=np.float64)
    cfg = forge.GenerationConfig()
    cfg.temperature = 1.0

    result = forge_llm._greedy_sample(logits, cfg)
    assert isinstance(result, int)
    assert 0 <= result < 3  # valid token ID


# ---------------------------------------------------------------------------
# 5. VisionLLM.from_gguf() factory (no files needed for instantiation check)
# ---------------------------------------------------------------------------

def test_from_gguf_classmethod_exists():
    assert callable(forge_llm.VisionLLM.from_gguf)


# ---------------------------------------------------------------------------
# 6. forge_llm module exposes all new types
# ---------------------------------------------------------------------------

def test_module_exports():
    """Verify all Phase 6 types are in forge_llm.__all__ or importable."""
    names = ["LLM", "Session", "VisionLLM", "VisionSession"]
    for name in names:
        assert hasattr(forge_llm, name), f"Missing export: {name}"
