"""Tests for the BackendManager and Backend abstraction."""
import os
import sys
import pytest

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge


class TestBackendManager:
    def test_instance_exists(self):
        mgr = forge.BackendManager.instance()
        assert mgr is not None

    def test_singleton(self):
        mgr1 = forge.BackendManager.instance()
        mgr2 = forge.BackendManager.instance()
        assert mgr1 is mgr2

    def test_cpu_backend_available(self):
        mgr = forge.BackendManager.instance()
        assert mgr.has_backend("cpu")

    def test_available_backends_not_empty(self):
        mgr = forge.BackendManager.instance()
        backends = mgr.available_backends()
        assert len(backends) > 0
        assert "cpu" in backends

    def test_backend_info_list(self):
        mgr = forge.BackendManager.instance()
        info_list = mgr.backend_info_list()
        assert len(info_list) > 0
        # Check first info has expected fields
        info = info_list[0]
        assert isinstance(info.name, str)
        assert info.available is True

    def test_get_cpu_backend(self):
        mgr = forge.BackendManager.instance()
        cpu = mgr.get_backend("cpu")
        assert cpu is not None
        assert cpu.name() == "cpu"

    def test_get_nonexistent_backend(self):
        mgr = forge.BackendManager.instance()
        backend = mgr.get_backend("nonexistent_backend")
        assert backend is None

    def test_has_cuda(self):
        mgr = forge.BackendManager.instance()
        # has_cuda should return a boolean
        result = mgr.has_cuda()
        assert isinstance(result, bool)

    def test_cuda_device_count(self):
        mgr = forge.BackendManager.instance()
        count = mgr.cuda_device_count()
        assert isinstance(count, int)
        assert count >= 0


class TestCPUBackend:
    @pytest.fixture(autouse=True)
    def setup(self):
        mgr = forge.BackendManager.instance()
        self.backend = mgr.get_backend("cpu")
        if self.backend is None:
            pytest.skip("CPU backend not available")

    def test_name(self):
        assert self.backend.name() == "cpu"

    def test_device_type(self):
        dt = self.backend.device_type()
        # DeviceType should be accessible
        assert dt is not None

    def test_capabilities_fp32(self):
        caps = self.backend.capabilities()
        assert self.backend.supports(forge.BackendCapability.FP32)

    def test_capabilities_quantized(self):
        # CPU backend should support quantized operations
        assert self.backend.supports(forge.BackendCapability.Quantized)

    def test_capabilities_unified_memory(self):
        # CPU has unified memory
        assert self.backend.supports(forge.BackendCapability.UnifiedMemory)

    def test_device_memory_total(self):
        total = self.backend.device_memory_total()
        assert isinstance(total, int)
        assert total > 0

    def test_device_memory_free(self):
        free = self.backend.device_memory_free()
        assert isinstance(free, int)
        assert free >= 0

    def test_device_id(self):
        did = self.backend.device_id()
        assert isinstance(did, int)
        assert did == 0


class TestBackendCapability:
    def test_capability_enum_values(self):
        # Note: BackendCapability.None is a Python keyword, test via integer comparison
        assert forge.BackendCapability.FP32 is not None
        assert forge.BackendCapability.FP16 is not None
        assert forge.BackendCapability.INT8 is not None
        assert forge.BackendCapability.Quantized is not None
        assert forge.BackendCapability.UnifiedMemory is not None
        assert forge.BackendCapability.StreamAsync is not None

    def test_capability_or_operation(self):
        combined = forge.BackendCapability.FP32 | forge.BackendCapability.FP16
        assert combined is not None

    def test_capability_and_operation(self):
        combined = forge.BackendCapability.FP32 | forge.BackendCapability.FP16
        result = combined & forge.BackendCapability.FP32
        # Result should be truthy (contains FP32)
        assert result


@pytest.mark.skipif(
    not forge.BackendManager.instance().has_cuda(),
    reason="CUDA not available"
)
class TestCUDABackend:
    @pytest.fixture(autouse=True)
    def setup(self):
        mgr = forge.BackendManager.instance()
        self.backend = mgr.get_backend("cuda", 0)
        if self.backend is None:
            pytest.skip("CUDA backend not available")

    def test_name(self):
        assert "cuda" in self.backend.name()

    def test_capabilities_fp32(self):
        assert self.backend.supports(forge.BackendCapability.FP32)

    def test_capabilities_fp16(self):
        assert self.backend.supports(forge.BackendCapability.FP16)

    def test_device_memory_total(self):
        total = self.backend.device_memory_total()
        assert isinstance(total, int)
        assert total > 0

    def test_device_memory_free(self):
        free = self.backend.device_memory_free()
        assert isinstance(free, int)
        assert free >= 0

    def test_device_id(self):
        did = self.backend.device_id()
        assert isinstance(did, int)
        assert did == 0

    def test_cuda_backend_in_available_list(self):
        mgr = forge.BackendManager.instance()
        backends = mgr.available_backends()
        assert "cuda" in backends
