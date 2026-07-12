import numpy as np
import forge


class TestTensorCreation:
    def test_create_fp32_cpu(self):
        t = forge.Tensor(forge.DataType.FP32, [3, 4], forge.DeviceType.CPU)
        assert t.shape() == [3, 4]
        assert t.numel() == 12
        assert t.dtype() == forge.DataType.FP32
        assert t.device() == forge.DeviceType.CPU

    def test_zero(self):
        t = forge.Tensor(forge.DataType.FP32, [2, 3], forge.DeviceType.CPU)
        t.zero_()
        arr = t.numpy()
        assert np.all(arr == 0)

    def test_numpy_roundtrip(self):
        t = forge.Tensor(forge.DataType.FP32, [2, 5], forge.DeviceType.CPU)
        arr = t.numpy()
        assert arr.shape == (2, 5)
        assert arr.dtype == np.float32

    def test_nbytes(self):
        t = forge.Tensor(forge.DataType.FP32, [4, 8], forge.DeviceType.CPU)
        assert t.nbytes() == 4 * 8 * 4

    def test_1d_tensor(self):
        t = forge.Tensor(forge.DataType.FP32, [10], forge.DeviceType.CPU)
        assert t.shape() == [10]
        assert t.numel() == 10

    def test_scalar_like_tensor(self):
        t = forge.Tensor(forge.DataType.FP32, [1], forge.DeviceType.CPU)
        assert t.numel() == 1


class TestTensorDevice:
    def test_to_device_cpu_to_cpu(self):
        t = forge.Tensor(forge.DataType.FP32, [2, 3], forge.DeviceType.CPU)
        t.to_device(forge.DeviceType.CPU)
        assert t.device() == forge.DeviceType.CPU
