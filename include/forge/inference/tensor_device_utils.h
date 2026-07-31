#pragma once

// 张量 device 搬运的小工具。
//
// Gemma4 的多处标量循环需要先把张量拉到 CPU 再放回原 device, 原本在
// gemma4_engine.cpp 里以 file-local static 形式存在。拆分为多个 executor 之后
// 需要共享, 因此提取为 inline helper。

#include "forge/tensor.h"

namespace forge {

// 若张量在 CUDA 上, 拷贝到 CPU 后返回副本; 否则原样返回。
inline TensorPtr ensure_cpu(const TensorPtr& t) {
    if (!t || t->device() == DeviceType::CPU) return t;
    auto cpu = std::make_shared<Tensor>(t->dtype(), t->shape(), DeviceType::CPU);
    cpu->copy_from(*t);
    return cpu;
}

// 把张量搬回目标 device。已在目标 device 上时原样返回。
inline TensorPtr restore_device(const TensorPtr& t, DeviceType target) {
    if (!t || t->device() == target) return t;
    auto on_dev = std::make_shared<Tensor>(t->dtype(), t->shape(), target);
    on_dev->copy_from(*t);
    return on_dev;
}

}  // namespace forge
