#pragma once

#include "nanoinfer/tensor.h"

namespace nanoinfer {
namespace dispatch {

enum class Backend { CPU, CUDA };

inline Backend resolve(DeviceType device) {
    switch (device) {
        case DeviceType::CUDA:
#ifdef USE_CUDA
            return Backend::CUDA;
#else
            return Backend::CPU;
#endif
        default:
            return Backend::CPU;
    }
}

template <typename CpuFn, typename GpuFn>
inline auto execute(CpuFn&& cpu_fn, GpuFn&& gpu_fn)
    -> decltype(cpu_fn())
{
    return cpu_fn();
}

#ifdef USE_CUDA
template <typename CpuFn, typename GpuFn>
inline auto execute(DeviceType device, CpuFn&& cpu_fn, GpuFn&& gpu_fn)
    -> decltype(cpu_fn())
{
    if (device == DeviceType::CUDA) {
        return gpu_fn();
    }
    return cpu_fn();
}

template <typename CpuFn, typename GpuFn>
inline auto execute(const TensorPtr& tensor, CpuFn&& cpu_fn, GpuFn&& gpu_fn)
    -> decltype(cpu_fn())
{
    return execute(tensor->device(), std::forward<CpuFn>(cpu_fn),
                   std::forward<GpuFn>(gpu_fn));
}
#else
template <typename CpuFn, typename GpuFn>
inline auto execute(DeviceType device, CpuFn&& cpu_fn, GpuFn&&)
    -> decltype(cpu_fn())
{
    (void)device;
    return cpu_fn();
}

template <typename CpuFn, typename GpuFn>
inline auto execute(const TensorPtr&, CpuFn&& cpu_fn, GpuFn&&)
    -> decltype(cpu_fn())
{
    return cpu_fn();
}
#endif

} // namespace dispatch
} // namespace nanoinfer
