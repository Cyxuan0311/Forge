#include "forge/backend.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "forge/logger.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

BackendStream::~BackendStream() {
    if (backend && handle) {
        backend->destroy_stream(*this);
    }
}

void BackendStream::synchronize() {
    if (backend) {
        backend->synchronize_stream(*this);
    }
}

// ============================================================================
// CPUBackend
// ============================================================================
class CPUBackend : public Backend {
public:
    DeviceType device_type() const override { return DeviceType::CPU; }
    std::string name() const override { return "cpu"; }

    void* allocate(size_t size) override {
        void* ptr = std::malloc(size);
        if (!ptr && size > 0) {
            LOG_ERROR("CPU malloc failed for " + std::to_string(size) + " bytes");
        }
        return ptr;
    }

    void deallocate(void* ptr, size_t) override {
        if (ptr)
            std::free(ptr);
    }

    void copy(void* dst, const void* src, size_t size, CopyKind) override {
        if (dst && src && size > 0)
            std::memcpy(dst, src, size);
    }

    void memset(void* ptr, int value, size_t size) override {
        if (ptr && size > 0)
            std::memset(ptr, value, size);
    }

    void synchronize() override {}

    BackendCapability capabilities() const override {
        return BackendCapability::FP32 | BackendCapability::FP16 | BackendCapability::Quantized |
               BackendCapability::UnifiedMemory;
    }

    size_t device_memory_total() const override {
        // Return a reasonable estimate for system RAM
        return static_cast<size_t>(16) * 1024 * 1024 * 1024;  // 16 GB placeholder
    }

    size_t device_memory_free() const override {
        // On CPU, we don't track free memory precisely
        return device_memory_total() / 2;  // placeholder
    }
};

// ============================================================================
// CUDABackend
// ============================================================================
#ifdef USE_CUDA
class CUDABackend : public Backend {
public:
    explicit CUDABackend(int device_id = 0) : device_id_(device_id) { cudaSetDevice(device_id); }

    DeviceType device_type() const override { return DeviceType::CUDA; }
    std::string name() const override { return "cuda:" + std::to_string(device_id_); }

    void* allocate(size_t size) override {
        void* ptr = nullptr;
        cudaSetDevice(device_id_);
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
            return nullptr;
        }
        return ptr;
    }

    void deallocate(void* ptr, size_t) override {
        if (ptr) {
            cudaSetDevice(device_id_);
            cudaFree(ptr);
        }
    }

    void copy(void* dst, const void* src, size_t size, CopyKind kind) override {
        cudaSetDevice(device_id_);
        cudaMemcpyKind cuda_kind;
        switch (kind) {
        case CopyKind::HostToHost:
            cuda_kind = cudaMemcpyHostToHost;
            break;
        case CopyKind::HostToDevice:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case CopyKind::DeviceToHost:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case CopyKind::DeviceToDevice:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            cuda_kind = cudaMemcpyDefault;
            break;
        }
        cudaMemcpy(dst, src, size, cuda_kind);
    }

    void memset(void* ptr, int value, size_t size) override {
        if (ptr && size > 0) {
            cudaSetDevice(device_id_);
            cudaMemset(ptr, value, size);
        }
    }

    void synchronize() override {
        cudaSetDevice(device_id_);
        cudaDeviceSynchronize();
    }

    BackendStream create_stream() override {
        BackendStream stream;
        stream.backend = this;
        cudaStream_t cuda_stream;
        cudaSetDevice(device_id_);
        cudaStreamCreate(&cuda_stream);
        stream.handle = cuda_stream;
        return stream;
    }

    void synchronize_stream(BackendStream& stream) override {
        if (stream.handle) {
            cudaSetDevice(device_id_);
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream.handle));
        }
    }

    void destroy_stream(BackendStream& stream) override {
        if (stream.handle) {
            cudaSetDevice(device_id_);
            cudaStreamDestroy(static_cast<cudaStream_t>(stream.handle));
            stream.handle = nullptr;
        }
    }

    BackendCapability capabilities() const override {
        return BackendCapability::FP32 | BackendCapability::FP16 | BackendCapability::INT8 |
               BackendCapability::Quantized | BackendCapability::StreamAsync;
    }

    size_t device_memory_total() const override {
        cudaSetDevice(device_id_);
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        return total_mem;
    }

    size_t device_memory_free() const override {
        cudaSetDevice(device_id_);
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        return free_mem;
    }

    int device_id() const override { return device_id_; }

private:
    int device_id_ = 0;
};
#endif

// ============================================================================
// Backend static factory methods
// ============================================================================
BackendStream Backend::create_stream() {
    return {};
}
void Backend::synchronize_stream(BackendStream&) {}
void Backend::destroy_stream(BackendStream&) {}

std::shared_ptr<Backend> Backend::create(DeviceType device) {
    if (device == DeviceType::CPU)
        return create_cpu();
    return create_cuda();
}

std::shared_ptr<Backend> Backend::create_cpu() {
    return std::make_shared<CPUBackend>();
}

std::shared_ptr<Backend> Backend::create_cuda(int device_id) {
#ifdef USE_CUDA
    return std::make_shared<CUDABackend>(device_id);
#else
    (void)device_id;
    LOG_ERROR("CUDA not available, falling back to CPU");
    return create_cpu();
#endif
}

// ============================================================================
// BackendManager
// ============================================================================
BackendManager::BackendManager() {
    // Register built-in backends
    register_backend("cpu", DeviceType::CPU,
                     [](int) -> std::shared_ptr<Backend> { return Backend::create_cpu(); });

#ifdef USE_CUDA
    register_backend("cuda", DeviceType::CUDA, [](int device_id) -> std::shared_ptr<Backend> {
        return Backend::create_cuda(device_id);
    });
#endif
}

BackendManager& BackendManager::instance() {
    static BackendManager mgr;
    return mgr;
}

std::shared_ptr<Backend> BackendManager::get_backend(DeviceType device) {
    if (device == DeviceType::CPU)
        return get_cpu_backend();
    return get_cuda_backend();
}

std::shared_ptr<Backend> BackendManager::get_cpu_backend() {
    if (!cpu_backend_) {
        cpu_backend_ = Backend::create_cpu();
    }
    return cpu_backend_;
}

std::shared_ptr<Backend> BackendManager::get_cuda_backend(int device_id) {
#ifdef USE_CUDA
    auto it = cuda_backends_.find(device_id);
    if (it != cuda_backends_.end())
        return it->second;
    auto backend = Backend::create_cuda(device_id);
    cuda_backends_[device_id] = backend;
    return backend;
#else
    (void)device_id;
    return get_cpu_backend();
#endif
}

void BackendManager::register_backend(const std::string& name, DeviceType device_type,
                                      BackendFactory factory) {
    BackendInfo info;
    info.name = name;
    info.device_type = device_type;
    info.factory = std::move(factory);
    info.available = true;

    // Check if the backend is actually usable
    if (device_type == DeviceType::CUDA) {
#ifndef USE_CUDA
        info.available = false;
#endif
    }

    registry_[name] = std::move(info);
    LOG_INFO("Registered backend: " + name +
             " (available=" + (registry_[name].available ? "true" : "false") + ")");
}

void BackendManager::unregister_backend(const std::string& name) {
    registry_.erase(name);
    cached_backends_.erase(name);
}

std::shared_ptr<Backend> BackendManager::get_backend(const std::string& name, int device_id) {
    // Check cache first
    std::string cache_key = name + ":" + std::to_string(device_id);
    auto it = cached_backends_.find(cache_key);
    if (it != cached_backends_.end())
        return it->second;

    // Look up in registry
    auto reg_it = registry_.find(name);
    if (reg_it == registry_.end()) {
        LOG_WARN("Backend not found: " + name);
        return nullptr;
    }

    if (!reg_it->second.available) {
        LOG_WARN("Backend not available: " + name);
        return nullptr;
    }

    auto backend = reg_it->second.factory(device_id);
    cached_backends_[cache_key] = backend;
    return backend;
}

std::vector<std::string> BackendManager::available_backends() const {
    std::vector<std::string> result;
    for (const auto& [name, info] : registry_) {
        if (info.available) {
            result.push_back(name);
        }
    }
    return result;
}

std::vector<BackendInfo> BackendManager::backend_info_list() const {
    std::vector<BackendInfo> result;
    for (const auto& [_, info] : registry_) {
        result.push_back(info);
    }
    return result;
}

bool BackendManager::has_backend(const std::string& name) const {
    auto it = registry_.find(name);
    return it != registry_.end() && it->second.available;
}

void BackendManager::register_backend(DeviceType device, std::shared_ptr<Backend> backend) {
    if (device == DeviceType::CPU) {
        cpu_backend_ = std::move(backend);
    } else {
        cuda_backends_[0] = std::move(backend);
    }
}

bool BackendManager::has_cuda() const {
#ifdef USE_CUDA
    int count = 0;
    cudaGetDeviceCount(&count);
    return count > 0;
#else
    return false;
#endif
}

int BackendManager::cuda_device_count() const {
#ifdef USE_CUDA
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
#else
    return 0;
#endif
}

std::vector<DeviceInfo> BackendManager::available_devices() {
    std::vector<DeviceInfo> devices;

    // CPU (lazy init if needed)
    auto cpu = get_cpu_backend();
    if (cpu) {
        DeviceInfo cpu_info;
        cpu_info.name = "cpu";
        cpu_info.type = DeviceType::CPU;
        cpu_info.device_id = -1;
        cpu_info.backend = cpu_backend_;
        cpu_info.memory_total = cpu_backend_->device_memory_total();
        cpu_info.memory_free = cpu_backend_->device_memory_free();
        cpu_info.capabilities = cpu_backend_->capabilities();
        devices.push_back(std::move(cpu_info));
    }

    // CUDA devices (enumerate all available, not just cached)
    int cuda_count = cuda_device_count();
    for (int i = 0; i < cuda_count; ++i) {
        auto gpu = get_cuda_backend(i);
        if (gpu) {
            DeviceInfo gpu_info;
            gpu_info.name = "cuda:" + std::to_string(i);
            gpu_info.type = DeviceType::CUDA;
            gpu_info.device_id = i;
            gpu_info.backend = gpu;
            gpu_info.memory_total = gpu->device_memory_total();
            gpu_info.memory_free = gpu->device_memory_free();
            gpu_info.capabilities = gpu->capabilities();
            devices.push_back(std::move(gpu_info));
        }
    }

    return devices;
}

}  // namespace forge
