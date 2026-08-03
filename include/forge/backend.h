#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "types.h"

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {

class Backend;

struct BackendStream {
    void* handle = nullptr;
    Backend* backend = nullptr;
    ~BackendStream();
    void synchronize();
};

// Backend capability flags
enum class BackendCapability : uint32_t {
    None = 0,
    FP32 = 1 << 0,
    FP16 = 1 << 1,
    INT8 = 1 << 2,
    Quantized = 1 << 3,      // Q4_0, Q8_0, etc.
    UnifiedMemory = 1 << 4,  // Host and device share address space
    StreamAsync = 1 << 5,    // Supports async stream operations
};

inline BackendCapability operator|(BackendCapability a, BackendCapability b) {
    return static_cast<BackendCapability>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline bool operator&(BackendCapability a, BackendCapability b) {
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

// Full capability description of a backend, used by OpDispatch::supports() to decide
// whether an op can run on this backend (aligns with ggml_backend_supports_op).
struct BackendCapabilities {
    DeviceType device = DeviceType::CPU;
    BackendCapability flags = BackendCapability::FP32;
    size_t alignment = 1;        // required byte alignment of tensor data
    int compute_capability = 0;  // GPU compute capability (major * 10 + minor), 0 for CPU

    bool has(BackendCapability c) const { return (flags & c); }
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual DeviceType device_type() const = 0;
    virtual std::string name() const = 0;

    // Memory management
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr, size_t size) = 0;

    // Backend-specific allocation size (for planner, aligns with ggml_backend_buft_get_alloc_size).
    // Returns the actual number of bytes the backend needs to allocate for a tensor of given
    // logical size. Default implementation returns the logical size.
    virtual size_t get_alloc_size(size_t logical_size) const { return logical_size; }

    // Memory pool (optional, for backends that support it)
    virtual void* pool_allocate(size_t size) { return allocate(size); }
    virtual void pool_deallocate(void* ptr, size_t size) { deallocate(ptr, size); }
    virtual void pool_reset() {}

    enum class CopyKind {
        HostToHost,
        HostToDevice,
        DeviceToHost,
        DeviceToDevice,
    };

    virtual void copy(void* dst, const void* src, size_t size, CopyKind kind) = 0;
    virtual void memset(void* ptr, int value, size_t size) = 0;
    virtual void synchronize() = 0;

    // Stream operations
    virtual BackendStream create_stream();
    virtual void synchronize_stream(BackendStream& stream);
    virtual void destroy_stream(BackendStream& stream);

    // Capability query
    virtual BackendCapability capabilities() const { return BackendCapability::FP32; }
    bool supports(BackendCapability cap) const { return (capabilities() & cap); }

    // Full capability description (device, flags, alignment, compute capability)
    virtual BackendCapabilities full_capabilities() const {
        BackendCapabilities caps;
        caps.device = device_type();
        caps.flags = capabilities();
        caps.alignment = 1;
        caps.compute_capability = 0;
        return caps;
    }

    // Device info
    virtual size_t device_memory_total() const { return 0; }
    virtual size_t device_memory_free() const { return 0; }
    virtual int device_id() const { return 0; }

    // Factory methods
    static std::shared_ptr<Backend> create(DeviceType device);
    static std::shared_ptr<Backend> create_cpu();
    static std::shared_ptr<Backend> create_cuda(int device_id = 0);
};

// Backend factory function type
using BackendFactory = std::function<std::shared_ptr<Backend>(int device_id)>;

struct BackendInfo {
    std::string name;  // e.g., "cpu", "cuda", "metal"
    DeviceType device_type;
    BackendFactory factory;
    bool available;  // Whether the backend is usable on this system
};

// Runtime device info for scheduling (BackendScheduler)
struct DeviceInfo {
    std::string name;  // "cpu", "cuda:0"
    DeviceType type;
    int device_id;  // -1 for CPU, 0+ for CUDA
    std::shared_ptr<Backend> backend;
    size_t memory_total = 0;
    size_t memory_free = 0;
    BackendCapability capabilities = BackendCapability::FP32;
};

class BackendManager {
public:
    static BackendManager& instance();

    // Get backend by device type (legacy API)
    std::shared_ptr<Backend> get_backend(DeviceType device);
    std::shared_ptr<Backend> get_cpu_backend();
    std::shared_ptr<Backend> get_cuda_backend(int device_id = 0);

    // Plugin-based registration
    void register_backend(const std::string& name, DeviceType device_type, BackendFactory factory);
    void unregister_backend(const std::string& name);

    // Get backend by name
    std::shared_ptr<Backend> get_backend(const std::string& name, int device_id = 0);

    // Query available backends
    std::vector<std::string> available_backends() const;
    std::vector<BackendInfo> backend_info_list() const;
    bool has_backend(const std::string& name) const;

    // Legacy registration (delegates to plugin API)
    void register_backend(DeviceType device, std::shared_ptr<Backend> backend);

    bool has_cuda() const;
    int cuda_device_count() const;

    // Enumerate all available devices with their info (for BackendScheduler)
    std::vector<DeviceInfo> available_devices();

private:
    BackendManager();

    std::shared_ptr<Backend> cpu_backend_;
    std::unordered_map<int, std::shared_ptr<Backend>> cuda_backends_;

    // Plugin registry
    std::unordered_map<std::string, BackendInfo> registry_;
    std::unordered_map<std::string, std::shared_ptr<Backend>> cached_backends_;
};

}  // namespace forge
