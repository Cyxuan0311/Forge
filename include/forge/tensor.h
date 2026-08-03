#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "types.h"

namespace forge {

class Backend;

// ---- Phase 1: Storage/Layout separation ----
// TensorStorage owns the raw memory buffer.
// Multiple Tensor views can share the same TensorStorage.
struct TensorStorage {
    void* base = nullptr;         // base pointer (may differ from view data ptr)
    size_t capacity = 0;          // total allocated bytes
    DeviceType device = DeviceType::CPU;
    Backend* backend = nullptr;   // which backend owns this memory

    // Returns true if this storage object is valid (has a base pointer)
    bool valid() const { return base != nullptr; }
};

// TensorLayout describes the logical view of a tensor.
// It references TensorStorage via byte_offset.
struct TensorLayout {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;   // element strides (not byte strides)
    size_t byte_offset = 0;         // byte offset from storage.base to data()
    size_t logical_bytes = 0;       // nbytes() — logical byte count
    size_t allocation_bytes = 0;    // bytes needed for allocation (planner uses this)
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor() = default;

    Tensor(DataType dtype, const std::vector<int64_t>& shape, DeviceType device = DeviceType::CPU);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    // ---- Data access (backward compatible) ----
    void* data() { return data_; }
    const void* data() const { return data_; }

    int64_t numel() const { return numel_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }

    DataType dtype() const { return dtype_; }
    DeviceType device() const { return device_; }
    size_t nbytes() const { return nbytes_; }

    void zero_();
    void copy_from(const Tensor& src);
    void to_device(DeviceType target);

    // ---- View / Slice (Phase 1: storage-aware) ----
    // Creates a view sharing the same storage.
    // Carries byte_offset, strides, and owner reference.
    Tensor view(const std::vector<int64_t>& new_shape) const;

    // Creates a slice view along a dimension.
    // Carries byte_offset relative to the base storage pointer.
    Tensor slice(int64_t dim, int64_t start, int64_t end) const;

    // ---- Storage / Layout accessors (Phase 1) ----
    const TensorStorage& storage() const { return storage_; }
    const TensorLayout& layout() const { return layout_; }

    // Byte offset of data() relative to storage().base.
    size_t byte_offset() const { return layout_.byte_offset; }

    // Number of bytes needed for allocation (planner uses this).
    // For owned tensors, equals nbytes(). For views, equals the view's allocated portion.
    size_t allocation_bytes() const { return layout_.allocation_bytes; }

    // Whether this tensor owns the underlying storage.
    bool owns_storage() const { return owns_storage_; }

    // Replace internal data pointer with externally-owned memory.
    // Returns the old data pointer (caller must free it via deallocate if owns_storage_ was true).
    // After this call, owns_storage_ is false (the external owner manages the memory).
    void* replace_data(void* new_data, size_t new_nbytes);

    // Create a tensor from an externally-owned buffer.
    static Tensor from_buffer(void* ptr, DataType dtype, const std::vector<int64_t>& shape,
                              DeviceType device = DeviceType::CPU, bool own = false);

private:
    void compute_strides();
    void allocate();
    void release();

    // ---- Old fields (backward compatible, kept for Phase 1 transition) ----
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DataType dtype_ = DataType::FP32;
    DeviceType device_ = DeviceType::CPU;
    int64_t numel_ = 0;
    size_t nbytes_ = 0;
    // Note: owns_data_ renamed to owns_storage_ for clarity
    bool owns_storage_ = true;

    // ---- New Phase 1 fields ----
    TensorStorage storage_;
    TensorLayout layout_;

    // Keeps the backing tensor alive when this is a view/slice.
    // Without this, view() creates a dangling pointer when the owning tensor is freed.
    std::shared_ptr<Tensor> backing_;
};

using TensorPtr = std::shared_ptr<Tensor>;

}  // namespace forge