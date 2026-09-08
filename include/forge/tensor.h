#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "types.h"

// 6-C-5.2: Tensor owns a device-side mirror of its FP8 scale, so it needs
// cudaStream_t in both CUDA and CPU-only builds. Same fallback pattern the rest of
// the public headers use (op_dispatch.h / backend.h): an opaque forward declaration
// of the CUDA stream type when we are not compiling against the CUDA runtime.
#ifdef USE_CUDA
#    include <cuda_runtime.h>
#else
struct CUstream_st;
typedef CUstream_st* cudaStream_t;
#endif

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

    // 6-C-5: FP8 per-tensor scale sidecar. Lives on the Tensor (data layer), NOT in
    // ComputeDTypePolicy, so it travels with the TensorPtr along ComputeGraph edges
    // (a node's output Tensor is fed directly as the next node's resolved_input).
    // 6-C-5.7: this sidecar is the ONLY source of an FP8 scale -- the legacy host
    // scale registry (fp8_scale_registry.cpp) has been deleted, so an unset sidecar
    // is a hard error for every FP8 consumer (no silent host fallback).
    bool has_fp8_scale() const { return has_fp8_scale_; }
    float fp8_scale() const { return fp8_scale_; }
    void set_fp8_scale(float s) {
        fp8_scale_ = s;
        has_fp8_scale_ = true;
        fp8_scale_dirty_ = true;  // device mirror must be re-uploaded
    }

    // 6-C-5.2: device-visible mirror of the scale. The returned pointer is STABLE
    // for this Tensor's lifetime (allocated once, then reused) -- that is what makes
    // it graph-visible: a captured graph keeps reading the same address, and calling
    // set_fp8_scale() + this accessor re-uploads into that SAME address so a replay
    // sees the new value without rebuilding the graphExec (6-C-4.1c proven pattern).
    // Returns nullptr when no scale is set (or on CPU-only builds / OOM).
    const float* fp8_scale_device(cudaStream_t stream = 0);
    const float* fp8_scale_device_ptr() const { return fp8_scale_device_; }

    // 6-C-5.1a: writable slot for DEVICE-COMPUTED scales (amax -> amax/240). Allocates
    // the stable scalar once and returns it WITHOUT uploading the host value: a
    // producer kernel writes the scale on device, so the pipeline needs no D2H/H2D
    // round trip and stays capturable by CUDA Graphs. After this call the DEVICE value
    // is authoritative (see fp8_scale_device_authoritative()) and fp8_scale_device()
    // will not clobber it.
    float* fp8_scale_device_slot(cudaStream_t stream = 0);
    bool fp8_scale_device_authoritative() const { return fp8_scale_device_authoritative_; }

    // 6-C-5.5c: explicit refresh of the HOST mirror for a DEVICE-computed scale
    // (5.1a / 5.5a producers). Until this is called, fp8_scale() is not meaningful for a
    // device-authoritative tensor. Deliberately opt-in and never used on the hot path:
    // a D2H sync is illegal inside CUDA Graph capture.
    bool sync_fp8_scale_from_device(cudaStream_t stream = 0);

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
    // 6-C-5.2: free the lazily-allocated device mirror of the FP8 scale sidecar.
    void free_fp8_scale_device();

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

    // 6-C-5: FP8 per-tensor scale sidecar (see public accessors above). Defaults to
    // unset so legacy behavior (no scale, or registry-lookup) is preserved.
    float fp8_scale_ = 0.0f;
    bool has_fp8_scale_ = false;
    // 6-C-5.2: lazily-allocated device mirror + pending-upload flag. Owned by this
    // Tensor (freed in the destructor, transferred by the move ops).
    float* fp8_scale_device_ = nullptr;
    bool fp8_scale_dirty_ = true;
    // 6-C-5.1a: true once the device value is produced ON device (fp8_scale_device_slot)
    // and the host mirror is therefore not authoritative.
    bool fp8_scale_device_authoritative_ = false;
};

using TensorPtr = std::shared_ptr<Tensor>;

}  // namespace forge