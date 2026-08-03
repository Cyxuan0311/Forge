#include "forge/tensor.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "forge/backend.h"
#include "forge/memory_pool.h"
#include "memory_counters.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

Tensor::Tensor(DataType dtype, const std::vector<int64_t>& shape, DeviceType device)
    : shape_(shape), dtype_(dtype), device_(device) {
    numel_ = 1;
    for (auto d : shape_)
        numel_ *= d;
    compute_strides();
    if (numel_ > 0)
        allocate();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      dtype_(other.dtype_),
      device_(other.device_),
      numel_(other.numel_),
      nbytes_(other.nbytes_),
      owns_storage_(other.owns_storage_),
      storage_(other.storage_),
      layout_(other.layout_),
      backing_(std::move(other.backing_)) {
    other.data_ = nullptr;
    other.numel_ = 0;
    other.nbytes_ = 0;
    other.owns_storage_ = false;
    other.storage_ = TensorStorage{};
    other.layout_ = TensorLayout{};
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        numel_ = other.numel_;
        nbytes_ = other.nbytes_;
        owns_storage_ = other.owns_storage_;
        storage_ = other.storage_;
        layout_ = other.layout_;
        backing_ = std::move(other.backing_);
        other.data_ = nullptr;
        other.numel_ = 0;
        other.nbytes_ = 0;
        other.owns_storage_ = false;
        other.storage_ = TensorStorage{};
        other.layout_ = TensorLayout{};
    }
    return *this;
}

Tensor::~Tensor() {
    release();
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (!shape_.empty()) {
        strides_.back() = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }
}

void Tensor::allocate() {
    if (is_quantized_type(dtype_)) {
        nbytes_ = compute_quantized_bytes(numel_, dtype_);
    } else {
        nbytes_ = numel_ * dtype_size(dtype_);
    }

    if (nbytes_ == 0)
        return;

    if (device_ == DeviceType::CPU) {
        data_ = std::malloc(nbytes_);
        if (!data_)
            throw std::runtime_error("CPU malloc failed");
    } else {
#ifdef USE_CUDA
        cudaError_t err = cudaMalloc(&data_, nbytes_);
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    owns_storage_ = true;

    // Initialize storage_ and layout_ fields
    storage_.base = data_;
    storage_.capacity = nbytes_;
    storage_.device = device_;
    layout_.shape = shape_;
    layout_.strides = strides_;
    layout_.byte_offset = 0;
    layout_.logical_bytes = nbytes_;
    layout_.allocation_bytes = nbytes_;
}

void Tensor::release() {
    if (owns_storage_ && data_) {
        if (device_ == DeviceType::CPU) {
            std::free(data_);
        } else {
#ifdef USE_CUDA
            cudaFree(data_);
#endif
        }
    }
    data_ = nullptr;
    nbytes_ = 0;
    numel_ = 0;
    owns_storage_ = false;
    storage_ = TensorStorage{};
    layout_ = TensorLayout{};
}

void Tensor::zero_() {
    if (!data_ || nbytes_ == 0)
        return;
    if (device_ == DeviceType::CPU) {
        std::memset(data_, 0, nbytes_);
    } else {
#ifdef USE_CUDA
        cudaMemset(data_, 0, nbytes_);
#endif
    }
}

void Tensor::copy_from(const Tensor& src) {
    if (numel_ != src.numel_)
        throw std::runtime_error("Tensor size mismatch in copy_from");
    if (dtype_ != src.dtype_)
        throw std::runtime_error("dtype mismatch in copy_from");

    if (device_ == DeviceType::CPU && src.device_ == DeviceType::CPU) {
        std::memcpy(data_, src.data_, nbytes_);
    } else if (device_ == DeviceType::CUDA && src.device_ == DeviceType::CPU) {
#ifdef USE_CUDA
        auto& ctr = MemoryCounters::instance();
        ctr.h2d_copy_count.fetch_add(1, std::memory_order_relaxed);
        ctr.h2d_bytes.fetch_add(nbytes_, std::memory_order_relaxed);
        cudaMemcpyAsync(data_, src.data_, nbytes_, cudaMemcpyHostToDevice);
#endif
    } else if (device_ == DeviceType::CPU && src.device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
        auto& ctr = MemoryCounters::instance();
        ctr.d2h_copy_count.fetch_add(1, std::memory_order_relaxed);
        ctr.d2h_bytes.fetch_add(nbytes_, std::memory_order_relaxed);
        cudaMemcpy(data_, src.data_, nbytes_, cudaMemcpyDeviceToHost);
#endif
    } else {
#ifdef USE_CUDA
        auto& ctr = MemoryCounters::instance();
        ctr.d2d_copy_count.fetch_add(1, std::memory_order_relaxed);
        ctr.d2d_bytes.fetch_add(nbytes_, std::memory_order_relaxed);
        cudaMemcpyAsync(data_, src.data_, nbytes_, cudaMemcpyDeviceToDevice);
#endif
    }
}

void Tensor::to_device(DeviceType target) {
    if (device_ == target)
        return;

    Tensor new_tensor(dtype_, shape_, target);
    new_tensor.copy_from(*this);

    release();
    data_ = new_tensor.data_;
    shape_ = std::move(new_tensor.shape_);
    strides_ = std::move(new_tensor.strides_);
    numel_ = new_tensor.numel_;
    nbytes_ = new_tensor.nbytes_;
    owns_storage_ = new_tensor.owns_storage_;
    storage_ = new_tensor.storage_;
    layout_ = new_tensor.layout_;
    device_ = target;

    new_tensor.owns_storage_ = false;
    new_tensor.data_ = nullptr;
    new_tensor.numel_ = 0;
    new_tensor.nbytes_ = 0;
    new_tensor.storage_ = TensorStorage{};
    new_tensor.layout_ = TensorLayout{};
}

void* Tensor::replace_data(void* new_data, size_t new_nbytes) {
    void* old_data = data_;
    bool old_owns = owns_storage_;

    data_ = new_data;
    nbytes_ = new_nbytes;
    owns_storage_ = false;

    // Update storage_ fields for external data
    storage_.base = new_data;
    storage_.capacity = new_nbytes;
    storage_.device = device_;
    layout_.byte_offset = 0;
    layout_.logical_bytes = new_nbytes;
    layout_.allocation_bytes = new_nbytes;

    if (old_owns) {
        return old_data;
    }
    return nullptr;
}

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    Tensor t;
    t.data_ = data_;
    t.shape_ = new_shape;
    t.dtype_ = dtype_;
    t.device_ = device_;
    t.numel_ = 1;
    for (auto d : new_shape)
        t.numel_ *= d;
    t.compute_strides();
    t.nbytes_ = nbytes_;
    t.owns_storage_ = false;

    // Phase 1: set storage_ sharing the parent's base, with correct byte_offset
    t.storage_ = storage_;
    t.layout_.shape = new_shape;
    t.layout_.strides = t.strides_;
    t.layout_.byte_offset = layout_.byte_offset;  // same offset as parent (view is just reshape)
    t.layout_.logical_bytes = t.nbytes_;
    t.layout_.allocation_bytes = t.nbytes_;

    // Keep the backing tensor alive to prevent use-after-free.
    try {
        t.backing_ = backing_ ? backing_ : const_cast<Tensor*>(this)->shared_from_this();
    } catch (const std::bad_weak_ptr&) {
        // This tensor is not managed by a shared_ptr — backing_ stays null.
    }
    return t;
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end) const {
    if (dim < 0 || dim >= static_cast<int64_t>(shape_.size()))
        throw std::runtime_error("slice: dim out of range");

    Tensor t;
    t.dtype_ = dtype_;
    t.device_ = device_;
    t.owns_storage_ = false;

    size_t offset_bytes = 0;

    if (is_quantized_type(dtype_)) {
        int64_t block_el = dtype_block_elements(dtype_);
        int64_t block_sz = dtype_block_size(dtype_);
        int64_t n_blocks_before = (start * (numel_ / shape_[dim]) + block_el - 1) / block_el;
        offset_bytes = n_blocks_before * block_sz;
        t.data_ = static_cast<char*>(data_) + offset_bytes;
        t.shape_ = shape_;
        t.shape_[dim] = end - start;
        t.numel_ = numel_ / shape_[dim] * (end - start);
        t.compute_strides();
        int64_t n_blocks_total = (t.numel_ + block_el - 1) / block_el;
        t.nbytes_ = n_blocks_total * block_sz;
    } else {
        offset_bytes = start * strides_[dim] * dtype_size(dtype_);
        t.data_ = static_cast<char*>(data_) + offset_bytes;
        t.shape_ = shape_;
        t.shape_[dim] = end - start;
        t.numel_ = numel_ / shape_[dim] * (end - start);
        t.compute_strides();
        t.nbytes_ = t.numel_ * dtype_size(dtype_);
    }

    // Phase 1: set storage_ sharing the parent's base, with correct byte_offset
    t.storage_ = storage_;
    t.layout_.shape = t.shape_;
    t.layout_.strides = t.strides_;
    t.layout_.byte_offset = layout_.byte_offset + offset_bytes;
    t.layout_.logical_bytes = t.nbytes_;
    t.layout_.allocation_bytes = t.nbytes_;

    // Keep the backing tensor alive to prevent use-after-free.
    try {
        t.backing_ = backing_ ? backing_ : const_cast<Tensor*>(this)->shared_from_this();
    } catch (const std::bad_weak_ptr&) {
        // This tensor is not managed by a shared_ptr — backing_ stays null.
    }
    return t;
}

Tensor Tensor::from_buffer(void* ptr, DataType dtype, const std::vector<int64_t>& shape,
                           DeviceType device, bool own) {
    Tensor t;
    t.data_ = ptr;
    t.dtype_ = dtype;
    t.shape_ = shape;
    t.device_ = device;
    t.owns_storage_ = own;
    t.numel_ = 1;
    for (auto d : shape)
        t.numel_ *= d;
    t.compute_strides();
    if (is_quantized_type(dtype)) {
        t.nbytes_ = compute_quantized_bytes(t.numel_, dtype);
    } else {
        t.nbytes_ = t.numel_ * dtype_size(dtype);
    }

    // Phase 1: initialize storage_ and layout_
    t.storage_.base = ptr;
    t.storage_.capacity = t.nbytes_;
    t.storage_.device = device;
    t.layout_.shape = shape;
    t.layout_.strides = t.strides_;
    t.layout_.byte_offset = 0;
    t.layout_.logical_bytes = t.nbytes_;
    t.layout_.allocation_bytes = t.nbytes_;

    return t;
}

}  // namespace forge