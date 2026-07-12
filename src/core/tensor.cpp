#include "forge/tensor.h"
#include "forge/memory_pool.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace forge {

Tensor::Tensor(DataType dtype, const std::vector<int64_t>& shape, DeviceType device)
    : shape_(shape), dtype_(dtype), device_(device) {
    numel_ = 1;
    for (auto d : shape_) numel_ *= d;
    compute_strides();
    if (numel_ > 0) allocate();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)), dtype_(other.dtype_),
      device_(other.device_), numel_(other.numel_), nbytes_(other.nbytes_),
      owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.numel_ = 0;
    other.nbytes_ = 0;
    other.owns_data_ = false;
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
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.numel_ = 0;
        other.nbytes_ = 0;
        other.owns_data_ = false;
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

    if (nbytes_ == 0) return;

    if (device_ == DeviceType::CPU) {
        data_ = std::malloc(nbytes_);
        if (!data_) throw std::runtime_error("CPU malloc failed");
    } else {
#ifdef USE_CUDA
        cudaError_t err = cudaMalloc(&data_, nbytes_);
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    owns_data_ = true;
}

void Tensor::release() {
    if (owns_data_ && data_) {
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
    owns_data_ = false;
}

void Tensor::zero_() {
    if (!data_ || nbytes_ == 0) return;
    if (device_ == DeviceType::CPU) {
        std::memset(data_, 0, nbytes_);
    } else {
#ifdef USE_CUDA
        cudaMemset(data_, 0, nbytes_);
#endif
    }
}

void Tensor::copy_from(const Tensor& src) {
    if (numel_ != src.numel_) throw std::runtime_error("Tensor size mismatch in copy_from");
    if (dtype_ != src.dtype_) throw std::runtime_error("dtype mismatch in copy_from");

    if (device_ == DeviceType::CPU && src.device_ == DeviceType::CPU) {
        std::memcpy(data_, src.data_, nbytes_);
    } else if (device_ == DeviceType::CUDA && src.device_ == DeviceType::CPU) {
#ifdef USE_CUDA
        cudaMemcpyAsync(data_, src.data_, nbytes_, cudaMemcpyHostToDevice);
#endif
    } else if (device_ == DeviceType::CPU && src.device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
        cudaMemcpy(data_, src.data_, nbytes_, cudaMemcpyDeviceToHost);
#endif
    } else {
#ifdef USE_CUDA
        cudaMemcpyAsync(data_, src.data_, nbytes_, cudaMemcpyDeviceToDevice);
#endif
    }
}

void Tensor::to_device(DeviceType target) {
    if (device_ == target) return;

    Tensor new_tensor(dtype_, shape_, target);
    new_tensor.copy_from(*this);

    release();
    data_ = new_tensor.data_;
    shape_ = std::move(new_tensor.shape_);
    strides_ = std::move(new_tensor.strides_);
    numel_ = new_tensor.numel_;
    nbytes_ = new_tensor.nbytes_;
    owns_data_ = new_tensor.owns_data_;
    device_ = target;

    new_tensor.owns_data_ = false;
    new_tensor.data_ = nullptr;
    new_tensor.numel_ = 0;
    new_tensor.nbytes_ = 0;
}

void* Tensor::replace_data(void* new_data, size_t new_nbytes) {
    void* old_data = data_;
    bool old_owns = owns_data_;

    data_ = new_data;
    nbytes_ = new_nbytes;
    owns_data_ = false;

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
    for (auto d : new_shape) t.numel_ *= d;
    t.compute_strides();
    t.nbytes_ = nbytes_;
    t.owns_data_ = false;
    return t;
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end) const {
    if (dim < 0 || dim >= static_cast<int64_t>(shape_.size()))
        throw std::runtime_error("slice: dim out of range");

    Tensor t;
    t.dtype_ = dtype_;
    t.device_ = device_;
    t.owns_data_ = false;

    if (is_quantized_type(dtype_)) {
        int64_t block_el = dtype_block_elements(dtype_);
        int64_t block_sz = dtype_block_size(dtype_);
        int64_t n_blocks_before = (start * (numel_ / shape_[dim]) + block_el - 1) / block_el;
        t.data_ = static_cast<char*>(data_) + n_blocks_before * block_sz;
        t.shape_ = shape_;
        t.shape_[dim] = end - start;
        t.numel_ = numel_ / shape_[dim] * (end - start);
        t.compute_strides();
        int64_t n_blocks_total = (t.numel_ + block_el - 1) / block_el;
        t.nbytes_ = n_blocks_total * block_sz;
    } else {
        auto offset_bytes = start * strides_[dim] * dtype_size(dtype_);
        t.data_ = static_cast<char*>(data_) + offset_bytes;
        t.shape_ = shape_;
        t.shape_[dim] = end - start;
        t.numel_ = numel_ / shape_[dim] * (end - start);
        t.compute_strides();
        t.nbytes_ = t.numel_ * dtype_size(dtype_);
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
    t.owns_data_ = own;
    t.numel_ = 1;
    for (auto d : shape) t.numel_ *= d;
    t.compute_strides();
    if (is_quantized_type(dtype)) {
        t.nbytes_ = compute_quantized_bytes(t.numel_, dtype);
    } else {
        t.nbytes_ = t.numel_ * dtype_size(dtype);
    }
    return t;
}

} // namespace forge
