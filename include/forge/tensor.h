#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include "types.h"

namespace forge {

class Tensor {
public:
    Tensor() = default;

    Tensor(DataType dtype, const std::vector<int64_t>& shape, DeviceType device = DeviceType::CPU);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

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
    Tensor view(const std::vector<int64_t>& new_shape) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end) const;

    // Replace internal data pointer with externally-owned memory.
    // Returns the old data pointer (caller must free it via deallocate if owns_data_ was true).
    // After this call, owns_data_ is false (the external owner manages the memory).
    void* replace_data(void* new_data, size_t new_nbytes);

    static Tensor from_buffer(void* ptr, DataType dtype, const std::vector<int64_t>& shape,
                               DeviceType device = DeviceType::CPU, bool own = false);

private:
    void compute_strides();
    void allocate();
    void release();

    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DataType dtype_ = DataType::FP32;
    DeviceType device_ = DeviceType::CPU;
    int64_t numel_ = 0;
    size_t nbytes_ = 0;
    bool owns_data_ = true;
};

using TensorPtr = std::shared_ptr<Tensor>;

} // namespace forge
