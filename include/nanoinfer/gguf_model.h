#pragma once

#include "model_loader.h"
#include <cstdint>
#include <variant>

namespace nanoinfer {

enum class GgmlDType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
};

struct GgufTensorInfo {
    std::string name;
    std::vector<int64_t> dims;
    GgmlDType dtype;
    int64_t offset;
    int64_t data_offset;
};

struct GgufLoadedTensor {
    std::string name;
    DataType dtype;
    GgmlDType orig_dtype;
    std::vector<int64_t> shape;
    int64_t file_offset;
    int64_t data_size;
    bool is_gguf_layout;
};

class GgufModel : public ModelLoader {
public:
    GgufModel() = default;
    ~GgufModel() override;

    bool load(const std::string& path) override;
    void close() override;

    bool has_tensor(const std::string& name) const override;
    TensorPtr get_tensor(const std::string& name, DeviceType device = DeviceType::CPU) const override;

    std::string get_metadata_str(const std::string& key, const std::string& default_val = "") const override;
    int64_t get_metadata_int(const std::string& key, int64_t default_val = 0) const override;
    double get_metadata_float(const std::string& key, double default_val = 0.0) const override;
    std::vector<int32_t> get_metadata_int_array(const std::string& key, const std::vector<int32_t>& default_val = {}) const override;

    bool supports_format(const std::string& path) const override;
    std::string format_name() const override { return "gguf"; }

    std::vector<std::string> tensor_names() const override;
    size_t num_tensors() const override { return tensors_.size(); }

    std::vector<int64_t> get_tensor_shape(const std::string& name) const override;

    const std::unordered_map<std::string, std::string>& metadata_str() const { return metadata_str_; }
    const std::unordered_map<std::string, int64_t>& metadata_int() const { return metadata_int_; }
    const std::unordered_map<std::string, double>& metadata_float() const { return metadata_float_; }
    const std::vector<GgufLoadedTensor>& tensors() const { return tensors_; }

private:
    int fd_ = -1;
    void* mapped_data_ = nullptr;
    size_t mapped_size_ = 0;

    std::unordered_map<std::string, std::string> metadata_str_;
    std::unordered_map<std::string, int64_t> metadata_int_;
    std::unordered_map<std::string, double> metadata_float_;
    std::unordered_map<std::string, std::vector<int32_t>> metadata_int_arrays_;

    std::vector<GgufLoadedTensor> tensors_;
    std::unordered_map<std::string, size_t> name_index_;
};

} // namespace nanoinfer
