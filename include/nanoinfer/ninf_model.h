#pragma once

#include "model_loader.h"
#include <cstdint>

namespace nanoinfer {

#pragma pack(push, 1)
struct NinfHeader {
    char magic[8];
    uint32_t version;
    uint32_t meta_offset;
    uint32_t meta_size;
    uint32_t tensor_count;
    uint64_t total_data_size;
};

struct TensorInfo {
    char name[64];
    uint32_t dtype;
    uint32_t ndim;
    uint64_t dims[4];
    uint64_t offset;
    uint64_t size;
};
#pragma pack(pop)

struct LoadedTensor {
    std::string name;
    DataType dtype;
    std::vector<int64_t> shape;
    int64_t file_offset;
    int64_t data_size;
};

class NinfModel : public ModelLoader {
public:
    NinfModel() = default;
    ~NinfModel() override;

    bool load(const std::string& path) override;
    void close() override;

    bool has_tensor(const std::string& name) const override;
    TensorPtr get_tensor(const std::string& name, DeviceType device = DeviceType::CPU) const override;

    std::string get_metadata_str(const std::string& key, const std::string& default_val = "") const override;
    int64_t get_metadata_int(const std::string& key, int64_t default_val = 0) const override;
    double get_metadata_float(const std::string& key, double default_val = 0.0) const override;
    std::vector<int32_t> get_metadata_int_array(const std::string& key, const std::vector<int32_t>& default_val = {}) const override;

    bool supports_format(const std::string& path) const override;
    std::string format_name() const override { return "ninf"; }

    std::vector<std::string> tensor_names() const override;
    size_t num_tensors() const override { return tensors_.size(); }

    std::vector<int64_t> get_tensor_shape(const std::string& name) const override;

    const NinfHeader& header() const { return header_; }
    const std::string& metadata() const { return metadata_; }

private:
    int fd_ = -1;
    void* mapped_data_ = nullptr;
    size_t mapped_size_ = 0;
    NinfHeader header_{};
    std::string metadata_;
    std::vector<LoadedTensor> tensors_;
    std::unordered_map<std::string, size_t> name_index_;
    mutable std::unordered_map<std::string, std::string> parsed_meta_;
};

} // namespace nanoinfer
