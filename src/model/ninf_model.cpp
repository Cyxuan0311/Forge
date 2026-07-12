#include "forge/ninf_model.h"
#include "../core/platform.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "forge/logger.h"

namespace forge {

class MiniJson {
public:
    explicit MiniJson(const std::string& json) : src_(json), pos_(0) {}

    std::unordered_map<std::string, std::string> flatten() {
        std::unordered_map<std::string, std::string> result;
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == '{') {
            parse_object(result, "");
        }
        return result;
    }

private:
    void skip_ws() {
        while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_])))
            ++pos_;
    }

    void parse_object(std::unordered_map<std::string, std::string>& out,
                      const std::string& prefix) {
        ++pos_;
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == '}') {
            ++pos_;
            return;
        }

        while (pos_ < src_.size()) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ':')
                ++pos_;
            skip_ws();

            std::string full_key = prefix.empty() ? key : (prefix + "." + key);

            if (pos_ < src_.size() && src_[pos_] == '{') {
                parse_object(out, full_key);
            } else if (pos_ < src_.size() && src_[pos_] == '[') {
                std::string val = parse_array();
                out[full_key] = val;
            } else if (pos_ < src_.size() && src_[pos_] == '"') {
                out[full_key] = parse_string();
            } else {
                out[full_key] = parse_primitive();
            }

            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < src_.size() && src_[pos_] == '}') {
                ++pos_;
                break;
            }
            break;
        }
    }

    std::string parse_string() {
        if (pos_ >= src_.size() || src_[pos_] != '"')
            return "";
        ++pos_;
        std::string result;
        while (pos_ < src_.size() && src_[pos_] != '"') {
            if (src_[pos_] == '\\' && pos_ + 1 < src_.size()) {
                ++pos_;
                switch (src_[pos_]) {
                case '"':
                    result += '"';
                    break;
                case '\\':
                    result += '\\';
                    break;
                case 'n':
                    result += '\n';
                    break;
                case 't':
                    result += '\t';
                    break;
                default:
                    result += src_[pos_];
                    break;
                }
            } else {
                result += src_[pos_];
            }
            ++pos_;
        }
        if (pos_ < src_.size())
            ++pos_;
        return result;
    }

    std::string parse_primitive() {
        size_t start = pos_;
        while (pos_ < src_.size() && src_[pos_] != ',' && src_[pos_] != '}' && src_[pos_] != ']' &&
               !std::isspace(static_cast<unsigned char>(src_[pos_]))) {
            ++pos_;
        }
        return src_.substr(start, pos_ - start);
    }

    std::string parse_array() {
        ++pos_;
        std::string result = "[";
        int depth = 1;
        while (pos_ < src_.size() && depth > 0) {
            if (src_[pos_] == '[')
                ++depth;
            else if (src_[pos_] == ']') {
                --depth;
                if (depth == 0)
                    break;
            }
            result += src_[pos_];
            ++pos_;
        }
        if (pos_ < src_.size())
            ++pos_;
        result += "]";
        return result;
    }

    const std::string& src_;
    size_t pos_;
};

static DataType dtype_from_uint32(uint32_t d) {
    switch (d) {
    case 0:
        return DataType::FP32;
    case 1:
        return DataType::FP16;
    case 2:
        return DataType::Q4_0;
    case 3:
        return DataType::Q4_1;
    case 4:
        return DataType::Q4_K;
    case 5:
        return DataType::INT8;
    default:
        return DataType::FP32;
    }
}

NinfModel::~NinfModel() {
    close();
}

bool NinfModel::supports_format(const std::string& path) const {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return false;

    char magic[8] = {};
    ssize_t n = read(fd, magic, 8);
    ::close(fd);

    if (n != 8)
        return false;
    return std::strncmp(magic, "NINFMODL", 8) == 0;
}

bool NinfModel::load(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        return false;

    forge_stat_t sb;
    if (forge_fstat(fd_, &sb) < 0) {
        close();
        return false;
    }
    mapped_size_ = static_cast<size_t>(sb.st_size);

    mapped_data_ = forge_mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == FORGE_MAP_FAILED) {
        close();
        return false;
    }

    std::memcpy(&header_, mapped_data_, sizeof(NinfHeader));

    if (std::strncmp(header_.magic, "NINFMODL", 8) != 0) {
        close();
        return false;
    }

    if (header_.meta_size > 0) {
        metadata_ = std::string(static_cast<const char*>(mapped_data_) + header_.meta_offset,
                                header_.meta_size);
    }

    const uint8_t* base = static_cast<const uint8_t*>(mapped_data_);
    size_t tensor_info_offset = header_.meta_offset + header_.meta_size;

    tensors_.resize(header_.tensor_count);
    for (uint32_t i = 0; i < header_.tensor_count; ++i) {
        TensorInfo info;
        std::memcpy(&info, base + tensor_info_offset + i * sizeof(TensorInfo), sizeof(TensorInfo));

        LoadedTensor& lt = tensors_[i];
        lt.name = std::string(info.name, strnlen(info.name, 64));
        lt.dtype = dtype_from_uint32(info.dtype);
        lt.shape.resize(info.ndim);
        for (uint32_t d = 0; d < info.ndim; ++d) {
            lt.shape[d] = static_cast<int64_t>(info.dims[d]);
        }
        lt.file_offset = static_cast<int64_t>(info.offset);
        lt.data_size = static_cast<int64_t>(info.size);

        name_index_[lt.name] = i;
    }

    LOG_INFO("NinfModel loaded: " + std::to_string(tensors_.size()) + " tensors from " + path);
    return true;
}

void NinfModel::close() {
    if (mapped_data_ && mapped_data_ != FORGE_MAP_FAILED) {
        forge_munmap(mapped_data_, mapped_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    mapped_size_ = 0;
    tensors_.clear();
    name_index_.clear();
    metadata_.clear();
}

bool NinfModel::has_tensor(const std::string& name) const {
    return name_index_.find(name) != name_index_.end();
}

TensorPtr NinfModel::get_tensor(const std::string& name, DeviceType device) const {
    auto it = name_index_.find(name);
    if (it == name_index_.end())
        return nullptr;

    const LoadedTensor& lt = tensors_[it->second];
    auto tensor = std::make_shared<Tensor>(lt.dtype, lt.shape, DeviceType::CPU);

    const uint8_t* base = static_cast<const uint8_t*>(mapped_data_);
    const void* src = base + lt.file_offset;

    if (tensor->data() && lt.data_size > 0) {
        size_t copy_size = std::min(static_cast<size_t>(lt.data_size), tensor->nbytes());
        std::memcpy(tensor->data(), src, copy_size);
    }

    if (device == DeviceType::CUDA) {
        tensor->to_device(DeviceType::CUDA);
    }

    return tensor;
}

std::string NinfModel::get_metadata_str(const std::string& key,
                                        const std::string& default_val) const {
    if (metadata_.empty())
        return default_val;
    if (parsed_meta_.empty()) {
        MiniJson parser(metadata_);
        const_cast<std::unordered_map<std::string, std::string>&>(parsed_meta_) = parser.flatten();
    }
    auto it = parsed_meta_.find(key);
    if (it != parsed_meta_.end())
        return it->second;
    return default_val;
}

int64_t NinfModel::get_metadata_int(const std::string& key, int64_t default_val) const {
    std::string val = get_metadata_str(key, "");
    if (val.empty())
        return default_val;
    return std::strtoll(val.c_str(), nullptr, 10);
}

double NinfModel::get_metadata_float(const std::string& key, double default_val) const {
    std::string val = get_metadata_str(key, "");
    if (val.empty())
        return default_val;
    return std::strtod(val.c_str(), nullptr);
}

std::vector<int32_t> NinfModel::get_metadata_int_array(
    const std::string& key, const std::vector<int32_t>& default_val) const {
    // ninf format doesn't support int arrays natively, return default
    return default_val;
}

std::vector<std::string> NinfModel::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& t : tensors_) {
        names.push_back(t.name);
    }
    return names;
}

std::vector<int64_t> NinfModel::get_tensor_shape(const std::string& name) const {
    auto it = name_index_.find(name);
    if (it == name_index_.end())
        return {};
    return tensors_[it->second].shape;
}

}  // namespace forge
