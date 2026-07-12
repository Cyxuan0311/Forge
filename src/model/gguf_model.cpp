#include "forge/gguf_model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "forge/logger.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

static DataType ggml_dtype_to_dtype(GgmlDType dt) {
    switch (dt) {
    case GgmlDType::F32:
        return DataType::FP32;
    case GgmlDType::F16:
        return DataType::FP16;
    case GgmlDType::Q4_0:
        return DataType::Q4_0;
    case GgmlDType::Q4_1:
        return DataType::Q4_1;
    case GgmlDType::Q5_0:
        return DataType::Q5_0;
    case GgmlDType::Q5_1:
        return DataType::Q5_1;
    case GgmlDType::Q8_0:
        return DataType::Q8_0;
    case GgmlDType::Q8_1:
        return DataType::INT8;
    case GgmlDType::Q2_K:
        return DataType::Q2_K;
    case GgmlDType::Q3_K:
        return DataType::Q3_K;
    case GgmlDType::Q4_K:
        return DataType::Q4_K;
    case GgmlDType::Q5_K:
        return DataType::Q5_K;
    case GgmlDType::Q6_K:
        return DataType::Q6_K;
    default:
        return DataType::FP32;
    }
}

GgufModel::~GgufModel() {
    close();
}

bool GgufModel::supports_format(const std::string& path) const {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return false;

    uint32_t magic = 0;
    ssize_t n = read(fd, &magic, 4);
    ::close(fd);

    if (n != 4)
        return false;
    return magic == 0x46554747;
}

bool GgufModel::load(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        return false;

    struct stat sb;
    if (fstat(fd_, &sb) < 0) {
        close();
        return false;
    }
    mapped_size_ = static_cast<size_t>(sb.st_size);

    mapped_data_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        close();
        return false;
    }

    const uint8_t* data = static_cast<const uint8_t*>(mapped_data_);

    uint32_t magic;
    std::memcpy(&magic, data, 4);
    if (magic != 0x46554747) {
        close();
        return false;
    }

    uint32_t version;
    std::memcpy(&version, data + 4, 4);

    uint64_t tensor_count, metadata_kv_count;
    size_t offset = 8;

    if (version >= 3) {
        std::memcpy(&tensor_count, data + offset, 8);
        offset += 8;
        std::memcpy(&metadata_kv_count, data + offset, 8);
        offset += 8;
    } else {
        uint32_t tc, mc;
        std::memcpy(&tc, data + offset, 4);
        offset += 4;
        std::memcpy(&mc, data + offset, 4);
        offset += 4;
        tensor_count = tc;
        metadata_kv_count = mc;
    }

    LOG_INFO("GGUF version=" + std::to_string(version) +
             " tensor_count=" + std::to_string(tensor_count) +
             " metadata_kv_count=" + std::to_string(metadata_kv_count));

    auto read_string = [&](size_t& off) -> std::string {
        uint64_t len;
        std::memcpy(&len, data + off, 8);
        off += 8;
        LOG_DEBUG("read_string: len=" + std::to_string(len) + " off=" + std::to_string(off));
        if (len > mapped_size_ - off) {
            LOG_ERROR("read_string: invalid len=" + std::to_string(len) +
                      " remaining=" + std::to_string(mapped_size_ - off));
            return "";
        }
        std::string s(reinterpret_cast<const char*>(data + off), len);
        off += len;
        return s;
    };

    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        std::string key = read_string(offset);

        uint32_t vtype;
        std::memcpy(&vtype, data + offset, 4);
        offset += 4;

        LOG_DEBUG("metadata[" + std::to_string(i) + "] key=" + key +
                  " vtype=" + std::to_string(vtype) + " offset=" + std::to_string(offset));

        switch (vtype) {
        case 0: {
            uint8_t val;
            std::memcpy(&val, data + offset, 1);
            offset += 1;
            metadata_int_[key] = static_cast<int64_t>(val);
            break;
        }
        case 1: {
            int8_t val;
            std::memcpy(&val, data + offset, 1);
            offset += 1;
            metadata_int_[key] = static_cast<int64_t>(val);
            break;
        }
        case 2: {
            uint16_t val;
            std::memcpy(&val, data + offset, 2);
            offset += 2;
            metadata_int_[key] = static_cast<int64_t>(val);
            break;
        }
        case 3: {
            int16_t val;
            std::memcpy(&val, data + offset, 2);
            offset += 2;
            metadata_int_[key] = static_cast<int64_t>(val);
            break;
        }
        case 4: {
            uint32_t val;
            std::memcpy(&val, data + offset, 4);
            offset += 4;
            metadata_int_[key] = static_cast<int64_t>(val);
            break;
        }
        case 5: {
            int32_t val;
            std::memcpy(&val, data + offset, 4);
            offset += 4;
            metadata_int_[key] = static_cast<int64_t>(val);
            break;
        }
        case 6: {
            float val;
            std::memcpy(&val, data + offset, 4);
            offset += 4;
            metadata_float_[key] = static_cast<double>(val);
            break;
        }
        case 7: {
            bool val;
            std::memcpy(&val, data + offset, 1);
            offset += 1;
            metadata_int_[key] = val ? 1 : 0;
            break;
        }
        case 8: {
            std::string val = read_string(offset);
            metadata_str_[key] = val;
            break;
        }
        case 9: {
            uint32_t arr_type;
            std::memcpy(&arr_type, data + offset, 4);
            offset += 4;
            uint64_t arr_len;
            std::memcpy(&arr_len, data + offset, 8);
            offset += 8;

            // Store INT32 arrays for MRoPE sections etc.
            if (arr_type == 5) {  // INT32 array
                std::vector<int32_t> arr(arr_len);
                for (uint64_t j = 0; j < arr_len; ++j) {
                    std::memcpy(&arr[j], data + offset, 4);
                    offset += 4;
                }
                metadata_int_arrays_[key] = std::move(arr);
            } else {
                for (uint64_t j = 0; j < arr_len; ++j) {
                    switch (arr_type) {
                    case 0:
                        offset += 1;
                        break;
                    case 1:
                        offset += 1;
                        break;
                    case 2:
                        offset += 2;
                        break;
                    case 3:
                        offset += 2;
                        break;
                    case 4:
                        offset += 4;
                        break;
                    case 5:
                        offset += 4;
                        break;
                    case 6:
                        offset += 4;
                        break;
                    case 7:
                        offset += 1;
                        break;
                    case 8:
                        read_string(offset);
                        break;
                    case 10:
                        offset += 8;
                        break;
                    case 11:
                        offset += 8;
                        break;
                    case 12:
                        offset += 8;
                        break;
                    default:
                        offset += 4;
                        break;
                    }
                }
            }
            break;
        }
        case 10: {
            uint64_t val;
            std::memcpy(&val, data + offset, 8);
            offset += 8;
            metadata_int_[key] = static_cast<int64_t>(val);
            break;
        }
        case 11: {
            int64_t val;
            std::memcpy(&val, data + offset, 8);
            offset += 8;
            metadata_int_[key] = val;
            break;
        }
        case 12: {
            double val;
            std::memcpy(&val, data + offset, 8);
            offset += 8;
            metadata_float_[key] = val;
            break;
        }
        default: {
            LOG_WARN("Unknown GGUF metadata type: " + std::to_string(vtype) + " for key: " + key);
            offset += 8;
            break;
        }
        }
    }

    std::vector<GgufTensorInfo> tensor_infos(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        tensor_infos[i].name = read_string(offset);

        uint32_t ndim;
        std::memcpy(&ndim, data + offset, 4);
        offset += 4;

        tensor_infos[i].dims.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            uint64_t dim;
            std::memcpy(&dim, data + offset, 8);
            offset += 8;
            tensor_infos[i].dims[d] = static_cast<int64_t>(dim);
        }

        uint32_t dtype_val;
        std::memcpy(&dtype_val, data + offset, 4);
        offset += 4;
        tensor_infos[i].dtype = static_cast<GgmlDType>(dtype_val);

        std::memcpy(&tensor_infos[i].offset, data + offset, 8);
        offset += 8;
    }

    uint64_t alignment = 32;
    auto it_align = metadata_int_.find("general.alignment");
    if (it_align != metadata_int_.end()) {
        alignment = static_cast<uint64_t>(it_align->second);
    }

    uint64_t data_offset = offset;
    if (data_offset % alignment != 0) {
        data_offset += alignment - (data_offset % alignment);
    }

    tensors_.resize(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GgufLoadedTensor& lt = tensors_[i];
        lt.name = tensor_infos[i].name;
        lt.orig_dtype = tensor_infos[i].dtype;
        lt.dtype = ggml_dtype_to_dtype(tensor_infos[i].dtype);
        lt.shape = tensor_infos[i].dims;
        std::reverse(lt.shape.begin(), lt.shape.end());
        if (lt.shape.size() == 2 && (lt.shape[0] > 1 || lt.shape[1] > 1)) {
            // Debug: tensor dimension reversal (disabled by default)
            // fprintf(stderr, "[DEBUG] tensor '%s' original_dims=[%ld,%ld]
            // reversed_dims=[%ld,%ld]\n",
            //         lt.name.c_str(),
            //         (long)tensor_infos[i].dims[0], (long)tensor_infos[i].dims[1],
            //         (long)lt.shape[0], (long)lt.shape[1]);
            // fflush(stderr);
        }
        lt.file_offset = static_cast<int64_t>(data_offset + tensor_infos[i].offset);
        lt.is_gguf_layout = true;

        int64_t nelements = 1;
        for (auto d : lt.shape)
            nelements *= d;
        if (is_quantized_type(lt.dtype)) {
            lt.data_size = static_cast<int64_t>(compute_quantized_bytes(nelements, lt.dtype));
        } else {
            lt.data_size = nelements * dtype_size(lt.dtype);
        }

        name_index_[lt.name] = static_cast<size_t>(i);
    }

    LOG_INFO("GgufModel loaded: " + std::to_string(tensors_.size()) + " tensors from " + path);

    // Log total tensor data size
    size_t total_tensor_bytes = 0;
    for (const auto& t : tensors_)
        total_tensor_bytes += t.data_size;
    LOG_INFO("GgufModel total tensor data: " + std::to_string(total_tensor_bytes / (1024 * 1024)) +
             " MB, file size: " + std::to_string(mapped_size_ / (1024 * 1024)) + " MB");

    return true;
}

void GgufModel::close() {
    if (mapped_data_ && mapped_data_ != MAP_FAILED) {
        munmap(mapped_data_, mapped_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    mapped_size_ = 0;
    tensors_.clear();
    name_index_.clear();
    metadata_str_.clear();
    metadata_int_.clear();
    metadata_float_.clear();
    metadata_int_arrays_.clear();
}

bool GgufModel::has_tensor(const std::string& name) const {
    return name_index_.find(name) != name_index_.end();
}

TensorPtr GgufModel::get_tensor(const std::string& name, DeviceType device) const {
    auto it = name_index_.find(name);
    if (it == name_index_.end())
        return nullptr;

    const GgufLoadedTensor& lt = tensors_[it->second];

    const uint8_t* base = static_cast<const uint8_t*>(mapped_data_);
    const void* src = base + lt.file_offset;

    // Bounds check
    if (lt.file_offset < 0 || static_cast<size_t>(lt.file_offset) >= mapped_size_) {
        LOG_ERROR("get_tensor: " + name + " invalid file_offset=" + std::to_string(lt.file_offset));
        return nullptr;
    }
    if (static_cast<size_t>(lt.file_offset) + lt.data_size > mapped_size_) {
        LOG_ERROR("get_tensor: " + name + " data extends beyond file: offset=" +
                  std::to_string(lt.file_offset) + " + size=" + std::to_string(lt.data_size) +
                  " > file_size=" + std::to_string(mapped_size_));
        return nullptr;
    }

    if (device == DeviceType::CPU) {
        // Zero-copy: directly reference mmap'd memory without copying
        // The GgufModel (and its mmap) must stay alive as long as this tensor exists
        auto tensor = std::make_shared<Tensor>(Tensor::from_buffer(
            const_cast<void*>(src), lt.dtype, lt.shape, DeviceType::CPU, false));
        return tensor;
    } else {
        // CUDA: allocate GPU memory and copy directly from mmap region
        // This avoids the intermediate CPU allocation + memcpy
#ifdef USE_CUDA
        auto tensor = std::make_shared<Tensor>(lt.dtype, lt.shape, DeviceType::CUDA);
        if (tensor->data() && lt.data_size > 0) {
            size_t copy_size = std::min(static_cast<size_t>(lt.data_size), tensor->nbytes());
            cudaMemcpy(tensor->data(), src, copy_size, cudaMemcpyHostToDevice);
        }
        return tensor;
#else
        // Fallback: allocate CPU tensor and copy (CUDA not available)
        auto tensor = std::make_shared<Tensor>(lt.dtype, lt.shape, DeviceType::CPU);
        if (tensor->data() && lt.data_size > 0) {
            size_t copy_size = std::min(static_cast<size_t>(lt.data_size), tensor->nbytes());
            std::memcpy(tensor->data(), src, copy_size);
        }
        return tensor;
#endif
    }
}

std::string GgufModel::get_metadata_str(const std::string& key,
                                        const std::string& default_val) const {
    auto it = metadata_str_.find(key);
    return it != metadata_str_.end() ? it->second : default_val;
}

int64_t GgufModel::get_metadata_int(const std::string& key, int64_t default_val) const {
    auto it = metadata_int_.find(key);
    return it != metadata_int_.end() ? it->second : default_val;
}

double GgufModel::get_metadata_float(const std::string& key, double default_val) const {
    auto it = metadata_float_.find(key);
    return it != metadata_float_.end() ? it->second : default_val;
}

std::vector<int32_t> GgufModel::get_metadata_int_array(
    const std::string& key, const std::vector<int32_t>& default_val) const {
    auto it = metadata_int_arrays_.find(key);
    return it != metadata_int_arrays_.end() ? it->second : default_val;
}

std::vector<std::string> GgufModel::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& t : tensors_) {
        names.push_back(t.name);
    }
    return names;
}

std::vector<int64_t> GgufModel::get_tensor_shape(const std::string& name) const {
    auto it = name_index_.find(name);
    if (it == name_index_.end())
        return {};
    return tensors_[it->second].shape;
}

}  // namespace forge
