#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "tensor.h"
#include "types.h"

namespace forge {

class WeightStore {
public:
    void set(const std::string& name, TensorPtr tensor);
    TensorPtr get(const std::string& name) const;
    TensorPtr get_or_null(const std::string& name) const;
    bool has(const std::string& name) const;
    const std::unordered_map<std::string, TensorPtr>& all() const;
    size_t size() const;
    void clear();
    size_t total_bytes() const;

    std::vector<std::string> weight_names() const;

    void to_device(DeviceType device);
    void to_device_layer(int layer_idx, DeviceType device, const std::string& prefix_pattern = "model.layers.{}");

private:
    std::unordered_map<std::string, TensorPtr> weights_;
};

} // namespace forge
