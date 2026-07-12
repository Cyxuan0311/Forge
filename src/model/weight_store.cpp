#include "forge/weight_store.h"
#include "forge/weight_mapper.h"

namespace forge {

void WeightStore::set(const std::string& name, TensorPtr tensor) {
    weights_[name] = std::move(tensor);
}

TensorPtr WeightStore::get(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) return nullptr;
    return it->second;
}

TensorPtr WeightStore::get_or_null(const std::string& name) const {
    return get(name);
}

bool WeightStore::has(const std::string& name) const {
    return weights_.find(name) != weights_.end();
}

const std::unordered_map<std::string, TensorPtr>& WeightStore::all() const {
    return weights_;
}

size_t WeightStore::size() const {
    return weights_.size();
}

void WeightStore::clear() {
    weights_.clear();
}

size_t WeightStore::total_bytes() const {
    size_t total = 0;
    for (const auto& [_, t] : weights_) {
        if (t) total += t->nbytes();
    }
    return total;
}

std::vector<std::string> WeightStore::weight_names() const {
    std::vector<std::string> names;
    names.reserve(weights_.size());
    for (const auto& [name, _] : weights_) {
        names.push_back(name);
    }
    return names;
}

void WeightStore::to_device(DeviceType device) {
    for (auto& [_, t] : weights_) {
        if (t) t->to_device(device);
    }
}

void WeightStore::to_device_layer(int layer_idx, DeviceType device, const std::string& prefix_pattern) {
    std::string prefix = WeightMapper::format_layer_prefix(prefix_pattern, layer_idx);
    std::string base = "layers." + std::to_string(layer_idx);
    for (auto& [name, t] : weights_) {
        if (t && (name.find(base) == 0 || name.find(prefix) == 0)) {
            t->to_device(device);
        }
    }
}

} // namespace forge
