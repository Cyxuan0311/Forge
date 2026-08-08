#include "meta_groups.h"

#include <algorithm>
#include <cstdio>

namespace forge::inspect {

namespace {

std::string format_value(const GgufSnapshot& snap, const std::string& key) {
    if (auto it = snap.meta_str.find(key); it != snap.meta_str.end()) {
        std::string v = it->second;
        if (v.size() > 120)
            v = v.substr(0, 117) + "...";
        return "\"" + v + "\"";
    }
    if (auto it = snap.meta_float.find(key); it != snap.meta_float.end()) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%g", it->second);
        return buf;
    }
    if (auto it = snap.meta_int.find(key); it != snap.meta_int.end()) {
        return std::to_string(it->second);
    }
    if (auto it = snap.meta_int_array.find(key); it != snap.meta_int_array.end()) {
        std::string s = "[";
        const size_t show = std::min<size_t>(it->second.size(), 12);
        for (size_t i = 0; i < show; ++i)
            s += (i ? ", " : "") + std::to_string(it->second[i]);
        if (it->second.size() > show)
            s += ", ...";
        s += "]";
        return s;
    }
    return "";
}

}  // namespace

std::vector<MetaGroup> group_metadata(const GgufSnapshot& snap, const std::string& arch) {
    // Collect all keys first so we can route them into groups in a stable order.
    std::vector<std::string> keys;
    for (const auto& kv : snap.meta_str)
        keys.push_back(kv.first);
    for (const auto& kv : snap.meta_int)
        keys.push_back(kv.first);
    for (const auto& kv : snap.meta_float)
        keys.push_back(kv.first);
    for (const auto& kv : snap.meta_int_array)
        keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    std::vector<MetaGroup> groups;
    auto group_for = [&](const std::string& name) -> MetaGroup& {
        for (auto& g : groups)
            if (g.name == name)
                return g;
        groups.push_back(MetaGroup{name, {}});
        return groups.back();
    };

    const std::string arch_prefix = arch + ".";
    for (const auto& key : keys) {
        std::string gname = "other";
        if (key.rfind("general.", 0) == 0)
            gname = "general";
        else if (key.rfind(arch_prefix, 0) == 0)
            gname = arch;
        else if (key.rfind("tokenizer.", 0) == 0)
            gname = "tokenizer";
        group_for(gname).entries.emplace_back(key, format_value(snap, key));
    }
    return groups;
}

}  // namespace forge::inspect
