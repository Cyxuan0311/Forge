#pragma once

#include <string>
#include <utility>
#include <vector>

#include "perf/gguf_scanner.h"

namespace forge::inspect {

// One metadata group (e.g. "general", "<arch>", "tokenizer", "other").
struct MetaGroup {
    std::string name;
    std::vector<std::pair<std::string, std::string>> entries;  // (key, formatted value)
};

// Groups a GGUF snapshot's metadata KV pairs by namespace and formats values.
std::vector<MetaGroup> group_metadata(const GgufSnapshot& snap, const std::string& arch);

}  // namespace forge::inspect
