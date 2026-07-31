#pragma once

// GraphKey: 计算图复用的完整判别键。
//
// 重构前 GraphCache 只比较 seq_len、gpu_layers 和 arch 三项, 这不足以保证安全:
// 同一 arch 在 prefill 与 decode、不同输入设备、不同 batch 组合下拓扑并不相同,
// 却会命中同一份缓存图。GraphKey 把所有影响拓扑的维度显式列出, 任何一项变化都
// 强制重建, 避免复用到语义不同的图。
//
// 注意: start_pos、seq_id、KV slot 不属于 key。它们在同一拓扑下变化, 应通过
// GraphRuntimeState 在执行期注入, 而不是参与缓存判别或被构建时永久捕获。

#include <string>

#include "forge/inference/forward_request.h"
#include "forge/tensor.h"

namespace forge {

struct GraphKey {
    int seq_len = 0;
    int gpu_layers = -1;
    std::string plan_id;
    DeviceType input_device = DeviceType::CPU;
    bool is_prefill = false;
    int batch_size = 1;
    int n_sequences = 1;

    bool operator==(const GraphKey& o) const {
        return seq_len == o.seq_len && gpu_layers == o.gpu_layers && plan_id == o.plan_id &&
               input_device == o.input_device && is_prefill == o.is_prefill &&
               batch_size == o.batch_size && n_sequences == o.n_sequences;
    }
    bool operator!=(const GraphKey& o) const { return !(*this == o); }

    std::string to_string() const;

    static GraphKey from_request(const ForwardRequest& req, const std::string& plan_id,
                                 int gpu_layers, DeviceType input_device);
};

}  // namespace forge
