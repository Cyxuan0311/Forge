#pragma once

#include <memory>
#include <string>
#include "types.h"

namespace forge {

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

// Per-tensor 混合精度策略（参考 llama.cpp Q4_K_M）
// 不同类别的权重使用不同量化精度：关键权重更高精度，其余默认精度
struct QuantPolicy {
    DataType default_type  = DataType::Q4_K;  // 默认量化类型
    DataType attn_wv_type  = DataType::Q4_K;  // value projection（默认与 default 相同 = 不启用）
    DataType ffn_down_type = DataType::Q4_K;  // FFN down proj
    DataType output_type   = DataType::Q4_K;  // output head

    bool enabled() const {
        return attn_wv_type != default_type ||
               ffn_down_type != default_type ||
               output_type != default_type;
    }

    // Q4_K_M 预设：关键权重更高精度
    static QuantPolicy q4_k_m() {
        QuantPolicy p;
        p.default_type  = DataType::Q4_K;
        p.attn_wv_type  = DataType::Q5_K;
        p.ffn_down_type = DataType::Q5_K;
        p.output_type   = DataType::Q6_K;
        return p;
    }
};

// 按 GGUF tensor 名称模式匹配选择量化类型
DataType select_quant_type(const QuantPolicy& policy, const std::string& tensor_name);

// 将 tensor 重新量化到目标类型（dequant → FP32 → re-quant）
TensorPtr requant_tensor(const TensorPtr& src, DataType target_type);

}  // namespace forge
