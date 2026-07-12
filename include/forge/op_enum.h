#pragma once

#include <cstdint>
#include <string>

namespace forge {

enum class OpType : uint32_t {
    NONE = 0,

    // Shape / data movement
    VIEW,
    PERMUTE,
    RESHAPE,
    TRANSPOSE,
    GET_ROWS,
    CPY,
    CONT,
    REPEAT,
    CONCAT,
    PAD,

    // Unary element-wise
    SILU,
    GELU,
    GELU_TANH,
    RELU,
    NEG,
    SQR,
    SQRT,
    EXP,
    LOG,

    // Binary element-wise
    ADD,
    SUB,
    MUL,
    DIV,
    SCALE,

    // Normalization
    RMS_NORM,
    LAYER_NORM,
    GROUP_NORM,

    // Matrix multiply
    MUL_MAT,
    MUL_MAT_TRANSB,
    MUL_MAT_ID,
    OUT_PROD,

    // Attention
    SOFT_MAX,
    SOFT_MAX_MASKED,
    FLASH_ATTN_EXT,
    FLASH_ATTN_GQA,

    // Position encoding
    ROPE,

    // Embedding
    EMBEDDING,

    // Reduction
    SUM,
    MEAN,
    ARGMAX,

    // GLU variants
    GLU,

    // KV cache operations
    CPY_K,
    CPY_V,
    GET_K,
    GET_V,

    // Custom / user-defined
    CUSTOM,

    COUNT,
};

inline std::string op_type_name(OpType op) {
    switch (op) {
    case OpType::NONE:
        return "none";
    case OpType::VIEW:
        return "view";
    case OpType::PERMUTE:
        return "permute";
    case OpType::RESHAPE:
        return "reshape";
    case OpType::TRANSPOSE:
        return "transpose";
    case OpType::GET_ROWS:
        return "get_rows";
    case OpType::CPY:
        return "cpy";
    case OpType::CONT:
        return "cont";
    case OpType::REPEAT:
        return "repeat";
    case OpType::CONCAT:
        return "concat";
    case OpType::PAD:
        return "pad";
    case OpType::SILU:
        return "silu";
    case OpType::GELU:
        return "gelu";
    case OpType::GELU_TANH:
        return "gelu_tanh";
    case OpType::RELU:
        return "relu";
    case OpType::NEG:
        return "neg";
    case OpType::SQR:
        return "sqr";
    case OpType::SQRT:
        return "sqrt";
    case OpType::EXP:
        return "exp";
    case OpType::LOG:
        return "log";
    case OpType::ADD:
        return "add";
    case OpType::SUB:
        return "sub";
    case OpType::MUL:
        return "mul";
    case OpType::DIV:
        return "div";
    case OpType::SCALE:
        return "scale";
    case OpType::RMS_NORM:
        return "rms_norm";
    case OpType::LAYER_NORM:
        return "layer_norm";
    case OpType::GROUP_NORM:
        return "group_norm";
    case OpType::MUL_MAT:
        return "mul_mat";
    case OpType::MUL_MAT_TRANSB:
        return "mul_mat_transB";
    case OpType::MUL_MAT_ID:
        return "mul_mat_id";
    case OpType::OUT_PROD:
        return "out_prod";
    case OpType::SOFT_MAX:
        return "soft_max";
    case OpType::SOFT_MAX_MASKED:
        return "soft_max_masked";
    case OpType::FLASH_ATTN_EXT:
        return "flash_attn_ext";
    case OpType::FLASH_ATTN_GQA:
        return "flash_attn_gqa";
    case OpType::ROPE:
        return "rope";
    case OpType::EMBEDDING:
        return "embedding";
    case OpType::SUM:
        return "sum";
    case OpType::MEAN:
        return "mean";
    case OpType::ARGMAX:
        return "argmax";
    case OpType::GLU:
        return "glu";
    case OpType::CPY_K:
        return "cpy_k";
    case OpType::CPY_V:
        return "cpy_v";
    case OpType::GET_K:
        return "get_k";
    case OpType::GET_V:
        return "get_v";
    case OpType::CUSTOM:
        return "custom";
    case OpType::COUNT:
        return "count";
    }
    return "unknown";
}

}  // namespace forge
