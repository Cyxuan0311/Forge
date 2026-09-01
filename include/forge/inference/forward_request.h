#pragma once

// ForwardRequest: 单次前向推理的显式请求对象。
//
// 重构前 start_pos / seq_id / seq_len 在 engine 的多层函数间以位置参数逐层传递,
// 任何一层遗漏或写错都不会被编译器发现。ForwardRequest 把这些运行时语义收敛到一个
// 结构体中, 让后续阶段(LayerExecutionContext / ExecutionPlan / GraphCache)可以直接
// 引用同一份请求状态, 而不是重新推导。
//
// 本阶段不改变任何执行语义: 旧的 forward(input_ids, start_pos, seq_id) 接口保留,
// 内部转换为 ForwardRequest 后再进入 engine 内部路径。

#include <cstdint>

#include "forge/tensor.h"

namespace forge {

struct ForwardRequest {
    // 输入 token id。从 hidden state 直接进入时(forward_from_hidden)为空。
    TensorPtr input_ids;

    // 已算好的 embedding（跳过 embedding lookup，直接进入 forward_layers）。
    // 用于 DFlash 的 embd batch 注入 / 共享 target embedding 的草稿输入。
    TensorPtr input_embeddings;

    // 从已算好的 embedding 构造请求（跳过 token embedding lookup）。
    static ForwardRequest from_embedding(const TensorPtr& emb, int n_tokens, int64_t start_pos,
                                         int seq_id = 0) {
        ForwardRequest req;
        req.input_embeddings = emb;
        req.n_tokens = n_tokens;
        req.start_pos = start_pos;
        req.seq_id = seq_id;
        req.is_prefill = n_tokens > 1;
        return req;
    }

    // 本次前向处理的 token 数量。
    // 注意: 不从 input_ids 推导, 因为 hidden-state 入口没有 input_ids。
    int n_tokens = 0;

    // KV cache 中的起始位置。
    int64_t start_pos = 0;

    // 序列 ID, 用于 KV cache 隔离。
    int seq_id = 0;

    // 是否为 prefill 阶段(多 token)。由构造函数按 n_tokens 推导, 可显式覆盖。
    bool is_prefill = false;

    // 是否返回所有位置的 logits, 而不是仅最后一个位置。
    bool return_all_logits = false;

    // 本次请求结束后的位置, 便于调用方推进 cursor。
    int64_t end_pos() const { return start_pos + static_cast<int64_t>(n_tokens); }

    // 从 token id tensor 构造请求(标准 forward 入口)。
    static ForwardRequest from_ids(const TensorPtr& ids, int64_t start_pos, int seq_id = 0) {
        ForwardRequest req;
        req.input_ids = ids;
        req.n_tokens = ids ? static_cast<int>(ids->numel()) : 0;
        req.start_pos = start_pos;
        req.seq_id = seq_id;
        req.is_prefill = req.n_tokens > 1;
        return req;
    }

    // 从已有 hidden state 构造请求(forward_from_hidden / 多模态 embedding 入口)。
    static ForwardRequest from_hidden(int n_tokens, int64_t start_pos, int seq_id = 0) {
        ForwardRequest req;
        req.n_tokens = n_tokens;
        req.start_pos = start_pos;
        req.seq_id = seq_id;
        req.is_prefill = n_tokens > 1;
        return req;
    }
};

}  // namespace forge