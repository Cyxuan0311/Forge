#pragma once

// ExecutionPlan: 模型「如何执行」的一次性决策结果。
//
// 重构前同一批语义分散在四处: ModelConfig 的 use_gqa/use_mla/use_ssm、
// ArchCapabilityRegistry 的 ArchCapability、engine_factory.cpp 的 EngineCapability
// 字符串表, 以及 engine 内部按 arch_type 的临时查表。任何一处漏改都会在 forward
// 阶段才暴露。ExecutionPlan 在 engine 创建时构建一次, 之后所有执行期代码只读 plan,
// 不再重新推导, 也不再按单个 bool 做 fallback。
//
// ExecutionPlan 只描述执行方式, 不负责执行。

#include <memory>
#include <string>

#include "forge/model.h"

namespace forge {

class LayerGraphBuilder;

// 由哪个 engine 类执行。Gemma4 只对应 arch "gemma4"(Gemma4Engine 仅为它注册)。
enum class EngineKind : int { Generic, DeepSeekMLA, Qwen35Hybrid, Gemma4, Phimoe };

enum class AttentionKind : int { GQA, MLA, HybridSSM };

enum class FfnKind : int { SiLUGated, GeGLU, SimpleGELU, MoE };

enum class RopeKind : int { None, Standard, NeoX, MRoPE, Proportional };

enum class MemoryKind : int { KV, MLA, Hybrid, PerLayerKV };

struct ExecutionPlan {
    EngineKind engine_kind = EngineKind::Generic;
    AttentionKind attention_kind = AttentionKind::GQA;
    FfnKind ffn_kind = FfnKind::SiLUGated;
    RopeKind rope_kind = RopeKind::Standard;
    MemoryKind memory_kind = MemoryKind::KV;

    NormType norm_kind = NormType::RMSNorm;
    bool has_qk_norm = false;
    bool has_qkv_bias = false;
    bool has_norm_bias = false;
    bool use_parallel_residual = false;
    bool use_embedding_scale = false;
    bool has_post_attention_norm = false;
    bool has_post_ffn_norm = false;

    // graph 执行能力。graph_builder 非空即代表支持, supports_graph 只是它的语义别名,
    // 不再是一份需要单独维护的架构名单。
    std::shared_ptr<LayerGraphBuilder> graph_builder;
    bool supports_graph = false;

    // 计划来源的架构名, 同时用作 graph cache key 的一部分。
    std::string arch;

    // 人类可读摘要, 用于日志和测试断言。
    std::string plan_id() const;
};

// 按架构名构建执行计划。cfg 提供运行期数值(FFN 类型、RoPE 类型、norm 类型等),
// ArchCapabilityRegistry 提供架构声明(embedding scale 等 cfg 中没有的语义)。
//
// 计划内部矛盾(例如要求 MoE 却落到 GenericEngine)会立即抛出, 不允许延迟到 forward
// 阶段才暴露。
ExecutionPlan build_execution_plan(const std::string& arch, const ModelConfig& cfg);

// 仅从 ArchCapability 推导一份最小 ModelConfig 并构建计划。用于启动期校验:
// 此时还没有真实模型, 只能验证架构声明本身与 engine 注册表是否自洽。
ExecutionPlan build_execution_plan_from_capability(const std::string& arch);

// EngineKind 对应的注册 engine 名, 供 EngineRegistry 在架构没有专用 engine 时解析。
const char* engine_name_for(EngineKind kind);

}  // namespace forge
