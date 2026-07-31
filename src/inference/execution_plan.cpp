#include "forge/inference/execution_plan.h"

#include <stdexcept>

#include "forge/inference/graph/layer_graph_builder.h"
#include "forge/logger.h"
#include "forge/op_dispatch.h"

namespace forge {

namespace {

const char* to_string(EngineKind k) {
    switch (k) {
        case EngineKind::Generic:      return "generic";
        case EngineKind::DeepSeekMLA:  return "deepseek_mla";
        case EngineKind::Qwen35Hybrid: return "qwen35_hybrid";
        case EngineKind::Gemma4:       return "gemma4";
    }
    return "?";
}

const char* to_string(AttentionKind k) {
    switch (k) {
        case AttentionKind::GQA:       return "gqa";
        case AttentionKind::MLA:       return "mla";
        case AttentionKind::HybridSSM: return "hybrid_ssm";
    }
    return "?";
}

const char* to_string(FfnKind k) {
    switch (k) {
        case FfnKind::SiLUGated: return "silu_gated";
        case FfnKind::GeGLU:     return "geglu";
        case FfnKind::SimpleGELU:return "simple_gelu";
        case FfnKind::MoE:       return "moe";
    }
    return "?";
}

const char* to_string(RopeKind k) {
    switch (k) {
        case RopeKind::None:         return "none";
        case RopeKind::Standard:     return "standard";
        case RopeKind::NeoX:         return "neox";
        case RopeKind::MRoPE:        return "mrope";
        case RopeKind::Proportional: return "proportional";
    }
    return "?";
}

const char* to_string(MemoryKind k) {
    switch (k) {
        case MemoryKind::KV:         return "kv";
        case MemoryKind::MLA:        return "mla";
        case MemoryKind::Hybrid:     return "hybrid";
        case MemoryKind::PerLayerKV: return "per_layer_kv";
    }
    return "?";
}

FfnKind ffn_kind_from(const ModelConfig& cfg) {
    switch (cfg.ffn_type) {
        case FFNType::SiLUGated:  return FfnKind::SiLUGated;
        case FFNType::GeGLU:      return FfnKind::GeGLU;
        case FFNType::SimpleGELU: return FfnKind::SimpleGELU;
        case FFNType::MoE:        return FfnKind::MoE;
    }
    return FfnKind::SiLUGated;
}

// RoPE 种类的判定顺序必须与 RopeExecutor::apply() 一致, 否则 plan 与实际执行漂移。
RopeKind rope_kind_from(const ModelConfig& cfg) {
    if (cfg.rope_type == RopeType::None)
        return RopeKind::None;
    if (cfg.use_mrope || cfg.rope_type == RopeType::MRoPE)
        return RopeKind::MRoPE;
    if (cfg.rope_type == RopeType::Proportional)
        return RopeKind::Proportional;
    if (cfg.use_neox_rope || cfg.rope_type == RopeType::NeoX)
        return RopeKind::NeoX;
    return RopeKind::Standard;
}

}  // namespace

std::string ExecutionPlan::plan_id() const {
    std::string id = arch;
    id += "/";
    id += to_string(engine_kind);
    id += "+";
    id += to_string(attention_kind);
    id += "+";
    id += to_string(ffn_kind);
    id += "+";
    id += to_string(rope_kind);
    id += "+";
    id += to_string(memory_kind);
    if (norm_kind == NormType::LayerNorm) id += "+layernorm";
    if (has_qk_norm)              id += "+qk_norm";
    if (has_qkv_bias)             id += "+qkv_bias";
    if (has_norm_bias)            id += "+norm_bias";
    if (use_parallel_residual)    id += "+parallel_residual";
    if (use_embedding_scale)      id += "+emb_scale";
    if (has_post_attention_norm)  id += "+post_attn_norm";
    if (has_post_ffn_norm)        id += "+post_ffn_norm";
    return id;
}

const char* engine_name_for(EngineKind kind) {
    switch (kind) {
        case EngineKind::Generic:      return "llama";
        case EngineKind::DeepSeekMLA:  return "deepseek_v2";
        case EngineKind::Qwen35Hybrid: return "qwen35";
        case EngineKind::Gemma4:       return "gemma4";
    }
    return "llama";
}

ExecutionPlan build_execution_plan(const std::string& arch, const ModelConfig& cfg) {
    // 架构未注册时使用默认 ArchCapability: Python API 允许直接构造自定义 ModelConfig,
    // 此时所有语义都来自 cfg。启动期校验走 build_execution_plan_from_capability(),
    // 那里要求架构必须已注册。
    const ArchCapability cap = ArchCapabilityRegistry::instance().get(arch);

    ExecutionPlan plan;
    plan.arch = arch;

    // ---- attention / engine / memory ----
    // 三者一起决定, 不允许只看单个 bool 做 fallback。
    if (cap.use_ssm || cfg.use_ssm) {
        plan.attention_kind = AttentionKind::HybridSSM;
        plan.engine_kind = EngineKind::Qwen35Hybrid;
        plan.memory_kind = MemoryKind::Hybrid;
    } else if (cap.use_mla || cfg.use_mla) {
        plan.attention_kind = AttentionKind::MLA;
        plan.engine_kind = EngineKind::DeepSeekMLA;
        plan.memory_kind = MemoryKind::MLA;
    } else {
        plan.attention_kind = AttentionKind::GQA;
        // per-layer embedding + 部分层共享 KV 是 Gemma4 专有布局, 无法由 GenericEngine 承载。
        bool gemma4_layout = (cfg.n_embd_per_layer > 0) || (cfg.n_layer_kv_from_start > 0) ||
                             !cfg.swa_layers.empty();
        if (arch == "gemma4" || gemma4_layout) {
            plan.engine_kind = EngineKind::Gemma4;
            plan.memory_kind = MemoryKind::PerLayerKV;
        } else {
            plan.engine_kind = EngineKind::Generic;
            plan.memory_kind = MemoryKind::KV;
        }
    }

    // ---- 算子策略 ----
    plan.ffn_kind = ffn_kind_from(cfg);
    plan.rope_kind = rope_kind_from(cfg);
    plan.norm_kind = cfg.norm_type;
    plan.has_qk_norm = cfg.use_qk_norm || cap.use_qk_norm;
    plan.has_qkv_bias = cap.has_qkv_bias;
    plan.has_norm_bias = cap.has_norm_bias;
    plan.use_parallel_residual = cfg.use_parallel_residual || cap.use_parallel_residual;
    plan.use_embedding_scale = cap.embedding_scale;
    plan.has_post_attention_norm = cfg.has_post_attention_norm || cap.has_post_attention_norm;
    plan.has_post_ffn_norm = cfg.has_post_ffn_norm || cap.has_post_ffn_norm;

    // ---- graph 支持 ----
    // 直接持有 builder 实例, 执行期不再按架构名重新查表。
    // Qwen3.5 的 state graph 未完成, create_graph_builder 返回 nullptr。
    register_builtin_op_kernels();
    if (plan.engine_kind != EngineKind::Qwen35Hybrid) {
        plan.graph_builder = create_graph_builder(arch);
    }
    plan.supports_graph = plan.graph_builder != nullptr;

    // ---- 一致性校验: 计划矛盾必须在创建阶段暴露 ----
    if (plan.engine_kind == EngineKind::Generic) {
        if (plan.ffn_kind == FfnKind::MoE) {
            throw std::runtime_error("build_execution_plan: architecture '" + arch +
                                     "' requires MoE FFN, which GenericEngine cannot execute");
        }
        if (plan.memory_kind != MemoryKind::KV) {
            throw std::runtime_error("build_execution_plan: architecture '" + arch +
                                     "' requires non-standard KV memory, which GenericEngine "
                                     "cannot execute");
        }
    }
    if (plan.norm_kind == NormType::LayerNorm && plan.engine_kind != EngineKind::Generic) {
        throw std::runtime_error("build_execution_plan: architecture '" + arch +
                                 "' requires LayerNorm, which only GenericEngine supports");
    }
    if (plan.use_parallel_residual && plan.engine_kind != EngineKind::Generic) {
        throw std::runtime_error("build_execution_plan: architecture '" + arch +
                                 "' requires parallel residual, which only GenericEngine supports");
    }

    return plan;
}

ExecutionPlan build_execution_plan_from_capability(const std::string& arch) {
    auto& cap_registry = ArchCapabilityRegistry::instance();
    if (!cap_registry.has(arch)) {
        throw std::runtime_error("build_execution_plan: architecture '" + arch +
                                 "' has no registered ArchCapability");
    }
    const ArchCapability cap = cap_registry.get(arch);

    // 只填充 ArchCapability 能表达的字段。其余保持默认, 由 plan 构建逻辑处理。
    ModelConfig cfg;
    cfg.arch_type = arch;
    cfg.norm_type = cap.norm_type;
    cfg.ffn_activation = cap.ffn_activation;
    cfg.use_mla = cap.use_mla;
    cfg.use_ssm = cap.use_ssm;
    cfg.use_gqa = cap.use_gqa;
    cfg.use_mrope = cap.use_mrope;
    cfg.use_neox_rope = cap.use_neox_rope;
    cfg.use_qk_norm = cap.use_qk_norm;
    cfg.use_parallel_residual = cap.use_parallel_residual;
    cfg.has_post_attention_norm = cap.has_post_attention_norm;
    cfg.has_post_ffn_norm = cap.has_post_ffn_norm;
    switch (cap.ffn_activation) {
        case ActivationType::GeGLU: cfg.ffn_type = FFNType::GeGLU; break;
        case ActivationType::GELU:  cfg.ffn_type = FFNType::SimpleGELU; break;
        default:                    cfg.ffn_type = FFNType::SiLUGated; break;
    }

    return build_execution_plan(arch, cfg);
}

}  // namespace forge
