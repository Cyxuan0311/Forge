#include "forge/engines/spark25_engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/inference/layers/ffn_executor.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

namespace forge {

namespace {

// Split-half partial RoPE: 旋转前 n_rot 维 (配对 x[d], x[d + n_rot/2]),
// dims [n_rot, head_dim) 原样拷贝。Spark-X2.5 全注意力层 n_rot=64 (head_dim=256),
// SWA 层 n_rot=256 (= head_dim, 全旋转)。
void rope_partial_cpu(const float* x, float* out, int seq_len, int num_heads, int head_dim,
                      int n_rot, int64_t pos, float theta) {
    const int half_rot = n_rot / 2;
    const int stride = num_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        const int64_t p = pos + s;
        for (int h = 0; h < num_heads; ++h) {
            float* head_out = out + (size_t)s * stride + h * head_dim;
            const float* head_in = x + (size_t)s * stride + h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                if (d >= n_rot) {
                    head_out[d] = head_in[d];
                } else if (d < half_rot) {
                    float freq = 1.0f / std::pow(theta, 2.0f * d / static_cast<float>(n_rot));
                    float angle = static_cast<float>(p) * freq;
                    float c = std::cos(angle);
                    float sn = std::sin(angle);
                    float x0 = head_in[d];
                    float x1 = head_in[d + half_rot];
                    head_out[d] = x0 * c - x1 * sn;
                    head_out[d + half_rot] = x0 * sn + x1 * c;
                }
                // d in [half_rot, n_rot): 已由配对 owner 写入
            }
        }
    }
}

// Per-head sigmoid 门控 (原地): attn[s, h*head_dim+d] *= sigmoid(gate[s, h])。
// sigmoid 融合在乘法里 (ops::sigmoid 未实现)。
void headwise_sigmoid_gate_mul_cpu(float* attn, const float* gate, int seq_len, int num_heads,
                                   int head_dim) {
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            float gv = gate[s * num_heads + h];
            float g = 1.0f / (1.0f + std::exp(-gv));
            float* head = attn + (size_t)s * num_heads * head_dim + h * head_dim;
            for (int d = 0; d < head_dim; ++d)
                head[d] *= g;
        }
    }
}

// ---- 调试插桩 (FORGE_DEBUG_SPARK=1; FORGE_DEBUG_SPARK_LAYER=N 选层, -1=全部) ----
static int g_spark_dbg = -1;
static bool spark_dbg_on() {
    if (g_spark_dbg < 0) {
        const char* e = std::getenv("FORGE_DEBUG_SPARK");
        g_spark_dbg = (e && std::atoi(e) != 0) ? 1 : 0;
    }
    return g_spark_dbg == 1;
}
static int spark_dbg_layer() {
    static int l = -2;
    if (l == -2) {
        const char* e = std::getenv("FORGE_DEBUG_SPARK_LAYER");
        l = e ? std::atoi(e) : 0;  // 默认只打第 0 层
    }
    return l;
}

static void spark_dump(const char* name, int layer, const TensorPtr& t) {
    if (!spark_dbg_on())
        return;
    int want = spark_dbg_layer();
    if (want >= 0 && layer != want && layer != -99)
        return;  // -99 用于 "始终打印最后一层"
    auto cpu = ensure_cpu(t);
    if (cpu->dtype() != DataType::FP32)
        return;
    const float* d = static_cast<const float*>(cpu->data());
    int64_t n = 1;
    for (auto s : cpu->shape())
        n *= s;
    if (n == 0)
        return;
    double mn = 1e30, mx = -1e30, sum = 0, sum2 = 0;
    long nan = 0;
    for (int64_t i = 0; i < n; ++i) {
        float v = d[i];
        if (std::isnan(v) || std::isinf(v)) {
            nan++;
            continue;
        }
        if (v < mn)
            mn = v;
        if (v > mx)
            mx = v;
        sum += v;
        sum2 += v * v;
    }
    double mean = sum / n;
    double std = std::sqrt(std::max(0.0, sum2 / n - mean * mean));
    fprintf(stderr,
            "[SPARK-DBG] %-16s L%2d n=%lld min=%.4g max=%.4g mean=%.4g std=%.4g nan=%ld\n",
            name, layer, (long long)n, mn, mx, mean, std, nan);
}

}  // namespace

Spark25Engine::Spark25Engine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx) {
    if (!init_weights()) {
        throw std::runtime_error("Spark25Engine: failed to initialize weights");
    }
}

bool Spark25Engine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

void Spark25Engine::init_kv_cache(const ModelConfig& cfg) {
    if (kv_cache_initialized())
        return;

    if (kv_cache_.nbytes() > 0 && kv_cache_.max_seq_len() > 0) {
        LOG_INFO("KV cache reusing resident buffers: max_seq_len=" +
                 std::to_string(kv_cache_.max_seq_len()) + ", " +
                 std::to_string(kv_cache_.nbytes() / (1024 * 1024)) + " MB");
        set_kv_cache_initialized(true);
        return;
    }

    int kv_max_seq = kv_cache_max_seq_len_ > 0 ? kv_cache_max_seq_len_ : cfg.max_seq_len;
    const int KV_MAX_SEQ_CAP = 4096;
    if (kv_cache_max_seq_len_ <= 0 && kv_max_seq > KV_MAX_SEQ_CAP) {
        LOG_INFO("Capping KV cache max_seq_len from " + std::to_string(kv_max_seq) + " to " +
                 std::to_string(KV_MAX_SEQ_CAP) + " to avoid OOM");
        kv_max_seq = KV_MAX_SEQ_CAP;
    }
    if (kv_max_seq < cfg.n_swa) {
        // SWA ring buffer 至少要容纳一个完整窗口
        kv_max_seq = cfg.n_swa;
    }

    DeviceType kv_dev;
    if (ctx_.params().offload_kqv) {
        bool all_layers_cuda = !layer_devices_.empty();
        for (const auto& t : layer_devices_) {
            if (!t.is_cuda()) {
                all_layers_cuda = false;
                break;
            }
        }
        kv_dev = all_layers_cuda ? DeviceType::CUDA : DeviceType::CPU;
    } else {
        kv_dev = DeviceType::CPU;
    }

    // 所有层同几何: n_kv_heads * head_dim (4 * 256)
    std::vector<int> kv_dims(cfg.num_layers, cfg.num_kv_heads * cfg.head_dim);

    LOG_INFO("KV cache init (Spark25 per-layer): layers=" + std::to_string(cfg.num_layers) +
             ", max_seq_len=" + std::to_string(kv_max_seq) +
             ", dev=" + (kv_dev == DeviceType::CUDA ? "CUDA" : "CPU") +
             ", n_swa=" + std::to_string(cfg.n_swa));

    kv_cache_.init_per_layer(cfg.num_layers, kv_dims, kv_max_seq, kv_dev);

    // SWA 层用 ring buffer 逐出; 全注意力层线性增长
    if (cfg.n_swa > 0) {
        std::vector<KVLayerPolicy> policies(cfg.num_layers, KVLayerPolicy::Full);
        for (int i = 0; i < cfg.num_layers; ++i) {
            if (i < (int)cfg.swa_layers.size() && cfg.swa_layers[i] == 1)
                policies[i] = KVLayerPolicy::SlidingWindow;
        }
        kv_cache_.set_layer_policies(policies, cfg.n_swa);
    }
    if (!layer_devices_.empty()) {
        kv_cache_.set_layer_devices(layer_devices_);
    }
    set_kv_cache_initialized(true);

    LOG_INFO("KV cache initialized successfully, actual size: " +
             std::to_string(kv_cache_.nbytes() / (1024 * 1024)) + " MB");
}

TensorPtr Spark25Engine::forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int layer_idx = lctx.layer_idx;
    const int seq_len = lctx.seq_len();
    const int64_t rope_pos = lctx.rope_pos();
    const DeviceType dev = lctx.device.type;
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;

    bool is_swa = (layer_idx < (int)cfg.swa_layers.size() && cfg.swa_layers[layer_idx] == 1);
    int n_rot = is_swa ? (cfg.rope_dim_count_swa > 0 ? cfg.rope_dim_count_swa : cfg.head_dim)
                       : (cfg.rope_dim_count > 0 ? cfg.rope_dim_count : cfg.head_dim);
    float theta = is_swa ? cfg.rope_theta_swa : cfg.rope_theta;

    // ---- 调试插桩：本层是否需要 dump ----
    auto dbg_want = [&]() -> bool {
        if (!spark_dbg_on())
            return false;
        int want = spark_dbg_layer();
        return want < 0 || layer_idx == want || layer_idx == cfg.num_layers - 1;
    };
    auto D = [&](const char* name, const TensorPtr& t) {
        if (dbg_want())
            spark_dump(name, layer_idx, t);
    };
    if (dbg_want()) {
        fprintf(stderr,
                "[SPARK-DBG] === L%2d is_swa=%d n_rot=%d theta=%.4g rope_pos=%lld seq_len=%d "
                "heads=%d kv=%d hd=%d ===\n",
                layer_idx, is_swa, n_rot, theta, (long long)rope_pos, seq_len, num_heads,
                num_kv_heads, head_dim);
        D("hidden_in", hidden);
    }

    // ---- 1. Pre-attention norm (门控输入也用它) ----
    TensorPtr normed;
    {
        PERF_SCOPE("layer/attn_norm");
        normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);
    }
    D("normed", normed);

    // ---- 2. QKV projection (loader 已把 fused attn_qkv 切分为 wq/wk/wv) ----
    TensorPtr q, k, v;
    {
        PERF_SCOPE("layer/qkv_proj");
        q = ops::matmul_transB(normed, lw.wq());
        k = ops::matmul_transB(normed, lw.wk());
        v = ops::matmul_transB(normed, lw.wv());
    }
    D("q", q);
    D("k", k);
    D("v", v);

    // ---- 3. RoPE (每层 n_rot / theta 不同) ----
    TensorPtr q_rope, k_rope;
    {
        PERF_SCOPE("layer/rope");
#ifdef USE_CUDA
        if (dev == DeviceType::CUDA && q->device() == DeviceType::CUDA &&
            k->device() == DeviceType::CUDA) {
            q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CUDA);
            k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CUDA);
            cuda::launch_rope_partial_gqa(static_cast<const float*>(q->data()),
                                          static_cast<const float*>(k->data()),
                                          static_cast<float*>(q_rope->data()),
                                          static_cast<float*>(k_rope->data()), num_heads,
                                          num_kv_heads, head_dim, n_rot, seq_len, rope_pos, theta);
        } else
#endif
        {
            q = ensure_cpu(q);
            k = ensure_cpu(k);
            q_rope = std::make_shared<Tensor>(DataType::FP32, q->shape(), DeviceType::CPU);
            k_rope = std::make_shared<Tensor>(DataType::FP32, k->shape(), DeviceType::CPU);
            rope_partial_cpu(static_cast<const float*>(q->data()),
                             static_cast<float*>(q_rope->data()), seq_len, num_heads, head_dim,
                             n_rot, rope_pos, theta);
            rope_partial_cpu(static_cast<const float*>(k->data()),
                             static_cast<float*>(k_rope->data()), seq_len, num_kv_heads, head_dim,
                             n_rot, rope_pos, theta);
            if (dev == DeviceType::CUDA) {
                auto q_dev = std::make_shared<Tensor>(DataType::FP32, q_rope->shape(),
                                                      DeviceType::CUDA);
                auto k_dev = std::make_shared<Tensor>(DataType::FP32, k_rope->shape(),
                                                      DeviceType::CUDA);
                q_dev->copy_from(*q_rope);
                k_dev->copy_from(*k_rope);
                q_rope = q_dev;
                k_rope = k_dev;
            }
        }
    }

    D("q_rope", q_rope);
    D("k_rope", k_rope);

    // ---- 4. KV cache update ----
    {
        PERF_SCOPE("layer/kv_cache_update");
        if (kv_cache_.update(layer_idx, lctx.seq_id(), lctx.start_pos(), k_rope, v, seq_len) < 0) {
            throw std::runtime_error(
                "KV cache overflow at layer " + std::to_string(layer_idx) +
                ": write_pos=" + std::to_string(lctx.start_pos()) +
                " seq_len=" + std::to_string(seq_len) +
                " exceeds max_seq_len=" + std::to_string(kv_cache_.max_seq_len()));
        }
    }

    // ---- 5. Attention ----
    // SWA 的滑动窗口由 KV cache 的 ring buffer 处理, get_key_filled()/filled()
    // 已只返回最后 window_size 个位置, 这里不再手工切片 (同 Gemma4Engine)。
    TensorPtr attn_out;
    {
        PERF_SCOPE("layer/attention");
        int total_len = kv_cache_.filled(layer_idx);
        auto k_c = kv_cache_.get_key_filled(layer_idx);
        auto v_c = kv_cache_.get_value_filled(layer_idx);

        if (dev == DeviceType::CUDA && k_c && k_c->device() == DeviceType::CPU) {
            auto k_cuda = std::make_shared<Tensor>(DataType::FP32, k_c->shape(), DeviceType::CUDA);
            k_cuda->copy_from(*k_c);
            k_c = k_cuda;
            auto v_cuda = std::make_shared<Tensor>(DataType::FP32, v_c->shape(), DeviceType::CUDA);
            v_cuda->copy_from(*v_c);
            v_c = v_cuda;
        }

        attn_out = ops::scaled_dot_product_attention_2d_gqa(q_rope, k_c, v_c, seq_len, total_len,
                                                            num_heads, num_kv_heads, head_dim,
                                                            nullptr, true);
    }
    D("attn_out", attn_out);

    // ---- 6. Per-head sigmoid 输出门控 (门控输入 = 归一化后的输入) ----
    {
        PERF_SCOPE("layer/attn_gate");
        auto gate = ops::matmul_transB(normed, lw.attn_gate());  // [T, n_head] F32
        D("gate_pre", gate);
        if (dbg_want()) {
            auto g = ensure_cpu(gate);
            const float* gd = static_cast<const float*>(g->data());
            int64_t gn = 1;
            for (auto s : g->shape())
                gn *= s;
            double ssum = 0;
            for (int64_t i = 0; i < gn; ++i)
                ssum += 1.0 / (1.0 + std::exp(-gd[i]));
            fprintf(stderr, "[SPARK-DBG] gate_sigmoid_mean=%.4g (n=%lld)\n", ssum / gn,
                    (long long)gn);
        }
#ifdef USE_CUDA
        if (dev == DeviceType::CUDA && attn_out->device() == DeviceType::CUDA &&
            gate->device() == DeviceType::CUDA) {
            cuda::launch_headwise_sigmoid_gate_mul(static_cast<float*>(attn_out->data()),
                                                   static_cast<const float*>(gate->data()),
                                                   seq_len, num_heads, head_dim);
        } else
#endif
        {
            attn_out = ensure_cpu(attn_out);
            gate = ensure_cpu(gate);
            headwise_sigmoid_gate_mul_cpu(static_cast<float*>(attn_out->data()),
                                          static_cast<const float*>(gate->data()), seq_len,
                                          num_heads, head_dim);
            if (dev == DeviceType::CUDA) {
                auto attn_dev =
                    std::make_shared<Tensor>(DataType::FP32, attn_out->shape(), DeviceType::CUDA);
                attn_dev->copy_from(*attn_out);
                attn_out = attn_dev;
            }
        }
    }

    D("attn_out_gated", attn_out);

    // ---- 7. Output projection + 残差 ----
    TensorPtr attn_proj;
    {
        PERF_SCOPE("layer/attn_proj");
        attn_proj = ops::matmul_transB(attn_out, lw.wo());
    }
    D("attn_proj", attn_proj);
    auto ffn_inp = ops::add(hidden, attn_proj);
    D("ffn_inp", ffn_inp);

    // ---- 8. FFN (gelu(gate)*up, PAR 结构 → 残差输入是 ffn_inp) ----
    TensorPtr ffn_normed;
    {
        PERF_SCOPE("layer/ffn_norm");
        ffn_normed = ops::rms_norm(ffn_inp, lw.ffn_norm(), cfg.rms_norm_eps);
    }
    D("ffn_normed", ffn_normed);
    TensorPtr ffn_out;
    {
        PERF_SCOPE("layer/ffn");
        ffn_out = FfnExecutor::apply(ffn_normed, ffn_inp, cfg, lw, seq_len, dev);
    }
    D("ffn_out", ffn_out);

    if (FfnExecutor::residual_fused(cfg, lw, seq_len, dev)) {
        D("layer_out", ffn_out);
        return ffn_out;  // 融合 kernel 已加残差
    }
    auto out = ops::add(ffn_inp, ffn_out);
    D("layer_out", out);
    return out;
}

}  // namespace forge
