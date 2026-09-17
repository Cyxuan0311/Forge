// BailingMoE3 GDN / KDA recurrent helpers (internal CPU ops). See gdn.h.

#include "gdn.h"

#include <cmath>
#include <cstring>

#include "forge/inference/tensor_device_utils.h"
#include "forge/quant_traits.h"

namespace forge {
namespace ops {

TensorPtr flatten_rows(const TensorPtr& t) {
    if (!t) return nullptr;
    auto cpu = ensure_cpu(t);
    const auto& sh = cpu->shape();
    if (sh.size() <= 1)
        return cpu;  // already flat
    int64_t a = sh[0];
    int64_t rest = 1;
    for (size_t i = 1; i < sh.size(); ++i) rest *= sh[i];
    // Reshape preserving dtype + storage (no copy). The source may be quantized;
    // reinterpreting its packed bytes as fp32 would read past the buffer (heap
    // corruption) — matmul_transB dequantizes quantized inputs itself.
    Tensor view_t = cpu->view(std::vector<int64_t>{a, rest});
    return std::make_shared<Tensor>(std::move(view_t));
}

// Weight layout must be [n_expert, rows, cols] (per-expert matrix is row-major
// [rows, cols]), which is how the GGUF loader hands us ffn_{gate,up,down}_exps.
// Note Forge stores tensor shapes outermost-first (i.e. reversed vs. the GGUF
// ne[] array), so this is shape[0] == n_expert, e.g. [128, 512, 1536].
TensorPtr extract_expert(const TensorPtr& t, int e, int n_expert) {
    if (!t) return nullptr;
    auto cpu = ensure_cpu(t);
    const auto& sh = cpu->shape();
    if (sh.size() < 3 || (n_expert > 0 && sh[0] != n_expert)) {
        // Callers must not pre-flatten: flatten_rows() turns [n_expert, rows, cols]
        // into [n_expert, rows*cols] and the per-expert shape is lost.
        return nullptr;
    }
    if (e < 0 || e >= sh[0]) return nullptr;
    const int64_t rows = sh[1];
    const int64_t cols = sh[2];

    auto out = std::make_shared<Tensor>(DataType::FP32,
                                        std::vector<int64_t>{rows, cols},
                                        DeviceType::CPU);
    float* od = static_cast<float*>(out->data());

    // Never reinterpret a quantized weight's bytes as float*: expert tensors are
    // IQ2_XS / IQ3_S and ensure_cpu() leaves them packed. Dequantize just this
    // expert's rows (experts are stored contiguously) into FP32.
    const DataType dt = cpu->dtype();
    if (dt == DataType::FP32) {
        const float* p = static_cast<const float*>(cpu->data()) + static_cast<int64_t>(e) * rows * cols;
        std::memcpy(od, p, static_cast<size_t>(rows) * cols * sizeof(float));
        return out;
    }

    auto row_fn = get_dequant_row_fn(dt);
    if (!row_fn)
        return nullptr;
    const int64_t block_el = dtype_block_elements(dt);
    const int64_t block_sz = dtype_block_size(dt);
    const int64_t blocks_per_row = (cols + block_el - 1) / block_el;
    const int64_t row_bytes = blocks_per_row * block_sz;
    const uint8_t* base = static_cast<const uint8_t*>(cpu->data());
    for (int64_t r = 0; r < rows; ++r) {
        row_fn(base + (static_cast<int64_t>(e) * rows + r) * row_bytes, od + r * cols,
               static_cast<int>(cols), 0);
    }
    return out;
}

void apply_rope_interleaved(float* vec, int dim, int64_t pos, float theta) {
    // Standard interleaved RoPE: for pair index d (i = 2d stepping by 2),
    //   freq = theta^(-2d/n_dims) = theta^(-i/dim)
    // matching llama.cpp's theta_scale = pow(freq_base, -2/n_dims) and Forge's
    // own rope.cpp / rope_executor.cpp (1/theta^(2d/head_dim)). Scaling by the
    // element index already doubles the pair index, so no extra factor of 2.
    for (int i = 0; i + 1 < dim; i += 2) {
        const float freq = std::pow(theta, -static_cast<float>(i) / static_cast<float>(dim));
        float angle = static_cast<float>(pos) * freq;
        float c = std::cos(angle);
        float s = std::sin(angle);
        float a = vec[i];
        float b = vec[i + 1];
        vec[i] = a * c - b * s;
        vec[i + 1] = a * s + b * c;
    }
}

void sigmoid_inplace(const TensorPtr& x) {
    if (!x)
        return;
    auto cpu = ensure_cpu(x);
    float* p = static_cast<float*>(cpu->data());
    const int64_t n = cpu->numel();
    for (int64_t i = 0; i < n; ++i) p[i] = 1.0f / (1.0f + std::exp(-p[i]));
}

void build_decay_gate(const TensorPtr& logits, const TensorPtr& ssm_dt_bias,
                      const TensorPtr& ssm_a, int seq_len, int num_heads, int head_dim,
                      float gate_lower_bound) {
    if (!logits)
        return;
    auto lc = ensure_cpu(logits);
    const int64_t d_inner = static_cast<int64_t>(num_heads) * head_dim;
    float* d = static_cast<float*>(lc->data());

    std::vector<float> bias(d_inner, 0.0f);
    if (ssm_dt_bias) {
        const float* b = static_cast<const float*>(ensure_cpu(ssm_dt_bias)->data());
        for (int64_t i = 0; i < d_inner; ++i) bias[i] = b[i];
    }
    std::vector<float> a(num_heads, 1.0f);
    if (ssm_a) {
        const float* av = static_cast<const float*>(ensure_cpu(ssm_a)->data());
        for (int h = 0; h < num_heads; ++h) a[h] = av[h];
    }

    for (int t = 0; t < seq_len; ++t) {
        float* row = d + static_cast<int64_t>(t) * d_inner;
        for (int h = 0; h < num_heads; ++h) {
            const float ah = a[h];
            float* p = row + static_cast<int64_t>(h) * head_dim;
            const float* pb = bias.data() + static_cast<int64_t>(h) * head_dim;
            for (int dd = 0; dd < head_dim; ++dd) {
                const float f = p[dd] + pb[dd];
                p[dd] = gate_lower_bound / (1.0f + std::exp(-(ah * f)));
            }
        }
    }
}

// x / sqrt(sum(x^2) + eps) applied per (token, head) — llama.cpp's
// build_gdn_l2_norm = rms_norm(x, eps/n) / sqrt(n).
static void l2_norm_heads(float* x, int seq_len, int num_heads, int head_dim, float eps) {
    const int64_t d_inner = static_cast<int64_t>(num_heads) * head_dim;
    for (int t = 0; t < seq_len; ++t) {
        float* row = x + static_cast<int64_t>(t) * d_inner;
        for (int h = 0; h < num_heads; ++h) {
            float* p = row + static_cast<int64_t>(h) * head_dim;
            float nrm = 0.0f;
            for (int d = 0; d < head_dim; ++d) nrm += p[d] * p[d];
            const float inv = 1.0f / std::sqrt(nrm + eps);
            for (int d = 0; d < head_dim; ++d) p[d] *= inv;
        }
    }
}

TensorPtr gdn_recurrence(const TensorPtr& q, const TensorPtr& k, const TensorPtr& v,
                         const TensorPtr& decay, const TensorPtr& beta, float* state,
                         int seq_len, int num_heads, int head_dim, float eps) {
    if (!q || !k || !v || !decay || !beta || !state)
        return nullptr;
    auto qc = ensure_cpu(q);
    auto kc = ensure_cpu(k);
    auto vc = ensure_cpu(v);
    auto dc = ensure_cpu(decay);
    auto bc = ensure_cpu(beta);

    const int64_t d_inner = static_cast<int64_t>(num_heads) * head_dim;
    float* qd = static_cast<float*>(qc->data());
    float* kd = static_cast<float*>(kc->data());
    const float* vd = static_cast<const float*>(vc->data());
    const float* gd = static_cast<const float*>(dc->data());
    const float* b = static_cast<const float*>(bc->data());

    l2_norm_heads(qd, seq_len, num_heads, head_dim, eps);
    l2_norm_heads(kd, seq_len, num_heads, head_dim, eps);

    auto out = std::make_shared<Tensor>(DataType::FP32,
                                        std::vector<int64_t>{seq_len, d_inner},
                                        DeviceType::CPU);
    float* od = static_cast<float*>(out->data());

    // Attention scores are scaled by 1/sqrt(head_dim) (ggml's fused op applies it
    // to the output rather than pre-scaling q).
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t state_size = static_cast<int64_t>(head_dim) * head_dim;
    std::vector<float> exp_decay(head_dim);
    std::vector<float> delta(head_dim);

    for (int h = 0; h < num_heads; ++h) {
        // S[i][j] lives at S[j * head_dim + i] (transposed, like ggml).
        float* S = state + static_cast<int64_t>(h) * state_size;
        for (int t = 0; t < seq_len; ++t) {
            const float* gt = gd + static_cast<int64_t>(t) * d_inner + h * head_dim;
            const float* qt = qd + static_cast<int64_t>(t) * d_inner + h * head_dim;
            const float* kt = kd + static_cast<int64_t>(t) * d_inner + h * head_dim;
            const float* vt = vd + static_cast<int64_t>(t) * d_inner + h * head_dim;
            float* ot = od + static_cast<int64_t>(t) * d_inner + h * head_dim;
            const float bt = b[static_cast<int64_t>(t) * num_heads + h];

            // 1) S[i][j] *= exp(decay[i])
            for (int i = 0; i < head_dim; ++i) exp_decay[i] = std::exp(gt[i]);
            for (int j = 0; j < head_dim; ++j) {
                float* Sj = S + static_cast<int64_t>(j) * head_dim;
                for (int i = 0; i < head_dim; ++i) Sj[i] *= exp_decay[i];
            }

            // 2) delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
            for (int j = 0; j < head_dim; ++j) {
                const float* Sj = S + static_cast<int64_t>(j) * head_dim;
                float dot = 0.0f;
                for (int i = 0; i < head_dim; ++i) dot += Sj[i] * kt[i];
                delta[j] = (vt[j] - dot) * bt;
            }

            // 3) S[i][j] += k[i] * delta[j]
            for (int j = 0; j < head_dim; ++j) {
                float* Sj = S + static_cast<int64_t>(j) * head_dim;
                const float dj = delta[j];
                for (int i = 0; i < head_dim; ++i) Sj[i] += kt[i] * dj;
            }

            // 4) out[j] = (sum_i S[i][j]*q[i]) * scale
            for (int j = 0; j < head_dim; ++j) {
                const float* Sj = S + static_cast<int64_t>(j) * head_dim;
                float dot = 0.0f;
                for (int i = 0; i < head_dim; ++i) dot += Sj[i] * qt[i];
                ot[j] = dot * scale;
            }
        }
    }
    return out;
}

TensorPtr gdn_output_gate(const TensorPtr& core_out, const TensorPtr& out_gate_logits,
                          const TensorPtr& ssm_norm, int seq_len, int num_heads,
                          int head_dim, float eps) {
    if (!core_out || !out_gate_logits)
        return nullptr;
    auto cc = ensure_cpu(core_out);
    auto gc = ensure_cpu(out_gate_logits);
    const int64_t d_inner = static_cast<int64_t>(num_heads) * head_dim;

    std::vector<float> nw(head_dim, 1.0f);
    if (ssm_norm) {
        const float* w = static_cast<const float*>(ensure_cpu(ssm_norm)->data());
        for (int d = 0; d < head_dim; ++d) nw[d] = w[d];
    }

    const float* cd = static_cast<const float*>(cc->data());
    const float* gd = static_cast<const float*>(gc->data());
    auto out = std::make_shared<Tensor>(DataType::FP32,
                                        std::vector<int64_t>{seq_len, d_inner},
                                        DeviceType::CPU);
    float* od = static_cast<float*>(out->data());

    for (int t = 0; t < seq_len; ++t) {
        const float* ct = cd + static_cast<int64_t>(t) * d_inner;
        const float* gt = gd + static_cast<int64_t>(t) * d_inner;
        float* ot = od + static_cast<int64_t>(t) * d_inner;
        for (int h = 0; h < num_heads; ++h) {
            const float* cp = ct + static_cast<int64_t>(h) * head_dim;
            const float* gp = gt + static_cast<int64_t>(h) * head_dim;
            float* op = ot + static_cast<int64_t>(h) * head_dim;
            // RMSNorm over the head's own head_dim entries (shared ssm_norm weight).
            float ms = 0.0f;
            for (int d = 0; d < head_dim; ++d) ms += cp[d] * cp[d];
            const float inv = 1.0f / std::sqrt(ms / static_cast<float>(head_dim) + eps);
            for (int d = 0; d < head_dim; ++d)
                op[d] = cp[d] * inv * nw[d] / (1.0f + std::exp(-gp[d]));
        }
    }
    return out;
}

}  // namespace ops
}  // namespace forge
