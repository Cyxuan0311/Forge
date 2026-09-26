// BailingMoE3 GDN / KDA recurrent helpers (internal CPU ops). See gdn.h.

#include "gdn.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "forge/inference/tensor_device_utils.h"
#include "forge/quant_traits.h"

// AVX2 vector primitives (forge::cpu::dot_f32 / scale_f32 / fmadd_f32 /
// mul_vec_f32) — scalar fallback provided when USE_AVX2 is not defined.
#include "simd.h"

namespace forge {
namespace ops {

TensorPtr flatten_rows(const TensorPtr& t) {
    if (!t) return nullptr;
    const auto& sh = t->shape();
    if (sh.size() <= 1)
        return t;  // already flat
    int64_t a = sh[0];
    int64_t rest = 1;
    for (size_t i = 1; i < sh.size(); ++i) rest *= sh[i];
    // Reshape preserving dtype + storage (no copy). The source may be quantized;
    // reinterpreting its packed bytes as fp32 would read past the buffer (heap
    // corruption) — matmul_transB dequantizes quantized inputs itself.
    //
    // 必须保留原设备: matmul_transB 只按输入 a 的设备派发 CUDA, 若权重视图被
    // 强制搬回 CPU 就会把主机指针当设备指针用 (illegal memory access)。
    Tensor view_t = t->view(std::vector<int64_t>{a, rest});
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

TensorPtr expert_quant_view(const TensorPtr& t, int e, int n_expert) {
    if (!t) return nullptr;
    // 与 flatten_rows 同理: 视图必须留在权重的原设备上, 否则 CUDA 激活 + CPU
    // 权重视图会被 matmul_transB 的 CUDA 分支当成设备指针 (illegal access)。
    const auto& sh = t->shape();
    if (sh.size() < 3 || (n_expert > 0 && sh[0] != n_expert)) {
        return nullptr;
    }
    if (e < 0 || e >= sh[0]) return nullptr;
    const int64_t rows = sh[1];
    const int64_t cols = sh[2];

    const DataType dt = t->dtype();
    int64_t row_bytes;
    if (is_quantized_type(dt)) {
        const int64_t block_el = dtype_block_elements(dt);
        const int64_t block_sz = dtype_block_size(dt);
        if (block_el <= 0 || block_sz <= 0) return nullptr;
        row_bytes = ((cols + block_el - 1) / block_el) * block_sz;
    } else {
        const size_t esz = dtype_size(dt);
        if (esz == 0) return nullptr;
        row_bytes = cols * static_cast<int64_t>(esz);
    }

    uint8_t* base = static_cast<uint8_t*>(t->data()) +
                    static_cast<int64_t>(e) * rows * row_bytes;
    // Borrowed: storage belongs to the layer weights, which outlive this view.
    Tensor view = Tensor::from_buffer(base, dt, {rows, cols}, t->device(), false);
    return std::make_shared<Tensor>(std::move(view));
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
            const float nrm = std::sqrt(forge::cpu::dot_f32(p, p, head_dim) + eps);
            forge::cpu::scale_f32(p, head_dim, 1.0f / nrm);
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

            // 1) S[i][j] *= exp(decay[i])  (per-column diagonal scaling)
            for (int i = 0; i < head_dim; ++i) exp_decay[i] = std::exp(gt[i]);
            for (int j = 0; j < head_dim; ++j) {
                float* Sj = S + static_cast<int64_t>(j) * head_dim;
                forge::cpu::mul_vec_f32(Sj, exp_decay.data(), head_dim);
            }

            // 2) delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
            for (int j = 0; j < head_dim; ++j) {
                const float* Sj = S + static_cast<int64_t>(j) * head_dim;
                const float dot = forge::cpu::dot_f32(Sj, kt, head_dim);
                delta[j] = (vt[j] - dot) * bt;
            }

            // 3) S[i][j] += k[i] * delta[j]
            for (int j = 0; j < head_dim; ++j) {
                float* Sj = S + static_cast<int64_t>(j) * head_dim;
                forge::cpu::fmadd_f32(Sj, kt, head_dim, delta[j]);
            }

            // 4) out[j] = (sum_i S[i][j]*q[i]) * scale
            for (int j = 0; j < head_dim; ++j) {
                const float* Sj = S + static_cast<int64_t>(j) * head_dim;
                ot[j] = forge::cpu::dot_f32(Sj, qt, head_dim) * scale;
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
            const float inv =
                1.0f / std::sqrt(forge::cpu::dot_f32(cp, cp, head_dim) /
                                     static_cast<float>(head_dim) +
                                 eps);
            for (int d = 0; d < head_dim; ++d)
                op[d] = cp[d] * inv * nw[d] / (1.0f + std::exp(-gp[d]));
        }
    }
    return out;
}

// ============================================================================
// Absorbed MLA attention (bailingmoe3)
// ============================================================================
TensorPtr bailingmoe3_mla_attention(const TensorPtr& q_flat,
                                    const TensorPtr& c_nope_norm,
                                    const TensorPtr& k_pe_rot,
                                    const TensorPtr& kb, const TensorPtr& v_all,
                                    const TensorPtr& gate, int start_pos,
                                    int seq_len, int total, int n_head,
                                    int kv_lora, int qk_nope_dim, int rope_dim,
                                    int v_dim, float theta, float scale) {
    const int qk_head_dim = qk_nope_dim + rope_dim;
    const int v_stride = n_head * v_dim;
    const int q_stride = n_head * qk_head_dim;

    // `k_pe_rot`/`v_all` arrive precomputed by the engine's incremental MLA
    // cache (rotated key per absolute position, and value contraction), so this
    // op only does the query-side contraction, causal softmax and context sum.
    auto qc = ensure_cpu(q_flat);
    auto cnc = ensure_cpu(c_nope_norm);
    auto kpr = ensure_cpu(k_pe_rot);
    auto kbc = ensure_cpu(kb);
    auto vad = ensure_cpu(v_all);
    auto gc = ensure_cpu(gate);
    const float* qf = static_cast<const float*>(qc->data());
    const float* cnp = static_cast<const float*>(cnc->data());
    const float* kpr_d = static_cast<const float*>(kpr->data());
    const float* kb_d = static_cast<const float*>(kbc->data());
    const float* va_d = static_cast<const float*>(vad->data());
    const float* gptr = gc ? static_cast<const float*>(gc->data()) : nullptr;

    auto out = std::make_shared<Tensor>(DataType::FP32,
                                        std::vector<int64_t>{seq_len, v_stride},
                                        DeviceType::CPU);
    float* odata = static_cast<float*>(out->data());

    std::vector<float> q_abs_full(static_cast<size_t>(seq_len) * n_head * kv_lora);
    std::vector<float> q_rope_rot(rope_dim);
    std::vector<float> scores(total);

    for (int s = 0; s < seq_len; ++s) {
        const int64_t qi = start_pos + s;
        const float* q_heads = qf + static_cast<int64_t>(s) * q_stride;
        float* o_heads = odata + static_cast<int64_t>(s) * v_stride;

        // q_abs[h] = kb[h] · q_nope[h]
        for (int h = 0; h < n_head; ++h) {
            const float* q_nope = q_heads + static_cast<int64_t>(h) * qk_head_dim;
            const float* kbh = kb_d + static_cast<int64_t>(h) * kv_lora * qk_nope_dim;
            float* qah = q_abs_full.data() +
                         (static_cast<int64_t>(s) * n_head + h) * kv_lora;
            for (int dout = 0; dout < kv_lora; ++dout)
                qah[dout] = forge::cpu::dot_f32(kbh + static_cast<int64_t>(dout) * qk_nope_dim,
                                                q_nope, qk_nope_dim);
        }

        for (int h = 0; h < n_head; ++h) {
            const float* q_rope = q_heads + static_cast<int64_t>(h) * qk_head_dim + qk_nope_dim;
            std::memcpy(q_rope_rot.data(), q_rope, q_rope_rot.size() * sizeof(float));
            apply_rope_interleaved(q_rope_rot.data(), rope_dim, qi, theta);
            const float* q_abs = q_abs_full.data() +
                                 (static_cast<int64_t>(s) * n_head + h) * kv_lora;
            const float* vV_h = va_d + static_cast<int64_t>(h) * v_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < total; ++j) {
                if (j > qi)
                    continue;  // causal mask (history indexed by absolute pos)
                const float dot_nope =
                    forge::cpu::dot_f32(q_abs, cnp + static_cast<int64_t>(j) * kv_lora, kv_lora);
                // k_pe_rot[j] already holds rope(c_rope[j], pos=j) — shared by
                // every head, so it is computed once per position, not per head.
                const float dot_rope = forge::cpu::dot_f32(
                    q_rope_rot.data(), kpr_d + static_cast<int64_t>(j) * rope_dim, rope_dim);
                const float sc = (dot_nope + dot_rope) * scale;
                scores[j] = sc;
                if (sc > max_score)
                    max_score = sc;
            }
            float sum = 0.0f;
            for (int j = 0; j < total; ++j) {
                if (j > qi)
                    continue;
                const float e = std::exp(scores[j] - max_score);
                scores[j] = e;
                sum += e;
            }
            const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            float* o = o_heads + static_cast<int64_t>(h) * v_dim;
            for (int d = 0; d < v_dim; ++d)
                o[d] = 0.0f;
            for (int j = 0; j < total; ++j) {
                if (j > qi)
                    continue;
                const float w = scores[j] * inv_sum;
                const float* vv = vV_h + static_cast<int64_t>(j) * v_stride;
                forge::cpu::fmadd_f32(o, vv, v_dim, w);
            }
            if (gptr) {
                const float gv = gptr[static_cast<int64_t>(s) * n_head + h];
                forge::cpu::scale_f32(o, v_dim, gv);
            }
        }
    }
    return out;
}

}  // namespace ops
}  // namespace forge
