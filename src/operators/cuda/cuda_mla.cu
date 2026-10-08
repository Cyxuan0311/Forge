// CUDA implementation of the absorbed MLA attention (bailingmoe3_mla_attention),
// mirroring operators/cpu/gdn.cpp. Inputs are device tensors; the output tensor
// is allocated on the device.
//
// Scheme: a value-contraction kernel computes v_all[j][h*v_dim+d] = vb[h*v_dim+d]
// · c_nope[j]; a q_abs-contraction kernel computes q_abs[s][h*kv_lora+dout] =
// kb[h*kv_lora+dout] · q_nope[s,h]; then one (query token, head) block does the
// causal softmax + context accumulation + output gate and writes the output row.

#include "cuda_gdn.h"
#include "cuda_common.h"

#include <cuda_runtime.h>
#include <cmath>

namespace forge {
namespace cuda {

// v_all[j*n_head*v_dim + h*v_dim + d] = vb[(h*v_dim+d)*kv_lora + k] · c_nope[j]
__global__ void k_mla_v(const float* vb, const float* cnp, float* v_all, int total,
                        int n_head, int kv_lora, int v_dim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n = static_cast<int64_t>(total) * n_head * v_dim;
    if (idx >= n) return;
    int64_t j = idx / static_cast<int64_t>(n_head * v_dim);
    int64_t rem = idx - j * n_head * v_dim;
    int h = static_cast<int>(rem / v_dim);
    int d = static_cast<int>(rem - h * v_dim);
    const float* vbh = vb + (static_cast<int64_t>(h) * v_dim + d) * kv_lora;
    const float* cj = cnp + j * kv_lora;
    float acc = 0.0f;
    for (int k = 0; k < kv_lora; ++k) acc += vbh[k] * cj[k];
    v_all[idx] = acc;
}

// q_abs_full[s*n_head*kv_lora + h*kv_lora + dout] =
//     sum_din kb[(h*kv_lora+dout)*qk_nope_dim + din] * q_nope[s,h][din]
__global__ void k_mla_qabs(const float* q_flat, const float* kb, float* q_abs_full,
                          int seq_len, int n_head, int kv_lora, int qk_nope_dim,
                          int qk_head_dim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n = static_cast<int64_t>(seq_len) * n_head * kv_lora;
    if (idx >= n) return;
    int64_t s = idx / static_cast<int64_t>(n_head * kv_lora);
    int64_t rem = idx - s * n_head * kv_lora;
    int h = static_cast<int>(rem / kv_lora);
    int dout = static_cast<int>(rem - h * kv_lora);
    const float* q_nope =
        q_flat + s * n_head * qk_head_dim + h * qk_head_dim;
    const float* krow = kb + (static_cast<int64_t>(h) * kv_lora + dout) * qk_nope_dim;
    float acc = 0.0f;
    for (int din = 0; din < qk_nope_dim; ++din) acc += krow[din] * q_nope[din];
    q_abs_full[idx] = acc;
}

// Split a flattened [total, latent_dim] latent into device c_nope [total, kv_lora]
// and c_rope [total, rope_dim] (latent_dim == kv_lora + rope_dim, nope first).
__global__ void k_mla_split(const float* latent, float* c_nope, float* c_rope,
                            int64_t total, int kv_lora, int rope_dim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n = total * (static_cast<int64_t>(kv_lora) + rope_dim);
    if (idx >= n) return;
    int64_t r = idx / (static_cast<int64_t>(kv_lora) + rope_dim);
    int c = static_cast<int>(idx - r * (static_cast<int64_t>(kv_lora) + rope_dim));
    const float* lrow = latent + r * (static_cast<int64_t>(kv_lora) + rope_dim);
    if (c < kv_lora)
        c_nope[r * kv_lora + c] = lrow[c];
    else
        c_rope[r * rope_dim + (c - kv_lora)] = lrow[c];
}

// (k_mla_score, the earlier per-(s,h) fused-scoring variant, was folded into
// k_mla_score_write below.)

// Write the (s,h) head row computed above into the output. Kept separate so the
// scoring kernel stays register-light; for decode seq_len==1 this is a single
// block anyway, so we instead inline the write in a combined kernel below.
__global__ void k_mla_score_write(const float* q_flat, const float* q_abs_full,
                                 const float* cnp, const float* k_pe_rot,
                                 const float* v_all, const float* gate,
                                 float* out, int start_pos, int seq_len, int total,
                                 int n_head, int kv_lora, int qk_nope_dim,
                                 int rope_dim, int v_dim, int qk_head_dim,
                                 float theta, float scale,
                                 const float* yarn_s, float yarn_mscale) {
    int s = static_cast<int>(blockIdx.x);
    int h = static_cast<int>(blockIdx.y);
    if (s >= seq_len || h >= n_head) return;
    int64_t qi = start_pos + s;
    const int v_stride = n_head * v_dim;
    const float* q_heads = q_flat + static_cast<int64_t>(s) * n_head * qk_head_dim;
    const float* q_rope = q_heads + h * qk_head_dim + qk_nope_dim;
    const float* q_abs = q_abs_full + static_cast<int64_t>(s) * n_head * kv_lora +
                         h * kv_lora;
    const float* vV_h = v_all + static_cast<int64_t>(h) * v_dim;

    // RoPE-rotate the query q_rope at position qi (interleaved pairs). This
    // depends only on the query, not on the history key, so it stays per-(s,h).
    float qr[128];
    for (int i = 0; i + 1 < rope_dim; i += 2) {
        float a = q_rope[i], b = q_rope[i + 1];
        float freq = __powf(theta, -static_cast<float>(i) / static_cast<float>(rope_dim));
        if (yarn_s) freq *= yarn_s[i];
        float ang = static_cast<float>(qi) * freq;
        float c = __cosf(ang) * yarn_mscale, sn = __sinf(ang) * yarn_mscale;
        qr[i] = a * c - b * sn;
        qr[i + 1] = a * sn + b * c;
    }

    float scores[4096];
    float max_score = -1e30f;
    for (int j = 0; j < total; ++j) {
        if (j > qi) continue;
        const float* cj = cnp + static_cast<int64_t>(j) * kv_lora;
        float dot_nope = 0.0f;
        for (int m = 0; m < kv_lora; ++m) dot_nope += q_abs[m] * cj[m];
        // k_pe_rot[j] is rope(c_rope[j], pos=j), precomputed by the engine and
        // shared across all heads — no per-head RoPE recomputation here.
        const float* kpr = k_pe_rot + static_cast<int64_t>(j) * rope_dim;
        float dot_rope = 0.0f;
        for (int i = 0; i + 1 < rope_dim; i += 2)
            dot_rope += qr[i] * kpr[i] + qr[i + 1] * kpr[i + 1];
        float sc = (dot_nope + dot_rope) * scale;
        scores[j] = sc;
        if (sc > max_score) max_score = sc;
    }
    float sum = 0.0f;
    for (int j = 0; j < total; ++j) {
        if (j > qi) continue;
        float e = __expf(scores[j] - max_score);
        scores[j] = e;
        sum += e;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    float* orow = out + static_cast<int64_t>(s) * v_stride + h * v_dim;
    for (int d = 0; d < v_dim; ++d) orow[d] = 0.0f;
    for (int j = 0; j < total; ++j) {
        if (j > qi) continue;
        float w = scores[j] * inv;
        const float* vv = vV_h + static_cast<int64_t>(j) * v_stride;
        for (int d = 0; d < v_dim; ++d) orow[d] += w * vv[d];
    }
    if (gate) {
        float gv = gate[static_cast<int64_t>(s) * n_head + h];
        for (int d = 0; d < v_dim; ++d) orow[d] *= gv;
    }
}

void mla_attention_launch(const TensorPtr& q_flat, const TensorPtr& c_nope_norm,
                         const TensorPtr& k_pe_rot, const TensorPtr& kb,
                         const TensorPtr& v_all, const TensorPtr& gate,
                         int start_pos, int seq_len, int total, int n_head,
                         int kv_lora, int qk_nope_dim, int rope_dim, int v_dim,
                         float theta, float scale, TensorPtr& out,
                         const float* yarn_s, float yarn_mscale) {
    const int qk_head_dim = qk_nope_dim + rope_dim;
    const int v_stride = n_head * v_dim;
    const int64_t qa_bytes =
        static_cast<int64_t>(seq_len) * n_head * kv_lora * sizeof(float);

    // Only the query-side contraction (q_abs) is recomputed here; k_pe_rot and
    // v_all are supplied precomputed by the engine's incremental MLA cache.
    float* qa_d = static_cast<float*>(scratch_pool().ensure(qa_bytes));

    const float* qf_d = static_cast<const float*>(q_flat->data());
    const float* cnp_d = static_cast<const float*>(c_nope_norm->data());
    const float* kpr_d = static_cast<const float*>(k_pe_rot->data());
    const float* kb_d = static_cast<const float*>(kb->data());
    const float* va_d = static_cast<const float*>(v_all->data());
    const float* g_d = gate ? static_cast<const float*>(gate->data()) : nullptr;

    int block = 256;
    int grid_qa = static_cast<int>((static_cast<int64_t>(seq_len) * n_head * kv_lora +
                                    block - 1) / block);
    k_mla_qabs<<<grid_qa, block>>>(qf_d, kb_d, qa_d, seq_len, n_head, kv_lora,
                                   qk_nope_dim, qk_head_dim);

    if (!out) {
        out = std::make_shared<Tensor>(DataType::FP32,
                                       std::vector<int64_t>{seq_len, v_stride},
                                       DeviceType::CUDA);
    }
    float* out_d = static_cast<float*>(out->data());
    dim3 grid(seq_len, n_head);
    k_mla_score_write<<<grid, 1>>>(qf_d, qa_d, cnp_d, kpr_d, va_d, g_d, out_d,
                                   start_pos, seq_len, total, n_head, kv_lora,
                                   qk_nope_dim, rope_dim, v_dim, qk_head_dim,
                                   theta, scale, yarn_s, yarn_mscale);
}

void mla_split_latent(const float* latent_dev, float* c_nope_dev, float* c_rope_dev,
                      int total, int kv_lora, int rope_dim) {
    int64_t n = static_cast<int64_t>(total) * (kv_lora + rope_dim);
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    k_mla_split<<<grid, block>>>(latent_dev, c_nope_dev, c_rope_dev, total, kv_lora,
                                 rope_dim);
}

// Rotate the new c_rope rows (absolute positions start_pos .. start_pos+total-1)
// into k_pe_rot. Identical to the RoPE the CPU reference applies per history key;
// the engine calls this once per token over only the new rows.
__global__ void k_mla_rope(const float* c_rope, float* k_pe_rot, int start_pos,
                           int total, int rope_dim, float theta,
                           const float* yarn_s, float yarn_mscale) {
    int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= total) return;
    int64_t base = j * rope_dim;
    const float* kp = c_rope + base;
    float* kr = k_pe_rot + base;
    float pos = static_cast<float>(start_pos) + static_cast<float>(j);
    for (int i = 0; i + 1 < rope_dim; i += 2) {
        const float a = kp[i], b = kp[i + 1];
        float freq =
            __powf(theta, -static_cast<float>(i) / static_cast<float>(rope_dim));
        if (yarn_s) freq *= yarn_s[i];
        const float ang = pos * freq;
        const float c = __cosf(ang) * yarn_mscale, sn = __sinf(ang) * yarn_mscale;
        kr[i] = a * c - b * sn;
        kr[i + 1] = a * sn + b * c;
    }
}

void mla_rope_launch(const TensorPtr& c_rope, TensorPtr& k_pe_rot, int start_pos,
                     int total, int rope_dim, float theta,
                     const float* yarn_s, float yarn_mscale) {
    int block = 256;
    int grid = static_cast<int>((static_cast<int64_t>(total) + block - 1) / block);
    k_mla_rope<<<grid, block>>>(static_cast<const float*>(c_rope->data()),
                                static_cast<float*>(k_pe_rot->data()), start_pos,
                                total, rope_dim, theta, yarn_s, yarn_mscale);
}

// Value contraction v_all[j][h*v_dim+d] = vb[(h*v_dim+d)*kv_lora+k] · c_nope_norm[j].
// Computes only the `total` supplied rows; the engine appends them to its cache.
void mla_value_contract_launch(const TensorPtr& vb, const TensorPtr& c_nope_norm,
                               TensorPtr& v_all, int total, int n_head,
                               int kv_lora, int v_dim) {
    int block = 256;
    int64_t n = static_cast<int64_t>(total) * n_head * v_dim;
    int grid = static_cast<int>((n + block - 1) / block);
    k_mla_v<<<grid, block>>>(static_cast<const float*>(vb->data()),
                             static_cast<const float*>(c_nope_norm->data()),
                             static_cast<float*>(v_all->data()), total, n_head,
                             kv_lora, v_dim);
}

}  // namespace cuda
}  // namespace forge
