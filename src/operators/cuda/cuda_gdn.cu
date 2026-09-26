// CUDA implementations of the bailingmoe3 GDN/MLA custom operators.
// These mirror operators/cpu/gdn.cpp + conv.cpp exactly.
//
// Stateful ops (conv1d_depthwise_silu, gdn_recurrence) are launched one kernel
// per token so the per-seq conv / S state is updated strictly in order on the
// default stream. For decode seq_len == 1, that is a single launch each.

#include "cuda_gdn.h"
#include "cuda_common.h"
#include "forge/perf_profiler.h"

#include <cuda_runtime.h>
#include <cmath>

namespace forge {
namespace cuda {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dev_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// ---------------------------------------------------------------------------
// sigmoid (stateless)
// ---------------------------------------------------------------------------
__global__ void k_sigmoid(float* d, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) d[i] = dev_sigmoid(d[i]);
}
void sigmoid_launch(const TensorPtr& t) {
    PERF_SCOPE("gdn/sigmoid");
    float* p = static_cast<float*>(t->data());
    int64_t n = t->numel();
    if (n == 0) return;
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    k_sigmoid<<<grid, block>>>(p, n);
}

// ---------------------------------------------------------------------------
// MoE weighted accumulation: acc[d] += w * src[d] for d in [0, n).
// Lets the routed-expert results be summed on the device instead of pulling
// every expert's [1, hidden_dim] output back to the host (one D2H per expert).
// ---------------------------------------------------------------------------
__global__ void k_axpy(float* acc, const float* src, float w, int n) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += w * src[i];
}
void moe_accumulate_launch(float* acc_dev, const float* src_dev, float w, int n) {
    if (n <= 0) return;
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    k_axpy<<<grid, block>>>(acc_dev, src_dev, w, n);
}

// ---------------------------------------------------------------------------
// conv1d_depthwise_silu — one kernel per token (stateful, sequential)
// x: [seq_len, dim]  w: [dim, k_size]  state_dev: [(k_size-1)*dim]
// ---------------------------------------------------------------------------
__global__ void k_conv_token(float* x, const float* w, float* state, int dim,
                             int k_size, int s) {
    int c = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    const int state_len = k_size - 1;
    const float raw = x[static_cast<int64_t>(s) * dim + c];
    float val = 0.0f;
    for (int k = 0; k < k_size; ++k) {
        float xv;
        if (k < state_len)
            xv = state[static_cast<int64_t>(k) * dim + c];
        else
            xv = raw;
        val += xv * w[static_cast<int64_t>(c) * k_size + k];
    }
    // SiLU: val * sigmoid(val) == val / (1 + exp(-val)).
    // (只写 sigmoid(val) 会漏掉乘 val, 结果恒为 ~0.5 — 与 CPU 的
    //  val / (1 + exp(-val)) 不一致。)
    val = val * dev_sigmoid(val);
    x[static_cast<int64_t>(s) * dim + c] = val;
    for (int k = 0; k < state_len - 1; ++k)
        state[static_cast<int64_t>(k) * dim + c] =
            state[static_cast<int64_t>(k + 1) * dim + c];
    state[static_cast<int64_t>(state_len - 1) * dim + c] = raw;
}
void conv1d_depthwise_silu_launch(const TensorPtr& x, const TensorPtr& w,
                                 float* conv_state_dev, int seq_len, int dim,
                                 int k_size) {
    PERF_SCOPE("gdn/conv");
    float* xd = static_cast<float*>(x->data());
    const float* wd = static_cast<const float*>(w->data());
    int block = 256;
    int grid = static_cast<int>((dim + block - 1) / block);
    for (int s = 0; s < seq_len; ++s)
        k_conv_token<<<grid, block>>>(xd, wd, conv_state_dev, dim, k_size, s);
}

// ---------------------------------------------------------------------------
// conv1d_depthwise_silu — fused 3-stream version (q/k/v in one launch).
// Same math as k_conv_token; linear id i maps to (stream, channel) so the
// three per-token convs share a single launch instead of three.
// ---------------------------------------------------------------------------
__global__ void k_conv3_token(float* x0, const float* w0, float* st0,
                              float* x1, const float* w1, float* st1,
                              float* x2, const float* w2, float* st2,
                              int dim, int k_size, int s) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= 3 * dim) return;
    const int stream = i / dim;
    const int c = i - stream * dim;
    float* x;
    const float* w;
    float* state;
    if (stream == 0) { x = x0; w = w0; state = st0; }
    else if (stream == 1) { x = x1; w = w1; state = st1; }
    else { x = x2; w = w2; state = st2; }
    const int state_len = k_size - 1;
    const float raw = x[static_cast<int64_t>(s) * dim + c];
    float val = 0.0f;
    for (int k = 0; k < k_size; ++k) {
        float xv;
        if (k < state_len)
            xv = state[static_cast<int64_t>(k) * dim + c];
        else
            xv = raw;
        val += xv * w[static_cast<int64_t>(c) * k_size + k];
    }
    val = val * dev_sigmoid(val);
    x[static_cast<int64_t>(s) * dim + c] = val;
    for (int k = 0; k < state_len - 1; ++k)
        state[static_cast<int64_t>(k) * dim + c] =
            state[static_cast<int64_t>(k + 1) * dim + c];
    state[static_cast<int64_t>(state_len - 1) * dim + c] = raw;
}
void conv3_depthwise_silu_launch(const TensorPtr& x0, const TensorPtr& w0,
                                 float* conv_state_dev0, const TensorPtr& x1,
                                 const TensorPtr& w1, float* conv_state_dev1,
                                 const TensorPtr& x2, const TensorPtr& w2,
                                 float* conv_state_dev2, int seq_len, int dim,
                                 int k_size) {
    PERF_SCOPE("gdn/conv");
    float* x0d = static_cast<float*>(x0->data());
    const float* w0d = static_cast<const float*>(w0->data());
    float* x1d = static_cast<float*>(x1->data());
    const float* w1d = static_cast<const float*>(w1->data());
    float* x2d = static_cast<float*>(x2->data());
    const float* w2d = static_cast<const float*>(w2->data());
    int block = 256;
    int grid = static_cast<int>((3 * dim + block - 1) / block);
    for (int s = 0; s < seq_len; ++s)
        k_conv3_token<<<grid, block>>>(x0d, w0d, conv_state_dev0, x1d, w1d,
                                       conv_state_dev1, x2d, w2d, conv_state_dev2,
                                       dim, k_size, s);
}

// ---------------------------------------------------------------------------
// build_decay_gate (stateless)
// decay[t, h*head_dim+dd] = lower_bound / (1 + exp(-a[h]*(x[t,h*head_dim+dd]+bias)))
// ---------------------------------------------------------------------------
__global__ void k_decay(const float* in, float* out, const float* bias,
                       const float* a, int d_inner, int num_heads, int head_dim,
                       float lower_bound) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= d_inner) return;
    // (h, dd) decomposition
    int h = static_cast<int>(i) / head_dim;
    int dd = static_cast<int>(i) - h * head_dim;
    float f = in[i] + bias[i];
    out[i] = lower_bound / (1.0f + __expf(-(a[h] * f)));
}
void build_decay_gate_launch(const TensorPtr& logits, const float* bias, const float* a,
                            int seq_len, int num_heads, int head_dim,
                            float gate_lower_bound) {
    PERF_SCOPE("gdn/decay");
    const int64_t d_inner = static_cast<int64_t>(num_heads) * head_dim;
    const float* in = static_cast<const float*>(logits->data());
    float* out = static_cast<float*>(logits->data());  // write in place
    // bias / a 是逐层 init 时一次性上传的设备端常量, 此处零拷贝直取。
    const float* dbias = bias;
    const float* da = a;
    int block = 256;
    // 每个 token 单独 launch, 故 grid 只覆盖一行的 d_inner 个元素。
    int grid = static_cast<int>((d_inner + block - 1) / block);
    for (int t = 0; t < seq_len; ++t) {
        const float* tin = in + static_cast<int64_t>(t) * d_inner;
        float* tout = out + static_cast<int64_t>(t) * d_inner;
        k_decay<<<grid, block>>>(tin, tout, dbias, da, d_inner, num_heads, head_dim,
                                gate_lower_bound);
    }
}

// ---------------------------------------------------------------------------
// gdn_recurrence — one kernel per token (stateful, sequential)
// q,k,v: [seq_len, d_inner]  decay: [seq_len, d_inner]  beta: [seq_len, num_heads]
// state_dev: [num_heads * head_dim * head_dim]  out: [seq_len, d_inner]
// ---------------------------------------------------------------------------
// 逐位对齐 gdn.cpp 的 GDN recurrence。原始实现 1 block / 64 线程但仅 num_heads
// 个线程活跃 (每 head 一线程), 且每线程在栈上开 ql[128]/kl[128]/exp_decay[128]
// (~384 寄存器) -> 严重 register spilling 到 local memory, 单 SM 串行、有效带宽
// 坍到 <1GB/s, 实测 2.5ms/layer (占 decode 45%)。
//
// 新版: 每 head 一个 block, head_dim 个线程 (每线程负责一行 j)。per-head 的
// q/k 归一化向量与 exp(decay) 由线程 0 以与 CPU 完全相同的串行顺序算入 shared
// memory; 每个 j 线程对 i 的归约仍保持串行 (与 gdn.cpp 的求和顺序逐位一致), 仅
// 把行 j 之间的并行暴露出来。无大栈数组, occupancy 从 16 线程/卡 提到 ~2048。
__global__ void k_recur_token(const float* q, const float* k, const float* v,
                             const float* decay, const float* beta, float* state,
                             float* out, int d_inner, int num_heads, int head_dim,
                             float eps, float scale, int seq_len) {
    int h = static_cast<int>(blockIdx.x);  // 每 head 一个 block
    if (h >= num_heads) return;
    int j = static_cast<int>(threadIdx.x);  // 每线程负责一行 j
    if (j >= head_dim) return;
    const int64_t state_size = static_cast<int64_t>(head_dim) * head_dim;
    float* S = state + static_cast<int64_t>(h) * state_size;
    // per-head 向量: 仅 3 个共享数组, 不再每个线程 384 寄存器。
    __shared__ float sq[128], sk[128], sd[128];

    for (int t = 0; t < seq_len; ++t) {
        const int64_t b = static_cast<int64_t>(t) * d_inner;
        const float* qt = q + b + static_cast<int64_t>(h) * head_dim;
        const float* kt = k + b + static_cast<int64_t>(h) * head_dim;
        const float* vt = v + b + static_cast<int64_t>(h) * head_dim;
        float* ot = out + b + static_cast<int64_t>(h) * head_dim;
        const float* gt = decay + b + static_cast<int64_t>(h) * head_dim;
        const float bt = beta[static_cast<int64_t>(t) * num_heads + h];

        // 线程 0 以与 CPU 相同的串行顺序算归一化向量 + exp(decay)。
        if (j == 0) {
            float qn = 0.0f, kn = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                qn += qt[i] * qt[i];
                kn += kt[i] * kt[i];
            }
            float qi = 1.0f / sqrtf(qn + eps);
            float ki = 1.0f / sqrtf(kn + eps);
            for (int i = 0; i < head_dim; ++i) {
                sq[i] = qt[i] * qi;
                sk[i] = kt[i] * ki;
            }
            for (int i = 0; i < head_dim; ++i) sd[i] = __expf(gt[i]);
        }
        __syncthreads();

        float vj = vt[j];  // v 为每行一个值 (下标 j)

        // 1) S[j][i] *= exp(decay[i])  — decay 按列下标 i 缩放 (与 gdn.cpp 一致)
        for (int i = 0; i < head_dim; ++i) S[static_cast<int64_t>(j) * head_dim + i] *= sd[i];
        // 2) delta[j] = (v[j] - sum_i S[j][i]*k[i]) * beta
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i)
            dot += S[static_cast<int64_t>(j) * head_dim + i] * sk[i];
        float delta = (vj - dot) * bt;
        // 3) S[j][i] += k[i] * delta[j]
        for (int i = 0; i < head_dim; ++i)
            S[static_cast<int64_t>(j) * head_dim + i] += sk[i] * delta;
        // 4) out[j] = (sum_i S[j][i]*q[i]) * scale
        float dot2 = 0.0f;
        for (int i = 0; i < head_dim; ++i)
            dot2 += S[static_cast<int64_t>(j) * head_dim + i] * sq[i];
        ot[j] = dot2 * scale;
        __syncthreads();
    }
}
void gdn_recurrence_launch(const TensorPtr& q, const TensorPtr& k, const TensorPtr& v,
                          const TensorPtr& decay, const TensorPtr& beta,
                          float* state_dev, int seq_len, int num_heads,
                          int head_dim, float eps, TensorPtr& out) {
    PERF_SCOPE("gdn/recur");
    const int64_t d_inner = static_cast<int64_t>(num_heads) * head_dim;
    const float* qd = static_cast<const float*>(q->data());
    const float* kd = static_cast<const float*>(k->data());
    const float* vd = static_cast<const float*>(v->data());
    const float* dd = static_cast<const float*>(decay->data());
    const float* bd = static_cast<const float*>(beta->data());
    if (!out)
        out = std::make_shared<Tensor>(DataType::FP32,
                                       std::vector<int64_t>{seq_len, d_inner},
                                       DeviceType::CUDA);
    float* od = static_cast<float*>(out->data());
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    // 每 head 一个 block, head_dim 个线程 (一行 j 一个线程)。
    int block = head_dim;
    int grid = num_heads;
    k_recur_token<<<grid, block>>>(qd, kd, vd, dd, bd, state_dev, od, d_inner,
                                   num_heads, head_dim, eps, scale, seq_len);
}

// ---------------------------------------------------------------------------
// gdn_output_gate — RMSNorm per head + sigmoid output gate (stateless)
// ---------------------------------------------------------------------------
__global__ void k_ogate(const float* c, const float* g, float* o, const float* nw,
                       int64_t total, int d_inner, int num_heads, int head_dim,
                       float eps) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // 覆盖 [0, seq_len*d_inner): 之前误用 d_inner 作上界, 只有 t==0 被处理,
    // 其余 token 直接 return, 输出被漏算。
    if (i >= total) return;
    int t = static_cast<int>(i) / d_inner;
    int hh = static_cast<int>(i - t * d_inner);
    int h = hh / head_dim;
    int d = hh - h * head_dim;
    const float* ct = c + static_cast<int64_t>(t) * d_inner + h * head_dim;
    const float* gt = g + static_cast<int64_t>(t) * d_inner + h * head_dim;
    float* ot = o + static_cast<int64_t>(t) * d_inner + h * head_dim;
    float ss = 0.0f;
    for (int j = 0; j < head_dim; ++j) ss += ct[j] * ct[j];
    float inv = 1.0f / sqrtf(ss / static_cast<float>(head_dim) + eps);
    ot[d] = ct[d] * inv * nw[d] / (1.0f + __expf(-gt[d]));
}
void gdn_output_gate_launch(const TensorPtr& core_out,
                           const TensorPtr& out_gate_logits,
                           const float* nw, int seq_len, int num_heads,
                           int head_dim, float eps, TensorPtr& out) {
    PERF_SCOPE("gdn/ogate");
    const int64_t d_inner = static_cast<int64_t>(num_heads) * head_dim;
    const float* cd = static_cast<const float*>(core_out->data());
    const float* gd = static_cast<const float*>(out_gate_logits->data());
    if (!out)
        out = std::make_shared<Tensor>(DataType::FP32,
                                       std::vector<int64_t>{seq_len, d_inner},
                                       DeviceType::CUDA);
    float* od = static_cast<float*>(out->data());
    // nw 是逐层 init 时一次性上传的设备端常量, 零拷贝直取。
    const float* dnw = nw;
    int block = 256;
    int64_t total = static_cast<int64_t>(seq_len) * d_inner;
    int grid = static_cast<int>((total + block - 1) / block);
    k_ogate<<<grid, block>>>(cd, gd, od, dnw, total, d_inner, num_heads, head_dim, eps);
}

}  // namespace cuda
}  // namespace forge
