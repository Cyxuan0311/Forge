#include "forge/cuda_kernels.h"

#include <cmath>

namespace forge {
namespace cuda {

// Partial split-half RoPE (Spark-X2.5): 旋转 dims [0, n_rot) (配对 x[d],
// x[d + n_rot/2]), dims [n_rot, head_dim) 原样拷贝。n_rot == head_dim 时退化为
// 标准 split-half RoPE。
__global__ void rope_partial_kernel(const float* x, float* out, int seq_len, int num_heads,
                                    int head_dim, int n_rot, int64_t pos, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int half_rot = n_rot / 2;
    if (d >= n_rot) {
        out[idx] = x[idx];
        return;
    }
    if (d >= half_rot)
        return;  // 配对 owner (d < half_rot) 负责写两个半区

    int s = idx / (num_heads * head_dim);

    float freq = 1.0f / powf(theta, (2.0f * d) / (float)n_rot);
    float angle = ((float)pos + (float)s) * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int i1 = idx + half_rot;
    float x0 = x[idx];
    float x1 = x[i1];
    out[idx] = x0 * cos_a - x1 * sin_a;
    out[i1] = x0 * sin_a + x1 * cos_a;
}

void launch_rope_partial_gqa(const float* q, const float* k, float* q_out, float* k_out,
                             int num_q_heads, int num_kv_heads, int head_dim, int n_rot,
                             int seq_len, int64_t pos, float theta, cudaStream_t stream) {
    int threads = 256;

    int q_total = seq_len * num_q_heads * head_dim;
    int q_blocks = (q_total + threads - 1) / threads;
    rope_partial_kernel<<<q_blocks, threads, 0, stream>>>(q, q_out, seq_len, num_q_heads, head_dim,
                                                          n_rot, pos, theta);

    int k_total = seq_len * num_kv_heads * head_dim;
    int k_blocks = (k_total + threads - 1) / threads;
    rope_partial_kernel<<<k_blocks, threads, 0, stream>>>(k, k_out, seq_len, num_kv_heads,
                                                          head_dim, n_rot, pos, theta);
}

// Per-head sigmoid 门控 (原地): x[s, h*head_dim+d] *= sigmoid(gate[s, h])。
__global__ void headwise_sigmoid_gate_mul_kernel(float* x, const float* gate, int seq_len,
                                                 int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total)
        return;

    int h = (idx / head_dim) % num_heads;
    int s = idx / (num_heads * head_dim);
    float g = 1.0f / (1.0f + expf(-gate[s * num_heads + h]));
    x[idx] *= g;
}

void launch_headwise_sigmoid_gate_mul(float* x, const float* gate, int seq_len, int num_heads,
                                      int head_dim, cudaStream_t stream) {
    int total = seq_len * num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    headwise_sigmoid_gate_mul_kernel<<<blocks, threads, 0, stream>>>(x, gate, seq_len, num_heads,
                                                                     head_dim);
}

}  // namespace cuda
}  // namespace forge
