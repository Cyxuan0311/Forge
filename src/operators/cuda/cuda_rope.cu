#include "cuda_rope.h"
#include <cmath>

namespace nanoinfer {
namespace cuda {

__global__ void rope_q_kernel(const float* x, float* x_out,
                               int seq_len, int num_heads, int head_dim, int64_t pos, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int half_dim = head_dim / 2;
    if (d >= half_dim) return;

    int h = (idx / head_dim) % num_heads;
    int s = idx / (num_heads * head_dim);

    float freq = 1.0f / powf(theta, (2.0f * d) / head_dim);
    float angle = (pos + s) * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int i0 = idx;
    int i1 = s * num_heads * head_dim + h * head_dim + d + half_dim;

    float x0 = x[i0], x1 = x[i1];
    x_out[i0] = x0 * cos_a - x1 * sin_a;
    x_out[i1] = x0 * sin_a + x1 * cos_a;
}

void launch_rope_fp32(const float* q, const float* k, float* q_out, float* k_out,
                      int num_heads, int head_dim, int seq_len, int64_t pos,
                      float theta, cudaStream_t stream) {
    int q_total = seq_len * num_heads * head_dim;
    int threads = 256;
    int blocks = (q_total + threads - 1) / threads;
    rope_q_kernel<<<blocks, threads, 0, stream>>>(q, q_out, seq_len, num_heads, head_dim, pos, theta);

    int k_total = seq_len * num_heads * head_dim;
    blocks = (k_total + threads - 1) / threads;
    rope_q_kernel<<<blocks, threads, 0, stream>>>(k, k_out, seq_len, num_heads, head_dim, pos, theta);
}

void launch_rope_gqa(const float* q, const float* k, float* q_out, float* k_out,
                     int num_q_heads, int num_kv_heads, int head_dim, int seq_len, int64_t pos,
                     float theta, cudaStream_t stream) {
    int threads = 256;

    int q_total = seq_len * num_q_heads * head_dim;
    int q_blocks = (q_total + threads - 1) / threads;
    rope_q_kernel<<<q_blocks, threads, 0, stream>>>(q, q_out, seq_len, num_q_heads, head_dim, pos, theta);

    int k_total = seq_len * num_kv_heads * head_dim;
    int k_blocks = (k_total + threads - 1) / threads;
    rope_q_kernel<<<k_blocks, threads, 0, stream>>>(k, k_out, seq_len, num_kv_heads, head_dim, pos, theta);
}

__global__ void expand_kv_kernel(const float* kv, float* out,
                                  int seq_len, int num_heads, int num_kv_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h = (idx / head_dim) % num_heads;
    int s = idx / (num_heads * head_dim);

    int kv_groups = num_heads / num_kv_heads;
    int kv_h = h / kv_groups;

    int src_idx = s * num_kv_heads * head_dim + kv_h * head_dim + d;
    out[idx] = kv[src_idx];
}

void launch_expand_kv(const float* kv, float* out,
                      int seq_len, int num_heads, int num_kv_heads, int head_dim,
                      cudaStream_t stream) {
    int total = seq_len * num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    expand_kv_kernel<<<blocks, threads, 0, stream>>>(kv, out, seq_len, num_heads, num_kv_heads, head_dim);
}

} // namespace cuda
} // namespace nanoinfer
