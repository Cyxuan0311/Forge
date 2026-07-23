#include <cmath>
#include <algorithm>

#include "cuda_common.h"
#include "cuda_ssm.h"

namespace forge {
namespace cuda {

// ============================================================================
// Kernel 1: SSM Preprocess
//   alpha += dt_bias, softplus, *= ssm_a, exp => gate_out
//   beta => sigmoid => beta_out
// ============================================================================
__global__ void ssm_preprocess_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    int seq_len, int num_v_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_v_heads;
    if (idx >= total) return;

    int s = idx / num_v_heads;
    int j = idx % num_v_heads;

    // alpha: softplus(alpha + dt_bias) * ssm_a
    float t = alpha[s * num_v_heads + j] + dt_bias[j];
    t = logf(1.0f + expf(t));
    t *= ssm_a[j];
    gate_out[s * num_v_heads + j] = t;

    // beta: sigmoid
    beta_out[s * num_v_heads + j] = 1.0f / (1.0f + expf(-beta[s * num_v_heads + j]));
}

void launch_ssm_preprocess(
    const float* alpha, const float* beta,
    const float* dt_bias, const float* ssm_a,
    float* gate_out, float* beta_out,
    int seq_len, int num_v_heads,
    cudaStream_t stream) {
    int total = seq_len * num_v_heads;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    ssm_preprocess_kernel<<<blocks, threads, 0, stream>>>(
        alpha, beta, dt_bias, ssm_a, gate_out, beta_out, seq_len, num_v_heads);
}

// ============================================================================
// Kernel 2: SSM Causal Conv1d (M=1 decode, handles seq_len > 1 with internal loop)
//   y = conv1d(x, weight) with persistent conv_state
// ============================================================================
__global__ void ssm_conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ conv_state,
    float* __restrict__ y,
    int seq_len, int conv_channels, int d_conv) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_channels) return;

    int state_len = d_conv - 1;

    for (int s = 0; s < seq_len; ++s) {
        const float* x_row = x + s * conv_channels;
        float* y_row = y + s * conv_channels;

        float val = 0.0f;
        for (int k = 0; k < state_len; ++k) {
            val += conv_state[k * conv_channels + c] * weight[c * d_conv + k];
        }
        val += x_row[c] * weight[c * d_conv + state_len];
        y_row[c] = val;

        for (int k = 0; k < state_len - 1; ++k) {
            conv_state[k * conv_channels + c] = conv_state[(k + 1) * conv_channels + c];
        }
        conv_state[(state_len - 1) * conv_channels + c] = x_row[c];
    }
}

void launch_ssm_conv1d(
    const float* x, const float* weight,
    float* conv_state, float* y,
    int seq_len, int conv_channels, int d_conv,
    cudaStream_t stream) {
    int threads = std::min(conv_channels, 1024);
    int blocks = (conv_channels + threads - 1) / threads;
    ssm_conv1d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, conv_state, y, seq_len, conv_channels, d_conv);
}

// ============================================================================
// Kernel 3: SSM SiLU + Split (fused)
//   conv_out --[SiLU]--> scatter to q, k, v
// ============================================================================
__global__ void ssm_silu_split_kernel(
    const float* __restrict__ conv_out,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int seq_len, int key_dim, int value_dim) {
    int conv_channels = 2 * key_dim + value_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * conv_channels;
    if (idx >= total) return;

    int s = idx / conv_channels;
    int c = idx % conv_channels;

    float x = conv_out[idx];
    float silu = x / (1.0f + expf(-x));

    if (c < key_dim) {
        q[s * key_dim + c] = silu;
    } else if (c < 2 * key_dim) {
        k[s * key_dim + (c - key_dim)] = silu;
    } else {
        v[s * value_dim + (c - 2 * key_dim)] = silu;
    }
}

void launch_ssm_silu_split(
    const float* conv_out,
    float* q, float* k, float* v,
    int seq_len, int key_dim, int value_dim,
    cudaStream_t stream) {
    int conv_channels = 2 * key_dim + value_dim;
    int total = seq_len * conv_channels;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    ssm_silu_split_kernel<<<blocks, threads, 0, stream>>>(
        conv_out, q, k, v, seq_len, key_dim, value_dim);
}

// ============================================================================
// Kernel 4: SSM Per-Head L2 Normalization (in-place)
//   Divides each head's vector by its L2 norm
// ============================================================================
__global__ void ssm_per_head_l2norm_kernel(
    float* __restrict__ data,
    int seq_len, int num_heads, int head_dim, float eps) {
    int head_idx = blockIdx.x;
    int total_heads = seq_len * num_heads;
    if (head_idx >= total_heads) return;

    int s = head_idx / num_heads;
    int h = head_idx % num_heads;
    float* head = data + s * num_heads * head_dim + h * head_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        sum_sq += head[i] * head[i];
    }

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum_sq);
    __syncthreads();

    float inv_rms = rsqrtf(s_sum + eps);

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        head[i] *= inv_rms;
    }
}

void launch_ssm_per_head_l2norm(
    float* data,
    int seq_len, int num_heads, int head_dim, float eps,
    cudaStream_t stream) {
    int total_heads = seq_len * num_heads;
    int threads = std::min(head_dim, 1024);
    ssm_per_head_l2norm_kernel<<<total_heads, threads, 0, stream>>>(
        data, seq_len, num_heads, head_dim, eps);
}

// ============================================================================
// Kernel 5: Gated Delta Net AR (core SSM recurrence)
//   One block per V-head, each thread handles one column of S matrix
// ============================================================================
__global__ void ssm_gated_delta_net_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ ssm_state,
    float* __restrict__ output,
    int seq_len,
    int head_k_dim,
    int head_v_dim,
    int num_k_heads,
    int num_v_heads) {
    int hv = blockIdx.x;
    if (hv >= num_v_heads) return;

    int tid = threadIdx.x;
    if (tid >= head_v_dim) return;

    extern __shared__ float smem[];
    float* smem_q_h = smem;
    float* smem_k_h = smem + head_k_dim;

    int head_repeat = num_v_heads / num_k_heads;
    int hk = hv / head_repeat;

    float* S = ssm_state + hv * head_v_dim * head_v_dim;
    float scale = rsqrtf(static_cast<float>(head_k_dim));

    int max_active_j = head_k_dim;
    if (head_v_dim < head_k_dim) max_active_j = head_v_dim;

    for (int s = 0; s < seq_len; ++s) {
        const float* k_h_global = k + s * num_k_heads * head_k_dim + hk * head_k_dim;
        const float* q_h_global = q + s * num_k_heads * head_k_dim + hk * head_k_dim;
        const float* v_h = v + s * num_v_heads * head_v_dim + hv * head_v_dim;
        float* out_h = output + s * num_v_heads * head_v_dim + hv * head_v_dim;
        float g = expf(gate[s * num_v_heads + hv]);
        float b = beta[s * num_v_heads + hv];

        for (int i = tid; i < head_k_dim; i += blockDim.x) {
            smem_k_h[i] = k_h_global[i];
            smem_q_h[i] = q_h_global[i];
        }
        __syncthreads();

        for (int j = 0; j < max_active_j; ++j) {
            S[j * head_v_dim + tid] *= g;
        }

        float sum = 0.0f;
        for (int j = 0; j < head_k_dim; ++j) {
            sum += S[j * head_v_dim + tid] * smem_k_h[j];
        }
        float delta_val = (v_h[tid] - sum) * b;

        for (int j = 0; j < max_active_j; ++j) {
            S[j * head_v_dim + tid] += smem_k_h[j] * delta_val;
        }

        float sum2 = 0.0f;
        for (int j = 0; j < head_k_dim; ++j) {
            sum2 += S[j * head_v_dim + tid] * smem_q_h[j];
        }
        out_h[tid] = sum2 * scale;
    }
}

void launch_ssm_gated_delta_net(
    const float* q, const float* k, const float* v,
    const float* gate, const float* beta,
    float* ssm_state, float* output,
    int seq_len, int head_k_dim, int head_v_dim,
    int num_k_heads, int num_v_heads,
    cudaStream_t stream) {
    int shared_bytes = 2 * head_k_dim * sizeof(float);
    int threads = head_v_dim;
    ssm_gated_delta_net_kernel<<<num_v_heads, threads, shared_bytes, stream>>>(
        q, k, v, gate, beta, ssm_state, output,
        seq_len, head_k_dim, head_v_dim, num_k_heads, num_v_heads);
}

// ============================================================================
// Kernel 6: SSM Gated Norm
//   Per-head RMSNorm on delta_out, then multiply by SiLU(z)
// ============================================================================
__global__ void ssm_gated_norm_kernel(
    float* __restrict__ delta_out,
    const float* __restrict__ z,
    const float* __restrict__ norm_w,
    int seq_len, int head_v_dim, int num_v_heads, float eps) {
    int head_idx = blockIdx.x;
    int total_heads = seq_len * num_v_heads;
    if (head_idx >= total_heads) return;

    int s = head_idx / num_v_heads;
    int hv = head_idx % num_v_heads;

    float* out_head = delta_out + s * num_v_heads * head_v_dim + hv * head_v_dim;
    const float* z_head = z + s * num_v_heads * head_v_dim + hv * head_v_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x) {
        sum_sq += out_head[i] * out_head[i];
    }

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum_sq);
    __syncthreads();

    float inv_rms = rsqrtf(s_sum / head_v_dim + eps);

    for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x) {
        float gz = z_head[i] / (1.0f + expf(-z_head[i]));
        out_head[i] = out_head[i] * inv_rms * norm_w[i] * gz;
    }
}

void launch_ssm_gated_norm(
    float* delta_out,
    const float* z, const float* norm_w,
    int seq_len, int head_v_dim, int num_v_heads, float eps,
    cudaStream_t stream) {
    int total_heads = seq_len * num_v_heads;
    int threads = std::min(head_v_dim, 1024);
    ssm_gated_norm_kernel<<<total_heads, threads, 0, stream>>>(
        delta_out, z, norm_w, seq_len, head_v_dim, num_v_heads, eps);
}

}  // namespace cuda
}  // namespace forge
