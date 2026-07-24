#include <cmath>
#include <cfloat>

#include "cuda_common.h"
#include "cuda_gemv_tmpl.cuh"

namespace forge {
namespace cuda {

// ============================================================================
// MoE Router Kernel — softmax over experts + top-K selection + renormalize
// ============================================================================
// Strategy:
//   Single-block kernel (n_expert ≤ 256 for Gemma4, typically 8).
//   Each thread handles one expert score. Warp-level reduction for max/exp/sum.
//   Top-K via sequential selection (small K, small n_expert).

__global__ void moe_router_kernel(
    const float* __restrict__ logits,     // [seq_len, n_expert]
    int* __restrict__ expert_indices,     // [seq_len, n_expert_used] (output)
    float* __restrict__ expert_weights,   // [seq_len, n_expert_used] (output)
    float* __restrict__ softmax_buf,      // [seq_len * n_expert] temporary
    int n_expert, int n_expert_used, int seq_len)
{
    // Process one token per block
    int s = blockIdx.x;
    int tid = threadIdx.x;
    int n = n_expert;

    const float* s_logits = logits + s * n_expert;
    float* s_buf = softmax_buf + s * n_expert;

    // Stage 1: softmax
    __shared__ float s_max_val;
    __shared__ float s_inv_sum;

    float my_val = (tid < n) ? s_logits[tid] : -FLT_MAX;

    // Max reduction via warp shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_val = fmaxf(my_val, __shfl_down_sync(0xFFFFFFFF, my_val, offset));
    }
    if (tid == 0) s_max_val = my_val;
    __syncthreads();

    // Exp + sum
    float my_exp = 0.0f;
    if (tid < n) {
        float v = expf(s_logits[tid] - s_max_val);
        s_buf[tid] = v;
        my_exp = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_exp += __shfl_down_sync(0xFFFFFFFF, my_exp, offset);
    }
    if (tid == 0) s_inv_sum = 1.0f / (my_exp + 1e-8f);
    __syncthreads();

    if (tid < n) s_buf[tid] *= s_inv_sum;
    __syncthreads();

    // Stage 2: top-K selection (sequential, n_expert ≤ 256, n_expert_used ≤ 8)
    int* s_indices = expert_indices + s * n_expert_used;
    float* s_weights = expert_weights + s * n_expert_used;

    if (tid < n_expert_used) {
        int best_idx = -1;
        float best_val = -1.0f;
        for (int e = 0; e < n_expert; ++e) {
            bool already = false;
            for (int k = 0; k < tid; ++k) {
                if (s_indices[k] == e) { already = true; break; }
            }
            if (!already && s_buf[e] > best_val) {
                best_val = s_buf[e];
                best_idx = e;
            }
        }
        s_indices[tid] = best_idx;
        s_weights[tid] = best_val;
    }
    __syncthreads();

    // Stage 3: renormalize top-K weights
    __shared__ float s_inv_topk;
    float topk_sum = 0.0f;
    if (tid < n_expert_used) topk_sum = s_weights[tid];
    for (int offset = 16; offset > 0; offset >>= 1) {
        topk_sum += __shfl_down_sync(0xFFFFFFFF, topk_sum, offset);
    }
    if (tid == 0) s_inv_topk = 1.0f / (topk_sum + 1e-8f);
    __syncthreads();

    if (tid < n_expert_used) s_weights[tid] *= s_inv_topk;
}

void launch_moe_router(
    const float* logits, int* expert_indices, float* expert_weights,
    float* softmax_buf, int n_expert, int n_expert_used, int seq_len,
    cudaStream_t stream)
{
    int threads = min(n_expert, 256);
    moe_router_kernel<<<seq_len, threads, 0, stream>>>(
        logits, expert_indices, expert_weights, softmax_buf,
        n_expert, n_expert_used, seq_len);
}

// ============================================================================
// MoE Router Scaling Kernel — pre-norm + inv_sqrt + element-wise scale
// ============================================================================
// Applies: x = rms_norm(x, ones, eps) * (1/sqrt(hidden_dim)) * scale_per_dim

__global__ void moe_router_scale_kernel(
    const float* __restrict__ x,            // [seq_len, hidden_dim]
    const float* __restrict__ scale,         // [hidden_dim] per-dim scale (ffn_gate_inp_s)
    float* __restrict__ out,                 // [seq_len, hidden_dim] (output)
    int hidden_dim, float inv_sqrt, float eps, int seq_len)
{
    int s = blockIdx.x;  // One block per token
    int tid = threadIdx.x;

    const float* x_row = x + s * hidden_dim;
    float* out_row = out + s * hidden_dim;
    const float* s_row = scale;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int d = tid; d < hidden_dim; d += blockDim.x) {
        sum_sq += x_row[d] * x_row[d];
    }

    __shared__ float s_sum;
    s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum_sq);
    __syncthreads();

    float rms = rsqrtf(s_sum / hidden_dim + eps);
    float factor = rms * inv_sqrt;

    for (int d = tid; d < hidden_dim; d += blockDim.x) {
        out_row[d] = x_row[d] * factor * s_row[d];
    }
}

void launch_moe_router_scale(
    const float* x, const float* scale, float* out,
    int hidden_dim, float inv_sqrt, float eps, int seq_len,
    cudaStream_t stream)
{
    int threads = min(hidden_dim, 1024);
    moe_router_scale_kernel<<<seq_len, threads, 0, stream>>>(
        x, scale, out, hidden_dim, inv_sqrt, eps, seq_len);
}

// ============================================================================
// MoE Expert GEMV Kernel — quantized 3D weight, split-K per row
// ============================================================================
// Weight layout: [N, K, n_expert] in quantized blocks.
// Each expert's rows are stored contiguously:
//   expert e starts at: q_w_3d + e * N * num_blocks_row * BS
//
// Grid: (token_index, expert_slot) — each block processes one (token, expert_k)
// Warp layout: split-K — multiple warps per output row, atomicAdd to combine.

template <DataType DT>
__global__ void moe_expert_gemv_kernel(
    const float* __restrict__ x,               // [n_tokens, K]
    const uint8_t* __restrict__ q_w_3d,        // [N, K, n_expert] quantized
    float* __restrict__ out,                    // [n_tokens, N] (output, pre-zeroed)
    const int* __restrict__ expert_indices,     // [n_tokens, n_expert_used] on GPU
    const float* __restrict__ expert_weights,   // [n_tokens, n_expert_used] on GPU
    int K, int N, int n_expert, int n_expert_used, int n_tokens)
{
    using Traits = CudaQuantTraits<DT>;
    constexpr int BE = Traits::block_elements;
    constexpr int BS = Traits::block_size;
    int num_blocks_row = (K + BE - 1) / BE;

    int token = blockIdx.x;
    int expert_k = blockIdx.y;
    if (token >= n_tokens || expert_k >= n_expert_used) return;

    int expert_idx = expert_indices[token * n_expert_used + expert_k];
    float weight = expert_weights[token * n_expert_used + expert_k];

    // Expert base pointer in 3D weight tensor
    const uint8_t* expert_base = q_w_3d + (size_t)expert_idx * N * num_blocks_row * BS;

    const float* x_token = x + token * K;

    // Each warp handles one output row with split-K
    // Simplify: use 1D thread indexing
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_in_block = tid / 32;

    int warps_per_block = blockDim.x / 32;
    int total_warps_this_block = warps_per_block;
    int warps_per_row = max(1, total_warps_this_block / N);  // how many warps share one row
    int rows_per_warp = max(1, N / total_warps_this_block);  // how many rows each warp handles

    // Simpler strategy: each warp handles a subset of rows
    for (int row = warp_in_block; row < N; row += warps_per_block) {
        const uint8_t* row_ptr = expert_base + (size_t)row * num_blocks_row * BS;

        float sum = 0.0f;
        for (int bi = lane; bi < num_blocks_row; bi += 32) {
            const uint8_t* block_ptr = row_ptr + bi * BS;
            int base = bi * BE;
            sum += Traits::dot_block(block_ptr, x_token, base, K);
        }

        // Warp reduce
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

        if (lane == 0) {
            atomicAdd(&out[token * N + row], weight * sum);
        }
    }
}

template <DataType DT>
void launch_moe_expert_gemv(
    const float* x, const void* q_w_3d, float* out,
    const int* expert_indices, const float* expert_weights,
    int K, int N, int n_expert, int n_expert_used, int n_tokens,
    cudaStream_t stream)
{
    // Each block handles one (token, expert_slot) pair
    dim3 grid(n_tokens, n_expert_used);
    // Use enough warps to cover N rows
    int warps = min(max(N, 4), 32);  // 4~32 warps per block
    int threads = warps * 32;
    moe_expert_gemv_kernel<DT><<<grid, threads, 0, stream>>>(
        x, static_cast<const uint8_t*>(q_w_3d), out,
        expert_indices, expert_weights,
        K, N, n_expert, n_expert_used, n_tokens);
}

// Explicit template instantiations
template void launch_moe_expert_gemv<DataType::Q4_0>(
    const float*, const void*, float*, const int*, const float*,
    int, int, int, int, int, cudaStream_t);
template void launch_moe_expert_gemv<DataType::Q4_K>(
    const float*, const void*, float*, const int*, const float*,
    int, int, int, int, int, cudaStream_t);
template void launch_moe_expert_gemv<DataType::Q6_K>(
    const float*, const void*, float*, const int*, const float*,
    int, int, int, int, int, cudaStream_t);
template void launch_moe_expert_gemv<DataType::Q8_0>(
    const float*, const void*, float*, const int*, const float*,
    int, int, int, int, int, cudaStream_t);

// ============================================================================
// GeGLU Split Kernel — gate_up fused: split + GELU(tanh) + multiply
// ============================================================================
// Input:  gate_up [n_tokens, 2*half_dim]  (gate in first half, up in second half)
// Output: out     [n_tokens, half_dim]    (gelu(gate) * up)

__global__ void gelu_tanh_multiply_split_kernel(
    const float* __restrict__ gate_up,    // [n_tokens, 2*half_dim]
    float* __restrict__ out,              // [n_tokens, half_dim]
    int half_dim, int total_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int s = idx / half_dim;
    int d = idx % half_dim;

    float g = gate_up[s * 2 * half_dim + d];              // gate half
    float u = gate_up[s * 2 * half_dim + half_dim + d];   // up half

    // GELU(tanh) approximation
    float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
    out[idx] = gelu * u;
}

void launch_gelu_tanh_multiply_split(
    const float* gate_up, float* out, int half_dim, int n_tokens,
    cudaStream_t stream)
{
    int total = n_tokens * half_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gelu_tanh_multiply_split_kernel<<<blocks, threads, 0, stream>>>(
        gate_up, out, half_dim, total);
}

}  // namespace cuda
}  // namespace forge
