#include <cmath>
#include <cfloat>

#include "cuda_common.h"
#include "cuda_gemv_common.cuh"  // block_q8_1_gemv, vec_dot_*_q8_1, GEMV_Q*_K macros
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
    __shared__ float s_inv_topk;

    // Single-thread sequential selection. The per-thread variant raced on
    // s_indices[k]: thread k read indices written by threads < k without a
    // barrier, so the same expert was often selected twice.
    if (tid == 0) {
        float topk_sum = 0.0f;
        for (int k = 0; k < n_expert_used; ++k) {
            int best = -1;
            float bv = -1.0f;
            for (int e = 0; e < n_expert; ++e) {
                bool used = false;
                for (int j = 0; j < k; ++j)
                    if (s_indices[j] == e) { used = true; break; }
                if (!used && s_buf[e] > bv) { bv = s_buf[e]; best = e; }
            }
            s_indices[k] = best;
            s_weights[k] = bv;
            topk_sum += bv;
        }
        s_inv_topk = 1.0f / (topk_sum + 1e-8f);
    }
    __syncthreads();

    // Stage 3: renormalize top-K weights
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

// ============================================================================
// Q8_1 activation quantization (batched) — Phase 1-A
// ============================================================================
// Quantize [n_tokens, K] FP32 activations to Q8_1 (one block_q8_1_gemv per 32
// elements, contiguous across tokens). Mirrors quantize_q8_1_kernel but handles
// the leading n_tokens dimension so the MoE path can quantize the whole batch
// once and reuse it across every routed expert (they share the activation).
__global__ void quantize_q8_1_batch_kernel(
    const float* __restrict__ x, block_q8_1_gemv* __restrict__ q8, int K, int n_tokens)
{
    int num_blocks = (K + 31) / 32;
    int total = n_tokens * num_blocks;
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= total) return;

    int tok = bi / num_blocks;
    int b = bi % num_blocks;
    const float* x_row = x + (size_t)tok * K;
    int base = b * 32;
    int end = min(base + 32, K);

    float amax = 0.0f;
    for (int j = base; j < end; ++j) {
        float ax = fabsf(x_row[j]);
        if (ax > amax) amax = ax;
    }

    float d_val = (amax > 1e-10f) ? (amax / 127.0f) : (1.0f / 127.0f);
    float inv_d = 1.0f / d_val;

    block_q8_1_gemv& blk = q8[bi];
    blk.ds = __halves2half2(__float2half(d_val), __float2half(0.0f));

    for (int j = 0; j < end - base; ++j) {
        float v = x_row[base + j] * inv_d;
        v = fminf(fmaxf(v, -128.0f), 127.0f);
        blk.qs[j] = static_cast<int8_t>(__float2int_rn(v));
    }
}

// ============================================================================
// MoE Expert GEMV Kernel — Q8_1 × quantized weight via shared dp4a primitives
// ============================================================================
// Weight layout: [N, K, n_expert] in quantized blocks.
// Each expert's rows are stored contiguously:
//   expert e starts at: q_w_3d + e * N * num_blocks_row * BS
//
// x is supplied pre-quantized to Q8_1 (quantize_q8_1_batch_kernel) so the
// per-token activation is dequantized-to-int8 exactly once, then every expert
// reuses it. The per-(row, block) dot dispatches on DT to the dp4a primitives
// in cuda_gemv_common.cuh (the same numerics as the decode GEMV path).

// Per (expert row, weight block) Q8_1 dot product. Specialized per DT.
template <DataType DT>
__device__ __forceinline__ float moe_dot_q8_1(
    const uint8_t* __restrict__ row_ptr, int bi,
    const block_q8_1_gemv* __restrict__ x_q8_token);

template <>
__device__ __forceinline__ float moe_dot_q8_1<DataType::Q4_0>(
    const uint8_t* __restrict__ row_ptr, int bi,
    const block_q8_1_gemv* __restrict__ x_q8_token) {
    return vec_dot_q4_0_q8_1(row_ptr + (size_t)bi * 18, x_q8_token + bi);
}

template <>
__device__ __forceinline__ float moe_dot_q8_1<DataType::Q8_0>(
    const uint8_t* __restrict__ row_ptr, int bi,
    const block_q8_1_gemv* __restrict__ x_q8_token) {
    return vec_dot_q8_0_q8_1(row_ptr + (size_t)bi * 34, x_q8_token + bi);
}

template <>
__device__ __forceinline__ float moe_dot_q8_1<DataType::Q4_K>(
    const uint8_t* __restrict__ row_ptr, int bi,
    const block_q8_1_gemv* __restrict__ x_q8_token) {
    constexpr int q8_stride = GEMV_QR4_K * GEMV_QI4_K / GEMV_QI8_1;  // 8
    float sum = 0.0f;
    const block_q8_1_gemv* row_q8 = x_q8_token + (size_t)bi * q8_stride;
    for (int iqs = 0; iqs < GEMV_QI4_K; iqs += 2)
        sum += vec_dot_q4_K_q8_1(row_ptr, row_q8, bi, iqs);
    return sum;
}

template <>
__device__ __forceinline__ float moe_dot_q8_1<DataType::Q6_K>(
    const uint8_t* __restrict__ row_ptr, int bi,
    const block_q8_1_gemv* __restrict__ x_q8_token) {
    constexpr int q8_stride = GEMV_QR6_K * GEMV_QI6_K / GEMV_QI8_1;  // 8
    float sum = 0.0f;
    const block_q8_1_gemv* row_q8 = x_q8_token + (size_t)bi * q8_stride;
    for (int iqs = 0; iqs < GEMV_QI6_K; ++iqs)
        sum += vec_dot_q6_K_q8_1(row_ptr, row_q8, bi, iqs);
    return sum;
}

template <DataType DT>
__global__ void moe_expert_gemv_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,   // [n_tokens, K/32] pre-quantized
    const uint8_t* __restrict__ q_w_3d,          // [N, K, n_expert] quantized
    float* __restrict__ out,                      // [n_tokens, N] (output, pre-zeroed)
    const int* __restrict__ expert_indices,       // [n_tokens, n_expert_used] on GPU
    const float* __restrict__ expert_weights,     // [n_tokens, n_expert_used] on GPU
    int K, int N, int n_expert, int n_expert_used, int n_tokens, int num_q8_blocks)
{
    constexpr int BE = CudaQuantTraits<DT>::block_elements;
    constexpr int BS = CudaQuantTraits<DT>::block_size;
    int num_blocks_row = (K + BE - 1) / BE;

    int token = blockIdx.x;
    int expert_k = blockIdx.y;
    if (token >= n_tokens || expert_k >= n_expert_used) return;

    int expert_idx = expert_indices[token * n_expert_used + expert_k];
    float weight = expert_weights[token * n_expert_used + expert_k];

    // Expert base pointer in 3D weight tensor
    const uint8_t* expert_base = q_w_3d + (size_t)expert_idx * N * num_blocks_row * BS;

    const block_q8_1_gemv* x_q8_token = x_q8 + (size_t)token * num_q8_blocks;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_in_block = tid / 32;

    int warps_per_block = blockDim.x / 32;

    // Each warp handles a subset of rows
    for (int row = warp_in_block; row < N; row += warps_per_block) {
        const uint8_t* row_ptr = expert_base + (size_t)row * num_blocks_row * BS;

        float sum = 0.0f;
        for (int bi = lane; bi < num_blocks_row; bi += 32) {
            sum += moe_dot_q8_1<DT>(row_ptr, bi, x_q8_token);
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
    // Phase 1-A: quantize the activation x ([n_tokens, K]) to Q8_1 ONCE. All
    // routed experts for a token share the same activation, so a single
    // quantization is reused across every (token, expert) GEMV — this is the
    // key win over the old scalar dot_block path.
    int num_q8_blocks = (K + 31) / 32;
    size_t q8_bytes = (size_t)n_tokens * num_q8_blocks * sizeof(block_q8_1_gemv);
    void* q8_buf = scratch_pool().ensure(q8_bytes);
    auto* x_q8 = static_cast<block_q8_1_gemv*>(q8_buf);

    int q_threads = 256;
    int q_blocks = (int)n_tokens * num_q8_blocks;
    quantize_q8_1_batch_kernel<<<q_blocks, q_threads, 0, stream>>>(x, x_q8, K, n_tokens);

    // Each block handles one (token, expert_slot) pair
    dim3 grid(n_tokens, n_expert_used);
    // Use enough warps to cover N rows
    int warps = min(max(N, 4), 32);  // 4~32 warps per block
    int threads = warps * 32;
    moe_expert_gemv_kernel<DT><<<grid, threads, 0, stream>>>(
        x_q8, static_cast<const uint8_t*>(q_w_3d), out,
        expert_indices, expert_weights,
        K, N, n_expert, n_expert_used, n_tokens, num_q8_blocks);
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
