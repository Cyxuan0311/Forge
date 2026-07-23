#pragma once

#include <cstdint>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_ssm_preprocess(
    const float* alpha, const float* beta,
    const float* dt_bias, const float* ssm_a,
    float* gate_out, float* beta_out,
    int seq_len, int num_v_heads,
    cudaStream_t stream = 0);

void launch_ssm_conv1d(
    const float* x, const float* weight,
    float* conv_state, float* y,
    int seq_len, int conv_channels, int d_conv,
    cudaStream_t stream = 0);

void launch_ssm_silu_split(
    const float* conv_out,
    float* q, float* k, float* v,
    int seq_len, int key_dim, int value_dim,
    cudaStream_t stream = 0);

void launch_ssm_per_head_l2norm(
    float* data,
    int seq_len, int num_heads, int head_dim, float eps,
    cudaStream_t stream = 0);

void launch_ssm_gated_delta_net(
    const float* q, const float* k, const float* v,
    const float* gate, const float* beta,
    float* ssm_state, float* output,
    int seq_len, int head_k_dim, int head_v_dim,
    int num_k_heads, int num_v_heads,
    cudaStream_t stream = 0);

void launch_ssm_gated_norm(
    float* delta_out,
    const float* z, const float* norm_w,
    int seq_len, int head_v_dim, int num_v_heads, float eps,
    cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
