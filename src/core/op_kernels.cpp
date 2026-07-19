#include <cmath>
#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/op_dispatch.h"
#include "forge/operators.h"

namespace forge {
namespace {

// ---- ADD ----
TensorPtr add_kernel(const std::vector<TensorPtr>& inputs, const int32_t*) {
    return ops::add(inputs[0], inputs[1]);
}

// ---- MUL ----
TensorPtr mul_kernel(const std::vector<TensorPtr>& inputs, const int32_t*) {
    return ops::multiply(inputs[0], inputs[1]);
}

// ---- SILU ----
TensorPtr silu_kernel(const std::vector<TensorPtr>& inputs, const int32_t*) {
    return ops::silu(inputs[0]);
}

// ---- RMS_NORM ----
// inputs[0] = x, inputs[1] = weight
// params[0..1] = float eps
TensorPtr rms_norm_kernel(const std::vector<TensorPtr>& inputs, const int32_t* params) {
    float eps = 1e-6f;
    if (params) {
        std::memcpy(&eps, params, sizeof(float));
    }
    return ops::rms_norm(inputs[0], inputs[1], eps);
}

// ---- MUL_MAT_TRANSB ----
// inputs[0] = a, inputs[1] = b, inputs[2] = bias (optional)
TensorPtr mul_mat_transb_kernel(const std::vector<TensorPtr>& inputs, const int32_t*) {
    TensorPtr bias = (inputs.size() > 2) ? inputs[2] : nullptr;
    return ops::matmul_transB(inputs[0], inputs[1], bias);
}

// ---- ROPE ----
// inputs[0] = tensor (either Q or K)
// params layout:
//   [0] = is_q (0 = K, 1 = Q)
//   [1] = num_heads (for Q) or num_kv_heads (for K)
//   [2] = head_dim
//   [3] = seq_len
//   [4..5] = int64_t start_pos
//   [6..7] = float rope_theta
//   [8] = use_neox (0/1)
//   [9] = device_type (0=CPU, 1=CUDA)
TensorPtr rope_kernel(const std::vector<TensorPtr>& inputs, const int32_t* params) {
    auto x = inputs[0];
    if (!params)
        return nullptr;

    int is_q = params[0];
    int num_h = params[1];  // num_heads (Q) or num_kv_heads (K)
    int head_dim = params[2];
    int seq_len = params[3];
    int64_t start_pos;
    std::memcpy(&start_pos, params + 4, sizeof(int64_t));
    float rope_theta;
    std::memcpy(&rope_theta, params + 6, sizeof(float));
    DeviceType dev = (params[9] == 1) ? DeviceType::CUDA : DeviceType::CPU;

    auto out = std::make_shared<Tensor>(DataType::FP32, x->shape(), dev);

    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        if (is_q) {
            cuda::launch_rope_gqa(static_cast<const float*>(x->data()), nullptr,
                                  static_cast<float*>(out->data()), nullptr, num_h, 0, head_dim,
                                  seq_len, start_pos, rope_theta);
        } else {
            cuda::launch_rope_gqa(nullptr, static_cast<const float*>(x->data()), nullptr,
                                  static_cast<float*>(out->data()), 0, num_h, head_dim, seq_len,
                                  start_pos, rope_theta);
        }
#endif
    } else {
        // CPU rope (adapted from rope only helpers)
        int half_dim = head_dim / 2;
        int stride = num_h * head_dim;
        const float* x_data = static_cast<const float*>(x->data());
        float* o_data = static_cast<float*>(out->data());
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_h; ++h) {
                for (int d = 0; d < half_dim; ++d) {
                    float freq = 1.0f / std::pow(rope_theta, 2.0f * d / head_dim);
                    float angle = (start_pos + s) * freq;
                    float cos_a = std::cos(angle);
                    float sin_a = std::sin(angle);
                    int idx0 = s * stride + h * head_dim + d;
                    int idx1 = idx0 + half_dim;
                    o_data[idx0] = x_data[idx0] * cos_a - x_data[idx1] * sin_a;
                    o_data[idx1] = x_data[idx0] * sin_a + x_data[idx1] * cos_a;
                }
            }
        }
    }
    return out;
}

// ---- FLASH_ATTN_GQA ----
// inputs[0] = q, inputs[1] = k, inputs[2] = v
// params:
//   [0] = num_heads
//   [1] = num_kv_heads
//   [2] = head_dim
//   [3] = causal (0/1)
//   [4] = device_type (0=CPU, 1=CUDA)
TensorPtr flash_attn_gqa_kernel(const std::vector<TensorPtr>& inputs, const int32_t* params) {
    auto q = inputs[0];
    auto k = inputs[1];
    auto v = inputs[2];
    if (!params)
        return nullptr;

    int num_heads = params[0];
    int num_kv_heads = params[1];
    int head_dim = params[2];
    bool causal = params[3] != 0;
    DeviceType dev = (params[4] == 1) ? DeviceType::CUDA : DeviceType::CPU;

    int seq_len_q = static_cast<int>(q->shape()[0]);
    int total_len = static_cast<int>(k->shape()[0]);

    if (dev == DeviceType::CUDA && num_kv_heads < num_heads) {
        auto attn_out = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len_q, num_heads * head_dim},
            DeviceType::CUDA);
        if (seq_len_q == 1) {
#ifdef USE_CUDA
            cuda::launch_flash_attention_gqa_decode(
                static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                static_cast<const float*>(v->data()), static_cast<float*>(attn_out->data()),
                total_len, num_heads, num_kv_heads, head_dim, nullptr);
#endif
        } else {
#ifdef USE_CUDA
            cuda::launch_flash_attention_gqa(
                static_cast<const float*>(q->data()), static_cast<const float*>(k->data()),
                static_cast<const float*>(v->data()), static_cast<float*>(attn_out->data()),
                seq_len_q, total_len, num_heads, num_kv_heads, head_dim, nullptr, true);
#endif
        }
        return attn_out;
    } else if (dev == DeviceType::CUDA) {
        return ops::scaled_dot_product_attention_2d(q, k, v, seq_len_q, total_len, num_heads,
                                                    head_dim, nullptr, causal);
    } else {
        // CPU path: GQA expand if needed
        TensorPtr k_expanded, v_expanded;
        if (num_kv_heads < num_heads) {
            int kv_groups = num_heads / num_kv_heads;
            k_expanded = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{total_len, num_heads * head_dim}, dev);
            v_expanded = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{total_len, num_heads * head_dim}, dev);
            const float* k_data = static_cast<const float*>(k->data());
            float* k_out = static_cast<float*>(k_expanded->data());
            const float* v_data = static_cast<const float*>(v->data());
            float* v_out = static_cast<float*>(v_expanded->data());
            for (int s = 0; s < total_len; ++s) {
                for (int h = 0; h < num_heads; ++h) {
                    int kv_h = h / kv_groups;
                    for (int d = 0; d < head_dim; ++d) {
                        k_out[s * num_heads * head_dim + h * head_dim + d] =
                            k_data[s * num_kv_heads * head_dim + kv_h * head_dim + d];
                        v_out[s * num_heads * head_dim + h * head_dim + d] =
                            v_data[s * num_kv_heads * head_dim + kv_h * head_dim + d];
                    }
                }
            }
        } else {
            k_expanded = k;
            v_expanded = v;
        }
        return ops::scaled_dot_product_attention_2d(q, k_expanded, v_expanded, seq_len_q, total_len,
                                                    num_heads, head_dim, nullptr, causal);
    }
}

// Auto-register all kernels at startup
static bool register_all_kernels() {
    auto& d = OpDispatch::instance();

    d.register_kernel(OpType::ADD, DeviceType::CPU, add_kernel);
    d.register_kernel(OpType::ADD, DeviceType::CUDA, add_kernel);

    d.register_kernel(OpType::MUL, DeviceType::CPU, mul_kernel);
    d.register_kernel(OpType::MUL, DeviceType::CUDA, mul_kernel);

    d.register_kernel(OpType::SILU, DeviceType::CPU, silu_kernel);
    d.register_kernel(OpType::SILU, DeviceType::CUDA, silu_kernel);

    d.register_kernel(OpType::RMS_NORM, DeviceType::CPU, rms_norm_kernel);
    d.register_kernel(OpType::RMS_NORM, DeviceType::CUDA, rms_norm_kernel);

    d.register_kernel(OpType::MUL_MAT_TRANSB, DeviceType::CPU, mul_mat_transb_kernel);
    d.register_kernel(OpType::MUL_MAT_TRANSB, DeviceType::CUDA, mul_mat_transb_kernel);

    d.register_kernel(OpType::ROPE, DeviceType::CPU, rope_kernel);
    d.register_kernel(OpType::ROPE, DeviceType::CUDA, rope_kernel);

    d.register_kernel(OpType::FLASH_ATTN_GQA, DeviceType::CPU, flash_attn_gqa_kernel);
    d.register_kernel(OpType::FLASH_ATTN_GQA, DeviceType::CUDA, flash_attn_gqa_kernel);

    return true;
}
static bool _kernels_registered = register_all_kernels();

}  // anonymous namespace
}  // namespace forge
