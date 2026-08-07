#include <cmath>
#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/op_dispatch.h"
#include "forge/operators.h"

#include "cpu/simd.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {
namespace {

// ---- Phase 3: dst-injection kernels ----
// These kernels write directly into dst->data(), eliminating graph-mode memcpy.

// ---- ADD (dst injection) ----
void add_kernel_dst(const std::vector<TensorPtr>& inputs, TensorPtr dst, const int32_t*) {
    auto a = inputs[0];
    auto b = inputs[1];
    int n = static_cast<int>(a->numel());
    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_add_bias(static_cast<const float*>(a->data()),
                              static_cast<const float*>(b->data()),
                              static_cast<float*>(dst->data()), n);
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* o_data = static_cast<float*>(dst->data());
        forge::cpu::add_f32_vec(a_data, b_data, o_data, n);
    }
}

// ---- MUL (dst injection) ----
void mul_kernel_dst(const std::vector<TensorPtr>& inputs, TensorPtr dst, const int32_t*) {
    auto a = inputs[0];
    auto b = inputs[1];
    int n = static_cast<int>(a->numel());
    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_multiply(static_cast<const float*>(a->data()),
                              static_cast<const float*>(b->data()),
                              static_cast<float*>(dst->data()), n);
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* o_data = static_cast<float*>(dst->data());
        forge::cpu::mul_f32_vec(a_data, b_data, o_data, n);
    }
}

// ---- SILU (dst injection) ----
void silu_kernel_dst(const std::vector<TensorPtr>& inputs, TensorPtr dst, const int32_t*) {
    auto x = inputs[0];
    int n = static_cast<int>(x->numel());
    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_silu(static_cast<const float*>(x->data()),
                          static_cast<float*>(dst->data()), n);
#endif
    } else {
        const float* x_data = static_cast<const float*>(x->data());
        float* o_data = static_cast<float*>(dst->data());
        for (int i = 0; i < n; ++i) {
            float v = x_data[i];
            o_data[i] = v / (1.0f + std::exp(-v));
        }
    }
}

// ---- RMS_NORM (dst injection) ----
// inputs[0] = x, inputs[1] = weight
// params[0..1] = float eps
void rms_norm_kernel_dst(const std::vector<TensorPtr>& inputs, TensorPtr dst,
                         const int32_t* params) {
    auto x = inputs[0];
    auto weight = (inputs.size() > 1) ? inputs[1] : nullptr;
    float eps = 1e-6f;
    if (params) {
        std::memcpy(&eps, params, sizeof(float));
    }

    int rows = static_cast<int>(x->shape()[0]);
    int cols = static_cast<int>(x->shape()[1]);

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        if (weight) {
            cuda::launch_rms_norm(static_cast<const float*>(x->data()),
                                  static_cast<const float*>(weight->data()),
                                  static_cast<float*>(dst->data()), rows, cols, eps);
        } else {
            std::vector<float> ones(cols, 1.0f);
            cuda::launch_rms_norm(static_cast<const float*>(x->data()), ones.data(),
                                  static_cast<float*>(dst->data()), rows, cols, eps);
        }
#endif
    } else {
        const float* x_data = static_cast<const float*>(x->data());
        const float* w_data = weight ? static_cast<const float*>(weight->data()) : nullptr;
        float* o_data = static_cast<float*>(dst->data());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (rows > 1)
#endif
        for (int r = 0; r < rows; ++r) {
            const float* x_row = x_data + r * cols;
            float* o_row = o_data + r * cols;
            forge::cpu::rms_norm_row_f32(x_row, w_data, o_row, cols, eps);
        }
    }
}

// ---- ROPE (dst injection) ----
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
void rope_kernel_dst(const std::vector<TensorPtr>& inputs, TensorPtr dst, const int32_t* params) {
    auto x = inputs[0];
    if (!params)
        return;

    int is_q = params[0];
    int num_h = params[1];
    int head_dim = params[2];
    int seq_len = params[3];
    int64_t start_pos;
    std::memcpy(&start_pos, params + 4, sizeof(int64_t));
    float rope_theta;
    std::memcpy(&rope_theta, params + 6, sizeof(float));
    DeviceType dev = (params[9] == 1) ? DeviceType::CUDA : DeviceType::CPU;

    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        if (is_q) {
            cuda::launch_rope_gqa(static_cast<const float*>(x->data()), nullptr, nullptr,
                                  static_cast<float*>(dst->data()), num_h, 0, head_dim,
                                  seq_len, start_pos, rope_theta);
        } else {
            cuda::launch_rope_gqa(nullptr, static_cast<const float*>(x->data()), nullptr,
                                  static_cast<float*>(dst->data()), 0, num_h, head_dim, seq_len,
                                  start_pos, rope_theta);
        }
#endif
    } else {
        int half_dim = head_dim / 2;
        int stride = num_h * head_dim;
        const float* x_data = static_cast<const float*>(x->data());
        float* o_data = static_cast<float*>(dst->data());
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
}

// ---- Legacy kernels (for ops not yet migrated to dst injection) ----

// ---- MUL_MAT_TRANSB (legacy) ----
TensorPtr mul_mat_transb_kernel(const std::vector<TensorPtr>& inputs, const int32_t*) {
    TensorPtr bias = (inputs.size() > 2) ? inputs[2] : nullptr;
    return ops::matmul_transB(inputs[0], inputs[1], bias);
}

// ---- FLASH_ATTN_GQA (legacy) ----
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

    // ---- Phase 3: dst-injection kernels ----
    // ADD/MUL/SILU/RMS_NORM/ROPE migrated to dst injection
    d.register_kernel_dst(OpType::ADD, DeviceType::CPU, add_kernel_dst);
    d.register_kernel_dst(OpType::ADD, DeviceType::CUDA, add_kernel_dst);

    d.register_kernel_dst(OpType::MUL, DeviceType::CPU, mul_kernel_dst);
    d.register_kernel_dst(OpType::MUL, DeviceType::CUDA, mul_kernel_dst);

    d.register_kernel_dst(OpType::SILU, DeviceType::CPU, silu_kernel_dst);
    d.register_kernel_dst(OpType::SILU, DeviceType::CUDA, silu_kernel_dst);

    d.register_kernel_dst(OpType::RMS_NORM, DeviceType::CPU, rms_norm_kernel_dst);
    d.register_kernel_dst(OpType::RMS_NORM, DeviceType::CUDA, rms_norm_kernel_dst);

    d.register_kernel_dst(OpType::ROPE, DeviceType::CPU, rope_kernel_dst);
    d.register_kernel_dst(OpType::ROPE, DeviceType::CUDA, rope_kernel_dst);

    // ---- Legacy kernels (ops not yet migrated) ----
    d.register_kernel(OpType::MUL_MAT_TRANSB, DeviceType::CPU, mul_mat_transb_kernel);
    d.register_kernel(OpType::MUL_MAT_TRANSB, DeviceType::CUDA, mul_mat_transb_kernel);

    d.register_kernel(OpType::FLASH_ATTN_GQA, DeviceType::CPU, flash_attn_gqa_kernel);
    d.register_kernel(OpType::FLASH_ATTN_GQA, DeviceType::CUDA, flash_attn_gqa_kernel);

    return true;
}
static bool _kernels_registered = register_all_kernels();

}  // anonymous namespace

// ---- 显式注册入口 ----
void register_builtin_op_kernels() {
    static bool done = false;
    if (done) return;
    register_all_kernels();
    done = true;
}

}  // namespace forge