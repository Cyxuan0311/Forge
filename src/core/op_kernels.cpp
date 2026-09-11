#include "cpu/common/thread_pool.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/fp8_utils.h"
#include "forge/op_dispatch.h"
#include "forge/operators.h"

#include "cpu/simd.h"


namespace forge {
namespace {

// ---- Phase 3: dst-injection kernels ----
// These kernels write directly into dst->data(), eliminating graph-mode memcpy.

// ---- ADD (dst injection) ----
// 6-A-6.4: stream forwarded to launch_* (default 0 = legacy stream).
// 6-A-8.0: native FP16 branch (FP16 storage, FP32 per-element compute, FP16
// out) -- the residual node of the end-to-end FP16 loop.
void add_kernel_dst(const std::vector<TensorPtr>& inputs, TensorPtr dst, const int32_t*,
                    cudaStream_t stream) {
    auto a = inputs[0];
    auto b = inputs[1];
    int n = static_cast<int>(a->numel());
    // 6-B-5: ADD is retired from require_no_bf16 -- it owns a real
    // __nv_bfloat16 kernel. The BF16 branch below runs before any gate/reject
    // so BF16 never reaches a float* cast.
    if (a->dtype() == DataType::BF16) {
        if (b->dtype() != DataType::BF16) {
            throw std::runtime_error(
                std::string("add_kernel_dst: BF16 a needs a BF16 b, got '") +
                dtype_name(b->dtype()) + "' (mixed dtypes fail loudly).");
        }
        const __nv_bfloat16* a_h = reinterpret_cast<const __nv_bfloat16*>(a->data());
        const __nv_bfloat16* b_h = reinterpret_cast<const __nv_bfloat16*>(b->data());
        __nv_bfloat16* o_h = reinterpret_cast<__nv_bfloat16*>(dst->data());
        if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            require_dtype_supported("add_kernel_dst", OpType::ADD, a->dtype());
            require_dtype_supported("add_kernel_dst", OpType::ADD, b->dtype());
            cuda::launch_add_bf16(a_h, b_h, o_h, n, stream);
#endif
        } else {
            // CPU path is FP32-only by design: upcast locally, add in FP32, store BF16.
            for (int i = 0; i < n; ++i) {
                o_h[i] = __float2bfloat16(__bfloat162float(a_h[i]) + __bfloat162float(b_h[i]));
            }
        }
        return;
    }
    // 6-B-1: remaining non-FP16/BF16 dtypes -- reject BF16 before the FP16
    // branch or any float* cast below. (ADD no longer reaches this line with
    // BF16; the branch above returns first.)
    require_no_bf16("add_kernel_dst", OpType::ADD, a->dtype());
    require_no_bf16("add_kernel_dst", OpType::ADD, b->dtype());
    // 6-C-1: FP8 has no kernel on any ADD path (CUDA fallback is covered by
    // require_dtype_supported via the op_supports_fp8 ledger; this gate covers
    // the bare CPU fallback below).
    require_no_fp8("add_kernel_dst", OpType::ADD, a->dtype());
    require_no_fp8("add_kernel_dst", OpType::ADD, b->dtype());
    if (a->dtype() == DataType::FP16) {
        const __half* a_h = reinterpret_cast<const __half*>(a->data());
        const __half* b_h = reinterpret_cast<const __half*>(b->data());
        __half* o_h = reinterpret_cast<__half*>(dst->data());
        if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            require_dtype_supported("add_kernel_dst", OpType::ADD, a->dtype());
            require_dtype_supported("add_kernel_dst", OpType::ADD, b->dtype());
            cuda::launch_add_fp16(a_h, b_h, o_h, n, stream);
#endif
        } else {
            // CPU path is FP32-only by design: upcast locally, add in FP32, store FP16.
            for (int i = 0; i < n; ++i) {
                o_h[i] = __float2half(__half2float(a_h[i]) + __half2float(b_h[i]));
            }
        }
        return;
    }
    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        require_dtype_supported("add_kernel_dst", OpType::ADD, a->dtype());
        require_dtype_supported("add_kernel_dst", OpType::ADD, b->dtype());
        cuda::launch_add_bias(static_cast<const float*>(a->data()),
                              static_cast<const float*>(b->data()),
                              static_cast<float*>(dst->data()), n, stream);
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* o_data = static_cast<float*>(dst->data());
        forge::cpu::add_f32_vec(a_data, b_data, o_data, n);
    }
}

// ---- MUL (dst injection) ----
// 6-A-8.0: native FP16 branch (FP16 storage, FP32 per-element compute, FP16
// out) -- the gate*up node of the end-to-end FP16 loop.
void mul_kernel_dst(const std::vector<TensorPtr>& inputs, TensorPtr dst, const int32_t*,
                    cudaStream_t stream) {
    auto a = inputs[0];
    auto b = inputs[1];
    int n = static_cast<int>(a->numel());
    // 6-B-5: MUL is retired from require_no_bf16 -- it owns a real
    // __nv_bfloat16 kernel. The BF16 branch below runs before any gate/reject
    // so BF16 never reaches a float* cast.
    if (a->dtype() == DataType::BF16) {
        if (b->dtype() != DataType::BF16) {
            throw std::runtime_error(
                std::string("mul_kernel_dst: BF16 a needs a BF16 b, got '") +
                dtype_name(b->dtype()) + "' (mixed dtypes fail loudly).");
        }
        const __nv_bfloat16* a_h = reinterpret_cast<const __nv_bfloat16*>(a->data());
        const __nv_bfloat16* b_h = reinterpret_cast<const __nv_bfloat16*>(b->data());
        __nv_bfloat16* o_h = reinterpret_cast<__nv_bfloat16*>(dst->data());
        if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            require_dtype_supported("mul_kernel_dst", OpType::MUL, a->dtype());
            require_dtype_supported("mul_kernel_dst", OpType::MUL, b->dtype());
            cuda::launch_multiply_bf16(a_h, b_h, o_h, n, stream);
#endif
        } else {
            // CPU path is FP32-only by design: upcast locally, multiply in FP32, store BF16.
            for (int i = 0; i < n; ++i) {
                o_h[i] = __float2bfloat16(__bfloat162float(a_h[i]) * __bfloat162float(b_h[i]));
            }
        }
        return;
    }
    // 6-B-1: remaining non-FP16/BF16 dtypes -- reject BF16 before the FP16
    // branch or any float* cast below. (MUL no longer reaches this line with
    // BF16; the branch above returns first.)
    require_no_bf16("mul_kernel_dst", OpType::MUL, a->dtype());
    require_no_bf16("mul_kernel_dst", OpType::MUL, b->dtype());
    // 6-C-1: FP8 gate -- mirrors ADD (covers the bare CPU fallback below).
    require_no_fp8("mul_kernel_dst", OpType::MUL, a->dtype());
    require_no_fp8("mul_kernel_dst", OpType::MUL, b->dtype());
    if (a->dtype() == DataType::FP16) {
        const __half* a_h = reinterpret_cast<const __half*>(a->data());
        const __half* b_h = reinterpret_cast<const __half*>(b->data());
        __half* o_h = reinterpret_cast<__half*>(dst->data());
        if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            require_dtype_supported("mul_kernel_dst", OpType::MUL, a->dtype());
            require_dtype_supported("mul_kernel_dst", OpType::MUL, b->dtype());
            cuda::launch_multiply_fp16(a_h, b_h, o_h, n, stream);
#endif
        } else {
            // CPU path is FP32-only by design: upcast locally, multiply in FP32, store FP16.
            for (int i = 0; i < n; ++i) {
                o_h[i] = __float2half(__half2float(a_h[i]) * __half2float(b_h[i]));
            }
        }
        return;
    }
    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        require_dtype_supported("mul_kernel_dst", OpType::MUL, a->dtype());
        require_dtype_supported("mul_kernel_dst", OpType::MUL, b->dtype());
        cuda::launch_multiply(static_cast<const float*>(a->data()),
                              static_cast<const float*>(b->data()),
                              static_cast<float*>(dst->data()), n, stream);
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* o_data = static_cast<float*>(dst->data());
        forge::cpu::mul_f32_vec(a_data, b_data, o_data, n);
    }
}

// ---- SILU (dst injection) ----
void silu_kernel_dst(const std::vector<TensorPtr>& inputs, TensorPtr dst, const int32_t*,
                     cudaStream_t stream) {
    auto x = inputs[0];
    int n = static_cast<int>(x->numel());
    // 6-B-3: SILU is retired from require_no_bf16 -- it owns a real
    // __nv_bfloat16 kernel. The BF16 branch below runs before any gate/reject
    // so BF16 never reaches a float* cast.
    if (x->dtype() == DataType::BF16) {
        const __nv_bfloat16* x_h = reinterpret_cast<const __nv_bfloat16*>(x->data());
        __nv_bfloat16* o_h = reinterpret_cast<__nv_bfloat16*>(dst->data());
        if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            require_dtype_supported("silu_kernel_dst", OpType::SILU, x->dtype());
            cuda::launch_silu_bf16(x_h, o_h, n, stream);
#endif
        } else {
            // CPU path is FP32-only by design: upcast locally, compute in FP32, store BF16.
            for (int i = 0; i < n; ++i) {
                float v = __bfloat162float(x_h[i]);
                o_h[i] = __float2bfloat16(v / (1.0f + std::exp(-v)));
            }
        }
        return;
    }
    // 6-B-1: remaining non-FP16/BF16 dtypes -- reject BF16 before the FP16
    // branch or any float* cast below. (SILU no longer reaches this line with
    // BF16; the branch above returns first.)
    require_no_bf16("silu_kernel_dst", OpType::SILU, x->dtype());
    // 6-C-3: native FP8 dst branch -- FP8 x routes to the real FP8 SiLU kernel
    // (SILU retired from the op_supports_fp8 ledger in 6-C-3). Mixed dtypes fall
    // through to require_no_fp8 below and fail loudly.
    if (x->dtype() == DataType::FP8_E4M3) {
        const uint8_t* x_p = static_cast<const uint8_t*>(x->data());
        uint8_t* o_p = static_cast<uint8_t*>(dst->data());
        if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            require_dtype_supported("silu_kernel_dst", OpType::SILU, x->dtype());
            cuda::launch_silu_fp8(x_p, o_p, n, stream);
#endif
        } else {
            // CPU: decode FP8, FP32 SiLU, encode FP8 (FP8 is a real native dtype
            // path here, mirroring the 6-C-2 RMSNorm CPU FP8 decode/encode).
            for (int i = 0; i < n; ++i) {
                float v = fp8_e4m3_to_fp32(x_p[i]);
                o_p[i] = fp32_to_fp8_e4m3(v / (1.0f + std::exp(-v)));
            }
        }
        return;
    }
    // 6-C-1: FP8 gate -- mirrors ADD/MUL (covers the bare CPU fallback below).
    require_no_fp8("silu_kernel_dst", OpType::SILU, x->dtype());
    if (x->dtype() == DataType::FP16) {
        const __half* x_h = reinterpret_cast<const __half*>(x->data());
        __half* o_h = reinterpret_cast<__half*>(dst->data());
        if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            cuda::launch_silu_fp16(x_h, o_h, n, stream);
#endif
        } else {
            std::vector<float> xf(n), of(n);
            for (int i = 0; i < n; ++i) xf[i] = __half2float(x_h[i]);
            for (int i = 0; i < n; ++i) of[i] = xf[i] / (1.0f + std::exp(-xf[i]));
            for (int i = 0; i < n; ++i) o_h[i] = __float2half(of[i]);
        }
        return;
    }
    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        require_dtype_supported("silu_kernel_dst", OpType::SILU, x->dtype());
        cuda::launch_silu(static_cast<const float*>(x->data()),
                          static_cast<float*>(dst->data()), n, stream);
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
                         const int32_t* params, cudaStream_t stream) {
    auto x = inputs[0];
    auto weight = (inputs.size() > 1) ? inputs[1] : nullptr;
    float eps = 1e-6f;
    if (params) {
        std::memcpy(&eps, params, sizeof(float));
    }

    int rows = static_cast<int>(x->shape()[0]);
    int cols = static_cast<int>(x->shape()[1]);

    // 6-B-2: RMS_NORM is retired from require_no_bf16 -- it owns a real
    // __nv_bfloat16 kernel (BF16 storage, FP32 reduction). The BF16 branch
    // below runs before any gate/reject so BF16 never reaches a float* cast.
    if (x->dtype() == DataType::BF16) {
        const __nv_bfloat16* x_h = reinterpret_cast<const __nv_bfloat16*>(x->data());
        const __nv_bfloat16* w_h =
            weight ? reinterpret_cast<const __nv_bfloat16*>(weight->data()) : nullptr;
        __nv_bfloat16* o_h = reinterpret_cast<__nv_bfloat16*>(dst->data());
        if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            require_dtype_supported("rms_norm_kernel_dst", OpType::RMS_NORM, x->dtype());
            if (weight) {
                require_dtype_supported("rms_norm_kernel_dst/weight", OpType::RMS_NORM,
                                        weight->dtype());
                if (weight->dtype() != DataType::BF16) {
                    throw std::runtime_error(
                        std::string("rms_norm_kernel_dst: BF16 x needs a BF16 weight, got '") +
                        dtype_name(weight->dtype()) + "' (mixed dtypes fail loudly).");
                }
                cuda::launch_rms_norm_bf16(x_h, w_h, o_h, rows, cols, eps, stream);
            } else {
                cuda::launch_rms_norm_unweighted_bf16(x_h, o_h, rows, cols, eps, stream);
            }
#endif
        } else {
            // CPU path is FP32-only by design: upcast locally, run the FP32 row kernel, store BF16.
            std::vector<float> xf(cols), of(rows * cols);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) xf[c] = __bfloat162float(x_h[r * cols + c]);
                const float* wrow = nullptr;
                std::vector<float> wf;
                if (weight) {
                    if (weight->dtype() != DataType::BF16) {
                        throw std::runtime_error(
                            std::string("rms_norm_kernel_dst (CPU): BF16 x needs a BF16 weight, got '") +
                            dtype_name(weight->dtype()) + "'.");
                    }
                    wf.assign(cols, 0.0f);
                    for (int c = 0; c < cols; ++c) wf[c] = __bfloat162float(w_h[c]);
                    wrow = wf.data();
                }
                forge::cpu::rms_norm_row_f32(xf.data(), wrow, of.data() + r * cols, cols, eps);
            }
            for (size_t k = 0; k < of.size(); ++k) o_h[k] = __float2bfloat16(of[k]);
        }
        return;
    }
    // 6-B-1: remaining non-FP16/BF16 dtypes (and BF16 for non-retired ops) --
    // reject BF16 before the FP16 branch or any float* cast below. (RMS_NORM no
    // longer reaches this line with BF16; the branch above returns first.)
    require_no_bf16("rms_norm_kernel_dst", OpType::RMS_NORM, x->dtype());
    if (weight) {
        require_no_bf16("rms_norm_kernel_dst/weight", OpType::RMS_NORM, weight->dtype());
    }
    // 6-C-2: native FP8 dst branch -- FP8 x + FP8 weight (or unweighted) route
    // to the real FP8 kernel (RMS_NORM retired from the op_supports_fp8
    // ledger). Mixed FP8/FP32 falls through to require_no_fp8 below and fails
    // loudly (no silent re-cast of the weight as uint8_t).
    if (x->dtype() == DataType::FP8_E4M3 &&
        (!weight || weight->dtype() == DataType::FP8_E4M3)) {
        const uint8_t* x_p = static_cast<const uint8_t*>(x->data());
        uint8_t* o_p = static_cast<uint8_t*>(dst->data());
        if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            if (weight) {
                cuda::launch_rms_norm_fp8(x_p, static_cast<const uint8_t*>(weight->data()),
                                          o_p, rows, cols, eps, stream);
            } else {
                cuda::launch_rms_norm_unweighted_fp8(x_p, o_p, rows, cols, eps, stream);
            }
#endif
        } else {
            // CPU: FP32-only row kernel -- decode FP8, FP32 row norm, encode FP8.
            std::vector<float> xf(cols), of(static_cast<size_t>(rows) * cols);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    xf[c] = fp8_e4m3_to_fp32(x_p[static_cast<size_t>(r) * cols + c]);
                }
                const float* wrow = nullptr;
                std::vector<float> wf;
                if (weight) {
                    const uint8_t* w_p = static_cast<const uint8_t*>(weight->data());
                    wf.assign(cols, 0.0f);
                    for (int c = 0; c < cols; ++c) wf[c] = fp8_e4m3_to_fp32(w_p[c]);
                    wrow = wf.data();
                }
                forge::cpu::rms_norm_row_f32(xf.data(), wrow,
                                             of.data() + static_cast<size_t>(r) * cols, cols,
                                             eps);
            }
            for (size_t k = 0; k < of.size(); ++k) o_p[k] = fp32_to_fp8_e4m3(of[k]);
        }
        return;
    }
    // 6-C-1: FP8 gate -- mirrors ADD/MUL (covers the bare CPU fallback below).
    require_no_fp8("rms_norm_kernel_dst", OpType::RMS_NORM, x->dtype());
    if (weight) {
        require_no_fp8("rms_norm_kernel_dst/weight", OpType::RMS_NORM, weight->dtype());
    }
    if (x->dtype() == DataType::FP16) {
        const __half* x_h = reinterpret_cast<const __half*>(x->data());
        const __half* w_h = weight ? reinterpret_cast<const __half*>(weight->data()) : nullptr;
        __half* o_h = reinterpret_cast<__half*>(dst->data());
        if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            if (weight) {
                cuda::launch_rms_norm_fp16(x_h, w_h, o_h, rows, cols, eps, stream);
            } else {
                std::vector<__half> ones(cols, __half(1.0f));
                cuda::launch_rms_norm_fp16(x_h, ones.data(), o_h, rows, cols, eps, stream);
            }
#endif
        } else {
            // CPU path is FP32-only by design: upcast locally, run the FP32 row kernel, store FP16.
            std::vector<float> xf(cols), of(rows * cols);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) xf[c] = __half2float(x_h[r * cols + c]);
                const float* wrow = nullptr;
                std::vector<float> wf;
                if (weight) {
                    wf.assign(cols, 0.0f);
                    for (int c = 0; c < cols; ++c) wf[c] = __half2float(w_h[c]);
                    wrow = wf.data();
                }
                forge::cpu::rms_norm_row_f32(xf.data(), wrow, of.data() + r * cols, cols, eps);
            }
            for (size_t k = 0; k < of.size(); ++k) o_h[k] = __float2half(of[k]);
        }
        return;
    }

    if (x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        require_dtype_supported("rms_norm_kernel_dst", OpType::RMS_NORM, x->dtype());
        if (weight) {
            require_dtype_supported("rms_norm_kernel_dst/weight", OpType::RMS_NORM,
                                    weight->dtype());
            cuda::launch_rms_norm(static_cast<const float*>(x->data()),
                                  static_cast<const float*>(weight->data()),
                                  static_cast<float*>(dst->data()), rows, cols, eps, stream);
        } else {
            std::vector<float> ones(cols, 1.0f);
            cuda::launch_rms_norm(static_cast<const float*>(x->data()), ones.data(),
                                  static_cast<float*>(dst->data()), rows, cols, eps, stream);
        }
#endif
    } else {
        const float* x_data = static_cast<const float*>(x->data());
        const float* w_data = weight ? static_cast<const float*>(weight->data()) : nullptr;
        float* o_data = static_cast<float*>(dst->data());

        cpu::parallel_for(rows, [&](int r) {
            const float* x_row = x_data + r * cols;
            float* o_row = o_data + r * cols;
            forge::cpu::rms_norm_row_f32(x_row, w_data, o_row, cols, eps);
            });
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
// 6-A-6.4: stream forwarded. ROPE stays FP32-only (no FP16 branch — see 6-A-6.3 gate).
void rope_kernel_dst(const std::vector<TensorPtr>& inputs, TensorPtr dst, const int32_t* params,
                     cudaStream_t stream) {
    auto x = inputs[0];
    if (!params)
        return;

    // 6-B-1 (P1): ROPE is FP32-only with no lowp branch -- any non-FP32 dtype
    // (including BF16) must fail loudly instead of being read as float below.
    // ROPE stays out of op_supports_lowp (not whitelisted); the gate + BF16
    // check together give the same loud behavior as the audited ops.
    require_dtype_supported("rope_kernel_dst", OpType::ROPE, x->dtype());
    require_no_bf16("rope_kernel_dst", OpType::ROPE, x->dtype());
    if (x->dtype() != DataType::FP32) {
        throw std::runtime_error(
            std::string("rope_kernel_dst: dtype '") + dtype_name(x->dtype()) +
            "' has no native kernel. ROPE is FP32-only; BF16 arrives with 6-B.");
    }

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
                                  seq_len, start_pos, rope_theta, stream);
        } else {
            cuda::launch_rope_gqa(nullptr, static_cast<const float*>(x->data()), nullptr,
                                  static_cast<float*>(dst->data()), 0, num_h, head_dim, seq_len,
                                  start_pos, rope_theta, stream);
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
// 6-A-6.4: stream plumbed through to ops::matmul_transB (native FP16 cuBLAS
// honors it). No dst injection here by design (see 6-A-6.4 scope: A).
TensorPtr mul_mat_transb_kernel(const std::vector<TensorPtr>& inputs, const int32_t*,
                                cudaStream_t stream) {
    TensorPtr bias = (inputs.size() > 2) ? inputs[2] : nullptr;
    return ops::matmul_transB(inputs[0], inputs[1], bias, stream);
}

// ---- FLASH_ATTN_GQA (legacy) ----
// 6-A-7: native FP16 branch added (all-FP16 QKV, FP32 math, FP16 out, stream
// honored from day one). FP32 path unchanged. Mixed dtypes / non-GQA CUDA /
// CPU: unchanged behavior. Unsupported FP16 configs throw loudly — never
// reinterpreted as float.
TensorPtr flash_attn_gqa_kernel(const std::vector<TensorPtr>& inputs, const int32_t* params,
                                cudaStream_t stream) {
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

    // 6-B-6 native BF16-Q decode: Q BF16, K/V FP32 (authoritative KV store),
    // GQA, seq_len_q == 1. 6-B-6.7 extends K/V to BF16 (authoritative BF16 KV):
    // the K/V dtype selects the kernel (FP32 -> bf16q launch, BF16 -> bf16
    // launch); mixed or other K/V dtypes keep loud rejection. 6-B-7.0 unlocks
    // BF16 prefill (all-BF16 QKV vectorized kernel) for the E2E closed loop.
    // Placed before the FP32 GQA branch so BF16 Q never reaches the
    // require_no_bf16 rejects below (retired call site).
    if (dev == DeviceType::CUDA && q->dtype() == DataType::BF16) {
        require_dtype_supported("flash_attn_gqa_kernel/q", OpType::FLASH_ATTN_GQA, q->dtype());
        require_dtype_supported("flash_attn_gqa_kernel/k", OpType::FLASH_ATTN_GQA, k->dtype());
        require_dtype_supported("flash_attn_gqa_kernel/v", OpType::FLASH_ATTN_GQA, v->dtype());
        const bool kv_fp32 = (k->dtype() == DataType::FP32 && v->dtype() == DataType::FP32);
        const bool kv_bf16 = (k->dtype() == DataType::BF16 && v->dtype() == DataType::BF16);
        if (!kv_fp32 && !kv_bf16) {
            throw std::runtime_error(
                "flash_attn_gqa_kernel: native BF16 decode needs all-FP32 or all-BF16 K/V, got '" +
                dtype_name(k->dtype()) + "' / '" + dtype_name(v->dtype()) +
                "' (mixed K/V fail loudly).");
        }
        if (num_kv_heads >= num_heads) {
            throw std::runtime_error(
                "flash_attn_gqa_kernel: BF16 non-GQA (MHA) has no native kernel yet (6-B-6: "
                "decode-first GQA only).");
        }
        if (head_dim != 16 && head_dim != 32 && head_dim != 64 && head_dim != 96 &&
            head_dim != 128) {
            throw std::runtime_error("flash_attn_gqa_kernel: BF16 attention unsupported head_dim=" +
                                     std::to_string(head_dim) + " (6-B-6: 16/32/64/96/128 only).");
        }
        auto attn_out = std::make_shared<Tensor>(
            DataType::BF16, std::vector<int64_t>{seq_len_q, num_heads * head_dim},
            DeviceType::CUDA);
#ifdef USE_CUDA
        if (seq_len_q != 1) {
            // 6-B-7.0: native BF16 prefill (E2E unlock). Prefill requires
            // all-BF16 QKV (no FP32-KV prefill kernel exists); the vectorized
            // kernel shares the decode causal convention.
            if (!kv_bf16) {
                throw std::runtime_error(
                    "flash_attn_gqa_kernel: native BF16 prefill needs all-BF16 QKV "
                    "(FP32-KV prefill has no kernel; decode accepts FP32 K/V).");
            }
            cuda::launch_flash_attention_gqa_prefill_bf16(
                reinterpret_cast<const __nv_bfloat16*>(q->data()),
                reinterpret_cast<const __nv_bfloat16*>(k->data()),
                reinterpret_cast<const __nv_bfloat16*>(v->data()),
                reinterpret_cast<__nv_bfloat16*>(attn_out->data()), seq_len_q, total_len,
                num_heads, num_kv_heads, head_dim, nullptr, causal, stream);
        } else if (kv_bf16) {
            cuda::launch_flash_attention_gqa_decode_bf16(
                reinterpret_cast<const __nv_bfloat16*>(q->data()),
                reinterpret_cast<const __nv_bfloat16*>(k->data()),
                reinterpret_cast<const __nv_bfloat16*>(v->data()),
                reinterpret_cast<__nv_bfloat16*>(attn_out->data()), total_len, num_heads,
                num_kv_heads, head_dim, nullptr, stream);
        } else {
            cuda::launch_flash_attention_gqa_decode_bf16q(
                reinterpret_cast<const __nv_bfloat16*>(q->data()),
                static_cast<const float*>(k->data()), static_cast<const float*>(v->data()),
                reinterpret_cast<__nv_bfloat16*>(attn_out->data()), total_len, num_heads,
                num_kv_heads, head_dim, nullptr, stream);
        }
#endif
        return attn_out;
    }

    // 6-A-7 native FP16 decode: Q/K/V all FP16, GQA, seq_len_q == 1.
    // Prefill (seq>1) FP16 lands in 7.2; mixed dtypes keep loud rejection.
    if (dev == DeviceType::CUDA && q->dtype() == DataType::FP16) {
        require_dtype_supported("flash_attn_gqa_kernel/q", OpType::FLASH_ATTN_GQA, q->dtype());
        require_dtype_supported("flash_attn_gqa_kernel/k", OpType::FLASH_ATTN_GQA, k->dtype());
        require_dtype_supported("flash_attn_gqa_kernel/v", OpType::FLASH_ATTN_GQA, v->dtype());
        if (k->dtype() != DataType::FP16 || v->dtype() != DataType::FP16) {
            throw std::runtime_error(
                "flash_attn_gqa_kernel: native FP16 attention needs all-FP16 QKV.");
        }
        if (num_kv_heads >= num_heads) {
            throw std::runtime_error(
                "flash_attn_gqa_kernel: FP16 non-GQA (MHA) has no native kernel yet (6-A-7: "
                "decode-first GQA only).");
        }
        if (head_dim != 16 && head_dim != 32 && head_dim != 64 && head_dim != 96 &&
            head_dim != 128) {
            throw std::runtime_error("flash_attn_gqa_kernel: FP16 attention unsupported head_dim=" +
                                     std::to_string(head_dim) + " (6-A-7: 16/32/64/96/128 only).");
        }
        auto attn_out = std::make_shared<Tensor>(
            DataType::FP16, std::vector<int64_t>{seq_len_q, num_heads * head_dim},
            DeviceType::CUDA);
#ifdef USE_CUDA
        if (seq_len_q == 1) {
            cuda::launch_flash_attention_gqa_decode_fp16(
                reinterpret_cast<const __half*>(q->data()),
                reinterpret_cast<const __half*>(k->data()),
                reinterpret_cast<const __half*>(v->data()),
                reinterpret_cast<__half*>(attn_out->data()), total_len, num_heads, num_kv_heads,
                head_dim, nullptr, stream);
        } else {
            // 6-A-7.2: vectorized prefill (causal convention matches the FP32
            // vectorized kernel: q_pos = kv_len - q_len + qi).
            cuda::launch_flash_attention_gqa_prefill_fp16(
                reinterpret_cast<const __half*>(q->data()),
                reinterpret_cast<const __half*>(k->data()),
                reinterpret_cast<const __half*>(v->data()),
                reinterpret_cast<__half*>(attn_out->data()), seq_len_q, total_len, num_heads,
                num_kv_heads, head_dim, nullptr, causal, stream);
        }
#endif
        return attn_out;
    }

    if (dev == DeviceType::CUDA && num_kv_heads < num_heads) {
        // 6-B-1: the FP32 GQA branch below reads Q/K/V as float* -- BF16 has no
        // kernel, so reject loudly here (the FP16 branch above is guarded by
        // q->dtype() == FP16; BF16 QKV would otherwise fall through silently).
        require_no_bf16("flash_attn_gqa_kernel/q", OpType::FLASH_ATTN_GQA, q->dtype());
        require_no_bf16("flash_attn_gqa_kernel/k", OpType::FLASH_ATTN_GQA, k->dtype());
        require_no_bf16("flash_attn_gqa_kernel/v", OpType::FLASH_ATTN_GQA, v->dtype());
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
        // 6-B-1: non-GQA FP32 fallback -- BF16 has no kernel, reject loudly.
        require_no_bf16("flash_attn_gqa_kernel/q", OpType::FLASH_ATTN_GQA, q->dtype());
        require_no_bf16("flash_attn_gqa_kernel/k", OpType::FLASH_ATTN_GQA, k->dtype());
        require_no_bf16("flash_attn_gqa_kernel/v", OpType::FLASH_ATTN_GQA, v->dtype());
        return ops::scaled_dot_product_attention_2d(q, k, v, seq_len_q, total_len, num_heads,
                                                    head_dim, nullptr, causal);
    } else {
        // 6-B-1: CPU path reads float* below -- BF16 has no kernel, reject loudly.
        require_no_bf16("flash_attn_gqa_kernel/q", OpType::FLASH_ATTN_GQA, q->dtype());
        require_no_bf16("flash_attn_gqa_kernel/k", OpType::FLASH_ATTN_GQA, k->dtype());
        require_no_bf16("flash_attn_gqa_kernel/v", OpType::FLASH_ATTN_GQA, v->dtype());
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