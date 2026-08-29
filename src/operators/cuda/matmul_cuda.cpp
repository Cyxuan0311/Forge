// Phase 4: CUDA dispatch for the matmul family.
// Moved verbatim out of src/operators/cpu/matmul.cpp (commit 1: pure move, behaviour unchanged).

#include "matmul_cuda.h"

#ifdef USE_CUDA

#    include <algorithm>
#    include <cstring>
#    include <stdexcept>

#    include <cuda_runtime.h>

#    include "forge/cuda_kernels.h"
#    include "forge/cuda_mem_pool.h"
#    include "forge/logger.h"
#    include "forge/operator_elementwise.h"
#    include "forge/operator_matmul.h"
#    include "forge/quant_traits.h"

namespace forge {
namespace ops {

namespace {

using DequantMatrixFn = void (*)(const void*, float*, int, int, cudaStream_t);

// Device-side full-matrix dequantization kernels. nullptr means the dtype has no
// CUDA dequant kernel.
DequantMatrixFn device_dequant_matrix_fn(DataType dt) {
    switch (dt) {
    case DataType::Q4_0:
        return cuda::launch_dequant_q4_0_matrix;
    case DataType::Q4_1:
        return cuda::launch_dequant_q4_1_matrix;
    case DataType::Q4_K:
        return cuda::launch_dequant_q4_k_matrix;
    case DataType::Q5_0:
        return cuda::launch_dequant_q5_0_matrix;
    case DataType::Q5_1:
        return cuda::launch_dequant_q5_1_matrix;
    case DataType::Q2_K:
        return cuda::launch_dequant_q2_k_matrix;
    case DataType::Q3_K:
        return cuda::launch_dequant_q3_k_matrix;
    case DataType::Q5_K:
        return cuda::launch_dequant_q5_k_matrix;
    case DataType::Q6_K:
        return cuda::launch_dequant_q6_k_matrix;
    case DataType::Q8_0:
        // num_rows/row_len signature is identical to (N, K).
        return cuda::launch_dequant_q8_0_matrix;
    case DataType::IQ2_S:
        return cuda::launch_dequant_iq2_s_matrix;
    case DataType::IQ2_XS:
        return cuda::launch_dequant_iq2_xs_matrix;
    case DataType::IQ2_XXS:
        return cuda::launch_dequant_iq2_xxs_matrix;
    case DataType::IQ3_S:
        return cuda::launch_dequant_iq3_s_matrix;
    case DataType::IQ4_NL:
        return cuda::launch_dequant_iq4_nl_matrix;
    default:
        return nullptr;
    }
}

cuda::GemvFn device_gemv_fn(DataType dt) {
    int idx = static_cast<int>(dt);
    if (idx < 0 || idx >= 21)
        return nullptr;
    return cuda::gemv_dispatch[idx];
}

cuda::GemvBatchFn device_gemv_batch_fn(DataType dt) {
    int idx = static_cast<int>(dt);
    if (idx < 0 || idx >= 21)
        return nullptr;
    return cuda::gemv_batch_dispatch[idx];
}

cuda::MmqFn device_mmq_fn(DataType dt) {
    int idx = static_cast<int>(dt);
    if (idx < 0 || idx >= 20)
        return nullptr;
    return cuda::mmq_dispatch[idx];
}

// Phase 4 hard rule: when CUDA cannot handle the request, report a capability failure
// with the reason. Never silently fall back to the host (no D2H/H2D staging).
[[noreturn]] void capability_failure(const char* op, DataType dt, int M, int K, int N) {
    std::string msg = std::string("CUDA capability failure in ") + op + ": dtype=" +
                      dtype_name(dt) + " M=" + std::to_string(M) + " K=" + std::to_string(K) +
                      " N=" + std::to_string(N) +
                      " has no CUDA kernel (no GEMV, no batched GEMV, no device dequant). "
                      "Cross-device fallback to CPU is disabled.";
    LOG_ERROR(msg);
    throw std::runtime_error(msg);
}

// RAII owner for a device scratch buffer drawn from the caching pool.
// Returns the block to the pool (instead of cudaFree) when the matmul call
// is done; kernels run ordered on the same stream, so the buffer is never
// recycled while the async GEMV that reads it is still in flight.
struct DeviceScratch {
    void* ptr = nullptr;
    ~DeviceScratch() {
        if (ptr)
            forge::cuda_mem::deallocate(ptr);
    }
    DeviceScratch() = default;
    explicit DeviceScratch(void* p) : ptr(p) {}
    DeviceScratch(const DeviceScratch&) = delete;
    DeviceScratch& operator=(const DeviceScratch&) = delete;
    DeviceScratch(DeviceScratch&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    float* as_float() { return static_cast<float*>(ptr); }
};

// Dequantize B on device into a pooled device scratch buffer.
DeviceScratch dequant_b_on_device(DequantMatrixFn fn, const TensorPtr& b, int N, int K) {
    size_t fp32_bytes = (size_t)N * K * sizeof(float);
    DeviceScratch scr(forge::cuda_mem::allocate(fp32_bytes));
    if (!scr.ptr)
        throw std::runtime_error("cuda_mem::allocate failed in dequant_b_on_device");
    fn(b->data(), scr.as_float(), N, K, 0);
    return scr;
}
}  // namespace

void cuda_apply_bias(const TensorPtr& out, const TensorPtr& bias, int M, int N) {
    const float* bias_data = static_cast<const float*>(bias->data());
    float* o_data = static_cast<float*>(out->data());

    TensorPtr bias_cuda;
    if (bias->device() == DeviceType::CPU) {
        bias_cuda = std::make_shared<Tensor>(bias->dtype(), bias->shape(), DeviceType::CUDA);
        bias_cuda->copy_from(*bias);
        bias_data = static_cast<const float*>(bias_cuda->data());
    }

    if (bias->ndim() == 1) {
        int bias_size = static_cast<int>(bias->shape()[0]);
        for (int m = 0; m < M; ++m) {
            cuda::launch_add_bias(o_data + m * N, bias_data, o_data + m * N, bias_size);
        }
    } else {
        int total = static_cast<int>(out->numel());
        cuda::launch_add_bias(o_data, bias_data, o_data, total);
    }
}

TensorPtr cuda_dequantize_matrix(const TensorPtr& b, int rows, int cols) {
    auto dequant_matrix = device_dequant_matrix_fn(b->dtype());
    if (!dequant_matrix)
        capability_failure("dequantize_matrix", b->dtype(), 0, cols, rows);
    auto b_fp32 = std::make_shared<Tensor>(DataType::FP32, b->shape(), DeviceType::CUDA);
    dequant_matrix(b->data(), static_cast<float*>(b_fp32->data()), rows, cols, 0);
    return b_fp32;
}

void cuda_matmul(const TensorPtr& a, const TensorPtr& b_fp32, const TensorPtr& out, int M, int K,
                 int N) {
    for (int m = 0; m < M; ++m) {
        cuda::launch_gemv(static_cast<const float*>(a->data()) + m * K,
                          static_cast<const float*>(b_fp32->data()),
                          static_cast<float*>(out->data()) + m * N, K, N);
    }
}

void cuda_matmul_transB(const TensorPtr& a, const TensorPtr& b, const TensorPtr& out, int M, int K,
                        int N) {
    const float* a_data = static_cast<const float*>(a->data());
    float* o_data = static_cast<float*>(out->data());
    const bool quantized = is_quantized_type(b->dtype());

    if (M > 1) {
        // Multi-row batch (prefill / speculative verification): prefer the
        // MMQ tiled kernel — weight scales/mins decode once per tile and are
        // reused across all M rows (dp4a int8 dot), giving sub-linear cost in
        // M. The batched-GEMV fallback re-reads every weight row once per
        // activation row (cost proportional to M).
        //
        // Whitelist: dtypes whose MMQ kernels pass the reference suite
        // (tests/test_mmq_reference.cpp). Q3_K has a known residual defect
        // and stays on the verified batched-GEMV path.
        // All six quantized dtypes now pass tests/test_mmq_reference.cpp.
        auto mmq = device_mmq_fn(b->dtype());
        if (mmq && quantized) {
            mmq(a_data, b->data(), o_data, M, K, N, 0);
        } else {
            auto gemv_batch = device_gemv_batch_fn(b->dtype());
            if (gemv_batch) {
                gemv_batch(a_data, b->data(), o_data, M, K, N, 0);
            } else if (quantized) {
                // No batched GEMV kernel: dequantize on device, then cuBLAS
                auto dequant_matrix = device_dequant_matrix_fn(b->dtype());
                if (!dequant_matrix)
                    capability_failure("matmul_transB", b->dtype(), M, K, N);
                auto b_scratch = dequant_b_on_device(dequant_matrix, b, N, K);
                cuda::launch_cublas_sgemm(a_data, b_scratch.as_float(), o_data, M, K, N, true);
            } else {
                cuda::launch_cublas_sgemm(a_data, static_cast<const float*>(b->data()), o_data,
                                          M, K, N, true);
            }
        }
    } else {
        // M == 1: single GEMV — use dispatch table for supported types
        auto gemv = device_gemv_fn(b->dtype());
        if (gemv) {
            gemv(a_data, b->data(), o_data, K, N, 0);
        } else if (quantized) {
            // No dedicated M=1 kernel: fall back to the batched GEMV (on-the-fly
            // dequant) with M=1 before dequantizing the whole weight to FP32.
            auto gemv_batch = device_gemv_batch_fn(b->dtype());
            if (gemv_batch) {
                gemv_batch(a_data, b->data(), o_data, 1, K, N, 0);
            } else {
                auto dequant_matrix = device_dequant_matrix_fn(b->dtype());
                if (!dequant_matrix)
                    capability_failure("matmul_transB", b->dtype(), M, K, N);
                auto b_scratch = dequant_b_on_device(dequant_matrix, b, N, K);
                cuda::launch_gemv_transB(a_data, b_scratch.as_float(), o_data, K, N);
            }
        } else {
            // FP32: use cuBLAS for large N (output_proj), custom GEMV for small N
            if (N > 4096) {
                cuda::launch_cublas_sgemm(a_data, static_cast<const float*>(b->data()), o_data, M, K,
                                          N, true);
            } else {
                cuda::launch_gemv_transB(a_data, static_cast<const float*>(b->data()), o_data, K, N);
            }
        }
    }
}

void cuda_matmul_transB_dual(const TensorPtr& a, const TensorPtr& b1, const TensorPtr& b2,
                             const TensorPtr& out, int M, int K, int N1, int N2) {
    (void)K;
    const int N = N1 + N2;
    auto out1 = ops::matmul_transB(a, b1);
    auto out2 = ops::matmul_transB(a, b2);

    float* o_data = static_cast<float*>(out->data());
    const float* o1_data = static_cast<const float*>(out1->data());
    const float* o2_data = static_cast<const float*>(out2->data());
    for (int m = 0; m < M; ++m) {
        cudaMemcpyAsync(o_data + m * N, o1_data + m * N1, N1 * sizeof(float),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(o_data + m * N + N1, o2_data + m * N2, N2 * sizeof(float),
                        cudaMemcpyDeviceToDevice);
    }
}

TensorPtr cuda_ffn_up_fused(const TensorPtr& input, const TensorPtr& w1, const TensorPtr& w3,
                            const TensorPtr& out, int M, int K, int intermediate_dim) {

    auto separate_path = [&]() {
        auto gate = ops::matmul_transB(input, w1);
        auto up = ops::matmul_transB(input, w3);
        return ops::silu_multiply(gate, up);
    };

    if (w1->dtype() == DataType::Q4_0 && w3->dtype() == DataType::Q4_0) {
        if (M == 1) {
            // Decode: single-token fused GEMV kernel (Q8_1 + dp4a)
            cuda::launch_ffn_up_fused_q4_0_q8_1(static_cast<const float*>(input->data()),
                                                w1->data(), w3->data(),
                                                static_cast<float*>(out->data()), K,
                                                intermediate_dim);
        } else {
            // M > 1: MMQ-backed matmuls (sub-linear in M) + silu_multiply.
            // The old fused batched-GEMV re-read every weight row per
            // activation row, so verification batches scaled linearly with M.
            return separate_path();
        }
    } else if (w1->dtype() == DataType::Q4_K && w3->dtype() == DataType::Q4_K) {
        if (M == 1) {
            // Decode: single-token fused GEMV kernel with Q8_1 + dp4a acceleration
            cuda::launch_ffn_up_fused_q4_k_q8_1(static_cast<const float*>(input->data()),
                                                 w1->data(), w3->data(),
                                                 static_cast<float*>(out->data()), K,
                                                 intermediate_dim);
        } else {
            // M > 1: MMQ-backed matmuls (sub-linear in M) + silu_multiply
            return separate_path();
        }
    } else if (w1->dtype() == DataType::Q5_K && w3->dtype() == DataType::Q5_K) {
        if (M == 1) {
            cuda::launch_ffn_up_fused_q5_k(static_cast<const float*>(input->data()), w1->data(),
                                           w3->data(), static_cast<float*>(out->data()), K,
                                           intermediate_dim);
        } else {
            return separate_path();
        }
    } else if (w1->dtype() == DataType::IQ4_XS && w3->dtype() == DataType::IQ4_XS) {
        if (M == 1) {
            // Decode: IQ4_XS gate + up fused with shared Q8_1 quantization
            cuda::launch_ffn_up_fused_iq4_xs_q8_1(static_cast<const float*>(input->data()),
                                                 w1->data(), w3->data(),
                                                 static_cast<float*>(out->data()), K,
                                                 intermediate_dim);
        } else {
            return separate_path();
        }
    } else if (w1->dtype() == DataType::Q3_K && w3->dtype() == DataType::Q3_K) {
        if (M == 1) {
            // Decode: Q3_K gate + Q3_K up fused with shared Q8_1 quantization
            cuda::launch_ffn_up_fused_q3k_q3k(static_cast<const float*>(input->data()), w1->data(),
                                               w3->data(), static_cast<float*>(out->data()), K,
                                               intermediate_dim);
        } else {
            return separate_path();
        }
    } else if (w1->dtype() == DataType::Q3_K && w3->dtype() == DataType::Q4_K) {
        if (M == 1) {
            // Decode: Q3_K gate + Q4_K up fused with shared Q8_1 quantization
            cuda::launch_ffn_up_fused_q3k_q4k(static_cast<const float*>(input->data()), w1->data(),
                                               w3->data(), static_cast<float*>(out->data()), K,
                                               intermediate_dim);
        } else {
            return separate_path();
        }
    } else if (w1->dtype() == DataType::Q2_K && w3->dtype() == DataType::Q2_K) {
        if (M == 1) {
            // Decode: Q2_K gate + Q2_K up fused with shared Q8_1 quantization
            cuda::launch_ffn_up_fused_q2k_q2k(static_cast<const float*>(input->data()), w1->data(),
                                               w3->data(), static_cast<float*>(out->data()), K,
                                               intermediate_dim);
        } else {
            return separate_path();
        }
    } else {
        return separate_path();
    }

    return out;
}

}  // namespace ops
}  // namespace forge

#endif  // USE_CUDA
