#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "simd.h"
#include "forge/logger.h"
#include "forge/operator_elementwise.h"
#include "forge/operator_matmul.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>

#    include "../cuda/matmul_cuda.h"
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

#if FORGE_USE_OPENBLAS
#    include <cblas.h>
#endif

#include "common/dequant.h"
#include "common/quant_helpers.h"
#include "common/quant_tables.h"

namespace forge {
namespace ops {

// Re-quantize Q8_0 weight to Q4_0 (dequantize → requantize, ~1-2s for 7B output weight)
TensorPtr requantize_q8_0_to_q4_0(const TensorPtr& q8_weight) {
    if (!q8_weight || q8_weight->dtype() != DataType::Q8_0)
        return nullptr;
    auto fp32 = dequantize_weight(q8_weight);
    return quantize_q4_0_weight(fp32);
}

TensorPtr dequantize_weight(const TensorPtr& weight) {
    if (!weight || !is_quantized_type(weight->dtype()))
        return weight;
    auto dequant_fn = get_dequant_row_fn(weight->dtype());
    if (!dequant_fn)
        return weight;

    int N = static_cast<int>(weight->shape()[0]);
    int K = static_cast<int>(weight->shape()[1]);
    auto fp32_weight = std::make_shared<Tensor>(DataType::FP32, weight->shape(), weight->device());
    const uint8_t* q_data = static_cast<const uint8_t*>(weight->data());
    float* out = static_cast<float*>(fp32_weight->data());

#pragma omp parallel
    {
        std::vector<float> row_buf(K);
#pragma omp for schedule(dynamic, 64)
        for (int n = 0; n < N; ++n) {
            dequant_fn(q_data, row_buf.data(), K, n);
            std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
        }
    }
    return fp32_weight;
}

// Scalar fp32 → fp16 conversion (truncation; matches pre-arch-split reference)
static inline uint16_t fp32_to_fp16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (x >> 13) & 0x3FFU;

    if (exponent <= 0) {
        // Zero or subnormal — round to zero for simplicity
        return static_cast<uint16_t>(sign);
    } else if (exponent >= 0x1F) {
        // Overflow to infinity
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

// Type-specific dequantize: Q4_0 weight tensor → FP32
TensorPtr dequantize_q4_0_weight(const TensorPtr& q_weight) {
    if (!q_weight || q_weight->dtype() != DataType::Q4_0) return nullptr;
    int N = static_cast<int>(q_weight->shape()[0]);
    int K = static_cast<int>(q_weight->shape()[1]);
    auto fp32 = std::make_shared<Tensor>(DataType::FP32, q_weight->shape(), q_weight->device());
    const uint8_t* q_data = static_cast<const uint8_t*>(q_weight->data());
    float* out = static_cast<float*>(fp32->data());
#pragma omp parallel
    {
        std::vector<float> row_buf(K);
#pragma omp for schedule(dynamic, 64)
        for (int n = 0; n < N; ++n) {
            dequantize_q4_0_row(q_data, row_buf.data(), K, n);
            std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
        }
    }
    return fp32;
}

// Type-specific dequantize: Q4_1 weight tensor → FP32
TensorPtr dequantize_q4_1_weight(const TensorPtr& q_weight) {
    if (!q_weight || q_weight->dtype() != DataType::Q4_1) return nullptr;
    int N = static_cast<int>(q_weight->shape()[0]);
    int K = static_cast<int>(q_weight->shape()[1]);
    auto fp32 = std::make_shared<Tensor>(DataType::FP32, q_weight->shape(), q_weight->device());
    const uint8_t* q_data = static_cast<const uint8_t*>(q_weight->data());
    float* out = static_cast<float*>(fp32->data());
#pragma omp parallel
    {
        std::vector<float> row_buf(K);
#pragma omp for schedule(dynamic, 64)
        for (int n = 0; n < N; ++n) {
            dequantize_q4_1_row(q_data, row_buf.data(), K, n);
            std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
        }
    }
    return fp32;
}

// Quantize FP32 weight to Q8_0 format (34 bytes per 32 elements)
TensorPtr quantize_q8_0_weight(const TensorPtr& fp32_weight) {
    if (!fp32_weight)
        return nullptr;

    const auto& shape = fp32_weight->shape();
    int N = static_cast<int>(shape[0]);
    int K = static_cast<int>(shape[1]);

    constexpr int BLOCK_EL = 32;
    constexpr int BLOCK_BYTES = 34;
    int blocks_per_row = (K + BLOCK_EL - 1) / BLOCK_EL;
    size_t row_bytes = static_cast<size_t>(blocks_per_row) * BLOCK_BYTES;

    auto q_weight = std::make_shared<Tensor>(DataType::Q8_0, shape, fp32_weight->device());
    const float* src = static_cast<const float*>(fp32_weight->data());
    uint8_t* dst = static_cast<uint8_t*>(q_weight->data());

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const float* row = src + n * K;
        uint8_t* q_row = dst + static_cast<size_t>(n) * row_bytes;

        for (int bi = 0; bi < blocks_per_row; ++bi) {
            int base = bi * BLOCK_EL;
            int remaining = K - base;
            int nel = remaining < BLOCK_EL ? remaining : BLOCK_EL;

            // Find max absolute value in this block
            float amax = 0.0f;
            for (int j = 0; j < nel; ++j) {
                float av = std::fabs(row[base + j]);
                if (av > amax)
                    amax = av;
            }

            // Compute scale: amax / 127
            float scale = amax / 127.0f;
            float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            // Encode scale as fp16
            uint16_t scale_fp16 = fp32_to_fp16_bits(scale);
            memcpy(q_row + bi * BLOCK_BYTES, &scale_fp16, 2);

            // Quantize values to int8
            int8_t* qs = reinterpret_cast<int8_t*>(q_row + bi * BLOCK_BYTES + 2);
            for (int j = 0; j < nel; ++j) {
                float v = row[base + j] * inv_scale;
                int qv = static_cast<int>(std::roundf(v));
                qs[j] = static_cast<int8_t>(std::max(-128, std::min(127, qv)));
            }
            // Zero-fill remaining bytes in partial block
            for (int j = nel; j < BLOCK_EL; ++j) {
                qs[j] = 0;
            }
        }
    }
    return q_weight;
}

// Quantize FP32 weight to Q4_0 format (18 bytes per 32 elements)
TensorPtr quantize_q4_0_weight(const TensorPtr& fp32_weight) {
    if (!fp32_weight || fp32_weight->dtype() != DataType::FP32) return nullptr;
    int N = static_cast<int>(fp32_weight->shape()[0]);
    int K = static_cast<int>(fp32_weight->shape()[1]);
    constexpr int QK = 32;
    constexpr int BLOCK_BYTES = 18;  // 2 bytes fp16 scale + 16 bytes nibbles
    int blocks_per_row = (K + QK - 1) / QK;
    int64_t row_bytes = blocks_per_row * BLOCK_BYTES;
    auto q4 = std::make_shared<Tensor>(DataType::Q4_0, fp32_weight->shape(), fp32_weight->device());
    const float* src = static_cast<const float*>(fp32_weight->data());
    uint8_t* dst = static_cast<uint8_t*>(q4->data());
#pragma omp parallel for schedule(dynamic, 64)
    for (int n = 0; n < N; ++n) {
        const float* row = src + n * K;
        uint8_t* row_dst = dst + n * row_bytes;
        for (int bi = 0; bi < blocks_per_row; ++bi) {
            int base = bi * QK;
            int n_el = (base + QK <= K) ? QK : (K - base);
            float amax = 0.0f;
            for (int j = 0; j < n_el; ++j) {
                float v = std::abs(row[base + j]);
                if (v > amax) amax = v;
            }
            float d = amax / 7.0f;  // Q4_0 range is [-8, 7]: max representable = 7*d
            float id = d > 0.0f ? 1.0f / d : 0.0f;
            uint16_t d_bits = fp32_to_fp16_bits(d);
            std::memcpy(row_dst + bi * BLOCK_BYTES, &d_bits, 2);
            uint8_t* qs = row_dst + bi * BLOCK_BYTES + 2;
            for (int j = 0; j < 16; ++j) {
                uint8_t lo = 0, hi = 0;
                if (j < n_el) {
                    int q = (int)(row[base + j] * id + (row[base + j] >= 0 ? 0.5f : -0.5f)) + 8;
                    if (q < 0) q = 0;
                    if (q > 15) q = 15;
                    lo = (uint8_t)q;
                }
                if (j + 16 < n_el) {
                    int q = (int)(row[base + 16 + j] * id + (row[base + 16 + j] >= 0 ? 0.5f : -0.5f)) + 8;
                    if (q < 0) q = 0;
                    if (q > 15) q = 15;
                    hi = (uint8_t)q;
                }
                qs[j] = lo | (hi << 4);
            }
        }
    }
    return q4;
}

static void apply_bias(TensorPtr& out, const TensorPtr& bias, int M, int N) {
    if (!bias)
        return;

    if (out->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda_apply_bias(out, bias, M, N);
#endif
        return;
    }

    const float* bias_data = static_cast<const float*>(bias->data());
    float* o_data = static_cast<float*>(out->data());

    if (bias->ndim() == 1) {
        int bias_size = static_cast<int>(bias->shape()[0]);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < bias_size && n < N; ++n) {
                o_data[m * N + n] += bias_data[n];
            }
        }
    } else {
        int total = static_cast<int>(out->numel());
        for (int i = 0; i < total; ++i) {
            o_data[i] += bias_data[i];
        }
    }
}

TensorPtr matmul(const TensorPtr& a, const TensorPtr& b, const TensorPtr& bias) {
    if (a->ndim() != 2 || b->ndim() != 2)
        throw std::runtime_error("matmul expects 2D tensors");

    TensorPtr b_fp32 = b;
    if (is_quantized_type(b->dtype())) {
        int N = static_cast<int>(b->shape()[0]);
        int K = static_cast<int>(b->shape()[1]);
        if (a->device() == DeviceType::CUDA) {
            // Phase 4: dequantize on device — no D2H/H2D staging.
#ifdef USE_CUDA
            b_fp32 = cuda_dequantize_matrix(b, N, K);
#else
            throw std::runtime_error("matmul: CUDA tensor without CUDA support");
#endif
        } else {
            auto dequant_fn = get_dequant_row_fn(b->dtype());
            if (!dequant_fn)
                throw std::runtime_error("Unsupported quantized type in matmul");
            if (b->device() != DeviceType::CPU)
                throw std::runtime_error(
                    "matmul: CPU path requires a host-resident quantized weight "
                    "(cross-device staging is disabled)");
            b_fp32 = std::make_shared<Tensor>(DataType::FP32, b->shape(), DeviceType::CPU);
            const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
            float* out = static_cast<float*>(b_fp32->data());
            std::vector<float> row_buf(K);
            for (int n = 0; n < N; ++n) {
                dequant_fn(q_data, row_buf.data(), K, n);
                std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
            }
        }
    }

    int M = static_cast<int>(a->shape()[0]);
    int K = static_cast<int>(a->shape()[1]);
    int K2 = static_cast<int>(b_fp32->shape()[0]);
    int N = static_cast<int>(b_fp32->shape()[1]);
    if (K != K2)
        throw std::runtime_error("matmul dimension mismatch");

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, a->device());

    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda_matmul(a, b_fp32, out, M, K, N);
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b_fp32->data());
        float* o_data = static_cast<float*>(out->data());
#if FORGE_USE_OPENBLAS
        PERF_SCOPE("matmul/fp32_gemm_openblas");
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a_data, K, b_data, N,
                    0.0f, o_data, N);
#elif defined(USE_AVX2)
        PERF_SCOPE("matmul/fp32_gemm_avx2");
        cpu::gemm_fp32_avx2(a_data, b_data, o_data, M, K, N);
#elif defined(USE_NEON)
        PERF_SCOPE("matmul/fp32_gemm_scalar");
        std::memset(o_data, 0, M * N * sizeof(float));
#    pragma omp parallel for schedule(dynamic) if (M * N > 64)
        for (int m = 0; m < M; ++m) {
            const float* a_row = a_data + m * K;
            float* o_row = o_data + m * N;
            for (int k = 0; k < K; ++k) {
                float a_val = a_row[k];
                const float* b_row = b_data + k * N;
                for (int n = 0; n < N; ++n) {
                    o_row[n] += a_val * b_row[n];
                }
            }
        }
#else
        PERF_SCOPE("matmul/fp32_gemm_scalar");
        std::memset(o_data, 0, M * N * sizeof(float));
#    pragma omp parallel for schedule(dynamic) if (M * N > 64)
        for (int m = 0; m < M; ++m) {
            const float* a_row = a_data + m * K;
            float* o_row = o_data + m * N;
            for (int k = 0; k < K; ++k) {
                float a_val = a_row[k];
                const float* b_row = b_data + k * N;
                for (int n = 0; n < N; ++n) {
                    o_row[n] += a_val * b_row[n];
                }
            }
        }
#endif
    }

    apply_bias(out, bias, M, N);
    return out;
}

// ============================================================================
// Batched MoE matmul: shared input (gate/up) and batched pairs (down)
// Reduces OpenMP fork/join overhead and redundant Q8_K quantization.
// ============================================================================

TensorPtr matmul_transB_shared_input(const TensorPtr& input,
                                      const std::vector<TensorPtr>& weights) {
    if (weights.empty()) return nullptr;

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weights[0]->shape()[0]);
    int n_w = static_cast<int>(weights.size());
    DataType dt = weights[0]->dtype();

    auto out = std::make_shared<Tensor>(DataType::FP32,
        std::vector<int64_t>{(int64_t)n_w, (int64_t)N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());

#ifdef USE_AVX2
    auto dot_fn = get_dot_q8k_fn(dt);
    // Batched fast path requires uniform dtype + shape across all weights.
    // Mixed-precision experts (per-expert dtypes differ) must fall back.
    for (int w = 1; w < n_w && dot_fn; ++w) {
        if (weights[w]->dtype() != dt ||
            static_cast<int>(weights[w]->shape()[0]) != N ||
            static_cast<int>(weights[w]->shape()[1]) != K) {
            dot_fn = nullptr;
        }
    }

    if (M == 1 && dot_fn) {
        PERF_SCOPE("matmul_transB/shared_input_multi_gemv");

        constexpr int QK_K = 256;
        const int nb = (K + QK_K - 1) / QK_K;

        // Quantize input ONCE — reused across all weight matrices
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(static_cast<const float*>(input->data()), q8_buf.data(), K);

        // Collect weight pointers
        std::vector<const uint8_t*> w_ptrs(n_w);
        for (int w = 0; w < n_w; ++w)
            w_ptrs[w] = static_cast<const uint8_t*>(weights[w]->data());

        // Row stride for this dtype
        int block_el = dtype_block_elements(dt);
        int block_bytes = dtype_block_size(dt);
        size_t row_stride = (size_t)((K + block_el - 1) / block_el) * block_bytes;

        // Single OpenMP region: all weight matrices' rows
        int total_N = n_w * N;
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_N; ++idx) {
            int w = idx / N;
            int n = idx % N;
            const uint8_t* row = w_ptrs[w] + (size_t)n * row_stride;
            o_data[w * N + n] = dot_fn(row, q8_buf.data(), nb);
        }
        return out;
    }
#elif defined(USE_NEON) || defined(USE_VSX)
    {
        auto dot_fn = cpu::get_dot_q8k_fn(dt);
        for (int w = 1; w < n_w && dot_fn; ++w) {
            if (weights[w]->dtype() != dt ||
                static_cast<int>(weights[w]->shape()[0]) != N ||
                static_cast<int>(weights[w]->shape()[1]) != K) {
                dot_fn = nullptr;
            }
        }

        if (M == 1 && dot_fn) {
            PERF_SCOPE("matmul_transB/shared_input_multi_gemv_neon");

            constexpr int QK_K = 256;
            const int nb = (K + QK_K - 1) / QK_K;

            std::vector<cpu::block_q8_K> q8_buf(nb);
            cpu::quantize_row_q8_K(static_cast<const float*>(input->data()), q8_buf.data(), K);

            std::vector<const uint8_t*> w_ptrs(n_w);
            for (int w = 0; w < n_w; ++w)
                w_ptrs[w] = static_cast<const uint8_t*>(weights[w]->data());

            int block_el = dtype_block_elements(dt);
            int block_bytes = dtype_block_size(dt);
            size_t row_stride = (size_t)((K + block_el - 1) / block_el) * block_bytes;

            int total_N = n_w * N;
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < total_N; ++idx) {
                int w = idx / N;
                int n = idx % N;
                const uint8_t* row = w_ptrs[w] + (size_t)n * row_stride;
                o_data[w * N + n] = dot_fn(row, q8_buf.data(), nb);
            }
            return out;
        }
    }
#endif

    // Fallback: sequential matmul_transB per weight
    for (int w = 0; w < n_w; ++w) {
        auto sub = ops::matmul_transB(input, weights[w]);
        memcpy(o_data + (size_t)w * N, sub->data(), N * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB_batched_pairs(const TensorPtr& inputs,
                                       const std::vector<TensorPtr>& weights) {
    if (weights.empty()) return nullptr;

    int n_pairs = static_cast<int>(weights.size());
    int K = static_cast<int>(inputs->shape()[1]);
    int N = static_cast<int>(weights[0]->shape()[0]);
    DataType dt = weights[0]->dtype();

    auto out = std::make_shared<Tensor>(DataType::FP32,
        std::vector<int64_t>{(int64_t)n_pairs, (int64_t)N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());

#ifdef USE_AVX2
    auto dot_fn = get_dot_q8k_fn(dt);
    // Batched fast path requires uniform dtype + shape across all weights.
    // Mixed-precision experts (per-expert dtypes differ) must fall back.
    for (int p = 1; p < n_pairs && dot_fn; ++p) {
        if (weights[p]->dtype() != dt ||
            static_cast<int>(weights[p]->shape()[0]) != N ||
            static_cast<int>(weights[p]->shape()[1]) != K) {
            dot_fn = nullptr;
        }
    }

    if (dot_fn) {
        PERF_SCOPE("matmul_transB/batched_pairs_multi_gemv");

        constexpr int QK_K = 256;
        const int nb = (K + QK_K - 1) / QK_K;

        // Quantize each input row to Q8_K (parallelized)
        std::vector<std::vector<cpu::block_q8_K>> q8_bufs(n_pairs,
            std::vector<cpu::block_q8_K>(nb));
        const float* in_data = static_cast<const float*>(inputs->data());
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < n_pairs; ++p)
            cpu::quantize_row_q8_K(in_data + (size_t)p * K, q8_bufs[p].data(), K);

        // Collect weight pointers
        std::vector<const uint8_t*> w_ptrs(n_pairs);
        for (int p = 0; p < n_pairs; ++p)
            w_ptrs[p] = static_cast<const uint8_t*>(weights[p]->data());

        int block_el = dtype_block_elements(dt);
        int block_bytes = dtype_block_size(dt);
        size_t row_stride = (size_t)((K + block_el - 1) / block_el) * block_bytes;

        // Single OpenMP region: all pairs' rows
        int total_N = n_pairs * N;
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_N; ++idx) {
            int p = idx / N;
            int n = idx % N;
            const uint8_t* row = w_ptrs[p] + (size_t)n * row_stride;
            o_data[p * N + n] = dot_fn(row, q8_bufs[p].data(), nb);
        }
        return out;
    }
#elif defined(USE_NEON) || defined(USE_VSX)
    {
        auto dot_fn = cpu::get_dot_q8k_fn(dt);
        for (int p = 1; p < n_pairs && dot_fn; ++p) {
            if (weights[p]->dtype() != dt ||
                static_cast<int>(weights[p]->shape()[0]) != N ||
                static_cast<int>(weights[p]->shape()[1]) != K) {
                dot_fn = nullptr;
            }
        }

        if (dot_fn) {
            PERF_SCOPE("matmul_transB/batched_pairs_multi_gemv_neon");

            constexpr int QK_K = 256;
            const int nb = (K + QK_K - 1) / QK_K;

            std::vector<std::vector<cpu::block_q8_K>> q8_bufs(n_pairs,
                std::vector<cpu::block_q8_K>(nb));
            const float* in_data = static_cast<const float*>(inputs->data());
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < n_pairs; ++p)
                cpu::quantize_row_q8_K(in_data + (size_t)p * K, q8_bufs[p].data(), K);

            std::vector<const uint8_t*> w_ptrs(n_pairs);
            for (int p = 0; p < n_pairs; ++p)
                w_ptrs[p] = static_cast<const uint8_t*>(weights[p]->data());

            int block_el = dtype_block_elements(dt);
            int block_bytes = dtype_block_size(dt);
            size_t row_stride = (size_t)((K + block_el - 1) / block_el) * block_bytes;

            int total_N = n_pairs * N;
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < total_N; ++idx) {
                int p = idx / N;
                int n = idx % N;
                const uint8_t* row = w_ptrs[p] + (size_t)n * row_stride;
                o_data[p * N + n] = dot_fn(row, q8_bufs[p].data(), nb);
            }
            return out;
        }
    }
#endif

    // Fallback: sequential
    for (int p = 0; p < n_pairs; ++p) {
        auto in_row = std::make_shared<Tensor>(DataType::FP32,
            std::vector<int64_t>{1, K}, DeviceType::CPU);
        memcpy(in_row->data(), static_cast<const float*>(inputs->data()) + (size_t)p * K,
               K * sizeof(float));
        auto sub = ops::matmul_transB(in_row, weights[p]);
        memcpy(o_data + (size_t)p * N, sub->data(), N * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB(const TensorPtr& a, const TensorPtr& b, const TensorPtr& bias) {
    if (a->ndim() != 2 || b->ndim() != 2)
        throw std::runtime_error("matmul_transB expects 2D tensors");

    int M = static_cast<int>(a->shape()[0]);
    int K = static_cast<int>(a->shape()[1]);
    int N = static_cast<int>(b->shape()[0]);
    int K2 = static_cast<int>(b->shape()[1]);
    if (K != K2)
        throw std::runtime_error("matmul_transB dimension mismatch: K=" + std::to_string(K) +
                                 " K2=" + std::to_string(K2) + " b_shape=[" +
                                 std::to_string(b->shape()[0]) + "," +
                                 std::to_string(b->shape()[1]) + "]");

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, a->device());

    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda_matmul_transB(a, b, out, M, K, N);
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        float* o_data = static_cast<float*>(out->data());

        if (is_quantized_type(b->dtype())) {
#ifdef USE_AVX2
            if (b->dtype() == DataType::Q4_0) {
                if (M == 1) {
#ifdef USE_AVX512_VNNI
                    if (forge::cpu::cached_has_avx512_vnni()) {
                        PERF_SCOPE("matmul_transB/q4_0_gemm_decode_vnni");
                        cpu::gemm_q4_0_decode_vnni(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                    K, N);
                    } else
#endif
                    {
                        // Try repacked weights for better cache locality
                        const uint8_t* repacked = get_repacked_q4_0(b->data(), K, N);
                        if (repacked) {
                            PERF_SCOPE("matmul_transB/q4_0_gemm_decode_repacked");
                            cpu::gemm_q4_0_decode_repacked_f16c_avx2(a_data, repacked, o_data, K, N);
                        } else {
                            PERF_SCOPE("matmul_transB/q4_0_gemm_decode");
                            cpu::gemm_q4_0_decode_f16c_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                              K, N);
                        }
                    }
                } else {
#ifdef USE_AVX512_VNNI
                    if (forge::cpu::cached_has_avx512_vnni()) {
                        PERF_SCOPE("matmul_transB/q4_0_gemm_batch_vnni");
                        cpu::gemm_q4_0_batch_vnni(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                   M, K, N);
                    } else
#endif
                    {
                        PERF_SCOPE("matmul_transB/q4_0_gemm_batch");
                        cpu::gemm_q4_0_batch_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                    M, K, N);
                    }
                }
            } else if (b->dtype() == DataType::Q8_0) {
                PERF_SCOPE("matmul_transB/q8_0_maddubs_gemv");
                cpu::gemv_q8_0_maddubs_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                     M, K, N);
            } else if (b->dtype() == DataType::Q4_1) {
                PERF_SCOPE("matmul_transB/q4_1_maddubs_gemv");
                cpu::gemv_q4_1_maddubs_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                     M, K, N);
            } else if (b->dtype() == DataType::Q5_0) {
                PERF_SCOPE("matmul_transB/q5_0_maddubs_gemv");
                cpu::gemv_q5_0_maddubs_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                     M, K, N);
            } else if (b->dtype() == DataType::Q5_1) {
                PERF_SCOPE("matmul_transB/q5_1_maddubs_gemv");
                cpu::gemv_q5_1_maddubs_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                     M, K, N);
            } else if (b->dtype() == DataType::Q4_K) {
                PERF_SCOPE("matmul_transB/q4_k_gemm");
                cpu::gemm_q4_K_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q6_K) {
                PERF_SCOPE("matmul_transB/q6_k_gemm");
                cpu::gemm_q6_K_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q2_K) {
                PERF_SCOPE("matmul_transB/q2_k_gemm");
                cpu::gemm_q2_K_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q3_K) {
                PERF_SCOPE("matmul_transB/q3_k_gemm");
                cpu::gemm_q3_K_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q5_K) {
                PERF_SCOPE("matmul_transB/q5_k_gemm");
                cpu::gemm_q5_K_avx2(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::IQ2_S) {
                PERF_SCOPE("matmul_transB/iq2_s_q8k_fused_gemv");
                gemv_iq2_s_q8k_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()),
                                            o_data, M, K, N);
            } else if (b->dtype() == DataType::IQ2_XS) {
                PERF_SCOPE("matmul_transB/iq2_xs_q8k_fused_gemv");
                gemv_iq2_xs_q8k_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()),
                                            o_data, M, K, N);
            } else if (b->dtype() == DataType::IQ3_S) {
                PERF_SCOPE("matmul_transB/iq3_s_q8k_fused_gemv");
                gemv_iq3_s_q8k_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()),
                                           o_data, M, K, N);
            } else if (b->dtype() == DataType::IQ4_NL) {
                PERF_SCOPE("matmul_transB/iq4_nl_q8k_fused_gemv");
                gemv_iq4_nl_q8k_transB_avx2(a_data, static_cast<const uint8_t*>(b->data()),
                                            o_data, M, K, N);
            } else
#elif defined(USE_NEON)
            if (b->dtype() == DataType::Q4_0) {
                if (M == 1) {
                    PERF_SCOPE("matmul_transB/q4_0_gemv_neon");
                    forge::cpu::gemv_q4_0_transB_neon(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                      M, K, N);
                } else {
                    PERF_SCOPE("matmul_transB/dequant+gemv");
                    auto dequant_fn = get_dequant_row_fn(b->dtype());
                    if (!dequant_fn)
                        throw std::runtime_error("Unsupported quantized type in matmul_transB: " + dtype_name(b->dtype()));
                    const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
                    int block_el = dtype_block_elements(b->dtype());
                    int block_bytes = dtype_block_size(b->dtype());
                    int blocks_per_row = (K + block_el - 1) / block_el;
                    size_t row_bytes = (size_t)blocks_per_row * block_bytes;
                    size_t expected_total = (size_t)N * row_bytes;
                    if (expected_total > b->nbytes()) {
                        fprintf(stderr, "[ERROR] matmul_transB: expected_total(%zu) > b->nbytes(%zu)! Buffer overflow risk!\n",
                                expected_total, b->nbytes());
                        fflush(stderr);
                    }
#pragma omp parallel
                    {
                        std::vector<float> row_buf(K);
#pragma omp for schedule(dynamic)
                        for (int n = 0; n < N; ++n) {
                            dequant_fn(q_data, row_buf.data(), K, n);
                            for (int m = 0; m < M; ++m) {
                                const float* a_row = a_data + m * K;
                                float sum = 0.0f;
                                for (int k = 0; k < K; ++k) {
                                    sum += a_row[k] * row_buf[k];
                                }
                                o_data[m * N + n] = sum;
                            }
                        }
                    }
                }
            } else if (b->dtype() == DataType::Q8_0) {
                PERF_SCOPE("matmul_transB/q8_0_gemv_neon");
                forge::cpu::gemv_q8_0_transB_neon(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                  M, K, N);
            } else if (b->dtype() == DataType::Q4_1) {
                PERF_SCOPE("matmul_transB/q4_1_gemv_neon");
                forge::cpu::gemv_q4_1_transB_neon(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                  M, K, N);
            } else if (b->dtype() == DataType::Q5_0) {
                PERF_SCOPE("matmul_transB/q5_0_gemv_neon");
                forge::cpu::gemv_q5_0_transB_neon(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                  M, K, N);
            } else if (b->dtype() == DataType::Q5_1) {
                PERF_SCOPE("matmul_transB/q5_1_gemv_neon");
                forge::cpu::gemv_q5_1_transB_neon(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                                  M, K, N);
            } else if (b->dtype() == DataType::Q4_K) {
                PERF_SCOPE("matmul_transB/q4_k_gemm_neon");
                cpu::gemm_q4_K_neon(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q6_K) {
                PERF_SCOPE("matmul_transB/q6_k_gemm_neon");
                cpu::gemm_q6_K_neon(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q2_K) {
                PERF_SCOPE("matmul_transB/q2_k_gemm_neon");
                cpu::gemm_q2_K_neon(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q3_K) {
                PERF_SCOPE("matmul_transB/q3_k_gemm_neon");
                cpu::gemm_q3_K_neon(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::Q5_K) {
                PERF_SCOPE("matmul_transB/q5_k_gemm_neon");
                cpu::gemm_q5_K_neon(a_data, static_cast<const uint8_t*>(b->data()), o_data,
                                     M, K, N);
            } else if (b->dtype() == DataType::IQ2_S) {
                PERF_SCOPE("matmul_transB/iq2_s_q8k_fused_gemv_neon");
                forge::cpu::gemv_iq2_s_q8k_transB_neon(a_data, static_cast<const uint8_t*>(b->data()),
                                                       o_data, M, K, N);
            } else if (b->dtype() == DataType::IQ2_XS) {
                PERF_SCOPE("matmul_transB/iq2_xs_q8k_fused_gemv_neon");
                forge::cpu::gemv_iq2_xs_q8k_transB_neon(a_data, static_cast<const uint8_t*>(b->data()),
                                                        o_data, M, K, N);
            } else if (b->dtype() == DataType::IQ3_S) {
                PERF_SCOPE("matmul_transB/iq3_s_q8k_fused_gemv_neon");
                forge::cpu::gemv_iq3_s_q8k_transB_neon(a_data, static_cast<const uint8_t*>(b->data()),
                                                       o_data, M, K, N);
            } else if (b->dtype() == DataType::IQ4_NL) {
                PERF_SCOPE("matmul_transB/iq4_nl_q8k_fused_gemv_neon");
                forge::cpu::gemv_iq4_nl_q8k_transB_neon(a_data, static_cast<const uint8_t*>(b->data()),
                                                        o_data, M, K, N);
            } else
#elif defined(USE_VSX)
            if (b->dtype() == DataType::Q4_0) {
                PERF_SCOPE("matmul_transB/q4_0_gemv_vsx");
                cpu::gemv_q4_0_transB_vsx(a_data, static_cast<const uint8_t*>(b->data()),
                                           o_data, M, K, N);
            } else if (b->dtype() == DataType::Q8_0) {
                PERF_SCOPE("matmul_transB/q8_0_gemv_vsx");
                cpu::gemv_q8_0_transB_vsx(a_data, static_cast<const uint8_t*>(b->data()),
                                           o_data, M, K, N);
            } else if (b->dtype() == DataType::Q4_1) {
                PERF_SCOPE("matmul_transB/q4_1_gemv_vsx");
                cpu::gemv_q4_1_transB_vsx(a_data, static_cast<const uint8_t*>(b->data()),
                                           o_data, M, K, N);
            } else if (b->dtype() == DataType::Q5_0) {
                PERF_SCOPE("matmul_transB/q5_0_gemv_vsx");
                cpu::gemv_q5_0_transB_vsx(a_data, static_cast<const uint8_t*>(b->data()),
                                           o_data, M, K, N);
            } else if (b->dtype() == DataType::Q5_1) {
                PERF_SCOPE("matmul_transB/q5_1_gemv_vsx");
                cpu::gemv_q5_1_transB_vsx(a_data, static_cast<const uint8_t*>(b->data()),
                                           o_data, M, K, N);
            } else if (b->dtype() == DataType::IQ2_S) {
                PERF_SCOPE("matmul_transB/iq2_s_q8k_fused_gemv_vsx");
                cpu::gemv_iq2_s_q8k_transB_vsx(a_data, static_cast<const uint8_t*>(b->data()),
                                               o_data, M, K, N);
            } else if (b->dtype() == DataType::IQ2_XS) {
                PERF_SCOPE("matmul_transB/iq2_xs_q8k_fused_gemv_vsx");
                cpu::gemv_iq2_xs_q8k_transB_vsx(a_data, static_cast<const uint8_t*>(b->data()),
                                                o_data, M, K, N);
            } else if (b->dtype() == DataType::IQ3_S) {
                PERF_SCOPE("matmul_transB/iq3_s_q8k_fused_gemv_vsx");
                cpu::gemv_iq3_s_q8k_transB_vsx(a_data, static_cast<const uint8_t*>(b->data()),
                                               o_data, M, K, N);
            } else if (b->dtype() == DataType::IQ4_NL) {
                PERF_SCOPE("matmul_transB/iq4_nl_q8k_fused_gemv_vsx");
                cpu::gemv_iq4_nl_q8k_transB_vsx(a_data, static_cast<const uint8_t*>(b->data()),
                                                o_data, M, K, N);
            } else if (b->dtype() == DataType::Q4_K) {
                PERF_SCOPE("matmul_transB/q4_k_gemm_vsx");
                cpu::gemm_q4_K_vsx(a_data, static_cast<const uint8_t*>(b->data()), o_data, M, K, N);
            } else if (b->dtype() == DataType::Q6_K) {
                PERF_SCOPE("matmul_transB/q6_k_gemm_vsx");
                cpu::gemm_q6_K_vsx(a_data, static_cast<const uint8_t*>(b->data()), o_data, M, K, N);
            } else if (b->dtype() == DataType::Q2_K) {
                PERF_SCOPE("matmul_transB/q2_k_gemm_vsx");
                cpu::gemm_q2_K_vsx(a_data, static_cast<const uint8_t*>(b->data()), o_data, M, K, N);
            } else if (b->dtype() == DataType::Q3_K) {
                PERF_SCOPE("matmul_transB/q3_k_gemm_vsx");
                cpu::gemm_q3_K_vsx(a_data, static_cast<const uint8_t*>(b->data()), o_data, M, K, N);
            } else if (b->dtype() == DataType::Q5_K) {
                PERF_SCOPE("matmul_transB/q5_k_gemm_vsx");
                cpu::gemm_q5_K_vsx(a_data, static_cast<const uint8_t*>(b->data()), o_data, M, K, N);
            } else
#endif
            {
                PERF_SCOPE("matmul_transB/dequant+gemv");
                auto dequant_fn = get_dequant_row_fn(b->dtype());
                if (!dequant_fn)
                    throw std::runtime_error("Unsupported quantized type in matmul_transB: " + dtype_name(b->dtype()));
                const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
                int block_el = dtype_block_elements(b->dtype());
                int block_bytes = dtype_block_size(b->dtype());
                int blocks_per_row = (K + block_el - 1) / block_el;
                size_t row_bytes = (size_t)blocks_per_row * block_bytes;
                size_t expected_total = (size_t)N * row_bytes;
                if (expected_total > b->nbytes()) {
                    fprintf(stderr, "[ERROR] matmul_transB: expected_total(%zu) > b->nbytes(%zu)! Buffer overflow risk!\n",
                            expected_total, b->nbytes());
                    fflush(stderr);
                }
#ifdef USE_AVX2
// For Q4_K/Q6_K: dequantize scalar + AVX2 dot product
#    pragma omp parallel
                {
                    std::vector<float> row_buf(K);
#    pragma omp for schedule(dynamic)
                    for (int n = 0; n < N; ++n) {
                        dequant_fn(q_data, row_buf.data(), K, n);
                        for (int m = 0; m < M; ++m) {
                            o_data[m * N + n] =
                                cpu::dot_product_avx2(a_data + m * K, row_buf.data(), K);
                        }
                    }
                }
#elif defined(USE_NEON)
#    if FORGE_USE_OPENBLAS
                PERF_SCOPE("matmul_transB/dequant+gemm_openblas");
                {
                    std::vector<float> b_fp32(N * K);
                    for (int n = 0; n < N; ++n) {
                        dequant_fn(q_data, &b_fp32[n * K], K, n);
                    }
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, a_data, K,
                                b_fp32.data(), K, 0.0f, o_data, N);
                }
#    else
#        pragma omp parallel
                {
                    std::vector<float> row_buf(K);
#        pragma omp for schedule(dynamic)
                    for (int n = 0; n < N; ++n) {
                        dequant_fn(q_data, row_buf.data(), K, n);
                        for (int m = 0; m < M; ++m) {
                            const float* a_row = a_data + m * K;
                            float sum = 0.0f;
                            for (int k = 0; k < K; ++k) {
                                sum += a_row[k] * row_buf[k];
                            }
                            o_data[m * N + n] = sum;
                        }
                    }
                }
#    endif
#else
#    if FORGE_USE_OPENBLAS
                PERF_SCOPE("matmul_transB/dequant+gemm_openblas");
                {
                    std::vector<float> b_fp32(N * K);
                    for (int n = 0; n < N; ++n) {
                        dequant_fn(q_data, &b_fp32[n * K], K, n);
                    }
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, a_data, K,
                                b_fp32.data(), K, 0.0f, o_data, N);
                }
#    else
#        pragma omp parallel
                {
                    std::vector<float> row_buf(K);
#        pragma omp for schedule(dynamic)
                    for (int n = 0; n < N; ++n) {
                        dequant_fn(q_data, row_buf.data(), K, n);
                        for (int m = 0; m < M; ++m) {
                            const float* a_row = a_data + m * K;
                            float sum = 0.0f;
                            for (int k = 0; k < K; ++k) {
                                sum += a_row[k] * row_buf[k];
                            }
                            o_data[m * N + n] = sum;
                        }
                    }
                }
#    endif
#endif
            }
        } else {
#if FORGE_USE_OPENBLAS
            PERF_SCOPE("matmul_transB/fp32_gemm_openblas");
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, a_data, K,
                        static_cast<const float*>(b->data()), K, 0.0f, o_data, N);
#elif defined(USE_AVX2)
            PERF_SCOPE("matmul_transB/fp32_gemv_avx2");
            cpu::gemv_fp32_transB_avx2(a_data, static_cast<const float*>(b->data()), o_data, M, K,
                                       N);
#elif defined(USE_NEON)
            PERF_SCOPE("matmul_transB/fp32_gemv_neon");
            forge::cpu::gemv_fp32_transB_neon(a_data, static_cast<const float*>(b->data()), o_data, M, K, N);
#elif defined(USE_VSX)
            PERF_SCOPE("matmul_transB/fp32_gemv_vsx");
            cpu::gemv_fp32_transB_vsx(a_data, static_cast<const float*>(b->data()), o_data, M, K, N);
#else
            PERF_SCOPE("matmul_transB/fp32_gemv_scalar");
            const float* b_data = static_cast<const float*>(b->data());
#    pragma omp parallel for schedule(dynamic) if (M * N > 64)
            for (int m = 0; m < M; ++m) {
                const float* a_row = a_data + m * K;
                float* o_row = o_data + m * N;
                for (int n = 0; n < N; ++n) {
                    const float* b_row = b_data + n * K;
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += a_row[k] * b_row[k];
                    }
                    o_row[n] = sum;
                }
            }
#endif
        }
    }

    apply_bias(out, bias, M, N);
    return out;
}

TensorPtr matmul_transB_dual(const TensorPtr& a, const TensorPtr& b1, const TensorPtr& b2) {
    if (a->ndim() != 2 || b1->ndim() != 2 || b2->ndim() != 2)
        throw std::runtime_error("matmul_transB_dual expects 2D tensors");

    int M = static_cast<int>(a->shape()[0]);
    int K = static_cast<int>(a->shape()[1]);
    int N1 = static_cast<int>(b1->shape()[0]);
    int N2 = static_cast<int>(b2->shape()[0]);
    int N = N1 + N2;

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, a->device());

    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda_matmul_transB_dual(a, b1, b2, out, M, K, N1, N2);
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        float* o_data = static_cast<float*>(out->data());

        auto compute_part = [&](const TensorPtr& b, int offset, int n_cols) {
            if (is_quantized_type(b->dtype())) {
                auto dequant_fn = get_dequant_row_fn(b->dtype());
                const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
                std::vector<float> row_buf(K);
                for (int n = 0; n < n_cols; ++n) {
                    dequant_fn(q_data, row_buf.data(), K, n);
                    for (int m = 0; m < M; ++m) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            sum += a_data[m * K + k] * row_buf[k];
                        }
                        o_data[m * N + offset + n] = sum;
                    }
                }
            } else {
                const float* b_data = static_cast<const float*>(b->data());
                for (int n = 0; n < n_cols; ++n) {
                    for (int m = 0; m < M; ++m) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            sum += a_data[m * K + k] * b_data[n * K + k];
                        }
                        o_data[m * N + offset + n] = sum;
                    }
                }
            }
        };

        compute_part(b1, 0, N1);
        compute_part(b2, N1, N2);
    }

    return out;
}

TensorPtr ffn_up_fused(const TensorPtr& input, const TensorPtr& w1, const TensorPtr& w3,
                       int intermediate_dim) {
    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, intermediate_dim},
                                        input->device());

    if (input->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        out = cuda_ffn_up_fused(input, w1, w3, out, M, K, intermediate_dim);
#endif
    } else {
        auto gate = ops::matmul_transB(input, w1);
        auto up = ops::matmul_transB(input, w3);
        out = ops::silu_multiply(gate, up);
    }

    return out;
}

TensorPtr matmul_transB_fused_qkv_q4_0(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv) {
    // All three weights must be Q4_0, same K dimension
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_0 expects 2D tensors");
    if (wq->dtype() != DataType::Q4_0 || wk->dtype() != DataType::Q4_0 ||
        wv->dtype() != DataType::Q4_0)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_qkv_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_transB_neon(a_data + m * K, static_cast<const uint8_t*>(wq->data()),
                                   static_cast<float*>(q_out->data()) + m * N_q, 1, K, N_q);
        cpu::gemv_q4_0_transB_neon(a_data + m * K, static_cast<const uint8_t*>(wk->data()),
                                   static_cast<float*>(k_out->data()) + m * N_k, 1, K, N_k);
        cpu::gemv_q4_0_transB_neon(a_data + m * K, static_cast<const uint8_t*>(wv->data()),
                                   static_cast<float*>(v_out->data()) + m * N_v, 1, K, N_v);
    }
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/fused_qkv_q4_0_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_transB_vsx(a_data + m * K, static_cast<const uint8_t*>(wq->data()),
                                  static_cast<float*>(q_out->data()) + m * N_q, 1, K, N_q);
        cpu::gemv_q4_0_transB_vsx(a_data + m * K, static_cast<const uint8_t*>(wk->data()),
                                  static_cast<float*>(k_out->data()) + m * N_k, 1, K, N_k);
        cpu::gemv_q4_0_transB_vsx(a_data + m * K, static_cast<const uint8_t*>(wv->data()),
                                  static_cast<float*>(v_out->data()) + m * N_v, 1, K, N_v);
    }
#else
    // Fallback: separate matmul_transB calls
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    // Return Q, K, V as separate tensors packaged in a concatenated format
    // The caller (llama_engine) will extract them
    // For simplicity, we use a vector-like structure:
    // Return Q in a custom way - actually let's return them separately.
    // Since TensorPtr can only return one tensor, we pack Q, K, V consecutively.
    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }

    // Store individual outputs as metadata for easy extraction
    // We'll use a convention: the returned tensor has shape [M, N_q + N_k + N_v]
    // The caller splits it using slice operations.
    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q4_0(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_0 expects 2D tensors");
    if (weight->dtype() != DataType::Q4_0)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_ffn_down_residual_avx2(a_data + m * K,
                                              static_cast<const uint8_t*>(weight->data()),
                                              r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_transB_neon(a_data + m * K, static_cast<const uint8_t*>(weight->data()), o_data + m * N, 1, K, N);
        for (int n = 0; n < N; ++n) o_data[m * N + n] += r_data[m * N + n];
    }
#elif defined(USE_VSX)
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_transB_vsx(a_data + m * K, static_cast<const uint8_t*>(weight->data()), o_data + m * N, 1, K, N);
        for (int n = 0; n < N; ++n) o_data[m * N + n] += r_data[m * N + n];
    }
#else
    // Fallback: separate matmul + add
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q4_1(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_1 expects 2D tensors");
    if (weight->dtype() != DataType::Q4_1)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q4_1 requires Q4_1 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_1 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q4_1");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_1_ffn_down_residual_avx2(a_data + m * K,
                                               static_cast<const uint8_t*>(weight->data()),
                                               r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_1_transB_neon(a_data + m * K, static_cast<const uint8_t*>(weight->data()), o_data + m * N, 1, K, N);
        for (int n = 0; n < N; ++n) o_data[m * N + n] += r_data[m * N + n];
    }
#elif defined(USE_VSX)
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_1_transB_vsx(a_data + m * K, static_cast<const uint8_t*>(weight->data()), o_data + m * N, 1, K, N);
        for (int n = 0; n < N; ++n) o_data[m * N + n] += r_data[m * N + n];
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

// Fused attention output projection + residual for Q4_0 decode.
// Computes: out = attn_out @ wo + hidden_residual  (single pass)
TensorPtr matmul_transB_fused_attn_proj_residual_q4_0(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_0 expects 2D tensors");
    if (weight->dtype() != DataType::Q4_0)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_transB_neon(a_data + m * K,
                                   static_cast<const uint8_t*>(weight->data()),
                                   o_data + m * N, 1, K, N);
        for (int n = 0; n < N; ++n) o_data[m * N + n] += r_data[m * N + n];
    }
#elif defined(USE_VSX)
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_transB_vsx(a_data + m * K,
                                  static_cast<const uint8_t*>(weight->data()),
                                  o_data + m * N, 1, K, N);
        for (int n = 0; n < N; ++n) o_data[m * N + n] += r_data[m * N + n];
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_attn_proj_residual_q4_k(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_k expects 2D tensors");
    if (weight->dtype() != DataType::Q4_K)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_k requires Q4_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_k_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/attn_proj_residual_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q4_K_ffn_down_residual_neon(a_data + m * K,
                                              static_cast<const uint8_t*>(weight->data()),
                                              r_data + m * N, o_data + m * N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/attn_proj_residual_q4_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q4_K_attn_proj_residual_vsx(a_data + m * K,
                                              static_cast<const uint8_t*>(weight->data()),
                                              r_data + m * N, o_data + m * N, K, N);
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_attn_proj_residual_q5_k(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_attn_proj_residual_q5_k expects 2D tensors");
    if (weight->dtype() != DataType::Q5_K)
        throw std::runtime_error("matmul_transB_fused_attn_proj_residual_q5_k requires Q5_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_attn_proj_residual_q5_k is CPU-only");
    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q5_k_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/attn_proj_residual_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q5_K_attn_proj_residual_neon(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/attn_proj_residual_q5_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q5_K_attn_proj_residual_vsx(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_attn_proj_residual_q6_k(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q6_k expects 2D tensors");
    if (weight->dtype() != DataType::Q6_K)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q6_k requires Q6_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q6_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q6_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q6_k_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/attn_proj_residual_q6_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) cpu::gemv_q6_K_attn_proj_residual_neon(a_data + m*K, static_cast<const uint8_t*>(weight->data()), r_data + m*N, o_data + m*N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/attn_proj_residual_q6_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) cpu::gemv_q6_K_attn_proj_residual_vsx(a_data + m*K, static_cast<const uint8_t*>(weight->data()), r_data + m*N, o_data + m*N, K, N);
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_attn_proj_residual_q2_k(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_attn_proj_residual_q2_k expects 2D tensors");
    if (weight->dtype() != DataType::Q2_K)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q2_k requires Q2_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q2_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q2_k_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) cpu::gemv_q2_K_attn_proj_residual_neon(a_data + m*K, static_cast<const uint8_t*>(weight->data()), r_data + m*N, o_data + m*N, K, N);
#elif defined(USE_VSX)
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) cpu::gemv_q2_K_attn_proj_residual_vsx(a_data + m*K, static_cast<const uint8_t*>(weight->data()), r_data + m*N, o_data + m*N, K, N);
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_attn_proj_residual_q3_k(const TensorPtr& input,
                                                       const TensorPtr& weight,
                                                       const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q3_k expects 2D tensors");
    if (weight->dtype() != DataType::Q3_K)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q3_k requires Q3_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error(
            "matmul_transB_fused_attn_proj_residual_q3_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/attn_proj_residual_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_attn_proj_residual_avx2(a_data + m * K,
                                                static_cast<const uint8_t*>(weight->data()),
                                                r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/attn_proj_residual_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q3_K_attn_proj_residual_neon(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/attn_proj_residual_q3_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q3_K_attn_proj_residual_vsx(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_qkv_q4_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_k expects 2D tensors");
    if (wq->dtype() != DataType::Q4_K || wk->dtype() != DataType::Q4_K ||
        wv->dtype() != DataType::Q4_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_k requires Q4_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_K_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_qkv_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_K_fused_qkv_neon(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/fused_qkv_q4_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_K_fused_qkv_vsx(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q4_k(const TensorPtr& input, const TensorPtr& w_gate,
                                          const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_k expects 2D tensors");
    if (w_gate->dtype() != DataType::Q4_K || w_up->dtype() != DataType::Q4_K)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_k requires Q4_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_K_fused_ffn_up_avx2(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_ffn_up_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_K_fused_ffn_up_neon(a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
                                         static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/fused_ffn_up_q4_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q4_K_fused_ffn_up_vsx(a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
                                        static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q3_k(const TensorPtr& input, const TensorPtr& w_gate,
                                         const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q3_k expects 2D tensors");
    if (w_gate->dtype() != DataType::Q3_K || w_up->dtype() != DataType::Q3_K)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q3_k requires Q3_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q3_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_fused_ffn_up_avx2(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_ffn_up_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_fused_ffn_up_neon(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/fused_ffn_up_q3_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q3_K_fused_ffn_up_vsx(a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
                                        static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

TensorPtr matmul_transB_fused_qk_q3_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qk_q3_k expects 2D tensors");
    if (wq->dtype() != DataType::Q3_K || wk->dtype() != DataType::Q3_K)
        throw std::runtime_error("matmul_transB_fused_qk_q3_k requires Q3_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qk_q3_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qk_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_fused_qk_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k, K, N_q, N_k);
    }
#elif defined(USE_NEON)
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
#endif

    int total_N = N_q + N_k;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB_fused_qkv_q3_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k expects 2D tensors");
    if (wq->dtype() != DataType::Q3_K || wk->dtype() != DataType::Q3_K ||
        wv->dtype() != DataType::Q3_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k requires Q3_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_qkv_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_K_fused_qkv_neon(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/fused_qkv_q3_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q3_K_fused_qkv_vsx(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB_fused_qkv_q3_k_q4_k(const TensorPtr& input, const TensorPtr& wq,
                                             const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k_q4_k expects 2D tensors");
    if (wq->dtype() != DataType::Q3_K || wk->dtype() != DataType::Q3_K ||
        wv->dtype() != DataType::Q4_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k_q4_k requires Q3_K (Q,K) + Q4_K (V)");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q3_k_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q3_k_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        gemv_q3_k_q4_k_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_qkv_q3_k_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_q4_k_fused_qkv_neon(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q6_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q6_k expects 2D tensors");
    if (weight->dtype() != DataType::Q6_K)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q6_k requires Q6_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q6_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);
    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q6_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q6_k_ffn_down_residual_avx2(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/ffn_down_residual_q6_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q6_K_ffn_down_residual_neon(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/ffn_down_residual_q6_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q6_K_ffn_down_residual_vsx(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif
    return out;
}


TensorPtr matmul_transB_fused_ffn_down_residual_q4_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_k expects 2D tensors");
    if (weight->dtype() != DataType::Q4_K)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q4_k requires Q4_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_k_ffn_down_residual_avx2(a_data + m * K,
                                               static_cast<const uint8_t*>(weight->data()),
                                               r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/ffn_down_residual_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q4_K_ffn_down_residual_neon(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/ffn_down_residual_q4_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q4_K_ffn_down_residual_vsx(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q5_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q5_k expects 2D tensors");
    if (weight->dtype() != DataType::Q5_K)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q5_k requires Q5_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q5_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q5_k_ffn_down_residual_avx2(a_data + m * K,
                                               static_cast<const uint8_t*>(weight->data()),
                                               r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/ffn_down_residual_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q5_K_ffn_down_residual_neon(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/ffn_down_residual_q5_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q5_K_ffn_down_residual_vsx(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q2_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q2_k expects 2D tensors");
    if (weight->dtype() != DataType::Q2_K)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q2_k requires Q2_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q2_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q2_k_ffn_down_residual_avx2(a_data + m * K,
                                               static_cast<const uint8_t*>(weight->data()),
                                               r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/ffn_down_residual_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q2_K_ffn_down_residual_neon(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/ffn_down_residual_q2_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q2_K_ffn_down_residual_vsx(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_down_residual_q3_k(const TensorPtr& input,
                                                     const TensorPtr& weight,
                                                     const TensorPtr& residual) {
    if (input->ndim() != 2 || weight->ndim() != 2 || residual->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q3_k expects 2D tensors");
    if (weight->dtype() != DataType::Q3_K)
        throw std::runtime_error(
            "matmul_transB_fused_ffn_down_residual_q3_k requires Q3_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q3_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q3_k_ffn_down_residual_avx2(a_data + m * K,
                                               static_cast<const uint8_t*>(weight->data()),
                                               r_data + m * N, o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/ffn_down_residual_q3_k");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q3_K_ffn_down_residual_neon(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/ffn_down_residual_q3_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q3_K_ffn_down_residual_vsx(a_data + m * K, static_cast<const uint8_t*>(weight->data()), r_data + m * N, o_data + m * N, K, N);
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q4_0(const TensorPtr& input, const TensorPtr& w_gate,
                                          const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_0 expects 2D tensors");
    if (w_gate->dtype() != DataType::Q4_0 || w_up->dtype() != DataType::Q4_0)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_fused_ffn_up_avx2(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_ffn_up_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_fused_ffn_up_neon(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q5_k(const TensorPtr& input, const TensorPtr& w_gate,
                                         const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q5_k expects 2D tensors");
    if (w_gate->dtype() != DataType::Q5_K || w_up->dtype() != DataType::Q5_K)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q5_k requires Q5_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q5_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        gemv_q5_k_fused_ffn_up_avx2(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_ffn_up_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q5_K_fused_ffn_up_neon(a_data + m * K, static_cast<const uint8_t*>(w_gate->data()), static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/fused_ffn_up_q5_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q5_K_fused_ffn_up_vsx(a_data + m * K, static_cast<const uint8_t*>(w_gate->data()), static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q2_k(const TensorPtr& input, const TensorPtr& w_gate,
                                           const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q2_k expects 2D tensors");
    if (w_gate->dtype() != DataType::Q2_K || w_up->dtype() != DataType::Q2_K)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q2_k requires Q2_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q2_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        gemv_q2_k_fused_ffn_up_avx2(
            a_data + m * K, static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_ffn_up_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q2_K_fused_ffn_up_neon(a_data + m * K, static_cast<const uint8_t*>(w_gate->data()), static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/fused_ffn_up_q2_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q2_K_fused_ffn_up_vsx(a_data + m * K, static_cast<const uint8_t*>(w_gate->data()), static_cast<const uint8_t*>(w_up->data()), o_data + m * N, K, N);
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

TensorPtr matmul_transB_fused_qkv_q5_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q5_k expects 2D tensors");
    if (wq->dtype() != DataType::Q5_K || wk->dtype() != DataType::Q5_K ||
        wv->dtype() != DataType::Q5_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q5_k requires Q5_K weights");
    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q5_K_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_qkv_q5_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q5_K_fused_qkv_neon(a_data + m * K, static_cast<const uint8_t*>(wq->data()), static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()), static_cast<float*>(q_out->data()) + m * N_q, static_cast<float*>(k_out->data()) + m * N_k, static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/fused_qkv_q5_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q5_K_fused_qkv_vsx(a_data + m * K, static_cast<const uint8_t*>(wq->data()), static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()), static_cast<float*>(q_out->data()) + m * N_q, static_cast<float*>(k_out->data()) + m * N_k, static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB_fused_qkv_q2_k(const TensorPtr& input, const TensorPtr& wq,
                                       const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q2_k expects 2D tensors");
    if (wq->dtype() != DataType::Q2_K || wk->dtype() != DataType::Q2_K ||
        wv->dtype() != DataType::Q2_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q2_k requires Q2_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q2_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q2_K_fused_qkv_avx2(
            a_data + m * K, static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
    }
#elif defined(USE_NEON)
    PERF_SCOPE("matmul_transB/fused_qkv_q2_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q2_K_fused_qkv_neon(a_data + m * K, static_cast<const uint8_t*>(wq->data()), static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()), static_cast<float*>(q_out->data()) + m * N_q, static_cast<float*>(k_out->data()) + m * N_k, static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
#elif defined(USE_VSX)
    PERF_SCOPE("matmul_transB/fused_qkv_q2_k_vsx");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m)
        cpu::gemv_q2_K_fused_qkv_vsx(a_data + m * K, static_cast<const uint8_t*>(wq->data()), static_cast<const uint8_t*>(wk->data()), static_cast<const uint8_t*>(wv->data()), static_cast<float*>(q_out->data()) + m * N_q, static_cast<float*>(k_out->data()) + m * N_k, static_cast<float*>(v_out->data()) + m * N_v, K, N_q, N_k, N_v);
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q,
                    N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k,
                    N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v,
                    N_v * sizeof(float));
    }
    return out;
}

}  // namespace ops
}  // namespace forge
