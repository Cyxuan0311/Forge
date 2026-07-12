#include "forge/operator_matmul.h"
#include "forge/operator_elementwise.h"
#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"
#include "cpu_gemv.h"
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdio>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if FORGE_USE_OPENBLAS
#include <cblas.h>
#endif

namespace forge {
namespace ops {

#ifdef USE_CUDA
namespace {
struct CudaScratchBuffer {
    void* ptr = nullptr;
    size_t capacity = 0;

    ~CudaScratchBuffer() {
        if (ptr) cudaFree(ptr);
    }

    void* ensure(size_t bytes) {
        if (bytes > capacity) {
            if (ptr) cudaFree(ptr);
            cudaMalloc(&ptr, bytes);
            capacity = bytes;
        }
        return ptr;
    }
};

static CudaScratchBuffer& get_scratch() {
    static thread_local CudaScratchBuffer buf;
    return buf;
}
}
#endif

static inline float fp16_to_fp32(uint16_t bits) {
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0) return 0.0f;
        float v = std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
        return sign ? -v : v;
    }
    float v = std::ldexp((1.0f + static_cast<float>(mantissa) / 1024.0f),
                          static_cast<int>(exponent) - 15);
    return sign ? -v : v;
}

static void dequantize_q4_0_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q4_0_BLOCK_SIZE = 18;
    int blocks_per_row = (K + 31) / 32;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q4_0_BLOCK_SIZE;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q4_0_BLOCK_SIZE;
        float scale = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs = block_ptr + 2;
        int base = bi * 32;
        for (int j = 0; j < 16 && base + j < K; ++j)
            out[base + j] = static_cast<float>((qs[j] & 0x0F) - 8) * scale;
        for (int j = 0; j < 16 && base + 16 + j < K; ++j)
            out[base + 16 + j] = static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale;
    }
}

static void dequantize_q4_1_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q4_1_BLOCK_SIZE = 20;
    int blocks_per_row = (K + 31) / 32;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q4_1_BLOCK_SIZE;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q4_1_BLOCK_SIZE;
        float d_val = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        float m_val = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr + 2));
        const uint8_t* qs = block_ptr + 4;
        int base = bi * 32;
        for (int j = 0; j < 16 && base + j < K; ++j)
            out[base + j] = static_cast<float>(qs[j] & 0x0F) * d_val + m_val;
        for (int j = 0; j < 16 && base + 16 + j < K; ++j)
            out[base + 16 + j] = static_cast<float>((qs[j] >> 4) & 0x0F) * d_val + m_val;
    }
}

static void dequantize_q8_0_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q8_0_BLOCK_SIZE = 34;
    int blocks_per_row = (K + 31) / 32;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q8_0_BLOCK_SIZE;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q8_0_BLOCK_SIZE;
        float scale = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block_ptr));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + 2);
        int base = bi * 32;
        for (int j = 0; j < 32 && base + j < K; ++j)
            out[base + j] = static_cast<float>(qs[j]) * scale;
    }
}

static constexpr int QK_K = 256;

static void dequantize_q6_k_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q6_K_BLOCK_SIZE = 210;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q6_K_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q6_K_BLOCK_SIZE;
        const uint8_t* ql = block_ptr;
        const uint8_t* qh = ql + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(qh + 64);
        uint16_t d_bits;
        memcpy(&d_bits, sc + 16, 2);
        float d = fp16_to_fp32(d_bits);

        float* y = out + bi * QK_K;
        const uint8_t* ql_cur = ql;
        const uint8_t* qh_cur = qh;
        const int8_t* sc_cur = sc;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql_cur[l +  0] & 0xF) | (((qh_cur[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql_cur[l + 32] & 0xF) | (((qh_cur[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql_cur[l +  0] >> 4) | (((qh_cur[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql_cur[l + 32] >> 4) | (((qh_cur[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * static_cast<float>(sc_cur[is + 0]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(sc_cur[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(sc_cur[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(sc_cur[is + 6]) * static_cast<float>(q4);
            }
            y += 128;
            ql_cur += 64;
            qh_cur += 32;
            sc_cur += 8;
        }
    }
}

static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void dequantize_q4_k_row(const uint8_t* q_data, float* out, int K, int row) {
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q4_K_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q4_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = fp16_to_fp32(d_bits);
        float dmin = fp16_to_fp32(dmin_bits);
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qs = block_ptr + 16;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * sc1;
            float m1_val = dmin * m1;
            float d2 = d * sc2;
            float m2_val = dmin * m2;
            int base = bi * QK_K + j;
            for (int l = 0; l < 32; ++l) {
                if (base + l < K)
                    out[base + l] = d1 * static_cast<float>(qs[l] & 0xF) - m1_val;
            }
            for (int l = 0; l < 32; ++l) {
                if (base + 32 + l < K)
                    out[base + 32 + l] = d2 * static_cast<float>(qs[l] >> 4) - m2_val;
            }
            qs += 32;
            is += 2;
        }
    }
}

using DequantRowFn = void(*)(const uint8_t*, float*, int, int);

static DequantRowFn get_dequant_row_fn(DataType dt) {
    switch (dt) {
        case DataType::Q4_0: return dequantize_q4_0_row;
        case DataType::Q4_1: return dequantize_q4_1_row;
        case DataType::Q4_K: return dequantize_q4_k_row;
        case DataType::Q8_0: return dequantize_q8_0_row;
        case DataType::Q6_K: return dequantize_q6_k_row;
        default: return nullptr;
    }
}

static int get_row_block_bytes(DataType dt) {
    switch (dt) {
        case DataType::Q4_0: return 18;
        case DataType::Q4_1: return 20;
        case DataType::Q4_K: return 144;
        case DataType::Q8_0: return 34;
        case DataType::Q6_K: return 210;
        default: return 0;
    }
}

static int get_row_block_elements(DataType dt) {
    switch (dt) {
        case DataType::Q4_0: case DataType::Q4_1: case DataType::Q8_0: return 32;
        case DataType::Q4_K: case DataType::Q6_K: return 256;
        default: return 1;
    }
}

TensorPtr dequantize_q4_0_weight(const TensorPtr& q_weight) {
    int N = static_cast<int>(q_weight->shape()[0]);
    int K = static_cast<int>(q_weight->shape()[1]);
    auto fp32_weight = std::make_shared<Tensor>(DataType::FP32, q_weight->shape(), q_weight->device());
    const uint8_t* q_data = static_cast<const uint8_t*>(q_weight->data());
    float* out = static_cast<float*>(fp32_weight->data());
    std::vector<float> row_buf(K);
    for (int n = 0; n < N; ++n) {
        dequantize_q4_0_row(q_data, row_buf.data(), K, n);
        std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
    }
    return fp32_weight;
}

TensorPtr dequantize_q4_1_weight(const TensorPtr& q_weight) {
    int N = static_cast<int>(q_weight->shape()[0]);
    int K = static_cast<int>(q_weight->shape()[1]);
    auto fp32_weight = std::make_shared<Tensor>(DataType::FP32, q_weight->shape(), q_weight->device());
    const uint8_t* q_data = static_cast<const uint8_t*>(q_weight->data());
    float* out = static_cast<float*>(fp32_weight->data());
    std::vector<float> row_buf(K);
    for (int n = 0; n < N; ++n) {
        dequantize_q4_1_row(q_data, row_buf.data(), K, n);
        std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
    }
    return fp32_weight;
}

// Encode a float value to IEEE 754 half-precision (fp16) as uint16_t
static uint16_t fp32_to_fp16_bits(float f) {
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

// Quantize an FP32 [N, K] weight tensor to Q8_0 format.
// Q8_0 block: 2 bytes fp16 scale + 32 bytes int8 values = 34 bytes per 32 elements.
TensorPtr quantize_q8_0_weight(const TensorPtr& fp32_weight) {
    if (!fp32_weight) return nullptr;

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
                if (av > amax) amax = av;
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

TensorPtr dequantize_weight(const TensorPtr& weight) {
    if (!weight || !is_quantized_type(weight->dtype())) return weight;
    auto dequant_fn = get_dequant_row_fn(weight->dtype());
    if (!dequant_fn) return weight;

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

static void apply_bias(TensorPtr& out, const TensorPtr& bias, int M, int N) {
    if (!bias) return;
    const float* bias_data = static_cast<const float*>(bias->data());
    float* o_data = static_cast<float*>(out->data());

    if (bias->device() != out->device()) {
        if (out->device() == DeviceType::CUDA && bias->device() == DeviceType::CPU) {
#ifdef USE_CUDA
            auto bias_cuda = std::make_shared<Tensor>(bias->dtype(), bias->shape(), DeviceType::CUDA);
            bias_cuda->copy_from(*bias);
            bias_data = static_cast<const float*>(bias_cuda->data());
#endif
        }
    }

    if (bias->ndim() == 1) {
        int bias_size = static_cast<int>(bias->shape()[0]);
        if (out->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            for (int m = 0; m < M; ++m) {
                cuda::launch_add_bias(o_data + m * N, bias_data, o_data + m * N, bias_size);
            }
#endif
        } else {
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < bias_size && n < N; ++n) {
                    o_data[m * N + n] += bias_data[n];
                }
            }
        }
    } else {
        int total = static_cast<int>(out->numel());
        if (out->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            cuda::launch_add_bias(o_data, bias_data, o_data, total);
#endif
        } else {
            for (int i = 0; i < total; ++i) {
                o_data[i] += bias_data[i];
            }
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
        auto dequant_fn = get_dequant_row_fn(b->dtype());
        if (!dequant_fn) throw std::runtime_error("Unsupported quantized type in matmul");
        b_fp32 = std::make_shared<Tensor>(DataType::FP32, b->shape(), DeviceType::CPU);
        const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
        std::vector<uint8_t> host_q;
        if (b->device() == DeviceType::CUDA) {
            size_t q_bytes = b->nbytes();
            host_q.resize(q_bytes);
#ifdef USE_CUDA
            cudaMemcpy(host_q.data(), q_data, q_bytes, cudaMemcpyDeviceToHost);
#endif
            q_data = host_q.data();
        }
        float* out = static_cast<float*>(b_fp32->data());
        std::vector<float> row_buf(K);
        for (int n = 0; n < N; ++n) {
            dequant_fn(q_data, row_buf.data(), K, n);
            std::memcpy(out + n * K, row_buf.data(), K * sizeof(float));
        }
        if (a->device() == DeviceType::CUDA) {
            b_fp32->to_device(DeviceType::CUDA);
        }
    }

    int M = static_cast<int>(a->shape()[0]);
    int K = static_cast<int>(a->shape()[1]);
    int K2 = static_cast<int>(b_fp32->shape()[0]);
    int N = static_cast<int>(b_fp32->shape()[1]);
    if (K != K2) throw std::runtime_error("matmul dimension mismatch");

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, a->device());

    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        for (int m = 0; m < M; ++m) {
            cuda::launch_gemv(
                static_cast<const float*>(a->data()) + m * K,
                static_cast<const float*>(b_fp32->data()),
                static_cast<float*>(out->data()) + m * N,
                K, N);
        }
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b_fp32->data());
        float* o_data = static_cast<float*>(out->data());
#if FORGE_USE_OPENBLAS
        PERF_SCOPE("matmul/fp32_gemm_openblas");
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, a_data, K, b_data, N, 0.0f, o_data, N);
#elif defined(USE_AVX2)
        PERF_SCOPE("matmul/fp32_gemm_avx2");
        cpu::gemm_fp32_avx2(a_data, b_data, o_data, M, K, N);
#else
        PERF_SCOPE("matmul/fp32_gemm_scalar");
        std::memset(o_data, 0, M * N * sizeof(float));
        #pragma omp parallel for schedule(dynamic) if(M * N > 64)
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

TensorPtr matmul_transB(const TensorPtr& a, const TensorPtr& b, const TensorPtr& bias) {
    if (a->ndim() != 2 || b->ndim() != 2)
        throw std::runtime_error("matmul_transB expects 2D tensors");

    int M = static_cast<int>(a->shape()[0]);
    int K = static_cast<int>(a->shape()[1]);
    int N = static_cast<int>(b->shape()[0]);
    int K2 = static_cast<int>(b->shape()[1]);
    if (K != K2) throw std::runtime_error("matmul_transB dimension mismatch: K=" +
        std::to_string(K) + " K2=" + std::to_string(K2) +
        " b_shape=[" + std::to_string(b->shape()[0]) + "," + std::to_string(b->shape()[1]) + "]");

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, a->device());

    if (a->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        // Threshold: for M <= 32, use batched GEMV (on-the-fly dequant)
        // to avoid dequantizing the entire weight matrix to FP32.
        const int GEMV_THRESHOLD = 32;

        if (M > 1 && M <= GEMV_THRESHOLD) {
            // Small batch: use batched GEMV with on-the-fly dequantization
            if (b->dtype() == DataType::Q4_0) {
                cuda::launch_gemv_q4_0_transB_batch(
                    static_cast<const float*>(a->data()),
                    b->data(), static_cast<float*>(out->data()),
                    M, K, N);
            } else if (b->dtype() == DataType::Q4_1) {
                cuda::launch_gemv_q4_1_transB_batch(
                    static_cast<const float*>(a->data()),
                    b->data(), static_cast<float*>(out->data()),
                    M, K, N);
            } else if (b->dtype() == DataType::Q6_K) {
                cuda::launch_gemv_q6_k_transB_batch(
                    static_cast<const float*>(a->data()),
                    b->data(), static_cast<float*>(out->data()),
                    M, K, N);
            } else if (b->dtype() == DataType::Q4_K) {
                // Q4_K batch: on-the-fly dequant GEMV (avoids full matrix dequant)
                cuda::launch_gemv_q4_k_transB_batch(
                    static_cast<const float*>(a->data()),
                    b->data(), static_cast<float*>(out->data()),
                    M, K, N);
            } else if (is_quantized_type(b->dtype())) {
                auto dequant_fn = get_dequant_row_fn(b->dtype());
                if (!dequant_fn) throw std::runtime_error("Unsupported quantized type in matmul_transB CUDA");
                auto b_fp32 = std::make_shared<Tensor>(DataType::FP32, b->shape(), DeviceType::CPU);
                const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
                std::vector<uint8_t> host_q;
                if (b->device() == DeviceType::CUDA) {
                    size_t q_bytes = b->nbytes();
                    host_q.resize(q_bytes);
                    cudaMemcpy(host_q.data(), q_data, q_bytes, cudaMemcpyDeviceToHost);
                    q_data = host_q.data();
                }
                float* fp32_out = static_cast<float*>(b_fp32->data());
                std::vector<float> row_buf(K);
                for (int n = 0; n < N; ++n) {
                    dequant_fn(q_data, row_buf.data(), K, n);
                    std::memcpy(fp32_out + n * K, row_buf.data(), K * sizeof(float));
                }
                b_fp32->to_device(DeviceType::CUDA);
                cuda::launch_cublas_sgemm(
                    static_cast<const float*>(a->data()),
                    static_cast<const float*>(b_fp32->data()),
                    static_cast<float*>(out->data()),
                    M, K, N, true);
            } else {
                cuda::launch_cublas_sgemm(
                    static_cast<const float*>(a->data()),
                    static_cast<const float*>(b->data()),
                    static_cast<float*>(out->data()),
                    M, K, N, true);
            }
        } else if (M > 1) {
            // Large batch: dequantize + cublas GEMM
            if (b->dtype() == DataType::Q4_0) {
                size_t fp32_bytes = (size_t)N * K * sizeof(float);
                float* b_fp32 = static_cast<float*>(get_scratch().ensure(fp32_bytes));
                cuda::launch_dequant_q4_0_matrix(b->data(), b_fp32, N, K);
                cuda::launch_cublas_sgemm(
                    static_cast<const float*>(a->data()),
                    b_fp32,
                    static_cast<float*>(out->data()),
                    M, K, N, true);
            } else if (b->dtype() == DataType::Q4_1) {
                size_t fp32_bytes = (size_t)N * K * sizeof(float);
                float* b_fp32 = static_cast<float*>(get_scratch().ensure(fp32_bytes));
                cuda::launch_dequant_q4_1_matrix(b->data(), b_fp32, N, K);
                cuda::launch_cublas_sgemm(
                    static_cast<const float*>(a->data()),
                    b_fp32,
                    static_cast<float*>(out->data()),
                    M, K, N, true);
            } else if (b->dtype() == DataType::Q4_K) {
                size_t fp32_bytes = (size_t)N * K * sizeof(float);
                float* b_fp32 = static_cast<float*>(get_scratch().ensure(fp32_bytes));
                cuda::launch_dequant_q4_k_matrix(b->data(), b_fp32, N, K);
                cuda::launch_cublas_sgemm(
                    static_cast<const float*>(a->data()),
                    b_fp32,
                    static_cast<float*>(out->data()),
                    M, K, N, true);
            } else if (b->dtype() == DataType::Q6_K) {
                size_t fp32_bytes = (size_t)N * K * sizeof(float);
                float* b_fp32 = static_cast<float*>(get_scratch().ensure(fp32_bytes));
                cuda::launch_dequant_q6_k_matrix(b->data(), b_fp32, N, K);
                cuda::launch_cublas_sgemm(
                    static_cast<const float*>(a->data()),
                    b_fp32,
                    static_cast<float*>(out->data()),
                    M, K, N, true);
            } else if (is_quantized_type(b->dtype())) {
                auto dequant_fn = get_dequant_row_fn(b->dtype());
                if (!dequant_fn) throw std::runtime_error("Unsupported quantized type in matmul_transB CUDA");
                auto b_fp32 = std::make_shared<Tensor>(DataType::FP32, b->shape(), DeviceType::CPU);
                const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
                std::vector<uint8_t> host_q;
                if (b->device() == DeviceType::CUDA) {
                    size_t q_bytes = b->nbytes();
                    host_q.resize(q_bytes);
                    cudaMemcpy(host_q.data(), q_data, q_bytes, cudaMemcpyDeviceToHost);
                    q_data = host_q.data();
                }
                float* fp32_out = static_cast<float*>(b_fp32->data());
                std::vector<float> row_buf(K);
                for (int n = 0; n < N; ++n) {
                    dequant_fn(q_data, row_buf.data(), K, n);
                    std::memcpy(fp32_out + n * K, row_buf.data(), K * sizeof(float));
                }
                b_fp32->to_device(DeviceType::CUDA);
                cuda::launch_cublas_sgemm(
                    static_cast<const float*>(a->data()),
                    static_cast<const float*>(b_fp32->data()),
                    static_cast<float*>(out->data()),
                    M, K, N, true);
            } else {
                cuda::launch_cublas_sgemm(
                    static_cast<const float*>(a->data()),
                    static_cast<const float*>(b->data()),
                    static_cast<float*>(out->data()),
                    M, K, N, true);
            }
        } else {
            // M == 1: single GEMV
            if (b->dtype() == DataType::Q4_0) {
                cuda::launch_gemv_q4_0_transB(
                    static_cast<const float*>(a->data()),
                    b->data(), static_cast<float*>(out->data()),
                    K, N);
            } else if (b->dtype() == DataType::Q4_1) {
                cuda::launch_gemv_q4_1_transB(
                    static_cast<const float*>(a->data()),
                    b->data(), static_cast<float*>(out->data()),
                    K, N);
            } else if (b->dtype() == DataType::Q4_K) {
                cuda::launch_gemv_q4_k_transB(
                    static_cast<const float*>(a->data()),
                    b->data(), static_cast<float*>(out->data()),
                    K, N);
            } else if (b->dtype() == DataType::Q6_K) {
                cuda::launch_gemv_q6_k_transB(
                    static_cast<const float*>(a->data()),
                    b->data(), static_cast<float*>(out->data()),
                    K, N);
            } else if (is_quantized_type(b->dtype())) {
                auto dequant_fn = get_dequant_row_fn(b->dtype());
                if (!dequant_fn) throw std::runtime_error("Unsupported quantized type in matmul_transB CUDA");
                auto b_fp32 = std::make_shared<Tensor>(DataType::FP32, b->shape(), DeviceType::CPU);
                const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
                std::vector<uint8_t> host_q;
                if (b->device() == DeviceType::CUDA) {
                    size_t q_bytes = b->nbytes();
                    host_q.resize(q_bytes);
                    cudaMemcpy(host_q.data(), q_data, q_bytes, cudaMemcpyDeviceToHost);
                    q_data = host_q.data();
                }
                float* fp32_out = static_cast<float*>(b_fp32->data());
                std::vector<float> row_buf(K);
                for (int n = 0; n < N; ++n) {
                    dequant_fn(q_data, row_buf.data(), K, n);
                    std::memcpy(fp32_out + n * K, row_buf.data(), K * sizeof(float));
                }
                b_fp32->to_device(DeviceType::CUDA);
                cuda::launch_gemv_transB(
                    static_cast<const float*>(a->data()),
                    static_cast<const float*>(b_fp32->data()),
                    static_cast<float*>(out->data()),
                    K, N);
            } else {
                // FP32: use cuBLAS for large N (output_proj), custom GEMV for small N
                if (N > 4096) {
                    cuda::launch_cublas_sgemm(
                        static_cast<const float*>(a->data()),
                        static_cast<const float*>(b->data()),
                        static_cast<float*>(out->data()),
                        M, K, N, true);
                } else {
                    cuda::launch_gemv_transB(
                        static_cast<const float*>(a->data()),
                        static_cast<const float*>(b->data()),
                        static_cast<float*>(out->data()),
                        K, N);
                }
            }
        }
#endif
    } else {
        const float* a_data = static_cast<const float*>(a->data());
        float* o_data = static_cast<float*>(out->data());

        if (is_quantized_type(b->dtype())) {
#ifdef USE_AVX2
            if (b->dtype() == DataType::Q4_0) {
                PERF_SCOPE("matmul_transB/q4_0_fused_gemv");
                cpu::gemv_q4_0_transB_avx2(
                    a_data, static_cast<const uint8_t*>(b->data()), o_data,
                    M, K, N);
            } else if (b->dtype() == DataType::Q8_0) {
                PERF_SCOPE("matmul_transB/q8_0_fused_gemv");
                cpu::gemv_q8_0_transB_avx2(
                    a_data, static_cast<const uint8_t*>(b->data()), o_data,
                    M, K, N);
            } else if (b->dtype() == DataType::Q4_1) {
                PERF_SCOPE("matmul_transB/q4_1_fused_gemv");
                cpu::gemv_q4_1_transB_avx2(
                    a_data, static_cast<const uint8_t*>(b->data()), o_data,
                    M, K, N);
            } else if (b->dtype() == DataType::Q4_K) {
                PERF_SCOPE("matmul_transB/q4_k_fused_gemv");
                cpu::gemv_q4_k_transB_avx2(
                    a_data, static_cast<const uint8_t*>(b->data()), o_data,
                    M, K, N);
            } else
#endif
            {
                PERF_SCOPE("matmul_transB/dequant+gemv");
                auto dequant_fn = get_dequant_row_fn(b->dtype());
                if (!dequant_fn) throw std::runtime_error("Unsupported quantized type in matmul_transB");
                const uint8_t* q_data = static_cast<const uint8_t*>(b->data());
#ifdef USE_AVX2
                // For Q4_K/Q6_K: dequantize scalar + AVX2 dot product
                #pragma omp parallel
                {
                    std::vector<float> row_buf(K);
                    #pragma omp for schedule(dynamic)
                    for (int n = 0; n < N; ++n) {
                        dequant_fn(q_data, row_buf.data(), K, n);
                        for (int m = 0; m < M; ++m) {
                            o_data[m * N + n] = cpu::dot_product_avx2(a_data + m * K, row_buf.data(), K);
                        }
                    }
                }
#else
#if FORGE_USE_OPENBLAS
                PERF_SCOPE("matmul_transB/dequant+gemm_openblas");
                {
                    std::vector<float> b_fp32(N * K);
                    for (int n = 0; n < N; ++n) {
                        dequant_fn(q_data, &b_fp32[n * K], K, n);
                    }
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                M, N, K, 1.0f, a_data, K, b_fp32.data(), K,
                                0.0f, o_data, N);
                }
#else
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
#endif
#endif
            }
        } else {
#if FORGE_USE_OPENBLAS
            PERF_SCOPE("matmul_transB/fp32_gemm_openblas");
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, a_data, K,
                        static_cast<const float*>(b->data()), K,
                        0.0f, o_data, N);
#elif defined(USE_AVX2)
            PERF_SCOPE("matmul_transB/fp32_gemv_avx2");
            cpu::gemv_fp32_transB_avx2(
                a_data, static_cast<const float*>(b->data()), o_data,
                M, K, N);
#else
            PERF_SCOPE("matmul_transB/fp32_gemv_scalar");
            const float* b_data = static_cast<const float*>(b->data());
            #pragma omp parallel for schedule(dynamic) if(M * N > 64)
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
        auto out1 = ops::matmul_transB(a, b1);
        auto out2 = ops::matmul_transB(a, b2);

        float* o_data = static_cast<float*>(out->data());
        const float* o1_data = static_cast<const float*>(out1->data());
        const float* o2_data = static_cast<const float*>(out2->data());
        for (int m = 0; m < M; ++m) {
            cudaMemcpyAsync(o_data + m * N, o1_data + m * N1, N1 * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(o_data + m * N + N1, o2_data + m * N2, N2 * sizeof(float), cudaMemcpyDeviceToDevice);
        }
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

TensorPtr ffn_up_fused(const TensorPtr& input, const TensorPtr& w1, const TensorPtr& w3, int intermediate_dim) {
    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, intermediate_dim}, input->device());

    // Threshold: for M <= 32, use batched GEMV (on-the-fly dequant) to avoid
    // dequantizing the entire weight matrix. For larger M, cublas GEMM after
    // dequantization is more efficient due to higher compute intensity.
    const int GEMV_THRESHOLD = 32;

    if (input->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        if (w1->dtype() == DataType::Q4_0 && w3->dtype() == DataType::Q4_0) {
            if (M == 1) {
                // Decode: single-token fused GEMV kernel
                cuda::launch_ffn_up_fused_q4_0(
                    static_cast<const float*>(input->data()),
                    w1->data(), w3->data(),
                    static_cast<float*>(out->data()),
                    K, intermediate_dim);
            } else if (M <= GEMV_THRESHOLD) {
                // Small batch prefill: batched GEMV with on-the-fly dequant
                cuda::launch_ffn_up_fused_q4_0_batch_gemv(
                    static_cast<const float*>(input->data()),
                    w1->data(), w3->data(),
                    static_cast<float*>(out->data()),
                    M, K, intermediate_dim);
            } else {
                // Large batch prefill: dequantize + cublas GEMM
                cuda::launch_ffn_up_fused_q4_0_batch(
                    static_cast<const float*>(input->data()),
                    w1->data(), w3->data(),
                    static_cast<float*>(out->data()),
                    M, K, intermediate_dim);
            }
        } else if (w1->dtype() == DataType::Q4_K && w3->dtype() == DataType::Q4_K) {
            if (M == 1) {
                // Decode: single-token fused GEMV kernel (shares x vector read)
                cuda::launch_ffn_up_fused_q4_k(
                    static_cast<const float*>(input->data()),
                    w1->data(), w3->data(),
                    static_cast<float*>(out->data()),
                    K, intermediate_dim);
            } else if (M <= GEMV_THRESHOLD) {
                // Q4_K small batch: separate GEMV + silu_multiply
                auto gate = ops::matmul_transB(input, w1);
                auto up = ops::matmul_transB(input, w3);
                out = ops::silu_multiply(gate, up);
            } else {
                cuda::launch_ffn_up_fused_q4_k_batch(
                    static_cast<const float*>(input->data()),
                    w1->data(), w3->data(),
                    static_cast<float*>(out->data()),
                    M, K, intermediate_dim);
            }
        } else {
            auto gate = ops::matmul_transB(input, w1);
            auto up = ops::matmul_transB(input, w3);
            out = ops::silu_multiply(gate, up);
        }
#endif
    } else {
        auto gate = ops::matmul_transB(input, w1);
        auto up = ops::matmul_transB(input, w3);
        out = ops::silu_multiply(gate, up);
    }

    return out;
}

TensorPtr matmul_transB_fused_qkv_q4_0(const TensorPtr& input,
                                          const TensorPtr& wq, const TensorPtr& wk, const TensorPtr& wv) {
    // All three weights must be Q4_0, same K dimension
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_0 expects 2D tensors");
    if (wq->dtype() != DataType::Q4_0 || wk->dtype() != DataType::Q4_0 || wv->dtype() != DataType::Q4_0)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_fused_qkv_avx2(
            a_data + m * K,
            static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()),
            static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v,
            K, N_q, N_k, N_v);
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
    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q, N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k, N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v, N_v * sizeof(float));
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
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_down_residual_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(weight->shape()[0]);

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/ffn_down_residual_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    const float* r_data = static_cast<const float*>(residual->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_ffn_down_residual_avx2(
            a_data + m * K,
            static_cast<const uint8_t*>(weight->data()),
            r_data + m * N,
            o_data + m * N,
            K, N);
    }
#else
    // Fallback: separate matmul + add
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_qkv_q4_k(const TensorPtr& input,
                                         const TensorPtr& wq, const TensorPtr& wk, const TensorPtr& wv) {
    if (input->ndim() != 2 || wq->ndim() != 2 || wk->ndim() != 2 || wv->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_k expects 2D tensors");
    if (wq->dtype() != DataType::Q4_K || wk->dtype() != DataType::Q4_K || wv->dtype() != DataType::Q4_K)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_k requires Q4_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_qkv_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N_q = static_cast<int>(wq->shape()[0]);
    int N_k = static_cast<int>(wk->shape()[0]);
    int N_v = static_cast<int>(wv->shape()[0]);

    auto q_out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_q}, DeviceType::CPU);
    auto k_out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_k}, DeviceType::CPU);
    auto v_out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N_v}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_qkv_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_K_fused_qkv_avx2(
            a_data + m * K,
            static_cast<const uint8_t*>(wq->data()),
            static_cast<const uint8_t*>(wk->data()),
            static_cast<const uint8_t*>(wv->data()),
            static_cast<float*>(q_out->data()) + m * N_q,
            static_cast<float*>(k_out->data()) + m * N_k,
            static_cast<float*>(v_out->data()) + m * N_v,
            K, N_q, N_k, N_v);
    }
#else
    q_out = ops::matmul_transB(input, wq);
    k_out = ops::matmul_transB(input, wk);
    v_out = ops::matmul_transB(input, wv);
#endif

    int total_N = N_q + N_k + N_v;
    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, total_N}, DeviceType::CPU);
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        std::memcpy(o_data + m * total_N, static_cast<float*>(q_out->data()) + m * N_q, N_q * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q, static_cast<float*>(k_out->data()) + m * N_k, N_k * sizeof(float));
        std::memcpy(o_data + m * total_N + N_q + N_k, static_cast<float*>(v_out->data()) + m * N_v, N_v * sizeof(float));
    }
    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q4_k(const TensorPtr& input,
                                             const TensorPtr& w_gate, const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_k expects 2D tensors");
    if (w_gate->dtype() != DataType::Q4_K || w_up->dtype() != DataType::Q4_K)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_k requires Q4_K weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_k is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q4_k");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_k_fused_ffn_up_avx2(
            a_data + m * K,
            static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()),
            o_data + m * N,
            K, N);
    }
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

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

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q6_k_ffn_down_residual_avx2(
            a_data + m * K,
            static_cast<const uint8_t*>(weight->data()),
            r_data + m * N,
            o_data + m * N,
            K, N);
    }
#else
    out = ops::matmul_transB(input, weight);
    out = ops::add(residual, out);
#endif

    return out;
}

TensorPtr matmul_transB_fused_ffn_up_q4_0(const TensorPtr& input,
                                             const TensorPtr& w_gate, const TensorPtr& w_up) {
    if (input->ndim() != 2 || w_gate->ndim() != 2 || w_up->ndim() != 2)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_0 expects 2D tensors");
    if (w_gate->dtype() != DataType::Q4_0 || w_up->dtype() != DataType::Q4_0)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_0 requires Q4_0 weights");
    if (input->device() != DeviceType::CPU)
        throw std::runtime_error("matmul_transB_fused_ffn_up_q4_0 is CPU-only");

    int M = static_cast<int>(input->shape()[0]);
    int K = static_cast<int>(input->shape()[1]);
    int N = static_cast<int>(w_gate->shape()[0]);

    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CPU);

#ifdef USE_AVX2
    PERF_SCOPE("matmul_transB/fused_ffn_up_q4_0");
    const float* a_data = static_cast<const float*>(input->data());
    float* o_data = static_cast<float*>(out->data());

    for (int m = 0; m < M; ++m) {
        cpu::gemv_q4_0_fused_ffn_up_avx2(
            a_data + m * K,
            static_cast<const uint8_t*>(w_gate->data()),
            static_cast<const uint8_t*>(w_up->data()),
            o_data + m * N,
            K, N);
    }
#else
    auto gate = ops::matmul_transB(input, w_gate);
    auto up = ops::matmul_transB(input, w_up);
    out = ops::silu_multiply(gate, up);
#endif

    return out;
}

} // namespace ops
} // namespace forge
