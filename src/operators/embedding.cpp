#include "nanoinfer/operator_embedding.h"
#include "nanoinfer/operator_matmul.h"
#include "nanoinfer/cuda_kernels.h"
#include "nanoinfer/perf_profiler.h"
#include <cmath>
#include <cstring>
#include <vector>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nanoinfer {
namespace ops {

static inline float fp16_to_fp32_embed(uint16_t bits) {
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

static void dequant_q4_0_rows(const uint8_t* q_data, float* out,
                               const int32_t* indices, int num_indices,
                               int vocab_size, int embed_dim) {
    const int Q4_0_BLOCK_SIZE = 18;
    int blocks_per_row = (embed_dim + 31) / 32;
    size_t row_bytes = blocks_per_row * Q4_0_BLOCK_SIZE;

    for (int i = 0; i < num_indices; ++i) {
        int vocab_idx = indices[i];
        if (vocab_idx < 0 || vocab_idx >= vocab_size) continue;

        const uint8_t* row_ptr = q_data + vocab_idx * row_bytes;
        float* out_row = out + i * embed_dim;

        for (int bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block_ptr = row_ptr + bi * Q4_0_BLOCK_SIZE;
            float scale = fp16_to_fp32_embed(*reinterpret_cast<const uint16_t*>(block_ptr));
            const uint8_t* qs = block_ptr + 2;
            int base = bi * 32;
            for (int j = 0; j < 16 && base + j < embed_dim; ++j) {
                out_row[base + j] = static_cast<float>((qs[j] & 0x0F) - 8) * scale;
            }
            for (int j = 0; j < 16 && base + 16 + j < embed_dim; ++j) {
                out_row[base + 16 + j] = static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale;
            }
        }
    }
}

static void dequant_q4_1_rows(const uint8_t* q_data, float* out,
                               const int32_t* indices, int num_indices,
                               int vocab_size, int embed_dim) {
    const int Q4_1_BLOCK_SIZE = 20;
    int blocks_per_row = (embed_dim + 31) / 32;
    size_t row_bytes = blocks_per_row * Q4_1_BLOCK_SIZE;

    for (int i = 0; i < num_indices; ++i) {
        int vocab_idx = indices[i];
        if (vocab_idx < 0 || vocab_idx >= vocab_size) continue;

        const uint8_t* row_ptr = q_data + vocab_idx * row_bytes;
        float* out_row = out + i * embed_dim;

        for (int bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block_ptr = row_ptr + bi * Q4_1_BLOCK_SIZE;
            float d_val = fp16_to_fp32_embed(*reinterpret_cast<const uint16_t*>(block_ptr));
            float m_val = fp16_to_fp32_embed(*reinterpret_cast<const uint16_t*>(block_ptr + 2));
            const uint8_t* qs = block_ptr + 4;
            int base = bi * 32;
            for (int j = 0; j < 16 && base + j < embed_dim; ++j) {
                out_row[base + j] = static_cast<float>(qs[j] & 0x0F) * d_val + m_val;
            }
            for (int j = 0; j < 16 && base + 16 + j < embed_dim; ++j) {
                out_row[base + 16 + j] = static_cast<float>((qs[j] >> 4) & 0x0F) * d_val + m_val;
            }
        }
    }
}

static void dequant_q8_0_rows(const uint8_t* q_data, float* out,
                               const int32_t* indices, int num_indices,
                               int vocab_size, int embed_dim) {
    const int Q8_0_BLOCK_SIZE = 34;
    int blocks_per_row = (embed_dim + 31) / 32;
    size_t row_bytes = blocks_per_row * Q8_0_BLOCK_SIZE;

    for (int i = 0; i < num_indices; ++i) {
        int vocab_idx = indices[i];
        if (vocab_idx < 0 || vocab_idx >= vocab_size) continue;

        const uint8_t* row_ptr = q_data + vocab_idx * row_bytes;
        float* out_row = out + i * embed_dim;

        for (int bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block_ptr = row_ptr + bi * Q8_0_BLOCK_SIZE;
            float scale = fp16_to_fp32_embed(*reinterpret_cast<const uint16_t*>(block_ptr));
            const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + 2);
            int base = bi * 32;
            for (int j = 0; j < 32 && base + j < embed_dim; ++j) {
                out_row[base + j] = static_cast<float>(qs[j]) * scale;
            }
        }
    }
}

static constexpr int QK_K_EMB = 256;

static void get_scale_min_k4_emb(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void dequant_q4_k_rows(const uint8_t* q_data, float* out,
                               const int32_t* indices, int num_indices,
                               int vocab_size, int embed_dim) {
    const int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (embed_dim + QK_K_EMB - 1) / QK_K_EMB;
    size_t row_bytes = blocks_per_row * Q4_K_BLOCK_SIZE;

    for (int i = 0; i < num_indices; ++i) {
        int vocab_idx = indices[i];
        if (vocab_idx < 0 || vocab_idx >= vocab_size) continue;

        const uint8_t* row_ptr = q_data + vocab_idx * row_bytes;
        float* out_row = out + i * embed_dim;

        for (int bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block_ptr = row_ptr + bi * Q4_K_BLOCK_SIZE;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block_ptr, 2);
            memcpy(&dmin_bits, block_ptr + 2, 2);
            float d = fp16_to_fp32_embed(d_bits);
            float dmin = fp16_to_fp32_embed(dmin_bits);
            const uint8_t* scales = block_ptr + 4;
            const uint8_t* qs = block_ptr + 16;

            int is = 0;
            for (int j = 0; j < QK_K_EMB; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4_emb(is, scales, &sc1, &m1);
                get_scale_min_k4_emb(is + 1, scales, &sc2, &m2);
                float d1 = d * sc1;
                float m1_val = dmin * m1;
                float d2 = d * sc2;
                float m2_val = dmin * m2;
                int base = bi * QK_K_EMB + j;
                for (int l = 0; l < 32; ++l) {
                    if (base + l < embed_dim)
                        out_row[base + l] = d1 * static_cast<float>(qs[l] & 0xF) - m1_val;
                }
                for (int l = 0; l < 32; ++l) {
                    if (base + 32 + l < embed_dim)
                        out_row[base + 32 + l] = d2 * static_cast<float>(qs[l] >> 4) - m2_val;
                }
                qs += 32;
                is += 2;
            }
        }
    }
}

static void dequant_q6_k_rows(const uint8_t* q_data, float* out,
                               const int32_t* indices, int num_indices,
                               int vocab_size, int embed_dim) {
    const int Q6_K_BLOCK_SIZE = 210;
    int blocks_per_row = (embed_dim + QK_K_EMB - 1) / QK_K_EMB;
    size_t row_bytes = blocks_per_row * Q6_K_BLOCK_SIZE;

    for (int i = 0; i < num_indices; ++i) {
        int vocab_idx = indices[i];
        if (vocab_idx < 0 || vocab_idx >= vocab_size) continue;

        const uint8_t* row_ptr = q_data + vocab_idx * row_bytes;
        float* out_row = out + i * embed_dim;

        for (int bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block_ptr = row_ptr + bi * Q6_K_BLOCK_SIZE;
            const uint8_t* ql = block_ptr;
            const uint8_t* qh = ql + 128;
            const int8_t* sc = reinterpret_cast<const int8_t*>(qh + 64);
            uint16_t d_bits;
            memcpy(&d_bits, sc + 16, 2);
            float d = fp16_to_fp32_embed(d_bits);

            float* y = out_row + bi * QK_K_EMB;
            const uint8_t* ql_cur = ql;
            const uint8_t* qh_cur = qh;
            const int8_t* sc_cur = sc;

            for (int n = 0; n < QK_K_EMB; n += 128) {
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
}

using DequantEmbFn = void(*)(const uint8_t*, float*, const int32_t*, int, int, int);

static DequantEmbFn get_dequant_emb_fn(DataType dt) {
    switch (dt) {
        case DataType::Q4_0: return dequant_q4_0_rows;
        case DataType::Q4_1: return dequant_q4_1_rows;
        case DataType::Q8_0: return dequant_q8_0_rows;
        case DataType::Q4_K: return dequant_q4_k_rows;
        case DataType::Q6_K: return dequant_q6_k_rows;
        default: return nullptr;
    }
}

TensorPtr embedding(const TensorPtr& weight, const TensorPtr& indices,
                    const TensorPtr& fp32_cache) {
    int num_indices = static_cast<int>(indices->numel());

    int embed_dim, vocab_size;
    if (weight->shape().size() >= 2 && weight->shape()[0] < weight->shape()[1]) {
        embed_dim = static_cast<int>(weight->shape()[0]);
        vocab_size = static_cast<int>(weight->shape()[1]);
    } else {
        vocab_size = static_cast<int>(weight->shape()[0]);
        embed_dim = static_cast<int>(weight->shape()[1]);
    }

    auto out = std::make_shared<Tensor>(DataType::FP32,
        std::vector<int64_t>{num_indices, embed_dim}, weight->device());

    if (weight->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        bool transposed = (weight->shape().size() >= 2 &&
                           weight->shape()[0] < weight->shape()[1]);
        if (weight->dtype() == DataType::Q4_0) {
            cuda::launch_embedding_q4_0(
                weight->data(),
                static_cast<const int32_t*>(indices->data()),
                static_cast<float*>(out->data()),
                num_indices, embed_dim, vocab_size, transposed);
        } else if (weight->dtype() == DataType::Q4_1) {
            cuda::launch_embedding_q4_1(
                weight->data(),
                static_cast<const int32_t*>(indices->data()),
                static_cast<float*>(out->data()),
                num_indices, embed_dim, vocab_size, transposed);
        } else if (weight->dtype() == DataType::Q4_K) {
            cuda::launch_embedding_q4_k(
                weight->data(),
                static_cast<const int32_t*>(indices->data()),
                static_cast<float*>(out->data()),
                num_indices, embed_dim, vocab_size, transposed);
        } else if (weight->dtype() == DataType::Q6_K) {
            cuda::launch_embedding_q6_k(
                weight->data(),
                static_cast<const int32_t*>(indices->data()),
                static_cast<float*>(out->data()),
                num_indices, embed_dim, vocab_size, transposed);
        } else if (is_quantized_type(weight->dtype())) {
            auto dequant_fn = get_dequant_emb_fn(weight->dtype());
            if (dequant_fn) {
                auto out_cpu = std::make_shared<Tensor>(DataType::FP32,
                    std::vector<int64_t>{num_indices, embed_dim}, DeviceType::CPU);
                const uint8_t* q_data = static_cast<const uint8_t*>(weight->data());
                std::vector<uint8_t> host_q;
                if (weight->device() == DeviceType::CUDA) {
                    size_t q_bytes = weight->nbytes();
                    host_q.resize(q_bytes);
                    cudaMemcpy(host_q.data(), q_data, q_bytes, cudaMemcpyDeviceToHost);
                    q_data = host_q.data();
                }
                const int32_t* idx_data = static_cast<const int32_t*>(indices->data());
                std::vector<int32_t> host_idx;
                if (indices->device() == DeviceType::CUDA) {
                    host_idx.resize(num_indices);
                    cudaMemcpy(host_idx.data(), idx_data, num_indices * sizeof(int32_t), cudaMemcpyDeviceToHost);
                    idx_data = host_idx.data();
                }
                float* o_data = static_cast<float*>(out_cpu->data());
                dequant_fn(q_data, o_data, idx_data, num_indices, vocab_size, embed_dim);
                out->copy_from(*out_cpu);
            } else {
                out->zero_();
            }
        } else {
            cuda::launch_embedding_fp32(
                static_cast<const float*>(weight->data()),
                static_cast<const int32_t*>(indices->data()),
                static_cast<float*>(out->data()),
                num_indices, embed_dim, vocab_size, transposed);
        }
#endif
    } else {
        const int32_t* idx_data = static_cast<const int32_t*>(indices->data());
        float* o_data = static_cast<float*>(out->data());

        if (is_quantized_type(weight->dtype())) {
            bool transposed = (weight->shape().size() >= 2 &&
                               weight->shape()[0] < weight->shape()[1]);
            if (transposed) {
                PERF_SCOPE("embedding/dequant_lookup");
                // Use pre-dequantized FP32 cache if available (avoids full dequant per call)
                TensorPtr fp32_weight = fp32_cache;
                if (!fp32_weight) {
                    if (weight->dtype() == DataType::Q4_0) {
                        fp32_weight = dequantize_q4_0_weight(weight);
                    } else if (weight->dtype() == DataType::Q4_1) {
                        fp32_weight = dequantize_q4_1_weight(weight);
                    } else {
                        std::memset(o_data, 0, num_indices * embed_dim * sizeof(float));
                        return out;
                    }
                }
                const float* w_data = static_cast<const float*>(fp32_weight->data());
                for (int i = 0; i < num_indices; ++i) {
                    int vocab_idx = idx_data[i];
                    if (vocab_idx < 0 || vocab_idx >= vocab_size) continue;
                    for (int d = 0; d < embed_dim; ++d) {
                        o_data[i * embed_dim + d] = w_data[d * vocab_size + vocab_idx];
                    }
                }
            } else {
                auto dequant_fn = get_dequant_emb_fn(weight->dtype());
                if (dequant_fn) {
                    dequant_fn(static_cast<const uint8_t*>(weight->data()),
                               o_data, idx_data, num_indices, vocab_size, embed_dim);
                } else {
                    std::memset(o_data, 0, num_indices * embed_dim * sizeof(float));
                }
            }
        } else {
            const float* w_data = static_cast<const float*>(weight->data());
            for (int i = 0; i < num_indices; ++i) {
                int vocab_idx = idx_data[i];
                if (vocab_idx < 0 || vocab_idx >= vocab_size) continue;
                std::memcpy(o_data + i * embed_dim,
                            w_data + vocab_idx * embed_dim,
                            embed_dim * sizeof(float));
            }
        }
    }
    return out;
}

} // namespace ops
} // namespace nanoinfer
