#include "forge/quant_policy.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "forge/logger.h"
#include "forge/tensor.h"

namespace forge {

// ---- FP16 helpers ----

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(uint32_t));
    uint32_t sign = (x >> 31) & 1;
    uint32_t exp  = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;
    uint16_t h;
    if (exp == 0xFF) {
        h = (sign << 15) | 0x7C00 | (mant ? 1 : 0);
    } else if (exp >= 143) {
        h = (sign << 15) | 0x7C00;
    } else if (exp >= 113) {
        h = (sign << 15) | ((exp - 112) << 10) | (mant >> 13);
    } else if (exp >= 103) {
        uint32_t sub_exp  = 126 - exp;
        uint32_t sub_mant = (mant | 0x800000) >> (sub_exp + 13);
        h = (sign << 15) | sub_mant;
    } else {
        h = static_cast<uint16_t>(sign << 15);
    }
    return h;
}

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) { x = sign << 31; }
        else {
            int e = -1;
            while (!(mant & 0x400)) { mant <<= 1; --e; }
            mant &= 0x3FF;
            x = (sign << 31) | ((127 + e - 14) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        x = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        x = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, sizeof(float));
    return f;
}

// ---- K-type scale encoding ----
// 12 bytes encode 8 pairs of (sc, m) as 6-bit unsigned values.
// Inverse of get_scale_min_k4 used in dequantize.

static void encode_scales_k4(const uint8_t sc[8], const uint8_t m[8], uint8_t* s) {
    std::memset(s, 0, 12);
    for (int j = 0; j < 4; ++j) {
        s[j]     = sc[j] & 63;
        s[j + 4] = m[j] & 63;
    }
    for (int j = 4; j < 8; ++j) {
        s[j - 4] |= (sc[j] >> 4) << 6;
        s[j]     |= (m[j] >> 4) << 6;
        s[j + 4]  = (sc[j] & 0xF) | ((m[j] & 0xF) << 4);
    }
}

// ---- Quantize functions (CPU, flat array) ----

static void quantize_f16_cpu(const float* data, uint8_t* q_data, int n) {
    auto* qs = reinterpret_cast<uint16_t*>(q_data);
    for (int i = 0; i < n; ++i) qs[i] = fp32_to_fp16(data[i]);
}

static void quantize_q8_0_cpu(const float* data, uint8_t* q_data, int n) {
    constexpr int BLOCK = 32;
    int nb = (n + BLOCK - 1) / BLOCK;
    for (int b = 0; b < nb; ++b) {
        int s = b * BLOCK, e = std::min(s + BLOCK, n);
        float amax = 0;
        for (int i = s; i < e; ++i) amax = std::max(amax, std::fabs(data[i]));
        float d = amax / 127.0f; if (d == 0) d = 1.0f;
        float id = 1.0f / d;
        uint16_t df = fp32_to_fp16(d);
        std::memcpy(q_data + b * 34, &df, 2);
        auto* qs = reinterpret_cast<int8_t*>(q_data + b * 34 + 2);
        for (int i = 0; i < BLOCK; ++i) {
            int idx = s + i;
            qs[i] = (idx < e) ? static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, data[idx] * id)))) : 0;
        }
    }
}

static void quantize_q4_0_cpu(const float* data, uint8_t* q_data, int n) {
    constexpr int BLOCK = 32;
    int nb = (n + BLOCK - 1) / BLOCK;
    for (int b = 0; b < nb; ++b) {
        int s = b * BLOCK, e = std::min(s + BLOCK, n);
        float amax = 0;
        for (int i = s; i < e; ++i) amax = std::max(amax, std::fabs(data[i]));
        float d = amax / -8.0f; if (d == 0) d = 1.0f;
        float id = 1.0f / d;
        uint16_t df = fp32_to_fp16(d);
        std::memcpy(q_data + b * 18, &df, 2);
        uint8_t* qs = q_data + b * 18 + 2;
        for (int i = 0; i < 16; ++i) {
            int i0 = s + i, i1 = s + i + 16;
            int8_t v0 = (i0 < e) ? static_cast<int8_t>(std::round(std::max(-8.0f, std::min(7.0f, data[i0] * id)))) : 0;
            int8_t v1 = (i1 < e) ? static_cast<int8_t>(std::round(std::max(-8.0f, std::min(7.0f, data[i1] * id)))) : 0;
            qs[i] = (static_cast<uint8_t>(v0) & 0x0F) | ((static_cast<uint8_t>(v1) & 0x0F) << 4);
        }
    }
}

// Q4_K: 144 bytes / 256 elements. 使用子块 scale/min 正确处理负值。
// 反量化公式: out = d * sc * qs_val - dmin * m, qs_val ∈ [0,15]
static void quantize_q4_k_cpu(const float* data, uint8_t* q_data, int n) {
    constexpr int QK = 256;
    int nb = (n + QK - 1) / QK;
    for (int b = 0; b < nb; ++b) {
        int bs = b * QK;
        uint8_t* block = q_data + b * 144;
        // 遍历 8 个子块，收集 sc/m 信息
        uint8_t sc_arr[8] = {}, m_arr[8] = {};
        float sub_max[8] = {}, sub_min[8] = {};
        for (int j = 0; j < 8; ++j) {
            for (int l = 0; l < 32; ++l) {
                int idx = bs + j * 32 + l;
                float v = (idx < n) ? data[idx] : 0.0f;
                if (l == 0) { sub_max[j] = sub_min[j] = v; }
                else { sub_max[j] = std::max(sub_max[j], v); sub_min[j] = std::min(sub_min[j], v); }
            }
        }
        // 计算超块 d / dmin
        float max_sc = 0, max_m = 0;
        for (int j = 0; j < 8; ++j) {
            float range = sub_max[j] - sub_min[j];
            max_sc = std::max(max_sc, range / 15.0f);
            max_m  = std::max(max_m, std::fabs(sub_min[j]));
        }
        float d = max_sc / 63.0f;    if (d == 0) d = 1.0f;
        float dm = max_m / 63.0f;    if (dm == 0) dm = 1.0f;
        uint16_t df = fp32_to_fp16(d), dminf = fp32_to_fp16(dm);
        std::memcpy(block, &df, 2);
        std::memcpy(block + 2, &dminf, 2);
        for (int j = 0; j < 8; ++j) {
            float range = sub_max[j] - sub_min[j];
            sc_arr[j] = static_cast<uint8_t>(std::min(63.0f, std::round(range / (15.0f * d))));
            m_arr[j]  = static_cast<uint8_t>(std::min(63.0f, std::round(-sub_min[j] / dm)));
            if (sc_arr[j] == 0) sc_arr[j] = 1;
        }
        encode_scales_k4(sc_arr, m_arr, block + 4);
        // 量化每个元素: qs_val = round((data + dmin*m) / (d*sc)), clamp [0,15]
        uint8_t* qs = block + 16;
        for (int j = 0; j < 8; ++j) {
            float inv_scale = (d * sc_arr[j] != 0) ? 1.0f / (d * sc_arr[j]) : 0;
            float offset = dm * m_arr[j];
            for (int l = 0; l < 16; ++l) {
                int i0 = bs + j * 32 + l, i1 = i0 + 16;
                float v0 = (i0 < n) ? data[i0] : -offset;
                float v1 = (i1 < n) ? data[i1] : -offset;
                int q0 = static_cast<int>(std::round((v0 + offset) * inv_scale));
                int q1 = static_cast<int>(std::round((v1 + offset) * inv_scale));
                q0 = std::max(0, std::min(15, q0));
                q1 = std::max(0, std::min(15, q1));
                qs[j * 16 + l] = static_cast<uint8_t>(q0 | (q1 << 4));
            }
        }
    }
}

// Q5_K: 176 bytes / 256 elements. 与 Q4_K 类似但 5-bit (qs_val ∈ [0,31])
static void quantize_q5_k_cpu(const float* data, uint8_t* q_data, int n) {
    constexpr int QK = 256;
    int nb = (n + QK - 1) / QK;
    for (int b = 0; b < nb; ++b) {
        int bs = b * QK;
        uint8_t* block = q_data + b * 176;
        uint8_t sc_arr[8] = {}, m_arr[8] = {};
        float sub_max[8] = {}, sub_min[8] = {};
        // Q5_K 子块: 4 × 64 元素, 每块 2 组 (sc, m)
        for (int j = 0; j < 8; ++j) {
            for (int l = 0; l < 32; ++l) {
                int idx = bs + j * 32 + l;
                float v = (idx < n) ? data[idx] : 0.0f;
                if (l == 0) { sub_max[j] = sub_min[j] = v; }
                else { sub_max[j] = std::max(sub_max[j], v); sub_min[j] = std::min(sub_min[j], v); }
            }
        }
        float max_sc = 0, max_m = 0;
        for (int j = 0; j < 8; ++j) {
            max_sc = std::max(max_sc, (sub_max[j] - sub_min[j]) / 31.0f);
            max_m  = std::max(max_m, std::fabs(sub_min[j]));
        }
        float d = max_sc / 63.0f;    if (d == 0) d = 1.0f;
        float dm = max_m / 63.0f;    if (dm == 0) dm = 1.0f;
        uint16_t df = fp32_to_fp16(d), dminf = fp32_to_fp16(dm);
        std::memcpy(block, &df, 2);
        std::memcpy(block + 2, &dminf, 2);
        for (int j = 0; j < 8; ++j) {
            sc_arr[j] = static_cast<uint8_t>(std::min(63.0f, std::round((sub_max[j] - sub_min[j]) / (31.0f * d))));
            m_arr[j]  = static_cast<uint8_t>(std::min(63.0f, std::round(-sub_min[j] / dm)));
            if (sc_arr[j] == 0) sc_arr[j] = 1;
        }
        encode_scales_k4(sc_arr, m_arr, block + 4);
        uint8_t* qh = block + 16;
        uint8_t* ql = block + 48;
        std::memset(qh, 0, 32);
        // 每 64 元素一组: 32 个低 nibble + 32 个高 nibble
        // u1/u2 掩码对应 qh 的 bit 位 (参考 dequantize_q5_k_row)
        for (int grp = 0; grp < 4; ++grp) {
            int grp_base = grp * 64;
            for (int l = 0; l < 32; ++l) {
                // 前半 32 元素
                int i0 = bs + grp_base + l;
                int i1 = bs + grp_base + l + 32;
                float offset0 = dm * m_arr[grp * 2], offset1 = dm * m_arr[grp * 2 + 1];
                float inv0 = (d * sc_arr[grp * 2] != 0) ? 1.0f / (d * sc_arr[grp * 2]) : 0;
                float inv1 = (d * sc_arr[grp * 2 + 1] != 0) ? 1.0f / (d * sc_arr[grp * 2 + 1]) : 0;
                float v0 = (i0 < n) ? data[i0] : -offset0;
                float v1 = (i1 < n) ? data[i1] : -offset1;
                int q0 = static_cast<int>(std::round((v0 + offset0) * inv0));
                int q1 = static_cast<int>(std::round((v1 + offset1) * inv1));
                q0 = std::max(0, std::min(31, q0));
                q1 = std::max(0, std::min(31, q1));
                ql[grp * 32 + l] = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
                if (q0 & 0x10) qh[l] |= (1 << (grp * 2));
                if (q1 & 0x10) qh[l] |= (1 << (grp * 2 + 1));
            }
        }
    }
}

// Q6_K: 210 bytes / 256 elements. 有符号 6-bit, int8 子块 scale.
// 反量化: out = d * int8(sc) * (q6_val - 32), q6_val ∈ [0,63]
static void quantize_q6_k_cpu(const float* data, uint8_t* q_data, int n) {
    constexpr int QK = 256;
    int nb = (n + QK - 1) / QK;
    for (int b = 0; b < nb; ++b) {
        int bs = b * QK;
        uint8_t* block = q_data + b * 210;
        uint8_t* ql = block;
        uint8_t* qh = ql + 128;
        int8_t* sc = reinterpret_cast<int8_t*>(qh + 64);
        // 计算超块 d: 使 int8(sc) × (q6-32) 覆盖最大范围
        float max_abs = 0;
        for (int i = 0; i < QK; ++i) {
            int idx = bs + i;
            if (idx < n) max_abs = std::max(max_abs, std::fabs(data[idx]));
        }
        // d * 127 * 31 ≥ max_abs  →  d = max_abs / (127*31)
        float d = max_abs / (127.0f * 31.0f);
        if (d == 0) d = 1.0f;
        uint16_t df = fp32_to_fp16(d);
        // 处理 2 个 128 元素组
        for (int grp = 0; grp < 2; ++grp) {
            int gb = bs + grp * 128;
            uint8_t* ql_g = ql + grp * 64;
            uint8_t* qh_g = qh + grp * 32;
            int8_t* sc_g  = sc + grp * 8;
            std::memset(ql_g, 0, 64);
            std::memset(qh_g, 0, 32);
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;  // scale index: 0 or 1
                // 找该子块 (16 元素) 的 amax
                float amax0 = 0, amax1 = 0, amax2 = 0, amax3 = 0;
                for (int k = 0; k < 16; ++k) {
                    int idx0 = gb + l + k;       if (idx0 < n) amax0 = std::max(amax0, std::fabs(data[idx0]));
                    int idx1 = gb + l + 32 + k;  if (idx1 < n) amax1 = std::max(amax1, std::fabs(data[idx1]));
                    int idx2 = gb + l + 64 + k;  if (idx2 < n) amax2 = std::max(amax2, std::fabs(data[idx2]));
                    int idx3 = gb + l + 96 + k;  if (idx3 < n) amax3 = std::max(amax3, std::fabs(data[idx3]));
                }
                sc_g[is + 0] = static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(std::round(amax0 / (d * 31.0f))))));
                sc_g[is + 2] = static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(std::round(amax1 / (d * 31.0f))))));
                sc_g[is + 4] = static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(std::round(amax2 / (d * 31.0f))))));
                sc_g[is + 6] = static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(std::round(amax3 / (d * 31.0f))))));
                if (sc_g[is + 0] == 0) sc_g[is + 0] = 1;
                if (sc_g[is + 2] == 0) sc_g[is + 2] = 1;
                if (sc_g[is + 4] == 0) sc_g[is + 4] = 1;
                if (sc_g[is + 6] == 0) sc_g[is + 6] = 1;
                for (int k = 0; k < 16; ++k) {
                    auto q6 = [&](int idx, int8_t s) -> int {
                        float v = (idx < n) ? data[idx] : 0;
                        float scale = d * s;
                        int q = (scale != 0) ? static_cast<int>(std::round(v / scale)) + 32 : 32;
                        return std::max(0, std::min(63, q));
                    };
                    int idx0 = gb + l + k;
                    int idx1 = gb + l + 32 + k;
                    int idx2 = gb + l + 64 + k;
                    int idx3 = gb + l + 96 + k;
                    int q0 = q6(idx0, sc_g[is + 0]);
                    int q1 = q6(idx1, sc_g[is + 2]);
                    int q2 = q6(idx2, sc_g[is + 4]);
                    int q3 = q6(idx3, sc_g[is + 6]);
                    // ql: 低 4 bit (q0,q2 共享 byte, q1,q3 共享 byte)
                    ql_g[l]      |= static_cast<uint8_t>((q0 & 0xF) | ((q2 & 0xF) << 4));
                    ql_g[l + 32] |= static_cast<uint8_t>((q1 & 0xF) | ((q3 & 0xF) << 4));
                    // qh: 高 2 bit
                    qh_g[l] |= static_cast<uint8_t>(((q0 >> 4) & 3) | (((q1 >> 4) & 3) << 2) |
                                                     (((q2 >> 4) & 3) << 4) | (((q3 >> 4) & 3) << 6));
                }
            }
        }
        std::memcpy(block + 208, &df, 2);
    }
}

// ---- Dispatch ----

static bool quantize_dispatch(const float* fp32_data, uint8_t* q_data, int64_t numel, DataType dt) {
    int n = static_cast<int>(numel);
    switch (dt) {
    case DataType::FP16: quantize_f16_cpu(fp32_data, q_data, n); return true;
    case DataType::Q4_0: quantize_q4_0_cpu(fp32_data, q_data, n); return true;
    case DataType::Q8_0: quantize_q8_0_cpu(fp32_data, q_data, n); return true;
    case DataType::Q4_K: quantize_q4_k_cpu(fp32_data, q_data, n); return true;
    case DataType::Q5_K: quantize_q5_k_cpu(fp32_data, q_data, n); return true;
    case DataType::Q6_K: quantize_q6_k_cpu(fp32_data, q_data, n); return true;
    default: return false;
    }
}

// ---- select_quant_type: 按 GGUF tensor 名称模式匹配 ----

DataType select_quant_type(const QuantPolicy& policy, const std::string& name) {
    // output.weight → output_type (最高精度)
    if (name == "output.weight") return policy.output_type;
    // blk.N.attn_v.weight / blk.N.attn_v.bias → attn_wv_type
    if (name.find("attn_v") != std::string::npos) return policy.attn_wv_type;
    // blk.N.ffn_down.weight → ffn_down_type
    if (name.find("ffn_down") != std::string::npos) return policy.ffn_down_type;
    return policy.default_type;
}

// ---- requant_tensor: dequant → FP32 → re-quant ----

TensorPtr requant_tensor(const TensorPtr& src, DataType target_type) {
    if (!src || src->dtype() == target_type) return src;

    // 确保源数据在 CPU
    TensorPtr cpu_src = src;
    if (src->device() == DeviceType::CUDA) {
        cpu_src = std::make_shared<Tensor>(src->dtype(), src->shape(), DeviceType::CPU);
        cpu_src->copy_from(*src);
    }

    int64_t numel = src->numel();
    int64_t n_rows = src->shape().size() >= 2 ? src->shape()[0] : 1;
    int64_t K = numel / n_rows;

    // Step 1: Dequantize 到 FP32
    auto fp32 = std::make_shared<Tensor>(DataType::FP32, src->shape(), DeviceType::CPU);
    float* fp32_data = static_cast<float*>(fp32->data());

    if (cpu_src->dtype() == DataType::FP32) {
        std::memcpy(fp32_data, cpu_src->data(), numel * sizeof(float));
    } else if (cpu_src->dtype() == DataType::FP16) {
        auto* src16 = static_cast<const uint16_t*>(cpu_src->data());
        for (int64_t i = 0; i < numel; ++i)
            fp32_data[i] = fp16_to_fp32(src16[i]);
    } else if (is_quantized_type(cpu_src->dtype())) {
        auto dequant_fn = get_dequant_row_fn(cpu_src->dtype());
        if (!dequant_fn) {
            LOG_WARN("requant_tensor: no dequant for " + dtype_name(cpu_src->dtype()));
            return src;
        }
        auto* q_data = static_cast<const uint8_t*>(cpu_src->data());
        for (int64_t r = 0; r < n_rows; ++r)
            dequant_fn(q_data, fp32_data, static_cast<int>(K), static_cast<int>(r));
    } else {
        LOG_WARN("requant_tensor: unsupported source type " + dtype_name(cpu_src->dtype()));
        return src;
    }

    // Step 2: Quantize 到目标类型
    auto result = std::make_shared<Tensor>(target_type, src->shape(), DeviceType::CPU);
    if (!quantize_dispatch(fp32_data, static_cast<uint8_t*>(result->data()), numel, target_type)) {
        LOG_WARN("requant_tensor: no quantize for target " + dtype_name(target_type));
        return src;
    }

    LOG_INFO("requant: " + dtype_name(src->dtype()) + " -> " + dtype_name(target_type) +
             " (" + std::to_string(numel) + " elements)");
    return result;
}

}  // namespace forge
