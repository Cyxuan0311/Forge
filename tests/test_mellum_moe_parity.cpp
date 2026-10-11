// CPU-vs-CUDA parity test for MellumMoe::apply (mellum's routed-only SiLU MoE).
//
// The Mellum hybrid-offload garbage output came from the fused GPU path routing
// gate/up/down through launch_moe_expert_gemv, whose atomicAdd semantics express
// Σ_k w_k·W_{e_k}·x and therefore apply the SiLU to the cross-expert SUM. The
// kernel-level oracle (test_moe_slot_reference.cpp) pins the new slot kernels;
// this test pins the LAYER wiring on top of them — router indices/weights layout,
// per-slot activation, weight applied exactly once (at the down step), buffer
// shapes and dtype dispatch — by running the same synthetic layer on CPU and on
// CUDA and comparing.
//
// No GGUF needed: LayerWeights is a name → tensor map, so the layer can be fed
// hand-built weights. The binary skips gracefully when no CUDA device exists.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "forge/inference/layer_execution_context.h"
#include "forge/inference/layers/mellum_moe.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/model.h"
#include "forge/operators.h"
#include "forge/quant_traits.h"
#include "forge/tensor.h"
#include "forge/types.h"

using forge::DataType;
using forge::DeviceTarget;
using forge::DeviceType;
using forge::ForwardRequest;
using forge::LayerExecutionContext;
using forge::LayerWeights;
using forge::ModelConfig;
using forge::Tensor;
using forge::TensorPtr;

namespace {

uint16_t halfbits(float f) {
    __half h = __float2half(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

// Random quantized weight bytes with every per-block scale field overwritten so
// no block can produce NaN/Inf, and small enough that a 3-layer FFN chain of
// full-range random quants stays in float range (a real GGUF has |w| ~ 0.06;
// random nibbles sit near max magnitude, so d must be ~1e-3/1e-4).
//
// Every row of the tensor carries its own blocks — walking only the first row
// leaves random f16 scales (up to 1e30) in the rest, which is what an earlier
// revision of this test did.
void fill_quant_bytes(uint8_t* p, size_t n, DataType dt, int K, std::mt19937& rng) {
    std::uniform_int_distribution<int> u8(0, 255);
    for (size_t i = 0; i < n; ++i)
        p[i] = (uint8_t)u8(rng);

    auto set_f16 = [](uint8_t* base, size_t off, float v) {
        uint16_t s = halfbits(v);
        base[off] = s & 0xff;
        base[off + 1] = (s >> 8) & 0xff;
    };

    int BE = 32, BS = 18;
    switch (dt) {
    case DataType::Q4_0:
        BE = 32;
        BS = 18;
        break;
    case DataType::Q8_0:
        BE = 32;
        BS = 34;
        break;
    case DataType::Q5_0:
        BE = 32;
        BS = 22;
        break;
    case DataType::Q4_K:
        BE = 256;
        BS = 144;
        break;
    case DataType::Q6_K:
        BE = 256;
        BS = 210;
        break;
    default:
        return;
    }
    const int nb = (K + BE - 1) / BE;
    const size_t row_bytes = (size_t)nb * BS;
    const size_t n_rows = row_bytes ? n / row_bytes : 0;

    for (size_t row = 0; row < n_rows; ++row) {
        uint8_t* r = p + row * row_bytes;
        for (int b = 0; b < nb; ++b) {
            switch (dt) {
            case DataType::Q4_0:
                set_f16(r, b * 18, 1e-3f);
                break;
            case DataType::Q8_0:
                set_f16(r, b * 34, 1e-3f);
                break;
            case DataType::Q5_0:
                set_f16(r, b * 22, 1e-3f);
                break;
            case DataType::Q4_K:
                set_f16(r, b * 144 + 0, 1e-4f);  // d
                set_f16(r, b * 144 + 2, 5e-5f);  // dmin
                break;
            case DataType::Q6_K:
                set_f16(r, b * 210 + 208, 1e-4f);
                break;  // d
            default:
                break;
            }
        }
    }
}

TensorPtr make_f32(const std::vector<float>& host, const std::vector<int64_t>& shape,
                   DeviceType dev) {
    auto t = std::make_shared<Tensor>(DataType::FP32, shape, dev);
    if (dev == DeviceType::CPU) {
        std::memcpy(t->data(), host.data(), host.size() * sizeof(float));
    } else {
        auto tmp = std::make_shared<Tensor>(DataType::FP32, shape, DeviceType::CPU);
        std::memcpy(tmp->data(), host.data(), host.size() * sizeof(float));
        t->copy_from(*tmp);
    }
    return t;
}

}  // namespace

namespace {

// One geometry: same synthetic layer evaluated on CPU and on CUDA.
bool run_case(int hidden, int n_ff, int n_expert, int n_expert_used, DataType gate_dt,
              DataType down_dt, int seq_len, std::mt19937& rng) {
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);

    LayerWeights lw;
    std::vector<float> norm_w(hidden);
    for (int i = 0; i < hidden; ++i)
        norm_w[i] = 0.5f + 0.5f * ur(rng);
    lw.set("ffn_norm", make_f32(norm_w, {hidden}, DeviceType::CPU));

    std::vector<float> gate_inp((size_t)n_expert * hidden);
    for (auto& v : gate_inp)
        v = ur(rng) * 0.5f;
    lw.set("ffn_gate_inp", make_f32(gate_inp, {n_expert, hidden}, DeviceType::CPU));

    auto make_experts = [&](DataType dt, int N, int K) {
        auto t =
            std::make_shared<Tensor>(dt, std::vector<int64_t>{n_expert, N, K}, DeviceType::CPU);
        fill_quant_bytes(static_cast<uint8_t*>(t->data()), t->nbytes(), dt, K, rng);
        return t;
    };
    lw.set("ffn_gate_exps", make_experts(gate_dt, n_ff, hidden));
    lw.set("ffn_up_exps", make_experts(gate_dt, n_ff, hidden));
    lw.set("ffn_down_exps", make_experts(down_dt, hidden, n_ff));

    std::vector<float> in_host((size_t)seq_len * hidden);
    for (auto& v : in_host)
        v = ur(rng);

    ModelConfig cfg;
    cfg.hidden_dim = hidden;
    cfg.n_expert = n_expert;
    cfg.n_expert_used = n_expert_used;
    cfg.rms_norm_eps = 1e-6f;
    cfg.ffn_type = forge::FFNType::MoE;
    cfg.ffn_activation = forge::ActivationType::SiLU_GELU;

    ForwardRequest req = ForwardRequest::from_embedding(nullptr, seq_len, 0, 0);

    auto run = [&](DeviceType dev) {
        LayerWeights w = lw;
        w.to_device(dev);
        LayerExecutionContext lctx{cfg, w, req};
        lctx.layer_idx = 0;
        lctx.device = dev == DeviceType::CUDA ? DeviceTarget::cuda() : DeviceTarget::cpu();
        auto in = make_f32(in_host, {seq_len, hidden}, dev);
        auto out = forge::MellumMoe::apply(in, lctx, nullptr);
        auto out_cpu = forge::ensure_cpu(out);
        return std::vector<float>(static_cast<const float*>(out_cpu->data()),
                                  static_cast<const float*>(out_cpu->data()) + out_cpu->numel());
    };

    const std::vector<float> cpu_out = run(DeviceType::CPU);
    const std::vector<float> gpu_out = run(DeviceType::CUDA);

    double dot = 0, na = 0, nb = 0, se = 0, scale = 0;
    int nan_gpu = 0, nan_cpu = 0;
    for (size_t i = 0; i < cpu_out.size(); ++i) {
        if (std::isnan(gpu_out[i]))
            ++nan_gpu;
        if (std::isnan(cpu_out[i]))
            ++nan_cpu;
        dot += (double)gpu_out[i] * cpu_out[i];
        na += (double)gpu_out[i] * gpu_out[i];
        nb += (double)cpu_out[i] * cpu_out[i];
        const double d = (double)gpu_out[i] - cpu_out[i];
        se += d * d;
        scale = std::max(scale, std::fabs((double)cpu_out[i]));
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    // Relative RMS: robust against single outlier elements, unlike max-abs on
    // random weights. CPU dequant-on-the-fly vs Q8_1 activation quantization is
    // not bit-exact but must agree closely. The old sum-before-activation
    // composition scores cos ~ 0.0 here (verified by reverting the fix).
    const double rel_rms = std::sqrt(se / std::max(nb, 1e-30));
    const bool ok = (nan_gpu == 0 && nan_cpu == 0) && (cos >= 0.99) && (rel_rms <= 5e-2);

    printf("  seq=%-2d k=%-2d/%-3d %-4s/%-4s cos=%.6f rel_rms=%.3e scale=%.3e nan=%d/%d  %s\n",
           seq_len, n_expert_used, n_expert, gate_dt == DataType::Q4_K ? "Q4K" : "Q4_0",
           down_dt == DataType::Q5_0 ? "Q5_0" : (down_dt == DataType::Q6_K ? "Q6_K" : "Q4_0"), cos,
           rel_rms, scale, nan_gpu, nan_cpu, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }
    std::mt19937 rng(20260911);

    bool all_ok = true;
    // Mellum's routing (8 of many) in decode and prefill-chunk shapes, plus the
    // top-1 / small-top-k edges and the K-quant dtype families.
    //
    // NOTE: Q8_0 down weights are deliberately NOT covered here. The CPU Q8_0
    // GEMV (gemv_q8_0_maddubs_transB_avx2) computes via _mm256_maddubs_epi16,
    // whose 16-bit pair sums saturate once both the full-range int8 weights and
    // the activations are large (~15-20% RMS error on this synthetic data,
    // measured; Q5_0/Q4_0/Q4_K are safe because their per-element weight range
    // keeps the pair sums far from int16 max). That is a pre-existing CPU-side
    // accuracy defect orthogonal to the GPU path under test here, so including
    // it would only make this oracle compare against a degraded reference.
    all_ok &= run_case(256, 256, 16, 8, DataType::Q4_K, DataType::Q5_0, 1, rng);
    all_ok &= run_case(256, 256, 16, 8, DataType::Q4_K, DataType::Q5_0, 4, rng);
    all_ok &= run_case(256, 256, 16, 2, DataType::Q4_K, DataType::Q5_0, 3, rng);
    all_ok &= run_case(256, 256, 16, 1, DataType::Q4_K, DataType::Q5_0, 1, rng);
    all_ok &= run_case(256, 256, 16, 8, DataType::Q4_K, DataType::Q6_K, 2, rng);
    all_ok &= run_case(256, 256, 8, 4, DataType::Q4_0, DataType::Q4_0, 2, rng);

    printf("%s\n", all_ok ? "mellum-moe-parity: PASS" : "mellum-moe-parity: FAIL");
    return all_ok ? 0 : 1;
}
