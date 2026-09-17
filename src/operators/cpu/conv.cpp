// BailingMoE3 depthwise causal conv1d kernel (internal CPU op).
// See conv.h for the state-layout convention.

#include "conv.h"

#include <cmath>
#include <cstring>

#include "forge/inference/tensor_device_utils.h"

namespace forge {
namespace ops {

TensorPtr load_conv_weight(const TensorPtr& w, int k_size, int dim) {
    auto cpu = ensure_cpu(w);
    const float* p = static_cast<const float*>(cpu->data());
    auto out = std::make_shared<Tensor>(DataType::FP32,
                                        std::vector<int64_t>{dim, k_size},
                                        DeviceType::CPU);
    float* o = static_cast<float*>(out->data());
    const auto& sh = cpu->shape();
    if (sh.size() == 3 && sh[1] == 1) {
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < k_size; ++j)
                o[i * k_size + j] = p[i * k_size + j];
    } else {
        std::memcpy(o, p, static_cast<size_t>(dim) * k_size * sizeof(float));
    }
    return out;
}

void conv1d_depthwise_silu(TensorPtr& x, const TensorPtr& weight, float* state,
                           int seq_len, int dim, int k_size) {
    auto xc = ensure_cpu(x);
    const float* w = static_cast<const float*>(ensure_cpu(weight)->data());
    float* d = static_cast<float*>(xc->data());
    const int state_len = k_size - 1;

    for (int s = 0; s < seq_len; ++s) {
        const float* xr = d + static_cast<int64_t>(s) * dim;
        for (int c = 0; c < dim; ++c) {
            // Capture the raw pre-conv projection BEFORE the SiLU overwrite
            // below: the state snapshot must store x_proj (llama.cpp snapshots
            // conv_x, whose tail is the raw projection), otherwise every token
            // after the first convolves SiLU'd history.
            const float raw = xr[c];
            float val = 0.0f;
            for (int k = 0; k < k_size; ++k) {
                float xv;
                if (k < state_len)
                    xv = state[static_cast<int64_t>(k) * dim + c];
                else
                    xv = raw;
                val += xv * w[static_cast<int64_t>(c) * k_size + k];
            }
            // SiLU(x) = x * sigmoid(x)
            val = val / (1.0f + std::exp(-val));
            d[static_cast<int64_t>(s) * dim + c] = val;

            for (int k = 0; k < state_len - 1; ++k)
                state[static_cast<int64_t>(k) * dim + c] =
                    state[static_cast<int64_t>(k + 1) * dim + c];
            state[static_cast<int64_t>(state_len - 1) * dim + c] = raw;
        }
    }
}

}  // namespace ops
}  // namespace forge
