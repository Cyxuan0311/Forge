#include "forge/operator_rope.h"
#include "forge/cuda_kernels.h"
#include "forge/perf_profiler.h"
#include <cmath>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

namespace forge {
namespace ops {

TensorPtr rope(const TensorPtr& q, const TensorPtr& k, int64_t pos, float theta) {
    int seq_len = static_cast<int>(q->shape()[0]);
    int num_heads = static_cast<int>(q->shape()[1]);
    int head_dim = static_cast<int>(q->shape()[2]);

    auto q_out = std::make_shared<Tensor>(DataType::FP32, q->shape(), q->device());
    auto k_out = std::make_shared<Tensor>(DataType::FP32, k->shape(), k->device());

    if (q->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_rope_fp32(
            static_cast<const float*>(q->data()),
            static_cast<const float*>(k->data()),
            static_cast<float*>(q_out->data()),
            static_cast<float*>(k_out->data()),
            num_heads, head_dim, seq_len, pos, theta
        );
#endif
    } else {
        PERF_SCOPE("rope/cpu");
        const float* q_data = static_cast<const float*>(q->data());
        const float* k_data = static_cast<const float*>(k->data());
        float* qo = static_cast<float*>(q_out->data());
        float* ko = static_cast<float*>(k_out->data());

        int half_dim = head_dim / 2;
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads; ++h) {
                for (int d = 0; d < half_dim; ++d) {
                    float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                    float angle = (pos + s) * freq;
                    float cos_a = std::cos(angle);
                    float sin_a = std::sin(angle);

                    int idx0 = s * num_heads * head_dim + h * head_dim + d;
                    int idx1 = idx0 + half_dim;

                    float q0 = q_data[idx0], q1 = q_data[idx1];
                    qo[idx0] = q0 * cos_a - q1 * sin_a;
                    qo[idx1] = q0 * sin_a + q1 * cos_a;

                    float k0 = k_data[idx0], k1 = k_data[idx1];
                    ko[idx0] = k0 * cos_a - k1 * sin_a;
                    ko[idx1] = k0 * sin_a + k1 * cos_a;
                }
            }
        }
    }

    return q_out;
}

} // namespace ops
} // namespace forge
