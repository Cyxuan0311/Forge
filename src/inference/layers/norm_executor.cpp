#include "forge/inference/layers/norm_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/operators.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {

TensorPtr NormExecutor::apply(const TensorPtr& x, const TensorPtr& weight, const TensorPtr& bias,
                              NormType type, float eps) {
    if (type == NormType::LayerNorm) {
        return ops::layer_norm(x, weight, bias, eps);
    }
    return ops::rms_norm(x, weight, eps);
}

TensorPtr NormExecutor::apply_qk_norm(const TensorPtr& x, const TensorPtr& norm_weight,
                                      int num_heads, int head_dim, float eps, DeviceType dev) {
    int seq_len = static_cast<int>(x->shape()[0]);
    int rows = seq_len * num_heads;

    if (dev == DeviceType::CUDA && x->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        const float* w_ptr = nullptr;
        TensorPtr w_cuda_tmp;
        if (norm_weight->dtype() == DataType::FP32) {
            if (norm_weight->device() == DeviceType::CUDA) {
                w_ptr = static_cast<const float*>(norm_weight->data());
            } else {
                w_cuda_tmp = std::make_shared<Tensor>(DataType::FP32, norm_weight->shape(),
                                                      DeviceType::CUDA);
                w_cuda_tmp->copy_from(*norm_weight);
                w_ptr = static_cast<const float*>(w_cuda_tmp->data());
            }
        }
        if (w_ptr) {
            auto x_out = std::make_shared<Tensor>(DataType::FP32, x->shape(), DeviceType::CUDA);
            cuda::launch_rms_norm(static_cast<const float*>(x->data()), w_ptr,
                                  static_cast<float*>(x_out->data()), rows, head_dim, eps);
            return x_out;
        }
#endif
    }

    // CPU path
    auto x_cpu = x;
    if (x_cpu->device() != DeviceType::CPU) {
        x_cpu = std::make_shared<Tensor>(DataType::FP32, x->shape(), DeviceType::CPU);
        x_cpu->copy_from(*x);
    }

    // Create mutable copy for in-place norm
    auto result = std::make_shared<Tensor>(DataType::FP32, x_cpu->shape(), DeviceType::CPU);
    std::memcpy(result->data(), x_cpu->data(), result->nbytes());

    float* data = static_cast<float*>(result->data());
    std::vector<float> nw(head_dim);
    if (norm_weight->dtype() == DataType::FP32) {
        if (norm_weight->device() == DeviceType::CUDA) {
            auto tmp =
                std::make_shared<Tensor>(DataType::FP32, norm_weight->shape(), DeviceType::CPU);
            tmp->copy_from(*norm_weight);
            std::memcpy(nw.data(), tmp->data(), head_dim * sizeof(float));
        } else {
            std::memcpy(nw.data(), norm_weight->data(), head_dim * sizeof(float));
        }
    } else {
        std::fill(nw.begin(), nw.end(), 1.0f);
    }

#pragma omp parallel for schedule(static) collapse(2) if (seq_len * num_heads > 4)
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            float* head_ptr = data + s * num_heads * head_dim + h * head_dim;
            float norm_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d) norm_sq += head_ptr[d] * head_ptr[d];
            float inv_rms = 1.0f / std::sqrt(norm_sq / head_dim + eps);
            for (int d = 0; d < head_dim; ++d) head_ptr[d] *= inv_rms * nw[d];
        }
    }

    return result;
}

}  // namespace forge
