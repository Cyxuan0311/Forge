#include "forge/inference/layers/ffn_executor.h"

#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

namespace forge {

bool FfnExecutor::residual_fused(const ModelConfig& cfg, const LayerWeights& lw, int seq_len,
                                 DeviceType dev) {
    if (cfg.ffn_type != FFNType::SiLUGated) return false;
    auto w2_dtype = lw.w2()->dtype();
    if (dev == DeviceType::CUDA && seq_len == 1) {
        return w2_dtype == DataType::Q4_0 || w2_dtype == DataType::Q4_K ||
               w2_dtype == DataType::Q5_K || w2_dtype == DataType::Q6_K;
    }
    if (dev == DeviceType::CPU && seq_len == 1) {
        return w2_dtype == DataType::Q4_0 || w2_dtype == DataType::Q4_1 ||
               w2_dtype == DataType::Q4_K || w2_dtype == DataType::Q6_K;
    }
    return false;
}

TensorPtr FfnExecutor::apply(const TensorPtr& x, const TensorPtr& residual, const ModelConfig& cfg,
                             const LayerWeights& lw, int seq_len, DeviceType dev) {
    switch (cfg.ffn_type) {
        case FFNType::SiLUGated: {
            TensorPtr ffn_mid;
            {
                PERF_SCOPE("layer/ffn_up");
                // Fused CUDA Q4_0 gate+up
                if (dev == DeviceType::CUDA && lw.w1()->dtype() == DataType::Q4_0 &&
                    lw.w3()->dtype() == DataType::Q4_0) {
                    ffn_mid = ops::ffn_up_fused(x, lw.w1(), lw.w3(), cfg.intermediate_dim);
                }
                // Fused CUDA Q3_K gate+up (shared Q8_1 quantization of x)
                else if (dev == DeviceType::CUDA && seq_len == 1 &&
                         lw.w1()->dtype() == DataType::Q3_K && lw.w3()->dtype() == DataType::Q3_K) {
                    ffn_mid = ops::ffn_up_fused(x, lw.w1(), lw.w3(), cfg.intermediate_dim);
                }
                // Fused CUDA Q5_K gate+up (shared x read)
                else if (dev == DeviceType::CUDA && seq_len == 1 &&
                         lw.w1()->dtype() == DataType::Q5_K && lw.w3()->dtype() == DataType::Q5_K) {
                    ffn_mid = ops::ffn_up_fused(x, lw.w1(), lw.w3(), cfg.intermediate_dim);
                }
                // Fused CUDA Q4_K gate+up (shared Q8_1 quantization of x)
                else if (dev == DeviceType::CUDA && seq_len == 1 &&
                         lw.w1()->dtype() == DataType::Q4_K && lw.w3()->dtype() == DataType::Q4_K) {
                    ffn_mid = ops::ffn_up_fused(x, lw.w1(), lw.w3(), cfg.intermediate_dim);
                }
                // Fused CUDA Q3_K gate + Q4_K up (mixed quantization, shared Q8_1)
                else if (dev == DeviceType::CUDA && seq_len == 1 &&
                         lw.w1()->dtype() == DataType::Q3_K && lw.w3()->dtype() == DataType::Q4_K) {
                    ffn_mid = ops::ffn_up_fused(x, lw.w1(), lw.w3(), cfg.intermediate_dim);
                }
                // Fused CUDA Q2_K gate+up (shared Q8_1 + dp4a)
                else if (dev == DeviceType::CUDA && seq_len == 1 &&
                         lw.w1()->dtype() == DataType::Q2_K && lw.w3()->dtype() == DataType::Q2_K) {
                    ffn_mid = ops::ffn_up_fused(x, lw.w1(), lw.w3(), cfg.intermediate_dim);
                }
                // Fused CPU Q4_0 gate+up
                else if (dev == DeviceType::CPU && seq_len == 1 && lw.w1() && lw.w3() &&
                         lw.w1()->dtype() == DataType::Q4_0 && lw.w3()->dtype() == DataType::Q4_0) {
                    ffn_mid = ops::matmul_transB_fused_ffn_up_q4_0(x, lw.w1(), lw.w3());
                }
                // Fused CPU Q4_K gate+up
                else if (dev == DeviceType::CPU && seq_len == 1 && lw.w1() && lw.w3() &&
                         lw.w1()->dtype() == DataType::Q4_K && lw.w3()->dtype() == DataType::Q4_K) {
                    ffn_mid = ops::matmul_transB_fused_ffn_up_q4_k(x, lw.w1(), lw.w3());
                }
                // Fused CPU Q5_K gate+up
                else if (dev == DeviceType::CPU && seq_len == 1 && lw.w1() && lw.w3() &&
                         lw.w1()->dtype() == DataType::Q5_K && lw.w3()->dtype() == DataType::Q5_K) {
                    ffn_mid = ops::matmul_transB_fused_ffn_up_q5_k(x, lw.w1(), lw.w3());
                }
                // Fused CPU Q3_K gate+up
                else if (dev == DeviceType::CPU && seq_len == 1 && lw.w1() && lw.w3() &&
                         lw.w1()->dtype() == DataType::Q3_K && lw.w3()->dtype() == DataType::Q3_K) {
                    ffn_mid = ops::matmul_transB_fused_ffn_up_q3_k(x, lw.w1(), lw.w3());
                }
                // Fused CPU Q2_K gate+up
                else if (dev == DeviceType::CPU && seq_len == 1 && lw.w1() && lw.w3() &&
                         lw.w1()->dtype() == DataType::Q2_K && lw.w3()->dtype() == DataType::Q2_K) {
                    ffn_mid = ops::matmul_transB_fused_ffn_up_q2_k(x, lw.w1(), lw.w3());
                } else {
                    auto gate = ops::matmul_transB(x, lw.w1());
                    auto up = ops::matmul_transB(x, lw.w3());
                    ffn_mid = ops::silu_multiply(gate, up);
                }
            }

            TensorPtr ffn_out;
            {
                PERF_SCOPE("layer/ffn_down");
                // Fused down_proj + residual add for decode (M=1)
                if (dev == DeviceType::CUDA && seq_len == 1) {
                    int K_down = static_cast<int>(lw.w2()->shape()[1]);
                    int N_down = static_cast<int>(lw.w2()->shape()[0]);
                    ffn_out = std::make_shared<Tensor>(DataType::FP32,
                                                       std::vector<int64_t>{1, N_down},
                                                       DeviceType::CUDA);
#ifdef USE_CUDA
                    auto w2_dtype = lw.w2()->dtype();
                    if (w2_dtype == DataType::Q4_0) {
                        cuda::launch_ffn_down_fused_q4_0(
                            static_cast<const float*>(ffn_mid->data()), lw.w2()->data(),
                            static_cast<const float*>(residual->data()),
                            static_cast<float*>(ffn_out->data()), K_down, N_down);
                    } else if (w2_dtype == DataType::Q4_K) {
                        cuda::launch_ffn_down_fused_q4_k_q8_1(
                            static_cast<const float*>(ffn_mid->data()), lw.w2()->data(),
                            static_cast<const float*>(residual->data()),
                            static_cast<float*>(ffn_out->data()), K_down, N_down);
                    } else if (w2_dtype == DataType::Q5_K) {
                        cuda::launch_ffn_down_fused_q5_k(
                            static_cast<const float*>(ffn_mid->data()), lw.w2()->data(),
                            static_cast<const float*>(residual->data()),
                            static_cast<float*>(ffn_out->data()), K_down, N_down);
                    } else if (w2_dtype == DataType::Q6_K) {
                        cuda::launch_ffn_down_fused_q6_k(
                            static_cast<const float*>(ffn_mid->data()), lw.w2()->data(),
                            static_cast<const float*>(residual->data()),
                            static_cast<float*>(ffn_out->data()), K_down, N_down);
                    } else {
                        ffn_out = ops::matmul_transB(ffn_mid, lw.w2());
                    }
#endif
                } else if (dev == DeviceType::CPU && seq_len == 1 &&
                           lw.w2()->dtype() == DataType::Q4_0) {
                    ffn_out = ops::matmul_transB_fused_ffn_down_residual_q4_0(ffn_mid, lw.w2(),
                                                                              residual);
                } else if (dev == DeviceType::CPU && seq_len == 1 &&
                           lw.w2()->dtype() == DataType::Q4_1) {
                    ffn_out = ops::matmul_transB_fused_ffn_down_residual_q4_1(ffn_mid, lw.w2(),
                                                                              residual);
                } else if (dev == DeviceType::CPU && seq_len == 1 &&
                           lw.w2()->dtype() == DataType::Q6_K) {
                    ffn_out = ops::matmul_transB_fused_ffn_down_residual_q6_k(ffn_mid, lw.w2(),
                                                                              residual);
                } else if (dev == DeviceType::CPU && seq_len == 1 &&
                           lw.w2()->dtype() == DataType::Q4_K) {
                    ffn_out = ops::matmul_transB_fused_ffn_down_residual_q4_k(ffn_mid, lw.w2(),
                                                                              residual);
                } else {
                    ffn_out = ops::matmul_transB(ffn_mid, lw.w2());
                }
            }
            return ffn_out;
        }

        case FFNType::GeGLU: {
            auto gate = ops::matmul_transB(x, lw.w1());
            auto up = ops::matmul_transB(x, lw.w3());
            auto gated = ops::gelu_multiply(gate, up);
            return ops::matmul_transB(gated, lw.w2());
        }

        case FFNType::SimpleGELU: {
            auto up = ops::matmul_transB(x, lw.w3());
            auto activated = ops::gelu(up);
            return ops::matmul_transB(activated, lw.w2());
        }

        case FFNType::MoE:
            // MoE is not handled here; the architectures served by GenericEngine
            // (Llama/Gemma/Gemma2/Falcon) never use it.
            throw std::runtime_error("FfnExecutor: MoE FFN not supported; use Gemma4Engine");

        default:
            throw std::runtime_error("FfnExecutor: unknown FFN type");
    }
}

}  // namespace forge
