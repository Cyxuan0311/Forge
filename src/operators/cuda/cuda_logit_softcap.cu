#include <cmath>

#include "cuda_common.h"

namespace forge {
namespace cuda {

__global__ void logit_softcap_kernel(
    float* __restrict__ logits,
    float cap,
    bool apply_softcap,
    const int* __restrict__ suppress_tokens,
    int num_suppress,
    int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Apply softcapping: logit = tanh(logit / cap) * cap
    if (idx < vocab_size) {
        if (apply_softcap) {
            logits[idx] = tanhf(logits[idx] / cap) * cap;
        }
    }

    // Suppress tokens (first num_suppress threads handle this)
    if (idx < num_suppress) {
        int tok_id = suppress_tokens[idx];
        if (tok_id >= 0 && tok_id < vocab_size) {
            logits[tok_id] = -INFINITY;
        }
    }
}

void launch_logit_softcap(
    float* logits, float cap, bool apply_softcap,
    const int* suppress_tokens, int num_suppress,
    int vocab_size, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    logit_softcap_kernel<<<blocks, threads, 0, stream>>>(
        logits, cap, apply_softcap, suppress_tokens, num_suppress, vocab_size);
}

}  // namespace cuda
}  // namespace forge
