#include "cuda_common.h"
#include "forge/cuda_kernels.h"

namespace forge {
namespace cuda {

// Fused Gumbel-noise + argmax kernel.
// Each element i gets score = logits[i]/temperature + gumbel_i.
// The Gumbel noise g_i = -log(-log(u_i)) where u_i ~ Uniform(0,1)
// is generated per-element using a hash-based PRNG seeded from
// base_seed and the element index, ensuring reproducibility.
// Block output: (score, idx) pair for the best element in the block.
struct BlockScore {
    float score;
    int32_t idx;
};

__global__ void gumbel_max_kernel(const float* __restrict__ logits,
                                   BlockScore* __restrict__ block_results,
                                   float temperature, uint64_t base_seed,
                                   int vocab_size) {
    extern __shared__ char smem[];
    float* s_score = reinterpret_cast<float*>(smem);
    int32_t* s_idx = reinterpret_cast<int32_t*>(smem + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    float best_score = -1e30f;
    int32_t best_idx = -1;

    for (int i = blockIdx.x * blockDim.x + tid; i < vocab_size; i += blockDim.x * gridDim.x) {
        float logit = logits[i];

        // Hash-based PRNG: generate uniform u in (0,1) from element index
        uint64_t rng = base_seed + (uint64_t)i * 6364136223846793005ULL;
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        rng *= 2685821657736338717ULL;
        // Mantissa bits only → float in [1, 2) → shift to (0, 1)
        union { uint32_t u; float f; } conv;
        conv.u = (static_cast<uint32_t>(rng) & 0x007FFFFFu) | 0x3F800000u;
        float u = conv.f - 1.0f;
        if (u <= 0.0f) u = 1e-37f;
        if (u >= 1.0f) u = 1.0f - 1e-37f;

        float gumbel = -__logf(-__logf(u));
        float score = logit / temperature + gumbel;

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    s_score[tid] = best_score;
    s_idx[tid] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_score[tid + s] > s_score[tid]) {
            s_score[tid] = s_score[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_results[blockIdx.x].score = s_score[0];
        block_results[blockIdx.x].idx = s_idx[0];
    }
}

__global__ void gumbel_final_kernel(const BlockScore* __restrict__ block_results,
                                    int32_t* __restrict__ out_token, int num_blocks) {
    float best_score = -1e30f;
    int32_t best_idx = -1;
    for (int i = 0; i < num_blocks; ++i) {
        if (block_results[i].score > best_score) {
            best_score = block_results[i].score;
            best_idx = block_results[i].idx;
        }
    }
    *out_token = best_idx;
}

void launch_gumbel_sample(const float* logits, int32_t* out_token,
                           float temperature, uint64_t seed,
                           int vocab_size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    if (blocks > 64) blocks = 64;

    size_t shared = threads * (sizeof(float) + sizeof(int32_t));

    // Scratch pool for block results (max 64 × BlockScore = 1024 bytes)
    void* scratch = scratch_pool().ensure(blocks * sizeof(BlockScore));
    auto* block_results = static_cast<BlockScore*>(scratch);

    gumbel_max_kernel<<<blocks, threads, shared, stream>>>(
        logits, block_results, temperature, seed, vocab_size);
    gumbel_final_kernel<<<1, 1, 0, stream>>>(
        block_results, out_token, blocks);
}

} // namespace cuda
} // namespace forge
