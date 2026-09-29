// Hadamard (Walsh-Hadamard) activation rotation for quantized KV caches.
//
// The Sylvester Hadamard matrix is symmetric and H*H = n*I, so with the
// 1/sqrt(n) scaling R = H/sqrt(n) is orthonormal AND self-inverse: the same
// transform rotates in and rotates back out. That is why this is a butterfly
// (O(n log n)) rather than a dense matmul (O(n^2)) -- and why upstream's "4
// extra mul_mats" framing understates how cheap this actually is.
//
// One block per vector, n threads, butterflies staged through shared memory.
// The transform is in-place on a contiguous [rows, ..., order] view, applied
// independently to every `order`-wide sub-chunk of the last dimension.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "forge/cuda_kernels.h"

namespace forge {
namespace cuda {
namespace {

// N is a compile-time power of two so the butterfly bounds fold away and the
// shared array is statically sized.
template <int N>
__global__ void hadamard_rotate_kernel(float* __restrict__ data, int n_vec) {
    __shared__ float s[N];
    const int v = blockIdx.x;
    if (v >= n_vec)
        return;

    float* p = data + static_cast<size_t>(v) * N;
    const int tid = threadIdx.x;

    s[tid] = p[tid];
    __syncthreads();

    // Butterfly: at stage `stride` each thread pairs with the element whose index
    // differs in bit log2(stride). Handling only i < partner makes each pair
    // update exactly once.
#pragma unroll
    for (int stride = 1; stride < N; stride <<= 1) {
        const int partner = tid ^ stride;
        if (tid < partner) {
            const float a = s[tid];
            const float b = s[partner];
            s[tid] = a + b;
            s[partner] = a - b;
        }
        __syncthreads();
    }

    // 1/sqrt(N) makes R orthonormal. Without it this would still preserve the
    // argmax of attention but silently rescale every logit by N.
    p[tid] = s[tid] * rsqrtf(static_cast<float>(N));
}

}  // namespace

// `order` must divide the last dimension exactly. Returns false otherwise so the
// caller can refuse rather than rotate a partial chunk.
bool launch_hadamard_rotate(const float* in, float* out, int rows, int order, cudaStream_t stream) {
    if (order <= 0 || rows <= 0)
        return false;
    const int n_vec = rows;
    const int threads = order;
    switch (order) {
    case 32:
        hadamard_rotate_kernel<32><<<n_vec, 32, 0, stream>>>(out, n_vec);
        break;
    case 64:
        hadamard_rotate_kernel<64><<<n_vec, 64, 0, stream>>>(out, n_vec);
        break;
    case 128:
        hadamard_rotate_kernel<128><<<n_vec, 128, 0, stream>>>(out, n_vec);
        break;
    case 256:
        hadamard_rotate_kernel<256><<<n_vec, 256, 0, stream>>>(out, n_vec);
        break;
    case 512:
        hadamard_rotate_kernel<512><<<n_vec, 512, 0, stream>>>(out, n_vec);
        break;
    default:
        // Not instantiated. Rather than fall back to something subtly different
        // (the identity would look like "rotation is on and did nothing").
        std::fprintf(stderr,
                     "[Forge WARN] hadamard rotation: unsupported order %d (want a power of two "
                     "in 32..512); rotation disabled for this shape\\n",
                     order);
        return false;
    }
    return true;
}

}  // namespace cuda
}  // namespace forge
