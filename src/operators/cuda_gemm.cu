#include "cuda_common.h"

namespace forge {
namespace cuda {

template <int T>
__global__ void gemm_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K, bool transB) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;

    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += T) {
        if (row < M && k0 + tx < K)
            As[ty][tx] = A[row * K + k0 + tx];
        else
            As[ty][tx] = 0.0f;

        if (transB) {
            if (col < N && k0 + ty < K)
                Bs[ty][tx] = B[col * K + k0 + ty];
            else
                Bs[ty][tx] = 0.0f;
        } else {
            if (k0 + ty < K && col < N)
                Bs[ty][tx] = B[(k0 + ty) * N + col];
            else
                Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < T; k++)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

constexpr int TILE_SIZE = 16;

void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, bool transB,
                       cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tiled_kernel<TILE_SIZE><<<grid, block, 0, stream>>>(A, B, C, M, N, K, transB);
}

}  // namespace cuda
}  // namespace forge
