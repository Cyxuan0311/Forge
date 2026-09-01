#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "forge/fp8_utils.h"

using namespace forge;

// Exact replica of the quantize kernel in cuda_quant.cu
__global__ void q_kern(const float* src, uint8_t* dst, int rows, int cols) {
    int row = blockIdx.x;
    int i = threadIdx.x;
    if (row >= rows)
        return;
    for (; i < cols; i += blockDim.x)
        dst[row * cols + i] = fp32_to_fp8_e4m3(src[row * cols + i]);
}

int main() {
    int rows = 8, cols = 64;
    int N = rows * cols;
    std::vector<float> h(N);
    srand(12345);
    for (int i = 0; i < N; ++i)
        h[i] = ((float)rand() / RAND_MAX - 0.5f) * 12.0f;  // [-6,6]

    float* d_src;
    uint8_t* d_dst;
    cudaMalloc(&d_src, N * 4);
    cudaMalloc(&d_dst, N);
    cudaMemcpy(d_src, h.data(), N * 4, cudaMemcpyHostToDevice);
    q_kern<<<rows, 256>>>(d_src, d_dst, rows, cols);
    cudaDeviceSynchronize();

    std::vector<uint8_t> out(N);
    cudaMemcpy(out.data(), d_dst, N, cudaMemcpyDeviceToHost);

    double maxe = 0, sume = 0;
    for (int i = 0; i < N; ++i) {
        float r = fp8_e4m3_to_fp32(out[i]);
        double e = fabs((double)r - (double)h[i]);
        if (e > maxe)
            maxe = e;
        sume += e;
    }
    printf("quantize-kernel roundtrip: N=%d max_abs_err=%.6f mean_abs_err=%.6f\n", N, maxe,
           sume / N);
    // reference: what the CPU conversion gives (should match kernel exactly)
    double ref_max = 0;
    for (int i = 0; i < N; ++i) {
        uint8_t b = fp32_to_fp8_e4m3(h[i]);
        float r = fp8_e4m3_to_fp32(b);
        double e = fabs((double)r - (double)h[i]);
        if (e > ref_max)
            ref_max = e;
    }
    printf("cpu-conversion reference max_abs_err=%.6f\n", ref_max);
    return 0;
}
