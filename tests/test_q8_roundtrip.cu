// Definitive Q8_0 KV quantize/dequantize round-trip + per-block scale check.
// Localizes whether the CUDA q8_0 quantize kernel writes correct per-block
// scales, or whether the defect lives elsewhere (attention / integration).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "forge/cuda_kernels.h"

using namespace forge::cuda;

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign << 31;
        } else {
            int e = -1;
            uint32_t m = mant;
            while (!(m & 0x400)) {
                m <<= 1;
                --e;
            }
            m &= 0x3FF;
            x = (sign << 31) | ((127 + e - 14) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        x = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        x = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, sizeof(float));
    return f;
}

int main() {
    const int n = 1024;  // kv_dim (MiMo-7B: 8 kv heads * 128 head_dim)
    const int rows = 4;  // simulate a few tokens
    const int BLOCK = 32;
    const int BLOCK_BYTES = 34;
    const int num_blocks = (n + BLOCK - 1) / BLOCK;
    const int row_bytes = num_blocks * BLOCK_BYTES;

    std::vector<float> h_in(n * rows);
    srand(12345);
    for (int i = 0; i < n * rows; ++i)
        h_in[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 10.0f;

    float* d_in;
    cudaMalloc(&d_in, n * rows * sizeof(float));
    uint8_t* d_q;
    cudaMalloc(&d_q, rows * row_bytes);
    float* d_out;
    cudaMalloc(&d_out, n * rows * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), n * rows * sizeof(float), cudaMemcpyHostToDevice);

    launch_quantize_q8_0_matrix(d_in, d_q, rows, n, 0);
    launch_dequant_q8_0_matrix(d_q, d_out, rows, n, 0);
    cudaDeviceSynchronize();

    std::vector<float> h_out(n * rows);
    std::vector<uint8_t> h_q(rows * row_bytes);
    cudaMemcpy(h_out.data(), d_out, n * rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_q.data(), d_q, rows * row_bytes, cudaMemcpyDeviceToHost);

    // 1) Round-trip error over all data.
    double sse = 0, sse_orig = 0;
    float max_abs = 0;
    for (int i = 0; i < n * rows; ++i) {
        float e = h_out[i] - h_in[i];
        sse += (double)e * e;
        sse_orig += (double)h_in[i] * h_in[i];
        if (std::fabs(e) > max_abs)
            max_abs = std::fabs(e);
    }
    double relL2 = std::sqrt(sse / sse_orig);

    // 2) Per-block scale check: does stored scale match true amax/127?
    int bad_blocks = 0;
    float worst_scale_err = 0;
    for (int r = 0; r < rows; ++r) {
        for (int bi = 0; bi < num_blocks; ++bi) {
            int start = bi * BLOCK, end = std::min(start + BLOCK, n);
            float amax = 0;
            int amax_idx = -1;
            for (int i = start; i < end; ++i) {
                float a = std::fabs(h_in[r * n + i]);
                if (a > amax) {
                    amax = a;
                    amax_idx = i;
                }
            }
            float true_d = (amax > 0) ? amax / 127.0f : 1.0f;
            uint16_t sb;
            std::memcpy(&sb, h_q.data() + r * row_bytes + bi * BLOCK_BYTES, 2);
            float stored_d = fp16_to_fp32(sb);
            float stored_amax = stored_d * 127.0f;
            float err = (true_d > 0) ? std::fabs(stored_d - true_d) / true_d : 0.0f;
            if (err > 0.005f) {
                ++bad_blocks;
                if (err > worst_scale_err)
                    worst_scale_err = err;
                if (bad_blocks <= 4) {
                    printf(
                        "  bad blk r=%d bi=%d: host_amax=%.4f stored_amax=%.4f "
                        "host_val@idx=%.4f\n",
                        r, bi, amax, stored_amax, h_in[r * n + amax_idx]);
                }
            }
        }
    }

    printf("Q8_0 roundtrip: max_abs=%.4f relL2=%.4f\n", max_abs, relL2);
    printf("per-block scale: bad_blocks=%d/%d worst_rel_err=%.4f\n", bad_blocks, rows * num_blocks,
           worst_scale_err);

    cudaFree(d_in);
    cudaFree(d_q);
    cudaFree(d_out);

    bool ok = (relL2 < 0.02) && (bad_blocks == 0);
    printf(ok ? "RESULT: PASS\n" : "RESULT: FAIL\n");
    return ok ? 0 : 1;
}
