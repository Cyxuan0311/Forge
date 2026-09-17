// Q3_K / Q2_K decode (M == 1) GEMV — standalone bandwidth + correctness audit.
//
// Mirrors tests/test_fp8_gemv_cuda.cu but exercises the production CUDA GEMV
// kernels for the K-quantized weights:
//   forge::cuda::launch_gemv_q3_k_smem  (Q3_K, 110B/256E -> 3.4375 BPW)
//   forge::cuda::launch_gemv_q2_k_q8_1 (Q2_K,  84B/256E -> 2.625  BPW)
//
// Contract under test:
//   M == 1, x: 1xK fp32 (quantized to Q8_1 internally), W: Q*_K weight in
//   Forge's native row-major [N, K] form, out[n] = sum_k dequant(W[n,k]) * x[k]
//   (the Q8_1 quantization of x introduces ~8-bit error; we check cosine only).
//
// Why random Q*_K bytes (no quantizer): the golden is produced by
// forge::ops::dequantize_q*_k_row on CPU from the SAME bytes the kernel reads,
// so the benchmark validates the kernel's dequant against the reference AND
// measures the true DRAM bandwidth of the Q*_K access pattern. We only force
// the per-block `d` scale to 1.0 fp16 (and dmin=0 for Q2_K) so random scales
// never produce NaN/Inf in the golden.
//
// DRAM math: effective_BW = N * K * BPW/8 / time. Q3_K is 3.4375 BPW, so for
// 2560x151936 the theoretical DRAM floor at 192 GB/s is ~0.87 ms; anything
// above that is kernel/MLP overhead, not weight traffic. Use this bench to
// measure the before/after of bandwidth tuning (e.g. the __ldg change in
// cuda_gemv_common.cuh).

#include <cuda_runtime.h>
#include <cuda_fp16.h>  // device half intrinsics (__float2half, __half2float, ...)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "cuda/cuda_gemv.h"      // forge::cuda::launch_gemv_q3_k_smem / _q2_k_q8_1
#include "cpu/common/dequant.h"  // forge::ops::dequantize_q3_k_row / _q2_k_row

using namespace forge;

static bool device_ok() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
    // K-quantized GEMV here targets Ada (sm_89+); skip gracefully otherwise.
    return p.major > 8 || (p.major == 8 && p.minor >= 9);
}

static double time_kernel(const std::function<void()>& f, int iters) {
    f();  // warmup (also forces lazy scratch_pool cudaMalloc)
    cudaStreamSynchronize(0);
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s, 0);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(e, 0);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms / iters;
}

// Fill random Q*_K bytes; keep per-block scale finite (force d = 1.0 fp16).
static void fill_qk(std::vector<uint8_t>& buf, int blocks_per_row, int N, std::mt19937& rng,
                    int BLOCK_BYTES) {
    std::uniform_int_distribution<int> u8(0, 255);
    const int d_off = (BLOCK_BYTES == 110) ? 108 : 80;  // Q3_K d@108, Q2_K d@80
    for (int r = 0; r < N; ++r) {
        uint8_t* row = buf.data() + (size_t)r * blocks_per_row * BLOCK_BYTES;
        for (int b = 0; b < blocks_per_row; ++b) {
            uint8_t* blk = row + b * BLOCK_BYTES;
            for (int j = 0; j < BLOCK_BYTES; ++j) blk[j] = (uint8_t)u8(rng);
            blk[d_off + 0] = 0x00;  // fp16 1.0 little-endian
            blk[d_off + 1] = 0x3C;
            if (BLOCK_BYTES == 84) {  // Q2_K dmin@82 -> 0
                blk[82] = 0x00;
                blk[83] = 0x00;
            }
        }
    }
}

// ============================================================================
// PROTOTYPE: Q3_K GEMV with cp.async per-lane async staging. NOT production.
//
// Hypothesis: the production kernel issues ~12 scattered global loads per Q3_K
// block (across the 16 iqs steps). Each is a small misaligned read that wastes
// DRAM sectors and is issued serially with data deps (low MLP). Here every lane
// issues ONE async copy (cp.async) of its whole 110-byte block into a private
// shared slot, then computes the PREVIOUS block from shared. There is NO
// cross-lane barrier: cp.async.wait_all is per-thread, so each lane waits only
// its own copy — preserving the 32 independent load streams that the earlier
// synchronous shared-mem rewrite destroyed (that one regressed 36% via a
// __syncthreads barrier). Q8_1 stays in L2-cached global (only ~152 KB, shared
// across all rows). Correctness is validated against dequantize_q3_k_row.
// ============================================================================
namespace bench_async {

#define BA_QK_K 256
#define BA_QI8_1 8
#define BA_QR3_K 4
#define BA_QI3_K (BA_QK_K / (4 * BA_QR3_K))  // 16

struct block_q8_1_gemv {
    half2 ds;
    int8_t qs[32];
};

static __device__ __forceinline__ int ba_get_int_b4(const void* x, const int& i32) {
    return ((const int*)x)[i32];
}
static __device__ __forceinline__ int ba_get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}

// Exact copy of production vec_dot_q3_K_q8_1 (cuda_gemv.cu), reading the Q3_K
// block from `vbq` — which we point at shared memory in this prototype.
// forge_dp4a -> __dp4a to avoid the forge::cuda namespace in this standalone bench.
static __device__ __forceinline__ float ba_vec_dot_q3_K_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    const uint8_t* bq3_K = (const uint8_t*)vbq + kbx * 110;
    const int bq8_offset = BA_QR3_K * (iqs / (BA_QI3_K / 2));
    const int scale_offset = iqs - iqs % BA_QI8_1 + (iqs % BA_QI8_1) / (BA_QI8_1 / 2);
    uint16_t d_bits;
    memcpy(&d_bits, bq3_K + 108, 2);
    const float d = __half2float(reinterpret_cast<const __half&>(d_bits));
    const int vl = ba_get_int_b2(bq3_K + 32, iqs);
    const int vh = ~ba_get_int_b2(bq3_K, iqs % (BA_QI3_K / 2)) >> bq8_offset;
    int    u[BA_QR3_K];
    float d8[BA_QR3_K];
    #pragma unroll
    for (int i = 0; i < BA_QR3_K; ++i) {
        u[i]  = ba_get_int_b4(bq8_1[bq8_offset + i].qs, iqs % BA_QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    float sumf = 0.0f;
    #pragma unroll
    for (int i = 0; i < BA_QR3_K; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low  = isc % (BA_QK_K / 32);
        const int sc_shift_low  = 4 * (isc / (BA_QK_K / 32));
        const int sc_low   = (bq3_K[96 + isc_low] >> sc_shift_low) & 0xF;
        const int isc_high = isc % (BA_QK_K / 64);
        const int sc_shift_high = 2 * (isc / (BA_QK_K / 64));
        const int sc_high  = ((bq3_K[96 + BA_QK_K / 32 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__global__ void ba_quantize_q8_1_kernel(const float* __restrict__ x,
                                        block_q8_1_gemv* __restrict__ q8, int K) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = (K + 31) / 32;
    if (bi >= num_blocks) return;
    int base = bi * 32;
    int end = min(base + 32, K);
    float amax = 0.0f;
    for (int j = base; j < end; ++j) {
        float ax = fabsf(x[j]);
        if (ax > amax) amax = ax;
    }
    float d_val = (amax > 1e-10f) ? (amax / 127.0f) : (1.0f / 127.0f);
    float inv_d = (d_val > 0.0f) ? (1.0f / d_val) : 0.0f;
    block_q8_1_gemv& blk = q8[bi];
    blk.ds = __halves2half2(__float2half(d_val), __float2half(0.0f));
    for (int j = 0; j < end - base; ++j) {
        float v = x[base + j] * inv_d;
        v = fminf(fmaxf(v, -128.0f), 127.0f);
        blk.qs[j] = static_cast<int8_t>(__float2int_rn(v));
    }
}

// Per-lane cp.async staging. STAGE=2 double-buffers: issue block b into
// slot b%2 while computing block b-1 from slot (b-1)%2 (issued previous iter).
template <int STAGE>
__global__ void ba_gemv_q3_k_async_kernel(
    const block_q8_1_gemv* __restrict__ x_q8,
    const uint8_t* __restrict__ q_weight,
    float* __restrict__ out, int K, int N, int src_stride)
{
    constexpr int Q3K_BE = 256, SLOT = 112;
    int num_blocks_row = (K + Q3K_BE - 1) / Q3K_BE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int warp_in_block = threadIdx.x / 32;
    if (warp_id >= N) return;

    const uint8_t* row_ptr = q_weight + (size_t)warp_id * num_blocks_row * src_stride;
    extern __shared__ uint8_t smem[];
    // Single-buffer: one 112B slot per lane; per-warp stride = 32*SLOT.
    uint8_t* mybuf = smem + (size_t)warp_in_block * (32 * SLOT) + (size_t)lane * SLOT;

    const int q8_stride = BA_QR3_K * BA_QI3_K / BA_QI8_1;  // 8
    float sum = 0.0f;
    int blocks_per_thread = (num_blocks_row + 31) / 32;

    // Single-buffer per-lane staging: copy this lane's Q3_K block into its own
    // shared slot, then compute from shared. No cross-lane barrier, no overlap.
    // This is the cp.async equivalent minus the async pipeline (used to validate
    // the staging hypothesis cleanly; cp.async itself is blocked by Q3_K's
    // 110B (misaligned) block stride — see notes in main).
    for (int b = 0; b < blocks_per_thread; ++b) {
        int bi = b * 32 + lane;
        if (bi < num_blocks_row) {
            const uint8_t* g = row_ptr + (size_t)bi * src_stride;
            uint8_t* d = mybuf;  // one 112B slot per lane
            #pragma unroll
            for (int c = 0; c < 28; ++c) {
                ((int*)(d + c * 4))[0] = ((const int*)(g + c * 4))[0];
            }
            const block_q8_1_gemv* row_q8 = x_q8 + (size_t)bi * q8_stride;
            #pragma unroll
            for (int iqs = 0; iqs < BA_QI3_K; ++iqs) {
                sum += ba_vec_dot_q3_K_q8_1(d, row_q8, 0, iqs);
            }
        }
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    if (lane == 0) out[warp_id] = sum;
}

static uint8_t* g_q8 = nullptr;
static size_t g_q8_cap = 0;

void ba_launch(const float* x, const uint8_t* w, float* out, int K, int N,
               cudaStream_t stream, int src_stride) {
    int num_q8 = (K + 31) / 32;
    size_t need = (size_t)num_q8 * sizeof(block_q8_1_gemv);
    if (need > g_q8_cap) {
        if (g_q8) cudaFree(g_q8);
        cudaMalloc(&g_q8, need + 256);
        g_q8_cap = need + 256;
        cudaMemset(g_q8, 0, g_q8_cap);  // zero pad so the last partial block's Q8_1 over-read is benign
    }
    block_q8_1_gemv* dq8 = reinterpret_cast<block_q8_1_gemv*>(g_q8);
    int q8_threads = 256;
    int q8_blocks = (num_q8 + q8_threads - 1) / q8_threads;
    ba_quantize_q8_1_kernel<<<q8_blocks, q8_threads, 0, stream>>>(x, dq8, K);

    int warps_per_block = 8;  // match production occupancy (32 warps/SM)
    int threads = warps_per_block * 32;
    int grid = (N + warps_per_block - 1) / warps_per_block;
    int smem = warps_per_block * 32 * 112;  // single-buffer: 28KB/block (< 48KB default)
    ba_gemv_q3_k_async_kernel<2><<<grid, threads, smem, stream>>>(dq8, w, out, K, N, src_stride);
    cudaError_t ek = cudaGetLastError();
    if (ek != cudaSuccess)
        printf("  [async launch err: %s (grid=%d threads=%d smem=%d)]\n",
               cudaGetErrorString(ek), grid, threads, smem);
}

}  // namespace bench_async

int main(int argc, char** argv) {
    if (!device_ok()) {
        printf("[q3k-gemv] SKIP: no Ada (sm_89+) GPU present\n");
        return 0;
    }
    const int iters = (argc > 1) ? std::atoi(argv[1]) : 200;

    // Decode shapes: (K, N). The 151936x2560 pair is the measured-baseline shape
    // from the Q3_K decode audit (was 1.935 ms / 86.4 GB/s on RTX 4050 Laptop).
    std::vector<std::pair<int, int>> shapes = {
        {4096, 4096},
        {151936, 2560},
        {3584, 18944},    // FFN gate (Qwen2.5-7B-ish)
        {18944, 3584},    // FFN down
        {11008, 4096},    // FP16-parity mixed-width check
    };

    auto run = [&](const char* kname, float bpw,
                   const std::function<void(const float*, const uint8_t*, float*, int, int)>& launcher,
                   void (*deq)(const uint8_t*, float*, int, int)) {
        for (auto& [K, N] : shapes) {
            const int QK_K = 256;
            int blocks_per_row = (K + QK_K - 1) / QK_K;
            const int BLOCK_BYTES = (bpw < 3.0f) ? 84 : 110;  // Q2_K vs Q3_K
            size_t qw_bytes = (size_t)N * blocks_per_row * BLOCK_BYTES;

            std::mt19937 rng(1234 + K * 7 + N);
            std::vector<uint8_t> hq(qw_bytes);
            fill_qk(hq, blocks_per_row, N, rng, BLOCK_BYTES);

            std::vector<float> hx(K);
            std::uniform_real_distribution<float> uf(-1.f, 1.f);
            for (int i = 0; i < K; ++i) hx[i] = uf(rng);

            // Golden: dequant weights -> FP32, then GEMV with FP32 x.
            std::vector<float> hW((size_t)N * K);
            for (int r = 0; r < N; ++r)
                deq(hq.data() + (size_t)r * blocks_per_row * BLOCK_BYTES,
                    hW.data() + (size_t)r * K, K, 0);
            std::vector<float> hgold(N, 0.f);
            for (int n = 0; n < N; ++n)
                for (int k = 0; k < K; ++k)
                    hgold[n] += hW[(size_t)n * K + k] * hx[k];

            // +64 pad: the async cp.async prototype reads 112B/block (2B past the
            // 110B block into the next 16B chunk); the pad keeps that over-read
            // in-bounds for the last block of the last row. Production path does
            // not over-read, so the pad is harmless dead space.
            uint8_t* dq = nullptr;
            float *dx = nullptr, *dout = nullptr;
            cudaMalloc(&dq, qw_bytes + 64);
            cudaMalloc(&dx, (size_t)K * sizeof(float));
            cudaMalloc(&dout, (size_t)N * sizeof(float));
            cudaMemcpy(dq, hq.data(), qw_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(dx, hx.data(), (size_t)K * sizeof(float), cudaMemcpyHostToDevice);

            auto launch = [&]() { launcher(dx, dq, dout, K, N); };
            double ms = time_kernel(launch, iters);

            std::vector<float> hout(N);
            cudaMemcpy(hout.data(), dout, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);

            double dot = 0, ng = 0, nt = 0, maxerr = 0;
            bool finite = true;
            for (int n = 0; n < N; ++n) {
                if (!std::isfinite(hout[n])) finite = false;
                dot += hout[n] * hgold[n];
                ng += hout[n] * hout[n];
                nt += hgold[n] * hgold[n];
                double e = std::fabs(hout[n] - hgold[n]);
                if (e > maxerr) maxerr = e;
            }
            double cosine = (ng > 0 && nt > 0) ? (dot / (std::sqrt(ng) * std::sqrt(nt))) : 0.0;
            double gbytes = (double)N * K * bpw / 8.0;
            double bw = gbytes / (ms / 1000.0) / 1e9;

            printf("  %-5s K=%7d N=%5d : us/tok=%8.3f  BW=%6.1f GB/s  BPW=%.2f  "
                   "maxerr=%9.3e  cos=%f%s\n",
                   kname, K, N, ms * 1000.0, bw, bpw, maxerr, cosine,
                   finite ? "" : "  **NaN/Inf**");

            cudaFree(dq);
            cudaFree(dx);
            cudaFree(dout);
        }
    };

    printf("=== Q3_K GEMV (M==1) ===\n");
    run("Q3_K", 3.4375f,
        [](const float* x, const uint8_t* w, float* o, int K, int N) {
            forge::cuda::launch_gemv_q3_k_smem(x, w, o, K, N, 0);
        },
        forge::ops::dequantize_q3_k_row);

    printf("\n=== Q3_K GEMV async cp.async, ALIGNED 112B stride (prototype) ===\n");
    printf("     (re-packs 110B blocks -> 112B/16-aligned once, outside timing;\n");
    printf("      isolates whether cp.async helps IF the Q3_K layout were aligned)\n");
    {
        auto run_async = [&](const char* kname, float bpw) {
            for (auto& [K, N] : shapes) {
                const int QK_K = 256;
                int nbr = (K + QK_K - 1) / QK_K;
                size_t qw110 = (size_t)N * nbr * 110;
                size_t qw112 = (size_t)N * nbr * 112;
                std::mt19937 rng(1234 + K * 7 + N);
                std::vector<uint8_t> hq(qw110);
                fill_qk(hq, nbr, N, rng, 110);
                std::vector<uint8_t> hqa(qw112, 0);
                for (int r = 0; r < N; ++r)
                    for (int b = 0; b < nbr; ++b)
                        memcpy(&hqa[((size_t)r * nbr + b) * 112],
                               &hq[((size_t)r * nbr + b) * 110], 110);
                std::vector<float> hx(K);
                std::uniform_real_distribution<float> uf(-1.f, 1.f);
                for (int i = 0; i < K; ++i) hx[i] = uf(rng);
                std::vector<float> hW((size_t)N * K);
                for (int r = 0; r < N; ++r)
                    forge::ops::dequantize_q3_k_row(hq.data() + (size_t)r * nbr * 110,
                                                    hW.data() + (size_t)r * K, K, 0);
                std::vector<float> hgold(N, 0.f);
                for (int n = 0; n < N; ++n)
                    for (int k = 0; k < K; ++k)
                        hgold[n] += hW[(size_t)n * K + k] * hx[k];
                uint8_t* dqa = nullptr;
                float *dx = nullptr, *dout = nullptr;
                cudaMalloc(&dqa, qw112);
                cudaMalloc(&dx, (size_t)K * sizeof(float));
                cudaMalloc(&dout, (size_t)N * sizeof(float));
                cudaMemcpy(dqa, hqa.data(), qw112, cudaMemcpyHostToDevice);
                cudaMemcpy(dx, hx.data(), (size_t)K * sizeof(float), cudaMemcpyHostToDevice);
                auto launch = [&]() { bench_async::ba_launch(dx, dqa, dout, K, N, 0, 112); };
                double ms = time_kernel(launch, iters);
                std::vector<float> hout(N);
                cudaMemcpy(hout.data(), dout, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);
                double dot = 0, ng = 0, nt = 0, maxerr = 0;
                bool finite = true;
                for (int n = 0; n < N; ++n) {
                    if (!std::isfinite(hout[n])) finite = false;
                    dot += hout[n] * hgold[n];
                    ng += hout[n] * hout[n];
                    nt += hgold[n] * hgold[n];
                    double e = std::fabs(hout[n] - hgold[n]);
                    if (e > maxerr) maxerr = e;
                }
                double cosine = (ng > 0 && nt > 0) ? (dot / (std::sqrt(ng) * std::sqrt(nt))) : 0.0;
                double gbytes = (double)N * K * bpw / 8.0;  // logical 110B weight traffic
                double bw = gbytes / (ms / 1000.0) / 1e9;
                printf("  %-5s K=%7d N=%5d : us/tok=%8.3f  BW=%6.1f GB/s  BPW=%.2f  "
                       "maxerr=%9.3e  cos=%f%s\n",
                       kname, K, N, ms * 1000.0, bw, bpw, maxerr, cosine,
                       finite ? "" : "  **NaN/Inf**");
                cudaFree(dqa);
                cudaFree(dx);
                cudaFree(dout);
            }
        };
        run_async("Q3K_A", 3.4375f);
    }

    printf("\n=== Q2_K GEMV (M==1) ===\n");
    run("Q2_K", 2.625f,
        [](const float* x, const uint8_t* w, float* o, int K, int N) {
            forge::cuda::launch_gemv_q2_k_q8_1(x, w, o, K, N, 0);
        },
        forge::ops::dequantize_q2_k_row);

    printf("\n[done] theoretical DRAM floor @192 GB/s for 2560x151936 Q3_K = %.3f ms\n",
           1e3 * ((double)2560 * 151936 * 3.4375 / 8.0) / 192e9);
    return 0;
}
