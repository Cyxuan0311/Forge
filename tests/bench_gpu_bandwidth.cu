// Phase 0: what bandwidth can this GPU actually reach?
//
// Why this exists. The optimisation plan for the FP16/BF16 LFM2.5-2.6B models was
// originally justified by "F16 GPU decode runs at 5.5 tok/s while Q8_0 runs at
// 43", i.e. a supposed 7x cliff. Re-measured back to back under identical flags,
// that is not what happens: F16 is ~29.8 tok/s and BF16 ~29.5, and the F16/Q8_0
// gap tracks the data ratio almost exactly.
//
//   Q8_0  2.9 GB / 23.2 ms = 125 GB/s
//   F16   5.4 GB / 33.6 ms = 161 GB/s
//   BF16  5.4 GB / 33.9 ms = 159 GB/s
//
// Note the 2-byte path achieves a HIGHER bandwidth than the 1-byte one, which is
// what a memory-roof-bound decode should look like. The open question is whether
// 161 GB/s is near the achievable peak (no GPU work justified) or well under it (a
// fused FP16 GEMV is worth writing). Arithmetic on tok/s cannot answer that, so
// measure the roof directly.
//
// Method: a large-buffer stream triad and a copy, timed with CUDA events, best of
// N after warmup. The buffer is sized to exceed L2 by a wide margin, otherwise
// this measures cache and not memory. Read-only bandwidth is reported separately
// from read+write, because decode is a pure streaming read: the number to compare
// against 161 GB/s is the read figure.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

#include <cuda_runtime.h>

namespace {

constexpr size_t kBufBytes = 1024ull * 1024 * 1024;  // 1 GiB per buffer

// Stream read: out[i] = in[i] * 2.0f + 1.0f. A dependent read of every element
// with no reuse, so it is bandwidth-bound if anything is.
__global__ void stream_read_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                                   size_t n4) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < n4; i += stride) {
        const float4 v = in[i];
        out[i] = make_float4(v.x * 2.0f + 1.0f, v.y * 2.0f + 1.0f, v.z * 2.0f + 1.0f,
                             v.w * 2.0f + 1.0f);
    }
}

// Read-only: reduce every element and write one value per block. The write
// traffic is negligible, so this isolates the READ roof -- the number decode is
// actually bounded by. The earlier stream kernel wrote as much as it read, so
// "read-only" had to be inferred by halving a figure that also included write
// stalls and read-for-ownership effects.
__global__ void read_only_kernel(const float4* __restrict__ in, float* __restrict__ sink,
                                 size_t n4) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    float acc = 0.0f;
    for (; i < n4; i += stride) {
        const float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // one atomic per block, into a tiny sink
    if (acc == 12345.678f) sink[0] = acc;  // never true; defeats dead-code removal
}

// Device-to-device copy.
__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < n4; i += stride) out[i] = in[i];
}

struct Timing {
    double ms = 0;
    double gbps = 0;
};

template <typename F>
Timing time_it(F launch, size_t n4, int iters) {
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    double best = 1e30;
    for (int it = 0; it < iters + 1; ++it) {  // +1 warmup, discarded
        cudaEventRecord(t0);
        launch();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        if (it > 0) best = std::min(best, static_cast<double>(ms));
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    Timing r;
    r.ms = best;
    r.gbps = (2.0 * kBufBytes) / (best * 1e-3) / 1e9;  // read + write
    return r;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no CUDA device; skipping\n");
        return 0;
    }

    float *d_in = nullptr, *d_out = nullptr;
    const size_t n4 = kBufBytes / sizeof(float4);
    if (cudaMalloc(&d_in, kBufBytes) != cudaSuccess ||
        cudaMalloc(&d_out, kBufBytes) != cudaSuccess) {
        std::printf("allocation of 2 GiB failed; skipping\n");
        return 0;
    }
    cudaMemset(d_in, 0, kBufBytes);
    cudaMemset(d_out, 0, kBufBytes);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("== GPU bandwidth roof ==\n");
    std::printf("device: %s (sm_%d%d, %d MiB VRAM, %.0f MHz mem clock)\n", prop.name, prop.major,
                prop.minor, prop.totalGlobalMem >> 20, prop.memoryClockRate / 1000.0);
    std::printf("buffer: %zu MiB per operand (%.0fx the %d MiB L2)\n", kBufBytes >> 20,
                double(kBufBytes) / double(prop.l2CacheSize), prop.l2CacheSize >> 20);

    // Saturate: enough waves to cover the device, sized from the SM count rather
    // than guessed. A fixed 2048-block grid can leave the memory system idle,
    // which would understate the roof and make a memory-bound decode look like
    // it has headroom when it does not.
    const int threads = 256;
    const int blocks = prop.multiProcessorCount * 16;
    std::printf("grid: %d blocks x %d threads (%d SMs, %zu float4/thread avg)\n", blocks, threads,
                prop.multiProcessorCount, n4 / (size_t(blocks) * threads));

    float* d_sink = nullptr;
    cudaMalloc(&d_sink, sizeof(float) * 4096);
    cudaMemset(d_sink, 0, sizeof(float) * 4096);

    const auto time_launch = [&](const std::function<void()>& launch, int iters) {
        cudaEvent_t t0 = nullptr, t1 = nullptr;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        double best = 1e30;
        for (int it = 0; it < iters + 1; ++it) {
            cudaEventRecord(t0);
            launch();
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms = 0;
            cudaEventElapsedTime(&ms, t0, t1);
            if (it > 0) best = std::min(best, static_cast<double>(ms));
        }
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        return best;
    };

    const double read_only_ms = time_launch(
        [&] {
            read_only_kernel<<<blocks, threads>>>(reinterpret_cast<const float4*>(d_in), d_sink,
                                                 n4);
        },
        7);
    const double copy_ms = time_launch(
        [&] {
            copy_kernel<<<blocks, threads>>>(
                reinterpret_cast<const float4*>(d_in), reinterpret_cast<float4*>(d_out), n4);
        },
        7);
    const double rw_ms = time_launch(
        [&] {
            stream_read_kernel<<<blocks, threads>>>(
                reinterpret_cast<const float4*>(d_in), reinterpret_cast<float4*>(d_out), n4);
        },
        7);

    // Read-only: kBufBytes read, writes negligible.
    const double read_gbps = kBufBytes / (read_only_ms * 1e-3) / 1e9;
    const double copy_gbps = 2.0 * kBufBytes / (copy_ms * 1e-3) / 1e9;
    const double rw_gbps = 2.0 * kBufBytes / (rw_ms * 1e-3) / 1e9;

    std::printf("\n  %-28s %8.2f ms  %7.1f GB/s\n", "read-only (decode's roof)", read_only_ms,
                read_gbps);
    std::printf("  %-28s %8.2f ms  %7.1f GB/s (r+w)\n", "device-to-device copy", copy_ms,
                copy_gbps);
    std::printf("  %-28s %8.2f ms  %7.1f GB/s (r+w)\n", "stream read+write", rw_ms, rw_gbps);

    const double gbps_theo = 2.0 * prop.memoryClockRate * 1000.0 *
                             (prop.memoryBusWidth / 8.0) / 1e9;
    std::printf("\n  %-28s %7.1f GB/s  (%d-bit bus, %.0f MHz, DDR => 2x)\n", "theoretical peak",
                gbps_theo, prop.memoryBusWidth, prop.memoryClockRate / 1000.0);
    std::printf("  %-28s %6.1f%%\n", "measured read / theoretical", 100.0 * read_gbps / gbps_theo);

    // The model-level figure: F16 (5.4 GB of weights) decoding at 29.8 tok/s
    // implies 33.6 ms/token, so 5.4 GB/33.6 ms = 161 GB/s IF every weight is
    // read once per token. Compare that against the measured read roof. If the
    // implied figure exceeds the roof, the traffic model is wrong, not the GPU.
    std::printf("\n-- verdict for the optimisation plan --\n");
    const double kModelGB = 5.4, kTokPerSec = 29.8;
    // GB per token x tokens per second = GB/s. (The earlier form divided by the
    // token period in ms, which mixed ms and s and printed 0.)
    const double implied = kModelGB * kTokPerSec;
    std::printf("  F16 decode implies:   %.0f GB/s  (%.1f GB @ %.1f tok/s, if all weights stream)\n",
                implied, kModelGB, kTokPerSec);
    std::printf("  measured read roof:  %.0f GB/s\n", read_gbps);
    if (implied <= read_gbps) {
        const double eff = 100.0 * implied / read_gbps;
        std::printf("  => decode is at %.0f%% of the read roof.\n", eff);
        if (eff >= 85.0) {
            std::printf("  => AT the roof. Do NOT write a fused FP16 GEMV; nothing to win.\n");
        } else {
            std::printf("  => Below the roof, so a fused FP16 GEMV has real headroom.\n");
        }
    } else {
        const double readable = read_gbps / kTokPerSec;  // GB readable in one token period
        std::printf("  => CONTRADICTION: decode implies %.0f GB/s but the card only reads %.0f GB/s.\n",
                    implied, read_gbps);
        std::printf("     So decode is NOT streaming all %.1f GB per token -- it reads at most\n",
                    kModelGB);
        std::printf("     ~%.1f GB. Either part of the model is not being read per token, or\n",
                    readable);
        std::printf("     it is resident in cache. The traffic model needs re-deriving before\n");
        std::printf("     any kernel is justified.\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
