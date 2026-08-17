// Runtime thread-count auto-detection for CPU inference.
//
// Decode is dominated by short quantized-GEMV bursts (e.g. Q4_0 FFN), where
// the optimal OpenMP thread count depends on the machine's memory subsystem,
// core topology and hyperthreading behaviour. A pure streaming-read probe
// cannot predict decode speed (streaming reads saturate with fewer threads
// than the compute-interleaved GEMVs do), so we run the real fused FFN-up
// kernel over synthetic weights sized to the target model and take the thread
// count with the lowest median burst time. One-time cost is ~200ms, negligible
// vs model load.

#include "forge/threads.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#if defined(_WIN32)
#    include <windows.h>
#else
#    include <dirent.h>
#    if defined(__linux__)
#        include <sched.h>
#    endif
#endif

#include "forge/operator_matmul.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace forge {

static constexpr int kProbeRowsDefault = 14336;  // Llama-3.1-8B FFN intermediate
static constexpr int kProbeKDefault = 4096;      // Llama-3.1-8B hidden dim
static constexpr int kProbeReps = 20;            // bursts per candidate (median)
static constexpr int kProbeRounds = 3;           // rounds per candidate
// Prefer the full physical count when the half-count is not clearly faster
// (>8%). Decode interleaves non-GEMV work (norms, attention, sampling, the
// slower Q6_K output layer) that benefits from more threads, and prefill runs
// at the same count and scales with more threads.
static constexpr double kPreferPhysicalMargin = 0.92;

static double now_seconds() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}

int physical_core_count() {
#if defined(_WIN32)
    // Count physical cores (not logical processors) via the core relation so
    // HT machines behave like the Linux path: candidates stop at logical-2 and
    // never include the oversubscribed full logical count.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::vector<uint8_t> buf(len);
        if (GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                    buf.data()),
                &len)) {
            int cores = 0;
            size_t off = 0;
            while (off < len) {
                auto* hdr =
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                        buf.data() + off);
                if (hdr->Relationship == RelationProcessorCore)
                    ++cores;
                if (hdr->Size == 0)
                    break;
                off += hdr->Size;
            }
            if (cores > 0)
                return cores;
        }
    }
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<int>(si.dwNumberOfProcessors);
#else
    // Linux/macOS/BSD: count unique core ids exposed by sysfs (Linux). On
    // systems without sysfs, fall back to an estimate of physical cores.
    std::vector<int> seen;
    DIR* dir = opendir("/sys/devices/system/cpu");
    if (!dir)
        return 0;
    dirent* e;
    while ((e = readdir(dir)) != nullptr) {
        if (strncmp(e->d_name, "cpu", 3) != 0)
            continue;
        char* end = nullptr;
        long idx = strtol(e->d_name + 3, &end, 10);
        if (*end != '\0' || idx < 0)
            continue;
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/%s/topology/core_id",
                 e->d_name);
        FILE* f = fopen(path, "r");
        if (!f)
            continue;
        int core = -1;
        if (fscanf(f, "%d", &core) == 1 && core >= 0) {
            if (std::find(seen.begin(), seen.end(), core) == seen.end())
                seen.push_back(core);
        }
        fclose(f);
    }
    closedir(dir);
    if (!seen.empty())
        return static_cast<int>(seen.size());
    int logical = static_cast<int>(std::thread::hardware_concurrency());
    return logical > 0 ? (logical + 1) / 2 : 0;
#endif
}

#if defined(__linux__)
// CPUs the process is actually allowed to run on (containers, cgroups,
// taskset). 0 when unavailable.
static int allowed_cpu_count() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = CPU_COUNT(&set);
        return n > 0 ? n : 0;
    }
    return 0;
}
#endif

static double gemv_ms(int threads, TensorPtr& t_in, TensorPtr& t_gate,
                      TensorPtr& t_up) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    // Each matmul in real decode is a fresh short OpenMP burst with other
    // sequential work (norms, attention, sampling) in between. Sustained
    // best-of-N timing hides the per-burst wake-up / HT-ramp cost and
    // mis-ranks oversubscribed counts, so we idle the pool between bursts and
    // report the median burst time per round, taking the best of two rounds to
    // tolerate a single catastrophically-thrashed round.
    double best = 1e30;
    for (int round = 0; round < kProbeRounds; ++round) {
        std::vector<double> bursts;
        for (int r = 0; r < kProbeReps; ++r) {
            double t0 = now_seconds();
            auto out = ops::matmul_transB_fused_ffn_up_q4_0(t_in, t_gate, t_up);
            double dt = now_seconds() - t0;
            volatile float sink = *static_cast<const float*>(out->data());
            (void)sink;
            bursts.push_back(dt);
            double t1 = now_seconds();
            while (now_seconds() - t1 < 1.0e-3) {
            }  // idle the team like decode's non-GEMV work
        }
        std::sort(bursts.begin(), bursts.end());
        best = std::min(best, bursts[bursts.size() / 2]);
    }
    return best;
}

int detect_best_cpu_threads() {
    return detect_best_cpu_threads(kProbeKDefault, kProbeRowsDefault);
}

int detect_best_cpu_threads(int hidden, int intermediate) {
    hidden = std::max(512, std::min(hidden, 16384));
    intermediate = std::max(2048, std::min(intermediate, 32768));

    int logical = static_cast<int>(std::thread::hardware_concurrency());
    if (logical < 1)
        logical = 1;
    int physical = physical_core_count();
    if (physical < 1)
        physical = logical;

#if defined(__linux__)
    // Cap candidate counts by the CPUs this process may actually use
    // (containers / cgroups / taskset) to avoid oversubscription.
    int allowed = allowed_cpu_count();
    if (allowed > 0) {
        logical = std::min(logical, allowed);
        if (physical > allowed)
            physical = allowed;
    }
#endif

    // Candidate set: physical/2, physical. Oversubscribed HT counts (logical-4,
    // logical-2) are excluded: the idle-pooled probe systematically over-ranks
    // them (per-burst HT-ramp cost is hidden between bursts), but real decode
    // loses badly with them — measured ~4x slower on an 8-physical P/E-hybrid
    // machine, where 14 threads (logical-2) thrashed against 8 physical cores.
    std::vector<int> candidates;
    candidates.push_back(physical / 2);
    candidates.push_back(physical);
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    for (int& c : candidates)
        c = std::max(1, std::min(c, logical));

    // Synthetic Q4_0 gate+up weights (row-granular like a real FFN layer),
    // sized to the target model's hidden/intermediate dims. Q4_0 block = 32
    // values / 18 bytes (fp16 scale + 16 packed 4-bit quants).
    constexpr int QK4_0 = 32;
    constexpr int Q4_0_BLOCK = 18;
    int nb = (hidden + QK4_0 - 1) / QK4_0;
    size_t mat_bytes = (size_t)intermediate * nb * Q4_0_BLOCK;
    std::vector<uint8_t> gate(mat_bytes), up(mat_bytes);
    for (size_t i = 0; i < mat_bytes; ++i) {
        gate[i] = static_cast<uint8_t>(i * 31);
        up[i] = static_cast<uint8_t>(i * 17);
    }
    // Force sane fp16 scales so the kernel never hits denormal/NaN timing
    // outliers (values are synthetic; only the timing matters).
    const uint16_t d_half = 0x3800;  // 0.5f in fp16
    for (size_t off = 0; off < mat_bytes; off += Q4_0_BLOCK) {
        std::memcpy(&gate[off], &d_half, 2);
        std::memcpy(&up[off], &d_half, 2);
    }
    std::vector<float> in(hidden, 0.5f);

    TensorPtr t_in = std::make_shared<Tensor>(Tensor::from_buffer(
        in.data(), DataType::FP32, std::vector<int64_t>{1, hidden}));
    TensorPtr t_gate = std::make_shared<Tensor>(Tensor::from_buffer(
        gate.data(), DataType::Q4_0,
        std::vector<int64_t>{(int64_t)intermediate, (int64_t)hidden}));
    TensorPtr t_up = std::make_shared<Tensor>(Tensor::from_buffer(
        up.data(), DataType::Q4_0,
        std::vector<int64_t>{(int64_t)intermediate, (int64_t)hidden}));

    int best_threads = physical;
    double best_ms = 1e30;
    for (int c : candidates) {
        double ms = gemv_ms(c, t_in, t_gate, t_up);
        // Prefer the full physical count unless the half-count is clearly
        // faster (see kPreferPhysicalMargin above).
        if (ms < best_ms * kPreferPhysicalMargin || best_threads < c) {
            best_ms = ms;
            best_threads = c;
        }
    }

#ifdef _OPENMP
    // Restore the logical default; engines set the real count per call from
    // ContextParams.
    omp_set_num_threads(logical);
#endif

    return best_threads;
}

}  // namespace forge