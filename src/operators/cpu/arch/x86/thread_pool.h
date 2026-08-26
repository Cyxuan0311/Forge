#pragma once

// Persistent pinned CPU thread pool for decode GEMV bursts.
//
// Decode issues hundreds of short quantized-GEMV kernels per step, each of
// which Forge used to run as an OpenMP `parallel for` (fork/join per call).
// On hybrid/thermal-limited machines that per-call fork/join plus thread
// migration measurably slows streaming: a benchmark of the exact same
// ggml_vec_dot_q2_K_q8_K loop ran 0.36 ms under llama.cpp's persistent pinned
// pool vs ~0.96 ms under OpenMP `schedule(dynamic,64)` (2.7x).
//
// This pool mirrors llama.cpp's ggml threadpool:
//   - persistent worker threads (no per-kernel fork/join)
//   - no CPU pinning by default (FORGE_PIN_THREADS=first|last opts in); the
//     OS scheduler places threads, which keeps hybrid P/E parts and VMs
//     (WSL2) from being locked onto slow efficiency cores
//   - ggml mul_mat chunk distribution: thread i starts at chunk i, then grabs
//     chunks via an atomic fetch_add so late work-stealing keeps DRAM streams
//     parallel.
//   - workers spin briefly on a generation counter (fast wake for back-to-back
//     kernels) then fall back to a condition variable (no busy spin while the
//     main thread does serial work between kernels).
//
// The calling thread participates as worker nth-1, exactly like ggml's main
// thread runs the mul_mat loop, so the configured thread count is not
// oversubscribed.
//
// The generation counter is bumped once per run *under the mutex together with
// the done counter reset*, so a worker can never miss a run (a spin-only
// start counter has a creation race: a worker that has not yet recorded its
// "last seen" value when the first run bumps the counter spins forever).

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    include <windows.h>
#else
#    include <pthread.h>
#    if defined(__linux__)
#        include <sched.h>
#    endif
#endif

#if defined(__x86_64__) || defined(__i386__)
#    include <immintrin.h>
#    define FORGE_POOL_PAUSE() _mm_pause()
#else
#    define FORGE_POOL_PAUSE() std::this_thread::yield()
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {
namespace cpu {

class ThreadPool {
public:
    explicit ThreadPool(int nthreads)
        : nth_(nthreads < 1 ? 1 : nthreads) {
        workers_.reserve(static_cast<size_t>(nth_ - 1));
        for (int i = 0; i < nth_ - 1; ++i) {
            workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return nth_; }

    // Run body over [0, n) in chunks of `chunk` rows, ggml mul_mat style.
    // body(i0, i1) must process rows [i0, i1). The calling thread participates
    // as worker (nth_-1).
    void parallel_for(int n, int chunk,
                      const std::function<void(int, int)>& body) {
        if (n <= 0) {
            return;
        }
        if (chunk < 1) {
            chunk = 1;
        }
        const int nchunks = (n + chunk - 1) / chunk;
        body_ = &body;
        n_ = n;
        chunk_ = chunk;
        nchunks_ = nchunks;

        {
            std::lock_guard<std::mutex> lk(mtx_);
            current_chunk_.store(nth_, std::memory_order_relaxed);
            done_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);  // wake workers
        }
        cv_.notify_all();

        // main thread participates as worker (nth_-1)
        int current = nth_ - 1;
        while (current < nchunks_) {
            const int i0 = current * chunk_;
            (*body_)(i0, std::min(n_, i0 + chunk_));
            current = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        }

        // wait until every worker finished its chunks
        std::unique_lock<std::mutex> lk(mtx_);
        cv_done_.wait(lk, [&] {
            return done_.load(std::memory_order_acquire) == nth_ - 1;
        });
    }

private:
    void worker_loop(int ith) {
        pin_to_core(ith);
        int mygen = 0;
        for (;;) {
            // fast path: spin briefly on the generation counter
            int spins = 0;
            int g = gen_.load(std::memory_order_acquire);
            while (g == mygen && !stop_.load(std::memory_order_relaxed)) {
                if (++spins >= kSpinLimit) {
                    break;
                }
                FORGE_POOL_PAUSE();
                g = gen_.load(std::memory_order_acquire);
            }
            if (g != mygen) {
                mygen = g;
                run_chunks(ith);
                continue;
            }
            if (stop_.load(std::memory_order_relaxed)) {
                return;
            }
            // slow path: wait on the condition variable
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [&] {
                return stop_.load(std::memory_order_relaxed) ||
                       gen_.load(std::memory_order_relaxed) != mygen;
            });
            if (stop_.load(std::memory_order_relaxed)) {
                return;
            }
            mygen = gen_.load(std::memory_order_relaxed);
            lk.unlock();
            run_chunks(ith);
        }
    }

    void run_chunks(int ith) {
        int current = ith;  // ggml: first chunk = thread id
        while (current < nchunks_) {
            const int i0 = current * chunk_;
            (*body_)(i0, std::min(n_, i0 + chunk_));
            current = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        }
        std::lock_guard<std::mutex> lk(mtx_);
        const int d = done_.fetch_add(1, std::memory_order_release) + 1;
        if (d == nth_ - 1) {
            cv_done_.notify_one();
        }
    }

    void pin_to_core(int ith) {
        // Hard pinning is opt-in via FORGE_PIN_THREADS=first|last.
        //
        // Default is NO pinning (llama.cpp behaviour): leave placement to the
        // OS scheduler. Rationale: on P/E hybrids the previous "last nth
        // allowed CPUs" heuristic parked every worker on efficiency cores
        // (measured ~40% slower per core on an i5-13500H), and inside VMs
        // (WSL2) Linux CPU ids do not reliably map to physical topology, so a
        // fixed mask can fight the host scheduler instead of helping it.
#if defined(__linux__)
        const char* mode = std::getenv("FORGE_PIN_THREADS");
        if (!mode || !*mode) {
            return;
        }
        const bool prefer_last = (std::strcmp(mode, "last") == 0);
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
            return;
        }
        std::vector<int> cpus;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &allowed)) {
                cpus.push_back(c);
            }
        }
        if (cpus.empty()) {
            return;
        }
        // first: the leading `nth_` allowed CPUs (performance cores when the
        // OS enumerates them first); last: the trailing `nth_` CPUs.
        const int n = static_cast<int>(cpus.size());
        int target;
        if (prefer_last) {
            target = n >= nth_ ? n - nth_ + ith : ith % n;
        } else {
            target = n >= nth_ ? ith : ith % n;
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[target], &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#elif defined(_WIN32)
        const char* mode = std::getenv("FORGE_PIN_THREADS");
        if (!mode || !*mode) {
            return;
        }
        const bool prefer_last = (std::strcmp(mode, "last") == 0);
        const int n = static_cast<int>(std::thread::hardware_concurrency());
        const int target =
            n >= nth_ ? (prefer_last ? n - nth_ + ith : ith) : ith;
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << target);
#else
        (void)ith;
#endif
    }

    static constexpr int kSpinLimit = 4096;

    int nth_;
    int n_ = 0;
    int chunk_ = 1;
    int nchunks_ = 0;
    const std::function<void(int, int)>* body_ = nullptr;
    std::vector<std::thread> workers_;
    std::atomic<int> current_chunk_{0};
    std::atomic<int> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable cv_done_;
};

// Process-wide pool, lazily created with the current OpenMP thread count
// (the engines call omp_set_num_threads(n_threads) before forwarding, so the
// first decode GEMV sees the configured count). Re-created only if the count
// changes between calls.
inline ThreadPool& cpu_thread_pool() {
#ifdef _OPENMP
    int want = omp_get_max_threads();
#else
    int want = static_cast<int>(std::thread::hardware_concurrency());
#endif
    if (want < 1) {
        want = 1;
    }
    static ThreadPool* pool = new ThreadPool(want);
    static int pool_nth = want;
    if (pool_nth != want) {
        delete pool;
        pool = new ThreadPool(want);
        pool_nth = want;
    }
    return *pool;
}

// Chunked parallel-for over the process-wide pinned pool (see ThreadPool).
inline void parallel_for_chunks(int n, int chunk,
                                const std::function<void(int, int)>& body) {
    cpu_thread_pool().parallel_for(n, chunk, body);
}

}  // namespace cpu
}  // namespace forge