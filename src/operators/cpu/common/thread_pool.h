#pragma once

// Persistent pinned CPU thread pool — Forge's built-in thread scheduler.
//
// This is the single, architecture-agnostic home of the pool. Every backend
// (x86/AVX2, arm64/NEON, ppc64/VSX, generic) and every caller (engines,
// layers, operators, sampler) includes "cpu/common/thread_pool.h", so there is
// exactly one process-wide pool.
//
// It replaces OpenMP's scheduling outright:
//   #pragma omp parallel for                    -> parallel_for_chunks()
//   #pragma omp parallel for + thread index     -> parallel_for_chunks_ex()
//   omp_get_max_threads() / omp_set_num_threads -> get_num_threads() /
//                                                  set_num_threads()
//
// Why: decode issues hundreds of short quantized-GEMV kernels per step. Each
// used to run as an OpenMP `parallel for` (fork/join per call). On
// hybrid/thermal-limited machines that per-call fork/join plus thread
// migration measurably slows streaming: a benchmark of the exact same
// ggml_vec_dot_q2_K_q8_K loop ran 0.36 ms under llama.cpp's persistent pinned
// pool vs ~0.96 ms under OpenMP `schedule(dynamic,64)` (2.7x).
//
// This pool mirrors llama.cpp's ggml threadpool:
//   - persistent worker threads (no per-kernel fork/join)
//   - no CPU pinning by default (FORGE_PIN_THREADS=first|last opts in); the
//     OS scheduler places threads, which keeps hybrid P/E parts and VMs
//     (WSL2) from being locked onto slow efficiency cores
//   - static contiguous partition by default, exactly like ggml's mul_mat
//     (`dr = (ne01 + nth - 1)/nth; ir0 = dr*ith`) and like the
//     `schedule(static)` pragmas this pool replaced: one contiguous range per
//     thread, zero synchronisation inside the loop.
//   - dynamic work stealing remains available (parallel_for_dynamic) for the
//     few loops whose per-iteration cost genuinely varies, but it is NOT the
//     default. Stealing costs one contended atomic fetch_add per chunk —
//     ~47 ns, because a single cache line bounces between every core — which
//     dominated short decode kernels when it was applied per element.
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

#include <algorithm>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "cpu/common/thread_policy.h"

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

namespace forge {
namespace cpu {

// How [0, n) is divided between the workers.
enum class Partition {
    Static,   // one contiguous range per thread, no synchronisation (default)
    Dynamic,  // atomic work stealing, one fetch_add per chunk
};

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

    // Static contiguous partition: thread i owns rows [i*dr, min((i+1)*dr, n))
    // with dr = ceil(n/nth) rounded up to a multiple of `chunk`. This is the
    // faithful replacement for `#pragma omp parallel for schedule(static)` and
    // for ggml's mul_mat row split. body(i0, i1) processes rows [i0, i1). The
    // calling thread participates as worker (nth_-1).
    void parallel_for(int n, int chunk,
                      const std::function<void(int, int)>& body) {
        dispatch(n, chunk, Partition::Static, &body, nullptr);
    }

    // Same, but the body also receives the worker index and the total worker
    // count: body(ith, nth, i0, i1). Use when a range needs a per-thread
    // scratch buffer or a strided decomposition.
    void parallel_for_ex(int n, int chunk,
                         const std::function<void(int, int, int, int)>& body) {
        dispatch(n, chunk, Partition::Static, nullptr, &body);
    }

    // Dynamic work stealing: thread i starts at chunk i, then grabs further
    // chunks with an atomic fetch_add. Only worth it when per-iteration cost
    // varies a lot (the original schedule(dynamic) sites); the atomic is
    // expensive, which is why this is no longer the default.
    void parallel_for_dynamic(int n, int chunk,
                              const std::function<void(int, int)>& body) {
        dispatch(n, chunk, Partition::Dynamic, &body, nullptr);
    }

private:
    void dispatch(int n, int chunk, Partition part,
                  const std::function<void(int, int)>* b2,
                  const std::function<void(int, int, int, int)>* b4) {
        if (n <= 0) {
            return;
        }
        if (chunk < 1) {
            chunk = 1;
        }
        const int nchunks = (n + chunk - 1) / chunk;
        body_ = b2;
        body_ex_ = b4;
        n_ = n;
        chunk_ = chunk;
        nchunks_ = nchunks;
        part_ = part;

        // Single chunk: no worker can contribute anything, so run it on the
        // calling thread and skip the wake/barrier entirely.
        //
        // Decode issues tens of thousands of these per response -- rms_norm and
        // rope with rows == 1, one per layer per token. Waking the pool for a
        // single chunk costs ~15 us of pure latency (measured in isolation) and
        // showed up as ~7 ms/token of rms_norm time, i.e. ~6% of decode.
        if (nchunks <= 1) {
            invoke(0, 0, n);
            return;
        }

        if (part == Partition::Static) {
            // ceil(n/nth), rounded up to a multiple of `chunk` so a range stays
            // chunk-aligned. With chunk == 1 this is exactly OpenMP's
            // schedule(static) split (GCC computes (n + nth - 1)/nth).
            const int base = (n + nth_ - 1) / nth_;
            dr_ = ((base + chunk - 1) / chunk) * chunk;
        } else {
            dr_ = 0;
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            current_chunk_.store(nth_, std::memory_order_relaxed);
            done_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);  // wake workers
        }
        cv_.notify_all();

        // main thread participates as worker (nth_-1)
        run_chunks(nth_ - 1, /*count_as_done=*/false);

        // Wait until every worker finished its chunks.
        //
        // NOTE: deliberately no spin here. dispatch() notifies the workers and
        // *then* runs the calling thread's own chunk, so the workers start
        // 10-50 us late (futex wake) and the calling thread almost always
        // finishes first. A join spin therefore never wins the race -- it burns
        // ~55 us spinning, still falls back to the futex, and meanwhile occupies
        // a core (or the SMT sibling of a still-running worker). Measured on
        // Qwen2.5-7B decode: it made every GEMM ~68 us slower, costing ~1.9 s
        // over a 256-token run -- exactly cancelling the rms_norm savings.
        std::unique_lock<std::mutex> lk(mtx_);
        cv_done_.wait(lk, [&] {
            return done_.load(std::memory_order_acquire) == nth_ - 1;
        });
    }

    void invoke(int ith, int i0, int i1) {
        if (body_ex_) {
            (*body_ex_)(ith, nth_, i0, i1);
        } else {
            (*body_)(i0, i1);
        }
    }

    // When count_as_done is true (worker threads) the completion is reported so
    // the caller can join; the calling thread itself must not report.
    void run_chunks(int ith, bool count_as_done) {
        if (part_ == Partition::Static) {
            // One contiguous range, no atomic, no stealing.
            const int i0 = std::min(ith * dr_, n_);
            const int i1 = std::min(i0 + dr_, n_);
            if (i0 < i1) {
                invoke(ith, i0, i1);
            }
        } else {
            int current = ith;  // ggml: first chunk = thread id
            while (current < nchunks_) {
                const int i0 = current * chunk_;
                invoke(ith, i0, std::min(n_, i0 + chunk_));
                current = current_chunk_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (!count_as_done) {
            return;
        }
        // Report completion with a plain atomic so the fast path never takes the
        // mutex. Only the last worker touches it, and only to wake a main thread
        // that already gave up spinning.
        if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 == nth_ - 1) {
            std::lock_guard<std::mutex> lk(mtx_);
            cv_done_.notify_all();
        }
    }

    void worker_loop(int ith) {
        pin_to_core(ith);
        const int spin_limit = thread_policy().spin_limit;
        int mygen = 0;
        for (;;) {
            // fast path: spin briefly on the generation counter
            int spins = 0;
            int g = gen_.load(std::memory_order_acquire);
            while (g == mygen && !stop_.load(std::memory_order_relaxed)) {
                if (++spins >= spin_limit) {
                    break;
                }
                FORGE_POOL_PAUSE();
                g = gen_.load(std::memory_order_acquire);
            }
            if (g != mygen) {
                mygen = g;
                run_chunks(ith, /*count_as_done=*/true);
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
            run_chunks(ith, /*count_as_done=*/true);
        }
    }

    void pin_to_core(int ith) {
        // Hard pinning is opt-in through the policy (FORGE_PIN_THREADS=first|last).
        //
        // Default is NO pinning (llama.cpp behaviour): leave placement to the
        // OS scheduler. Rationale: on P/E hybrids the previous "last nth
        // allowed CPUs" heuristic parked every worker on efficiency cores
        // (measured ~40% slower per core on an i5-13500H), and inside VMs
        // (WSL2) Linux CPU ids do not reliably map to physical topology, so a
        // fixed mask can fight the host scheduler instead of helping it.
        const ThreadPinMode mode = thread_policy().pin;
        if (mode == ThreadPinMode::None) {
            return;
        }
        const bool prefer_last = (mode == ThreadPinMode::Last);
#if defined(__linux__)
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
        const int n = static_cast<int>(std::thread::hardware_concurrency());
        const int target =
            n >= nth_ ? (prefer_last ? n - nth_ + ith : ith) : ith;
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << target);
#else
        (void)ith;
        (void)prefer_last;
#endif
    }

    int nth_;
    int n_ = 0;
    int chunk_ = 1;
    int nchunks_ = 0;
    int dr_ = 0;
    Partition part_ = Partition::Static;
    const std::function<void(int, int)>* body_ = nullptr;
    const std::function<void(int, int, int, int)>* body_ex_ = nullptr;
    std::vector<std::thread> workers_;
    std::atomic<int> current_chunk_{0};
    std::atomic<int> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable cv_done_;
};

// ---- thread-count state (replaces omp_get/set_num_threads) ----------------

inline int default_thread_count() {
    // FORGE_NUM_THREADS wins; OMP_NUM_THREADS is honoured so existing scripts
    // that sized the old OpenMP runtime keep working.
    const char* env = std::getenv("FORGE_NUM_THREADS");
    if (!env || !*env) {
        env = std::getenv("OMP_NUM_THREADS");
    }
    if (env && *env) {
        const int v = std::atoi(env);
        if (v > 0) {
            return v;
        }
    }
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    return hw > 0 ? hw : 1;
}

inline int& thread_count_state() {
    static int n = default_thread_count();
    return n;
}

// Current configured worker count. This is the built-in replacement for
// omp_get_max_threads().
inline int get_num_threads() {
    return thread_count_state();
}

// Set the worker count (replaces omp_set_num_threads()). The pool is resized
// lazily on the next parallel_for, so back-to-back set() calls with no work in
// between cost nothing.
inline void set_num_threads(int n) {
    if (n >= 1) {
        thread_count_state() = n;
    }
}

// Process-wide pool, lazily created with the configured thread count.
// Re-created only if the count changes between calls.
inline ThreadPool& cpu_thread_pool() {
    const int want = get_num_threads();
    static ThreadPool* pool = new ThreadPool(want);
    static int pool_nth = want;
    if (pool_nth != want) {
        delete pool;
        pool = new ThreadPool(want);
        pool_nth = want;
    }
    return *pool;
}

// Statically partitioned parallel-for over the process-wide pool: each worker
// gets one contiguous range of rows. `chunk` only sets the alignment of those
// ranges. This is the drop-in replacement for `schedule(static)`.
inline void parallel_for_chunks(int n, int chunk,
                                const std::function<void(int, int)>& body) {
    cpu_thread_pool().parallel_for(n, chunk, body);
}

// Variant whose body also receives (worker index, worker count).
inline void parallel_for_chunks_ex(
    int n, int chunk, const std::function<void(int, int, int, int)>& body) {
    cpu_thread_pool().parallel_for_ex(n, chunk, body);
}

// Dynamic work stealing — the replacement for `schedule(dynamic)`. Reserve it
// for loops whose per-iteration cost is genuinely uneven; the per-chunk atomic
// makes it a net loss on uniform, bandwidth-bound kernels.
inline void parallel_for_chunks_dynamic(
    int n, int chunk, const std::function<void(int, int)>& body) {
    cpu_thread_pool().parallel_for_dynamic(n, chunk, body);
}

// Element-wise dynamic variant: body(i) runs once per i in [0, n), handed out
// in `chunk`-sized units. The direct replacement for
// `#pragma omp parallel for schedule(dynamic, k)`.
template <typename Fn>
inline void parallel_for_dynamic(int n, int chunk, Fn&& body) {
    if (n <= 0) {
        return;
    }
    parallel_for_chunks_dynamic(n, chunk, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            body(i);
        }
    });
}

// Two-dimensional parallel-for (the OpenMP `collapse(2)` replacement):
// body(i0, i1) runs once per (i0, i1) with i0 in [0, n0) and i1 in [0, n1).
//
// The 2-D space is flattened row-major (idx = i0 * n1 + i1) so consecutive
// indices stay contiguous in the inner dimension, matching the memory layout of
// the tensors these loops walk.
namespace detail {

// Shared flattening for the 2-D variants; `dispatch` receives the flattened
// iteration count and the per-range body.
template <typename Fn, typename Dispatch>
inline void parallel_for2_impl(int n0, int n1, Fn&& body, Dispatch&& dispatch) {
    if (n0 <= 0 || n1 <= 0) {
        return;
    }
    const int64_t total = (int64_t)n0 * (int64_t)n1;
    if (total > (int64_t)INT_MAX) {
        // Pathological; fall back to a serial sweep rather than overflowing.
        for (int i0 = 0; i0 < n0; ++i0)
            for (int i1 = 0; i1 < n1; ++i1) body(i0, i1);
        return;
    }
    dispatch((int)total, [&](int a, int b) {
        for (int idx = a; idx < b; ++idx) {
            body((int)((int64_t)idx / n1), (int)((int64_t)idx % n1));
        }
    });
}

}  // namespace detail

// Statically split; the `schedule(static) collapse(2)` replacement.
template <typename Fn>
inline void parallel_for2(int n0, int n1, Fn&& body) {
    detail::parallel_for2_impl(n0, n1, body, [](int total, auto&& fn) {
        parallel_for_chunks(total, 1, fn);
    });
}

// Work-stealing variant; the `schedule(dynamic) collapse(2)` replacement. Use
// where the per-(i0, i1) cost varies, e.g. causal attention where row i touches
// i+1 keys.
template <typename Fn>
inline void parallel_for2_dynamic(int n0, int n1, Fn&& body) {
    detail::parallel_for2_impl(n0, n1, body, [](int total, auto&& fn) {
        parallel_for_chunks_dynamic(total, 1, fn);
    });
}

// Element-wise parallel-for: body(i) runs once per i in [0, n).
//
// Statically split into one contiguous range per worker, matching the
// `schedule(static)` pragmas this replaced and ggml's mul_mat row split. A
// per-element dynamic hand-out (what this used to do) costs one contended
// atomic per element, which measurably dominated short decode kernels.
template <typename Fn>
inline void parallel_for(int n, Fn&& body) {
    if (n <= 0) {
        return;
    }
    parallel_for_chunks(n, 1, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            body(i);
        }
    });
}

}  // namespace cpu
}  // namespace forge
