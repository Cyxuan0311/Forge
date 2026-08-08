#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace forge::inspect {

// Minimal C++20 thread pool used by the parallel GGUF scanner.
// Tasks must not throw and must not outlive the parallel_for call.
class ThreadPool {
public:
    explicit ThreadPool(size_t n = 0) {
        if (n == 0)
            n = std::thread::hardware_concurrency();
        if (n == 0)
            n = 1;
        workers_.reserve(n);
        for (size_t i = 0; i < n; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    ~ThreadPool() {
        stop_ = true;
        cv_.notify_all();
        for (auto& t : workers_)
            if (t.joinable())
                t.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t size() const { return workers_.size(); }

    // Reconfigures the worker count. Drains nothing (callers only resize when
    // idle); any in-flight tasks must have completed first.
    void resize(size_t n) {
        if (n == 0)
            n = std::thread::hardware_concurrency();
        if (n == 0)
            n = 1;
        if (n == workers_.size())
            return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_)
            if (t.joinable())
                t.join();
        workers_.clear();
        stop_ = false;
        workers_.reserve(n);
        for (size_t i = 0; i < n; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
    }

    // Runs f(i) for i in [begin, end), split into size() contiguous ranges.
    // Blocks until all ranges complete. Thread-safe only if f writes disjoint slots.
    template <typename F>
    void parallel_for(size_t begin, size_t end, F&& f) {
        if (begin >= end)
            return;
        const size_t n = workers_.size();
        const size_t range = end - begin;
        if (n == 1 || range < n) {
            for (size_t i = begin; i < end; ++i)
                f(i);
            return;
        }
        std::atomic<size_t> remaining{n};
        const size_t per = (range + n - 1) / n;
        for (size_t t = 0; t < n; ++t) {
            const size_t s = begin + t * per;
            const size_t e = std::min(end, s + per);
            if (s >= e) {
                remaining.fetch_sub(1);
                continue;
            }
            submit([this, &f, s, e, &remaining]() {
                for (size_t i = s; i < e; ++i)
                    f(i);
                remaining.fetch_sub(1);
                done_cv_.notify_all();
            });
        }
        std::unique_lock<std::mutex> lk(done_mu_);
        done_cv_.wait(lk, [&] { return remaining.load() == 0; });
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty())
                    return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::mutex done_mu_;
    std::condition_variable done_cv_;
};

// Process-wide pool reused across calls (avoids re-spawning threads per file).
// Resized to `n` workers on first use or when `n` changes.
inline ThreadPool& cached_pool(size_t n) {
    static ThreadPool pool;
    pool.resize(n);
    return pool;
}

}  // namespace forge::inspect
