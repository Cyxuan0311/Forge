#pragma once

namespace forge {

// Detect the number of physical CPU cores (0 when undetectable).
int physical_core_count();

// Probe the real fused FFN-up GEMV (short bursts with idle between, as in
// decode) across candidate thread counts and return the thread count with the
// lowest median burst time. `hidden`/`intermediate` size the synthetic Q2_K
// weights to the target model (defaults target Llama-3.1-8B class). Falls back
// to a sane default when detection or probing fails.
int detect_best_cpu_threads();
int detect_best_cpu_threads(int hidden, int intermediate);

}  // namespace forge