#pragma once
// Q4_0 Weight Repack Cache (x86 / AVX2).
// Repacked weights are keyed by the original Tensor data pointer.
// This ensures each weight matrix is repacked at most once.

#include <cstddef>
#include <cstdint>

namespace forge {
namespace ops {

#ifdef USE_AVX2

// Get or create repacked Q4_0 weights for decode.
// Returns nullptr if repack is not applicable (e.g., N < 4).
const uint8_t* get_repacked_q4_0(const void* orig_data, int64_t K, int64_t N);

// Get or create 8-row-interleaved Q4_K weights for decode (block_q4_Kx8).
// Returns nullptr if repack is not applicable (N % 8 != 0 or K % 256 != 0).
const uint8_t* get_repacked_q4_K(const void* orig_data, int64_t K, int64_t N);

// Get or create 8-row-interleaved Q6_K weights for decode (block_q6_Kx8).
// Returns nullptr if repack is not applicable (N % 8 != 0 or K % 256 != 0).
const uint8_t* get_repacked_q6_K(const void* orig_data, int64_t K, int64_t N);

// Get or create 8-row-interleaved Q4_0 weights for decode (block_q4_0x8,
// llama-style LUT layout). Returns nullptr if N % 8 != 0 or K % 32 != 0.
const uint8_t* get_repacked_q4_0_8x8(const void* orig_data, int64_t K, int64_t N);

#endif  // USE_AVX2

}  // namespace ops
}  // namespace forge