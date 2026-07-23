#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if FORGE_USE_CUBLAS
#    include <cublas_v2.h>
#endif
#include <cstdint>

namespace forge {
namespace cuda {

// Thread-local scratch memory pool for temporary device allocations.
// Avoids per-kernel cudaMalloc/cudaFree overhead.
struct CudaScratchPool {
    void* ptr = nullptr;
    size_t capacity = 0;
    ~CudaScratchPool() {
        if (ptr)
            cudaFree(ptr);
    }
    void* ensure(size_t bytes) {
        if (bytes > capacity) {
            if (ptr)
                cudaFree(ptr);
            cudaMalloc(&ptr, bytes);
            capacity = bytes;
        }
        return ptr;
    }
};

inline CudaScratchPool& scratch_pool() {
    static CudaScratchPool pool;
    return pool;
}

#if FORGE_USE_CUBLAS
// Lazily-initialized cublas handle (one per process).
inline cublasHandle_t get_cublas_handle(cudaStream_t stream = 0) {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    if (stream) {
        cublasSetStream(handle, stream);
    }
    return handle;
}
#endif

// Device helper: unpack Q4_0 signed nibble
__device__ __forceinline__ int q4_unpack(const uint8_t* qs, int j) {
    if (j < 16) {
        return (qs[j] & 0x0F) - 8;
    } else {
        return ((qs[j - 16] >> 4) & 0x0F) - 8;
    }
}

// Device helper: unpack Q4_1 unsigned nibble
__device__ __forceinline__ int q4_unpack_unsigned(const uint8_t* qs, int j) {
    if (j < 16) {
        return qs[j] & 0x0F;
    } else {
        return (qs[j - 16] >> 4) & 0x0F;
    }
}

// Device helper: Q4_K scale/min extraction
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Device helper: Q3_K scale unpacking (12 packed bytes → 16 signed 6-bit scales)
__device__ __forceinline__ void q3_k_unpack_scales(const uint8_t* scales_raw, int8_t* scales_out) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    memcpy(aux, scales_raw, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    memcpy(scales_out, aux, 16);
}

// ---- dp4a helper (sm_61+) ----
// 4-way int8 dot product: a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + c
__device__ __forceinline__ int forge_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int r = c;
    for (int i = 0; i < 4; i++)
        r += ((a >> (i * 8)) & 0xFF) * ((b >> (i * 8)) & 0xFF);
    return r;
#endif
}

// Pack 4 int8 values into a 32-bit integer for dp4a
__device__ __forceinline__ int pack_int8_4(int8_t v0, int8_t v1, int8_t v2, int8_t v3) {
    // Note: use unsigned cast to avoid sign-extension during shift
    return (static_cast<uint32_t>(static_cast<uint8_t>(v0))) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v1)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24);
}

// Read 4 consecutive int8 values from memory and pack into 32-bit for dp4a
__device__ __forceinline__ int load_pack_int8_4(const int8_t* ptr) {
    int result;
    memcpy(&result, ptr, 4);  // prevents the compiler from assuming alignment
    return result;
}

}  // namespace cuda
}  // namespace forge
