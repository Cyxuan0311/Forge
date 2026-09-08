// =========================================================================
// cuda_fp8_quant.cu — 6-C-5.1a/5.1b: W8A8 FP8 E4M3 per-tensor quantize producer
//
// Fully device-side pipeline (no D2H/H2D), so it is capturable by CUDA Graphs:
//   amax reduction -> scale = amax/240 -> q = encode(x / scale)
// The scale lands directly in the output Tensor's stable device scalar, obtained
// from Tensor::fp8_scale_device_slot().
// =========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

#include "forge/fp8_quant.h"
#include "forge/fp8_utils.h"

namespace forge {

namespace {

constexpr int kBlock = 256;

// Grow-only scratch for the block amax partials. Deliberately static and never freed
// during a capture: allocating/freeing inside a captured region is not safe, and after
// warm-up the buffer is large enough that no allocation happens at all.
float* amax_partials_scratch(int blocks) {
    static float* buf = nullptr;
    static int cap = 0;
    if (blocks > cap) {
        if (buf)
            cudaFree(buf);
        if (cudaMalloc(reinterpret_cast<void**>(&buf),
                       static_cast<size_t>(blocks) * sizeof(float)) != cudaSuccess) {
            buf = nullptr;
            cap = 0;
            return nullptr;
        }
        cap = blocks;
    }
    return buf;
}

// 6-C-5.1b: NaN/Inf must be detected EXPLICITLY. fmaxf(NaN, x) returns x, so a bare
// abs-max reduction would silently swallow a NaN and never reject it. Each block
// therefore reduces TWO values: the finite abs-max and a "saw non-finite" flag.
// Layout: partials[2*b] = abs-max, partials[2*b + 1] = non-finite flag.
// 6-C-5.6a: RELU is a template flag so the FP8 activation reuses THIS reduction instead
// of growing a third FP8 scale implementation. Fusing relu here also means the
// activation produces its FP8 output without materializing an FP32 tensor (5.6d).
template <bool RELU>
__global__ void fp8_amax_partial_kernel(const float* __restrict__ x, int n,
                                        float* __restrict__ partials) {
    __shared__ float sm[kBlock];
    __shared__ float sbad[kBlock];
    const int tid = threadIdx.x;

    float m = 0.f;
    float bad = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = x[i];
        if (RELU)
            v = fmaxf(v, 0.f);
        if (!isfinite(v))
            bad = 1.f;
        else
            m = fmaxf(m, fabsf(v));
    }

    sm[tid] = m;
    sbad[tid] = bad;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sm[tid] = fmaxf(sm[tid], sm[tid + s]);
            sbad[tid] = fmaxf(sbad[tid], sbad[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[2 * blockIdx.x] = sm[0];
        partials[2 * blockIdx.x + 1] = sbad[0];
    }
}

// 6-C-5.1b: finalize amax and apply the scale contract. Writes the scale into the
// Tensor-owned device scalar. `amax_out` is optional (may be null) and exists so tests
// can inspect amax without turning this into a synchronizing production path.
__global__ void fp8_amax_finalize_kernel(const float* __restrict__ partials, int npartials,
                                         float* __restrict__ scale_out,
                                         float* __restrict__ amax_out) {
    __shared__ float sm[kBlock];
    __shared__ float sbad[kBlock];
    const int tid = threadIdx.x;

    float m = 0.f;
    float bad = 0.f;
    for (int i = tid; i < npartials; i += blockDim.x) {
        m = fmaxf(m, partials[2 * i]);
        bad = fmaxf(bad, partials[2 * i + 1]);
    }

    sm[tid] = m;
    sbad[tid] = bad;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sm[tid] = fmaxf(sm[tid], sm[tid + s]);
            sbad[tid] = fmaxf(sbad[tid], sbad[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Non-finite anywhere -> amax is non-finite -> REJECT (scale becomes NaN).
        const float amax = (sbad[0] > 0.f) ? NAN : sm[0];
        float scale;
        if (!isfinite(amax)) {
            scale = NAN;  // REJECT: do not let NaN/Inf poison the graph silently
        } else if (amax <= 0.f) {
            scale = 1.f;  // all-zero input -> scale 1, output is all zeros
        } else {
            scale = amax / kFp8E4M3Max;
        }
        if (amax_out)
            *amax_out = amax;
        *scale_out = scale;
    }
}

// 6-C-5.1a: encode with the scale read from DEVICE memory (no host round trip).
// 6-C-5.6a: same RELU flag as the reduction so activation and plain quantize encode
// with identical semantics.
template <bool RELU>
__global__ void fp8_quantize_scaled_kernel(const float* __restrict__ x, uint8_t* __restrict__ q,
                                           int n, const float* __restrict__ scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float s = *scale;
    float v = x[i];
    if (RELU)
        v = fmaxf(v, 0.f);
    q[i] = fp32_to_fp8_e4m3(v / s);
}

}  // namespace

Fp8AmaxStatus fp8_amax_status(float amax) {
    if (!std::isfinite(amax))
        return Fp8AmaxStatus::NonFinite;
    if (amax <= 0.f)
        return Fp8AmaxStatus::Zero;
    return Fp8AmaxStatus::Ok;
}

float fp8_w8a8_scale_from_amax(float amax) {
    switch (fp8_amax_status(amax)) {
    case Fp8AmaxStatus::Ok:
        return amax / kFp8E4M3Max;
    case Fp8AmaxStatus::Zero:
        return 1.f;
    case Fp8AmaxStatus::NonFinite:
    default:
        return NAN;
    }
}

// 6-C-5.5a: the ONE reduction pipeline, shared by the W8A8 input producer (5.1a) and the
// FP8 output producer (5.5a). Both write the scale into a caller-supplied device slot so
// neither depends on a host scale, and neither allocates (given warm scratch).
void quantize_fp8_e4m3_impl(const float* x, uint8_t* q, int n, float* scale_dev,
                            cudaStream_t stream, bool apply_relu = false) {
    const int pblocks = (n + kBlock - 1) / kBlock;
    // 2 floats per block: [abs-max, non-finite flag] (6-C-5.1b).
    float* partials = amax_partials_scratch(2 * pblocks);
    if (!partials)
        throw std::runtime_error("quantize_fp8_e4m3: failed to allocate amax scratch");

    const int qblocks = (n + kBlock - 1) / kBlock;
    if (apply_relu) {
        fp8_amax_partial_kernel<true><<<pblocks, kBlock, 0, stream>>>(x, n, partials);
        fp8_amax_finalize_kernel<<<1, kBlock, 0, stream>>>(partials, pblocks, scale_dev,
                                                           /*amax_out=*/nullptr);
        fp8_quantize_scaled_kernel<true><<<qblocks, kBlock, 0, stream>>>(x, q, n, scale_dev);
    } else {
        fp8_amax_partial_kernel<false><<<pblocks, kBlock, 0, stream>>>(x, n, partials);
        fp8_amax_finalize_kernel<<<1, kBlock, 0, stream>>>(partials, pblocks, scale_dev,
                                                           /*amax_out=*/nullptr);
        fp8_quantize_scaled_kernel<false><<<qblocks, kBlock, 0, stream>>>(x, q, n, scale_dev);
    }
}

TensorPtr quantize_fp8_e4m3_w8a8(const TensorPtr& src, cudaStream_t stream) {
    if (!src)
        throw std::invalid_argument("quantize_fp8_e4m3_w8a8: null source tensor");
    if (src->dtype() != DataType::FP32)
        throw std::invalid_argument("quantize_fp8_e4m3_w8a8: source must be FP32");
    if (src->numel() <= 0)
        throw std::invalid_argument("quantize_fp8_e4m3_w8a8: empty source tensor");

    const int n = static_cast<int>(src->numel());

    // 1) FP8 output Tensor (same shape/device), 2) its stable device scale slot.
    auto out = std::make_shared<Tensor>(DataType::FP8_E4M3, src->shape(), src->device());
    float* scale_dev = out->fp8_scale_device_slot(stream);
    if (!scale_dev)
        throw std::runtime_error("quantize_fp8_e4m3_w8a8: failed to allocate device scale slot");

    // 3) device amax reduction, 4) scale = amax / 240, 5) encode (all on device)
    quantize_fp8_e4m3_impl(static_cast<const float*>(src->data()),
                           static_cast<uint8_t*>(out->data()), n, scale_dev, stream);

    return out;
}

TensorPtr quantize_fp8_e4m3_output(const TensorPtr& src, cudaStream_t stream) {
    if (!src)
        throw std::invalid_argument("quantize_fp8_e4m3_output: null source tensor");
    if (src->dtype() != DataType::FP32)
        throw std::invalid_argument("quantize_fp8_e4m3_output: source must be FP32");
    if (src->numel() <= 0)
        throw std::invalid_argument("quantize_fp8_e4m3_output: empty source tensor");

    const int n = static_cast<int>(src->numel());
    auto out = std::make_shared<Tensor>(DataType::FP8_E4M3, src->shape(), src->device());
    float* scale_dev = out->fp8_scale_device_slot(stream);
    if (!scale_dev)
        throw std::runtime_error("quantize_fp8_e4m3_output: failed to allocate device scale slot");

    quantize_fp8_e4m3_impl(static_cast<const float*>(src->data()),
                           static_cast<uint8_t*>(out->data()), n, scale_dev, stream);
    return out;
}

void quantize_fp8_e4m3_output_into(const TensorPtr& src, const TensorPtr& dst, cudaStream_t stream,
                                   bool apply_relu) {
    if (!src || !dst)
        throw std::invalid_argument("quantize_fp8_e4m3_output_into: null tensor");
    if (src->dtype() != DataType::FP32)
        throw std::invalid_argument("quantize_fp8_e4m3_output_into: source must be FP32");
    if (dst->dtype() != DataType::FP8_E4M3)
        throw std::invalid_argument("quantize_fp8_e4m3_output_into: dst must be FP8_E4M3");
    if (src->numel() != dst->numel())
        throw std::invalid_argument("quantize_fp8_e4m3_output_into: shape mismatch");

    const int n = static_cast<int>(src->numel());
    // Reuse the dst's EXISTING stable slot. It must already exist (allocated outside
    // capture) -- allocating here would break graph capture.
    float* scale_dev = dst->fp8_scale_device_slot(stream);
    if (!scale_dev)
        throw std::runtime_error("quantize_fp8_e4m3_output_into: failed to get device scale slot");

    // 6-C-5.6a: with apply_relu this IS the FP8 activation producer -- it reuses the
    // 5.1b/5.5b-verified scale contract and materializes no intermediate FP32 tensor.
    quantize_fp8_e4m3_impl(static_cast<const float*>(src->data()),
                           static_cast<uint8_t*>(dst->data()), n, scale_dev, stream, apply_relu);
}

}  // namespace forge
