// Tests for the Hadamard activation rotation (Phase 5).
//
// What is asserted here is the algebra, because that is what the layer depends
// on and what can actually be checked without a model:
//
//  1. Orthonormality. R*R = I. If this fails, rotation is not a reparametrisation
//     -- every logit gets rescaled and attention is simply broken.
//  2. Self-inverse. R = R^-1 (the Hadamard matrix is symmetric), so rotate then
//     un-rotate is the identity. This is what lets undo_v() be the same call.
//  3. Dot-product preservation. (Rq).(Rk) = q.k. The entire point. A stride or
//     ordering slip here produces plausible output that is quietly wrong.
//  4. The attention identity end to end: softmax(q.k) @ (R v), un-rotated, must
//     equal the unrotated attention output.
//  5. Incoherence: one large component spreads over all n coordinates.
//
// Deliberately NOT asserted: that rotation reduces quantisation error. It does
// not always. Forge's q4_0 is a per-32-block symmetric-amax quantiser, and for
// that family the effect changes sign with the outlier ratio -- spreading a
// dominant coordinate raises the scale of every block it touches, so rotation
// helps at moderate ratios and hurts once one coordinate dominates. The measured
// numbers are printed as information; the decision is made against real K/V in
// the docs, not asserted here.
//
// CPU is checked against the explicit matrix and CUDA is checked against CPU, so
// the two paths cannot drift apart.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/inference/layers/attn_rot_executor.h"
#include "forge/operators.h"
#include "forge/tensor.h"

using namespace forge;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const char* what) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL  %s\n", what);
    }
}

static TensorPtr vec1d(const std::vector<float>& v, DeviceType dev = DeviceType::CPU) {
    auto t = std::make_shared<Tensor>(DataType::FP32,
                                      std::vector<int64_t>{static_cast<int64_t>(v.size())}, dev);
    if (dev == DeviceType::CPU) {
        std::memcpy(t->data(), v.data(), v.size() * sizeof(float));
    } else {
        cudaMemcpy(t->data(), v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    return t;
}

static std::vector<float> read(const TensorPtr& t) {
    const float* p = static_cast<const float*>(t->data());
    return std::vector<float>(p, p + t->numel());
}

static std::vector<float> pull(const TensorPtr& t) {
    if (t->device() == DeviceType::CPU)
        return read(t);
    std::vector<float> out(static_cast<size_t>(t->numel()));
    cudaMemcpy(out.data(), t->data(), out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

static double dot(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
        s += static_cast<double>(a[i]) * b[i];
    return s;
}

// Full-order rotation through the shipped op, so the helper and the op cannot
// drift. Note undo_v() is deliberately NOT usable for this: it rotates in
// 64-wide V sub-blocks, which is a different transform when n > 64.
static std::vector<float> rotate_cpu(std::vector<float> v, int order) {
    TensorPtr t = vec1d(v);
    ops::hadamard_rotate(t, order);
    return read(t);
}

int main() {
    std::printf("== AttnRot: Hadamard activation rotation ==\n");
    char buf[200];

    // ---- 1. matrix properties ------------------------------------------------
    for (int n : {8, 16, 32, 64, 128, 256}) {
        const std::vector<float> h = hadamard_matrix(n);

        double max_off = 0, max_diag = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0;
                for (int k = 0; k < n; ++k)
                    s += static_cast<double>(h[i * n + k]) * h[j * n + k];
                const double d = s - (i == j ? 1.0 : 0.0);
                if (i == j)
                    max_diag = std::max(max_diag, std::fabs(d));
                else
                    max_off = std::max(max_off, std::fabs(d));
            }
        }
        std::snprintf(buf, sizeof(buf), "H(%d) orthonormal, max |RR^T - I| = %.2e", n,
                      std::max(max_off, max_diag));
        check(max_off < 1e-5 && max_diag < 1e-5, buf);

        double max_asym = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                max_asym = std::max<double>(max_asym, std::fabs(h[i * n + j] - h[j * n + i]));
        std::snprintf(buf, sizeof(buf), "H(%d) symmetric (self-inverse), max asym = %.2e", n,
                      max_asym);
        check(max_asym < 1e-7, buf);
    }

    // ---- 2. butterfly == explicit matrix, round trip, incoherence -----------
    {
        const int n = 64;
        const std::vector<float> h = hadamard_matrix(n);
        std::vector<float> v(n);
        for (int i = 0; i < n; ++i)
            v[i] = std::sin(0.37f * i) + 0.1f * i;
        const std::vector<float> got = rotate_cpu(v, n);

        std::vector<float> want(n, 0.0f);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                want[i] += h[i * n + j] * v[j];
        double err = 0;
        for (int i = 0; i < n; ++i)
            err = std::max<double>(err, std::fabs(got[i] - want[i]));
        std::snprintf(buf, sizeof(buf), "butterfly(n=%d) == explicit H @ v, max err %.2e", n, err);
        check(err < 1e-5, buf);

        const std::vector<float> back = rotate_cpu(got, n);
        double rt = 0;
        for (int i = 0; i < n; ++i)
            rt = std::max<double>(rt, std::fabs(back[i] - v[i]));
        std::snprintf(buf, sizeof(buf), "rotate then unrotate == identity, max err %.2e", rt);
        check(rt < 1e-4, buf);

        double nv = 0, ng = 0;
        for (int i = 0; i < n; ++i) {
            nv += static_cast<double>(v[i]) * v[i];
            ng += static_cast<double>(got[i]) * got[i];
        }
        std::snprintf(buf, sizeof(buf), "rotation preserves ||v|| (%.6f -> %.6f)", std::sqrt(nv),
                      std::sqrt(ng));
        check(std::fabs(std::sqrt(nv) - std::sqrt(ng)) < 1e-3, buf);

        // The mechanism itself: one large component spreads over all n
        // coordinates, so the largest one drops by ~sqrt(n). This is the
        // incoherence property the feature exists to provide.
        {
            std::vector<float> e0(n, 0.0f);
            e0[7] = 1.0f;
            const std::vector<float> re = rotate_cpu(e0, n);
            float mx = 0;
            for (int i = 0; i < n; ++i)
                mx = std::max(mx, std::fabs(re[i]));
            const double expect = 1.0 / std::sqrt(static_cast<double>(n));
            std::snprintf(buf, sizeof(buf), "incoherence: unit spike -> max %.5f (1/sqrt(n) = %.5f)",
                          mx, expect);
            check(std::fabs(mx - expect) < 1e-5, buf);
        }

        // Informational: the sign of the quantisation effect is not fixed.
        const auto sqerr = [](const std::vector<float>& x, int order) {
            double e = 0;
            for (int b = 0; b < order / 32; ++b) {
                float amax = 0;
                for (int i = 0; i < 32; ++i)
                    amax = std::max(amax, std::fabs(x[b * 32 + i]));
                const float sc = amax / 7.0f;
                if (sc <= 0)
                    continue;
                for (int i = 0; i < 32; ++i) {
                    const float d = x[b * 32 + i] - std::round(x[b * 32 + i] / sc) * sc;
                    e += static_cast<double>(d) * d;
                }
            }
            return e;
        };
        std::printf("      [info] 4-bit per-32-block MSE, spike/rms vs ratio (sign is NOT fixed):\n");
        for (int ratio : {8, 100, 1000}) {
            std::vector<float> sp(n, 0.0f);
            for (int i = 0; i < n; ++i)
                sp[i] = std::sin(0.3f * i) * 0.05f;
            sp[7] = 0.05f * static_cast<float>(ratio);
            const double eb = sqerr(sp, n), ea = sqerr(rotate_cpu(sp, n), n);
            std::printf("      [info]   spike/rms %5d : %.3e -> %.3e  (%.2fx)\n", ratio, eb, ea,
                        eb / std::max(1e-30, ea));
        }
    }

    // ---- 3. dot product preservation ----------------------------------------
    {
        const int n = 64;
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; ++i) {
            a[i] = std::sin(0.11f * i) * (i == 5 ? 40.0f : 1.0f);
            b[i] = std::cos(0.29f * i);
        }
        const double before = dot(a, b), after = dot(rotate_cpu(a, n), rotate_cpu(b, n));
        std::snprintf(buf, sizeof(buf), "q.k preserved: %.6f -> %.6f (rel %.2e)", before, after,
                      std::fabs(after - before) / std::max(1e-9, std::fabs(before)));
        check(std::fabs(after - before) < 1e-3 * std::max(1.0, std::fabs(before)), buf);
    }

    // ---- 4. the full attention identity -------------------------------------
    {
        const int n = 64, T = 40, D = 64;
        std::vector<float> q(D), k(T * D), v(T * D);
        for (int i = 0; i < D; ++i)
            q[i] = std::sin(0.7f * i);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < D; ++i) {
                k[t * D + i] = std::sin(0.3f * t + 0.13f * i);
                v[t * D + i] = std::cos(0.21f * t + 0.41f * i);
            }
        const std::vector<float> rq = rotate_cpu(q, n), rk = rotate_cpu(k, n),
                               rv = rotate_cpu(v, n);

        std::vector<float> lr(T), lrot(T);
        for (int t = 0; t < T; ++t) {
            const std::vector<float> kr(k.begin() + t * D, k.begin() + (t + 1) * D);
            const std::vector<float> krr(rk.begin() + t * D, rk.begin() + (t + 1) * D);
            lr[t] = static_cast<float>(dot(q, kr));
            lrot[t] = static_cast<float>(dot(rq, krr));
        }
        double lerr = 0;
        for (int t = 0; t < T; ++t)
            lerr = std::max<double>(lerr, std::fabs(lr[t] - lrot[t]));
        std::snprintf(buf, sizeof(buf), "attention logits identical, max err %.2e", lerr);
        check(lerr < 1e-2, buf);

        const float mx = *std::max_element(lr.begin(), lr.end());
        std::vector<float> p(T);
        double z = 0;
        for (int t = 0; t < T; ++t) {
            p[t] = std::exp(lr[t] - mx);
            z += p[t];
        }
        for (auto& x : p)
            x = static_cast<float>(x / z);

        std::vector<float> oref(D, 0.0f), orot(D, 0.0f);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < D; ++i) {
                oref[i] += p[t] * v[t * D + i];
                orot[i] += p[t] * rv[t * D + i];
            }
        const std::vector<float> rec = rotate_cpu(orot, n);
        double oerr = 0, scale = 0;
        for (int i = 0; i < D; ++i) {
            oerr = std::max<double>(oerr, std::fabs(rec[i] - oref[i]));
            scale = std::max<double>(scale, std::fabs(oref[i]));
        }
        std::snprintf(buf, sizeof(buf),
                      "un-rotated attention output == plain output, max err %.2e (|out| %.3f)",
                      oerr, scale);
        check(oerr < 1e-3 * std::max(1.0, scale), buf);
    }

    // ---- 5. order selection and V sub-blocking ------------------------------
    {
        check(rotation_order_for_head_dim(64) == 64, "head_dim 64 -> order 64");
        check(rotation_order_for_head_dim(128) == 128, "head_dim 128 -> order 128 (largest)");
        check(rotation_order_for_head_dim(256) == 256, "head_dim 256 -> order 256 (largest)");
        check(rotation_order_for_head_dim(192) == 64, "head_dim 192 -> order 64 (192 = 2^6*3)");
        check(rotation_order_for_head_dim(32) == 0, "head_dim 32 -> disabled (below min 64)");
        check(rotation_order_for_head_dim(0) == 0, "head_dim 0 -> disabled");

        const int n = 128;
        std::vector<float> v(n);
        for (int i = 0; i < n; ++i)
            v[i] = (i < 64) ? std::sin(0.3f * i) : (10.0f + std::cos(0.9f * i));
        AttnRotExecutor e;
        e.configure(true, n, DeviceType::CPU);
        check(e.v_block() == 64, "V block is 64");
        check(e.order() == 128, "K/Q order is the full head dim");
        e.apply_v(vec1d(v));
        const std::vector<float> got = read(e.apply_v(vec1d(v)));

        const std::vector<float> b1 = rotate_cpu(std::vector<float>(v.begin() + 64, v.end()), 64);
        double err = 0;
        for (int i = 0; i < 64; ++i)
            err = std::max<double>(err, std::fabs(got[64 + i] - b1[i]));
        std::snprintf(buf, sizeof(buf), "V 64-wide sub-blocks are independent, max err %.2e", err);
        check(err < 1e-5, buf);

        // apply_qk must rotate BOTH sides by the same order, or the dot product
        // is not preserved -- which is what this pair of checks pins down.
        std::vector<float> q(128), k(128);
        for (int i = 0; i < 128; ++i) {
            q[i] = std::sin(0.2f * i);
            k[i] = std::cos(0.45f * i);
        }
        const double d0 = dot(q, k);
        TensorPtr tq = vec1d(q), tk = vec1d(k);
        e.apply_qk(tq, tk, nullptr, nullptr);
        const double d1 = dot(read(tq), read(tk));
        std::snprintf(buf, sizeof(buf), "apply_qk preserves q.k: %.6f -> %.6f", d0, d1);
        check(std::fabs(d1 - d0) < 1e-3 * std::max(1.0, std::fabs(d0)), buf);
    }

    // ---- 6. disabled is an exact no-op, and carries a signature -------------
    {
        std::vector<float> v(64);
        for (int i = 0; i < 64; ++i)
            v[i] = static_cast<float>(i);
        AttnRotExecutor off;
        off.configure(false, 128, DeviceType::CPU);
        check(!off.enabled(), "configure(false) disables");
        check(off.signature() == 0, "disabled signature is 0");
        TensorPtr t = vec1d(v);
        check(off.apply_v(t) == t && read(t) == v, "apply_v is an exact no-op when disabled");

        AttnRotExecutor on;
        on.configure(true, 128, DeviceType::CPU);
        check(on.signature() != 0, "enabled signature is non-zero (cache identity guard)");
        // Different order must not collide, or a cache could be read with the
        // wrong rotation.
        AttnRotExecutor o2;
        o2.configure(true, 64, DeviceType::CPU);
        check(on.signature() != o2.signature(), "signatures differ across rotation orders");
    }

    // ---- 7. unsupported head dims are refused, not silently wrong -----------
    {
        AttnRotExecutor e;
        e.configure(true, 48, DeviceType::CPU);
        check(!e.enabled(), "head_dim 48 (not a multiple of 64) refuses rotation");
        e.configure(true, 80, DeviceType::CPU);
        check(!e.enabled(), "head_dim 80 (not a multiple of 64) refuses rotation");
        e.configure(true, 1024, DeviceType::CPU);
        check(!e.enabled(), "order 1024 (> kernel max 512) refuses rotation");
    }

    // ---- 8. CUDA matches CPU -------------------------------------------------
    {
        std::printf("-- CUDA --\n");
        for (int n : {64, 128, 256}) {
            std::vector<float> v(n);
            for (int i = 0; i < n; ++i)
                v[i] = std::sin(0.13f * i * (n / 64.0f)) + 0.05f * static_cast<float>((i * 7) % n);
            const std::vector<float> ref = rotate_cpu(v, n);
            TensorPtr d = vec1d(v, DeviceType::CUDA);
            const bool ok = ops::hadamard_rotate(d, n);
            check(ok, "CUDA launcher accepts a supported order");
            cudaDeviceSynchronize();
            const std::vector<float> got = pull(d);
            double err = 0;
            for (int i = 0; i < n; ++i)
                err = std::max<double>(err, std::fabs(got[i] - ref[i]));
            std::snprintf(buf, sizeof(buf), "CUDA rotate(n=%d) == CPU, max err %.2e", n, err);
            check(err < 1e-5, buf);

            // ...and back again, since self-inverse is what un-rotates the output.
            check(ops::hadamard_rotate(d, n), "CUDA second call accepted");
            cudaDeviceSynchronize();
            const std::vector<float> rt = pull(d);
            double e2 = 0;
            for (int i = 0; i < n; ++i)
                e2 = std::max<double>(e2, std::fabs(rt[i] - v[i]));
            std::snprintf(buf, sizeof(buf), "CUDA round trip == input, max err %.2e", e2);
            check(e2 < 1e-4, buf);
        }
        TensorPtr d = vec1d(std::vector<float>(64, 1.0f), DeviceType::CUDA);
        check(!ops::hadamard_rotate(d, 48), "CUDA launcher refuses order 48 instead of guessing");
        check(!ops::hadamard_rotate(d, 1024), "CUDA launcher refuses order 1024");
    }

    std::printf("\n=== %d/%d checks passed ===\n", g_pass, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}
