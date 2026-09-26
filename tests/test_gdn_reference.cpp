// Reference cross-check for the BailingMoE3 / KDA recurrent (gated delta net)
// kernel against llama.cpp's ggml_compute_forward_gated_delta_net (KDA branch,
// ggml/src/ggml-cpu/ops.cpp ~11002-11032):
//
//   state is transposed: s_out[j*S_v + i] = S[i][j]
//   1) delta[i] = exp(g[i]);  S[i][j] *= delta[i]            (gate on key axis i)
//   2) delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
//   3) S[i][j] += k[i] * delta[j]
//   4) out[j]   = (sum_i S[i][j]*q[i]) * scale,  scale = 1/sqrt(S_v)
//
// Hand-computed expectations use eps = 0 so l2_norm_heads() is the identity on
// already-unit q/k, keeping the arithmetic exact and easy to verify by hand.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "forge/tensor.h"
#include "operators/cpu/gdn.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#    include <stdexcept>
#    include <string>

#    include "forge/inference/layers/bailingmoe3_recurrent_memory.h"
#    include "operators/cuda/cuda_gdn.h"
#endif

namespace {

using namespace forge;

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

bool near(float a, float b, float tol = 1e-5f) {
    return std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= tol;
}

TensorPtr make_tensor(const std::vector<float>& data, const std::vector<int64_t>& shape) {
    auto t = std::make_shared<Tensor>(DataType::FP32, shape, DeviceType::CPU);
    std::memcpy(t->data(), data.data(), data.size() * sizeof(float));
    return t;
}

std::vector<float> flatten2(const TensorPtr& t) {
    const float* p = static_cast<const float*>(t->data());
    return std::vector<float>(p, p + t->numel());
}

// ---- Hand-computed reference (head_dim=4, num_heads=1, seq_len=2) ----------
//
// q = [1,0,0,0]            (unit, unchanged by l2_norm with eps=0)
// k = [0.5,0.5,0.5,0.5]    (unit)
// v0 = [1,2,3,4],  v1 = [2,4,6,8]
// g  = -0.693147f (exp = 0.5) for both tokens, all 4 channels
// beta = 0.5 for both tokens
// scale = 1/sqrt(4) = 0.5
//
// t=0 (S starts at zero):
//   delta  = v0 * beta                  = [0.5, 1.0, 1.5, 2.0]
//   S[i][j] = k[i]*delta[j] = 0.5*delta = [0.25, 0.5, 0.75, 1.0]  (every i)
//   out0[j] = S[0][j]*q[0] * 0.5        = [0.125, 0.25, 0.375, 0.5]
//
// t=1:
//   S *= 0.5                            = [0.125, 0.25, 0.375, 0.5]
//   sum_i S[i][j]*k[i] = 2*S[0][j]      = [0.25, 0.5, 0.75, 1.0]
//   delta = (v1 - sum)*0.5              = [0.875, 1.75, 2.625, 3.5]
//   S[i][j] += k[i]*delta[j]            = [0.5625, 1.125, 1.6875, 2.25]
//   out1[j] = S[0][j]*0.5               = [0.28125, 0.5625, 0.84375, 1.125]

const float kLogHalf = -0.693147f;  // exp() == 0.5

void test_two_token_sequence() {
    std::printf("[1] two-token sequence vs hand-computed llama.cpp reference\n");

    const int seq_len = 2, num_heads = 1, head_dim = 4;
    const int d_inner = num_heads * head_dim;  // 4

    // q, k, v laid out [seq_len, d_inner]
    std::vector<float> q = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f};
    std::vector<float> k = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<float> v = {1.f, 2.f, 3.f, 4.f, 2.f, 4.f, 6.f, 8.f};

    // decay (log-gate) [seq_len, d_inner]
    std::vector<float> decay(8, kLogHalf);
    // beta [seq_len, num_heads]
    std::vector<float> beta = {0.5f, 0.5f};

    std::vector<float> state(num_heads * head_dim * head_dim, 0.0f);

    auto out = ops::gdn_recurrence(make_tensor(q, {seq_len, d_inner}),
                                   make_tensor(k, {seq_len, d_inner}),
                                   make_tensor(v, {seq_len, d_inner}),
                                   make_tensor(decay, {seq_len, d_inner}),
                                   make_tensor(beta, {seq_len, num_heads}),
                                   state.data(), seq_len, num_heads, head_dim, 0.0f);
    check(out != nullptr, "gdn_recurrence returns a tensor");
    if (!out) return;
    check(out->numel() == (int64_t)(seq_len * d_inner), "output size == seq_len*d_inner");

    auto got = flatten2(out);

    const std::vector<float> want0 = {0.125f, 0.25f, 0.375f, 0.5f};
    const std::vector<float> want1 = {0.28125f, 0.5625f, 0.84375f, 1.125f};

    bool ok0 = true, ok1 = true;
    for (int j = 0; j < head_dim; ++j) {
        if (!near(got[j], want0[j])) ok0 = false;
        if (!near(got[d_inner + j], want1[j])) ok1 = false;
    }
    check(ok0, "token 0 matches hand-computed [0.125 0.25 0.375 0.5]");
    check(ok1, "token 1 matches hand-computed [0.28125 0.5625 0.84375 1.125]");

    if (!ok0) {
        std::printf("      got t0:");
        for (int j = 0; j < head_dim; ++j) std::printf(" %.6f", got[j]);
        std::printf("\n");
    }
    if (!ok1) {
        std::printf("      got t1:");
        for (int j = 0; j < head_dim; ++j) std::printf(" %.6f", got[d_inner + j]);
        std::printf("\n");
    }
}

// Decoding one token at a time must reproduce the batched result: this pins down
// that the recurrent state S is carried correctly across calls (decode path).
void test_state_carry_across_calls() {
    std::printf("[2] step-by-step decode reproduces the batched result\n");

    const int num_heads = 1, head_dim = 4, d_inner = 4;

    std::vector<float> q1 = {1.f, 0.f, 0.f, 0.f};
    std::vector<float> k1 = {0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<float> v1 = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> d1(4, kLogHalf);
    std::vector<float> b1 = {0.5f};

    std::vector<float> q2 = {1.f, 0.f, 0.f, 0.f};
    std::vector<float> k2 = {0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<float> v2 = {2.f, 4.f, 6.f, 8.f};
    std::vector<float> d2(4, kLogHalf);
    std::vector<float> b2 = {0.5f};

    std::vector<float> state(num_heads * head_dim * head_dim, 0.0f);

    auto o1 = ops::gdn_recurrence(make_tensor(q1, {1, d_inner}),
                                  make_tensor(k1, {1, d_inner}),
                                  make_tensor(v1, {1, d_inner}),
                                  make_tensor(d1, {1, d_inner}),
                                  make_tensor(b1, {1, num_heads}), state.data(), 1,
                                  num_heads, head_dim, 0.0f);
    check(o1 != nullptr, "first decode step returns a tensor");
    if (!o1) return;
    auto g1 = flatten2(o1);

    const std::vector<float> want0 = {0.125f, 0.25f, 0.375f, 0.5f};
    bool ok0 = true;
    for (int j = 0; j < head_dim; ++j)
        if (!near(g1[j], want0[j])) ok0 = false;
    check(ok0, "step 1 == batched token 0");

    auto o2 = ops::gdn_recurrence(make_tensor(q2, {1, d_inner}),
                                  make_tensor(k2, {1, d_inner}),
                                  make_tensor(v2, {1, d_inner}),
                                  make_tensor(d2, {1, d_inner}),
                                  make_tensor(b2, {1, num_heads}), state.data(), 1,
                                  num_heads, head_dim, 0.0f);
    check(o2 != nullptr, "second decode step returns a tensor");
    if (!o2) return;
    auto g2 = flatten2(o2);

    const std::vector<float> want1 = {0.28125f, 0.5625f, 0.84375f, 1.125f};
    bool ok1 = true;
    for (int j = 0; j < head_dim; ++j)
        if (!near(g2[j], want1[j])) ok1 = false;
    check(ok1, "step 2 == batched token 1 (state carried across calls)");
    if (!ok1) {
        std::printf("      got:");
        for (int j = 0; j < head_dim; ++j) std::printf(" %.6f", g2[j]);
        std::printf("\n");
    }
}

}  // namespace

int main() {
    test_two_token_sequence();
    test_state_carry_across_calls();
    std::printf("gdn_reference: failures=%d\n", g_failures);
    std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
