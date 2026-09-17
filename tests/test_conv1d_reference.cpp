// Reference cross-check for the BailingMoE3 / KDA depthwise causal conv1d
// against llama.cpp:
//
//   bailingmoe3.cpp:197   conv_x = concat(conv_state, x_proj, 0)
//   bailingmoe3.cpp:213   out    = ggml_ssm_conv(conv_x, conv_weight)
//   bailingmoe3.cpp:214   out    = ggml_silu(out)
//   ops.cpp:9741-9751     out[ch][t] = sum_{i<nc} conv_x[t+i][ch] * w[ch*nc + i]
//
// Two properties are pinned down:
//   1) the sliding window is [state_oldest .. state_newest, x[t]] and the weight
//      tap order is w[0]..w[nc-1] with w[nc-1] multiplying the current input;
//   2) the conv state stores the RAW pre-conv projection (x_proj), NOT the
//      post-SiLU output -- llama.cpp snapshots conv_x, whose tail is x_proj.
//      Because forge's kernel is in-place, writing the SiLU result before
//      shifting the state would silently feed SiLU'd history to later tokens.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "forge/tensor.h"
#include "operators/cpu/conv.h"

namespace {

using namespace forge;

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

bool near(float a, float b, float tol = 1e-4f) {
    const float scale = std::fmax(1.0f, std::fabs(b));
    return std::fabs(a - b) <= tol * scale;
}

TensorPtr make_tensor(const std::vector<float>& data, const std::vector<int64_t>& shape) {
    auto t = std::make_shared<Tensor>(DataType::FP32, shape, DeviceType::CPU);
    std::memcpy(t->data(), data.data(), data.size() * sizeof(float));
    return t;
}

std::vector<float> flatten(const TensorPtr& t) {
    const float* p = static_cast<const float*>(t->data());
    return std::vector<float>(p, p + t->numel());
}

// Independent reference following llama.cpp's semantics. `x` is the raw
// (pre-conv) input and is NOT modified; `state` is updated in place.
void ref_conv1d(const std::vector<float>& x, const std::vector<float>& w,
                std::vector<float>& state, int seq_len, int dim, int k_size,
                std::vector<float>& out) {
    const int state_len = k_size - 1;
    out.assign((size_t) seq_len * dim, 0.0f);

    for (int s = 0; s < seq_len; ++s) {
        for (int c = 0; c < dim; ++c) {
            float val = 0.0f;
            for (int k = 0; k < k_size; ++k) {
                // window: oldest state ... newest state, then current input
                const float xv = (k < state_len) ? state[k * dim + c] : x[s * dim + c];
                val += xv * w[c * k_size + k];
            }
            val = val / (1.0f + std::exp(-val));  // SiLU
            out[s * dim + c] = val;

            // state shift: keep the RAW input, not the SiLU output
            const float raw = x[s * dim + c];
            for (int k = 0; k < state_len - 1; ++k) state[k * dim + c] = state[(k + 1) * dim + c];
            state[(state_len - 1) * dim + c] = raw;
        }
    }
}

void test_conv_window_and_state() {
    std::printf("[1] depthwise causal conv1d vs llama.cpp reference\n");

    const int dim = 2;       // d_inner
    const int k_size = 4;    // d_conv
    const int seq_len = 2;

    // state layout [time * dim + channel], 3 (k_size-1) timesteps of history
    std::vector<float> state = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    // input layout [token * dim + channel]
    const std::vector<float> x = {10.f, 20.f, 30.f, 40.f};
    // weight layout [channel * k_size + tap]
    const std::vector<float> w = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

    // reference (operates on copies)
    std::vector<float> ref_state = state;
    std::vector<float> want;
    ref_conv1d(x, w, ref_state, seq_len, dim, k_size, want);

    // forge (in-place on x)
    auto xt = make_tensor(x, {seq_len, dim});
    auto wt = make_tensor(w, {dim, k_size});
    ops::conv1d_depthwise_silu(xt, wt, state.data(), seq_len, dim, k_size);

    auto got = flatten(xt);

    bool ok_out = true;
    for (size_t i = 0; i < want.size(); ++i) {
        if (!near(got[i], want[i])) ok_out = false;
    }
    check(ok_out, "conv+SiLU output matches reference (tap order & window)");
    if (!ok_out) {
        std::printf("      want:");
        for (float v : want) std::printf(" %.5f", v);
        std::printf("\n      got :");
        for (float v : got) std::printf(" %.5f", v);
        std::printf("\n");
    }

    // The state must hold the raw pre-conv inputs after processing.
    bool ok_state = true;
    for (size_t i = 0; i < ref_state.size(); ++i) {
        if (!near(state[i], ref_state[i])) ok_state = false;
    }
    check(ok_state, "conv state stores RAW pre-conv input (not post-SiLU)");
    if (!ok_state) {
        std::printf("      want state:");
        for (float v : ref_state) std::printf(" %.5f", v);
        std::printf("\n      got  state:");
        for (float v : state) std::printf(" %.5f", v);
        std::printf("\n");
    }
}

// A single-token call from a zero state must leave raw x in the state: this is
// the decode path, where a polluted state corrupts every following token.
void test_single_token_state_is_raw() {
    std::printf("[2] single-token decode leaves raw input in conv state\n");

    const int dim = 1, k_size = 4, seq_len = 1;
    const int state_len = k_size - 1;

    std::vector<float> state(state_len * dim, 0.0f);  // zero history
    const std::vector<float> x = {2.0f};
    const std::vector<float> w = {0.0f, 0.0f, 0.0f, 1.0f};  // identity on current input

    auto xt = make_tensor(x, {seq_len, dim});
    auto wt = make_tensor(w, {dim, k_size});
    ops::conv1d_depthwise_silu(xt, wt, state.data(), seq_len, dim, k_size);

    // conv = 2.0 * 1.0 = 2.0 ; SiLU(2.0) = 2.0/(1+e^-2) = 1.7616
    auto got = flatten(xt);
    check(near(got[0], 2.0f / (1.0f + std::exp(-2.0f))), "output == SiLU(2.0)");

    // state must be the raw 2.0, NOT the SiLU'd 1.7616
    check(near(state[state_len - 1], 2.0f),
          "newest state slot == raw input 2.0 (not SiLU'd 1.7616)");
    if (!near(state[state_len - 1], 2.0f)) {
        std::printf("      got state=%.6f\n", state[state_len - 1]);
    }
}

}  // namespace

int main() {
    test_conv_window_and_state();
    test_single_token_state_is_raw();
    std::printf("conv1d_reference: failures=%d\n", g_failures);
    std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
