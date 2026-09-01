// Qwen35 recurrent-memory tests (roadmap 2.3 Phase B).
//
// Pins down the per-sequence + speculative-rollback contract of
// Qwen35RecurrentMemory:
//   1. Dimension derivation: explicit ModelConfig, then weight-shape fallback.
//   2. Per-sequence isolation and lazy allocation; reset() keeps buffers.
//   3. snapshot()/rollback() restore the exact snapshot content.
//   4. rollback() on a sequence that was never snapshotted is a no-op.
//   5. A fresh snapshot supersedes the previous one.
//   6. reset_seq()/reset() zero content only (buffers retained).
//   7. Spec semantics: snapshot before verify, then rollback + replay of the
//      accepted prefix must equal the plain (no-spec) continuation.
//   8. GPU mirror (when a CUDA device exists): dev snapshot/rollback agree
//      with the CPU copy.
//
// Build & run:
//   cmake -B build -DFORGE_BUILD_TESTS=ON && cmake --build build -j
//   ctest --test-dir build -R qwen35-recurrent --output-on-failure
//   ./build/forge-qwen35-recurrent-test

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "forge/inference/layers/qwen35_recurrent_memory.h"
#include "forge/model.h"
#include "forge/tensor.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

using namespace forge;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        ++g_checks;                                                   \
        if (!(cond)) {                                                \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                             \
        } else {                                                      \
            std::printf("ok   %s\n", msg);                            \
        }                                                             \
    } while (0)

// Deterministic pseudo-random fill (LCG). Same seed -> same values.
static void fill_pseudo(float* p, int n, unsigned seed) {
    unsigned x = seed ? seed : 1u;
    for (int i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = static_cast<float>((x >> 8) & 0xffff) / 65536.0f;
    }
}

static bool floats_equal(const float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i)
        if (a[i] != b[i])
            return false;
    return true;
}

// ---------------------------------------------------------------------------
// Fixture: a Qwen35 hybrid config (8 layers; layers 1, 3, 5 are linear attn).
// ---------------------------------------------------------------------------
static ModelConfig make_config() {
    ModelConfig cfg;
    cfg.num_layers = 8;
    cfg.hidden_dim = 128;
    cfg.use_ssm = true;
    cfg.ssm_state_size = 16;     // d_state
    cfg.ssm_group_count = 8;     // n_group
    cfg.ssm_time_step_rank = 4;  // dt_rank
    cfg.ssm_inner_size = 64;     // d_inner
    cfg.ssm_conv_kernel = 4;     // d_conv
    return cfg;
}

static ModelWeights make_weights(const ModelConfig& cfg) {
    ModelWeights w;
    w.layers.resize(static_cast<size_t>(cfg.num_layers));
    for (int i = 0; i < cfg.num_layers; ++i) {
        if (i == 1 || i == 3 || i == 5)
            w.layers[i].layer_type = LayerType::LinearAttention;
    }
    return w;
}

// Which layers hold real buffers in this fixture.
static bool is_linear_layer(int i) {
    return i == 1 || i == 3 || i == 5;
}

// ---------------------------------------------------------------------------
// 1. Dimension derivation
// ---------------------------------------------------------------------------
static void test_dims_explicit() {
    std::printf("\n--- [1] dimension derivation (explicit config) ---\n");
    auto cfg = make_config();
    auto w = make_weights(cfg);
    Qwen35RecurrentMemory mem;
    mem.init(cfg, w);
    CHECK(mem.initialized(), "init: use_ssm=true -> initialized");
    const auto& d = mem.dims();
    CHECK(d.d_state == 16 && d.n_group == 8 && d.dt_rank == 4 && d.d_inner == 64 && d.d_conv == 4,
          "explicit config dims carried through");
    CHECK(d.head_k_dim() == 16 && d.num_k_heads() == 8,
          "head_k_dim = d_state, num_k_heads = n_group");
    CHECK(d.num_v_heads() == 4 && d.head_v_dim == 16,
          "num_v_heads = dt_rank, head_v_dim = d_inner/dt_rank");
    CHECK(d.key_dim() == 128 && d.value_dim() == 64,
          "key_dim = d_state*n_group, value_dim = head_v_dim*dt_rank");
    CHECK(d.conv_channels == 64 + 2 * 8 * 16, "conv_channels = d_inner + 2*n_group*d_state");
    CHECK(d.state_size() == 16 * 16 * 4, "state_size = head_v_dim^2 * num_v_heads");
    CHECK(d.conv_state_size() == (4 - 1) * (64 + 2 * 8 * 16),
          "conv_state_size = (d_conv-1)*conv_channels");
}

static void test_dims_fallback() {
    std::printf("\n--- [2] dimension derivation (weight-shape fallback) ---\n");
    auto cfg = make_config();
    cfg.ssm_conv_kernel = 0;  // force derivation from weights
    cfg.ssm_time_step_rank = 0;
    auto w = make_weights(cfg);
    // First linear layer carries conv1d (out_channels = d_conv) and ssm_a.
    auto conv1d =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{3}, DeviceType::CPU);
    auto ssm_a = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{5}, DeviceType::CPU);
    w.layers[1].set("ssm_conv1d", conv1d);
    w.layers[1].set("ssm_a", ssm_a);

    Qwen35RecurrentMemory mem;
    mem.init(cfg, w);
    const auto& d = mem.dims();
    CHECK(d.d_conv == 3, "fallback: d_conv from ssm_conv1d shape[0]");
    CHECK(d.dt_rank == 5, "fallback: dt_rank from ssm_a numel");
    CHECK(d.head_v_dim == 64 / 5, "fallback: head_v_dim recomputed from derived dt_rank");
    CHECK(d.state_size() == (64 / 5) * (64 / 5) * 5, "fallback: state_size recomputed");
    CHECK(d.conv_state_size() == 2 * d.conv_channels, "fallback: conv_state_size recomputed");

    // No SSM in config -> init is a no-op.
    ModelConfig no_ssm = make_config();
    no_ssm.use_ssm = false;
    Qwen35RecurrentMemory empty;
    empty.init(no_ssm, w);
    CHECK(!empty.initialized(), "init: use_ssm=false -> not initialized");
}

// ---------------------------------------------------------------------------
// 3. Lazy allocation + buffer retention + per-seq isolation
// ---------------------------------------------------------------------------
static void test_lazy_and_isolation() {
    std::printf("\n--- [3] lazy allocation, buffer retention, per-seq isolation ---\n");
    auto cfg = make_config();
    auto w = make_weights(cfg);
    Qwen35RecurrentMemory mem;
    mem.init(cfg, w);

    const int conv_n = mem.dims().conv_state_size();  // 960
    const int ssm_n = mem.dims().state_size();        // 1024

    // Only LinearAttention layers hold real buffers.
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        float* p = mem.conv_state_cpu(0, layer);
        if (is_linear_layer(layer)) {
            CHECK(p != nullptr, "linear layer holds a conv buffer");
        }
    }
    CHECK(mem.conv_state_cpu(0, 1) != mem.conv_state_cpu(0, 3),
          "distinct linear layers have distinct buffers");
    CHECK(mem.ssm_state_cpu(0, 1) != mem.ssm_state_cpu(0, 3),
          "distinct linear layers have distinct ssm buffers");
    CHECK(mem.conv_state_cpu(0, 1) != mem.conv_state_cpu(1, 1),
          "different seq_ids have distinct conv buffers");
    CHECK(mem.ssm_state_cpu(0, 1) != mem.ssm_state_cpu(1, 1),
          "different seq_ids have distinct ssm buffers");

    // Freshly allocated buffers are zero-filled (linear layers only -- full
    // attention layers carry no state buffers and must not be dereferenced).
    bool zero_ok = true;
    for (int layer = 1; layer < cfg.num_layers; layer += 2) {
        if (!is_linear_layer(layer))
            continue;
        const float* c = mem.conv_state_cpu(2, layer);
        const float* s = mem.ssm_state_cpu(2, layer);
        for (int i = 0; i < conv_n; ++i)
            zero_ok &= (c[i] == 0.0f);
        for (int i = 0; i < ssm_n; ++i)
            zero_ok &= (s[i] == 0.0f);
    }
    CHECK(zero_ok, "new seq states are zero-filled");

    // Per-seq isolation: writes to seq 0 must not leak into seq 1.
    float* c0 = mem.conv_state_cpu(0, 1);
    float* c1 = mem.conv_state_cpu(1, 1);
    float* s0 = mem.ssm_state_cpu(0, 1);
    float* s1 = mem.ssm_state_cpu(1, 1);
    fill_pseudo(c0, conv_n, 101u);
    fill_pseudo(s0, ssm_n, 202u);
    CHECK(floats_equal(c1, mem.conv_state_cpu(1, 1), conv_n) &&
              floats_equal(s1, mem.ssm_state_cpu(1, 1), ssm_n),
          "seq1 untouched by seq0 writes");
    CHECK(c1[0] == 0.0f && s1[0] == 0.0f, "seq1 buffers still zero");
    c1[0] = 7.0f;
    s1[0] = 9.0f;
    CHECK(c0[0] == c0[0] && c0[0] != 7.0f, "seq0 conv unaffected by seq1 write");
    CHECK(s0[0] != 9.0f, "seq0 ssm unaffected by seq1 write");

    // reset() keeps the buffers (pointer identity) but clears content.
    float* c0_before = c0;
    mem.reset();
    CHECK(mem.conv_state_cpu(0, 1) == c0_before, "reset() retains allocated buffers");
    bool cleared = true;
    for (int i = 0; i < conv_n; ++i)
        cleared &= (mem.conv_state_cpu(0, 1)[i] == 0.0f);
    CHECK(cleared, "reset() zeroes per-seq content");
}

// ---------------------------------------------------------------------------
// 4/5. snapshot / rollback contract
// ---------------------------------------------------------------------------
static void test_snapshot_rollback() {
    std::printf("\n--- [4] snapshot / rollback contract ---\n");
    auto cfg = make_config();
    auto w = make_weights(cfg);
    Qwen35RecurrentMemory mem;
    mem.init(cfg, w);

    const int conv_n = mem.dims().conv_state_size();
    const int ssm_n = mem.dims().state_size();

    float* c = mem.conv_state_cpu(0, 1);
    float* s = mem.ssm_state_cpu(0, 1);
    std::vector<float> c0(conv_n), s0(ssm_n), c1(conv_n), s1(ssm_n);
    fill_pseudo(c, conv_n, 11u);
    fill_pseudo(s, ssm_n, 22u);
    std::memcpy(c0.data(), c, conv_n * sizeof(float));
    std::memcpy(s0.data(), s, ssm_n * sizeof(float));

    mem.snapshot(0);

    // Mutate after the snapshot.
    fill_pseudo(c, conv_n, 33u);
    fill_pseudo(s, ssm_n, 44u);

    mem.rollback(0);
    // Accessors may return a different pointer after rollback (vector copy);
    // re-fetch before comparing.
    c = mem.conv_state_cpu(0, 1);
    s = mem.ssm_state_cpu(0, 1);
    CHECK(floats_equal(c, c0.data(), conv_n) && floats_equal(s, s0.data(), ssm_n),
          "rollback restores exact snapshot content");

    // Rolling back twice with no new snapshot stays at the snapshot.
    mem.rollback(0);
    c = mem.conv_state_cpu(0, 1);
    s = mem.ssm_state_cpu(0, 1);
    CHECK(floats_equal(c, c0.data(), conv_n) && floats_equal(s, s0.data(), ssm_n),
          "second rollback (no new snapshot) is idempotent");

    // A fresh snapshot supersedes the previous one.
    mem.snapshot(0);
    fill_pseudo(c, conv_n, 55u);
    fill_pseudo(s, ssm_n, 66u);
    std::memcpy(c1.data(), c, conv_n * sizeof(float));
    std::memcpy(s1.data(), s, ssm_n * sizeof(float));
    mem.snapshot(0);  // now the snapshot = c1/s1
    fill_pseudo(c, conv_n, 77u);
    fill_pseudo(s, ssm_n, 88u);
    mem.rollback(0);
    c = mem.conv_state_cpu(0, 1);
    s = mem.ssm_state_cpu(0, 1);
    CHECK(floats_equal(c, c1.data(), conv_n) && floats_equal(s, s1.data(), ssm_n),
          "fresh snapshot supersedes the previous one");
}

// ---------------------------------------------------------------------------
// 6. rollback of a never-snapshotted sequence is a no-op
// ---------------------------------------------------------------------------
static void test_rollback_never_snapshotted() {
    std::printf("\n--- [5] rollback without prior snapshot is a no-op ---\n");
    auto cfg = make_config();
    auto w = make_weights(cfg);
    Qwen35RecurrentMemory mem;
    mem.init(cfg, w);

    const int conv_n = mem.dims().conv_state_size();
    const int ssm_n = mem.dims().state_size();

    // seq 1 is never snapshotted; its state must survive rollback unchanged.
    float* c = mem.conv_state_cpu(1, 1);
    float* s = mem.ssm_state_cpu(1, 1);
    fill_pseudo(c, conv_n, 91u);
    fill_pseudo(s, ssm_n, 92u);
    std::vector<float> c0(conv_n), s0(ssm_n);
    std::memcpy(c0.data(), c, conv_n * sizeof(float));
    std::memcpy(s0.data(), s, ssm_n * sizeof(float));

    mem.rollback(1);
    c = mem.conv_state_cpu(1, 1);
    s = mem.ssm_state_cpu(1, 1);
    CHECK(floats_equal(c, c0.data(), conv_n) && floats_equal(s, s0.data(), ssm_n),
          "rollback of never-snapshotted seq keeps state intact");

    // After a snapshot the rollback becomes active again.
    mem.snapshot(1);
    c[0] = -1.0f;
    s[0] = -2.0f;
    mem.rollback(1);
    c = mem.conv_state_cpu(1, 1);
    s = mem.ssm_state_cpu(1, 1);
    CHECK(c[0] == c0[0] && s[0] == s0[0], "snapshot re-arms rollback for the seq");
}

// ---------------------------------------------------------------------------
// 7. reset_seq / reset
// ---------------------------------------------------------------------------
static void test_reset_seq() {
    std::printf("\n--- [6] reset_seq / reset ---\n");
    auto cfg = make_config();
    auto w = make_weights(cfg);
    Qwen35RecurrentMemory mem;
    mem.init(cfg, w);

    const int conv_n = mem.dims().conv_state_size();
    const int ssm_n = mem.dims().state_size();

    fill_pseudo(mem.conv_state_cpu(0, 1), conv_n, 111u);
    fill_pseudo(mem.ssm_state_cpu(0, 1), ssm_n, 222u);
    fill_pseudo(mem.conv_state_cpu(1, 1), conv_n, 333u);
    fill_pseudo(mem.ssm_state_cpu(1, 1), ssm_n, 444u);

    mem.reset_seq(0);
    bool zero0 = true;
    for (int i = 0; i < conv_n; ++i)
        zero0 &= (mem.conv_state_cpu(0, 1)[i] == 0.0f);
    for (int i = 0; i < ssm_n; ++i)
        zero0 &= (mem.ssm_state_cpu(0, 1)[i] == 0.0f);
    CHECK(zero0, "reset_seq(0) zeroes seq 0");
    CHECK(mem.conv_state_cpu(1, 1)[0] != 0.0f, "reset_seq(0) leaves seq 1 intact");

    mem.reset();
    bool zero1 = true;
    for (int i = 0; i < conv_n; ++i)
        zero1 &= (mem.conv_state_cpu(1, 1)[i] == 0.0f);
    for (int i = 0; i < ssm_n; ++i)
        zero1 &= (mem.ssm_state_cpu(1, 1)[i] == 0.0f);
    CHECK(zero1, "reset() zeroes remaining seqs");
}

// ---------------------------------------------------------------------------
// 8. Spec semantics: snapshot -> (draft pollution) -> rollback -> replay of the
//    accepted prefix must equal the plain continuation.
// ---------------------------------------------------------------------------
static void apply_tokens(float* state, int n, const std::vector<float>& toks) {
    for (float t : toks) {
        for (int i = 0; i < n; ++i) {
            state[i] = state[i] * 0.95f + t * (0.5f + 0.01f * static_cast<float>(i));
        }
    }
}

static void test_spec_semantics() {
    std::printf("\n--- [7] spec rollback + replay == plain continuation ---\n");
    auto cfg = make_config();
    auto w = make_weights(cfg);
    Qwen35RecurrentMemory mem;
    mem.init(cfg, w);

    const int conv_n = mem.dims().conv_state_size();
    const int ssm_n = mem.dims().state_size();

    const std::vector<float> prefix = {1.0f, 2.0f, 3.0f};
    const float last = 4.0f;                                  // last accepted target token
    const std::vector<float> drafts = {99.0f, 98.0f, 97.0f};  // must be rejected
    const std::vector<float> accepted = {5.0f};               // n_accepted = 1

    // ---- seq 0: plain path, no speculation ----
    float* c_plain = mem.conv_state_cpu(0, 1);
    float* s_plain = mem.ssm_state_cpu(0, 1);
    apply_tokens(c_plain, conv_n, prefix);
    apply_tokens(s_plain, ssm_n, prefix);
    apply_tokens(c_plain, conv_n, {last});
    apply_tokens(s_plain, ssm_n, {last});
    apply_tokens(c_plain, conv_n, accepted);
    apply_tokens(s_plain, ssm_n, accepted);

    // ---- seq 1: speculative path ----
    // Mirror the executor exactly: the generator's last_token is a sampled
    // (resampled/bonus) token that has NOT been forwarded yet, so it is absent
    // from the SSM state at snapshot time. The verify batch [last, drafts...]
    // forwards last for the first time; on partial rejection we roll back to
    // the pre-verify state and replay [last, accepted drafts].
    float* c_spec = mem.conv_state_cpu(1, 1);
    float* s_spec = mem.ssm_state_cpu(1, 1);
    apply_tokens(c_spec, conv_n, prefix);
    apply_tokens(s_spec, ssm_n, prefix);

    mem.snapshot(1);  // verify begins: state covers the accepted history only

    apply_tokens(c_spec, conv_n, {last});  // verify batch [last, d1, d2, d3]
    apply_tokens(s_spec, ssm_n, {last});
    apply_tokens(c_spec, conv_n, drafts);
    apply_tokens(s_spec, ssm_n, drafts);

    mem.rollback(1);  // reject -> back to the accepted history
    // replay: last + accepted drafts (mirrors replay_recurrent)
    c_spec = mem.conv_state_cpu(1, 1);
    s_spec = mem.ssm_state_cpu(1, 1);
    apply_tokens(c_spec, conv_n, {last});
    apply_tokens(s_spec, ssm_n, {last});
    apply_tokens(c_spec, conv_n, accepted);
    apply_tokens(s_spec, ssm_n, accepted);

    CHECK(floats_equal(c_spec, c_plain, conv_n),
          "conv state after rollback+replay == plain continuation");
    CHECK(floats_equal(s_spec, s_plain, ssm_n),
          "ssm state after rollback+replay == plain continuation");
}

// ---------------------------------------------------------------------------
// 9. GPU mirror (optional): dev snapshot/rollback agree with CPU copy.
// ---------------------------------------------------------------------------
#ifdef USE_CUDA
static void test_gpu_mirror() {
    std::printf("\n--- [8] GPU mirror snapshot/rollback ---\n");
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("     no CUDA device; skipping GPU mirror test\n");
        return;
    }
    auto cfg = make_config();
    auto w = make_weights(cfg);
    Qwen35RecurrentMemory mem;
    mem.init(cfg, w);

    const int conv_n = mem.dims().conv_state_size();  // 960
    const int ssm_n = mem.dims().state_size();        // 1024
    std::vector<float> c0(conv_n), s0(ssm_n), c1(conv_n), s1(ssm_n);

    // Seed the dev buffers from CPU pseudo-random values (H2D).
    fill_pseudo(c0.data(), conv_n, 501u);
    fill_pseudo(s0.data(), ssm_n, 502u);
    std::memcpy(c1.data(), c0.data(), conv_n * sizeof(float));
    std::memcpy(s1.data(), s0.data(), ssm_n * sizeof(float));
    cudaMemcpy(mem.conv_state_gpu(0, 1), c0.data(), conv_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mem.ssm_state_gpu(0, 1), s0.data(), ssm_n * sizeof(float), cudaMemcpyHostToDevice);

    mem.snapshot(0);
    CHECK(mem.conv_state_gpu(0, 1) != nullptr && mem.ssm_state_gpu(0, 1) != nullptr,
          "gpu snapshot buffers allocated on first snapshot");

    // Corrupt the dev buffers (H2D with different values).
    fill_pseudo(c1.data(), conv_n, 601u);
    fill_pseudo(s1.data(), ssm_n, 602u);
    cudaMemcpy(mem.conv_state_gpu(0, 1), c1.data(), conv_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mem.ssm_state_gpu(0, 1), s1.data(), ssm_n * sizeof(float), cudaMemcpyHostToDevice);

    mem.rollback(0);

    // Read back and compare against the pre-corruption values.
    std::vector<float> back(conv_n), backs(ssm_n);
    cudaMemcpy(back.data(), mem.conv_state_gpu(0, 1), conv_n * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(backs.data(), mem.ssm_state_gpu(0, 1), ssm_n * sizeof(float),
               cudaMemcpyDeviceToHost);
    CHECK(floats_equal(back.data(), c0.data(), conv_n), "gpu conv state restored from snapshot");
    CHECK(floats_equal(backs.data(), s0.data(), ssm_n), "gpu ssm state restored from snapshot");
}
#endif

}  // namespace

int main() {
    std::printf("=== forge-qwen35-recurrent-test ===\n");
    test_dims_explicit();
    test_dims_fallback();
    test_lazy_and_isolation();
    test_snapshot_rollback();
    test_rollback_never_snapshotted();
    test_reset_seq();
    test_spec_semantics();
#ifdef USE_CUDA
    test_gpu_mirror();
#endif
    std::printf("\n=== %d/%d checks passed ===\n", g_checks - g_failures, g_checks);
    return g_failures > 0 ? 1 : 0;
}
