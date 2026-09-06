// =========================================================================
// test_kernel_selector.cpp — Phase 4 unified KernelSelector unit tests
//
// Covers:
//   1. The pure policy rules folded in from Phase 3/3.5 — decode GEMV family,
//      prefill MMQ/TensorCore boundaries (incl. the sm_arch dimension), the
//      FP32 large-N threshold, and the MoE dtype support set.
//   2. Env overrides (FORGE_MATMUL_PATH / FORGE_FORCE_TC / FORGE_FORCE_MMQ)
//      used by the benchmark harness stay authoritative.
//   3. The Phase 4-B measured-dispatch cache: identical problems select the
//      same variant twice, cache hit statistics are observable, recorded
//      measured winners override the heuristic, and reset clears everything.
//
// Pure policy / host logic only: no kernels are launched. The executable links
// the CUDA libraries for the dispatch-table lookups.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "forge/kernel_selector.h"
#include "forge/types.h"

namespace {

int failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

forge::KernelProblem matmul_problem(forge::DataType dt, int M, int N, int K, int sm = 89) {
    forge::KernelProblem p;
    p.domain = forge::KernelDomain::MatMul;
    p.dtype = dt;
    p.M = M;
    p.N = N;
    p.K = K;
    p.sm_arch = sm;
    return p;
}

forge::KernelProblem moe_problem(forge::DataType dt) {
    forge::KernelProblem p;
    p.domain = forge::KernelDomain::MoE;
    p.dtype = dt;
    p.M = 16;
    p.N = 4096;
    p.K = 4096;
    p.sm_arch = 89;
    return p;
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    std::string s = std::string(name) + "=" + value;
    _putenv(s.c_str());
#else
    setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#ifdef _WIN32
    std::string s = std::string(name) + "=";
    _putenv(s.c_str());
#else
    unsetenv(name);
#endif
}

// ---- 1. Pure policy: decode GEMV family ----

void test_decode_policy() {
    // FP32 weights: custom GEMV below/equal the large-N threshold, cuBLAS above.
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::FP32, 1, 4000, 4096)) ==
          forge::KernelVariant::GEMV_FP32_TRANSB);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::FP32, 1, 4096, 4096)) ==
          forge::KernelVariant::GEMV_FP32_TRANSB);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::FP32, 1, 16384, 4096)) ==
          forge::KernelVariant::GEMV_FP32_CUBLAS);

    // Quantized weights with a dedicated single-row kernel -> Q8_1 GEMV.
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q8_0, 1, 4096, 4096)) ==
          forge::KernelVariant::GEMV_Q8_1);

    // FP32 prefill -> cuBLAS.
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::FP32, 8, 4096, 4096)) ==
          forge::KernelVariant::MATMUL_CUBLAS);
}

// ---- 1. Pure policy: prefill MMQ / TensorCore rule ----

void test_prefill_rule() {
    // Phase 5-C per-dtype conservative thresholds (see kernel_selector.cpp
    // kTcMinM): Q8_0/Q5_K/Q6_K -> TensorCore from M >= 96, Q4_K/Q4_0 -> TC from
    // M >= 128. Each threshold carries a safety margin above its measured
    // MMQ->TC crossover band (Q8_0 ~48-64, Q5_K ~64-80, Q6_K ~80, Q4_K/Q4_0
    // ~96-110) and is NOT the crossover point; the 4-B autotuner owns the
    // fine-grained selection inside the bands once opted in.

    // Q4_0 / Q4_K: high thresholds (measured ~96-110 + margin -> 128).
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_0, 512, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_0, 128, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_0, 96, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_0, 32, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_K, 128, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_K, 96, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_K, 64, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ);

    // Q8_0: low threshold (measured ~48-64 + margin -> 96).
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q8_0, 128, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q8_0, 96, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q8_0, 64, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q8_0, 32, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ);

    // Q6_K: measured crossover ~M=80 (Phase 5-A best-of-5 interleaved scan) ->
    // conservative threshold 96 (TC already 1.28x+ at 96, and 4-B autotune
    // measures the 96-112 band anyway when opted in).
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q6_K, 128, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q6_K, 96, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q6_K, 64, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q6_K, 32, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ);

    // Q5_K: observed band M≈64-80 (Phase 5-B, best-of-3 reports ~64, best-of-5
    // ~80 -- a band, not an exact integer) -> conservative threshold 96.
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q5_K, 128, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q5_K, 96, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q5_K, 64, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q5_K, 32, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ);

    // Plateau coverage for the 5-D acceptance grid rows M=200/512: every dtype
    // stays on TC well past its conservative threshold (Q4_0's M=512 check is
    // the block above this one).
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_0, 200, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_K, 200, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_K, 512, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q8_0, 200, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q8_0, 512, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q6_K, 200, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q6_K, 512, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q5_K, 200, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q5_K, 512, 4096, 4096, 89)) ==
          forge::KernelVariant::MMQ_TENSORCORE);

    // Pre-TensorCore hardware (sm75): always MMQ even for large M.
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_0, 512, 4096, 4096, 75)) ==
          forge::KernelVariant::MMQ);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q8_0, 512, 4096, 4096, 75)) ==
          forge::KernelVariant::MMQ);
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q6_K, 512, 4096, 4096, 75)) ==
          forge::KernelVariant::MMQ);

    // Same problem twice -> same variant (pure policy determinism).
    CHECK(forge::select_kernel(matmul_problem(forge::DataType::Q4_0, 256, 4096, 4096, 89)) ==
          forge::select_kernel(matmul_problem(forge::DataType::Q4_0, 256, 4096, 4096, 89)));
}

// ---- 1. Pure policy: MoE dtype support set ----

void test_moe_policy() {
    CHECK(forge::select_kernel(moe_problem(forge::DataType::Q4_0)) ==
          forge::KernelVariant::MOE_EXPERT_GEMV);
    CHECK(forge::select_kernel(moe_problem(forge::DataType::Q4_K)) ==
          forge::KernelVariant::MOE_EXPERT_GEMV);
    CHECK(forge::select_kernel(moe_problem(forge::DataType::Q6_K)) ==
          forge::KernelVariant::MOE_EXPERT_GEMV);
    CHECK(forge::select_kernel(moe_problem(forge::DataType::Q8_0)) ==
          forge::KernelVariant::MOE_EXPERT_GEMV);
    // Anything else is silently unsupported (caller no-ops, as before).
    CHECK(forge::select_kernel(moe_problem(forge::DataType::Q5_1)) ==
          forge::KernelVariant::UNSUPPORTED);
    CHECK(forge::select_kernel(moe_problem(forge::DataType::FP32)) ==
          forge::KernelVariant::UNSUPPORTED);
}

// ---- 1. Names ----

void test_names() {
    CHECK(std::strcmp(forge::kernel_variant_name(forge::KernelVariant::MMQ), "MMQ") == 0);
    CHECK(std::strcmp(forge::kernel_variant_name(forge::KernelVariant::MMQ_TENSORCORE),
                      "MMQ_TENSORCORE") == 0);
    CHECK(std::strcmp(forge::kernel_variant_name(forge::KernelVariant::MOE_EXPERT_GEMV),
                      "MOE_EXPERT_GEMV") == 0);
    CHECK(std::strcmp(forge::kernel_variant_name(forge::KernelVariant::UNSUPPORTED),
                      "UNSUPPORTED") == 0);
    CHECK(std::strcmp(forge::kernel_domain_name(forge::KernelDomain::MatMul), "matmul") == 0);
    CHECK(std::strcmp(forge::kernel_domain_name(forge::KernelDomain::MoE), "moe") == 0);
}

// ---- 2. Env overrides stay authoritative ----

void test_env_overrides() {
    auto p_large = matmul_problem(forge::DataType::Q4_0, 512, 4096, 4096, 89);  // rule: TC
    auto p_small = matmul_problem(forge::DataType::Q4_0, 32, 4096, 4096, 89);   // rule: MMQ

    set_env("FORGE_FORCE_MMQ", "1");
    CHECK(forge::select_kernel(p_large) == forge::KernelVariant::MMQ);
    unset_env("FORGE_FORCE_MMQ");

    set_env("FORGE_FORCE_TC", "1");
    CHECK(forge::select_kernel(p_small) == forge::KernelVariant::MMQ_TENSORCORE);
    unset_env("FORGE_FORCE_TC");

    set_env("FORGE_MATMUL_PATH", "mmq");
    CHECK(forge::select_kernel(p_large) == forge::KernelVariant::MMQ);
    set_env("FORGE_MATMUL_PATH", "tc");
    CHECK(forge::select_kernel(p_small) == forge::KernelVariant::MMQ_TENSORCORE);
    unset_env("FORGE_MATMUL_PATH");
}

// ---- 3. Phase 4-B measured-dispatch cache ----

void test_cache() {
    forge::kernel_cache_reset();
    auto stats = forge::kernel_cache_stats();
    CHECK(stats.lookups == 0 && stats.hits == 0 && stats.misses == 0 &&
          stats.measured == 0 && stats.tc_selected == 0 && stats.dp4a_selected == 0);

    // First consult of an eligible signature: cache miss, auto rule recorded.
    const auto q0 = matmul_problem(forge::DataType::Q4_0, 128, 4096, 4096, 89);
    bool hit = false;
    CHECK(forge::select_kernel_cached(q0, &hit) == forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(!hit);
    // Pure policy is untouched by the cache.
    CHECK(forge::select_kernel(q0) == forge::KernelVariant::MMQ_TENSORCORE);

    // Same problem again: same variant, served from the cache (hit).
    CHECK(forge::select_kernel_cached(q0, &hit) == forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(hit);

    // A different M for the same dtype/shape is a distinct signature.
    const auto q0_small = matmul_problem(forge::DataType::Q4_0, 32, 4096, 4096, 89);
    CHECK(forge::select_kernel_cached(q0_small, &hit) == forge::KernelVariant::MMQ);
    CHECK(!hit);

    // Recorded (measured) winner overrides the heuristic for that signature.
    forge::kernel_cache_record(q0, forge::KernelVariant::MMQ);
    CHECK(forge::select_kernel_cached(q0, &hit) == forge::KernelVariant::MMQ);
    CHECK(hit);
    CHECK(forge::select_kernel(q0) == forge::KernelVariant::MMQ_TENSORCORE);  // pure unaffected

    // Non-eligible slices never touch the cache (decode + MoE consult bypass).
    const auto decode = matmul_problem(forge::DataType::Q8_0, 1, 4096, 4096, 89);
    CHECK(forge::select_kernel_cached(decode, &hit) == forge::KernelVariant::GEMV_Q8_1);
    CHECK(!hit);
    CHECK(forge::select_kernel_cached(moe_problem(forge::DataType::Q8_0), &hit) ==
          forge::KernelVariant::MOE_EXPERT_GEMV);
    CHECK(!hit);

    stats = forge::kernel_cache_stats();
    // Four cacheable consults: q0 (miss, hit), q0_small (miss), q0 after the
    // override record (hit). decode + MoE bypass the cache entirely.
    CHECK(stats.lookups == 4);
    CHECK(stats.hits == 2);
    CHECK(stats.misses == 2);
    CHECK(stats.measured == 1);
    CHECK(stats.tc_selected == 0);    // the single record chose MMQ (dp4a)
    CHECK(stats.dp4a_selected == 1);

    // A TensorCore record is counted separately from a DP4A record.
    forge::kernel_cache_record(q0_small, forge::KernelVariant::MMQ_TENSORCORE);
    stats = forge::kernel_cache_stats();
    CHECK(stats.measured == 2);
    CHECK(stats.tc_selected == 1);
    CHECK(stats.dp4a_selected == 1);

    // Phase 5-C: a measured autotune result may also override the per-dtype
    // static threshold *downward* (pure rule: Q8_0 below M=96 -> MMQ; a real
    // workload may measure TC faster at M=64 and record the flip).
    const auto q8_64 = matmul_problem(forge::DataType::Q8_0, 64, 4096, 4096, 89);
    CHECK(forge::select_kernel(q8_64) == forge::KernelVariant::MMQ);  // pure rule
    forge::kernel_cache_record(q8_64, forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(forge::select_kernel_cached(q8_64, &hit) == forge::KernelVariant::MMQ_TENSORCORE);
    CHECK(hit);
    stats = forge::kernel_cache_stats();
    CHECK(stats.measured == 3);
    CHECK(stats.tc_selected == 2);
    CHECK(stats.dp4a_selected == 1);

    forge::kernel_cache_reset();
    stats = forge::kernel_cache_stats();
    CHECK(stats.lookups == 0 && stats.hits == 0 && stats.misses == 0 &&
          stats.measured == 0 && stats.tc_selected == 0 && stats.dp4a_selected == 0);
}

}  // namespace

int main() {
    std::printf("kernel-selector: decode/prefill/MoE policy + env overrides + cache\n");

    test_names();
    test_decode_policy();
    test_prefill_rule();
    test_moe_policy();
    test_env_overrides();
    test_cache();

    if (failures == 0)
        std::printf("kernel-selector: OK\n");
    else
        std::printf("kernel-selector: %d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
