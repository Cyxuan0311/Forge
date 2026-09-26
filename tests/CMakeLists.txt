# Forge test suite.
#
# Responsibilities of THIS file:
#   * Compile the C++ test/bench sources under tests/ and register them with
#     CTest.
#   * Cross-compile architecture-specific kernel tests (ARM64 NEON, PPC64 VSX)
#     and run them under QEMU so they verify on x86 CI hosts.
#
# Responsibilities of the PARENT (top-level) CMakeLists.txt:
#   * Own the master on/off switch: option(FORGE_BUILD_TESTS ...) -- default OFF.
#   * Call enable_testing() and add_subdirectory(tests) only when enabled.
# This file assumes it is only ever included when FORGE_BUILD_TESTS is ON.
#
# Run everything with CTest:
#   cmake -B build -DFORGE_BUILD_TESTS=ON && cmake --build build -j
#   ctest --test-dir build --output-on-failure
#
# Benchmarks are long-running and gated by a separate switch
# (FORGE_BUILD_BENCHMARKS, default OFF) so they stay out of the default run.

option(FORGE_BUILD_BENCHMARKS "Build the micro-benchmark executables" OFF)

# Tests include src/operators headers through repo-relative paths such as
# "cpu/common/thread_pool.h"; make sure that base directory is on the include
# search path for every test target defined in this directory.
include_directories(${PROJECT_SOURCE_DIR}/src/operators)

# Libraries shared by the native (host-compiled) test executables.
set(FORGE_TEST_LIBS forge_infer forge_model forge_ops forge_core
                    forge_tokenizer forge_agent)

# -----------------------------------------------------------------------------
# Helper: link a native test like a regular executable (libs + optional
# CUDA/OpenMP). Libraries are wrapped in --start-group/--no-whole-group (GNU ld)
# because the arch-registration objects inside forge_model reference engine
# classes defined in forge_infer and vice versa, so plain left-to-right
# scanning can fail to resolve symbols.
# -----------------------------------------------------------------------------
function(forge_link_native target)
  target_include_directories(
    ${target}
    PRIVATE ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/operators/cpu
            ${PROJECT_SOURCE_DIR}/src/operators/cpu/common
            ${CUDAToolkit_INCLUDE_DIRS})
  if(APPLE)
    set(group_open "")
    set(group_close "")
  else()
    set(group_open "-Wl,--start-group")
    set(group_close "-Wl,--end-group")
  endif()
  target_link_libraries(
    ${target}
    PRIVATE ${group_open} ${FORGE_TEST_LIBS} ${group_close}
            $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>)
  if(CUDAToolkit_FOUND)
    target_compile_definitions(${target} PRIVATE USE_CUDA)
    target_link_libraries(
      ${target} PRIVATE CUDA::cudart $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>)
  endif()
endfunction()

# -----------------------------------------------------------------------------
# Native unit tests (compiled on the host).
#
# New host tests are added by listing their .cpp here. Cross-architecture
# sources (test_arm64_*, test_ppc64_*, tests/arch/**) MUST NOT be listed here
# -- they are handled by the QEMU cross section below.
# -----------------------------------------------------------------------------

# Speculative decoding (Phase 1): draft-token verification, batched forward
# parity, and greedy end-to-end parity. Needs tinyllama gguf; skipped gracefully
# when the fixture is absent.
add_executable(forge-spec-test test_spec_verify.cpp)
forge_link_native(forge-spec-test)
add_test(NAME spec-verify COMMAND forge-spec-test)

# DFlash / DSPark (Phase 2): iSWA routing + config parse. Part 1 loads the real
# dflash GGUF to verify parse_dflash_config populates the sliding-window
# metadata; Part 2 is a self-contained KVCache ring-buffer test proving the
# iSWA routing. Engine integration (Part 3) needs a matched drafter+target pair
# and is skipped unless FORGE_TEST_DFLASH_MODEL + FORGE_TEST_TARGET_MODEL are set.
add_executable(forge-dflash-test test_dflash.cpp)
forge_link_native(forge-dflash-test)
add_test(NAME dflash COMMAND forge-dflash-test)

# KV auto-capacity sizing (roadmap 1.2): per-sequence footprint arithmetic and
# the max_num_seqs derivation from a device memory budget. Host-only.
add_executable(forge-kv-sizing-test test_kv_sizing.cpp)
forge_link_native(forge-kv-sizing-test)
add_test(NAME kv-sizing COMMAND forge-kv-sizing-test)

# KV page offload (roadmap 1.1): whole-sequence host swap (evict/bring back)
# with bit-exact restore, so swapped-out requests resume with identical
# attention results. Host-only; exercised end-to-end through PagedKVStorage.
add_executable(forge-kv-offload-test test_kv_offload.cpp)
forge_link_native(forge-kv-offload-test)
add_test(NAME kv-offload COMMAND forge-kv-offload-test)

# Roadmap 2.2: contiguous KV cache defragmentation (host-only).
add_executable(forge-kv-defrag-test test_kv_defrag.cpp)
forge_link_native(forge-kv-defrag-test)
add_test(NAME kv-defrag COMMAND forge-kv-defrag-test)

# Roadmap 2.3 Phase B: Qwen35 recurrent-memory per-sequence state and
# speculative snapshot/rollback contract (host-only; GPU mirror optional).
add_executable(forge-qwen35-recurrent-test test_qwen35_recurrent.cpp)
forge_link_native(forge-qwen35-recurrent-test)
add_test(NAME qwen35-recurrent COMMAND forge-qwen35-recurrent-test)

# GDN (KDA gated delta net) reference cross-check: compares forge's
# gdn_recurrence() against hand-computed values derived from llama.cpp's
# ggml_compute_forward_gated_delta_net (KDA branch), and pins down that the
# recurrent state is carried correctly across step-by-step decode calls.
add_executable(forge-gdn-reference-test test_gdn_reference.cpp)
forge_link_native(forge-gdn-reference-test)
target_include_directories(
  forge-gdn-reference-test
  PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
add_test(NAME gdn-reference COMMAND forge-gdn-reference-test)

# Depthwise causal conv1d reference cross-check (sliding-window tap order + the
# requirement that the conv state keeps the RAW pre-conv projection, matching
# llama.cpp's conv_x snapshot of x_proj).
add_executable(forge-conv1d-reference-test test_conv1d_reference.cpp)
forge_link_native(forge-conv1d-reference-test)
target_include_directories(
  forge-conv1d-reference-test
  PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
add_test(NAME conv1d-reference COMMAND forge-conv1d-reference-test)

# IQ-family dequant row functions vs llama.cpp scalar reference: pins the
# IQ1_M qh pointer advance (must NOT double-step) and the IQ2_XXS group-scale
# formula d*(0.5+s)*0.25, plus the IQ4_NL 32-element block path.
add_executable(forge-dequant-reference-test test_dequant_reference.cpp)
forge_link_native(forge-dequant-reference-test)
target_include_directories(
  forge-dequant-reference-test
  PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
add_test(NAME dequant-reference COMMAND forge-dequant-reference-test)

# Roadmap 2.5: prefill whole-segment quantized write (requires CUDA).
add_executable(forge-kv-quant-segment-test test_kv_quant_segment.cpp)
forge_link_native(forge-kv-quant-segment-test)
add_test(NAME kv-quant-segment COMMAND forge-kv-quant-segment-test)

# Parallel GGUF scanner (inspect).
add_executable(forge-inspect-scanner-test test_inspect_scanner.cpp)
target_include_directories(
  forge-inspect-scanner-test
  PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
          ${CUDAToolkit_INCLUDE_DIRS})
if(APPLE)
  set(group_open "")
  set(group_close "")
else()
  set(group_open "-Wl,--start-group")
  set(group_close "-Wl,--end-group")
endif()
target_link_libraries(
  forge-inspect-scanner-test
  PRIVATE ${group_open} forge_inspect ${FORGE_TEST_LIBS} ${group_close})
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge-inspect-scanner-test PRIVATE USE_CUDA)
  target_link_libraries(forge-inspect-scanner-test PRIVATE CUDA::cudart)
endif()
add_test(NAME inspect-scanner COMMAND forge-inspect-scanner-test)

# Generic scalar fallback kernels (standalone, no libs).
add_executable(forge-generic-kernels-test test_generic_scalar_kernels.cpp)
target_compile_definitions(forge-generic-kernels-test
                           PRIVATE FORGE_ARCH_GENERIC)
target_include_directories(
  forge-generic-kernels-test PRIVATE ${PROJECT_SOURCE_DIR}
                                     ${PROJECT_SOURCE_DIR}/src/operators)
add_test(NAME generic-kernels COMMAND forge-generic-kernels-test)

# Phase 6-A-1: ComputeDTypePolicy + dtype contract (explicit reject instead of
# silently reinterpreting FP16/BF16 as float). Host logic only, no kernels.
# Compiles the small quant_traits support sources directly so the executable
# stays fast to build instead of linking the full library set.
add_executable(
  forge-dtype-contract-test
  test_dtype_contract.cpp
  ${PROJECT_SOURCE_DIR}/src/operators/cpu/quant_traits.cpp
  ${PROJECT_SOURCE_DIR}/src/operators/cpu/common/dequant.cpp
  ${PROJECT_SOURCE_DIR}/src/operators/cpu/common/quant_tables.cpp)
target_include_directories(forge-dtype-contract-test
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME dtype-contract COMMAND forge-dtype-contract-test)

# Agent module: tool-call dialect parsing (forge_text / xml_function /
# qwen_json + auto-detect) and <think> reasoning separation. Pure host logic —
# compiles the forge_agent sources directly so the executable stays fast to
# build (same pattern as dtype-contract above).
add_executable(
  forge-tool-dialects-test
  test_tool_dialects.cpp
  ${PROJECT_SOURCE_DIR}/src/agent/tool_dialect.cpp
  ${PROJECT_SOURCE_DIR}/src/agent/reasoning_filter.cpp
  ${PROJECT_SOURCE_DIR}/src/agent/agent_pipeline.cpp)
target_include_directories(
  forge-tool-dialects-test
  PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/third_party)
add_test(NAME tool-dialects COMMAND forge-tool-dialects-test)

# Q3_K MMQ decode bisect -- a host-only mechanical debugging tool (not a
# normal ctest assertion suite). Built on demand, not registered with CTest.
add_executable(
  forge-q3k-decode-bisect
  q3k_decode_bisect.cpp
  ${PROJECT_SOURCE_DIR}/src/operators/cpu/common/dequant.cpp
  ${PROJECT_SOURCE_DIR}/src/operators/cpu/common/quant_tables.cpp)
target_include_directories(
  forge-q3k-decode-bisect PRIVATE ${PROJECT_SOURCE_DIR}/include
                                  ${PROJECT_SOURCE_DIR})

# -----------------------------------------------------------------------------
# CUDA MMQ kernels vs CPU-dequant golden (built only when CUDA is present).
# The binary skips gracefully at runtime when no GPU is available.
# -----------------------------------------------------------------------------
if(CUDAToolkit_FOUND)
  add_executable(
    forge-mmq-ref-test
    test_mmq_reference.cpp
    ${PROJECT_SOURCE_DIR}/src/operators/cpu/common/dequant.cpp)
  forge_link_native(forge-mmq-ref-test)
  add_test(NAME mmq-ref COMMAND forge-mmq-ref-test)

  # Phase 4: unified kernel selector policy + measured-dispatch cache. Host
  # logic only (links the CUDA libraries for the dispatch-table lookups).
  add_executable(forge-kernel-selector-test test_kernel_selector.cpp)
  forge_link_native(forge-kernel-selector-test)
  add_test(NAME kernel-selector COMMAND forge-kernel-selector-test)

  # Phase 6-A-2: native FP16 RMSNorm correctness (FP32 reference parity) +
  # micro-bench (FP16 vs FP32 kernel). Skips gracefully when no GPU present.
  add_executable(forge-rms-norm-fp16-test test_rms_norm_fp16.cpp)
  forge_link_native(forge-rms-norm-fp16-test)
  add_test(NAME rms-norm-fp16 COMMAND forge-rms-norm-fp16-test)

  # Phase 6-B-2: native BF16 RMSNorm correctness (FP32 reference parity) +
  # micro-bench (BF16 vs FP32 + FP16 byte-parity sanity, two workload
  # regimes) + single-node graph smoke. Skips gracefully when no GPU present.
  add_executable(forge-rms-norm-bf16-test test_rms_norm_bf16.cpp)
  forge_link_native(forge-rms-norm-bf16-test)
  add_test(NAME rms-norm-bf16 COMMAND forge-rms-norm-bf16-test)

  # Phase 6-C-2: native FP8 E4M3 RMSNorm correctness (decoded-input FP32
  # reference, cosine primary gate) + 2-hop chain validation + stream A/B
  # determinism + graph smoke + micro-bench (FP8 vs BF16 vs FP32). Skips
  # gracefully when no GPU present.
  add_executable(forge-rms-norm-fp8-test test_rms_norm_fp8.cpp)
  forge_link_native(forge-rms-norm-fp8-test)
  add_test(NAME rms-norm-fp8 COMMAND forge-rms-norm-fp8-test)

  # Phase 6-C-3: native FP8 E4M3 activation family (silu / gelu / gelu_tanh +
  # gated silu_multiply / gelu_multiply) correctness (FP32 reference parity,
  # cosine primary gate) + micro-bench (FP8 vs FP32 + FP16 byte-parity sanity,
  # two workload regimes) + single-node FP8 SILU graph smoke + mixed-dtype loud
  # reject. Skips gracefully when no GPU present.
  add_executable(forge-activation-fp8-test test_activation_fp8.cpp)
  forge_link_native(forge-activation-fp8-test)
  add_test(NAME activation-fp8 COMMAND forge-activation-fp8-test)

  # Phase 6-A-3: native FP16 SiLU correctness (FP32 reference parity) +
  # micro-bench (FP16 vs FP32 kernel, two workload regimes). Skips gracefully
  # when no GPU present.
  add_executable(forge-silu-fp16-test test_silu_fp16.cpp)
  forge_link_native(forge-silu-fp16-test)
  add_test(NAME silu-fp16 COMMAND forge-silu-fp16-test)

  # Phase 6-A-4: native FP16 activation family (GELU / GELU-tanh / gated SiLU-GeGLU)
  # correctness (FP32 reference parity) + micro-bench (FP16 vs FP32, two regimes).
  # Skips gracefully when no GPU present.
  add_executable(forge-activation-fp16-test test_activation_fp16.cpp)
  forge_link_native(forge-activation-fp16-test)
  add_test(NAME activation-fp16 COMMAND forge-activation-fp16-test)

  # Phase 6-B-3: native BF16 activation family correctness (FP32 reference
  # parity) + micro-bench (BF16 vs FP32 + FP16 byte-parity sanity, two workload
  # regimes) + single-node SILU graph smoke. Skips gracefully w/o GPU.
  add_executable(forge-activation-bf16-test test_activation_bf16.cpp)
  forge_link_native(forge-activation-bf16-test)
  add_test(NAME activation-bf16 COMMAND forge-activation-bf16-test)

  # Phase 6-A-5: native FP16 Matmul (6-A-5.1 direct launch + 6-A-5.2 ops route +
  # 6-A-5.3 real FP16 compute chain RMSNorm->Matmul->SiLU). FP16 storage in/out,
  # FP32 accumulate; checks the no-implicit-FP32 dtype invariant end-to-end.
  add_executable(forge-matmul-fp16-test test_matmul_fp16.cpp)
  forge_link_native(forge-matmul-fp16-test)
  add_test(NAME matmul-fp16 COMMAND forge-matmul-fp16-test)

  add_executable(forge-matmul-fp16-chain-test test_matmul_fp16_chain.cpp)
  forge_link_native(forge-matmul-fp16-chain-test)
  add_test(NAME matmul-fp16-chain COMMAND forge-matmul-fp16-chain-test)

  # Phase 6-B-4.1/4.2/4.4: native BF16 matmul (CUDA_R_16BF, FP32 accumulate).
  # Direct launch + ops::matmul_transB route (BF16 in -> BF16 out, mixed loud)
  # + M-sweep bench (best-of-3 alternating arms). Skips gracefully w/o GPU.
  add_executable(forge-matmul-bf16-test test_matmul_bf16.cpp)
  forge_link_native(forge-matmul-bf16-test)
  add_test(NAME matmul-bf16 COMMAND forge-matmul-bf16-test)

  # Phase 6-B-4.3: native BF16 chain (RMSNorm -> Matmul -> silu_multiply).
  # Dtype invariant (all BF16) + FP32 reference parity. Skips gracefully w/o GPU.
  add_executable(forge-matmul-bf16-chain-test test_matmul_bf16_chain.cpp)
  forge_link_native(forge-matmul-bf16-chain-test)
  add_test(NAME matmul-bf16-chain COMMAND forge-matmul-bf16-chain-test)

  # Phase 6-A-6: native FP16 Graph / Binding integration. Runs the SAME operator
  # chain (RMSNorm -> Matmul -> SiLU) through ComputeGraph instead of standalone
  # calls: graph builder / planner / dst injection must not promote FP16 to FP32
  # (checked with the dtype trace), and CUDA Graph capture/replay must keep the
  # FP16 buffer address/size stable. Skips gracefully when no GPU present.
  add_executable(forge-graph-fp16-test test_graph_fp16.cpp)
  forge_link_native(forge-graph-fp16-test)
  add_test(NAME graph-fp16 COMMAND forge-graph-fp16-test)

  # Phase 6-B-5: native BF16 graph integration (RMSNorm -> Matmul -> SiLU ->
  # Add -> Mul). Dtype propagation + planner sizing + dst injection + dtype
  # trace + CUDA Graph capture/replay, all BF16. Skips gracefully w/o GPU.
  add_executable(forge-graph-bf16-test test_graph_bf16.cpp)
  forge_link_native(forge-graph-bf16-test)
  add_test(NAME graph-bf16 COMMAND forge-graph-bf16-test)

  # Phase 6-A-7.1: native FP16 attention decode (FP16 QKV storage, FP32 QK^T /
  # online-softmax / PV, FP16 out). Direct-launch parity vs CPU FP32 ref +
  # FP32-kernel self-consistency, dtype hard asserts, ordinary-vs-capture
  # stream bitwise equality, ComputeGraph node capture/replay, selector unit
  # test, loud failure on unsupported head_dim. Skips gracefully when no GPU.
  add_executable(forge-attn-fp16-test test_attention_fp16.cpp)
  forge_link_native(forge-attn-fp16-test)
  add_test(NAME attn-fp16 COMMAND forge-attn-fp16-test)

  # Phase 6-B-6: native BF16-Q decode (BF16 Q + FP32 KV, FP32 math, BF16 out).
  # Correctness + stream A/B + selector/loud surface + graph node + kv-sweep
  # bench. Skips gracefully w/o GPU.
  add_executable(forge-attn-bf16-test test_attention_bf16.cpp)
  forge_link_native(forge-attn-bf16-test)
  add_test(NAME attn-bf16 COMMAND forge-attn-bf16-test)

  # Phase 6-A-8: end-to-end native FP16 validation. Minimal 2-layer llama-style
  # closed loop (RMSNorm -> QKV -> FP16 attention -> o_proj -> residual ->
  # RMSNorm -> gate/up -> SiLU -> gate*up -> residual -> next layer -> final
  # norm -> logits), all-FP16 storage through the real ComputeGraph machinery.
  # Gates: (1) dtype trace -- no implicit FP16->FP32->FP16 on the main chain,
  # (2) FP16 vs FP32 logits parity (max_abs/max_rel/cosine/top-k, fixed
  # thresholds), (3) CUDA Graph ordinary -> capture -> replay x N bitwise
  # stable on the whole graph. Prefill and decode dispatch modes both covered.
  # 8.0 section chain-validates the new native FP16 ADD / MUL kernels.
  # Skips gracefully when no GPU present.
  add_executable(forge-e2e-fp16-test test_e2e_fp16.cpp)
  forge_link_native(forge-e2e-fp16-test)
  add_test(NAME e2e-fp16 COMMAND forge-e2e-fp16-test)

  # Phase 6-B-7: end-to-end BF16 validation. 2-layer closed loop (RMSNorm ->
  # QKV proj -> BF16 attention + BF16 KV -> residual -> gated FFN -> logits),
  # prefill + decode; decode consumes KV produced by the real prefill write
  # path. Skips gracefully w/o GPU.
  add_executable(forge-e2e-bf16-test test_e2e_bf16.cpp)
  forge_link_native(forge-e2e-bf16-test)
  add_test(NAME e2e-bf16 COMMAND forge-e2e-bf16-test)

  # Q8_0 quantize/dequantize round-trip + per-block scale validation.
  add_executable(forge-q8-roundtrip-test test_q8_roundtrip.cu)
  forge_link_native(forge-q8-roundtrip-test)
  add_test(NAME q8-roundtrip COMMAND forge-q8-roundtrip-test)

  # Phase 1-A: MoE expert GEMV (Q8_1 + dp4a) must match the main-path 2D
  # batch GEMV (shared vec_dot primitives) for Q4_0/Q8_0/Q4_K/Q6_K.
  add_executable(forge-moe-dp4a-ref-test test_moe_dp4a_reference.cpp)
  forge_link_native(forge-moe-dp4a-ref-test)
  add_test(NAME moe-dp4a-ref COMMAND forge-moe-dp4a-ref-test)

  # Phase 1-B + 1-C: dual matmul (M==1 slice write) and output_proj cooperative
  # reduction vs FP32 CPU reference; output_proj also checked for run-to-run
  # bit-exactness (no atomicAdd).
  add_executable(
    forge-phase1-bc-ref-test
    test_phase1_bc_reference.cpp
    ${PROJECT_SOURCE_DIR}/src/operators/cpu/common/dequant.cpp)
  forge_link_native(forge-phase1-bc-ref-test)
  add_test(NAME phase1-bc-ref COMMAND forge-phase1-bc-ref-test)

  # Phase 2-C: Vectorized FP32 prefill vs Scalar prefill parity (bit-exact
  # reformat) and both vs a CPU naive GQA reference.
  add_executable(forge-attn-backend-parity-test
                 test_attention_backend_parity.cpp)
  forge_link_native(forge-attn-backend-parity-test)
  add_test(NAME attn-backend-parity COMMAND forge-attn-backend-parity-test)

  # Phase 2-B: FP8 (E4M3/E5M2) fused prefill vs FP32 prefill over the same
  # dequantized KV (bit-exact, isolates kernel correctness from FP8 error).
  add_executable(forge-attn-fp8-prefill-test test_attention_fp8_prefill.cpp)
  forge_link_native(forge-attn-fp8-prefill-test)
  add_test(NAME attn-fp8-prefill COMMAND forge-attn-fp8-prefill-test)

  # Phase 6-C-4.1d: W8A8 FP8 E4M3 GEMM production backend (CUTLASS). Parity vs
  # FP32 reference + FP8-output ULP + KernelSelector wiring. Skips gracefully when
  # no Ada GPU is present (launchers return false pre-sm_89).
  add_executable(forge-fp8-gemm-test test_fp8_gemm_cuda.cu)
  forge_link_native(forge-fp8-gemm-test)
  add_test(NAME fp8-gemm COMMAND forge-fp8-gemm-test)

  # Phase 6-C-4.2: M==1 FP8 E4M3 GEMV — standalone benchmark + correctness audit.
  # Hand-written kernel (independent of the 6-C-4.1 CUTLASS GEMM path). Correctness
  # (golden = decode(A)xdecode(W)), scale fusion equivalence, FP8 vs FP16/FP32
  # latency/bandwidth at decode model shapes. No KernelSelector wiring yet. Skips
  # gracefully when no Ada GPU is present.
  add_executable(forge-fp8-gemv-bench test_fp8_gemv_cuda.cu)
  forge_link_native(forge-fp8-gemv-bench)
  add_test(NAME fp8-gemv-bench COMMAND forge-fp8-gemv-bench)

  # Q3_K / Q2_K decode (M==1) GEMV — bandwidth + correctness audit. Exercises the
  # production CUDA GEMV kernels (launch_gemv_q3_k_smem / launch_gemv_q2_k_q8_1)
  # against a CPU dequant golden. Used to measure the effect of bandwidth tuning
  # (e.g. the __ldg change in cuda_gemv_common.cuh). Skips without an Ada GPU.
  add_executable(forge-q3k-gemv-bench test_q3k_gemv_cuda.cu)
  forge_link_native(forge-q3k-gemv-bench)
  add_test(NAME q3k-gemv-bench COMMAND forge-q3k-gemv-bench)

  # Phase 5-D: Q3_K FP16 whole-matrix dequant (launch_dequant_q3_k_matrix_fp16)
  # + the Q3_K -> FP16 -> cuBLAS TensorCore prefill route it unlocks. Parity
  # against the existing FP32 dequant and a host FP32 GEMM golden. Skips
  # gracefully without a CUDA device or cuBLAS.
  add_executable(forge-q3k-dequant-fp16-test test_q3k_dequant_fp16.cu)
  forge_link_native(forge-q3k-dequant-fp16-test)
  add_test(NAME q3k-dequant-fp16 COMMAND forge-q3k-dequant-fp16-test)

  # Phase 6-C-5.6: minimal deterministic W8A8 FP8 E2E chain (producer -> FP8
  # GEMM/GEMV -> FP32) driven through the ops boundary. Gates: FP32-out parity vs
  # decode golden, bitwise determinism, scale source counters (registry == 0,
  # sidecar > 0), device-authoritative sidecars, fused-ReLU activation producer
  # (5.6a). Skips gracefully when no Ada GPU is present.
  add_executable(forge-fp8-w8a8-e2e-test test_fp8_w8a8_e2e.cu)
  forge_link_native(forge-fp8-w8a8-e2e-test)
  add_test(NAME fp8-w8a8-e2e COMMAND forge-fp8-w8a8-e2e-test)

  # Phase 2-B + 2-C: KV-tiled prefill across every KV format policy
  # (FP32/F16/Q8_0/FP8-E4M3/FP8-E5M2), contiguous and paged.
  # Phase 2-B + 2-C: KV-tiled prefill AND decode across every KV format policy
  # (FP32/F16/Q4_0/Q8_0/FP8-E4M3/FP8-E5M2), contiguous and paged. Decode reuses
  # the same tiled kernel (q_len == 1) and is checked against the legacy decode.
  add_executable(forge-tiled-attn-formats-test test_tiled_prefill_formats.cpp)
  forge_link_native(forge-tiled-attn-formats-test)
  add_test(NAME tiled-attn-formats COMMAND forge-tiled-attn-formats-test)

  # Phase 2-D: persistent per-(seq, layer) device page-table mirror
  # (incremental upload + multi-sequence isolation + dirty re-upload).
  add_executable(forge-kv-paged-mirror-test test_kv_paged_mirror.cpp)
  forge_link_native(forge-kv-paged-mirror-test)
  add_test(NAME kv-paged-mirror COMMAND forge-kv-paged-mirror-test)

  # Phase 2-B + 2-C: throughput micro-benchmark (tiled vs legacy decode/prefill).
  # Built like a test but NOT registered with CTest (long-running; run manually):
  #   ./build/tests/forge-bench-tiled-attn
  add_executable(forge-bench-tiled-attn bench_tiled_attn.cpp)
  forge_link_native(forge-bench-tiled-attn)

  # Phase 2 round 1 (6-A-7 Phase 2): FP16 baseline vs FP32 path, real kernel
  # timing, Decode/Prefill separate. NOT registered with CTest:
  #   ./build/tests/forge-bench-fp16-baseline
  add_executable(forge-bench-fp16-baseline bench_fp16_baseline.cpp)
  forge_link_native(forge-bench-fp16-baseline)
endif()

# -----------------------------------------------------------------------------
# Quantized dot-kernel comparisons (x86 AVX2 hosts only).
# -----------------------------------------------------------------------------
if(FORGE_CPU_HAS_AVX2)
  foreach(iq_test IN ITEMS iq2xs iq23s iq13xxs)
    add_executable(
      forge-${iq_test}-compare-test
      test_${iq_test}_compare.cpp
      ${PROJECT_SOURCE_DIR}/src/operators/cpu/common/quant_tables.cpp)
    target_link_libraries(forge-${iq_test}-compare-test
                          PRIVATE forge_cpu_features)
    target_include_directories(
      forge-${iq_test}-compare-test
      PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include
              ${PROJECT_SOURCE_DIR}/src/operators/cpu
              ${PROJECT_SOURCE_DIR}/src/operators/cpu/common)
    add_test(NAME ${iq_test}-compare COMMAND forge-${iq_test}-compare-test)
  endforeach()
else()
  message(STATUS "tests: skipping IQ2XS/IQ23S compare tests (no AVX2 host)")
endif()

# -----------------------------------------------------------------------------
# Cross-architecture kernel tests (ARM64 NEON, PPC64 VSX) are OPT-IN.
#
# They require a cross toolchain + QEMU and pull in arch-specific headers that
# do not build on every host, so they are gated behind FORGE_BUILD_CROSS_TESTS
# (default OFF). Enable explicitly with -DFORGE_BUILD_CROSS_TESTS=ON.
#
# The actual cross-build logic lives in cmake/cross_tests.cmake.
# -----------------------------------------------------------------------------
option(FORGE_BUILD_CROSS_TESTS
       "Build cross-architecture (ARM64/PPC64) kernel tests under QEMU" OFF)
if(FORGE_BUILD_CROSS_TESTS)
  include(${PROJECT_SOURCE_DIR}/cmake/cross_tests.cmake)
endif()

# -----------------------------------------------------------------------------
# Micro-benchmarks (gated by FORGE_BUILD_BENCHMARKS, default OFF; not
# registered with CTest -- run the binaries manually).
# -----------------------------------------------------------------------------
if(FORGE_BUILD_BENCHMARKS)
  # Several benchmarks control threads and do their own timing directly through
  # OpenMP (omp_set_num_threads / omp_get_wtime), so the OpenMP runtime must be
  # linked in -- it is not part of the ordinary test link interface.
  find_package(OpenMP)
  foreach(bench IN ITEMS ffn_q2k moe_multigemv moe_real output_proj_q6k
                         prefill_gemm cpu_decode)
    add_executable(forge-bench-${bench} bench_${bench}.cpp)
    # bench_cpu_decode.cpp includes arch headers by repo-relative path
    # ("src/operators/cpu/arch/x86/fused.h"); give it the source root.
    if(bench STREQUAL "cpu_decode")
      target_include_directories(forge-bench-cpu_decode
                                 PRIVATE ${PROJECT_SOURCE_DIR})
    endif()
    forge_link_native(forge-bench-${bench})
    if(OpenMP_CXX_FOUND)
      target_link_libraries(forge-bench-${bench} PRIVATE OpenMP::OpenMP_CXX)
    endif()
  endforeach()
endif()
