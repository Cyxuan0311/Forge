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

# Libraries shared by the native (host-compiled) test executables.
set(FORGE_TEST_LIBS forge_infer forge_model forge_ops forge_core
                    forge_tokenizer)

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
            $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>
            $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
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
  PRIVATE ${group_open} forge_inspect ${FORGE_TEST_LIBS} ${group_close}
          $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
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

  # Q8_0 quantize/dequantize round-trip + per-block scale validation.
  add_executable(forge-q8-roundtrip-test test_q8_roundtrip.cu)
  forge_link_native(forge-q8-roundtrip-test)
  add_test(NAME q8-roundtrip COMMAND forge-q8-roundtrip-test)
endif()

# -----------------------------------------------------------------------------
# Quantized dot-kernel comparisons (x86 AVX2 hosts only).
# -----------------------------------------------------------------------------
if(FORGE_CPU_HAS_AVX2)
  foreach(iq_test IN ITEMS iq2xs iq23s)
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
  foreach(bench IN ITEMS ffn_q2k moe_multigemv moe_real output_proj_q6k
                         prefill_gemm cpu_decode)
    add_executable(forge-bench-${bench} bench_${bench}.cpp)
    forge_link_native(forge-bench-${bench})
  endforeach()
endif()
