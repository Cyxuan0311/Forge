# Forge test suite.
#
# Layout:
#   - Native test/bench executables are regular CMake targets linked against
#     the forge libraries built by the parent project.
#   - Architecture-specific kernel tests (ARM64 NEON, PPC64 VSX) cannot run on
#     this host, so they are cross-compiled with <triple>-linux-gnu-g++ and run
#     under QEMU (both registered with CTest via CMAKE_CROSSCOMPILING_EMULATOR-
#     style wrappers). They are enabled only when the toolchain AND the qemu
#     binary are found.
#
# Run everything with CTest:
#   cmake -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
#
# Benchmarks are long-running; keep them out of the default ctest run by
# configuring with -DFORGE_BUILD_BENCHMARKS=OFF.

option(FORGE_BUILD_BENCHMARKS "Build the micro-benchmark executables" ON)

set(FORGE_TEST_LIBS
    forge_infer forge_model forge_ops forge_core forge_tokenizer)

# Common helper: link a native test like forge-cli (libs + optional CUDA/OpenMP).
# Libraries are wrapped in --start-group/--no-whole-group (GNU ld): the arch
# registration objects inside forge_model reference engine classes defined in
# forge_infer and vice versa, so plain left-to-right scanning can fail.
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
    ${target} PRIVATE ${group_open} ${FORGE_TEST_LIBS} ${group_close}
                      $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>
                      $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
  if(CUDAToolkit_FOUND)
    target_compile_definitions(${target} PRIVATE USE_CUDA)
    target_link_libraries(${target} PRIVATE CUDA::cudart
                                            $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>)
  endif()
endfunction()

# =========================================================================
# Native tests
# =========================================================================

# ---- Speculative decoding (Phase 1) --------------------------------------
#  1. verify_draft_tokens branch coverage (greedy + resample-consistency)
#  2. forward_batch multi-token logits vs sequential single-token forwards
#  3. End-to-end greedy parity: Generator output identical with spec on/off
# Tests 2 and 3 need models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf and are
# skipped with a notice when the file is missing.
add_executable(forge-spec-test test_spec_verify.cpp)
forge_link_native(forge-spec-test)
add_test(NAME spec-verify COMMAND forge-spec-test)

# ---- Parallel GGUF scanner (inspect) --------------------------------------
add_executable(forge-inspect-scanner-test test_inspect_scanner.cpp)
target_include_directories(
  forge-inspect-scanner-test PRIVATE ${PROJECT_SOURCE_DIR}/include
                                     ${PROJECT_SOURCE_DIR}/src
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

# ---- Generic scalar fallback kernels (standalone, no libs) ----------------
add_executable(forge-generic-kernels-test test_generic_scalar_kernels.cpp)
target_compile_definitions(forge-generic-kernels-test PRIVATE FORGE_ARCH_GENERIC)
target_include_directories(
  forge-generic-kernels-test PRIVATE ${PROJECT_SOURCE_DIR}
                                     ${PROJECT_SOURCE_DIR}/src/operators)
add_test(NAME generic-kernels COMMAND forge-generic-kernels-test)

# =========================================================================
# Quantized dot-kernel comparisons (x86 AVX2 hosts only)
# =========================================================================

if(FORGE_CPU_HAS_AVX2)
  foreach(iq_test IN ITEMS iq2xs iq23s)
    add_executable(forge-${iq_test}-compare-test test_${iq_test}_compare.cpp
                                                 ${PROJECT_SOURCE_DIR}/src/operators/cpu/common/quant_tables.cpp)
    target_link_libraries(forge-${iq_test}-compare-test PRIVATE forge_cpu_features)
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

# =========================================================================
# Cross-architecture kernel tests (QEMU)
# =========================================================================

# Helper: locate cross gcc + qemu emulator pair and register a ctest entry.
#   name        test label prefix
#   src         test source
#   triple_gcc  cross compiler binary name
#   qemu_bin    qemu binary names tried in order
#   cflags      arch-specific compile flags
function(forge_add_cross_test name src triple_gcc qemu_candidates cflags)
  find_program(${name}_GCC ${triple_gcc})
  set(qemu_found "")
  foreach(q IN LISTS qemu_candidates)
    find_program(${name}_QEMU ${q})
    if(${name}_QEMU)
      set(qemu_found ${${name}_QEMU})
      break()
    endif()
    unset(${name}_QEMU CACHE)
  endforeach()

  if(NOT ${name}_GCC OR NOT qemu_found)
    message(STATUS "tests: skipping ${name} (need ${triple_gcc} + qemu)")
    return()
  endif()

  set(out_dir ${PROJECT_BINARY_DIR}/tests/cross)
  file(MAKE_DIRECTORY ${out_dir})
  set(out_bin ${out_dir}/${name})

  add_custom_command(
    OUTPUT ${out_bin}
    COMMAND ${${name}_GCC} -std=c++17 -O2 -static ${cflags} -I${PROJECT_SOURCE_DIR}
            -I${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/${src}
            ${PROJECT_SOURCE_DIR}/tests/cross_host_mem_stub.cpp -o ${out_bin}
    DEPENDS ${PROJECT_SOURCE_DIR}/${src} ${PROJECT_SOURCE_DIR}/tests/cross_host_mem_stub.cpp
    COMMENT "Cross-building ${name} (${triple_gcc})"
    VERBATIM)
  # Build the cross binary as part of the default `cmake --build build`.
  add_custom_target(${name}-build ALL DEPENDS ${out_bin})

  # ctest wrapper script so `ctest` transparently runs under qemu.
  set(runner ${out_dir}/${name}.sh)
  file(WRITE ${runner}
       "#!/bin/sh\nexec ${qemu_found} ${out_bin}\n")
  execute_process(COMMAND chmod +x ${runner})

  add_test(NAME ${name} COMMAND ${runner})
  message(STATUS "tests: ${name} -> ${triple_gcc} + ${qemu_found}")
endfunction()

forge_add_cross_test(
  neon-arm64-kernels tests/test_arm64_neon_kernels.cpp
  aarch64-linux-gnu-g++ "qemu-aarch64-static;qemu-aarch64"
  "-DUSE_NEON;-march=armv8.2-a+dotprod")

forge_add_cross_test(
  vsx-ppc64-kernels tests/test_ppc64_vsx_kernels.cpp
  powerpc64le-linux-gnu-g++ "qemu-ppc64le-static;qemu-ppc64le"
  "-DUSE_VSX;-mcpu=power8")

# =========================================================================
# Micro-benchmarks (not registered with ctest; run manually)
# =========================================================================

if(FORGE_BUILD_BENCHMARKS)
  foreach(bench IN ITEMS ffn_q2k moe_multigemv moe_real output_proj_q6k prefill_gemm)
    add_executable(forge-bench-${bench} bench_${bench}.cpp)
    forge_link_native(forge-bench-${bench})
  endforeach()
endif()
