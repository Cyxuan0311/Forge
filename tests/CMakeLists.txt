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
                                  ${PROJECT_SOURCE_DIR}/src)

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
# Cross-architecture kernel tests (cross-compile + QEMU).
#
# These sources use NEON/VSX intrinsics and cannot be compiled by the host
# compiler, so each is cross-built with <triple>-linux-gnu-g++ and executed
# under QEMU via CTest. A test is registered only when BOTH the cross compiler
# and its qemu emulator are found; otherwise it is silently skipped.
#
# Included here:
#   * tests/test_arm64_neon_kernels.cpp, tests/test_ppc64_vsx_kernels.cpp
#     (top-level historical locations)
#   * every *.cpp under tests/arch/arm64/ and tests/arch/ppc64/ (auto-discovered)
# -----------------------------------------------------------------------------

# Helper: locate a cross gcc + qemu emulator pair and register a ctest entry.
#   name        test label prefix
#   src         test source (relative to PROJECT_SOURCE_DIR)
#   triple_gcc  cross compiler binary name
#   qemu_candidates  qemu binary names tried in order
#   cflags      arch-specific compile flags
#   inc_dirs    extra -I include directories (relative to PROJECT_SOURCE_DIR)
function(
  forge_add_cross_test
  name
  src
  triple_gcc
  qemu_candidates
  cflags
  inc_dirs)
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

  set(inc_args "")
  foreach(d IN LISTS inc_dirs)
    list(APPEND inc_args -I${PROJECT_SOURCE_DIR}/${d})
  endforeach()

  add_custom_command(
    OUTPUT ${out_bin}
    COMMAND
      ${${name}_GCC} -std=c++17 -O2 -static ${cflags} -I${PROJECT_SOURCE_DIR}
      -I${PROJECT_SOURCE_DIR}/include ${inc_args} ${PROJECT_SOURCE_DIR}/${src}
      ${PROJECT_SOURCE_DIR}/tests/cross_host_mem_stub.cpp -o ${out_bin}
    DEPENDS ${PROJECT_SOURCE_DIR}/${src}
            ${PROJECT_SOURCE_DIR}/tests/cross_host_mem_stub.cpp
    COMMENT "Cross-building ${name} (${triple_gcc})"
    VERBATIM)
  # Build the cross binary as part of the default `cmake --build build`.
  add_custom_target(${name}-build ALL DEPENDS ${out_bin})

  # ctest wrapper script so `ctest` transparently runs under qemu.
  set(runner ${out_dir}/${name}.sh)
  file(WRITE ${runner} "#!/bin/sh\nexec ${qemu_found} ${out_bin}\n")
  execute_process(COMMAND chmod +x ${runner})

  add_test(NAME ${name} COMMAND ${runner})
  message(STATUS "tests: ${name} -> ${triple_gcc} + ${qemu_found}")
endfunction()

# Top-level historical cross tests.
forge_add_cross_test(
  neon-arm64-kernels tests/test_arm64_neon_kernels.cpp aarch64-linux-gnu-g++
  "qemu-aarch64-static;qemu-aarch64" "-DUSE_NEON;-march=armv8.2-a+dotprod" "")

forge_add_cross_test(
  vsx-ppc64-kernels tests/test_ppc64_vsx_kernels.cpp powerpc64le-linux-gnu-g++
  "qemu-ppc64le-static;qemu-ppc64le" "-DUSE_VSX;-mcpu=power8" "")

# Auto-discover every *.cpp under tests/arch/<arch>/ and cross-build it with
# the matching toolchain + include path.
file(GLOB ARM64_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/arch/arm64/*.cpp")
foreach(src IN LISTS ARM64_SOURCES)
  get_filename_component(stem ${src} NAME_WE)
  forge_add_cross_test(
    arm64-${stem} tests/arch/arm64/${stem}.cpp aarch64-linux-gnu-g++
    "qemu-aarch64-static;qemu-aarch64" "-DUSE_NEON;-march=armv8.2-a+dotprod"
    "src/operators/cpu/arch/arm64")
endforeach()

file(GLOB PPC64_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/arch/ppc64/*.cpp")
foreach(src IN LISTS PPC64_SOURCES)
  get_filename_component(stem ${src} NAME_WE)
  forge_add_cross_test(
    ppc64-${stem} tests/arch/ppc64/${stem}.cpp powerpc64le-linux-gnu-g++
    "qemu-ppc64le-static;qemu-ppc64le" "-DUSE_VSX;-mcpu=power8"
    "src/operators/cpu/arch/ppc64")
endforeach()

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
