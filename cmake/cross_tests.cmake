# Cross-architecture kernel tests (cross-compile + QEMU).
#
# Included from tests/CMakeLists.txt ONLY when FORGE_BUILD_CROSS_TESTS=ON.
# Must NOT be processed by default: the arch-specific sources require a cross
# toolchain + QEMU and pull in headers that do not build on every developer
# machine (e.g. x86 CI hosts).
#
# Note: this file is pulled in via include(), so CMAKE_CURRENT_SOURCE_DIR still
# points at tests/ (the including listfile). The arch globs below use
# PROJECT_SOURCE_DIR explicitly to stay unambiguous regardless of where this
# .cmake physically lives.
#
# Sources handled here:
#   * tests/test_arm64_neon_kernels.cpp, tests/test_ppc64_vsx_kernels.cpp
#     (top-level historical locations)
#   * every *.cpp under tests/arch/arm64/ and tests/arch/ppc64/ (auto-discovered)

# Helper: locate a cross gcc + qemu emulator pair and register a ctest entry.
#   name            test label prefix
#   src             test source (relative to PROJECT_SOURCE_DIR)
#   triple_gcc      cross compiler binary name
#   qemu_candidates qemu binary names tried in order
#   cflags          arch-specific compile flags
#   inc_dirs        extra -I include directories (relative to PROJECT_SOURCE_DIR)
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
     "${PROJECT_SOURCE_DIR}/tests/arch/arm64/*.cpp")
foreach(src IN LISTS ARM64_SOURCES)
  get_filename_component(stem ${src} NAME_WE)
  forge_add_cross_test(
    arm64-${stem} tests/arch/arm64/${stem}.cpp aarch64-linux-gnu-g++
    "qemu-aarch64-static;qemu-aarch64" "-DUSE_NEON;-march=armv8.2-a+dotprod"
    "src/operators/cpu/arch/arm64;src/operators/cpu/common;tests/arch")
endforeach()

file(GLOB PPC64_SOURCES CONFIGURE_DEPENDS
     "${PROJECT_SOURCE_DIR}/tests/arch/ppc64/*.cpp")
foreach(src IN LISTS PPC64_SOURCES)
  get_filename_component(stem ${src} NAME_WE)
  forge_add_cross_test(
    ppc64-${stem} tests/arch/ppc64/${stem}.cpp powerpc64le-linux-gnu-g++
    "qemu-ppc64le-static;qemu-ppc64le" "-DUSE_VSX;-mcpu=power8"
    "src/operators/cpu/arch/ppc64;src/operators/cpu/common;tests/arch")
endforeach()
