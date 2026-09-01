# detect_cpu_features.cmake — Two-level CPU feature detection for Forge.
#
# Level 1: Architecture
#   FORGE_ARCH_X86   / FORGE_ARCH_ARM64   / FORGE_ARCH_PPC64   / FORGE_ARCH_GENERIC
#   Override with -DFORGE_ARCH=X86|ARM64|PPC64|GENERIC
#
# Level 2: SIMD features (native builds only; cross-compile uses FORGE_CPU_FEATURES)
#   FORGE_CPU_HAS_AVX2           — AVX2 + FMA + F16C
#   FORGE_CPU_HAS_AVX512_VNNI    — AVX-512F + BW + VL + VNNI + OSXSAVE
#   FORGE_CPU_HAS_NEON           — ARM NEON (always true on arm64)
#   FORGE_CPU_HAS_DOTPROD        — ARMv8.2-a dotprod
#   FORGE_CPU_HAS_MATMUL_INT8    — ARMv8.2-a i8mm
#   FORGE_CPU_HAS_VSX            — PowerPC VSX (POWER7+, always on ppc64/ppc64le)
#
# Output variables:
#   FORGE_ARCH                — X86 | ARM64 | PPC64 | GENERIC
#   FORGE_CPU_HAS_AVX2        — TRUE | FALSE
#   FORGE_CPU_HAS_AVX512_VNNI — TRUE | FALSE
#   FORGE_CPU_HAS_NEON        — TRUE | FALSE
#   FORGE_CPU_HAS_DOTPROD     — TRUE | FALSE
#   FORGE_CPU_HAS_MATMUL_INT8 — TRUE | FALSE
#   FORGE_CPU_HAS_VSX         — TRUE | FALSE

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# MSVC has no -fsyntax-only; use /Zs for syntax-only checks.
if(MSVC)
  set(_forge_syntax_only "/Zs")
else()
  set(_forge_syntax_only "-fsyntax-only")
endif()

# ---- Level 1: Architecture detection ----
if(DEFINED FORGE_ARCH)
  string(TOUPPER "${FORGE_ARCH}" FORGE_ARCH)
  if(NOT FORGE_ARCH MATCHES "^(X86|ARM64|PPC64|GENERIC)$")
    message(
      FATAL_ERROR
        "FORGE_ARCH must be X86, ARM64, PPC64, or GENERIC (got '${FORGE_ARCH}')"
    )
  endif()
  message(STATUS "CPU arch: ${FORGE_ARCH} (user override)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)")
  set(FORGE_ARCH "X86")
  message(
    STATUS
      "CPU arch: X86 (detected from CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})"
  )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|armv[78])")
  set(FORGE_ARCH "ARM64")
  message(
    STATUS
      "CPU arch: ARM64 (detected from CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})"
  )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64|ppc64le|powerpc64|powerpc64le)")
  set(FORGE_ARCH "PPC64")
  message(
    STATUS
      "CPU arch: PPC64 (detected from CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})"
  )
else()
  set(FORGE_ARCH "GENERIC")
  message(
    STATUS
      "CPU arch: GENERIC (unknown CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})"
  )
endif()

# ---- Level 2: SIMD feature detection ----
set(FORGE_CPU_HAS_AVX2 FALSE)
set(FORGE_CPU_HAS_AVX512_VNNI FALSE)
set(FORGE_CPU_HAS_NEON FALSE)
set(FORGE_CPU_HAS_DOTPROD FALSE)
set(FORGE_CPU_HAS_MATMUL_INT8 FALSE)
set(FORGE_CPU_HAS_VSX FALSE)

if(FORGE_ARCH STREQUAL "X86")
  # Cross-compile: user specifies features via FORGE_CPU_FEATURES list
  if(DEFINED FORGE_CPU_FEATURES)
    message(
      STATUS "CPU features: ${FORGE_CPU_FEATURES} (cross-compile override)")
    if("AVX2" IN_LIST FORGE_CPU_FEATURES)
      set(FORGE_CPU_HAS_AVX2 TRUE)
    endif()
    if("AVX512_VNNI" IN_LIST FORGE_CPU_FEATURES)
      set(FORGE_CPU_HAS_AVX512_VNNI TRUE)
    endif()
  elseif(CMAKE_CROSSCOMPILING)
    message(
      STATUS
        "Cross-compiling for x86 — SIMD detection skipped. Set FORGE_CPU_FEATURES to enable."
    )
  else()
    # Native build: probe CPU features via compile-time predefined macros. Uses
    # check_cxx_source_compiles (no try_run) — WSL/9p safe. The compiler's
    # -march=native (or FORGE_TUNE_MARCH) determines which SIMD macros are
    # defined; we simply check for them without executing.

    # Determine effective arch probe flags (mirrors root CMakeLists.txt logic)
    set(_forge_probe_flags "")
    if(FORGE_TUNE_MARCH)
      set(_forge_probe_flags "${FORGE_TUNE_MARCH}")
    elseif(FORGE_NATIVE_BUILD)
      if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        set(_forge_probe_flags "-mcpu=apple-m1")
      elseif(MSVC)
        set(_forge_probe_flags "/arch:AVX2")
      else()
        set(_forge_probe_flags "-march=native")
      endif()
    endif()

    if(_forge_probe_flags STREQUAL "")
      message(
        STATUS
          "No arch probe flags — SIMD detection skipped. Set FORGE_CPU_FEATURES to enable."
      )
    else()
      message(
        STATUS
          "Probing CPU features via compile-time macros (flags: ${_forge_probe_flags})..."
      )

      # Directly compile a probe translation unit with the probe flags using
      # execute_process — fully avoids check_cxx_source_compiles / try_run
      # quirks (WSL/9p safe: only -c, no linking or execution). The length of
      # the probe source determines success: 0-byte object => feature available.
      set(_probe_dir "${CMAKE_BINARY_DIR}/CMakeFiles/forge_cpu_probe")

      # AVX2: check __AVX2__ && __FMA__ && __F16C__ (GCC/Clang); MSVC defines
      # _M_AVX2 with /arch:AVX2 and has no __FMA__/__F16C__ macros (both are
      # implied by /arch:AVX2), so the probe must differ per compiler.
      file(MAKE_DIRECTORY "${_probe_dir}")
      if(MSVC)
        set(_probe_avx2_src
            "#if !defined(_M_AVX2)\n#error \"AVX2 not available\"\n#endif\nint probe_avx2(){ return 0; }\n")
      else()
        set(_probe_avx2_src
            "#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)\n"
            "#error \"AVX2 not available\"\n#endif\nint probe_avx2(){ return 0; }\n")
      endif()
      file(WRITE "${_probe_dir}/probe_avx2.cpp" "${_probe_avx2_src}")
      execute_process(
        COMMAND
          "${CMAKE_CXX_COMPILER}" ${CMAKE_CXX_COMPILER_ARG1} ${CMAKE_CXX_FLAGS}
          "${_forge_probe_flags}" ${_forge_syntax_only} "${_probe_dir}/probe_avx2.cpp"
        RESULT_VARIABLE _avx2_probe_reply
        ERROR_VARIABLE _avx2_probe_err)
      if(_avx2_probe_reply EQUAL 0)
        set(FORGE_CPU_HAS_AVX2 TRUE)
      else()
        set(FORGE_CPU_HAS_AVX2 FALSE)
        message(STATUS "    AVX2 probe compiler output: ${_avx2_probe_err}")
      endif()

      if(FORGE_CPU_HAS_AVX2)
        message(STATUS "  AVX2: YES (via ${_forge_probe_flags})")
        if(MSVC)
          check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
        else()
          check_cxx_compiler_flag("-mavx2 -mfma -mf16c" COMPILER_SUPPORTS_AVX2)
        endif()
        if(NOT COMPILER_SUPPORTS_AVX2)
          message(
            WARNING
              "  AVX2 macros detected but compiler does not support -mavx2 -mfma -mf16c"
          )
          set(FORGE_CPU_HAS_AVX2 FALSE)
        endif()
      else()
        message(
          WARNING
            "  AVX2: NO — compile-time probe failed with ${_forge_probe_flags}")
        message(WARNING "    Override with: -DFORGE_CPU_FEATURES=AVX2")
      endif()

      # AVX-512 VNNI: check __AVX512F__ && __AVX512BW__ && __AVX512VL__ &&
      # __AVX512VNNI__
      file(
        WRITE "${_probe_dir}/probe_avx512.cpp"
        "#if !defined(__AVX512F__) || !defined(__AVX512BW__) || !defined(__AVX512VL__) || !defined(__AVX512VNNI__)\n"
        "#error \"AVX-512 VNNI not available\"\n"
        "#endif\n"
        "int probe_avx512(){ return 0; }\n")
      execute_process(
        COMMAND
          "${CMAKE_CXX_COMPILER}" ${CMAKE_CXX_COMPILER_FLAG1} ${CMAKE_CXX_FLAGS}
          "${_forge_probe_flags}" ${_forge_syntax_only} "${_probe_dir}/probe_avx512.cpp"
        RESULT_VARIABLE _avx512_probe
        ERROR_VARIABLE _avx512_probe_err)
      if(_avx512_probe EQUAL 0)
        set(FORGE_CPU_HAS_AVX512_VNNI TRUE)
      else()
        set(FORGE_CPU_HAS_AVX512_VNNI FALSE)
        message(
          STATUS "    AVX-512 probe compiler output: ${_avx512_probe_err}")
      endif()

      if(FORGE_CPU_HAS_AVX512_VNNI)
        message(STATUS "  AVX-512 VNNI: YES (via ${_forge_probe_flags})")
        if(MSVC)
          check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512_VNNI)
        else()
          check_cxx_compiler_flag(
            "-mavx512f -mavx512bw -mavx512vnni -mavx512vl -mf16c"
            COMPILER_SUPPORTS_AVX512_VNNI)
        endif()
        if(NOT COMPILER_SUPPORTS_AVX512_VNNI)
          message(
            WARNING
              "  AVX-512 VNNI macros detected but compiler does not support flags"
          )
          set(FORGE_CPU_HAS_AVX512_VNNI FALSE)
        endif()
      else()
        message(
          WARNING
            "  AVX-512 VNNI: NO — compile-time probe failed with ${_forge_probe_flags}"
        )
        message(
          WARNING "    Override with: -DFORGE_CPU_FEATURES=AVX2;AVX512_VNNI")
      endif()

      unset(_avx2_probe)
      unset(_avx2_probe_err)
      unset(_avx512_probe)
      unset(_avx512_probe_err)
    endif()
  endif()

elseif(FORGE_ARCH STREQUAL "ARM64")
  # ARM NEON is always available on AArch64 (v8+ baseline)
  set(FORGE_CPU_HAS_NEON TRUE)

  if(DEFINED FORGE_CPU_FEATURES)
    message(
      STATUS "CPU features: ${FORGE_CPU_FEATURES} (cross-compile override)")
    if("DOTPROD" IN_LIST FORGE_CPU_FEATURES)
      set(FORGE_CPU_HAS_DOTPROD TRUE)
    endif()
    if("MATMUL_INT8" IN_LIST FORGE_CPU_FEATURES)
      set(FORGE_CPU_HAS_MATMUL_INT8 TRUE)
    endif()
  elseif(CMAKE_CROSSCOMPILING)
    message(
      STATUS
        "Cross-compiling for ARM64 — feature detection skipped. Set FORGE_CPU_FEATURES."
    )
  else()
    # Native ARM64: probe via compile-time predefined macros. Mirrors root
    # CMakeLists.txt: macOS Apple Silicon uses -mcpu=apple-m1, generic Linux
    # ARM64 uses -march=native / FORGE_TUNE_MARCH. Uses execute_process
    # -fsyntax-only (WSL/9p safe, no try_run).
    set(_forge_arm_probe_flags "")
    if(FORGE_TUNE_MARCH)
      set(_forge_arm_probe_flags "${FORGE_TUNE_MARCH}")
    elseif(FORGE_NATIVE_BUILD)
      if(APPLE)
        # Apple M1 is the min base for all M-series (M1/M2/M3/M4). All support
        # NEON + dotprod + i8mm.
        set(_forge_arm_probe_flags "-mcpu=apple-m1")
      else()
        set(_forge_arm_probe_flags "-march=native")
      endif()
    endif()

    if(_forge_arm_probe_flags STREQUAL "")
      message(
        STATUS
          "No arch probe flags — ARM feature detection skipped. Set FORGE_CPU_FEATURES."
      )
    else()
      message(
        STATUS
          "Probing ARM64 features via compile-time macros (flags: ${_forge_arm_probe_flags})..."
      )
      set(_probe_dir "${CMAKE_BINARY_DIR}/CMakeFiles/forge_cpu_probe")

      # NEON: check __ARM_NEON (should always be true on AArch64)
      file(MAKE_DIRECTORY "${_probe_dir}")
      file(WRITE "${_probe_dir}/probe_neon.cpp"
           "#if !defined(__ARM_NEON)\n" "#error \"NEON not available\"\n"
           "#endif\n" "int probe_neon(){ return 0; }\n")
      execute_process(
        COMMAND
          "${CMAKE_CXX_COMPILER}" ${CMAKE_CXX_COMPILER_ARG1} ${CMAKE_CXX_FLAGS}
          "${_forge_arm_probe_flags}" ${_forge_syntax_only}
          "${_probe_dir}/probe_neon.cpp"
        RESULT_VARIABLE _neon_result
        ERROR_VARIABLE _neon_err)
      if(_neon_result EQUAL 0)
        message(STATUS "  NEON: YES (via ${_forge_arm_probe_flags})")
      else()
        message(
          WARNING
            "  NEON: NO — unexpected on AArch64. Override with FORGE_CPU_FEATURES."
        )
        set(FORGE_CPU_HAS_NEON FALSE)
      endif()

      # DOTPROD: check __ARM_FEATURE_DOTPROD
      file(
        WRITE "${_probe_dir}/probe_dotprod.cpp"
        "#if !defined(__ARM_FEATURE_DOTPROD)\n"
        "#error \"dotprod not available\"\n" "#endif\n"
        "int probe_dotprod(){ return 0; }\n")
      execute_process(
        COMMAND
          "${CMAKE_CXX_COMPILER}" ${CMAKE_CXX_COMPILER_FLAG1} ${CMAKE_CXX_FLAGS}
          "${_forge_arm_probe_flags}" ${_forge_syntax_only}
          "${_probe_dir}/probe_dotprod.cpp"
        RESULT_VARIABLE _dotprod_result
        ERROR_VARIABLE _dotprod_err)
      if(_dotprod_result EQUAL 0)
        set(FORGE_CPU_HAS_DOTPROD TRUE)
        message(STATUS "  DOTPROD: YES (via ${_forge_arm_probe_flags})")
      else()
        message(
          WARNING
            "  DOTPROD: NO — compile-time probe failed with ${_forge_arm_probe_flags}"
        )
        message(WARNING "    Override with: -DFORGE_CPU_FEATURES=DOTPROD")
      endif()

      # MATMUL_INT8: check __ARM_FEATURE_MATMUL_INT8 (ARMv8.6+i8mm, Apple M1+)
      file(
        WRITE "${_probe_dir}/probe_i8mm.cpp"
        "#if !defined(__ARM_FEATURE_MATMUL_INT8)\n"
        "#error \"i8mm not available\"\n" "#endif\n"
        "int probe_i8mm(){ return 0; }\n")
      execute_process(
        COMMAND
          "${CMAKE_CXX_COMPILER}" ${CMAKE_CXX_COMPILER_FLAG1} ${CMAKE_CXX_FLAGS}
          "${_forge_arm_probe_flags}" ${_forge_syntax_only}
          "${_probe_dir}/probe_i8mm.cpp"
        RESULT_VARIABLE _i8mm_result
        ERROR_VARIABLE _i8mm_err)
      if(_i8mm_result EQUAL 0)
        set(FORGE_CPU_HAS_MATMUL_INT8 TRUE)
        message(STATUS "  MATMUL_INT8: YES (via ${_forge_arm_probe_flags})")
      else()
        message(
          WARNING
            "  MATMUL_INT8: NO — compile-time probe failed with ${_forge_arm_probe_flags}"
        )
        message(
          WARNING "    Override with: -DFORGE_CPU_FEATURES=DOTPROD;MATMUL_INT8")
      endif()

      unset(_neon_result)
      unset(_dotprod_result)
      unset(_i8mm_result)
    endif()
  endif()

elseif(FORGE_ARCH STREQUAL "PPC64")
  # VSX is available on POWER7+ (POWER8 is the baseline for ppc64le). VSX
  # provides 128-bit SIMD vector float/int operations via <altivec.h>.
  set(FORGE_CPU_HAS_VSX TRUE)

  if(DEFINED FORGE_CPU_FEATURES)
    message(
      STATUS "CPU features: ${FORGE_CPU_FEATURES} (cross-compile override)")
  elseif(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling for PPC64 — VSX enabled by default.")
  else()
    # Native PPC64: probe via compile-time predefined macros.
    set(_probe_dir "${CMAKE_BINARY_DIR}/CMakeFiles/forge_cpu_probe")

    # VSX: check __VSX__ (should be defined on POWER7+)
    file(MAKE_DIRECTORY "${_probe_dir}")
    file(WRITE "${_probe_dir}/probe_vsx.cpp"
         "#if !defined(__VSX__)\n" "#error \"VSX not available\"\n" "#endif\n"
         "int probe_vsx(){ return 0; }\n")
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" ${CMAKE_CXX_COMPILER_ARG1}
              ${CMAKE_CXX_FLAGS} ${_forge_syntax_only} "${_probe_dir}/probe_vsx.cpp"
      RESULT_VARIABLE _vsx_result
      ERROR_VARIABLE _vsx_err)
    if(_vsx_result EQUAL 0)
      message(STATUS "  VSX: YES")
    else()
      message(
        WARNING
          "  VSX: NO — unexpected on PPC64. Override with FORGE_CPU_FEATURES.")
      set(FORGE_CPU_HAS_VSX FALSE)
    endif()
    unset(_vsx_result)
    unset(_vsx_err)
  endif()

else()
  message(STATUS "CPU features: GENERIC (no SIMD)")
endif()
