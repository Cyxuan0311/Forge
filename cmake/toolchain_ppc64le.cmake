# CMake toolchain file for cross-compiling Forge to PPC64LE (PowerPC little-endian).
# Usage:
#   cmake -B build_ppc64 \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain_ppc64le.cmake \
#         -DCMAKE_BUILD_TYPE=Release
#
# Prerequisites:
#   Ubuntu/Debian: sudo apt-get install g++-powerpc64le-linux-gnu qemu-user-static
#
# For QEMU verification after cross-compile:
#   cmake --build build_ppc64
#   cd build_ppc64 && ctest --verbose

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)

# ---- Compiler ----
set(CMAKE_C_COMPILER powerpc64le-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER powerpc64le-linux-gnu-g++)

# ---- Sysroot (Ubuntu cross-compilation default) ----
# powerpc64le-linux-gnu-g++ hardcodes /usr/powerpc64le-linux-gnu in its
# GCC specs. Set CMAKE_SYSROOT to "/" to avoid double-prefixing, same
# as the aarch64 toolchain.
set(CMAKE_SYSROOT "/")
set(CMAKE_FIND_ROOT_PATH /usr/powerpc64le-linux-gnu)

# Search only in sysroot for libs/headers (not host)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ---- QEMU for test execution ----
set(CMAKE_CROSSCOMPILING_EMULATOR
    qemu-ppc64le-static;-L;/usr/powerpc64le-linux-gnu)

# ---- CPU feature overrides for cross-compile ----
# VSX is available on POWER7+; POWER8 is the baseline for ppc64le.
set(FORGE_CPU_FEATURES
    "VSX"
    CACHE STRING "PPC SIMD features: VSX")

# ---- Build options ----
set(FORGE_NATIVE_BUILD
    OFF
    CACHE BOOL "" FORCE)
set(FORGE_TUNE_MARCH
    "-mcpu=power8"
    CACHE STRING "Tune for POWER8 (ppc64le baseline)")

# Disable CUDA (no NVIDIA GPU on Power typically, unless POWER9+NVLink)
set(USE_CUBLAS
    OFF
    CACHE BOOL "" FORCE)

# Suppress GCC ABI-change notes (layout of aggregates containing vectors...)
# These are harmless for cross-compilation and only warn about GCC 5 ABI.
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wno-psabi"
    CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -Wno-psabi"
    CACHE STRING "" FORCE)
