# CMake toolchain file for cross-compiling Forge to AArch64 (ARM64).
# Usage:
#   cmake -B build_arm64 \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain_aarch64.cmake \
#         -DCMAKE_BUILD_TYPE=Release
#
# Prerequisites:
#   Ubuntu/Debian: sudo apt-get install g++-aarch64-linux-gnu qemu-user-static
#   macOS:         Already native ARM64, no cross-compile needed.
#
# For QEMU verification after cross-compile:
#   cmake --build build_arm64
#   cd build_arm64 && ctest --verbose

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# ---- Compiler ----
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# ---- Sysroot (Ubuntu cross-compilation default) ----
# aarch64-linux-gnu-g++ already hardcodes /usr/aarch64-linux-gnu
# in its GCC specs (STARTFILE_SPEC / LIB_SPEC).
# Setting CMAKE_SYSROOT=/usr/aarch64-linux-gnu would double-add the
# prefix, causing linker "cannot find ... inside /usr/aarch64-linux-gnu"
# errors (effective path becomes /usr/aarch64-linux-gnu/usr/aarch64-linux-gnu/...).
# Set CMAKE_SYSROOT to "/" so the linker uses host-root as its sysroot.
# GCC specs still prepend their own /usr/aarch64-linux-gnu prefix, and
# the resulting paths (e.g. /usr/aarch64-linux-gnu/lib/libc.so.6) resolve
# correctly against the root sysroot.
# find_*() uses CMAKE_FIND_ROOT_PATH alone (see below).
set(CMAKE_SYSROOT "/")
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

# Search only in sysroot for libs/headers (not host)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ---- QEMU for test execution ----
# CMake cross-compile mode normally can't run test binaries.
# With qemu-user-static + binfmt_misc, executables run transparently.
# If ctest doesn't work, run manually: qemu-aarch64-static -L /usr/aarch64-linux-gnu ./test_binary
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64-static;-L;/usr/aarch64-linux-gnu)

# ---- CPU feature overrides for cross-compile ----
# Since cpuid probes can't run on the host, specify features explicitly.
# Apple M1/M2/M3/M4: DOTPROD;MATMUL_INT8
# Generic ARMv8.2+ (e.g. Cortex-A76/A78/X1): DOTPROD
# Generic ARMv8.0 (e.g. Cortex-A72): (none)
# Raspberry Pi 4 (Cortex-A72): (none)
# Raspberry Pi 5 (Cortex-A76): DOTPROD
set(FORGE_CPU_FEATURES
    "NEON;DOTPROD"
    CACHE STRING "ARM SIMD features: NEON, DOTPROD, MATMUL_INT8")

# ---- Build options ----
# FORGE_NATIVE_BUILD is meaningless in cross-compile; use FORGE_TUNE_MARCH.
set(FORGE_NATIVE_BUILD
    OFF
    CACHE BOOL "" FORCE)
set(FORGE_TUNE_MARCH
    "-march=armv8.2-a+dotprod"
    CACHE STRING "Tune for ARMv8.2-A + dotprod")

# Disable CUDA (no NVIDIA GPU on ARM64 SBCs typically)
set(USE_CUBLAS
    OFF
    CACHE BOOL "" FORCE)
