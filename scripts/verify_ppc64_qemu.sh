#!/bin/bash
# PPC64 QEMU cross-verification script for Forge VSX kernels.
#
# Verifies PowerPC64 VSX kernel correctness on an x86 host via QEMU
# user-mode emulation. Works in two modes:
#
#   MODE 1 (fast): Cross-compile & run standalone VSX kernel unit tests.
#   MODE 2 (full): Cross-compile the full Forge project & build forge-cli.
#
# Prerequisites (Ubuntu/Debian):
#   sudo apt-get install g++-powerpc64le-linux-gnu qemu-user-static
#
# Usage:
#   ./scripts/verify_ppc64_qemu.sh          # Mode 1 (unit tests)
#   ./scripts/verify_ppc64_qemu.sh --full   # Mode 2 (full project)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

require_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo -e "${RED}ERROR: $1 not found. Install with: sudo apt-get install g++-powerpc64le-linux-gnu qemu-user-static${NC}"
        exit 1
    fi
}

echo "============================================"
echo " Forge PPC64 VSX QEMU Verification"
echo "============================================"

# ---- Check prerequisites ----
require_tool powerpc64le-linux-gnu-g++
require_tool qemu-ppc64le-static

CXX=powerpc64le-linux-gnu-g++
CXXFLAGS=(-std=c++17 -mcpu=power8 -O2 -DUSE_VSX)

# Verify the cross-compiler actually targets powerpc64le
ARCH_CHECK=$("$CXX" -dumpmachine 2>/dev/null || echo "unknown")
if [[ "$ARCH_CHECK" != powerpc64le* ]]; then
    echo -e "${RED}ERROR: $CXX does not target powerpc64le (got: $ARCH_CHECK)${NC}"
    exit 1
fi
echo -e "${GREEN}Cross-compiler: $CXX ($ARCH_CHECK)${NC}"

# ---- Mode 1: Standalone VSX kernel unit tests ----
echo ""
echo "--- Building standalone VSX kernel tests ---"

UNIT_TEST_SRC="$PROJECT_DIR/tests/test_ppc64_vsx_kernels.cpp"
UNIT_TEST_BIN="$PROJECT_DIR/build_ppc64_qemu/test_vsx_kernels"

mkdir -p "$(dirname "$UNIT_TEST_BIN")"

# Static linking avoids QEMU dynamic loader issues
echo "Compiling..."
"$CXX" "${CXXFLAGS[@]}" -static \
    -I"$PROJECT_DIR" \
    -I"$PROJECT_DIR/include" \
    -I"$PROJECT_DIR/src/operators/cpu" \
    -I"$PROJECT_DIR/src/operators/cpu/arch/ppc64" \
    "$UNIT_TEST_SRC" \
    -o "$UNIT_TEST_BIN" \
    -lm

echo "Running via QEMU..."
if qemu-ppc64le-static "$UNIT_TEST_BIN"; then
    echo ""
    echo -e "${GREEN}All VSX kernel unit tests PASSED${NC}"
else
    echo ""
    echo -e "${RED}VSX kernel unit tests FAILED${NC}"
    exit 1
fi

# ---- Mode 2: Full project build (optional) ----
if [[ "${1:-}" == "--full" ]]; then
    echo ""
    echo "--- Building full Forge project for PPC64 ---"

    BUILD_DIR="$PROJECT_DIR/build_ppc64_qemu/full"
    rm -rf "$BUILD_DIR"

    cmake -B "$BUILD_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$PROJECT_DIR/cmake/toolchain_ppc64le.cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        -DFORGE_CPU_FEATURES="VSX" \
        -DFORGE_TUNE_MARCH="-mcpu=power8" \
        -DFORGE_USE_CUDA=OFF \
        -DFORGE_BUILD_PYTHON=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        "$PROJECT_DIR"

    cmake --build "$BUILD_DIR" -j"$(nproc)"

    echo ""
    echo -e "${GREEN}Full PPC64 project build PASSED${NC}"
    echo ""
    echo "To verify inference, run on native PPC64 hardware or use:"
    echo "  cd $BUILD_DIR && ctest --verbose"
    echo ""
    echo "Note: ctest uses QEMU via CMAKE_CROSSCOMPILING_EMULATOR."
fi

echo ""
echo "============================================"
echo -e "${GREEN} PPC64 QEMU Verification Complete${NC}"
echo "============================================"
