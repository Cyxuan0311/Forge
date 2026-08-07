#!/bin/bash
# ARM64 QEMU cross-verification script for Forge NEON kernels.
#
# Verifies ARM64 NEON kernel correctness on an x86 host via QEMU user-mode
# emulation. Works in two modes:
#
#   MODE 1 (fast): Cross-compile & run standalone NEON kernel unit tests.
#   MODE 2 (full): Cross-compile the full Forge project & run verify_inference.
#
# Prerequisites (Ubuntu/Debian):
#   sudo apt-get install g++-aarch64-linux-gnu qemu-user-static
#
# Usage:
#   ./scripts/verify_arm64_qemu.sh          # Mode 1 (unit tests)
#   ./scripts/verify_arm64_qemu.sh --full   # Mode 2 (full project)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

require_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo -e "${RED}ERROR: $1 not found. Install with: sudo apt-get install g++-aarch64-linux-gnu qemu-user-static${NC}"
        exit 1
    fi
}

echo "============================================"
echo " Forge ARM64 NEON QEMU Verification"
echo "============================================"

# ---- Check prerequisites ----
require_tool aarch64-linux-gnu-g++
require_tool qemu-aarch64-static

CXX=aarch64-linux-gnu-g++
CXXFLAGS=(-std=c++17 -march=armv8.2-a+dotprod -O2 -DUSE_NEON)

# Verify the cross-compiler actually targets aarch64
ARCH_CHECK=$("$CXX" -dumpmachine 2>/dev/null || echo "unknown")
if [[ "$ARCH_CHECK" != aarch64* ]]; then
    echo -e "${RED}ERROR: $CXX does not target aarch64 (got: $ARCH_CHECK)${NC}"
    exit 1
fi
echo -e "${GREEN}Cross-compiler: $CXX ($ARCH_CHECK)${NC}"

# ---- Mode 1: Standalone NEON kernel unit tests ----
echo ""
echo "--- Building standalone NEON kernel tests ---"

UNIT_TEST_SRC="$PROJECT_DIR/tests/test_arm64_neon_kernels.cpp"
UNIT_TEST_BIN="$PROJECT_DIR/build_arm64_qemu/test_neon_kernels"

mkdir -p "$(dirname "$UNIT_TEST_BIN")"

# Static linking avoids QEMU dynamic loader issues
echo "Compiling..."
"$CXX" "${CXXFLAGS[@]}" -static \
    -I"$PROJECT_DIR" \
    -I"$PROJECT_DIR/include" \
    -I"$PROJECT_DIR/src/operators/cpu" \
    -I"$PROJECT_DIR/src/operators/cpu/arch/arm64" \
    "$UNIT_TEST_SRC" \
    -o "$UNIT_TEST_BIN" \
    -lm

echo "Running via QEMU..."
if qemu-aarch64-static "$UNIT_TEST_BIN"; then
    echo ""
    echo -e "${GREEN}All NEON kernel unit tests PASSED${NC}"
else
    echo ""
    echo -e "${RED}NEON kernel unit tests FAILED${NC}"
    exit 1
fi

# ---- Mode 2: Full project build (optional) ----
if [[ "${1:-}" == "--full" ]]; then
    echo ""
    echo "--- Building full Forge project for ARM64 ---"

    BUILD_DIR="$PROJECT_DIR/build_arm64_qemu/full"
    rm -rf "$BUILD_DIR"

    cmake -B "$BUILD_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$PROJECT_DIR/cmake/toolchain_aarch64.cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        -DFORGE_CPU_FEATURES="NEON;DOTPROD" \
        -DFORGE_TUNE_MARCH="-march=armv8.2-a+dotprod" \
        -DFORGE_USE_CUDA=OFF \
        -DFORGE_BUILD_PYTHON=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        "$PROJECT_DIR"

    cmake --build "$BUILD_DIR" -j"$(nproc)"

    echo ""
    echo -e "${GREEN}Full ARM64 project build PASSED${NC}"
    echo ""
    echo "To verify inference, run on native ARM64 hardware or use:"
    echo "  cd $BUILD_DIR && ctest --verbose"
    echo ""
    echo "Note: ctest uses QEMU via CMAKE_CROSSCOMPILING_EMULATOR."
fi

echo ""
echo "============================================"
echo -e "${GREEN} ARM64 QEMU Verification Complete${NC}"
echo "============================================"
