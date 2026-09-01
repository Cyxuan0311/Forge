#!/bin/bash
# Forge Interactive Build Script
# Usage: ./build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}       Forge Build Configuration${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo
}

print_option() {
    echo -e "  ${GREEN}$1${NC} - $2"
}

ask_option() {
    local prompt="$1"
    local default="$2"
    local result
    read -rp "$(echo -e "${YELLOW}$prompt [${default}]: ${NC}")" result
    echo "${result:-$default}"
}

detect_cuda() {
    if command -v nvcc &>/dev/null; then
        echo -e "${GREEN}CUDA found: $(nvcc --version | grep -oP 'release \K[0-9.]+')${NC}"
        return 0
    elif [ -d "/usr/local/cuda" ] || [ -d "/usr/local/cuda-12.4" ]; then
        echo -e "${GREEN}CUDA found (not in PATH)${NC}"
        return 0
    fi
    echo -e "${RED}CUDA not found${NC}"
    return 1
}

detect_gpu_arch() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "86;89"
        return
    fi
    local arch
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$arch" ]; then
        echo "$arch"
    else
        echo "86;89"
    fi
}

print_header

# --- Build Type ---
echo -e "${CYAN}Build Type:${NC}"
echo -e "  [1] Release   (optimized, -O3)"
echo -e "  [2] Debug     (debug symbols, -g)"
echo -e "  [3] RelWithDebInfo (optimized + debug info)"
BUILD_TYPE=$(ask_option "Select build type" "1")
case $BUILD_TYPE in
    1) BUILD_TYPE="Release" ;;
    2) BUILD_TYPE="Debug" ;;
    3) BUILD_TYPE="RelWithDebInfo" ;;
    *) BUILD_TYPE="Release" ;;
esac
echo -e "  → Build type: ${GREEN}$BUILD_TYPE${NC}"
echo

# --- CUDA ---
echo -e "${CYAN}CUDA Support:${NC}"
if detect_cuda; then
    USE_CUDA=$(ask_option "Enable CUDA? (y/n)" "y")
    case $USE_CUDA in
        y|Y|yes) USE_CUDA="ON" ;;
        *) USE_CUDA="OFF" ;;
    esac
else
    USE_CUDA="OFF"
    echo -e "  → CUDA: ${RED}disabled (not found)${NC}"
fi
echo

# --- cuBLAS ---
if [ "$USE_CUDA" = "ON" ]; then
    echo -e "${CYAN}cuBLAS:${NC}"
    USE_CUBLAS=$(ask_option "Use cuBLAS for GPU GEMM? (y/n)" "y")
    case $USE_CUBLAS in
        y|Y|yes) USE_CUBLAS="ON" ;;
        *) USE_CUBLAS="OFF" ;;
    esac
    echo -e "  → cuBLAS: ${GREEN}$USE_CUBLAS${NC}"
    echo

    # --- CUDA Architecture ---
    DEFAULT_ARCH=$(detect_gpu_arch)
    echo -e "${CYAN}CUDA Architecture:${NC}"
    echo -e "  Detected GPU arch: ${GREEN}$DEFAULT_ARCH${NC}"
    CUDA_ARCH=$(ask_option "CUDA architectures (e.g. 75;86;89)" "$DEFAULT_ARCH")
    echo -e "  → CUDA arch: ${GREEN}$CUDA_ARCH${NC}"
    echo
fi

# --- OpenBLAS ---
echo -e "${CYAN}OpenBLAS:${NC}"
USE_OPENBLAS=$(ask_option "Use OpenBLAS for CPU GEMM? (y/n)" "n")
case $USE_OPENBLAS in
    y|Y|yes) USE_OPENBLAS="ON" ;;
    *) USE_OPENBLAS="OFF" ;;
esac
echo -e "  → OpenBLAS: ${GREEN}$USE_OPENBLAS${NC}"
echo

# --- PerfProfiler ---
echo -e "${CYAN}PerfProfiler:${NC}"
FORGE_PROFILING=$(ask_option "Enable PerfProfiler instrumentation? (y/n)" "n")
case $FORGE_PROFILING in
    y|Y|yes) FORGE_PROFILING="ON" ;;
    *) FORGE_PROFILING="OFF" ;;
esac
echo -e "  → PerfProfiler: ${GREEN}$FORGE_PROFILING${NC}"
echo

# --- Cross-arch tests (ARM64/PPC64 under QEMU) ---
echo -e "${CYAN}Cross-arch tests:${NC}"
USE_CROSS_TESTS=$(ask_option "Build ARM/PPC cross tests under QEMU? (y/n)" "n")
case $USE_CROSS_TESTS in
    y|Y|yes) USE_CROSS_TESTS="ON" ;;
    *) USE_CROSS_TESTS="OFF" ;;
esac
echo -e "  → Cross tests: ${GREEN}$USE_CROSS_TESTS${NC}"
echo

# --- Native build (-march=native) ---
echo -e "${CYAN}Native build:${NC}"
USE_NATIVE=$(ask_option "Tune for this CPU (-march=native)? (y/n)" "y")
case $USE_NATIVE in
    y|Y|yes) USE_NATIVE="ON" ;;
    *) USE_NATIVE="OFF" ;;
esac
echo -e "  → Native build: ${GREEN}$USE_NATIVE${NC}"
echo

# --- Python bindings ---
echo -e "${CYAN}Python bindings:${NC}"
USE_PYTHON=$(ask_option "Build Python bindings? (y/n)" "y")
case $USE_PYTHON in
    y|Y|yes) USE_PYTHON="ON" ;;
    *) USE_PYTHON="OFF" ;;
esac
echo -e "  → Python: ${GREEN}$USE_PYTHON${NC}"
echo

# --- Tests ---
echo -e "${CYAN}Tests:${NC}"
USE_TESTS=$(ask_option "Build forge test executables? (y/n)" "n")
case $USE_TESTS in
    y|Y|yes) USE_TESTS="ON" ;;
    *) USE_TESTS="OFF" ;;
esac
echo -e "  → Tests: ${GREEN}$USE_TESTS${NC}"
echo

# --- Benchmarks ---
echo -e "${CYAN}Benchmarks:${NC}"
USE_BENCH=$(ask_option "Build micro-benchmarks? (y/n)" "n")
case $USE_BENCH in
    y|Y|yes) USE_BENCH="ON" ;;
    *) USE_BENCH="OFF" ;;
esac
echo -e "  → Benchmarks: ${GREEN}$USE_BENCH${NC}"
echo

# --- Parallel Jobs ---
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
echo -e "${CYAN}Build Parallelism:${NC}"
JOBS=$(ask_option "Parallel jobs" "$NPROC")
echo -e "  → Jobs: ${GREEN}$JOBS${NC}"
echo

# --- Build Directory ---
BUILD_DIR=$(ask_option "Build directory" "build")
echo -e "  → Build dir: ${GREEN}$BUILD_DIR${NC}"
echo

# --- Summary ---
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}       Build Summary${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "  Build Type:     ${GREEN}$BUILD_TYPE${NC}"
echo -e "  CUDA:           ${GREEN}$USE_CUDA${NC}"
if [ "$USE_CUDA" = "ON" ]; then
    echo -e "  cuBLAS:         ${GREEN}$USE_CUBLAS${NC}"
    echo -e "  CUDA Arch:      ${GREEN}$CUDA_ARCH${NC}"
fi
echo -e "  OpenBLAS:       ${GREEN}$USE_OPENBLAS${NC}"
echo -e "  PerfProfiler:   ${GREEN}$FORGE_PROFILING${NC}"
echo -e "  Cross Tests:    ${GREEN}$USE_CROSS_TESTS${NC}"
echo -e "  Native Build:   ${GREEN}$USE_NATIVE${NC}"
echo -e "  Python:         ${GREEN}$USE_PYTHON${NC}"
echo -e "  Tests:          ${GREEN}$USE_TESTS${NC}"
echo -e "  Benchmarks:     ${GREEN}$USE_BENCH${NC}"
echo -e "  Parallel Jobs:  ${GREEN}$JOBS${NC}"
echo -e "  Build Dir:      ${GREEN}$BUILD_DIR${NC}"
echo -e "${CYAN}========================================${NC}"
echo

# --- Confirm ---
CONFIRM=$(ask_option "Start build? (y/n)" "y")
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo -e "${RED}Build cancelled.${NC}"
    exit 0
fi

# --- Configure ---
echo
echo -e "${GREEN}[1/2] Configuring CMake...${NC}"
CMAKE_ARGS=(
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [ "$USE_CUDA" = "ON" ]; then
    CMAKE_ARGS+=(
        -DFORGE_USE_CUDA=ON
        -DUSE_CUBLAS="$USE_CUBLAS"
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    )
else
    CMAKE_ARGS+=(-DFORGE_USE_CUDA=OFF)
fi

if [ "$USE_OPENBLAS" = "ON" ]; then
    CMAKE_ARGS+=(-DUSE_OPENBLAS=ON)
fi

CMAKE_ARGS+=(-DFORGE_PROFILING="$FORGE_PROFILING")
CMAKE_ARGS+=(-DFORGE_NATIVE_BUILD="$USE_NATIVE")
CMAKE_ARGS+=(-DFORGE_BUILD_PYTHON="$USE_PYTHON")
CMAKE_ARGS+=(-DFORGE_BUILD_TESTS="$USE_TESTS")
# FORGE_BUILD_CROSS_TESTS / FORGE_BUILD_BENCHMARKS are defined inside
# tests/CMakeLists.txt, so only pass them when the test suite is enabled;
# otherwise CMake warns "Manually-specified variables were not used".
if [ "$USE_TESTS" = "ON" ]; then
  CMAKE_ARGS+=(-DFORGE_BUILD_CROSS_TESTS="$USE_CROSS_TESTS")
  CMAKE_ARGS+=(-DFORGE_BUILD_BENCHMARKS="$USE_BENCH")
fi

cmake "${CMAKE_ARGS[@]}"

# --- Build ---
echo
echo -e "${GREEN}[2/2] Building...${NC}"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}       Build Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

# Show output locations
if [ -f "$BUILD_DIR/forge-cli" ]; then
    echo -e "  CLI:       ${CYAN}$BUILD_DIR/forge-cli${NC}"
fi
if [ -f "$BUILD_DIR/Release/forge-cli.exe" ]; then
    echo -e "  CLI:       ${CYAN}$BUILD_DIR/Release/forge-cli.exe${NC}"
fi
echo -e "  Python:    ${CYAN}$BUILD_DIR/ (or $BUILD_DIR/Release/)${NC}"
echo
