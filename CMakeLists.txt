cmake_minimum_required(VERSION 3.18)
project(NanoInfer VERSION 0.5.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Wno-deprecated-gpu-targets")

# Detect AVX2 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2 -mfma -mf16c" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2/FMA/F16C supported - enabling optimized CPU kernels")
    add_compile_definitions(USE_AVX2)
else()
    message(STATUS "AVX2 not supported - using scalar CPU kernels")
endif()

option(USE_CUBLAS "Use cuBLAS for GPU matrix operations" ON)
if(USE_CUBLAS)
    message(STATUS "cuBLAS enabled - using cuBLAS for GPU GEMM")
else()
    message(STATUS "cuBLAS disabled - using pure CUDA tiled GEMM kernel")
endif()

option(USE_OPENBLAS "Use OpenBLAS for CPU FP32 matrix operations" OFF)
if(USE_OPENBLAS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENBLAS REQUIRED openblas)
    if(OPENBLAS_FOUND)
        message(STATUS "OpenBLAS found at ${OPENBLAS_LIBRARY_DIRS}")
    endif()
else()
    message(STATUS "OpenBLAS disabled - using hand-tuned CPU kernels")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(PYBIND11_LTO_CXX_FLAGS "")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# ---- Core library (static) ----
add_library(nanoinfer_core STATIC
    src/core/tensor.cpp
    src/core/memory_pool.cpp
    src/core/backend.cpp
    src/core/backend_scheduler.cpp
    src/core/op_kernels.cpp
    src/core/op_dispatch.cpp
    src/core/memory_planner.cpp
    src/core/compute_graph.cpp
    src/debug/logger.cpp
    src/inference/kv_cache.cpp
)
target_include_directories(nanoinfer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(nanoinfer_core PUBLIC USE_CUDA)
target_link_libraries(nanoinfer_core PUBLIC
    CUDA::cudart
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# ---- Operators library (static) ----
add_library(nanoinfer_ops STATIC
    src/operators/norm.cpp
    src/operators/activation.cpp
    src/operators/matmul.cpp
    src/operators/embedding.cpp
    src/operators/rope.cpp
    src/operators/elementwise.cpp
    src/operators/attention.cpp
    src/operators/cuda_norm.cu
    src/operators/cuda_embedding.cu
    src/operators/cuda_gemv.cu
    src/operators/cuda_attention.cu
    src/operators/cuda_quant.cu
    src/operators/cuda_gemm.cu
)
target_include_directories(nanoinfer_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(nanoinfer_ops PUBLIC USE_CUDA)
target_compile_definitions(nanoinfer_ops PRIVATE NANOINFER_USE_CUBLAS=$<BOOL:${USE_CUBLAS}>)
target_compile_definitions(nanoinfer_ops PRIVATE NANOINFER_USE_OPENBLAS=$<BOOL:${USE_OPENBLAS}>)
target_include_directories(nanoinfer_ops PRIVATE ${OPENBLAS_INCLUDE_DIRS})
target_link_libraries(nanoinfer_ops PUBLIC
    nanoinfer_core
    $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>
    $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# ---- Model library (static) ----
add_library(nanoinfer_model STATIC
    src/model/model.cpp
    src/model/model_loader.cpp
    src/model/ninf_model.cpp
    src/model/gguf_model.cpp
    src/inference/engine_factory.cpp
)
target_include_directories(nanoinfer_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(nanoinfer_model PUBLIC USE_CUDA)
target_link_libraries(nanoinfer_model PUBLIC
    nanoinfer_core
)

# ---- Tokenizer library (static) ----
add_library(nanoinfer_tokenizer STATIC
    src/tokenizer/tokenizer.cpp
)
target_include_directories(nanoinfer_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(nanoinfer_tokenizer PUBLIC
    nanoinfer_core
)

# ---- Vision library (static) ----
add_library(nanoinfer_vision STATIC
    src/vision/vision_encoder.cpp
)
target_include_directories(nanoinfer_vision PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(nanoinfer_vision PUBLIC USE_CUDA)
target_link_libraries(nanoinfer_vision PUBLIC
    nanoinfer_core
    nanoinfer_ops
    nanoinfer_model
    nanoinfer_infer
)

# ---- Inference library (static) ----
add_library(nanoinfer_infer STATIC
    src/inference/perf_profiler.cpp
    src/inference/engines/transformer_engine.cpp
    src/inference/engines/llama_engine.cpp
    src/inference/engines/deepseek_engine.cpp
    src/inference/engines/qwen35_engine.cpp
    src/inference/engines/layer_graph_builder.cpp
    src/inference/engines/llama_graph_builder.cpp
    src/inference/engines/deepseek_graph_builder.cpp
    src/inference/engines/qwen35_graph_builder.cpp
    src/inference/context.cpp
    src/inference/sampler.cpp
    src/inference/generator.cpp
    src/inference/paged_kv_cache.cpp
    src/inference/request_scheduler.cpp
)
target_include_directories(nanoinfer_infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(nanoinfer_infer PUBLIC USE_CUDA)
target_link_libraries(nanoinfer_infer PUBLIC
    nanoinfer_core
    nanoinfer_ops
    nanoinfer_model
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# ---- CLI executable ----
add_executable(nanoinfer-cli
    src/cli/nanoinfer_cli.cpp
    src/cli/cli_args.cpp
    src/cli/cli_chat_template.cpp
    src/cli/cli_generation.cpp
    src/cli/cli_interactive.cpp
)
target_include_directories(nanoinfer-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(nanoinfer-cli PRIVATE USE_CUDA)
target_link_libraries(nanoinfer-cli PRIVATE
    nanoinfer_infer
    nanoinfer_model
    nanoinfer_ops
    nanoinfer_core
    nanoinfer_tokenizer
    nanoinfer_vision
    CUDA::cudart
    $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>
    $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

set_target_properties(nanoinfer-cli PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86;89"
)

# ---- Python bindings ----
pybind11_add_module(nanoinfer
    src/bindings/nanoinfer_py.cpp
    src/bindings/core_types.cpp
    src/bindings/model.cpp
    src/bindings/tokenizer.cpp
    src/bindings/multimodal.cpp
    src/bindings/scheduler.cpp
    src/bindings/backend.cpp
    src/bindings/logger.cpp
    src/bindings/profiler.cpp
)
target_link_libraries(nanoinfer PRIVATE
    nanoinfer_infer
    nanoinfer_model
    nanoinfer_ops
    nanoinfer_core
    nanoinfer_tokenizer
    nanoinfer_vision
    Python3::Python
    Python3::NumPy
    $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>
)
target_include_directories(nanoinfer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(nanoinfer PRIVATE
    VERSION_INFO=${PROJECT_VERSION}
    USE_CUDA
)
set_target_properties(nanoinfer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86;89"
)