cmake_minimum_required(VERSION 3.18)
project(
  Forge
  VERSION 0.5.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Wno-deprecated-gpu-targets")
endif()

# Detect AVX2 support (x86/x86_64 only)
include(CheckCXXCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)")
  check_cxx_compiler_flag("-mavx2 -mfma -mf16c" COMPILER_SUPPORTS_AVX2)
  if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2/FMA/F16C supported - enabling optimized CPU kernels")
    add_compile_definitions(USE_AVX2)
  else()
    message(STATUS "AVX2 not supported - using scalar CPU kernels")
  endif()
else()
  message(
    STATUS
      "Non-x86 architecture (${CMAKE_SYSTEM_PROCESSOR}) - using scalar CPU kernels"
  )
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
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_JOBS 0)
set(PYBIND11_LTO_CXX_FLAGS "")

# ---- Optional CUDA support ----
option(FORGE_USE_CUDA "Enable CUDA GPU acceleration" ON)

if(FORGE_USE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found - building with GPU support")
  else()
    message(STATUS "CUDA not found - building CPU-only")
  endif()
else()
  message(STATUS "CUDA disabled by user - building CPU-only")
endif()

# ---- Auto-detect CUDA architecture ----
set(FORGE_CUDA_ARCH "86;89")
if(CUDAToolkit_FOUND)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(detect_cuda_arch)
  detect_cuda_architecture(FORGE_CUDA_ARCH)
endif()

find_package(OpenMP)
find_package(
  Python3
  COMPONENTS Interpreter Development NumPy
  REQUIRED)

set(FETCHCONTENT_BASE_DIR
    "${CMAKE_SOURCE_DIR}/.cache/fetchcontent"
    CACHE PATH "FetchContent cache directory")

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://ghproxy.net/https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6)
FetchContent_MakeAvailable(pybind11)

# ---- Core library (static) ----
add_library(
  forge_core STATIC
  src/core/tensor.cpp
  src/core/memory_pool.cpp
  src/core/backend.cpp
  src/core/backend_scheduler.cpp
  src/core/op_kernels.cpp
  src/core/op_dispatch.cpp
  src/core/memory_planner.cpp
  src/core/compute_graph.cpp
  src/debug/logger.cpp
  src/inference/kv_cache.cpp)
target_include_directories(forge_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                             ${CUDAToolkit_INCLUDE_DIRS})
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge_core PUBLIC USE_CUDA)
  target_link_libraries(forge_core PUBLIC CUDA::cudart)
endif()
target_link_libraries(forge_core
                      PUBLIC $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)

# ---- Operators library (static) ----
add_library(
  forge_ops STATIC
  src/operators/cpu/norm.cpp
  src/operators/cpu/activation.cpp
  src/operators/cpu/matmul.cpp
  src/operators/cpu/quant_traits.cpp
  src/operators/cpu/embedding.cpp
  src/operators/cpu/rope.cpp
  src/operators/cpu/elementwise.cpp
  src/operators/cpu/attention.cpp)
if(CUDAToolkit_FOUND)
  target_sources(
    forge_ops
    PRIVATE src/operators/cuda/cuda_rms_norm.cu
            src/operators/cuda/cuda_activation.cu
            src/operators/cuda/cuda_elementwise.cu
            src/operators/cuda/cuda_rope.cu
            src/operators/cuda/cuda_flash_attn.cu
            src/operators/cuda/cuda_embedding.cu
            src/operators/cuda/cuda_gemv.cu
            src/operators/cuda/cuda_gemv_instances.cu
            src/operators/cuda/cuda_fused.cu
            src/operators/cuda/cuda_quant.cu
            src/operators/cuda/cuda_gemm.cu)
  target_compile_definitions(forge_ops PUBLIC USE_CUDA)
endif()

target_include_directories(forge_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                            ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(forge_ops
                           PRIVATE FORGE_USE_CUBLAS=$<BOOL:${USE_CUBLAS}>)
target_compile_definitions(forge_ops
                           PRIVATE FORGE_USE_OPENBLAS=$<BOOL:${USE_OPENBLAS}>)
target_include_directories(forge_ops PRIVATE ${OPENBLAS_INCLUDE_DIRS})
target_link_libraries(
  forge_ops PUBLIC forge_core $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>
                   $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
if(CUDAToolkit_FOUND)
  target_link_libraries(forge_ops PUBLIC $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>)
endif()

# ---- Model library (static) ----
add_library(
  forge_model STATIC
  src/model/model.cpp
  src/model/model_loader.cpp
  src/model/ninf_model.cpp
  src/model/gguf_model.cpp
  src/model/arch_config_parser.cpp
  src/model/arch_weight_init.cpp
  src/model/arch_capability.cpp
  src/model/arch_registrations.cpp
  src/model/model_weights.cpp
  src/model/weight_store.cpp
  src/model/weight_mapper.cpp
  src/inference/engine_factory.cpp)
target_include_directories(
  forge_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                     ${CUDAToolkit_INCLUDE_DIRS})
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge_model PUBLIC USE_CUDA)
endif()
target_link_libraries(forge_model PUBLIC forge_core)

# ---- Tokenizer library (static) ----
add_library(forge_tokenizer STATIC src/tokenizer/tokenizer.cpp)
target_include_directories(forge_tokenizer
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(forge_tokenizer PUBLIC forge_core)

# ---- Vision library (static) ----
add_library(forge_vision STATIC src/vision/vision_encoder.cpp)
target_include_directories(
  forge_vision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                      ${CUDAToolkit_INCLUDE_DIRS})
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge_vision PUBLIC USE_CUDA)
endif()
target_link_libraries(forge_vision PUBLIC forge_core forge_ops forge_model
                                          forge_infer)

# ---- Inference library (static) ----
add_library(
  forge_infer STATIC
  src/inference/perf_profiler.cpp
  src/inference/engines/transformer_engine.cpp
  src/inference/engines/deepseek_engine.cpp
  src/inference/engines/qwen35_engine.cpp
  src/inference/engines/gemma4_engine.cpp
  src/inference/engines/generic_engine.cpp
  src/inference/engines/layer_graph_builder.cpp
  src/inference/engines/llama_graph_builder.cpp
  src/inference/engines/deepseek_graph_builder.cpp
  src/inference/engines/qwen35_graph_builder.cpp
  src/inference/context.cpp
  src/inference/sampler.cpp
  src/inference/generator.cpp
  src/inference/paged_kv_cache.cpp
  src/inference/request_scheduler.cpp)
target_include_directories(
  forge_infer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                     ${CUDAToolkit_INCLUDE_DIRS})
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge_infer PUBLIC USE_CUDA)
endif()
target_link_libraries(
  forge_infer PUBLIC forge_core forge_ops forge_model
                     $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)

# ---- CLI executable ----
add_executable(
  forge-cli
  src/cli/forge_cli.cpp src/cli/cli_args.cpp src/cli/cli_chat_template.cpp
  src/cli/cli_generation.cpp src/cli/cli_interactive.cpp)
target_include_directories(forge-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                             ${CUDAToolkit_INCLUDE_DIRS})
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge-cli PRIVATE USE_CUDA)
endif()
target_link_libraries(
  forge-cli
  PRIVATE forge_infer
          forge_model
          forge_ops
          forge_core
          forge_tokenizer
          forge_vision
          $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>
          $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
if(CUDAToolkit_FOUND)
  target_link_libraries(forge-cli PRIVATE CUDA::cudart
                                          $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>)
  set_target_properties(
    forge-cli
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON
               CUDA_ARCHITECTURES "${FORGE_CUDA_ARCH}")
endif()

# ---- Python bindings ----
pybind11_add_module(
  forge
  src/bindings/forge_py.cpp
  src/bindings/core_types.cpp
  src/bindings/model.cpp
  src/bindings/tokenizer.cpp
  src/bindings/multimodal.cpp
  src/bindings/scheduler.cpp
  src/bindings/backend.cpp
  src/bindings/logger.cpp
  src/bindings/profiler.cpp)
target_link_libraries(
  forge
  PRIVATE forge_infer
          forge_model
          forge_ops
          forge_core
          forge_tokenizer
          forge_vision
          Python3::Python
          Python3::NumPy
          $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>)
target_include_directories(forge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                         ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(forge PRIVATE VERSION_INFO=${PROJECT_VERSION})
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge PRIVATE USE_CUDA)
  set_target_properties(
    forge
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON
               CUDA_ARCHITECTURES "${FORGE_CUDA_ARCH}")
endif()
