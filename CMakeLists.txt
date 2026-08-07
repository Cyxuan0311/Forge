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

option(FORGE_NATIVE_BUILD
       "Build with -march=native (x86) or -mcpu=apple-m1 (macOS arm64)" ON)
option(FORGE_TUNE_MARCH "Override -march flag (empty = auto)" "")

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
  if(FORGE_NATIVE_BUILD)
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
      # macOS arm64: -march=native is not well-supported by Apple Clang
      set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=apple-m1")
      set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Wno-deprecated-gpu-targets")
      message(STATUS "Native build: -mcpu=apple-m1 (macOS arm64)")
    else()
      set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
      set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Wno-deprecated-gpu-targets")
      message(STATUS "Native build: -march=native")
    endif()
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Wno-deprecated-gpu-targets")
    message(STATUS "Portable build: no -march flag (FORGE_NATIVE_BUILD=OFF)")
  endif()
  if(FORGE_TUNE_MARCH)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${FORGE_TUNE_MARCH}")
    message(STATUS "Custom march: ${FORGE_TUNE_MARCH}")
  endif()
endif()

# ---- CPU feature detection (two-level: arch + SIMD) ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(detect_cpu_features)

# ---- forge_cpu_features INTERFACE target ----
# All targets that need arch-specific headers or SIMD compile flags link to
# this.
add_library(forge_cpu_features INTERFACE)
target_compile_definitions(forge_cpu_features
                           INTERFACE FORGE_ARCH_${FORGE_ARCH})
target_include_directories(forge_cpu_features
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/operators)

if(FORGE_CPU_HAS_AVX2)
  target_compile_definitions(forge_cpu_features INTERFACE USE_AVX2)
  target_compile_options(forge_cpu_features INTERFACE -mavx2 -mfma -mf16c)
endif()
if(FORGE_CPU_HAS_AVX512_VNNI)
  target_compile_definitions(forge_cpu_features INTERFACE USE_AVX512_VNNI)
  target_compile_options(
    forge_cpu_features INTERFACE -mavx512f -mavx512bw -mavx512vnni -mavx512vl
                                 -mf16c)
endif()
if(FORGE_CPU_HAS_NEON)
  target_compile_definitions(forge_cpu_features INTERFACE USE_NEON)
endif()
if(FORGE_CPU_HAS_DOTPROD)
  target_compile_definitions(forge_cpu_features INTERFACE USE_DOTPROD)
  target_compile_options(
    forge_cpu_features
    INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=armv8.2-a+dotprod>)
endif()
if(FORGE_CPU_HAS_MATMUL_INT8)
  target_compile_definitions(forge_cpu_features INTERFACE USE_MATMUL_INT8)
  target_compile_options(
    forge_cpu_features
    INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=armv8.2-a+dotprod+i8mm>)
endif()
if(FORGE_CPU_HAS_VSX)
  target_compile_definitions(forge_cpu_features INTERFACE USE_VSX)
  target_compile_options(forge_cpu_features INTERFACE -mcpu=power8)
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

option(FORGE_PROFILING "Enable PerfProfiler instrumentation" OFF)
if(FORGE_PROFILING)
  message(STATUS "PerfProfiler enabled - FORGE_PROFILING=ON")
  add_compile_definitions(FORGE_PROFILING=1)
else()
  message(STATUS "PerfProfiler disabled - FORGE_PROFILING=OFF (zero-cost)")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_JOBS 0)
set(PYBIND11_LTO_CXX_FLAGS "")

# ---- Optional CUDA support ----
# CUDA is unsupported on Apple platforms (macOS deprecated CUDA in 10.14, and
# nvcc cannot compile for arm64). Intel Macs can technically use CUDA with
# outdated drivers, but we disable it for simplicity.
option(FORGE_USE_CUDA "Enable CUDA GPU acceleration" ON)

if(APPLE)
  message(STATUS "CUDA disabled on Apple platforms - building CPU-only")
elseif(FORGE_USE_CUDA)
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

option(FORGE_BUILD_PYTHON
       "Build Python bindings (requires Python3 development)" ON)
if(FORGE_BUILD_PYTHON)
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
else()
  message(
    STATUS "Python bindings disabled (FORGE_BUILD_PYTHON=OFF) - CLI-only build")
endif()

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
target_include_directories(
  forge_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/operators)
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge_core PUBLIC USE_CUDA)
  target_link_libraries(forge_core PUBLIC CUDA::cudart)
endif()
target_link_libraries(forge_core
                      PUBLIC $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
target_link_libraries(forge_core PUBLIC forge_cpu_features)

# ---- Operators library (static) ----
# Defined in src/operators/CMakeLists.txt
add_subdirectory(src/operators)

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
  src/model/quant_policy.cpp
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
add_library(
  forge_vision STATIC
  src/vision/vision_encoder.cpp src/vision/vision_registry.cpp
  src/vision/vit_shared.cpp src/vision/qwen3vl_encoder.cpp)
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
  src/inference/execution_plan.cpp
  src/inference/engines/transformer_engine.cpp
  src/inference/engines/deepseek_engine.cpp
  src/inference/engines/qwen35_engine.cpp
  src/inference/engines/gemma4_engine.cpp
  src/inference/engines/phimoe_engine.cpp
  src/inference/engines/generic_engine.cpp
  src/inference/layers/norm_executor.cpp
  src/inference/layers/rope_executor.cpp
  src/inference/layers/attention_selector.cpp
  src/inference/layers/attention_executor.cpp
  src/inference/layers/ffn_executor.cpp
  src/inference/layers/mla_executor.cpp
  src/inference/layers/deepseek_layer_executor.cpp
  src/inference/layers/gemma4_embedding.cpp
  src/inference/layers/gemma4_attention.cpp
  src/inference/layers/gemma4_moe.cpp
  src/inference/layers/qwen35_recurrent_memory.cpp
  src/inference/layers/qwen35_rope.cpp
  src/inference/layers/qwen35_full_attention.cpp
  src/inference/layers/qwen35_linear_attention.cpp
  src/inference/graph/graph_key.cpp
  src/inference/graph/layer_graph_builder.cpp
  src/inference/graph/llama_graph_builder.cpp
  src/inference/graph/deepseek_graph_builder.cpp
  src/inference/graph/qwen35_graph_builder.cpp
  src/inference/graph/graph_runtime.cpp
  src/inference/graph/graph_cuda_runner.cpp
  src/inference/context.cpp
  src/inference/sampler.cpp
  src/inference/generator.cpp
  src/inference/paged_kv_cache.cpp
  src/inference/attention_mask.cpp
  src/inference/request_scheduler.cpp
  src/inference/speculative.cpp)
target_include_directories(
  forge_infer
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/operators)
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge_infer PUBLIC USE_CUDA)
endif()
target_link_libraries(
  forge_infer PUBLIC forge_core forge_ops forge_model
                     $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
target_link_libraries(forge_infer PUBLIC forge_cpu_features)

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
if(FORGE_BUILD_PYTHON)
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
endif()
