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

  # Default to the upstream repo. Users behind a mirror can override with:
  #   -DFORGE_PYBIND11_REPO=https://ghproxy.net/https://github.com/pybind/pybind11.git
  # or transparently via: git config --global url."https://ghproxy.net/https://github.com/".insteadOf "https://github.com/"
  set(FORGE_PYBIND11_REPO
      "https://github.com/pybind/pybind11.git"
      CACHE STRING "pybind11 git repository URL (mirror-friendly)")

  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY "${FORGE_PYBIND11_REPO}"
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
  src/core/fp8_scale_debug.cpp # 6-C-5.0: FP8 scale debug/trace instrumentation
  src/core/host_mem_pool.cpp
  src/core/cuda_mem_pool.cpp
  src/core/backend.cpp
  src/core/backend_scheduler.cpp
  src/core/op_kernels.cpp
  src/core/op_dispatch.cpp
  src/core/memory_planner.cpp
  src/core/compute_graph.cpp
  src/debug/logger.cpp
  src/inference/kv_block_swapper.cpp
  src/inference/kv_cache.cpp
  src/inference/kv_memory.cpp
  src/inference/kv_sizing.cpp
  src/inference/kv_storage.cpp
  src/inference/kv_working_set.cpp
  src/inference/prefix_cache.cpp)
target_include_directories(
  forge_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/operators)
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge_core PUBLIC USE_CUDA)
  target_link_libraries(forge_core PUBLIC CUDA::cudart)
endif()
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

# ---- Agent module (static) ----
# Pluggable tool-call dialects + <think> reasoning separation for the server.
# Pure host logic (nlohmann/json + std); no engine or tokenizer dependency.
add_library(
  forge_agent STATIC src/agent/tool_dialect.cpp src/agent/reasoning_filter.cpp
                     src/agent/agent_pipeline.cpp)
target_include_directories(
  forge_agent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                     ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

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
  src/inference/chat_template.cpp
  src/inference/engines/transformer_engine.cpp
  src/inference/engines/expert_page_cache.cpp
  src/inference/engines/deepseek_engine.cpp
  src/inference/engines/dflash_engine.cpp
  src/inference/dflash_draft_provider.cpp
  src/inference/engines/qwen35_engine.cpp
  src/inference/engines/lfm2_engine.cpp
  src/inference/engines/gemma4_engine.cpp
  src/inference/engines/gemma_embedding2_engine.cpp
  src/inference/engines/spark25_engine.cpp
  src/inference/engines/phimoe_engine.cpp
  src/inference/engines/bailingmoe3_engine.cpp
  src/inference/engines/xing4_engine.cpp
  src/inference/engines/generic_engine.cpp
  src/inference/layers/norm_executor.cpp
  src/inference/layers/rope_executor.cpp
  src/inference/layers/attn_rot_executor.cpp
  src/inference/layers/attention_selector.cpp
  src/inference/layers/attention_executor.cpp
  src/inference/layers/ffn_executor.cpp
  src/inference/layers/mla_executor.cpp
  src/inference/layers/deepseek_layer_executor.cpp
  src/inference/layers/gemma4_embedding.cpp
  src/inference/layers/gemma4_attention.cpp
  src/inference/layers/gemma4_moe.cpp
  src/inference/layers/qwen35_recurrent_memory.cpp
  src/inference/layers/bailingmoe3_recurrent_memory.cpp
  src/inference/layers/qwen35_rope.cpp
  src/inference/layers/qwen35_full_attention.cpp
  src/inference/layers/qwen35_linear_attention.cpp
  src/inference/layers/qwen3vl_full_attention.cpp
  src/inference/layers/xing4_mhc.cpp
  src/inference/graph/graph_key.cpp
  src/inference/graph/layer_graph_builder.cpp
  src/inference/graph/llama_graph_builder.cpp
  src/inference/graph/deepseek_graph_builder.cpp
  src/inference/graph/qwen35_graph_builder.cpp
  src/inference/graph/qwen3vl_graph_builder.cpp
  src/inference/graph/graph_runtime.cpp
  src/inference/graph/graph_cuda_runner.cpp
  src/inference/context.cpp
  src/inference/sampler.cpp
  src/inference/metrics.cpp
  src/inference/generator.cpp
  src/inference/attention_mask.cpp
  src/inference/request_scheduler.cpp
  src/inference/speculative.cpp
  src/inference/speculative_executor.cpp
  src/inference/kv_cache_dtype.cpp
  src/inference/cli_common_args.cpp
  src/inference/cli_spec_args.cpp
  src/inference/cli_dashboard_args.cpp
  src/inference/model_draft_provider.cpp
  src/inference/mtp_draft_provider.cpp
  src/inference/threads.cpp
  # Jinja engine (vendored from llama.cpp, MIT licensed)
  third_party/jinja/lexer.cpp
  third_party/jinja/parser.cpp
  third_party/jinja/runtime.cpp
  third_party/jinja/jinja_string.cpp
  third_party/jinja/value.cpp
  third_party/jinja/caps.cpp
  third_party/jinja/unicode.cpp)
target_include_directories(
  forge_infer
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/operators
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/jinja
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge_infer PUBLIC USE_CUDA)
endif()
target_link_libraries(forge_infer PUBLIC forge_core forge_ops forge_model)
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
          forge_dashboard
          forge_vision
          $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>)
if(CUDAToolkit_FOUND)
  target_link_libraries(forge-cli PRIVATE CUDA::cudart
                                          $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>)
  set_target_properties(
    forge-cli
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON
               CUDA_ARCHITECTURES "${FORGE_CUDA_ARCH}")
endif()

# ---- forge-server (OpenAI-compatible HTTP API server) ----
add_library(forge_httplib STATIC third_party/cpp-httplib/httplib.cpp
                                 third_party/cpp-httplib/httplib.h)
target_include_directories(
  forge_httplib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpp-httplib)
find_package(Threads REQUIRED)
target_link_libraries(forge_httplib PUBLIC Threads::Threads)

# ---- Web dashboard (shared by forge-cli and forge-server) ----
#
# The frontend lives in web/ (Vue 3 + TypeScript + Vite + ECharts) and is built
# by npm. Its output is embedded into the binary rather than loaded from disk:
# the release tarball ships a bare `forge-cli` with no asset directory beside it
# (see .github/workflows/release.yml), so a runtime file lookup would need a
# --dashboard-root escape hatch and would break for every installed copy.
#
# Default is OFF so that a plain `cmake -B build` needs no Node.js toolchain.
# When it is ON and npm is missing we warn and degrade instead of failing: the
# dashboard still serves /api/metrics, and /dashboard/ explains how to rebuild.
option(FORGE_BUILD_DASHBOARD
       "Build the web frontends (dashboard + chat UI) with npm (requires Node.js >= 18)" OFF)
find_program(FORGE_NPM_EXECUTABLE NAMES npm)

# ---- Standalone chat UI (web-chat/) ----------------------------------------
# A Vue 3 + Vite project mirroring web/ (the dashboard), built by the same npm
# pipeline and embedded with FUNC_PREFIX=chat so the accessors become
# chat_assets()/chat_asset_count(). forge-server serves it on the API port
# under /chat/ (see server_routes.cpp); the dashboard keeps its own loopback
# port. Without Node.js the chat tables degrade to the empty stubs in
# dashboard_server.cpp, same as the dashboard's.
set(_forge_chat_dir ${CMAKE_CURRENT_SOURCE_DIR}/web-chat)
set(_forge_chat_dist ${CMAKE_CURRENT_BINARY_DIR}/web-chat/dist)
set(_forge_chat_assets_src ${CMAKE_CURRENT_BINARY_DIR}/chat_assets.cpp)

if(FORGE_BUILD_DASHBOARD AND FORGE_NPM_EXECUTABLE)
  set(_forge_web_dir ${CMAKE_CURRENT_SOURCE_DIR}/web)
  set(_forge_web_dist ${CMAKE_CURRENT_BINARY_DIR}/web/dist)
  # `npm ci` deletes and recreates node_modules, so two concurrent builds racing
  # on it fail with ENOTEMPTY. Gating it behind a stamp file means it runs once
  # per lockfile change instead of on every `cmake --build`, which removes that
  # race in practice and stops every incremental build from spending a minute
  # reinstalling ~200 packages it already has.
  set(_forge_npm_stamp ${CMAKE_CURRENT_BINARY_DIR}/forge-dashboard-npm.stamp)
  add_custom_command(
    OUTPUT ${_forge_npm_stamp}
    # --cache pins the store inside the build tree so a dashboard build never
    # writes to ~/.npm (read-only CI containers) and never pollutes the repo.
    COMMAND ${CMAKE_COMMAND} -E env npm ci --prefix ${_forge_web_dir} -- --cache
            ${CMAKE_CURRENT_BINARY_DIR}/npm-cache
    COMMAND ${CMAKE_COMMAND} -E touch ${_forge_npm_stamp}
    WORKING_DIRECTORY ${_forge_web_dir}
    DEPENDS ${_forge_web_dir}/package.json ${_forge_web_dir}/package-lock.json
    COMMENT "npm ci (web dashboard dependencies)"
    VERBATIM)
  add_custom_target(forge-dashboard-deps DEPENDS ${_forge_npm_stamp})
  # Every frontend source file is an explicit build dependency.
  #
  # An explicit list, not file(GLOB_RECURSE ... CONFIGURE_DEPENDS). The
  # CONFIGURE_DEPENDS variant re-globbed at build time, reported "GLOB
  # mismatch!" on every invocation, and re-ran configure mid-build until
  # `cmake --build` wedged. A couple of dozen paths cost nothing to write down
  # and make the dependency set deterministic.
  #
  # Some dependency list is mandatory either way: without one, editing a .vue
  # file left dist/ (and therefore the embedded binary) untouched, so the build
  # silently relinked the OLD bundle -- a change that looks like it did nothing.
  set(_forge_web_sources
      ${_forge_web_dir}/index.html
      ${_forge_web_dir}/vite.config.ts
      ${_forge_web_dir}/tsconfig.json
      ${_forge_web_dir}/env.d.ts
      ${_forge_web_dir}/src/main.ts
      ${_forge_web_dir}/src/App.vue
      ${_forge_web_dir}/src/api.ts
      ${_forge_web_dir}/src/types.ts
      ${_forge_web_dir}/src/theme.ts
      ${_forge_web_dir}/src/style.css
      ${_forge_web_dir}/src/echarts.ts
      ${_forge_web_dir}/src/charts/gauge.ts
      ${_forge_web_dir}/src/charts/latency.ts
      ${_forge_web_dir}/src/charts/lines.ts
      ${_forge_web_dir}/src/charts/operators.ts
      ${_forge_web_dir}/src/charts/option.ts
      ${_forge_web_dir}/src/charts/timeseries.ts
      ${_forge_web_dir}/src/composables/palette.ts
      ${_forge_web_dir}/src/composables/theme.ts
      ${_forge_web_dir}/src/export/format.ts
      ${_forge_web_dir}/src/export/io.ts
      ${_forge_web_dir}/src/components/EChart.vue
      ${_forge_web_dir}/src/components/ExportMenu.vue
      ${_forge_web_dir}/src/components/GaugePanel.vue
      ${_forge_web_dir}/src/components/LatencyChart.vue
      ${_forge_web_dir}/src/components/OperatorChart.vue
      ${_forge_web_dir}/src/components/RequestTable.vue
      ${_forge_web_dir}/src/components/StatTile.vue
      ${_forge_web_dir}/src/components/ThemeSlider.vue
      ${_forge_web_dir}/src/components/TimeSeriesChart.vue)
  add_custom_command(
    OUTPUT ${_forge_web_dist}/index.html
    # --outDir sends the bundle into the build tree so a `npm run build` never
    # leaves generated files in the source checkout. --emptyOutDir is what
    # suppresses Vite's interactive confirmation for an outDir outside the
    # project root, which would otherwise hang a non-interactive build.
    COMMAND ${CMAKE_COMMAND} -E env npm run build -- --outDir ${_forge_web_dist}
            --emptyOutDir
    WORKING_DIRECTORY ${_forge_web_dir}
    DEPENDS ${_forge_npm_stamp} ${_forge_web_sources}
    COMMENT "npm run build (web dashboard)"
    VERBATIM)
  add_custom_target(forge-dashboard-assets ALL
                    DEPENDS ${_forge_web_dist}/index.html)

  set(_forge_dashboard_assets_src
      ${CMAKE_CURRENT_BINARY_DIR}/dashboard_assets.cpp)
  add_custom_command(
    OUTPUT ${_forge_dashboard_assets_src}
    COMMAND
      ${CMAKE_COMMAND} -DINPUT_DIR=${_forge_web_dist}
      -DOUTPUT_FILE=${_forge_dashboard_assets_src} -P
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedDashboardAssets.cmake
    # Depend on the FILE, not on the forge-dashboard-assets TARGET. A target
    # dependency gives ordering but no file-level staleness, so make happily
    # considered a dashboard_assets.cpp from hours earlier up to date and the
    # binary kept serving the old bundle. Vite rewrites index.html on every
    # build, so its mtime is a reliable "the bundle changed" signal even though
    # the hashed files live in a sibling directory.
    DEPENDS ${_forge_web_dist}/index.html
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedDashboardAssets.cmake
    COMMENT "Embedding web dashboard assets"
    VERBATIM)

  # ---- chat UI: npm ci + build + embed (mirrors the dashboard pipeline) ----
  set(_forge_chat_npm_stamp ${CMAKE_CURRENT_BINARY_DIR}/forge-chat-npm.stamp)
  add_custom_command(
    OUTPUT ${_forge_chat_npm_stamp}
    COMMAND ${CMAKE_COMMAND} -E env npm ci --prefix ${_forge_chat_dir} -- --cache
            ${CMAKE_CURRENT_BINARY_DIR}/npm-cache
    COMMAND ${CMAKE_COMMAND} -E touch ${_forge_chat_npm_stamp}
    WORKING_DIRECTORY ${_forge_chat_dir}
    DEPENDS ${_forge_chat_dir}/package.json ${_forge_chat_dir}/package-lock.json
    COMMENT "npm ci (web chat UI dependencies)"
    VERBATIM)
  add_custom_target(forge-chat-deps DEPENDS ${_forge_chat_npm_stamp})

  set(_forge_chat_sources
      ${_forge_chat_dir}/index.html
      ${_forge_chat_dir}/vite.config.ts
      ${_forge_chat_dir}/tsconfig.json
      ${_forge_chat_dir}/env.d.ts
      ${_forge_chat_dir}/package.json
      ${_forge_chat_dir}/src/main.ts
      ${_forge_chat_dir}/src/App.vue
      ${_forge_chat_dir}/src/api.ts
      ${_forge_chat_dir}/src/markdown.ts
      ${_forge_chat_dir}/src/store.ts
      ${_forge_chat_dir}/src/types.ts
      ${_forge_chat_dir}/src/style.css
      ${_forge_chat_dir}/src/components/Composer.vue
      ${_forge_chat_dir}/src/components/ConnectionPanel.vue
      ${_forge_chat_dir}/src/components/ModalShell.vue
      ${_forge_chat_dir}/src/components/MessageItem.vue
      ${_forge_chat_dir}/src/components/Sidebar.vue
      ${_forge_chat_dir}/src/components/SettingsPanel.vue
      ${_forge_chat_dir}/src/components/ThinkingIndicator.vue
      ${_forge_chat_dir}/src/components/TopBar.vue)
  add_custom_command(
    OUTPUT ${_forge_chat_dist}/index.html
    COMMAND ${CMAKE_COMMAND} -E env npm run build -- --outDir ${_forge_chat_dist}
            --emptyOutDir
    WORKING_DIRECTORY ${_forge_chat_dir}
    DEPENDS ${_forge_chat_npm_stamp} ${_forge_chat_sources}
    COMMENT "npm run build (web chat UI)"
    VERBATIM)
  add_custom_target(forge-chat-assets ALL DEPENDS ${_forge_chat_dist}/index.html)

  add_custom_command(
    OUTPUT ${_forge_chat_assets_src}
    COMMAND
      ${CMAKE_COMMAND} -DINPUT_DIR=${_forge_chat_dist}
      -DOUTPUT_FILE=${_forge_chat_assets_src} -DFUNC_PREFIX=chat -P
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedDashboardAssets.cmake
    # File-level staleness on the dist shell (see the dashboard embed above).
    DEPENDS ${_forge_chat_dist}/index.html
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedDashboardAssets.cmake
    COMMENT "Embedding web chat UI assets"
    VERBATIM)

  add_library(forge_dashboard STATIC src/dashboard/dashboard_server.cpp
                                     src/dashboard/launcher.cpp
                                     ${_forge_dashboard_assets_src}
                                     ${_forge_chat_assets_src})
  target_compile_definitions(forge_dashboard PUBLIC FORGE_DASHBOARD_EMBEDDED=1)
  add_dependencies(forge_dashboard forge-dashboard-assets forge-chat-deps)
else()
  if(FORGE_BUILD_DASHBOARD)
    message(
      WARNING
        "FORGE_BUILD_DASHBOARD=ON but npm was not found. Building without the "
        "web frontend: /api/metrics will still work, /dashboard/ will explain "
        "how to rebuild. Install Node.js >= 18 and reconfigure.")
  else()
    message(
      STATUS
        "Web dashboard frontend OFF (-DFORGE_BUILD_DASHBOARD=ON to build it with npm)"
    )
  endif()
  add_library(forge_dashboard STATIC src/dashboard/dashboard_server.cpp
                                     src/dashboard/launcher.cpp)
endif()

target_include_directories(
  forge_dashboard PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                         ${CMAKE_CURRENT_SOURCE_DIR}/src
                         ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(forge_dashboard PUBLIC forge_infer forge_httplib)
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge_dashboard PUBLIC USE_CUDA)
endif()

add_executable(
  forge-server src/server/server_main.cpp src/server/server_state.cpp
               src/server/server_routes.cpp)
target_include_directories(
  forge-server
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}/src/server
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(CUDAToolkit_FOUND)
  target_compile_definitions(forge-server PRIVATE USE_CUDA)
endif()
target_link_libraries(
  forge-server
  PRIVATE forge_infer
          forge_model
          forge_ops
          forge_core
          forge_tokenizer
          forge_agent
          forge_dashboard
          forge_vision
          forge_httplib
          $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>)
if(CUDAToolkit_FOUND)
  target_link_libraries(
    forge-server PRIVATE CUDA::cudart $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>)
  set_target_properties(
    forge-server
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON
               CUDA_ARCHITECTURES "${FORGE_CUDA_ARCH}")
endif()

# ---- forge-inspect (GGUF introspection tool) ----
add_library(
  forge_inspect STATIC
  src/inspect/perf/gguf_scanner.cpp src/inspect/color.cpp
  src/inspect/arch_ops.cpp src/inspect/meta_groups.cpp src/inspect/report.cpp
  src/inspect/format.cpp)
target_include_directories(
  forge_inspect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/inspect)
target_link_libraries(forge_inspect PUBLIC forge_model forge_core forge_ops
                                           forge_infer)

add_executable(forge-inspect src/inspect/main.cpp)
target_include_directories(
  forge-inspect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                        ${CMAKE_CURRENT_SOURCE_DIR}/src/inspect)
target_link_libraries(forge-inspect PRIVATE forge_inspect forge_model
                                            forge_core forge_ops forge_infer)

# ---- Python bindings ----
if(FORGE_BUILD_PYTHON)
  pybind11_add_module(
    forge
    src/bindings/forge_py.cpp
    src/bindings/core_types.cpp
    src/bindings/model.cpp
    src/bindings/tokenizer.cpp
    src/bindings/chat_template.cpp
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

  # Copy tools/python/forge_llm.py to build directory alongside forge.so
  add_custom_command(
    TARGET forge
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_SOURCE_DIR}/tools/python/forge_llm.py"
      "$<TARGET_FILE_DIR:forge>/forge_llm.py"
    COMMENT "Copying forge_llm.py to build dir")
endif()

# ---- Tests ----
# Master switch for the test suite. Default OFF: the test executables (and the
# cross-architecture QEMU targets) are only built when explicitly requested.
option(FORGE_BUILD_TESTS "Build forge test executables" OFF)
if(FORGE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
