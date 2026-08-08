# forge_ops — CPU/CUDA operator kernels (static library) This file is the sole
# owner of forge_ops. Previously the definition was inlined in the root
# CMakeLists.txt; now it lives here.

add_library(
  forge_ops STATIC
  cpu/norm.cpp
  cpu/activation.cpp
  cpu/matmul.cpp
  cpu/quant_traits.cpp
  cpu/embedding.cpp
  cpu/rope.cpp
  cpu/elementwise.cpp
  cpu/attention.cpp
  cpu/common/dequant.cpp
  cpu/common/quant_tables.cpp)

# Architecture-specific sources and compile definitions (see
# cpu_arch_split_plan.md §3.2)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)")
  target_sources(forge_ops PRIVATE cpu/arch/x86/repack.cpp)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)")
  # arm64 sources added in Phase 2
else()

endif()

if(CUDAToolkit_FOUND)
  target_sources(
    forge_ops
    PRIVATE cuda/cuda_rms_norm.cu
            cuda/matmul_cuda.cpp
            cuda/cuda_activation.cu
            cuda/cuda_elementwise.cu
            cuda/cuda_rope.cu
            cuda/cuda_flash_attn.cu
            cuda/cuda_embedding.cu
            cuda/cuda_gemv.cu
            cuda/cuda_gemv_instances.cu
            cuda/cuda_fused.cu
            cuda/cuda_quant.cu
            cuda/cuda_gemm.cu
            cuda/cuda_fused_attn.cu
            cuda/cuda_paged_kv.cu
            cuda/cuda_logit_softcap.cu
            cuda/cuda_moe.cu
            cuda/cuda_sampler.cu
            cuda/cuda_ssm.cu
            cuda/cuda_mmq.cu)
  target_compile_definitions(forge_ops PUBLIC USE_CUDA)
  set_target_properties(forge_ops PROPERTIES CUDA_ARCHITECTURES
                                             "${FORGE_CUDA_ARCH}")
endif()

target_include_directories(forge_ops PUBLIC ${CMAKE_SOURCE_DIR}/include
                                            ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(forge_ops
                           PRIVATE FORGE_USE_CUBLAS=$<BOOL:${USE_CUBLAS}>)
target_compile_definitions(forge_ops
                           PRIVATE FORGE_USE_OPENBLAS=$<BOOL:${USE_OPENBLAS}>)
target_include_directories(forge_ops PRIVATE ${OPENBLAS_INCLUDE_DIRS})
target_link_libraries(
  forge_ops
  PUBLIC forge_core forge_cpu_features
         $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>
         $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
if(CUDAToolkit_FOUND)
  target_link_libraries(forge_ops PUBLIC $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>)
endif()
