add_library(
  forge_ops STATIC
  dispatch.cpp
  norm.cpp
  activation.cpp
  elementwise.cpp
  matmul.cpp
  embedding.cpp
  rope.cpp
  attention.cpp)
if(CUDAToolkit_FOUND)
  target_sources(
    forge_ops
    PRIVATE cuda/cuda_rms_norm.cu
            cuda/cuda_activation.cu
            cuda/cuda_elementwise.cu
            cuda/cuda_rope.cu
            cuda/cuda_flash_attn.cu
            cuda/cuda_embedding.cu
            cuda/cuda_gemv.cu
            cuda/cuda_gemv_instances.cu
            cuda/cuda_fused.cu
            cuda/cuda_quant.cu
            cuda/cuda_gemm.cu)
  target_include_directories(forge_ops PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(forge_ops PUBLIC USE_CUDA)
  target_compile_definitions(forge_ops
                             PRIVATE FORGE_USE_CUBLAS=$<BOOL:${USE_CUBLAS}>)
  target_link_libraries(forge_ops PUBLIC $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>)
endif()
target_compile_definitions(forge_ops
                           PRIVATE FORGE_USE_OPENBLAS=$<BOOL:${USE_OPENBLAS}>)
target_include_directories(forge_ops PRIVATE ${OPENBLAS_INCLUDE_DIRS})
target_link_libraries(
  forge_ops PUBLIC forge_core $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>
                   $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
