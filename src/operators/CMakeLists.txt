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
  cpu/conv.cpp
  cpu/gdn.cpp
  cpu/common/dequant.cpp
  cpu/common/quant_tables.cpp
)# 6-C-5.7: interim FP8 scale registry (fp8_scale_registry.cpp) deleted

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
            cuda/cuda_tiled_prefill.cu
            cuda/cuda_paged_kv.cu
            cuda/cuda_logit_softcap.cu
            cuda/cuda_moe.cu
            cuda/cuda_sampler.cu
            cuda/cuda_ssm.cu
            cuda/cuda_mmq.cu
            cuda/kernel_selector.cpp)
  target_compile_definitions(forge_ops PUBLIC USE_CUDA)
  set_target_properties(forge_ops PROPERTIES CUDA_ARCHITECTURES
                                             "${FORGE_CUDA_ARCH}")

  # 6-C-4.1d: W8A8 FP8 E4M3 GEMM production backend (CUTLASS, Ada).
  # The CUTLASS Sm89 device code only instantiates for sm_89+ (guarded inside the
  # TU), so it compiles cleanly in a sm_86;89 build. CUTLASS headers are attached
  # ONLY to this translation unit (not the whole target) to keep the rest of the
  # build CUTLASS-free. When CUTLASS is absent or no Ada arch is selected, a stub
  # mirrors the launcher API so forge_ops always links.
  set(FORGE_FP8_CUTLASS_FOUND FALSE)
  if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
    foreach(_arch ${FORGE_CUDA_ARCH})
      if(_arch EQUAL 89
         OR _arch EQUAL 90
         OR _arch EQUAL 91
         OR _arch EQUAL 92)
        set(FORGE_FP8_CUTLASS_FOUND TRUE)
      endif()
    endforeach()
  endif()
  if(FORGE_FP8_CUTLASS_FOUND)
    target_sources(forge_ops PRIVATE cuda/cuda_fp8_gemm.cu)
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_fp8_gemm.cu
      PROPERTIES INCLUDE_DIRECTORIES
                 "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
  else()
    target_sources(forge_ops PRIVATE cuda/cuda_fp8_gemm_stub.cu)
  endif()

  # 6-C-4.2/4.3: W8A8 FP8 E4M3 GEMV production backend. PURE CUDA (no CUTLASS
  # dependency) -- the sm_89 ada FP8 decode GEMV is a hand-written, memory-bound
  # kernel, not a Tensor-Core op. So it always compiles for the sm_89+ target and
  # is unconditionally linked. The stub (cuda_fp8_gemv_stub.cu) is a structural
  # mirror of the GEMM split and is intentionally NOT linked (linking both would be
  # a duplicate-symbol error); it exists only as a reference/fallback template.
  target_sources(forge_ops PRIVATE cuda/cuda_fp8_gemv.cu)

  # 6-C-5.1a: W8A8 FP8 E4M3 per-tensor quantize PRODUCER (device-side amax -> scale ->
  # encode, no D2H/H2D). Also pure CUDA; separate from the KV-cache FP8 quantize in
  # cuda_quant.cu, which uses its own per-(row,head) scale buffer.
  target_sources(forge_ops PRIVATE cuda/cuda_fp8_quant.cu)
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
         $<$<BOOL:${USE_OPENBLAS}>:${OPENBLAS_LIBRARIES}>)
if(CUDAToolkit_FOUND)
  target_link_libraries(forge_ops PUBLIC $<$<BOOL:${USE_CUBLAS}>:CUDA::cublas>)
endif()
