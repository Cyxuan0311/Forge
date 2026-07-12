# detect_cuda_architecture(<output_var>)
#
# Compiles and runs a tiny CUDA program to detect the current GPU's
# compute capability (major.minor, e.g. "89" for RTX 4090).
# Sets <output_var> on success; leaves it unchanged on failure.
#
function(detect_cuda_architecture OUTPUT_VAR)
    if(CMAKE_CROSSCOMPILING)
        return()
    endif()

    set(_src "${CMAKE_BINARY_DIR}/detect_cuda_arch.cu")
    set(_bin "${CMAKE_BINARY_DIR}/detect_cuda_arch${CMAKE_EXECUTABLE_SUFFIX}")

    file(WRITE "${_src}"
"#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf(\"%d%d\", prop.major, prop.minor);
    return 0;
}"
    )

    execute_process(
        COMMAND "${CMAKE_CUDA_COMPILER}" -o "${_bin}" "${_src}"
        RESULT_VARIABLE _compile_ret
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _compile_ret EQUAL 0)
        return()
    endif()

    execute_process(
        COMMAND "${_bin}"
        OUTPUT_VARIABLE _arch
        RESULT_VARIABLE _run_ret
    )
    if(_run_ret EQUAL 0 AND _arch)
        string(STRIP "${_arch}" _arch)
        message(STATUS "Detected CUDA architecture: ${_arch}")
        set("${OUTPUT_VAR}" "${_arch}" PARENT_SCOPE)
    endif()
endfunction()
