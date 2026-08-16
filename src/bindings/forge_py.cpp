/**
 * Forge Python Bindings - Main Entry Point
 *
 * This file defines the PYBIND11_MODULE entry point and delegates
 * registration to individual module files:
 *   - core_types.cpp: Enums, Tensor, ModelConfig, VisionConfig
 *   - model.cpp: Model, InferenceContext
 *   - tokenizer.cpp: Tokenizer
 *   - multimodal.cpp: MultimodalModel
 *   - scheduler.cpp: RequestScheduler, SamplerConfig
 *   - backend.cpp: Backend, BackendManager
 *   - logger.cpp: Logger
 *   - profiler.cpp: PerfProfiler
 */

#include "common.h"
#include "forge/threads.h"
#ifdef _OPENMP
#    include <omp.h>
#endif

// Thread count requested via forge.set_num_threads(). create_context() applies
// it to the engine ContextParams so the engine's omp_set_num_threads() (which
// overrides the global) uses the requested value instead of the default of 4.
int forge_global_cpu_threads = 4;

PYBIND11_MODULE(forge, m) {
    m.doc() = "Forge: Lightweight LLM inference engine";
    m.attr("__version__") = "0.7.0";

    register_core_types(m);
    register_model(m);
    register_tokenizer(m);
    register_chat_template(m);
    register_multimodal(m);
    register_scheduler(m);
    register_backend(m);
    register_logger(m);
    register_profiler(m);

    m.def(
        "set_num_threads",
        [](int n) {
#ifdef _OPENMP
            if (n < 1)
                n = 1;
            omp_set_num_threads(n);
#endif
            forge_global_cpu_threads = n;
        },
        py::arg("n"), "Set number of CPU threads for inference (OpenMP)");

    m.def(
        "auto_detect_threads",
        [](int hidden, int intermediate) {
            int n = (hidden > 0 && intermediate > 0)
                        ? forge::detect_best_cpu_threads(hidden, intermediate)
                        : forge::detect_best_cpu_threads();
#ifdef _OPENMP
            omp_set_num_threads(n);
#endif
            forge_global_cpu_threads = n;
            return n;
        },
        py::arg("hidden") = 0, py::arg("intermediate") = 0,
        "Probe the fused Q2_K GEMV and pick the best CPU thread count for "
        "decode. hidden/intermediate size the synthetic probe weights to the "
        "model (0 = Llama-8B-class defaults). Applies it for subsequent "
        "create_context() calls.");
}
