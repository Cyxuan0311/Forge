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
#ifdef _OPENMP
#    include <omp.h>
#endif

PYBIND11_MODULE(forge, m) {
    m.doc() = "Forge: Lightweight LLM inference engine";
    m.attr("__version__") = "0.5.0";

    register_core_types(m);
    register_model(m);
    register_tokenizer(m);
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
        },
        py::arg("n"), "Set number of CPU threads for inference (OpenMP)");
}
