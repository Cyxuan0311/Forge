#include "common.h"

void register_profiler(py::module_& m) {
    py::class_<PerfProfiler>(m, "_PerfProfiler")
        .def_static("instance", &PerfProfiler::instance, py::return_value_policy::reference)
        .def("enable", &PerfProfiler::enable)
        .def("disable", &PerfProfiler::disable)
        .def("enabled", &PerfProfiler::enabled)
        .def("reset", &PerfProfiler::reset)
        .def("print_summary", &PerfProfiler::print_summary)
        .def("summary", [](PerfProfiler& p) {
            py::dict result;
            for (const auto& [name, rec] : p.summary()) {
                py::dict entry;
                entry["count"] = rec.count;
                entry["total_ms"] = rec.total_ms;
                entry["avg_ms"] = rec.count > 0 ? rec.total_ms / rec.count : 0.0;
                entry["min_ms"] = rec.min_ms;
                entry["max_ms"] = rec.max_ms;
                entry["last_ms"] = rec.last_ms;
                result[name.c_str()] = entry;
            }
            return result;
        });

    // Convenience module-level functions
    m.def("profiler_enable", []() { PerfProfiler::instance().enable(); });
    m.def("profiler_disable", []() { PerfProfiler::instance().disable(); });
    m.def("profiler_reset", []() { PerfProfiler::instance().reset(); });
    m.def("profiler_enabled", []() { return PerfProfiler::instance().enabled(); });
    m.def("profiler_set_cuda_events", [](bool use_cuda) {
        PerfProfiler::instance().set_use_cuda_events(use_cuda);
    });
    m.def("profiler_summary", []() {
        // Must flush deferred CUDA events first, same as print_summary()
#ifdef USE_CUDA
        PerfProfiler::instance().flush_deferred();
#endif
        py::dict result;
        auto data = PerfProfiler::instance().summary();
        for (const auto& [name, rec] : data) {
            py::dict entry;
            entry["count"] = rec.count;
            entry["total_ms"] = rec.total_ms;
            entry["avg_ms"] = rec.count > 0 ? rec.total_ms / rec.count : 0.0;
            entry["min_ms"] = rec.min_ms;
            entry["max_ms"] = rec.max_ms;
            entry["last_ms"] = rec.last_ms;
            result[name.c_str()] = entry;
        }
        return result;
    });
    m.def("profiler_print", []() { PerfProfiler::instance().print_summary(); });
}
