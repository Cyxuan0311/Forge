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
        })
        .def("set_trace_enabled", &PerfProfiler::set_trace_enabled)
        .def("trace_enabled", &PerfProfiler::trace_enabled)
        .def("trace_to_json", [](PerfProfiler& p) { return p.trace_to_json(); })
        .def("trace_overflow", [](PerfProfiler& p) { return p.trace_overflow(); })
        .def("trace_size", [](PerfProfiler& p) { return p.trace_size(); });

    // Convenience module-level functions
    m.def("profiler_enable", []() {
#if FORGE_PROFILING == 1
        PerfProfiler::instance().enable();
#else
        fprintf(stderr,
                "[Warning] profiler_enable() called but FORGE_PROFILING=0 at build time.\n"
                "         C++ operator-level profiling is disabled (all PERF_SCOPE are no-ops).\n"
                "         Rebuild with: cmake .. -DFORGE_PROFILING=ON\n");
#endif
    });
    m.def("profiler_disable", []() { PerfProfiler::instance().disable(); });
    m.def("profiler_reset", []() { PerfProfiler::instance().reset(); });
    m.def("profiler_enabled", []() { return PerfProfiler::instance().enabled(); });
    m.def("profiler_set_cuda_events",
          [](bool use_cuda) { PerfProfiler::instance().set_use_cuda_events(use_cuda); });
    m.def("_profiler_record_test", [](const std::string& name, double ms) {
        PerfEventId id = PerfProfiler::instance().intern_name(name.c_str());
        PerfProfiler::instance().record(id, ms);
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
    m.def("profiler_set_trace_enabled", [](bool v) { PerfProfiler::instance().set_trace_enabled(v); });
    m.def("profiler_trace_enabled", []() { return PerfProfiler::instance().trace_enabled(); });
    m.def("profiler_trace_to_json", []() { return PerfProfiler::instance().trace_to_json(); });
    m.def("profiler_trace_overflow", []() { return PerfProfiler::instance().trace_overflow(); });
    m.def("profiler_trace_size", []() { return PerfProfiler::instance().trace_size(); });
}
