#include "common.h"

void register_logger(py::module_& m) {
    py::class_<PyLogger>(m, "Logger")
        .def_static("set_level", &PyLogger::set_level, py::arg("level"))
        .def_static("level", &PyLogger::level)
        .def_static("set_python_sink", &PyLogger::set_python_sink, py::arg("callback"))
        .def_static("reset_sink", &PyLogger::reset_sink);
}
