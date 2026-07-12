#include "common.h"

void register_backend(py::module_& m) {
    py::enum_<BackendCapability>(m, "BackendCapability")
        .value("None", BackendCapability::None)
        .value("FP32", BackendCapability::FP32)
        .value("FP16", BackendCapability::FP16)
        .value("INT8", BackendCapability::INT8)
        .value("Quantized", BackendCapability::Quantized)
        .value("UnifiedMemory", BackendCapability::UnifiedMemory)
        .value("StreamAsync", BackendCapability::StreamAsync)
        .def("__or__", [](BackendCapability a, BackendCapability b) { return a | b; })
        .def("__and__", [](BackendCapability a, BackendCapability b) { return a & b; });

    py::class_<BackendInfo>(m, "BackendInfo")
        .def_readonly("name", &BackendInfo::name)
        .def_readonly("device_type", &BackendInfo::device_type)
        .def_readonly("available", &BackendInfo::available);

    py::class_<Backend, std::shared_ptr<Backend>>(m, "Backend")
        .def("device_type", &Backend::device_type)
        .def("name", &Backend::name)
        .def("capabilities", &Backend::capabilities)
        .def("supports", &Backend::supports, py::arg("cap"))
        .def("device_memory_total", &Backend::device_memory_total)
        .def("device_memory_free", &Backend::device_memory_free)
        .def("device_id", &Backend::device_id);

    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def_readonly("name", &DeviceInfo::name)
        .def_readonly("device_type", &DeviceInfo::type)
        .def_readonly("device_id", &DeviceInfo::device_id)
        .def_readonly("memory_total", &DeviceInfo::memory_total)
        .def_readonly("memory_free", &DeviceInfo::memory_free);

    py::class_<BackendManager>(m, "BackendManager")
        .def_static("instance", &BackendManager::instance, py::return_value_policy::reference)
        .def("available_backends", &BackendManager::available_backends)
        .def("backend_info_list", &BackendManager::backend_info_list)
        .def("available_devices", &BackendManager::available_devices)
        .def("has_backend", &BackendManager::has_backend, py::arg("name"))
        .def("get_backend",
             (std::shared_ptr<Backend>(BackendManager::*)(const std::string&, int)) &
                 BackendManager::get_backend,
             py::arg("name"), py::arg("device_id") = 0)
        .def("has_cuda", &BackendManager::has_cuda)
        .def("cuda_device_count", &BackendManager::cuda_device_count);
}
