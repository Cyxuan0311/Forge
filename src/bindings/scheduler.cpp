#include "common.h"

void register_scheduler(py::module_& m) {
    py::class_<SamplerConfig>(m, "SamplerConfig")
        .def(py::init<>())
        .def(py::init<float, int, float, float, bool, uint64_t>(),
             py::arg("temperature") = 1.0f,
             py::arg("top_k") = 0,
             py::arg("top_p") = 1.0f,
             py::arg("repeat_penalty") = 1.0f,
             py::arg("do_sample") = true,
             py::arg("seed") = 0)
        .def_readwrite("temperature", &SamplerConfig::temperature)
        .def_readwrite("top_k", &SamplerConfig::top_k)
        .def_readwrite("top_p", &SamplerConfig::top_p)
        .def_readwrite("repeat_penalty", &SamplerConfig::repeat_penalty)
        .def_readwrite("do_sample", &SamplerConfig::do_sample)
        .def_readwrite("seed", &SamplerConfig::seed);

    py::enum_<RequestStatus>(m, "RequestStatus")
        .value("Waiting", RequestStatus::Waiting)
        .value("Prefilling", RequestStatus::Prefilling)
        .value("Decoding", RequestStatus::Decoding)
        .value("Finished", RequestStatus::Finished)
        .value("Failed", RequestStatus::Failed);

    py::class_<GenerateRequest>(m, "GenerateRequest")
        .def_readonly("request_id", &GenerateRequest::request_id)
        .def_readonly("status", &GenerateRequest::status)
        .def_readonly("output_tokens", &GenerateRequest::output_tokens)
        .def_readonly("num_generated", &GenerateRequest::num_generated)
        .def_readonly("finish_reason", &GenerateRequest::finish_reason);

    py::class_<PyRequestScheduler>(m, "RequestScheduler")
        .def(py::init<PyModel&, int, int>(),
             py::arg("model"),
             py::arg("block_size") = 16,
             py::arg("max_num_seqs") = 4)
        .def("submit", &PyRequestScheduler::submit,
             py::arg("prompt_tokens"),
             py::arg("max_new_tokens") = 256,
             py::arg("eos_token_id") = -1,
             py::arg("sampler_config") = SamplerConfig{})
        .def("step", &PyRequestScheduler::step)
        .def("get_finished", &PyRequestScheduler::get_finished)
        .def("num_active", &PyRequestScheduler::num_active)
        .def("num_waiting", &PyRequestScheduler::num_waiting)
        .def("has_pending", &PyRequestScheduler::has_pending)
        .def("abort", &PyRequestScheduler::abort, py::arg("request_id"))
        .def("reset", &PyRequestScheduler::reset);
}
