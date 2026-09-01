#include "common.h"

void register_scheduler(py::module_& m) {
    py::class_<SamplerConfig>(m, "SamplerConfig")
        .def(py::init<>())
        .def(py::init<float, int, float, float, int, bool, uint64_t>(),
             py::arg("temperature") = 1.0f, py::arg("top_k") = 0, py::arg("top_p") = 1.0f,
             py::arg("repeat_penalty") = 1.0f, py::arg("repeat_last_n") = 64,
             py::arg("do_sample") = true, py::arg("seed") = 0)
        .def_readwrite("temperature", &SamplerConfig::temperature)
        .def_readwrite("top_k", &SamplerConfig::top_k)
        .def_readwrite("top_p", &SamplerConfig::top_p)
        .def_readwrite("repeat_penalty", &SamplerConfig::repeat_penalty)
        .def_readwrite("repeat_last_n", &SamplerConfig::repeat_last_n)
        .def_readwrite("do_sample", &SamplerConfig::do_sample)
        .def_readwrite("seed", &SamplerConfig::seed)
        .def_readwrite("logit_softcapping", &SamplerConfig::logit_softcapping);

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
        .def_readonly("finish_reason", &GenerateRequest::finish_reason)
        .def_readonly("prefix_len", &GenerateRequest::prefix_len)
        .def_readonly("from_cache", &GenerateRequest::from_cache)
        .def_readonly("prefill_done", &GenerateRequest::prefill_done);

    py::class_<CachedPrompt>(m, "CachedPrompt")
        .def_readonly("tokens", &CachedPrompt::tokens)
        .def_readonly("seq_id", &CachedPrompt::seq_id)
        .def_readonly("valid", &CachedPrompt::valid);

    py::class_<PyRequestScheduler>(m, "RequestScheduler")
        .def(py::init<py::object, int, int>(), py::arg("model"), py::arg("block_size") = 16,
             py::arg("max_num_seqs") = 4)
        .def("submit", &PyRequestScheduler::submit, py::arg("prompt_tokens"),
             py::arg("max_new_tokens") = 256, py::arg("eos_token_id") = -1,
             py::arg("sampler_config") = SamplerConfig{})
        .def("step", &PyRequestScheduler::step)
        .def("get_finished", &PyRequestScheduler::get_finished)
        .def("get_all_requests", &PyRequestScheduler::get_all_requests)
        .def("num_active", &PyRequestScheduler::num_active)
        .def("num_waiting", &PyRequestScheduler::num_waiting)
        .def("has_pending", &PyRequestScheduler::has_pending)
        .def("abort", &PyRequestScheduler::abort, py::arg("request_id"))
        .def("reset", &PyRequestScheduler::reset)
        .def_property_readonly("prefix_cache_hits", &PyRequestScheduler::prefix_cache_hits)
        .def_property_readonly("prefix_cache_misses", &PyRequestScheduler::prefix_cache_misses)
        .def_property("n_batch", &PyRequestScheduler::n_batch, &PyRequestScheduler::set_n_batch)
        .def_property("n_ubatch", &PyRequestScheduler::n_ubatch, &PyRequestScheduler::set_n_ubatch)
        .def_property("n_threads", &PyRequestScheduler::n_threads,
                      &PyRequestScheduler::set_n_threads)
        .def_property("n_threads_batch", &PyRequestScheduler::n_threads_batch,
                      &PyRequestScheduler::set_n_threads_batch)
        .def("memory_stats", &PyRequestScheduler::memory_stats)
        // Roadmap 1.3: chunked prefill / continuous-batching metrics
        .def_property("prefill_chunk_size", &PyRequestScheduler::prefill_chunk_size,
                      &PyRequestScheduler::set_prefill_chunk_size)
        .def_property_readonly("last_step_prefill_tokens",
                               &PyRequestScheduler::last_step_prefill_tokens)
        .def_property_readonly("last_step_decode_tokens",
                               &PyRequestScheduler::last_step_decode_tokens)
        .def_property_readonly("last_step_decode_ratio",
                               &PyRequestScheduler::last_step_decode_ratio)
        .def_property_readonly("max_step_latency_ms", &PyRequestScheduler::max_step_latency_ms)
        .def_property_readonly("interleaved_steps", &PyRequestScheduler::interleaved_steps)
        .def_property_readonly("prefill_chunks_issued", &PyRequestScheduler::prefill_chunks_issued)
        // Roadmap 1.1: KV host offload (swap)
        .def_property("kv_swap_watermark", &PyRequestScheduler::kv_swap_watermark,
                      &PyRequestScheduler::set_kv_swap_watermark)
        .def_property_readonly("swap_events", &PyRequestScheduler::swap_events)
        .def_property_readonly("num_offloaded_pages", &PyRequestScheduler::num_offloaded_pages)
        .def_property_readonly("num_brought_back_pages",
                               &PyRequestScheduler::num_brought_back_pages)
        .def_property_readonly("num_free_pages", &PyRequestScheduler::num_free_pages)
        .def_property_readonly("num_total_pages", &PyRequestScheduler::num_total_pages)
        .def_property_readonly("host_pool_bytes", &PyRequestScheduler::host_pool_bytes)
        // Phase 6: high-level generate() wrapper
        .def("generate", &PyRequestScheduler::generate, py::arg("prompt_tokens"),
             py::arg("generation_config") = GenerationConfig{},
             py::arg("sampler_config") = SamplerConfig{},
             "Submit prompt and run step loop until finished. Returns list of GenerateRequest.");
}
