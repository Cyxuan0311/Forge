#include "common.h"
#include "forge/chat_template.h"

void register_chat_template(py::module_& m) {
    py::class_<ChatTemplateMessage>(m, "ChatTemplateMessage")
        .def(py::init<>())
        .def_readwrite("role", &ChatTemplateMessage::role)
        .def_readwrite("content", &ChatTemplateMessage::content);

    py::class_<ChatTemplateInput>(m, "ChatTemplateInput")
        .def(py::init<>())
        .def_readwrite("messages", &ChatTemplateInput::messages)
        .def_readwrite("add_generation_prompt", &ChatTemplateInput::add_generation_prompt)
        .def_readwrite("system_prompt", &ChatTemplateInput::system_prompt);

    py::class_<ChatTemplateEngine>(m, "ChatTemplateEngine")
        .def(py::init<>())
        .def_static("from_tokenizer", &ChatTemplateEngine::from_tokenizer, py::arg("tokenizer"))
        .def("uses_jinja", &ChatTemplateEngine::uses_jinja)
        .def("supports_system_role", &ChatTemplateEngine::supports_system_role)
        .def("render", &ChatTemplateEngine::render, py::arg("input"))
        .def("apply", &ChatTemplateEngine::apply, py::arg("input"));
}