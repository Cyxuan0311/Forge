#include "common.h"

void register_tokenizer(py::module_& m) {
    py::enum_<TokenizerModelType>(m, "TokenizerModelType")
        .value("SPM", TokenizerModelType::SPM)
        .value("BPE", TokenizerModelType::BPE);

    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("load_from_gguf", &Tokenizer::load_from_gguf, py::arg("path"))
        .def(
            "encode",
            [](const Tokenizer& tok, const std::string& text, bool add_bos, bool add_eos,
               bool add_dummy_prefix) {
                return tok.encode(text, add_bos, add_eos, add_dummy_prefix);
            },
            py::arg("text"), py::arg("add_bos") = true, py::arg("add_eos") = false,
            py::arg("add_dummy_prefix") = true)
        .def(
            "decode",
            [](const Tokenizer& tok, const std::vector<int32_t>& ids, bool skip_special,
               bool strip_leading_space) {
                return tok.decode(ids, skip_special, strip_leading_space);
            },
            py::arg("ids"), py::arg("skip_special") = true, py::arg("strip_leading_space") = true)
        .def("decode_token", &Tokenizer::decode_token, py::arg("id"))
        .def("token_to_id", &Tokenizer::token_to_id, py::arg("token"))
        .def("id_to_token", &Tokenizer::id_to_token, py::arg("id"))
        .def("token_score", &Tokenizer::token_score, py::arg("id"))
        .def("token_type", &Tokenizer::token_type, py::arg("id"))
        .def_property_readonly("vocab_size", &Tokenizer::vocab_size)
        .def_property_readonly("bos_token_id", &Tokenizer::bos_token_id)
        .def_property_readonly("eos_token_id", &Tokenizer::eos_token_id)
        .def_property_readonly("pad_token_id", &Tokenizer::pad_token_id)
        .def_property_readonly("unk_token_id", &Tokenizer::unk_token_id)
        .def_property_readonly("model_type", &Tokenizer::model_type)
        .def_property_readonly("chat_template", &Tokenizer::chat_template)
        .def_property_readonly("is_loaded", &Tokenizer::is_loaded);
}
