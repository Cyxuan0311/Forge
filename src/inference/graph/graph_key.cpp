#include "forge/inference/graph/graph_key.h"

namespace forge {

std::string GraphKey::to_string() const {
    std::string s = plan_id;
    s += "/seq=" + std::to_string(seq_len);
    s += "/gpu=" + std::to_string(gpu_layers);
    s += "/dev=";
    s += (input_device == DeviceType::CUDA) ? "cuda" : "cpu";
    s += is_prefill ? "/prefill" : "/decode";
    s += "/bs=" + std::to_string(batch_size);
    s += "/nseq=" + std::to_string(n_sequences);
    return s;
}

GraphKey GraphKey::from_request(const ForwardRequest& req, const std::string& plan_id,
                                int gpu_layers, DeviceType input_device) {
    GraphKey key;
    key.seq_len = req.n_tokens;
    key.gpu_layers = gpu_layers;
    key.plan_id = plan_id;
    key.input_device = input_device;
    key.is_prefill = req.is_prefill;
    key.batch_size = 1;
    key.n_sequences = 1;
    return key;
}

}  // namespace forge
