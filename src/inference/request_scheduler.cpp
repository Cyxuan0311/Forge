#include "forge/request_scheduler.h"
#include "forge/engine.h"
#include "forge/engines/llama_engine.h"
#include "forge/logger.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace forge {

RequestScheduler::RequestScheduler(Model& model, int block_size, int max_num_seqs)
    : model_(model), ctx_(model), sampler_(SamplerConfig{}), max_num_seqs_(max_num_seqs) {
    const auto& cfg = model_.config();
    DeviceType dev = model_.device();
    paged_cache_.init(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim,
                      cfg.max_seq_len, block_size, max_num_seqs, dev);

    auto engine = EngineRegistry::instance().create(cfg.arch_type, model_, ctx_);
    if (engine) {
        ctx_.set_engine(std::move(engine));
    }
}

int RequestScheduler::submit(const std::vector<int32_t>& prompt_tokens,
                              int max_new_tokens,
                              int eos_token_id,
                              const SamplerConfig& sampler_cfg,
                              GenerateRequest::Callback callback) {
    std::lock_guard<std::mutex> lock(mutex_);

    int req_id = next_request_id_++;
    GenerateRequest req;
    req.request_id = req_id;
    req.prompt_tokens = prompt_tokens;
    req.max_new_tokens = max_new_tokens;
    req.eos_token_id = eos_token_id;
    req.sampler_config = sampler_cfg;
    req.callback = callback;
    req.status = RequestStatus::Waiting;
    req.current_pos = 0;

    requests_[req_id] = std::move(req);
    waiting_queue_.push(req_id);

    LOG_DEBUG("RequestScheduler: submitted request " + std::to_string(req_id) +
              " with " + std::to_string(prompt_tokens.size()) + " prompt tokens");

    return req_id;
}

bool RequestScheduler::step() {
    std::lock_guard<std::mutex> lock(mutex_);

    schedule();

    if (active_ids_.empty()) return false;

    std::vector<int> still_active;
    for (int rid : active_ids_) {
        auto it = requests_.find(rid);
        if (it == requests_.end()) continue;

        auto& req = it->second;
        bool ok = true;

        if (req.status == RequestStatus::Prefilling) {
            ok = prefill_request(req);
        } else if (req.status == RequestStatus::Decoding) {
            ok = decode_step(req);
        }

        if (!ok) {
            req.status = RequestStatus::Failed;
            req.finish_reason = "error";
            paged_cache_.release_seq(rid);
            continue;
        }

        if (req.status == RequestStatus::Finished || req.status == RequestStatus::Failed) {
            paged_cache_.release_seq(rid);
            continue;
        }

        still_active.push_back(rid);
    }

    active_ids_ = std::move(still_active);
    return true;
}

void RequestScheduler::schedule() {
    while (!waiting_queue_.empty() &&
           static_cast<int>(active_ids_.size()) < max_num_seqs_ &&
           paged_cache_.num_free_blocks() > 0) {
        int rid = waiting_queue_.front();
        waiting_queue_.pop();

        auto it = requests_.find(rid);
        if (it == requests_.end()) continue;

        if (paged_cache_.allocate_seq(rid) < 0) {
            waiting_queue_.push(rid);
            break;
        }

        it->second.status = RequestStatus::Prefilling;
        active_ids_.push_back(rid);

        LOG_DEBUG("RequestScheduler: scheduled request " + std::to_string(rid));
    }
}

bool RequestScheduler::prefill_request(GenerateRequest& req) {
    const auto& cfg = model_.config();
    DeviceType dev = ctx_.device();
    auto* engine = ctx_.engine();
    if (!engine) return false;

    int prompt_len = static_cast<int>(req.prompt_tokens.size());

    auto input_ids = std::make_shared<Tensor>(DataType::INT32,
        std::vector<int64_t>{prompt_len}, DeviceType::CPU);
    std::memcpy(input_ids->data(), req.prompt_tokens.data(), prompt_len * sizeof(int32_t));

    if (dev == DeviceType::CUDA) {
        input_ids->to_device(DeviceType::CUDA);
    }

    auto logits = engine->forward(input_ids, 0);

    auto last_logits = std::make_shared<Tensor>(logits->slice(0, prompt_len - 1, prompt_len));

    sampler_.set_config(req.sampler_config);
    int token_id = sampler_.sample(last_logits, prompt_len - 1);

    req.output_tokens.push_back(token_id);
    req.num_generated = 1;
    req.current_pos = prompt_len;
    req.status = RequestStatus::Decoding;

    if (req.callback) {
        req.callback(req.request_id, token_id, 0, req.status);
    }

    if (req.eos_token_id >= 0 && token_id == req.eos_token_id) {
        req.status = RequestStatus::Finished;
        req.finish_reason = "eos";
    }

    return true;
}

bool RequestScheduler::decode_step(GenerateRequest& req) {
    DeviceType dev = ctx_.device();
    auto* engine = ctx_.engine();
    if (!engine) return false;

    int32_t last_token = req.output_tokens.back();

    auto input_ids = std::make_shared<Tensor>(DataType::INT32,
        std::vector<int64_t>{1}, DeviceType::CPU);
    *static_cast<int32_t*>(input_ids->data()) = last_token;

    if (dev == DeviceType::CUDA) {
        input_ids->to_device(DeviceType::CUDA);
    }

    auto logits = engine->forward(input_ids, req.current_pos);

    sampler_.set_config(req.sampler_config);
    int token_id = sampler_.sample(logits, req.current_pos);

    req.output_tokens.push_back(token_id);
    req.num_generated++;
    req.current_pos++;

    if (req.callback) {
        req.callback(req.request_id, token_id, req.num_generated - 1, req.status);
    }

    if (req.eos_token_id >= 0 && token_id == req.eos_token_id) {
        req.status = RequestStatus::Finished;
        req.finish_reason = "eos";
    } else if (req.num_generated >= req.max_new_tokens) {
        req.status = RequestStatus::Finished;
        req.finish_reason = "length";
    }

    return true;
}

std::vector<GenerateRequest> RequestScheduler::get_finished() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<GenerateRequest> finished;
    std::vector<int> to_remove;

    for (auto& [rid, req] : requests_) {
        if (req.status == RequestStatus::Finished || req.status == RequestStatus::Failed) {
            finished.push_back(std::move(req));
            to_remove.push_back(rid);
        }
    }

    for (int rid : to_remove) {
        requests_.erase(rid);
    }

    return finished;
}

std::vector<GenerateRequest> RequestScheduler::get_all_requests() const {
    std::vector<GenerateRequest> result;
    for (const auto& [rid, req] : requests_) {
        result.push_back(req);
    }
    return result;
}

int RequestScheduler::num_active() const {
    return static_cast<int>(active_ids_.size());
}

int RequestScheduler::num_waiting() const {
    return static_cast<int>(waiting_queue_.size());
}

bool RequestScheduler::has_pending() const {
    return !active_ids_.empty() || !waiting_queue_.empty();
}

void RequestScheduler::abort(int request_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = requests_.find(request_id);
    if (it == requests_.end()) return;

    it->second.status = RequestStatus::Failed;
    it->second.finish_reason = "aborted";
    paged_cache_.release_seq(request_id);

    active_ids_.erase(
        std::remove(active_ids_.begin(), active_ids_.end(), request_id),
        active_ids_.end());
}

void RequestScheduler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [rid, req] : requests_) {
        if (req.status != RequestStatus::Finished && req.status != RequestStatus::Failed) {
            paged_cache_.release_seq(rid);
        }
    }

    requests_.clear();
    active_ids_.clear();
    while (!waiting_queue_.empty()) waiting_queue_.pop();
}

} // namespace forge
