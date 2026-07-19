#include "forge/request_scheduler.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "forge/engine.h"
#include "forge/kv_cache.h"
#include "forge/logger.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

// ---- Prompt hash (FNV-1a) ----

size_t RequestScheduler::hash_prompt(const std::vector<int32_t>& tokens) {
    // FNV-1a 64-bit hash
    size_t h = 14695981039346656037ULL;
    for (int32_t t : tokens) {
        h ^= static_cast<size_t>(t);
        h *= 1099511628211ULL;
    }
    return h;
}

// ---- Prefix cache: try to find a cached prompt ----

bool RequestScheduler::try_prefix_cache(GenerateRequest& req) {
    if (static_cast<int>(req.prompt_tokens.size()) < MIN_CACHE_PROMPT_LEN)
        return false;

    auto* engine = ctx_.engine();
    if (!engine)
        return false;

    KVCache* kv = engine->kv_cache();
    if (!kv)
        return false;

    size_t h = hash_prompt(req.prompt_tokens);
    auto it = prompt_cache_.find(h);

    if (it != prompt_cache_.end() && it->second.valid) {
        auto& cached = it->second;

        // Verify the cached seq_id still has cells in the KV cache
        if (cached.seq_id >= 0 && kv->seq_filled(0, cached.seq_id) >=
            static_cast<int>(cached.tokens.size())) {
            // Cache hit: zero-copy share prefix via seq_cp
            kv->seq_cp(cached.seq_id, req.request_id, 0,
                       static_cast<int64_t>(cached.tokens.size()));

            req.prefix_len = static_cast<int>(cached.tokens.size());
            req.prefix_seq_id = cached.seq_id;
            req.from_cache = true;
            req.current_pos = req.prefix_len;

            prefix_cache_hits_++;
            LOG_DEBUG("Prefix cache HIT: req=" + std::to_string(req.request_id) +
                      " prefix_len=" + std::to_string(req.prefix_len) +
                      " from seq=" + std::to_string(cached.seq_id));
            return true;
        } else {
            // Cached seq_id's KV has been evicted — invalidate
            cached.valid = false;
            LOG_DEBUG("Prefix cache EVICTED: seq=" + std::to_string(cached.seq_id));
        }
    }

    // Cache miss: register this prompt for future requests
    // The request will process the full prompt, and we'll register after forward completes
    req.prefix_len = 0;
    req.prefix_seq_id = -1;
    req.from_cache = false;

    prefix_cache_misses_++;
    return false;
}

// ---- Prefix cache: evict entry for a given seq_id ----

void RequestScheduler::evict_prefix_cache(int seq_id) {
    for (auto& [h, cached] : prompt_cache_) {
        if (cached.seq_id == seq_id) {
            cached.valid = false;
            LOG_DEBUG("Prefix cache evicted: seq=" + std::to_string(seq_id));
            break;
        }
    }
}

// ---- Prefix cache: preserve when the owning sequence finishes ----

void RequestScheduler::preserve_prefix_cache(int seq_id, int prompt_len) {
    auto* engine = ctx_.engine();
    if (!engine)
        return;

    KVCache* kv = engine->kv_cache();
    if (!kv)
        return;

    // Find the cache entry owned by this seq_id
    for (auto& [h, cached] : prompt_cache_) {
        if (cached.seq_id == seq_id && cached.valid) {
            // Allocate a new persistent seq_id to hold the prefix KV cells
            int new_seq_id = next_request_id_++;
            kv->seq_cp(seq_id, new_seq_id, 0, prompt_len);
            kv->seq_rm(seq_id, 0, prompt_len);

            // Update cache entry to point to the new persistent seq_id
            cached.seq_id = new_seq_id;
            LOG_DEBUG("Prefix cache PRESERVED: old_seq=" + std::to_string(seq_id) +
                      " → new_seq=" + std::to_string(new_seq_id) +
                      " prefix_len=" + std::to_string(prompt_len));
            break;
        }
    }
}

// ---- Constructor ----

RequestScheduler::RequestScheduler(Model& model, int block_size, int max_num_seqs)
    : model_(model), ctx_(model), sampler_(SamplerConfig{}), max_num_seqs_(max_num_seqs) {
    const auto& cfg = model_.config();
    DeviceType dev = model_.device();
    paged_cache_.init(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len, block_size,
                      max_num_seqs, dev);

    auto engine = EngineRegistry::instance().create(cfg.arch_type, model_, ctx_);
    if (engine) {
        ctx_.set_engine(std::move(engine));
    }
}

int RequestScheduler::submit(const std::vector<int32_t>& prompt_tokens, int max_new_tokens,
                             int eos_token_id, const SamplerConfig& sampler_cfg,
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

    LOG_DEBUG("RequestScheduler: submitted request " + std::to_string(req_id) + " with " +
              std::to_string(prompt_tokens.size()) + " prompt tokens");

    return req_id;
}

bool RequestScheduler::step() {
    std::lock_guard<std::mutex> lock(mutex_);

    schedule();

    if (active_ids_.empty())
        return false;

    auto* engine = ctx_.engine();
    if (!engine)
        return false;

    KVCache* kv = engine->kv_cache();

    // Build InferenceBatch from active requests
    InferenceBatch batch;
    std::vector<int> batch_rid;  // maps batch item index → request ID

    for (int rid : active_ids_) {
        auto it = requests_.find(rid);
        if (it == requests_.end())
            continue;

        auto& req = it->second;
        if (req.status != RequestStatus::Prefilling && req.status != RequestStatus::Decoding)
            continue;

        InferenceBatchItem item;
        item.seq_id = rid;
        item.logits = true;

        if (req.status == RequestStatus::Prefilling) {
            // Check prefix cache for this request
            if (!req.from_cache) {
                try_prefix_cache(req);
            }

            if (req.from_cache && req.prefix_len > 0) {
                // Cache hit: only process tokens after the cached prefix
                int suffix_start = req.prefix_len;
                int suffix_len = static_cast<int>(req.prompt_tokens.size()) - suffix_start;
                if (suffix_len > 0) {
                    item.tokens.assign(req.prompt_tokens.begin() + suffix_start,
                                       req.prompt_tokens.end());
                } else {
                    // Entire prompt is cached — need at least one token for sampling
                    // Use the last token of the prefix
                    item.tokens = {req.prompt_tokens.back()};
                    suffix_start = static_cast<int>(req.prompt_tokens.size()) - 1;
                    suffix_len = 1;
                }
                item.start_pos = suffix_start;
                item.positions.resize(item.tokens.size());
                for (size_t j = 0; j < item.tokens.size(); j++)
                    item.positions[j] = static_cast<int64_t>(suffix_start + j);
            } else {
                // Cache miss: process full prompt
                item.tokens = req.prompt_tokens;
                item.start_pos = 0;
                item.positions.resize(req.prompt_tokens.size());
                for (size_t j = 0; j < req.prompt_tokens.size(); j++)
                    item.positions[j] = static_cast<int64_t>(j);
            }
        } else {  // Decoding
            item.tokens = {req.output_tokens.back()};
            item.start_pos = req.current_pos;
            item.positions = {req.current_pos};
        }

        batch.items.push_back(std::move(item));
        batch_rid.push_back(rid);
    }

    if (batch.empty())
        return false;

    // Batch forward pass
    TensorPtr logits_batch;
    try {
        logits_batch = engine->forward_batch(batch);
    } catch (const std::exception& e) {
        LOG_ERROR("forward_batch failed: " + std::string(e.what()));
        for (int rid : active_ids_) {
            auto it = requests_.find(rid);
            if (it != requests_.end()) {
                it->second.status = RequestStatus::Failed;
                it->second.finish_reason = "error";
                paged_cache_.release_seq(rid);
            }
        }
        active_ids_.clear();
        return false;
    }

    // Sample per-sequence tokens and update state
    std::vector<int> still_active;

    for (int i = 0; i < batch.size(); i++) {
        int rid = batch_rid[i];
        auto it = requests_.find(rid);
        if (it == requests_.end())
            continue;

        auto& req = it->second;

        // Extract this sequence's logits from batch result
        TensorPtr seq_logits;
        if (logits_batch && logits_batch->ndim() >= 2 &&
            static_cast<int>(logits_batch->shape()[0]) == batch.size()) {
            seq_logits = std::make_shared<Tensor>(logits_batch->slice(0, i, i + 1));
        } else if (logits_batch) {
            if (i != batch.size() - 1) {
                still_active.push_back(rid);
                continue;
            }
            seq_logits = logits_batch;
        } else {
            req.status = RequestStatus::Failed;
            req.finish_reason = "error";
            paged_cache_.release_seq(rid);
            continue;
        }

        // Sample with per-request config
        sampler_.set_config(req.sampler_config);
        int64_t sample_pos = batch.items[i].start_pos +
                             static_cast<int64_t>(batch.items[i].tokens.size()) - 1;
        int token_id = sampler_.sample(seq_logits, sample_pos);

        // Update request state
        req.output_tokens.push_back(token_id);
        req.num_generated++;

        if (req.status == RequestStatus::Prefilling) {
            req.current_pos = static_cast<int>(req.prompt_tokens.size());
            req.status = RequestStatus::Decoding;

            // Register this prompt in the prefix cache (if eligible)
            if (!req.from_cache &&
                static_cast<int>(req.prompt_tokens.size()) >= MIN_CACHE_PROMPT_LEN && kv) {
                size_t h = hash_prompt(req.prompt_tokens);
                // Only register if not already cached
                if (prompt_cache_.find(h) == prompt_cache_.end()) {
                    prompt_cache_[h] = {req.prompt_tokens, rid, true};
                    LOG_DEBUG("Prefix cache REGISTER: req=" + std::to_string(rid) +
                              " tokens=" + std::to_string(req.prompt_tokens.size()));
                }
            }
        } else {
            req.current_pos++;
        }

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

        if (req.status == RequestStatus::Finished || req.status == RequestStatus::Failed) {
            // Handle prefix cache: if this request owns a cached prefix, preserve it
            if (kv && req.from_cache && req.prefix_seq_id >= 0) {
                // The request used a cached prefix — just clean up its own cells
                // The prefix cells are still owned by prefix_seq_id (and possibly other sequences)
                // Remove only this request's cells beyond the prefix
                if (req.current_pos > req.prefix_len) {
                    kv->seq_rm(rid, req.prefix_len, req.current_pos);
                }
                // Remove this seq_id from prefix cells
                kv->seq_rm(rid, 0, req.prefix_len);
            } else if (kv) {
                // This request owns a prefix cache entry — preserve it
                preserve_prefix_cache(rid, static_cast<int>(req.prompt_tokens.size()));
            }

            paged_cache_.release_seq(rid);
            continue;
        }

        still_active.push_back(rid);
    }

    active_ids_ = std::move(still_active);
    return true;
}

void RequestScheduler::schedule() {
    while (!waiting_queue_.empty() && static_cast<int>(active_ids_.size()) < max_num_seqs_ &&
           paged_cache_.num_free_blocks() > 0) {
        int rid = waiting_queue_.front();
        waiting_queue_.pop();

        auto it = requests_.find(rid);
        if (it == requests_.end())
            continue;

        if (paged_cache_.allocate_seq(rid) < 0) {
            waiting_queue_.push(rid);
            break;
        }

        it->second.status = RequestStatus::Prefilling;
        active_ids_.push_back(rid);

        LOG_DEBUG("RequestScheduler: scheduled request " + std::to_string(rid));
    }
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
    if (it == requests_.end())
        return;

    it->second.status = RequestStatus::Failed;
    it->second.finish_reason = "aborted";
    paged_cache_.release_seq(request_id);

    active_ids_.erase(std::remove(active_ids_.begin(), active_ids_.end(), request_id),
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
    while (!waiting_queue_.empty())
        waiting_queue_.pop();
    prompt_cache_.clear();
    prefix_cache_hits_ = 0;
    prefix_cache_misses_ = 0;
}

}  // namespace forge
