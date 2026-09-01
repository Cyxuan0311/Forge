#include "forge/request_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "forge/engine.h"
#include "forge/engines/transformer_engine.h"
#include "forge/kv_cache.h"
#include "forge/kv_memory.h"
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

    KVMemory* memory = engine->kv_memory();
    if (!memory)
        return false;

    size_t h = hash_prompt(req.prompt_tokens);
    auto it = prompt_cache_.find(h);

    if (it != prompt_cache_.end() && it->second.valid) {
        auto& cached = it->second;

        // Verify the cached seq_id still has cells in the KV cache
        if (cached.seq_id >= 0 && memory->storage().seq_filled(0, cached.seq_id) >=
                                      static_cast<int>(cached.tokens.size())) {
            // Cache hit: zero-copy share prefix via seq_share
            memory->seq_share(cached.seq_id, req.request_id, 0,
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

    KVMemory* memory = engine->kv_memory();
    if (!memory)
        return;

    // Find the cache entry owned by this seq_id
    for (auto& [h, cached] : prompt_cache_) {
        if (cached.seq_id == seq_id && cached.valid) {
            // Allocate a new persistent seq_id to hold the prefix KV cells
            int new_seq_id = next_request_id_++;
            memory->seq_share(seq_id, new_seq_id, 0, prompt_len);
            memory->seq_remove(seq_id, 0, prompt_len);

            // Update cache entry to point to the new persistent seq_id
            cached.seq_id = new_seq_id;
            LOG_DEBUG("Prefix cache PRESERVED: old_seq=" + std::to_string(seq_id) + " → new_seq=" +
                      std::to_string(new_seq_id) + " prefix_len=" + std::to_string(prompt_len));
            break;
        }
    }
}

// ---- Page-level prefix cache (paged mode) ----

bool RequestScheduler::try_prefix_cache_paged(GenerateRequest& req) {
    auto* engine = ctx_.engine();
    if (!engine)
        return false;

    KVMemory* memory = engine->kv_memory();
    if (!memory || !memory->is_paged())
        return false;

    int prefix_len = prefix_cache_.try_lookup(req.prompt_tokens, req.request_id, memory->storage());

    if (prefix_len > 0) {
        req.prefix_len = prefix_len;
        req.prefix_seq_id = -1;  // not used in paged mode; PrefixCache tracks internally
        req.from_cache = true;
        req.current_pos = prefix_len;
        memory->record_prefix_hit(prefix_len);
        return true;
    }

    req.prefix_len = 0;
    req.prefix_seq_id = -1;
    req.from_cache = false;
    return false;
}

void RequestScheduler::finish_request_paged(GenerateRequest& req) {
    auto* engine = ctx_.engine();
    if (!engine)
        return;

    KVMemory* memory = engine->kv_memory();
    if (!memory || !memory->is_paged())
        return;

    if (req.from_cache) {
        // Request used a cached prefix: release its prefix reference.
        // Suffix pages (beyond prefix) are released by release_seq_kv below.
        prefix_cache_.release_prefix(req.request_id, memory->storage());
    } else if (static_cast<int>(req.prompt_tokens.size()) >= MIN_CACHE_PROMPT_LEN) {
        // Request owns a new prefix: register it in the page-level cache.
        // register_prefix transfers prefix page ownership to a cache seq_id
        // and removes them from the request's page table.
        prefix_cache_.register_prefix(req.prompt_tokens, req.request_id, memory->storage());
    }

    // Evict LRU entries to keep cache bounded
    prefix_cache_.evict_lru(PrefixCache::DEFAULT_MAX_ENTRIES, memory->storage());
}

// ---- Constructor ----

RequestScheduler::RequestScheduler(Model& model, int block_size, int max_num_seqs)
    : model_(model), ctx_(model), sampler_(SamplerConfig{}), max_num_seqs_(max_num_seqs) {
    (void)block_size;  // block_size no longer needed with engine KVCache

    // Check for paged storage mode (same env var as PyModel::create_context).
    const char* storage_mode_env = std::getenv("FORGE_KV_STORAGE_MODE");
    if (storage_mode_env && std::string(storage_mode_env) == "paged") {
        ctx_.params_mut().kv_storage_mode = KVStorageMode::Paged;
        paged_mode_ = true;
    }

    auto engine = EngineRegistry::instance().create(model_.config().arch_type, model_, ctx_);
    if (engine) {
        // Match the engine's layer placement to the model's device. Without this
        // the engine's default gpu_layers_ = -1 maps every layer to CUDA while
        // the weights still live on CPU (CPU-loaded model), and the first
        // forward hangs on mixed-device ops (norm with a CPU weight pointer).
        if (auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine.get())) {
            tfm_eng->set_gpu_layers(ctx_.params().device == DeviceType::CUDA ? -1 : 0);
        }
        ctx_.set_engine(std::move(engine));
    }
}

// ---- KV cache cleanup ----

void RequestScheduler::release_seq_kv(int seq_id, int prompt_len) {
    auto* engine = ctx_.engine();
    if (!engine)
        return;
    KVMemory* memory = engine->kv_memory();
    if (!memory)
        return;

    // Remove the sequence from all its KV cells
    if (prompt_len > 0) {
        memory->seq_remove(seq_id, 0, prompt_len);
    } else {
        // No position info: release all cells
        memory->release_sequence(seq_id);
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

    // Roadmap 1.3: step latency is the metric a long prefill used to blow up.
    const auto step_start = std::chrono::steady_clock::now();

    schedule();

    // Roadmap 1.1: under KV memory pressure, evict low-value active sequences'
    // pages to the host pool before building the batch. Evicted sequences sit
    // out this step; their pages are brought back before their next forward.
    const std::vector<int> suspended = try_swap_out();

    if (active_ids_.empty())
        return false;

    auto* engine = ctx_.engine();
    if (!engine)
        return false;

    KVMemory* memory = engine->kv_memory();

    // Build InferenceBatch from active requests
    InferenceBatch batch;
    std::vector<int> batch_rid;  // maps batch item index → request ID
    // Roadmap 1.3: per-item flag telling the sampling loop whether this item
    // produced logits. Intermediate prefill chunks do not, so they must not be
    // sampled (the engine zero-fills their logits row).
    std::vector<char> sample_flags;
    int step_prefill_tokens = 0;
    int step_decode_tokens = 0;

    // Effective prefill chunk size: <0 disables chunking (legacy whole-prompt
    // behaviour), 0 follows n_ubatch, >0 is an explicit override.
    const int configured_chunk = prefill_chunk_size_;
    const bool chunking_enabled = configured_chunk >= 0;
    int chunk = configured_chunk;
    if (chunk == 0)
        chunk = ctx_.params().n_ubatch;
    if (chunk <= 0)
        chunk = 256;

    for (int rid : active_ids_) {
        auto it = requests_.find(rid);
        if (it == requests_.end())
            continue;

        auto& req = it->second;
        if (req.status != RequestStatus::Prefilling && req.status != RequestStatus::Decoding)
            continue;

        // Roadmap 1.1: sequences whose pages were just swapped out skip this
        // step — fetching them back would defeat the purpose of the eviction.
        if (std::find(suspended.begin(), suspended.end(), rid) != suspended.end())
            continue;

        // Bring any previously-evicted pages of this sequence back onto the
        // device before the forward pass reads them.
        if (memory && memory->is_paged())
            memory->storage().bring_back_seq(rid);

        InferenceBatchItem item;
        item.seq_id = rid;
        item.logits = true;

        if (req.status == RequestStatus::Prefilling) {
            // Prefix cache lookup happens once, on the first chunk.
            if (req.prefill_done == 0 && !req.from_cache) {
                if (paged_mode_)
                    try_prefix_cache_paged(req);
                else
                    try_prefix_cache(req);
            }

            const int prompt_len = static_cast<int>(req.prompt_tokens.size());
            if (prompt_len <= 0) {
                // Nothing to prefill; drop the request from this step.
                req.status = RequestStatus::Failed;
                req.finish_reason = "empty_prompt";
                release_seq_kv(rid);
                continue;
            }

            // Tokens already covered by a prefix-cache hit need no forward pass.
            int begin = req.prefill_done;
            if (req.from_cache && req.prefix_len > begin)
                begin = req.prefix_len;
            if (begin > prompt_len)
                begin = prompt_len;

            int len = prompt_len - begin;
            if (chunking_enabled)
                len = std::min(len, chunk);

            if (len <= 0) {
                // Every token is already covered by the cached prefix — still
                // need one token to sample the first output.
                begin = prompt_len - 1;
                len = 1;
            }

            const bool last_chunk = (begin + len >= prompt_len);

            item.tokens.assign(req.prompt_tokens.begin() + begin,
                               req.prompt_tokens.begin() + begin + len);
            item.start_pos = begin;
            item.positions.resize(len);
            for (int j = 0; j < len; j++)
                item.positions[j] = static_cast<int64_t>(begin + j);

            // Only the final chunk yields a token; intermediate chunks just
            // advance the prompt cursor.
            item.logits = last_chunk;
            sample_flags.push_back(last_chunk ? 1 : 0);
            step_prefill_tokens += len;
            prefill_chunks_issued_++;
        } else {  // Decoding
            item.tokens = {req.output_tokens.back()};
            item.start_pos = req.current_pos;
            item.positions = {req.current_pos};
            item.logits = true;
            sample_flags.push_back(1);
            step_decode_tokens += 1;
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
                release_seq_kv(rid);
            }
        }
        active_ids_.clear();
        return false;
    }

    // Sample per-sequence tokens and update state
    std::vector<int> still_active;

    // Bring logits to CPU if needed (already on CPU from forward_batch, but just in case)
    TensorPtr logits_cpu = logits_batch;
    if (logits_batch && logits_batch->device() == DeviceType::CUDA) {
        logits_cpu =
            std::make_shared<Tensor>(DataType::FP32, logits_batch->shape(), DeviceType::CPU);
        logits_cpu->copy_from(*logits_batch);
    }

    for (int i = 0; i < batch.size(); i++) {
        int rid = batch_rid[i];
        auto it = requests_.find(rid);
        if (it == requests_.end())
            continue;

        auto& req = it->second;

        // Roadmap 1.3: every chunk advanced the prompt cursor, including the
        // intermediate ones that carry no logits.
        if (req.status == RequestStatus::Prefilling)
            req.prefill_done += static_cast<int>(batch.items[i].tokens.size());

        // Intermediate prefill chunks produced no logits (the engine zero-fills
        // their row), so they must not be sampled — they just stay active.
        if (!sample_flags[i]) {
            still_active.push_back(rid);
            continue;
        }

        // Extract this sequence's logits from batch result [n_seq, vocab_size]
        TensorPtr seq_logits;
        if (logits_cpu && logits_cpu->ndim() >= 2 &&
            static_cast<int>(logits_cpu->shape()[0]) == batch.size()) {
            int vocab_size = static_cast<int>(logits_cpu->shape()[1]);
            seq_logits = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{1, vocab_size}, DeviceType::CPU);
            const float* src = static_cast<const float*>(logits_cpu->data()) + i * vocab_size;
            std::memcpy(seq_logits->data(), src, vocab_size * sizeof(float));
        } else if (logits_cpu) {
            // Fallback: only last sequence's logits available
            if (i != batch.size() - 1) {
                still_active.push_back(rid);
                continue;
            }
            seq_logits = logits_cpu;
        } else {
            req.status = RequestStatus::Failed;
            req.finish_reason = "error";
            release_seq_kv(rid);
            continue;
        }

        // Sample with per-request config
        sampler_.set_config(req.sampler_config);
        int64_t sample_pos =
            batch.items[i].start_pos + static_cast<int64_t>(batch.items[i].tokens.size()) - 1;
        int token_id = sampler_.sample(seq_logits, sample_pos);

        // Update request state
        req.output_tokens.push_back(token_id);
        req.num_generated++;

        if (req.status == RequestStatus::Prefilling) {
            req.current_pos = static_cast<int>(req.prompt_tokens.size());
            req.status = RequestStatus::Decoding;

            // Register this prompt in the legacy prefix cache (contiguous mode only).
            // Paged mode registers on request finish via finish_request_paged().
            if (!paged_mode_ && !req.from_cache &&
                static_cast<int>(req.prompt_tokens.size()) >= MIN_CACHE_PROMPT_LEN && memory) {
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
            // Handle prefix cache cleanup
            if (paged_mode_) {
                // Paged mode: page-level prefix cache handles register/release
                finish_request_paged(req);
            } else if (memory && req.from_cache && req.prefix_seq_id >= 0) {
                // Contiguous mode: request used a cached prefix — clean up its own cells
                // The prefix cells are still owned by prefix_seq_id (and possibly other sequences)
                // Remove only this request's cells beyond the prefix
                if (req.current_pos > req.prefix_len) {
                    memory->seq_remove(rid, req.prefix_len, req.current_pos);
                }
                // Remove this seq_id from prefix cells
                memory->seq_remove(rid, 0, req.prefix_len);
            } else if (memory) {
                // Contiguous mode: this request owns a prefix cache entry — preserve it
                preserve_prefix_cache(rid, static_cast<int>(req.prompt_tokens.size()));
            }

            release_seq_kv(rid);
            continue;
        }

        still_active.push_back(rid);
    }

    active_ids_ = std::move(still_active);

    // Roadmap 1.3: publish this step's interleaving metrics. A step that ran
    // both prefill and decode work proves decode was not starved.
    last_step_prefill_tokens_ = step_prefill_tokens;
    last_step_decode_tokens_ = step_decode_tokens;
    if (step_prefill_tokens > 0 && step_decode_tokens > 0)
        interleaved_steps_++;

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start)
            .count();
    if (elapsed_ms > max_step_latency_ms_)
        max_step_latency_ms_ = elapsed_ms;

    // Roadmap 1.1: bring_back happens mid-step (before the forward pass), so
    // refresh the swap counters after the step completes.
    update_swap_stats();

    return true;
}

void RequestScheduler::schedule() {
    auto* engine = ctx_.engine();
    KVMemory* memory = engine ? engine->kv_memory() : nullptr;

    while (!waiting_queue_.empty() && static_cast<int>(active_ids_.size()) < max_num_seqs_) {
        // Check KV cache capacity: need at least some free slots.
        // If KV cache is not yet initialized (max_seq_len == 0), allow admission
        // since it will be initialized on first forward.
        bool has_capacity = true;
        if (memory && memory->max_seq_len() > 0) {
            int free_slots = memory->num_free_slots();
            if (free_slots <= 0)
                has_capacity = false;
        }

        if (!has_capacity)
            break;

        int rid = waiting_queue_.front();
        waiting_queue_.pop();

        auto it = requests_.find(rid);
        if (it == requests_.end())
            continue;

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

// =========================================================================
// Roadmap 1.1: KV host offload (swap)
// =========================================================================

std::vector<int> RequestScheduler::try_swap_out() {
    std::vector<int> suspended;

    auto* engine = ctx_.engine();
    if (!engine)
        return suspended;
    KVMemory* memory = engine->kv_memory();
    if (!memory || !memory->is_paged())
        return suspended;
    auto& storage = memory->storage();

    const int total_pages = storage.total_page_capacity();
    if (total_pages <= 0)
        return suspended;

    // Keep at least `kv_swap_watermark_` of the page capacity free on the
    // device. Watermark <= 0 disables proactive swap-out.
    const int min_free = static_cast<int>(kv_swap_watermark_ * total_pages);
    if (min_free <= 0)
        return suspended;

    // Evicting a page releases its device storage, so the "free" measure that
    // matters is the device footprint (referenced, non-evicted pages) against
    // total capacity.
    const int in_use = storage.num_device_pages_in_use();
    if (total_pages - in_use >= min_free)
        return suspended;

    // Victims are the active requests holding the fewest pages (cheapest to
    // move). They keep their KV state on the host pool and resume next step.
    std::vector<int> candidates;
    for (int rid : active_ids_) {
        auto it = requests_.find(rid);
        if (it == requests_.end())
            continue;
        const auto& req = it->second;
        if (req.status != RequestStatus::Prefilling && req.status != RequestStatus::Decoding)
            continue;
        candidates.push_back(rid);
    }
    std::sort(candidates.begin(), candidates.end(),
              [&](int a, int b) { return storage.seq_num_pages(a) < storage.seq_num_pages(b); });

    for (int rid : candidates) {
        if (total_pages - storage.num_device_pages_in_use() >= min_free)
            break;
        if (storage.offload_seq(rid)) {
            suspended.push_back(rid);
            swap_events_++;
            LOG_DEBUG("RequestScheduler: swapped out request " + std::to_string(rid));
        }
    }

    update_swap_stats();
    return suspended;
}

void RequestScheduler::update_swap_stats() {
    auto* engine = ctx_.engine();
    if (!engine)
        return;
    KVMemory* memory = engine->kv_memory();
    if (!memory || !memory->is_paged())
        return;
    auto& storage = memory->storage();
    num_offloaded_pages_ = storage.num_offloaded_pages();
    num_brought_back_pages_ = storage.num_brought_back_pages();
}

int RequestScheduler::num_free_pages() const {
    const auto* engine = ctx_.engine();
    if (!engine)
        return 0;
    const KVMemory* memory = engine->kv_memory();
    if (!memory || !memory->is_paged())
        return 0;
    return memory->storage().num_free_pages();
}

int RequestScheduler::num_total_pages() const {
    const auto* engine = ctx_.engine();
    if (!engine)
        return 0;
    const KVMemory* memory = engine->kv_memory();
    if (!memory || !memory->is_paged())
        return 0;
    return memory->storage().total_page_capacity();
}

size_t RequestScheduler::host_pool_bytes() const {
    const auto* engine = ctx_.engine();
    if (!engine)
        return 0;
    const KVMemory* memory = engine->kv_memory();
    if (!memory || !memory->is_paged())
        return 0;
    return memory->storage().host_pool_bytes();
}

void RequestScheduler::abort(int request_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = requests_.find(request_id);
    if (it == requests_.end())
        return;

    it->second.status = RequestStatus::Failed;
    it->second.finish_reason = "aborted";
    release_seq_kv(request_id);

    active_ids_.erase(std::remove(active_ids_.begin(), active_ids_.end(), request_id),
                      active_ids_.end());
}

void RequestScheduler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [rid, req] : requests_) {
        if (req.status != RequestStatus::Finished && req.status != RequestStatus::Failed) {
            release_seq_kv(rid);
        }
    }

    requests_.clear();
    active_ids_.clear();
    while (!waiting_queue_.empty())
        waiting_queue_.pop();
    prompt_cache_.clear();
    prefix_cache_.clear();
    prefix_cache_hits_ = 0;
    prefix_cache_misses_ = 0;

    // Roadmap 1.3: clear the interleaving metrics too.
    last_step_prefill_tokens_ = 0;
    last_step_decode_tokens_ = 0;
    max_step_latency_ms_ = 0.0;
    interleaved_steps_ = 0;
    prefill_chunks_issued_ = 0;

    // Roadmap 1.1: clear the swap counters.
    swap_events_ = 0;
    num_offloaded_pages_ = 0;
    num_brought_back_pages_ = 0;
}

}  // namespace forge
