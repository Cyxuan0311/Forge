#include "forge/attention_mask.h"

#include <cmath>
#include <cstring>

#include "forge/inference_batch.h"
#include "forge/kv_cache.h"
#include "forge/logger.h"

namespace forge {

static constexpr float MASK_NEG_INF = -1e30f;

TensorPtr AttentionMaskBuilder::build_kq_mask(const InferenceBatch& ubatch,
                                               const KVCache& kv_cache,
                                               int layer,
                                               int num_heads,
                                               int head_dim,
                                               bool causal,
                                               DeviceType dev) {
    int total_q = ubatch.n_tokens();
    int kv_len = kv_cache.filled(layer);

    if (total_q == 0 || kv_len == 0)
        return nullptr;

    // For single-sequence batches with causal attention, no mask needed
    if (ubatch.size() <= 1 && causal)
        return nullptr;

    auto offsets = ubatch.token_offsets();

    auto mask = std::make_shared<Tensor>(DataType::FP32,
                                          std::vector<int64_t>{total_q, kv_len},
                                          DeviceType::CPU);
    float* mask_data = static_cast<float*>(mask->data());

    // Initialize all to -inf (mask out everything)
    for (int i = 0; i < total_q * kv_len; ++i)
        mask_data[i] = MASK_NEG_INF;

    // For each query sequence, enable attention to its own KV positions
    for (int si = 0; si < ubatch.size(); si++) {
        const auto& item = ubatch.items[si];
        int seq_id = item.seq_id;
        int q_start = offsets[si];
        int q_len = static_cast<int>(item.tokens.size());

        // Iterate over all KV positions
        const auto& cells = kv_cache.layers()[layer].cells;
        for (int kv_pos = 0; kv_pos < kv_len && kv_pos < static_cast<int>(cells.size()); ++kv_pos) {
            const auto& cell = cells[kv_pos];

            // Skip free cells (already -inf)
            if (cell.is_free())
                continue;

            // Skip cells not owned by this sequence
            if (!cell.has_seq(seq_id))
                continue;

            // This KV position belongs to this sequence — enable attention
            // from query tokens that are at or after this KV position (causal)
            for (int qi = 0; qi < q_len; qi++) {
                int64_t query_pos = item.start_pos + qi;
                if (causal && static_cast<int64_t>(cell.pos) > query_pos) {
                    // Future KV position — masked by causal constraint
                    continue;
                }
                mask_data[(q_start + qi) * kv_len + kv_pos] = 0.0f;
            }
        }
    }

    if (dev == DeviceType::CUDA) {
        auto mask_cuda = std::make_shared<Tensor>(DataType::FP32, mask->shape(), DeviceType::CUDA);
        mask_cuda->copy_from(*mask);
        return mask_cuda;
    }

    return mask;
}

TensorPtr AttentionMaskBuilder::build_decode_mask(const InferenceBatch& ubatch,
                                                   const KVCache& kv_cache,
                                                   int layer,
                                                   DeviceType dev) {
    int n_seqs = ubatch.size();
    int kv_len = kv_cache.filled(layer);

    if (n_seqs == 0 || kv_len == 0)
        return nullptr;

    // For single-sequence decode, no mask needed
    if (n_seqs <= 1)
        return nullptr;

    // Each sequence contributes 1 query token.
    // Mask shape: [n_seqs, kv_len]
    auto mask = std::make_shared<Tensor>(DataType::FP32,
                                          std::vector<int64_t>{n_seqs, kv_len},
                                          DeviceType::CPU);
    float* mask_data = static_cast<float*>(mask->data());

    // Initialize all to -inf
    for (int i = 0; i < n_seqs * kv_len; ++i)
        mask_data[i] = MASK_NEG_INF;

    const auto& cells = kv_cache.layers()[layer].cells;

    for (int si = 0; si < n_seqs; si++) {
        int seq_id = ubatch.items[si].seq_id;

        for (int kv_pos = 0; kv_pos < kv_len && kv_pos < static_cast<int>(cells.size()); ++kv_pos) {
            const auto& cell = cells[kv_pos];
            if (cell.is_free() || !cell.has_seq(seq_id))
                continue;

            // Decode: query_pos is always the latest position, which is >=
            // all previously computed KV positions for this sequence.
            // So causal constraint is automatically satisfied.
            mask_data[si * kv_len + kv_pos] = 0.0f;
        }
    }

    if (dev == DeviceType::CUDA) {
        auto mask_cuda = std::make_shared<Tensor>(DataType::FP32, mask->shape(), DeviceType::CUDA);
        mask_cuda->copy_from(*mask);
        return mask_cuda;
    }

    return mask;
}

}  // namespace forge
