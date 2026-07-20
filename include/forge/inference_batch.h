#pragma once

#include <cstdint>
#include <vector>

namespace forge {

// A single sequence within an inference batch.
// Each item represents one sequence's tokens, positions, and metadata.
struct InferenceBatchItem {
    std::vector<int32_t> tokens;     // token IDs for this sequence
    std::vector<int64_t> positions;  // per-token positions (for MRoPE / position-dependent ops)
    int64_t start_pos = 0;          // starting position in KV cache
    int seq_id = 0;                 // sequence ID for KV cache isolation
    bool logits = true;             // whether this sequence needs logits output
};

// A batch of sequences for parallel inference.
// Follows the "flat [total_tokens, dim]" approach like llama.cpp's llama_batch,
// but groups by sequence for clarity.
struct InferenceBatch {
    std::vector<InferenceBatchItem> items;
    bool all_logits = false;  // true = 返回所有位置的 logits [total_tokens, vocab]

    // Total number of tokens across all sequences
    int n_tokens() const {
        int total = 0;
        for (const auto& item : items)
            total += static_cast<int>(item.tokens.size());
        return total;
    }

    // True if any sequence has more than 1 token (prefill phase)
    bool is_prefill() const {
        for (const auto& item : items)
            if (static_cast<int>(item.tokens.size()) > 1)
                return true;
        return false;
    }

    // True if all sequences have exactly 1 token (decode phase)
    bool is_decode() const {
        for (const auto& item : items)
            if (static_cast<int>(item.tokens.size()) != 1)
                return false;
        return !items.empty();
    }

    // Offsets[i] = starting token index for sequence i in the flat token array
    std::vector<int> token_offsets() const {
        std::vector<int> offsets;
        offsets.reserve(items.size());
        int offset = 0;
        for (const auto& item : items) {
            offsets.push_back(offset);
            offset += static_cast<int>(item.tokens.size());
        }
        return offsets;
    }

    bool empty() const { return items.empty(); }
    int size() const { return static_cast<int>(items.size()); }
};

// Split a batch into micro-batches, each with at most n_ubatch tokens.
// Strategy: per-sequence chunking — if a single sequence has more tokens than
// n_ubatch, it is split into multiple micro-batches. This avoids cross-sequence
// masking complexity and ensures each micro-batch is self-contained.
//
// Example: batch has 2 sequences with 500 and 10 tokens, n_ubatch=256:
//   micro-batch 0: seq_0 tokens [0, 256)
//   micro-batch 1: seq_0 tokens [256, 500)
//   micro-batch 2: seq_1 tokens [0, 10)
//
// If total tokens <= n_ubatch, returns a single micro-batch (the original batch).
std::vector<InferenceBatch> split_batch(const InferenceBatch& batch, int n_ubatch);

}  // namespace forge
