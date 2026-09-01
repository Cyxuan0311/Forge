#pragma once

// DFlash / DSPark draft provider for speculative decoding (Phase 3 of
// DFLASH_DSPARK_PLAN.md). Implements IDraftProvider by driving a DflashEngine
// (standalone lightweight transformer sharing the target's token_embd/lm_head)
// to propose `n_spec` candidate tokens in a single parallel forward.
//
// A draft round:
//   1. take multi-layer target hiddens for the committed prefix (target must
//      have captures_layer_hiddens() enabled),
//   2. encode -> context feature (cached in the engine),
//   3. precompute_context_kv: inject target prefix K/V into the draft KV cache,
//   4. decode([anchor, MASK*n_spec]) non-causally -> [n_spec+1, vocab] logits,
//   5. greedy-sample n_spec tokens (anchor row -> first draft, MASK rows -> rest).
//
// Verification (greedy argmax + resample-consistency) is reused unchanged from
// the SpeculativeExecutor; DFlash is lossless regardless of any offset.

#include <memory>
#include <string>
#include <vector>

#include "forge/speculative.h"

namespace forge {

class InferenceContext;
class InferenceEngine;

class DFlashDraftProvider : public IDraftProvider {
public:
    // `spec` supplies the drafter GGUF path (draft_model_path), target layer
    // override (draft_target_layers), MASK token id (draft_mask_token_id) and
    // n_spec (draft_n_spec). `target` is the engine whose token_embd/lm_head the
    // drafter borrows; `target_vocab` must equal the drafter's vocabulary.
    DFlashDraftProvider(const SpeculativeConfig& spec, InferenceContext& ctx,
                        InferenceEngine& target, int target_vocab);
    ~DFlashDraftProvider() override;

    bool valid() const;

    void begin(const std::vector<int32_t>& prompt) override;
    std::vector<int32_t> draft(int32_t last_token, int n_draft) override;
    void accept(const std::vector<int32_t>& tokens) override;
    void reset() override;
    const char* name() const override { return is_dspark_ ? "dspark" : "dflash"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool is_dspark_ = false;
};

}  // namespace forge
