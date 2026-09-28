#pragma once

// DFlash / DSPark draft provider for speculative decoding (Phase 3 of
// DFLASH_DSPARK_PLAN.md). Implements IDraftProvider by driving a DflashEngine
// (standalone lightweight transformer sharing the target's token_embd/lm_head)
// to propose `n_spec` candidate tokens in a single parallel forward.
//
// A draft round (incremental context KV, vLLM dflash style):
//   1. append the newly confirmed tokens' K/V into the persistent drafter KV
//      cache (O(delta) per round; no full-prefix rebuild, no target re-forward),
//   2. decode([anchor, MASK*n_spec]) non-causally at the anchor's real position
//      -> [n_spec+1, vocab] logits,
//   3. greedy-sample the MASK rows only (row 0 predicts the anchor itself and is
//      never part of the draft; DFlash2 replaces this with a selector-lattice walk).
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
    // DFlash maintains a committed ledger strictly aligned with the target's
    // confirmation boundary (the bookkeeping basis for the incremental context
    // KV); it must receive the confirmation stream even for rounds where it
    // was not selected, otherwise the ledger gets a hole and the next selected
    // round injects the delta at the wrong offset.
    bool is_global() const override { return true; }

    // Target hidden-state handoff: the encoder input is the concatenation of
    // the drafter's configured target_layers, which the target's verify forward
    // already computed. Consuming it removes one full target forward per round.
    const std::vector<int>* capture_layers() const override;
    void set_verify_hiddens(std::shared_ptr<Tensor> hiddens) override;

    // Drop the stash. Called when a token that was NOT produced by the captured
    // verify forward may enter the next delta range.
    void clear_verify_hiddens();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool is_dspark_ = false;
};

}  // namespace forge
