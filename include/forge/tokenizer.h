#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace forge {

enum class TokenizerModelType : int {
    SPM,  // SentencePiece (llama, mistral, etc.)
    BPE,  // GPT-2 style BPE (qwen2, gpt2, etc.)
};

struct TokenizerVocab {
    std::vector<std::string> tokens;                       // id -> token string
    std::vector<float> scores;                             // id -> score
    std::vector<int32_t> token_types;                      // id -> token type
    std::unordered_map<std::string, int32_t> token_to_id;  // token string -> id
};

class Tokenizer {
public:
    Tokenizer() = default;

    /// Load tokenizer from a GGUF model file.
    bool load_from_gguf(const std::string& path);

    /// Encode text into token IDs.
    /// @param add_bos If true, prepend BOS token.
    /// @param add_eos If true, append EOS token.
    /// @param add_dummy_prefix If true, add SPM dummy prefix (▁) at the start.
    ///                         Should be false when encoding text that follows a
    ///                         special token or newline (no leading space desired).
    /// @param parse_special If true, scan text for special tokens (e.g. <|im_start|>,
    ///                      [INST]) and encode them as single tokens rather than
    ///                      splitting into subword pieces.
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true, bool add_eos = false,
                                bool add_dummy_prefix = true, bool parse_special = false) const;

    /// Decode token IDs back to text.
    /// @param strip_leading_space If true, remove the leading space from SPM dummy prefix.
    ///                           Should be false when decoding partial sequences (streaming).
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true,
                       bool strip_leading_space = true) const;

    /// Decode a single token ID.
    std::string decode_token(int32_t id) const;

    // Accessors
    int32_t vocab_size() const { return static_cast<int32_t>(vocab_.tokens.size()); }
    int32_t bos_token_id() const { return bos_id_; }
    int32_t eos_token_id() const { return eos_id_; }
    int32_t pad_token_id() const { return pad_id_; }
    int32_t unk_token_id() const { return unk_id_; }
    TokenizerModelType model_type() const { return model_type_; }
    const std::string& chat_template() const { return chat_template_; }

    /// Whether the model metadata requests a BOS token to be prepended on raw
    /// text encoding. Defaults to true (llama.cpp's default for models that omit
    /// the key); set to false when GGUF `tokenizer.ggml.add_bos_token` = false.
    bool add_bos_token() const { return add_bos_token_; }
    /// Whether the model metadata requests an EOS token to be appended on raw
    /// text encoding. Defaults to false.
    bool add_eos_token() const { return add_eos_token_; }

    /// GGUF `tokenizer.ggml.pre` value (e.g. "spark2_5", "llama3", "qwen2", ...).
    /// Empty when the field is absent. Drives pre-tokenization regex selection.
    const std::string& pre_type() const { return pre_type_; }

    /// Look up token ID by string.
    int32_t token_to_id(const std::string& token) const;

    /// Look up token string by ID.
    std::string id_to_token(int32_t id) const;

    /// Look up token score by ID.
    float token_score(int32_t id) const;

    /// Look up token type by ID.
    int32_t token_type(int32_t id) const;

    bool is_loaded() const { return loaded_; }

private:
    // SPM encoding
    std::vector<int32_t> encode_spm(const std::string& text, bool add_bos, bool add_eos,
                                    bool add_dummy_prefix = true) const;
    std::vector<int32_t> encode_spm_greedy(const std::string& spm_text) const;

    // BPE encoding
    std::vector<int32_t> encode_bpe(const std::string& text, bool add_bos, bool add_eos) const;

    // Byte-level BPE pre-tokenization (GPT-2 style)
    std::vector<std::string> bpe_pre_tokenize(const std::string& text) const;

    // spark2_5 pre-tokenization (CJK grouping, 1-3 digit numbers, etc.)
    // Used when tokenizer.ggml.pre == "spark2_5".
    std::vector<std::string> bpe_pre_tokenize_spark25(const std::string& text) const;

    // Apply BPE merges to a word
    std::vector<std::string> bpe_apply(const std::vector<std::string>& word) const;

    // Convert text to byte-level representation (Ġ for space, etc.)
    std::string text_to_byte_level(const std::string& text) const;

    // Convert byte-level representation back to text
    std::string byte_level_to_text(const std::string& text) const;

    // Unicode helpers
    static uint32_t utf8_to_unicode(const std::string& text, size_t& offset);
    static std::string unicode_to_utf8(uint32_t cp);
    static bool is_word_char(uint32_t cp);

    TokenizerVocab vocab_;
    TokenizerModelType model_type_ = TokenizerModelType::SPM;

    // BPE merge rules: pair -> rank (lower = higher priority)
    std::unordered_map<std::string, int> bpe_ranks_;

    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t pad_id_ = -1;
    int32_t unk_id_ = -1;

    // Raw text encoding flags from GGUF metadata (tokenizer.ggml.add_*_token).
    // Defaults mirror llama.cpp: BOS on, EOS off, for models that omit the key.
    bool add_bos_token_ = true;
    bool add_eos_token_ = false;

    std::string chat_template_;
    std::string pre_type_;  // from tokenizer.ggml.pre (empty = unspecified)
    bool loaded_ = false;

    // Byte-level BPE lookup tables
    std::unordered_map<uint8_t, std::string> byte_to_unicode_char_;
    std::unordered_map<std::string, uint8_t> unicode_char_to_byte_;

    // Special token cache for parse_special (lazy-initialized)
    mutable std::vector<std::pair<std::string, int32_t>> special_tokens_cache_;
    mutable bool special_tokens_cached_ = false;
    void ensure_special_tokens_cached() const;
};

}  // namespace forge
