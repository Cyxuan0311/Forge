#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <memory>

namespace forge {

enum class TokenizerModelType : int {
    SPM,    // SentencePiece (llama, mistral, etc.)
    BPE,    // GPT-2 style BPE (qwen2, gpt2, etc.)
};

struct TokenizerVocab {
    std::vector<std::string> tokens;       // id -> token string
    std::vector<float> scores;             // id -> score
    std::vector<int32_t> token_types;      // id -> token type
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
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true,
                                bool add_eos = false, bool add_dummy_prefix = true) const;

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

    std::string chat_template_;
    bool loaded_ = false;

    // Byte-level BPE lookup tables
    std::unordered_map<uint8_t, std::string> byte_to_unicode_char_;
    std::unordered_map<std::string, uint8_t> unicode_char_to_byte_;
};

} // namespace forge
