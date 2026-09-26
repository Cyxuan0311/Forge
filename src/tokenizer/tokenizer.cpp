#include "forge/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <map>
#include <queue>
#include <regex>
#include <sstream>

#include "forge/logger.h"

namespace forge {

// ===== Unicode helpers =====

uint32_t Tokenizer::utf8_to_unicode(const std::string& text, size_t& offset) {
    uint8_t b0 = static_cast<uint8_t>(text[offset]);

    if (b0 < 0x80) {
        offset += 1;
        return b0;
    } else if ((b0 & 0xE0) == 0xC0) {
        if (offset + 1 >= text.size()) {
            offset = text.size();
            return 0xFFFD;
        }
        uint32_t cp = (b0 & 0x1F) << 6;
        cp |= (static_cast<uint8_t>(text[offset + 1]) & 0x3F);
        offset += 2;
        return cp;
    } else if ((b0 & 0xF0) == 0xE0) {
        if (offset + 2 >= text.size()) {
            offset = text.size();
            return 0xFFFD;
        }
        uint32_t cp = (b0 & 0x0F) << 12;
        cp |= (static_cast<uint8_t>(text[offset + 1]) & 0x3F) << 6;
        cp |= (static_cast<uint8_t>(text[offset + 2]) & 0x3F);
        offset += 3;
        return cp;
    } else if ((b0 & 0xF8) == 0xF0) {
        if (offset + 3 >= text.size()) {
            offset = text.size();
            return 0xFFFD;
        }
        uint32_t cp = (b0 & 0x07) << 18;
        cp |= (static_cast<uint8_t>(text[offset + 1]) & 0x3F) << 12;
        cp |= (static_cast<uint8_t>(text[offset + 2]) & 0x3F) << 6;
        cp |= (static_cast<uint8_t>(text[offset + 3]) & 0x3F);
        offset += 4;
        return cp;
    }

    offset += 1;
    return 0xFFFD;
}

std::string Tokenizer::unicode_to_utf8(uint32_t cp) {
    std::string result;
    if (cp < 0x80) {
        result += static_cast<char>(cp);
    } else if (cp < 0x800) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}

bool Tokenizer::is_word_char(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||    // CJK Unified Ideographs
           (cp >= 0x3400 && cp <= 0x4DBF) ||    // CJK Extension A
           (cp >= 0x20000 && cp <= 0x2A6DF) ||  // CJK Extension B
           (cp >= 0x3000 && cp <= 0x303F) ||    // CJK Symbols
           (cp >= 0x3040 && cp <= 0x309F) ||    // Hiragana
           (cp >= 0x30A0 && cp <= 0x30FF) ||    // Katakana
           (cp >= 0xAC00 && cp <= 0xD7AF);      // Hangul
}

// ===== GGUF metadata reader =====

struct GgufMetaReader {
    const uint8_t* data;
    size_t size;
    size_t offset;

    bool check(size_t n) const { return offset + n <= size; }

    std::string read_str() {
        if (!check(8))
            return "";
        uint64_t len;
        std::memcpy(&len, data + offset, 8);
        if (len > size - offset - 8)
            return "";
        offset += 8;
        std::string s(reinterpret_cast<const char*>(data + offset), len);
        offset += len;
        return s;
    }

    uint32_t read_u32() {
        if (!check(4))
            return 0;
        uint32_t v;
        std::memcpy(&v, data + offset, 4);
        offset += 4;
        return v;
    }

    uint8_t read_u8() {
        if (!check(1))
            return 0;
        uint8_t v = data[offset];
        offset += 1;
        return v;
    }

    int32_t read_i32() {
        if (!check(4))
            return 0;
        int32_t v;
        std::memcpy(&v, data + offset, 4);
        offset += 4;
        return v;
    }

    uint64_t read_u64() {
        if (!check(8))
            return 0;
        uint64_t v;
        std::memcpy(&v, data + offset, 8);
        offset += 8;
        return v;
    }

    float read_f32() {
        if (!check(4))
            return 0.0f;
        float v;
        std::memcpy(&v, data + offset, 4);
        offset += 4;
        return v;
    }

    void skip_value(uint32_t vtype) {
        switch (vtype) {
        case 0:
        case 1:
        case 7:
            offset += 1;
            break;
        case 2:
        case 3:
            offset += 2;
            break;
        case 4:
        case 5:
        case 6:
            offset += 4;
            break;
        case 10:
        case 12:
            offset += 8;
            break;
        case 8:
            read_str();
            break;
        case 9: {
            uint32_t arr_type = read_u32();
            uint64_t arr_len = read_u64();
            static const int type_sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 4, 8};
            if (arr_type == 8) {
                for (uint64_t j = 0; j < arr_len; ++j)
                    read_str();
            } else if (arr_type < 13) {
                offset += arr_len * type_sizes[arr_type];
            } else {
                offset += arr_len * 4;
            }
            break;
        }
        default:
            offset += 8;
            break;
        }
    }
};

// ===== Load from GGUF =====

bool Tokenizer::load_from_gguf(const std::string& path) {
    // Read the file
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        LOG_ERROR("Tokenizer: Cannot open file: " + path);
        return false;
    }

    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // We only need the metadata section, which is at the beginning.
    // Large vocab models (e.g., Qwen with 152K tokens) can have metadata > 4MB.
    // Read up to 64MB to cover all metadata.
    size_t read_size = std::min(file_size, static_cast<size_t>(64 * 1024 * 1024));
    std::vector<uint8_t> buf(read_size);
    size_t nread = fread(buf.data(), 1, read_size, fp);
    fclose(fp);

    if (nread < 16) {
        LOG_ERROR("Tokenizer: File too small: " + path);
        return false;
    }

    uint32_t magic;
    std::memcpy(&magic, buf.data(), 4);
    if (magic != 0x46554747) {
        LOG_ERROR("Tokenizer: Not a GGUF file: " + path);
        return false;
    }

    uint32_t version;
    std::memcpy(&version, buf.data() + 4, 4);

    GgufMetaReader reader{buf.data(), buf.size(), 8};

    uint64_t tensor_count, meta_count;
    if (version >= 3) {
        tensor_count = reader.read_u64();
        meta_count = reader.read_u64();
    } else {
        tensor_count = reader.read_u32();
        meta_count = reader.read_u32();
    }

    // Initialize byte-level BPE tables
    // GPT-2 byte-level encoding: map each byte to a unique unicode character
    // Following the same mapping as in GPT-2 encoder.py
    {
        // The GPT-2 byte-to-unicode mapping
        // Bytes 33-126, 161-172, 174-255 map to themselves
        // Other bytes (0-32, 127-160, 173) map to 256+ unicode chars
        std::vector<int> byte_to_unicode(256);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                byte_to_unicode[b] = b;
            } else {
                byte_to_unicode[b] = 256 + n;
                n++;
            }
        }

        for (int b = 0; b < 256; ++b) {
            std::string ustr = unicode_to_utf8(static_cast<uint32_t>(byte_to_unicode[b]));
            byte_to_unicode_char_[static_cast<uint8_t>(b)] = ustr;
            unicode_char_to_byte_[ustr] = static_cast<uint8_t>(b);
        }
    }

    // Parse metadata
    std::string tok_model;
    std::vector<std::string> tok_tokens;
    std::vector<float> tok_scores;
    std::vector<int32_t> tok_token_types;
    std::vector<std::string> tok_merges;

    for (uint64_t i = 0; i < meta_count; ++i) {
        std::string key = reader.read_str();
        uint32_t vtype = reader.read_u32();

        if (key == "tokenizer.ggml.model") {
            tok_model = reader.read_str();
        } else if (key == "tokenizer.ggml.tokens") {
            uint32_t arr_type = reader.read_u32();
            uint64_t arr_len = reader.read_u64();
            tok_tokens.resize(arr_len);
            for (uint64_t j = 0; j < arr_len; ++j) {
                tok_tokens[j] = reader.read_str();
            }
        } else if (key == "tokenizer.ggml.scores") {
            uint32_t arr_type = reader.read_u32();
            uint64_t arr_len = reader.read_u64();
            tok_scores.resize(arr_len);
            for (uint64_t j = 0; j < arr_len; ++j) {
                tok_scores[j] = reader.read_f32();
            }
        } else if (key == "tokenizer.ggml.token_type") {
            uint32_t arr_type = reader.read_u32();
            uint64_t arr_len = reader.read_u64();
            tok_token_types.resize(arr_len);
            for (uint64_t j = 0; j < arr_len; ++j) {
                tok_token_types[j] = reader.read_i32();
            }
        } else if (key == "tokenizer.ggml.merges") {
            uint32_t arr_type = reader.read_u32();
            uint64_t arr_len = reader.read_u64();
            tok_merges.resize(arr_len);
            for (uint64_t j = 0; j < arr_len; ++j) {
                tok_merges[j] = reader.read_str();
            }
        } else if (key == "tokenizer.ggml.bos_token_id") {
            bos_id_ = static_cast<int32_t>(reader.read_u32());
        } else if (key == "tokenizer.ggml.eos_token_id") {
            eos_id_ = static_cast<int32_t>(reader.read_u32());
        } else if (key == "tokenizer.ggml.padding_token_id") {
            pad_id_ = static_cast<int32_t>(reader.read_u32());
        } else if (key == "tokenizer.ggml.unknown_token_id") {
            unk_id_ = static_cast<int32_t>(reader.read_u32());
        } else if (key == "tokenizer.ggml.add_bos_token" || key == "llama.add_bos_token") {
            // GGUF BOOL is a single byte (vtype 7).
            add_bos_token_ = (reader.read_u8() != 0);
        } else if (key == "tokenizer.ggml.add_eos_token" || key == "llama.add_eos_token") {
            add_eos_token_ = (reader.read_u8() != 0);
        } else if (key == "tokenizer.chat_template") {
            chat_template_ = reader.read_str();
        } else if (key == "tokenizer.ggml.pre") {
            pre_type_ = reader.read_str();
        } else {
            reader.skip_value(vtype);
        }
    }

    if (tok_tokens.empty()) {
        LOG_ERROR("Tokenizer: No tokens found in GGUF file");
        return false;
    }

    // Set model type
    if (tok_model == "gpt2" || tok_model == "bpe") {
        model_type_ = TokenizerModelType::BPE;
    } else {
        model_type_ = TokenizerModelType::SPM;
    }

    // Build vocab
    vocab_.tokens = std::move(tok_tokens);
    vocab_.scores = std::move(tok_scores);
    vocab_.token_types = std::move(tok_token_types);

    if (vocab_.scores.size() < vocab_.tokens.size()) {
        vocab_.scores.resize(vocab_.tokens.size(), 0.0f);
    }
    if (vocab_.token_types.size() < vocab_.tokens.size()) {
        vocab_.token_types.resize(vocab_.tokens.size(), 0);
    }

    for (int32_t i = 0; i < static_cast<int32_t>(vocab_.tokens.size()); ++i) {
        vocab_.token_to_id[vocab_.tokens[i]] = i;
    }

    // Build BPE merge ranks
    if (model_type_ == TokenizerModelType::BPE) {
        for (int i = 0; i < static_cast<int>(tok_merges.size()); ++i) {
            bpe_ranks_[tok_merges[i]] = i;
        }
    }

    // Set default special token IDs if not found
    if (bos_id_ < 0) {
        auto it = vocab_.token_to_id.find("<s>");
        if (it != vocab_.token_to_id.end())
            bos_id_ = it->second;
    }
    if (eos_id_ < 0) {
        auto it = vocab_.token_to_id.find("</s>");
        if (it != vocab_.token_to_id.end())
            eos_id_ = it->second;
    }
    if (unk_id_ < 0) {
        auto it = vocab_.token_to_id.find("<unk>");
        if (it != vocab_.token_to_id.end())
            unk_id_ = it->second;
    }

    LOG_INFO("Tokenizer loaded: model=" + tok_model + " pre=" + (pre_type_.empty() ? "<unspecified>" : pre_type_) +
             " vocab_size=" + std::to_string(vocab_.tokens.size()) + " bos=" + std::to_string(bos_id_) +
             " eos=" + std::to_string(eos_id_) + " merges=" + std::to_string(tok_merges.size()));

    loaded_ = true;
    return true;
}

// ===== Encode =====

void Tokenizer::ensure_special_tokens_cached() const {
    if (special_tokens_cached_)
        return;

    special_tokens_cache_.clear();

    for (int32_t i = 0; i < static_cast<int32_t>(vocab_.tokens.size()); ++i) {
        const auto& token = vocab_.tokens[i];
        if (token.empty())
            continue;

        bool is_special = false;

        // 1. Check GGUF token type: CONTROL(3), USER_DEFINED(4), UNUSED(5) are special
        if (i < static_cast<int32_t>(vocab_.token_types.size())) {
            int tt = vocab_.token_types[i];
            if (tt == 3 || tt == 4 || tt == 5) {
                is_special = true;
            }
        }

        // 2. Heuristic: tokens containing control-like characters are special
        if (!is_special) {
            for (char c : token) {
                if (c == '<' || c == '>' || c == '|' || c == '[' || c == ']') {
                    is_special = true;
                    break;
                }
            }
        }

        // 3. Skip byte tokens (type 6) and pure whitespace tokens
        if (is_special) {
            if (i < static_cast<int32_t>(vocab_.token_types.size()) &&
                vocab_.token_types[i] == 6) {
                continue;  // skip byte tokens
            }
            // Skip pure whitespace tokens
            bool all_space = true;
            for (char c : token) {
                if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                    all_space = false;
                    break;
                }
            }
            if (!all_space) {
                special_tokens_cache_.push_back({token, i});
            }
        }
    }

    // Sort by length descending for longest-match greedy
    std::sort(special_tokens_cache_.begin(), special_tokens_cache_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    special_tokens_cached_ = true;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos,
                                       bool add_dummy_prefix, bool parse_special) const {
    if (!loaded_)
        return {};

    if (!parse_special) {
        std::vector<int32_t> result;
        if (model_type_ == TokenizerModelType::BPE) {
            result = encode_bpe(text, add_bos, add_eos);
        } else {
            result = encode_spm(text, add_bos, add_eos, add_dummy_prefix);
        }
        return result;
    }

    // ---- parse_special mode ----
    ensure_special_tokens_cached();

    std::vector<int32_t> result;

    if (add_bos && bos_id_ >= 0) {
        result.push_back(bos_id_);
    }

    size_t pos = 0;
    while (pos < text.size()) {
        // Try to match a special token at current position (longest match first)
        bool matched = false;
        for (const auto& [token_str, tid] : special_tokens_cache_) {
            if (text.compare(pos, token_str.size(), token_str) == 0) {
                result.push_back(tid);
                pos += token_str.size();
                matched = true;
                break;
            }
        }

        if (matched)
            continue;

        // Find the next occurrence of any special token
        size_t next_special = std::string::npos;
        for (const auto& [token_str, tid] : special_tokens_cache_) {
            size_t found = text.find(token_str, pos);
            if (found != std::string::npos && found < next_special) {
                next_special = found;
            }
        }

        if (next_special == std::string::npos) {
            // No more special tokens — encode the rest as regular text
            std::string regular = text.substr(pos);
            auto regular_ids = encode(regular, false, false, add_dummy_prefix && (pos == 0), false);
            result.insert(result.end(), regular_ids.begin(), regular_ids.end());
            break;
        }

        // Encode the regular text segment before the next special token
        if (next_special > pos) {
            std::string regular = text.substr(pos, next_special - pos);
            auto regular_ids = encode(regular, false, false, add_dummy_prefix && (pos == 0), false);
            result.insert(result.end(), regular_ids.begin(), regular_ids.end());
        }
        pos = next_special;
    }

    if (add_eos && eos_id_ >= 0) {
        result.push_back(eos_id_);
    }

    return result;
}

std::vector<int32_t> Tokenizer::encode_spm(const std::string& text, bool add_bos, bool add_eos,
                                           bool add_dummy_prefix) const {
    std::vector<int32_t> result;

    if (add_bos && bos_id_ >= 0) {
        result.push_back(bos_id_);
    }

    // SentencePiece encoding:
    // 1. Replace spaces with ▁ (U+2581)
    // 2. Add ▁ prefix at the beginning (SPM "add_dummy_prefix")
    //    - If text starts with space, the first space acts as dummy prefix (absorbed)
    //    - If text doesn't start with space, prepend ▁
    // 3. Run bigram-merge tokenization on the ▁-substituted text
    //    (control characters are included and handled by the merge algorithm)

    const uint8_t SPM_SPACE_UTF8[] = {0xE2, 0x96, 0x81};  // ▁ U+2581
    const std::string spm_space(reinterpret_cast<const char*>(SPM_SPACE_UTF8), 3);

    // SPM adds a ▁ prefix at the beginning of text, but if the text
    // already starts with a space (which becomes ▁), the dummy prefix
    // is absorbed (not duplicated).
    // "Hello" -> "▁Hello", " Hello" -> "▁Hello", "  Hello" -> "▁▁Hello"

    bool starts_with_space = (!text.empty() && text[0] == ' ');

    std::string spm_text;
    spm_text.reserve(text.size() * 2);

    // Add dummy prefix if text doesn't start with space and add_dummy_prefix is true
    if (add_dummy_prefix && !starts_with_space) {
        spm_text += spm_space;
    }

    for (size_t i = 0; i < text.size();) {
        uint8_t b = static_cast<uint8_t>(text[i]);

        if (b == ' ') {
            // Space -> ▁
            spm_text += spm_space;
            i += 1;
        } else {
            // All other characters (including control chars) go into spm_text
            // The bigram-merge algorithm will handle them:
            // - If a control char matches a vocab token (e.g., \r -> ▁\r or \r), it's used
            // - If not, it falls back to <0xHH> byte token
            int char_len = 1;
            if (b >= 0xF0)
                char_len = 4;
            else if (b >= 0xE0)
                char_len = 3;
            else if (b >= 0xC0)
                char_len = 2;
            spm_text.append(text, i, char_len);
            i += char_len;
        }
    }

    // Run bigram-merge tokenization
    auto ids = encode_spm_greedy(spm_text);
    result.insert(result.end(), ids.begin(), ids.end());

    if (add_eos && eos_id_ >= 0) {
        result.push_back(eos_id_);
    }

    return result;
}

// SPM tokenization using bigram-merge algorithm (like llama.cpp)
// This matches SentencePiece's Unigram model encoding:
// 1. Split text into UTF-8 characters
// 2. Repeatedly merge the highest-scoring adjacent pair
// 3. Fallback to byte tokens for unmatched characters
std::vector<int32_t> Tokenizer::encode_spm_greedy(const std::string& spm_text) const {
    if (spm_text.empty())
        return {};

    // Symbol: a node in the linked list of text segments
    struct Symbol {
        int prev;  // index of previous symbol, -1 if none
        int next;  // index of next symbol, -1 if none
        const char* text;
        size_t n;  // length of this symbol's text
    };

    // Bigram: a candidate merge of two adjacent symbols
    struct Bigram {
        struct Comparator {
            bool operator()(const Bigram& l, const Bigram& r) const {
                return (l.score < r.score) || (l.score == r.score && l.left > r.left);
            }
        };
        int left;
        int right;
        float score;
        size_t size;
    };

    // Step 1: Split text into UTF-8 characters as initial symbols
    std::vector<Symbol> symbols;
    int index = 0;
    size_t offs = 0;
    while (offs < spm_text.size()) {
        Symbol sym;
        size_t len = 1;
        auto b = static_cast<uint8_t>(spm_text[offs]);
        if (b >= 0xF0)
            len = 4;
        else if (b >= 0xE0)
            len = 3;
        else if (b >= 0xC0)
            len = 2;
        len = std::min(len, spm_text.size() - offs);

        sym.text = spm_text.c_str() + offs;
        sym.n = len;
        sym.prev = index - 1;
        sym.next = (offs + len >= spm_text.size()) ? -1 : index + 1;
        symbols.push_back(sym);
        offs += len;
        index++;
    }

    // Reverse merge map: merged text -> (left_idx, right_idx)
    std::map<std::string, std::pair<int, int>> rev_merge;

    // Check if scores are all zero (common in quantized GGUF files)
    bool scores_all_zero = true;
    for (size_t i = 0; i < vocab_.scores.size() && i < 1000; ++i) {
        if (vocab_.scores[i] != 0.0f) {
            scores_all_zero = false;
            break;
        }
    }

    // Step 2: Seed the work queue with all possible 2-character merges
    using Queue = std::priority_queue<Bigram, std::vector<Bigram>, Bigram::Comparator>;
    Queue work_queue;

    auto try_add_bigram = [&](int left, int right) {
        if (left < 0 || right < 0 || left >= (int)symbols.size() || right >= (int)symbols.size())
            return;
        if (symbols[left].n == 0 || symbols[right].n == 0)
            return;

        std::string text(symbols[left].text, symbols[left].n + symbols[right].n);
        auto it = vocab_.token_to_id.find(text);
        if (it == vocab_.token_to_id.end())
            return;

        int32_t tid = it->second;
        if (tid < 0 || tid >= static_cast<int32_t>(vocab_.tokens.size()))
            return;

        Bigram bigram;
        bigram.left = left;
        bigram.right = right;
        // When scores are all zero, use heuristic scoring:
        // - Pure-space tokens (only ▁ chars) get very low score (-1e9)
        //   to prevent premature merging of spaces
        // - Other tokens use -tid as score (lower ID = higher priority)
        if (scores_all_zero) {
            bool is_space_only = true;
            const uint8_t SPM_SPACE_UTF8[] = {0xE2, 0x96, 0x81};
            for (size_t k = 0; k < text.size();) {
                if (k + 2 < text.size() && static_cast<uint8_t>(text[k]) == SPM_SPACE_UTF8[0] &&
                    static_cast<uint8_t>(text[k + 1]) == SPM_SPACE_UTF8[1] &&
                    static_cast<uint8_t>(text[k + 2]) == SPM_SPACE_UTF8[2]) {
                    k += 3;
                } else {
                    is_space_only = false;
                    break;
                }
            }
            bigram.score = is_space_only ? -1e9f : -static_cast<float>(tid);
        } else {
            bigram.score = vocab_.scores[tid];
        }
        bigram.size = text.size();
        work_queue.push(bigram);
        rev_merge[text] = std::make_pair(left, right);
    };

    for (int i = 1; i < (int)symbols.size(); ++i) {
        try_add_bigram(i - 1, i);
    }

    // Step 3: Keep merging the highest-scoring pair
    while (!work_queue.empty()) {
        auto bigram = work_queue.top();
        work_queue.pop();

        auto& left_sym = symbols[bigram.left];
        auto& right_sym = symbols[bigram.right];

        // Skip if already merged or size mismatch
        if (left_sym.n == 0 || right_sym.n == 0 || left_sym.n + right_sym.n != bigram.size) {
            continue;
        }

        // Merge right into left
        left_sym.n += right_sym.n;
        right_sym.n = 0;

        // Remove right from the chain
        left_sym.next = right_sym.next;
        if (right_sym.next >= 0) {
            symbols[right_sym.next].prev = bigram.left;
        }

        // Try new merges with neighbors
        try_add_bigram(left_sym.prev, bigram.left);
        try_add_bigram(bigram.left, left_sym.next);
    }

    // Step 4: Collect tokens from the symbol chain
    std::vector<int32_t> output;
    for (int i = 0; i != -1; i = symbols[i].next) {
        auto& sym = symbols[i];
        if (sym.n == 0)
            continue;

        std::string text(sym.text, sym.n);
        auto it = vocab_.token_to_id.find(text);
        if (it != vocab_.token_to_id.end()) {
            output.push_back(it->second);
        } else {
            // Check if this was a merge that can be resegmented
            auto p = rev_merge.find(text);
            if (p != rev_merge.end()) {
                // Recursively resegment
                // For simplicity, fall back to byte tokens
                for (size_t j = 0; j < sym.n; ++j) {
                    uint8_t b = static_cast<uint8_t>(sym.text[j]);
                    char buf[8];
                    snprintf(buf, sizeof(buf), "<0x%02X>", b);
                    auto bit = vocab_.token_to_id.find(buf);
                    if (bit != vocab_.token_to_id.end()) {
                        output.push_back(bit->second);
                    } else if (unk_id_ >= 0) {
                        output.push_back(unk_id_);
                    }
                }
            } else {
                // Output as byte tokens
                for (size_t j = 0; j < sym.n; ++j) {
                    uint8_t b = static_cast<uint8_t>(sym.text[j]);
                    char buf[8];
                    snprintf(buf, sizeof(buf), "<0x%02X>", b);
                    auto bit = vocab_.token_to_id.find(buf);
                    if (bit != vocab_.token_to_id.end()) {
                        output.push_back(bit->second);
                    } else if (unk_id_ >= 0) {
                        output.push_back(unk_id_);
                    }
                }
            }
        }
    }

    return output;
}

std::string Tokenizer::text_to_byte_level(const std::string& text) const {
    std::string result;
    for (size_t i = 0; i < text.size();) {
        uint8_t b = static_cast<uint8_t>(text[i]);
        // Check if this is a multi-byte UTF-8 character
        int char_len = 1;
        if (b >= 0xF0)
            char_len = 4;
        else if (b >= 0xE0)
            char_len = 3;
        else if (b >= 0xC0)
            char_len = 2;

        // For each byte of the character, map to unicode char
        for (int j = 0; j < char_len && (i + j) < text.size(); ++j) {
            auto it = byte_to_unicode_char_.find(static_cast<uint8_t>(text[i + j]));
            if (it != byte_to_unicode_char_.end()) {
                result += it->second;
            } else {
                result += text[i + j];
            }
        }
        i += char_len;
    }
    return result;
}

std::string Tokenizer::byte_level_to_text(const std::string& text) const {
    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        // Try to match a unicode char to byte
        bool found = false;
        // Try longest match first (multi-byte UTF-8 sequences in the mapping)
        for (size_t len = std::min(static_cast<size_t>(4), text.size() - pos); len >= 1; --len) {
            std::string candidate = text.substr(pos, len);
            auto it = unicode_char_to_byte_.find(candidate);
            if (it != unicode_char_to_byte_.end()) {
                result += static_cast<char>(it->second);
                pos += len;
                found = true;
                break;
            }
        }
        if (!found) {
            result += text[pos];
            pos += 1;
        }
    }
    return result;
}

namespace {

// ---- spark2_5 pre-tokenization support ----
// Mirrors llama.cpp's LLAMA_VOCAB_PRE_TYPE_SPARK2_5 regex set:
//   \p{N}{1,3} | [CJK]+ | (punct+letters | word | punct/sym | newline | ws) | \p{N}
// Implemented by hand over Unicode codepoints because std::regex does not
// support \p{} property classes.

enum class UC : int { WS, DIGIT, LETTER, PUNCT, SYMBOL, ACCENT, CJK, OTHER };

static bool u_is_han(uint32_t c) {
    return (c>=0x4E00&&c<=0x9FFF) || (c>=0x3400&&c<=0x4DBF) ||
           (c>=0x20000&&c<=0x2A6DF) || (c>=0x2A700&&c<=0x2B73F) ||
           (c>=0x2B740&&c<=0x2B81F) || (c>=0x2B820&&c<=0x2CEAF) ||
           (c>=0x2CEB0&&c<=0x2EBEF) || (c>=0xF900&&c<=0xFAFF) ||
           (c>=0x2F800&&c<=0x2FA1F);
}
static bool u_is_hira(uint32_t c) { return c>=0x3040 && c<=0x309F; }
static bool u_is_kana(uint32_t c) { return c>=0x30A0 && c<=0x30FF; }
static bool u_is_hangul(uint32_t c) {
    return (c>=0xAC00&&c<=0xD7A3) || (c>=0x1100&&c<=0x11FF) || (c>=0x3130&&c<=0x318F);
}
static bool u_is_accent(uint32_t c) {
    if (c>=0x0300&&c<=0x036F) return true;
    if (c>=0x0483&&c<=0x0489) return true;
    if (c>=0x0591&&c<=0x05BD) return true;
    if (c>=0x0610&&c<=0x061A) return true;
    if (c>=0x064B&&c<=0x065F) return true;
    if (c>=0x1AB0&&c<=0x1AFF) return true;
    if (c>=0x1DC0&&c<=0x1DFF) return true;
    if (c>=0x20D0&&c<=0x20FF) return true;
    if (c>=0xFE20&&c<=0xFE2F) return true;
    return false;
}
static bool u_is_digit(uint32_t c) {
    if (c>='0' && c<='9') return true;
    static const std::pair<uint32_t,uint32_t> nd[] = {
        {0x0660,0x0669},{0x06F0,0x06F9},{0x07C0,0x07C9},{0x0966,0x096F},
        {0x09E6,0x09EF},{0x0A66,0x0A6F},{0x0AE6,0x0AEF},{0x0B66,0x0B6F},
        {0x0BE6,0x0BEF},{0x0C66,0x0C6F},{0x0CE6,0x0CEF},{0x0D66,0x0D6F},
        {0x0E50,0x0E59},{0x0ED0,0x0ED9},{0x0F20,0x0F29},{0x1040,0x1049},
        {0x17E0,0x17E9},{0x1810,0x1819},{0x1946,0x194F},{0x19D0,0x19D9},
        {0x1A80,0x1A89},{0x1A90,0x1A99},{0x1B50,0x1B59},{0x1BB0,0x1BB9},
        {0xA620,0xA629},{0xA8D0,0xA8D9},{0xA900,0xA909},{0xA9D0,0xA9D9},
        {0xAA50,0xAA59},{0xABF0,0xABF9},{0xFF10,0xFF19},
    };
    for (const auto& r : nd) if (c>=r.first && c<=r.second) return true;
    return false;
}
static bool u_is_punct(uint32_t c) {
    if (c>=0x200B && c<=0x200F) return true;
    if (c>=0x2010 && c<=0x2027) return true;
    if (c>=0x202A && c<=0x202E) return true;
    if (c>=0x2030 && c<=0x205E) return true;
    if (c>=0x205F && c<=0x206F) return true;
    if (c>=0x2E00 && c<=0x2E7F) return true;
    if (c>=0x3001 && c<=0x303F) return true;
    if (c>=0xFE30 && c<=0xFE4F) return true;
    if (c>=0xFE50 && c<=0xFE6B) return true;
    return false;
}
static bool u_is_symbol(uint32_t c) {
    if (c>=0x20A0 && c<=0x20B8) return true;
    if (c>=0x2100 && c<=0x214F) return true;
    if (c>=0x2150 && c<=0x218F) return true;
    if (c>=0x2190 && c<=0x21FF) return true;
    if (c>=0x2200 && c<=0x22FF) return true;
    if (c>=0x2300 && c<=0x23FF) return true;
    if (c>=0x2400 && c<=0x243F) return true;
    if (c>=0x2460 && c<=0x24FF) return true;
    if (c>=0x2500 && c<=0x25FF) return true;
    if (c>=0x2600 && c<=0x26FF) return true;
    if (c>=0x2700 && c<=0x27BF) return true;
    if (c>=0x27C0 && c<=0x27EF) return true;
    if (c>=0x2B00 && c<=0x2BFF) return true;
    if (c>=0x1F100 && c<=0x1F1FF) return true;
    return false;
}

static UC u_class(uint32_t c) {
    if (c < 0x80) {
        if (c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v') return UC::WS;
        if (c>='0'&&c<='9') return UC::DIGIT;
        if ((c>='A'&&c<='Z')||(c>='a'&&c<='z')) return UC::LETTER;
        static const char* ps = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        if (strchr(ps, (char)c)) return UC::PUNCT;
        return UC::OTHER;
    }
    if (c>=0xFF00 && c<=0xFFEF) {  // fullwidth / halfwidth forms
        if (c>=0xFF10 && c<=0xFF19) return UC::DIGIT;
        if ((c>=0xFF21&&c<=0xFF3A)||(c>=0xFF41&&c<=0xFF5A)) return UC::LETTER;
        if (c>=0xFF01 && c<=0xFF60) return UC::PUNCT;  // fullwidth ASCII punctuation
        return UC::OTHER;
    }
    if (c==0x0085||c==0x00A0||c==0x1680||(c>=0x2000&&c<=0x200A)||c==0x2028||c==0x2029||c==0x202F||c==0x205F||c==0x3000)
        return UC::WS;
    if (u_is_han(c)||u_is_hira(c)||u_is_kana(c)) return UC::CJK;
    if (u_is_hangul(c)) return UC::LETTER;
    if (u_is_accent(c)) return UC::ACCENT;
    if (u_is_punct(c)) return UC::PUNCT;
    if (u_is_symbol(c)) return UC::SYMBOL;
    return UC::LETTER;  // unknown non-ASCII scripts treated as letters
}

static bool u_is_ascii_letter(uint32_t c) {
    return (c>='A'&&c<='Z')||(c>='a'&&c<='z');
}
static const char* k_ascii_punctsym = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
static bool u_is_ascii_punctsym(uint32_t c) {
    return c < 0x80 && strchr(k_ascii_punctsym, (char)c) != nullptr;
}

}  // namespace

std::vector<std::string> Tokenizer::bpe_pre_tokenize(const std::string& text) const {
    if (pre_type_ == "spark2_5") {
        return bpe_pre_tokenize_spark25(text);
    }

    // GPT-2 pre-tokenization regex
    // Pattern: '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    std::vector<std::string> chunks;

    // Simplified GPT-2 pre-tokenizer
    // Split into: contractions, words (with optional leading space), numbers,
    // punctuation, whitespace
    static const std::regex gpt2_pattern(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+)",
        std::regex::icase | std::regex::optimize);

    auto it = std::sregex_iterator(text.begin(), text.end(), gpt2_pattern);
    auto end = std::sregex_iterator();

    for (; it != end; ++it) {
        chunks.push_back(it->str());
    }

    return chunks;
}

std::vector<std::string> Tokenizer::bpe_pre_tokenize_spark25(const std::string& text) const {
    std::vector<std::string> chunks;

    // Decode to codepoints with their byte offsets.
    std::vector<uint32_t> cpts;
    std::vector<size_t> boff;
    size_t off = 0;
    while (off < text.size()) {
        size_t s = off;
        uint32_t cp = utf8_to_unicode(text, off);
        cpts.push_back(cp);
        boff.push_back(s);
    }
    boff.push_back(text.size());
    const size_t n = cpts.size();

    auto emit = [&](size_t i, size_t len) {
        size_t a = boff[i], b = boff[i + len];
        chunks.push_back(text.substr(a, b - a));
    };

    size_t i = 0;
    while (i < n) {
        uint32_t cp = cpts[i];
        UC cl = u_class(cp);
        size_t consume = 1;

        if (cl == UC::DIGIT) {
            // alt1: \p{N}{1,3}
            size_t j = i;
            while (j < n && u_class(cpts[j]) == UC::DIGIT && (j - i) < 3) ++j;
            consume = j - i;
        } else if (cl == UC::CJK) {
            // alt2: [CJK]+
            size_t j = i;
            while (j < n && u_class(cpts[j]) == UC::CJK) ++j;
            consume = j - i;
        } else {
            bool matched = false;
            // alt3a: [punct/sym ASCII][A-Za-z]+
            if (!matched && u_is_ascii_punctsym(cp) && i + 1 < n &&
                u_is_ascii_letter(cpts[i + 1])) {
                size_t j = i + 1;
                while (j < n && u_is_ascii_letter(cpts[j])) ++j;
                consume = j - i; matched = true;
            }
            // alt3b: [^\r\n\L\P\S]? [\L\M]+
            // NOTE: in llama.cpp CJK (Han/Hira/Kata) are Unicode *letters* (\p{L}),
            // so the word rule also covers them. We treat CJK as word-like here
            // (alt2 already grabbed any standalone CJK run at this position).
            if (!matched) {
                auto wordish = [](UC c) { return c == UC::LETTER || c == UC::ACCENT || c == UC::CJK; };
                bool is_letterish = wordish(cl);
                bool prefix_ok = (cp != '\r' && cp != '\n' && !wordish(cl) &&
                                  cl != UC::PUNCT && cl != UC::SYMBOL);
                bool letter_after = (i + 1 < n) && wordish(u_class(cpts[i + 1]));
                if (is_letterish) {
                    size_t j = i;
                    while (j < n && wordish(u_class(cpts[j]))) ++j;
                    consume = j - i; matched = true;
                } else if (prefix_ok && letter_after) {
                    size_t j = i + 1;
                    while (j < n && wordish(u_class(cpts[j]))) ++j;
                    consume = j - i; matched = true;
                }
            }
            // alt3c:  ? [\P\S]+
            if (!matched) {
                bool ps = (cl == UC::PUNCT || cl == UC::SYMBOL);
                if (ps) {
                    size_t j = i;
                    while (j < n && (u_class(cpts[j]) == UC::PUNCT || u_class(cpts[j]) == UC::SYMBOL)) ++j;
                    consume = j - i; matched = true;
                } else if (cp == ' ' && i + 1 < n &&
                           (u_class(cpts[i + 1]) == UC::PUNCT || u_class(cpts[i + 1]) == UC::SYMBOL)) {
                    size_t j = i + 1;
                    while (j < n && (u_class(cpts[j]) == UC::PUNCT || u_class(cpts[j]) == UC::SYMBOL)) ++j;
                    consume = j - i; matched = true;
                }
            }
            // alt3d: [\r\n]
            if (!matched && (cp == '\r' || cp == '\n')) { consume = 1; matched = true; }
            // alt3e: \s+(?!\S)
            if (!matched && cl == UC::WS) {
                size_t j = i;
                while (j < n && u_class(cpts[j]) == UC::WS) ++j;
                if (j == n) { consume = j - i; matched = true; }       // trailing whitespace
                else if ((j - i) >= 2) { consume = (j - i) - 1; matched = true; } // keep 1 for next token
            }
            // alt3f: \s+
            if (!matched && cl == UC::WS) {
                size_t j = i;
                while (j < n && u_class(cpts[j]) == UC::WS) ++j;
                consume = j - i; matched = true;
            }
            // fallback: a single codepoint
            if (!matched) consume = 1;
        }
        emit(i, consume);
        i += consume;
    }
    return chunks;
}

std::vector<std::string> Tokenizer::bpe_apply(const std::vector<std::string>& word) const {
    if (word.size() <= 1)
        return word;

    std::vector<std::string> tokens = word;

    while (tokens.size() > 1) {
        // Find the pair with the lowest rank
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = 0;

        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            std::string pair = tokens[i] + " " + tokens[i + 1];
            auto it = bpe_ranks_.find(pair);
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_rank == std::numeric_limits<int>::max())
            break;

        // Merge the best pair
        std::vector<std::string> new_tokens;
        size_t i = 0;
        while (i < tokens.size()) {
            if (i == best_idx) {
                new_tokens.push_back(tokens[i] + tokens[i + 1]);
                i += 2;
            } else {
                new_tokens.push_back(tokens[i]);
                i += 1;
            }
        }
        tokens = std::move(new_tokens);
    }

    return tokens;
}

std::vector<int32_t> Tokenizer::encode_bpe(const std::string& text, bool add_bos,
                                           bool add_eos) const {
    std::vector<int32_t> result;

    if (add_bos && bos_id_ >= 0) {
        result.push_back(bos_id_);
    }

    // Pre-tokenize
    auto chunks = bpe_pre_tokenize(text);

    for (const auto& chunk : chunks) {
        // Convert to byte-level representation
        std::string byte_level = text_to_byte_level(chunk);

        // Split into individual characters for BPE
        std::vector<std::string> chars;
        size_t pos = 0;
        while (pos < byte_level.size()) {
            uint8_t b = static_cast<uint8_t>(byte_level[pos]);
            int char_len = 1;
            if (b >= 0xF0)
                char_len = 4;
            else if (b >= 0xE0)
                char_len = 3;
            else if (b >= 0xC0)
                char_len = 2;

            if (pos + char_len <= byte_level.size()) {
                chars.push_back(byte_level.substr(pos, char_len));
            } else {
                chars.push_back(byte_level.substr(pos, 1));
                char_len = 1;
            }
            pos += char_len;
        }

        // Apply BPE merges
        auto bpe_tokens = bpe_apply(chars);

        // Look up token IDs
        for (const auto& tok : bpe_tokens) {
            auto it = vocab_.token_to_id.find(tok);
            if (it != vocab_.token_to_id.end()) {
                result.push_back(it->second);
            } else if (unk_id_ >= 0) {
                result.push_back(unk_id_);
            }
        }
    }

    if (add_eos && eos_id_ >= 0) {
        result.push_back(eos_id_);
    }

    return result;
}

// ===== Decode =====

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special,
                              bool strip_leading_space) const {
    if (!loaded_)
        return "";

    std::string result;
    for (int32_t id : ids) {
        if (id < 0 || id >= static_cast<int32_t>(vocab_.tokens.size()))
            continue;

        if (skip_special) {
            // Skip BOS, EOS, PAD tokens
            if (id == bos_id_ || id == eos_id_ || id == pad_id_)
                continue;
            // Skip special tokens (GGUF types 3/4/5: CONTROL like <s>, and
            // USER_DEFINED like <|im_end|> / <|im_start|>). Must stay in sync
            // with ensure_special_tokens_cached(). NOT byte tokens (type 6:
            // <0xHH>) — they represent actual text content.
            if (id < static_cast<int32_t>(vocab_.token_types.size())) {
                int tt = vocab_.token_types[id];
                if (tt == 3 || tt == 4 || tt == 5)
                    continue;  // CONTROL / USER_DEFINED / UNUSED
            }
        }

        result += vocab_.tokens[id];
    }

    // Convert SPM ▁ to space and <0xHH> byte tokens to characters
    if (model_type_ == TokenizerModelType::SPM) {
        std::string decoded;
        decoded.reserve(result.size());
        for (size_t i = 0; i < result.size();) {
            // Check for ▁ (U+2581, encoded as E2 96 81 in UTF-8)
            if (i + 2 < result.size() && static_cast<uint8_t>(result[i]) == 0xE2 &&
                static_cast<uint8_t>(result[i + 1]) == 0x96 &&
                static_cast<uint8_t>(result[i + 2]) == 0x81) {
                decoded += ' ';
                i += 3;
            }
            // Check for <0xHH> byte tokens
            else if (i + 5 <= result.size() && result[i] == '<' && result[i + 1] == '0' &&
                     result[i + 2] == 'x') {
                // Parse hex
                auto hex_val = [](char c) -> int {
                    if (c >= '0' && c <= '9')
                        return c - '0';
                    if (c >= 'A' && c <= 'F')
                        return c - 'A' + 10;
                    if (c >= 'a' && c <= 'f')
                        return c - 'a' + 10;
                    return -1;
                };
                int hi = hex_val(result[i + 3]);
                int lo = hex_val(result[i + 4]);
                if (hi >= 0 && lo >= 0 && result[i + 5] == '>') {
                    decoded += static_cast<char>(hi * 16 + lo);
                    i += 6;
                } else {
                    decoded += result[i];
                    i += 1;
                }
            } else {
                decoded += result[i];
                i += 1;
            }
        }
        // Remove the leading space (from dummy prefix ▁)
        // SPM encodes with a dummy ▁ at the start, which becomes a space.
        // The original text didn't have this leading space.
        // Only strip when decoding a complete sequence (not partial/streaming).
        if (strip_leading_space && !decoded.empty() && decoded[0] == ' ') {
            decoded.erase(decoded.begin());
        }
        result = std::move(decoded);
    } else {
        // BPE: convert byte-level representation back to text
        result = byte_level_to_text(result);
    }

    return result;
}

std::string Tokenizer::decode_token(int32_t id) const {
    if (!loaded_ || id < 0 || id >= static_cast<int32_t>(vocab_.tokens.size()))
        return "";
    return decode({id}, false);
}

// ===== Accessors =====

int32_t Tokenizer::token_to_id(const std::string& token) const {
    auto it = vocab_.token_to_id.find(token);
    return it != vocab_.token_to_id.end() ? it->second : unk_id_;
}

std::string Tokenizer::id_to_token(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(vocab_.tokens.size()))
        return "";
    return vocab_.tokens[id];
}

float Tokenizer::token_score(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(vocab_.scores.size()))
        return 0.0f;
    return vocab_.scores[id];
}

int32_t Tokenizer::token_type(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(vocab_.token_types.size()))
        return 0;
    return vocab_.token_types[id];
}

}  // namespace forge
