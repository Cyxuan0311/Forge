"""Tests for the Tokenizer class.

Tests are split into:
- Basic tests that don't require a GGUF model file
- Integration tests that require TinyLlama GGUF model
"""
import os
import sys
import pytest

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TINYLLAMA_Q4_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")


def tinyllama_available():
    return os.path.exists(TINYLLAMA_Q4_PATH)


# ============================================================================
# Basic tests (no GGUF model required)
# ============================================================================

class TestTokenizerBasic:
    def test_create_tokenizer(self):
        tok = forge.Tokenizer()
        assert tok is not None

    def test_not_loaded_initially(self):
        tok = forge.Tokenizer()
        assert not tok.is_loaded

    def test_load_nonexistent_file(self):
        tok = forge.Tokenizer()
        result = tok.load_from_gguf("/nonexistent/model.gguf")
        assert result is False or not tok.is_loaded

    def test_encode_before_load_returns_empty_or_raises(self):
        tok = forge.Tokenizer()
        try:
            ids = tok.encode("hello")
            # If no exception, should return empty or invalid result
            assert isinstance(ids, list)
        except Exception:
            pass  # Expected: raises exception

    def test_decode_before_load_returns_empty_or_raises(self):
        tok = forge.Tokenizer()
        try:
            text = tok.decode([1, 2, 3])
            assert isinstance(text, str)
        except Exception:
            pass  # Expected: raises exception


# ============================================================================
# TinyLlama SPM tokenizer tests
# ============================================================================

@pytest.mark.skipif(not tinyllama_available(), reason="TinyLlama GGUF model not found")
class TestTokenizerSPM:
    @pytest.fixture(autouse=True)
    def setup_tokenizer(self):
        self.tok = forge.Tokenizer()
        self.tok.load_from_gguf(TINYLLAMA_Q4_PATH)

    def test_is_loaded(self):
        assert self.tok.is_loaded

    def test_vocab_size(self):
        assert self.tok.vocab_size == 32000

    def test_special_token_ids(self):
        assert self.tok.bos_token_id == 1
        assert self.tok.eos_token_id == 2

    def test_model_type_spm(self):
        assert self.tok.model_type == forge.TokenizerModelType.SPM

    def test_chat_template_not_empty(self):
        template = self.tok.chat_template
        assert len(template) > 0

    def test_encode_simple(self):
        ids = self.tok.encode("Hello", add_bos=False)
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_encode_with_bos(self):
        ids = self.tok.encode("Hello", add_bos=True)
        assert ids[0] == self.tok.bos_token_id

    def test_encode_without_bos(self):
        ids = self.tok.encode("Hello", add_bos=False)
        if len(ids) > 0:
            assert ids[0] != self.tok.bos_token_id

    def test_encode_with_eos(self):
        ids = self.tok.encode("Hello", add_bos=False, add_eos=True)
        assert ids[-1] == self.tok.eos_token_id

    def test_encode_empty_string(self):
        ids = self.tok.encode("", add_bos=False)
        # Empty string may produce 0 or 1 token depending on implementation
        assert isinstance(ids, list)

    def test_encode_single_char(self):
        ids = self.tok.encode("a", add_bos=False)
        assert len(ids) > 0

    def test_encode_spaces(self):
        ids = self.tok.encode("  hello  ", add_bos=False)
        assert len(ids) > 0

    def test_encode_multiline(self):
        ids = self.tok.encode("line1\nline2", add_bos=False)
        assert len(ids) > 0

    def test_encode_unicode_chinese(self):
        ids = self.tok.encode("你好世界", add_bos=False)
        assert len(ids) > 0

    def test_encode_unicode_japanese(self):
        ids = self.tok.encode("こんにちは", add_bos=False)
        assert len(ids) > 0

    def test_encode_unicode_emoji(self):
        ids = self.tok.encode("Hello 🌍", add_bos=False)
        assert len(ids) > 0

    def test_encode_long_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        ids = self.tok.encode(text, add_bos=False)
        assert len(ids) > 0

    def test_encode_deterministic(self):
        ids1 = self.tok.encode("Hello world", add_bos=False)
        ids2 = self.tok.encode("Hello world", add_bos=False)
        assert ids1 == ids2

    def test_encode_different_texts_different_ids(self):
        ids1 = self.tok.encode("Hello", add_bos=False)
        ids2 = self.tok.encode("World", add_bos=False)
        assert ids1 != ids2

    def test_encode_dummy_prefix(self):
        # With dummy prefix (default), SPM adds a leading space
        ids_with = self.tok.encode("hello", add_bos=False, add_dummy_prefix=True)
        ids_without = self.tok.encode("hello", add_bos=False, add_dummy_prefix=False)
        # They may differ because dummy prefix adds a space token
        assert isinstance(ids_with, list)
        assert isinstance(ids_without, list)

    def test_decode_simple(self):
        ids = self.tok.encode("Hello", add_bos=False)
        text = self.tok.decode(ids, skip_special=True)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_decode_roundtrip(self):
        original = "Hello world"
        ids = self.tok.encode(original, add_bos=False)
        decoded = self.tok.decode(ids, skip_special=True)
        # After stripping, the decoded text should contain the original words
        assert "Hello" in decoded or "hello" in decoded

    def test_decode_single_token(self):
        ids = self.tok.encode("a", add_bos=False)
        text = self.tok.decode(ids, skip_special=False)
        assert isinstance(text, str)

    def test_decode_empty_ids(self):
        text = self.tok.decode([], skip_special=True)
        assert text == ""

    def test_decode_token(self):
        # decode_token should return a string for a valid token ID
        text = self.tok.decode_token(1)
        assert isinstance(text, str)

    def test_token_to_id_known_tokens(self):
        bos_id = self.tok.token_to_id("<s>")
        assert bos_id == 1

        eos_id = self.tok.token_to_id("</s>")
        assert eos_id == 2

    def test_token_to_id_unknown(self):
        unk_id = self.tok.token_to_id("this_token_definitely_does_not_exist_xyz123")
        assert unk_id == self.tok.unk_token_id or unk_id == -1

    def test_id_to_token_roundtrip(self):
        # For known tokens, id_to_token should return a string
        for tid in [0, 1, 2, 100, 1000]:
            token = self.tok.id_to_token(tid)
            assert isinstance(token, str)

    def test_token_to_id_id_to_token_roundtrip(self):
        # token_to_id -> id_to_token should return the original token
        for token_str in ["<s>", "</s>", "the", "a"]:
            tid = self.tok.token_to_id(token_str)
            if tid >= 0:
                recovered = self.tok.id_to_token(tid)
                assert recovered == token_str

    def test_token_score(self):
        score = self.tok.token_score(0)
        assert isinstance(score, float)

    def test_token_type(self):
        ttype = self.tok.token_type(0)
        assert isinstance(ttype, int)

    def test_vocab_size_consistency(self):
        # vocab_size should match the number of tokens we can access
        vs = self.tok.vocab_size
        assert vs > 0
        # id_to_token for the last valid ID should work
        token = self.tok.id_to_token(vs - 1)
        assert isinstance(token, str)

    def test_encode_decode_chinese_roundtrip(self):
        original = "你好世界"
        ids = self.tok.encode(original, add_bos=False)
        decoded = self.tok.decode(ids, skip_special=True)
        # Chinese text should roundtrip reasonably
        assert "你" in decoded or len(decoded) > 0

    def test_encode_special_token_handling(self):
        # Encoding text that contains special token strings
        ids = self.tok.encode("<s>Hello</s>", add_bos=False)
        assert isinstance(ids, list)

    def test_decode_skip_special(self):
        # Use real tokens from encoding to avoid invalid UTF-8 issues
        ids = self.tok.encode("Hello world", add_bos=True)
        text_without = self.tok.decode(ids, skip_special=True)
        # skip_special=True should not contain <s> or </s>
        assert "<s>" not in text_without
        assert "</s>" not in text_without

    def test_decode_strip_leading_space(self):
        ids = self.tok.encode("hello", add_bos=False)
        text_stripped = self.tok.decode(ids, skip_special=True, strip_leading_space=True)
        text_not_stripped = self.tok.decode(ids, skip_special=True, strip_leading_space=False)
        assert isinstance(text_stripped, str)
        assert isinstance(text_not_stripped, str)


@pytest.mark.skipif(not tinyllama_available(), reason="TinyLlama GGUF model not found")
class TestTokenizerSPMAdvanced:
    @pytest.fixture(autouse=True)
    def setup_tokenizer(self):
        self.tok = forge.Tokenizer()
        self.tok.load_from_gguf(TINYLLAMA_Q4_PATH)

    def test_encode_numbers(self):
        ids = self.tok.encode("12345", add_bos=False)
        assert len(ids) > 0

    def test_encode_punctuation(self):
        ids = self.tok.encode("Hello, world! How are you?", add_bos=False)
        assert len(ids) > 0

    def test_encode_tab(self):
        ids = self.tok.encode("hello\tworld", add_bos=False)
        assert len(ids) > 0

    def test_encode_multiple_newlines(self):
        ids = self.tok.encode("\n\n\n", add_bos=False)
        assert len(ids) > 0

    def test_encode_mixed_language(self):
        ids = self.tok.encode("Hello 你好 こんにちは", add_bos=False)
        assert len(ids) > 0

    def test_encode_code_snippet(self):
        code = "def foo(x):\n    return x + 1"
        ids = self.tok.encode(code, add_bos=False)
        assert len(ids) > 0

    def test_encode_json_like(self):
        text = '{"key": "value", "number": 42}'
        ids = self.tok.encode(text, add_bos=False)
        assert len(ids) > 0

    def test_encode_repeated_text(self):
        text = "abc " * 50
        ids = self.tok.encode(text, add_bos=False)
        assert len(ids) > 0

    def test_long_text_token_count(self):
        # Longer text should generally produce more tokens
        short_ids = self.tok.encode("Hello", add_bos=False)
        long_ids = self.tok.encode("Hello world, this is a longer sentence with more words.", add_bos=False)
        assert len(long_ids) >= len(short_ids)

    def test_encode_whitespace_only(self):
        ids = self.tok.encode("   ", add_bos=False)
        assert isinstance(ids, list)

    def test_decode_partial_sequence(self):
        # Decode should work with partial token sequences
        ids = self.tok.encode("Hello world", add_bos=False)
        if len(ids) > 1:
            partial = self.tok.decode(ids[:2], skip_special=True)
            assert isinstance(partial, str)

    def test_multiple_load_calls(self):
        # Loading tokenizer multiple times should work
        tok2 = forge.Tokenizer()
        tok2.load_from_gguf(TINYLLAMA_Q4_PATH)
        ids1 = self.tok.encode("test", add_bos=False)
        ids2 = tok2.encode("test", add_bos=False)
        assert ids1 == ids2
