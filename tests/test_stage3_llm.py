"""Stage 3: LLM high-level API and chat template tests."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "build"))

import forge
import forge_llm


# ====================================================================
#  ChatTemplate tests
# ====================================================================

class TestChatTemplateDetection:
    def test_detect_plain_with_none_tokenizer(self):
        tmpl = forge_llm.detect_chat_template(None)
        assert tmpl.name == "plain"

    def test_detect_by_template_name(self):
        tmpl = forge_llm.detect_chat_template(None, "chatml")
        assert tmpl.name == "chatml"

    def test_detect_by_template_name_unknown(self):
        tmpl = forge_llm.detect_chat_template(None, "nonexistent")
        assert tmpl.name == "plain"

    def test_all_templates_registered(self):
        names = set(forge_llm._TEMPLATES.keys())
        expected = {"chatml", "llama3", "zephyr", "llama2", "gemma4", "deepseek", "phi3", "plain"}
        assert names == expected

    def test_detect_tinyllama(self):
        tok = forge.Tokenizer()
        tok.load_from_gguf("models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
        tmpl = forge_llm.detect_chat_template(tok)
        assert tmpl.name == "zephyr"


class TestChatTemplateApply:
    """Test template application with a real tokenizer."""

    @pytest.fixture
    def tok(self):
        gguf_path = os.path.join(os.path.dirname(__file__), "..", "models",
                                 "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
        if not os.path.exists(gguf_path):
            pytest.skip("TinyLlama GGUF not found")
        t = forge.Tokenizer()
        t.load_from_gguf(gguf_path)
        return t

    def test_plain_template(self, tok):
        tmpl = forge_llm._PlainTemplate()
        ids = tmpl.apply(tok, [{"role": "user", "content": "Hello"}])
        assert len(ids) > 1
        decoded = tok.decode(ids, skip_special=True)
        assert "user" in decoded or "Hello" in decoded

    def test_chatml_template(self, tok):
        tmpl = forge_llm._ChatMLTemplate()
        ids = tmpl.apply(tok, [{"role": "user", "content": "Hi"}])
        # Verify it generates valid token sequence
        assert len(ids) > 0

    def test_chatml_with_generation_prompt(self, tok):
        tmpl = forge_llm._ChatMLTemplate()
        ids = tmpl.apply(tok, [{"role": "user", "content": "Hi"}], add_generation_prompt=True)
        assert len(ids) > 0

    def test_llama3_template(self, tok):
        tmpl = forge_llm._Llama3Template()
        ids = tmpl.apply(tok, [{"role": "user", "content": "Hello"}])
        assert len(ids) > 0

    def test_zephyr_template(self, tok):
        tmpl = forge_llm._ZephyrTemplate()
        ids = tmpl.apply(tok, [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ])
        assert len(ids) > 0

    def test_gemma4_template(self, tok):
        tmpl = forge_llm._Gemma4Template()
        ids = tmpl.apply(tok, [{"role": "user", "content": "Hi"}])
        assert len(ids) > 0

    def test_deepseek_template(self, tok):
        tmpl = forge_llm._DeepSeekTemplate()
        ids = tmpl.apply(tok, [{"role": "user", "content": "Hi"}])
        assert len(ids) > 0

    def test_phi3_template(self, tok):
        tmpl = forge_llm._Phi3Template()
        ids = tmpl.apply(tok, [{"role": "user", "content": "Hi"}])
        assert len(ids) > 0

    def test_llama2_template(self, tok):
        tmpl = forge_llm._Llama2Template()
        ids = tmpl.apply(tok, [{"role": "user", "content": "Hi"}])
        assert len(ids) > 0


# ====================================================================
#  LLM class tests
# ====================================================================

# Shared fixture for tests that need model+tokenizer+context
@pytest.fixture
def llm_with_ctx(model_path):
    model = forge.Model()
    model.load(model_path, arch_type="llama",
               vocab_size=100, hidden_dim=32, intermediate_dim=64,
               num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
               device="cpu")
    tok = forge.Tokenizer()
    # Use TinyLlama GGUF for tokenizer if available, otherwise skip
    gguf_path = os.path.join(os.path.dirname(__file__), "..", "models",
                             "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
    if not os.path.exists(gguf_path):
        pytest.skip("TinyLlama GGUF not found for tokenizer")
    tok.load_from_gguf(gguf_path)

    llm = forge_llm.LLM()
    llm._model = model
    llm._tokenizer = tok
    llm._ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
    llm._template = forge_llm.detect_chat_template(tok)
    llm._device = "cpu"
    return llm


class TestLLMConstruction:
    def test_llm_attributes(self, llm_with_ctx):
        llm = llm_with_ctx
        assert llm.tokenizer is not None
        assert llm.context is not None
        assert llm.eos_id() >= 0

    def test_generate_from_tokens(self, llm_with_ctx):
        result = llm_with_ctx.generate([1, 2, 3], max_new_tokens=2)
        assert isinstance(result, str)

    def test_chat_with_templates(self, llm_with_ctx):
        result = llm_with_ctx.chat([{"role": "user", "content": "Hi"}], max_new_tokens=2)
        assert isinstance(result, str)

    def test_generate_stream(self, llm_with_ctx):
        chunks = list(llm_with_ctx.generate_stream([1, 2, 3], max_new_tokens=2))
        assert len(chunks) >= 1


# ====================================================================
#  Session tests
# ====================================================================

class TestSession:
    @pytest.fixture
    def session(self, model_path):
        model = forge.Model()
        model.load(model_path, arch_type="llama",
                   vocab_size=100, hidden_dim=32, intermediate_dim=64,
                   num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                   device="cpu")
        gguf_path = os.path.join(os.path.dirname(__file__), "..", "models",
                                 "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
        if not os.path.exists(gguf_path):
            pytest.skip("TinyLlama GGUF not found")
        tok = forge.Tokenizer()
        tok.load_from_gguf(gguf_path)

        llm = forge_llm.LLM()
        llm._model = model
        llm._tokenizer = tok
        llm._ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        llm._template = forge_llm.detect_chat_template(tok)
        llm._device = "cpu"
        return llm.create_session(system_msg="You are helpful.")

    def test_session_creation(self, session):
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "system"

    def test_session_send(self, session):
        response = session.send("Hi there", max_new_tokens=2)
        assert isinstance(response, str)
        assert len(session.messages) == 3  # system + user + assistant

    def test_session_chat_stream(self, session):
        chunks = list(session.chat_stream("Hello", max_new_tokens=2))
        assert len(chunks) >= 1

    def test_session_reset(self, session):
        session.send("Hi", max_new_tokens=1)
        assert len(session.messages) >= 2
        session.reset()
        assert len(session.messages) == 0

    def test_session_fork(self, session):
        session.send("Hi", max_new_tokens=1)
        original_count = len(session.messages)
        forked = session.fork()
        assert len(forked.messages) == original_count

    def test_session_fork_independent(self, session):
        forked = session.fork(system_msg="Forked system")
        assert forked.messages[0]["content"] == "Forked system"
        assert session.messages[0]["content"] == "You are helpful."
