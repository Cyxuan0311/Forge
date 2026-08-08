"""Forge high-level LLM API — pure Python layer over forge bindings.

Provides LLM and Session classes with unified chat template handling,
modeled after llama-cpp-python's user experience.

Usage:
    llm = LLM.from_gguf("model.gguf")
    # Streaming generation
    for text in llm.generate_stream("What is AI?"):
        print(text, end="")
    # Multi-turn chat
    for text in llm.chat_stream([{"role": "user", "content": "Hello!"}]):
        print(text, end="")
"""

import os
import sys

# Ensure forge bindings are importable (from build/ directory)
_build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(_build_dir) and _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)

import forge  # noqa: E402


# ===========================================================================
#  Unified Chat Template Engine
# ===========================================================================


class _ChatTemplate:
    """Base class for chat templates."""

    name = "plain"

    def apply(self, tokenizer, messages, add_generation_prompt=True) -> list:
        """Convert messages to token IDs."""
        raise NotImplementedError

    @staticmethod
    def detect(tokenizer) -> "_ChatTemplate":
        """Auto-detect the appropriate chat template from tokenizer metadata."""
        template_str = tokenizer.chat_template

        # 1. Try detection from GGUF chat_template string
        if template_str:
            tl = template_str.lower()
            if "<|im_start|>" in tl or "chatml" in tl:
                return _ChatMLTemplate()
            if "<|start_header_id|>" in tl or "llama3" in tl or "llama 3" in tl:
                return _Llama3Template()
            if "<|user|>" in tl and "<|assistant|>" in tl:
                return _ZephyrTemplate()
            if "[inst]" in tl or "llama-2" in tl:
                return _Llama2Template()
            if "<|turn" in tl or "gemma" in tl:
                return _Gemma4Template()
            if "<｜User｜>" in tl or "deepseek" in tl:
                return _DeepSeekTemplate()
            if "phi" in tl or "<|user|" in tl:
                return _Phi3Template()

        # 2. Fallback: detect by model_type
        try:
            model_type = str(tokenizer.model_type).lower() if tokenizer.model_type else ""
        except Exception:
            model_type = ""
        if "qwen" in model_type:
            return _ChatMLTemplate()
        if "llama" in model_type:
            return _Llama3Template()
        if "gemma" in model_type:
            return _Gemma4Template()
        if "deepseek" in model_type:
            return _DeepSeekTemplate()
        if "phi" in model_type:
            return _Phi3Template()

        return _PlainTemplate()


class _ChatMLTemplate(_ChatTemplate):
    """ChatML format: <|im_start|>role\ncontent<|im_end|>\n (Qwen, MiniCPM, MiMo)."""

    name = "chatml"

    def apply(self, tokenizer, messages, add_generation_prompt=True):
        im_start = tokenizer.token_to_id("<|im_start|>")
        im_end = tokenizer.token_to_id("<|im_end|>")
        if im_start < 0 or im_end < 0:
            return _PlainTemplate().apply(tokenizer, messages, add_generation_prompt)

        ids = []
        for msg in messages:
            ids.append(im_start)
            ids.extend(tokenizer.encode(msg["role"] + "\n", add_bos=False))
            ids.extend(tokenizer.encode(msg["content"], add_bos=False))
            ids.append(im_end)
            ids.extend(tokenizer.encode("\n", add_bos=False))

        if add_generation_prompt:
            ids.append(im_start)
            ids.extend(tokenizer.encode("assistant\n", add_bos=False))

        return ids


class _Llama3Template(_ChatTemplate):
    """LLaMA 3.1 format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>."""

    name = "llama3"

    def apply(self, tokenizer, messages, add_generation_prompt=True):
        start_h = tokenizer.token_to_id("<|start_header_id|>")
        end_h = tokenizer.token_to_id("<|end_header_id|>")
        eot = tokenizer.token_to_id("<|eot_id|>")

        ids = []
        bos = tokenizer.bos_token_id
        if bos >= 0:
            ids.append(bos)

        for msg in messages:
            ids.append(start_h)
            ids.extend(tokenizer.encode(msg["role"], add_bos=False))
            ids.append(end_h)
            ids.extend(tokenizer.encode("\n\n", add_bos=False))
            ids.extend(tokenizer.encode(msg["content"], add_bos=False))
            ids.append(eot)

        if add_generation_prompt:
            ids.append(start_h)
            ids.extend(tokenizer.encode("assistant", add_bos=False))
            ids.append(end_h)
            ids.extend(tokenizer.encode("\n\n", add_bos=False))

        return ids


class _ZephyrTemplate(_ChatTemplate):
    """Zephyr format: <|user|>\n...<eos>\n<|assistant|>\n...<eos>\n (TinyLlama)."""

    name = "zephyr"

    def apply(self, tokenizer, messages, add_generation_prompt=True):
        ids = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                ids.extend(tokenizer.encode("<|system|>\n", add_bos=False))
                ids.extend(tokenizer.encode(msg["content"], add_bos=False, add_dummy_prefix=False))
                ids.append(tokenizer.eos_token_id)
                ids.extend(tokenizer.encode("\n", add_bos=False, add_dummy_prefix=False))
            elif role == "user":
                ids.extend(tokenizer.encode("<|user|>\n", add_bos=False))
                ids.extend(tokenizer.encode(msg["content"], add_bos=False, add_dummy_prefix=False))
                ids.append(tokenizer.eos_token_id)
                ids.extend(tokenizer.encode("\n", add_bos=False, add_dummy_prefix=False))
            elif role == "assistant":
                ids.extend(
                    tokenizer.encode("<|assistant|>\n", add_bos=False, add_dummy_prefix=False)
                )
                ids.extend(tokenizer.encode(msg["content"], add_bos=False, add_dummy_prefix=False))
                ids.append(tokenizer.eos_token_id)
                ids.extend(tokenizer.encode("\n", add_bos=False, add_dummy_prefix=False))

        if add_generation_prompt:
            ids.extend(tokenizer.encode("<|assistant|>\n", add_bos=False, add_dummy_prefix=False))

        return ids


class _Llama2Template(_ChatTemplate):
    """Llama-2 format: [INST] ... [/INST] ..."""

    name = "llama2"

    def apply(self, tokenizer, messages, add_generation_prompt=True):
        ids = []
        for msg in messages:
            if msg["role"] == "user":
                ids.append(tokenizer.bos_token_id)
                ids.extend(tokenizer.encode("[INST] ", add_bos=False))
                ids.extend(tokenizer.encode(msg["content"], add_bos=False, add_dummy_prefix=False))
                ids.extend(tokenizer.encode(" [/INST]", add_bos=False))
            elif msg["role"] == "assistant":
                ids.extend(tokenizer.encode(" ", add_bos=False, add_dummy_prefix=False))
                ids.extend(tokenizer.encode(msg["content"], add_bos=False, add_dummy_prefix=False))
                ids.append(tokenizer.eos_token_id)

        return ids


class _Gemma4Template(_ChatTemplate):
    """Gemma 4 format: <|turn>role\ncontent<turn|>\n with optional thinking."""

    name = "gemma4"

    def __init__(self, enable_thinking=False):
        self._thinking = enable_thinking

    def apply(self, tokenizer, messages, add_generation_prompt=True):
        bos = tokenizer.bos_token_id
        turn_start = tokenizer.token_to_id("<|turn>")
        turn_end = tokenizer.token_to_id("<turn|>")
        newline = 108  # Gemma4 newline token

        user_id = tokenizer.token_to_id("user")
        model_id = tokenizer.token_to_id("model")
        system_id = tokenizer.token_to_id("system")

        ids = []
        if bos >= 0:
            ids.append(bos)

        if self._thinking:
            think_id = tokenizer.token_to_id("<|think|>")
            if think_id is not None and think_id >= 0:
                ids.append(turn_start)
                ids.append(system_id)
                ids.append(newline)
                ids.append(think_id)
                ids.append(newline)
                ids.append(turn_end)
                ids.append(newline)

        for msg in messages:
            role = msg["role"]
            if role == "system":
                role_id = system_id
            elif role == "user":
                role_id = user_id
            elif role == "assistant":
                role_id = model_id
            else:
                role_id = user_id

            ids.append(turn_start)
            ids.append(role_id)
            ids.append(newline)
            ids.extend(tokenizer.encode(msg["content"], add_bos=False))
            ids.append(turn_end)
            ids.append(newline)

        if add_generation_prompt:
            ids.append(turn_start)
            ids.append(model_id)
            ids.append(newline)

        return ids


class _DeepSeekTemplate(_ChatTemplate):
    """DeepSeek-R1 format: <｜User｜>content<｜Assistant｜>content"""

    name = "deepseek"

    def apply(self, tokenizer, messages, add_generation_prompt=True):
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        user_id = tokenizer.token_to_id("<｜User｜>")
        asst_id = tokenizer.token_to_id("<｜Assistant｜>")

        ids = []
        if bos >= 0:
            ids.append(bos)

        for msg in messages:
            if msg["role"] == "system":
                ids.extend(tokenizer.encode(msg["content"], add_bos=False))
            elif msg["role"] == "user":
                ids.append(user_id)
                ids.extend(tokenizer.encode(msg["content"], add_bos=False))
            elif msg["role"] == "assistant":
                ids.append(asst_id)
                ids.extend(tokenizer.encode(msg["content"], add_bos=False))
                ids.append(eos)

        if add_generation_prompt:
            ids.append(asst_id)

        return ids


class _Phi3Template(_ChatTemplate):
    """Phi-3 Mini format: <|user|>\ncontent<|end|>\n<|assistant|>\ncontent<|end|>\n"""

    name = "phi3"

    def apply(self, tokenizer, messages, add_generation_prompt=True):
        ids = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            ids.extend(tokenizer.encode(f"<|{role}|>\n", add_bos=False))
            ids.extend(tokenizer.encode(content, add_bos=False))
            ids.extend(tokenizer.encode("<|end|>\n", add_bos=False))

        if add_generation_prompt:
            ids.extend(tokenizer.encode("<|assistant|>\n", add_bos=False))

        return ids


class _PlainTemplate(_ChatTemplate):
    """Plain fallback: role: content\\n"""

    name = "plain"

    def apply(self, tokenizer, messages, add_generation_prompt=True):
        ids = []
        bos = tokenizer.bos_token_id
        if bos >= 0:
            ids.append(bos)
        for msg in messages:
            ids.extend(tokenizer.encode(f"{msg['role']}: {msg['content']}\n", add_bos=False))
        if add_generation_prompt:
            ids.extend(tokenizer.encode("assistant: ", add_bos=False))
        return ids


# Registry of all supported templates
_TEMPLATES = {
    "chatml": _ChatMLTemplate,
    "llama3": _Llama3Template,
    "zephyr": _ZephyrTemplate,
    "llama2": _Llama2Template,
    "gemma4": _Gemma4Template,
    "deepseek": _DeepSeekTemplate,
    "phi3": _Phi3Template,
    "plain": _PlainTemplate,
}


def detect_chat_template(tokenizer, template_name=None) -> _ChatTemplate:
    """Detect or select a chat template.

    Args:
        tokenizer: forge.Tokenizer instance (may be None for plain fallback).
        template_name: Optional explicit template name (e.g. "chatml", "llama3").
                       If None, auto-detection is used.

    Returns:
        A _ChatTemplate instance.
    """
    if template_name and template_name in _TEMPLATES:
        return _TEMPLATES[template_name]()
    if tokenizer is None:
        return _PlainTemplate()
    return _ChatTemplate.detect(tokenizer)


# ===========================================================================
#  LLM — High-level model interface
# ===========================================================================


class LLM:
    """High-level interface for loading a model and generating text.

    Example:
        llm = LLM.from_gguf("model.gguf")
        print(llm.generate("What is AI?"))
        llm.chat_stream([{"role": "user", "content": "Hello"}])
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._ctx = None
        self._template = None
        self._device = "cpu"

    # ---- Constructors ----

    @classmethod
    def from_gguf(
        cls,
        model_path,
        device="cuda",
        n_gpu_layers=-1,
        kv_dtype="fp16",
        template_name=None,
        **kwargs,
    ):
        """Create an LLM from a GGUF model file.

        Args:
            model_path: Path to .gguf file.
            device: "cuda" or "cpu".
            n_gpu_layers: Number of layers on GPU (-1 = all).
            kv_dtype: KV cache dtype ("fp32", "fp16", "q8_0", "q4_0", "q4_k").
            template_name: Optional chat template name override.
            **kwargs: Additional args passed to create_context.

        Returns:
            LLM instance ready for generation.
        """
        llm = cls()
        llm._device = device

        # Load model
        llm._model = forge.Model()
        llm._model.load_gguf(model_path, device=device)

        # Load tokenizer
        llm._tokenizer = forge.Tokenizer()
        llm._tokenizer.load_from_gguf(model_path)

        # Create context
        llm._ctx = llm._model.create_context(kv_cache_dtype=kv_dtype, gpu_layers=n_gpu_layers)

        # Detect chat template
        llm._template = detect_chat_template(llm._tokenizer, template_name)

        return llm

    # ---- Properties ----

    @property
    def model(self) -> forge.Model:
        return self._model

    @property
    def tokenizer(self) -> forge.Tokenizer:
        return self._tokenizer

    @property
    def context(self) -> forge.InferenceContext:
        return self._ctx

    @property
    def template(self) -> _ChatTemplate:
        return self._template

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    # ---- Generation ----

    def _to_tokens(self, text_or_tokens):
        """Convert text or token list to token IDs."""
        if isinstance(text_or_tokens, str):
            return self._tokenizer.encode(text_or_tokens, add_bos=True)
        return list(text_or_tokens)

    def _make_config(
        self,
        max_new_tokens=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.1,
        stop_token_ids=None,
        **kwargs,
    ):
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        cfg.temperature = temperature
        cfg.top_k = top_k
        cfg.top_p = top_p
        cfg.repeat_penalty = repeat_penalty
        cfg.do_sample = temperature > 0
        if self._tokenizer:
            cfg.eos_token_id = self._tokenizer.eos_token_id
        if stop_token_ids:
            cfg.stop_token_ids = list(stop_token_ids)
        return cfg

    def generate(
        self,
        prompt,
        max_new_tokens=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.1,
        stop_token_ids=None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt (non-streaming).

        Args:
            prompt: Text string or list of token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k, top_p: Sampling parameters.
            repeat_penalty: Repetition penalty.
            stop_token_ids: Additional stop token IDs.
            **kwargs: Extra arguments passed to GenerationConfig.

        Returns:
            Generated text string.
        """
        tokens = self._to_tokens(prompt)
        cfg = self._make_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop_token_ids=stop_token_ids,
            **kwargs,
        )
        result = self._ctx.generate_kv(tokens, cfg)
        return self._tokenizer.decode(list(result.token_ids), skip_special=True)

    def generate_stream(
        self,
        prompt,
        max_new_tokens=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.1,
        stop_token_ids=None,
        **kwargs,
    ):
        """Streaming generate text from a prompt.

        Yields:
            Decoded text fragments as they become available.
        """
        tokens = self._to_tokens(prompt)
        cfg = self._make_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop_token_ids=stop_token_ids,
            **kwargs,
        )

        token_buffer = []

        def on_token(tid, step):
            token_buffer.append(tid)
            if len(token_buffer) >= 4:
                try:
                    text = self._tokenizer.decode(token_buffer, skip_special=True)
                    token_buffer.clear()
                    return text
                except UnicodeDecodeError:
                    pass
            return None

        texts = []

        def callback(tid, step):
            token_buffer.append(tid)
            if len(token_buffer) >= 4 or tid == cfg.eos_token_id:
                try:
                    text = self._tokenizer.decode(token_buffer, skip_special=True)
                    token_buffer.clear()
                    texts.append(text)
                except UnicodeDecodeError:
                    pass

        self._ctx.generate_stream_kv(tokens, cfg, callback)

        # Decode any remaining tokens
        if token_buffer:
            try:
                text = self._tokenizer.decode(token_buffer, skip_special=True)
                token_buffer.clear()
                texts.append(text)
            except UnicodeDecodeError:
                pass

        # Yield accumulated texts
        for text in texts:
            yield text

    def _apply_chat_template(self, messages):
        """Apply chat template to messages, return token IDs."""
        return self._template.apply(self._tokenizer, messages, add_generation_prompt=True)

    def chat(
        self,
        messages,
        max_new_tokens=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.1,
        stop_token_ids=None,
        **kwargs,
    ) -> str:
        """Multi-turn chat (non-streaming).

        Args:
            messages: List of {"role": str, "content": str} dicts.
            max_new_tokens, temperature, top_k, top_p, repeat_penalty: Generation params.
            stop_token_ids: Additional stop token IDs.
            **kwargs: Extra arguments passed to GenerationConfig.

        Returns:
            Assistant response text.
        """
        tokens = self._apply_chat_template(messages)
        return self.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop_token_ids=stop_token_ids,
            **kwargs,
        )

    def chat_stream(
        self,
        messages,
        max_new_tokens=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.1,
        stop_token_ids=None,
        **kwargs,
    ):
        """Multi-turn streaming chat.

        Yields:
            Decoded text fragments.
        """
        tokens = self._apply_chat_template(messages)
        yield from self.generate_stream(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop_token_ids=stop_token_ids,
            **kwargs,
        )

    def create_session(self, system_msg=None, **kwargs) -> "Session":
        """Create a stateful chat session.

        Args:
            system_msg: Optional system message string.
            **kwargs: Extra arguments for session-level generation defaults.

        Returns:
            Session instance.
        """
        return Session(self, system_msg=system_msg, **kwargs)

    def reset(self):
        """Reset KV cache (start a fresh conversation)."""
        if self._ctx:
            self._ctx.reset_kv()

    def eos_id(self) -> int:
        """Return the EOS token ID."""
        if self._tokenizer:
            return self._tokenizer.eos_token_id
        return -1


# ===========================================================================
#  Session — Stateful multi-turn conversation
# ===========================================================================


class Session:
    """Stateful multi-turn chat session with KV cache persistence.

    In paged KV mode (FORGE_KV_STORAGE_MODE=paged), fork() shares the
    KV prefix pages between parent and child via zero-copy seq_share,
    avoiding redundant KV computation.

    Example:
        session = llm.create_session(system_msg="You are a helpful assistant.")
        for text in session.chat_stream("Hello!"):
            print(text, end="")
        for text in session.chat_stream("How are you?"):
            print(text, end="")
        session.reset()  # Clear conversation
    """

    _next_seq_id = 1  # seq_id 0 reserved for default generation

    def __init__(self, llm: LLM, system_msg=None, **kwargs):
        self._llm = llm
        self._messages = []
        self._gen_defaults = kwargs
        self._prefix_token_count = 0  # template tokens fed to KV so far
        self._seq_id = Session._next_seq_id
        Session._next_seq_id += 1

        if system_msg:
            self._messages.append({"role": "system", "content": system_msg})

    @property
    def seq_id(self) -> int:
        return self._seq_id

    @property
    def messages(self) -> list:
        return list(self._messages)

    @property
    def llm(self) -> LLM:
        return self._llm

    @property
    def prefix_token_count(self) -> int:
        """Number of template tokens that have been filled into KV cache."""
        return self._prefix_token_count

    def _apply_template(self, add_generation_prompt=True) -> list:
        return self._llm.template.apply(
            self._llm.tokenizer, self._messages, add_generation_prompt=add_generation_prompt
        )

    def send(
        self,
        message,
        role="user",
        max_new_tokens=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.1,
        **kwargs,
    ) -> str:
        """Send a message and get the response (non-streaming).

        Args:
            message: Text content of the message.
            role: Role of the message ("user" or "system").
            **kwargs: Override default generation parameters.

        Returns:
            Assistant response text.
        """
        self._messages.append({"role": role, "content": message})
        tokens = self._apply_template()
        cfg = self._llm._make_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            **kwargs,
        )
        result = self._llm.context.generate_kv(tokens, cfg)
        response = self._llm.tokenizer.decode(list(result.token_ids), skip_special=True)
        self._messages.append({"role": "assistant", "content": response})
        self._prefix_token_count = len(tokens) + result.num_generated_tokens
        return response

    def chat(self, message, **kwargs) -> str:
        """Alias for send()."""
        return self.send(message, **kwargs)

    def chat_stream(
        self,
        message,
        role="user",
        max_new_tokens=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.1,
        **kwargs,
    ):
        """Send a message and stream the response.

        Yields:
            Decoded text fragments.
        """
        self._messages.append({"role": role, "content": message})
        tokens = self._apply_template()

        cfg = self._llm._make_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            **kwargs,
        )

        token_buffer = []
        full_response = []

        def callback(tid, step):
            token_buffer.append(tid)
            full_response.append(tid)
            if len(token_buffer) >= 4 or tid == cfg.eos_token_id:
                try:
                    self._llm.tokenizer.decode(token_buffer, skip_special=True)
                    token_buffer.clear()
                except UnicodeDecodeError:
                    pass

        self._llm.context.generate_stream_kv(tokens, cfg, callback)

        # Update prefix count
        self._prefix_token_count = len(tokens) + len(full_response)

        # Decode the full response
        if full_response:
            response_text = self._llm.tokenizer.decode(full_response, skip_special=True)
            self._messages.append({"role": "assistant", "content": response_text})
            yield response_text
        else:
            self._messages.append({"role": "assistant", "content": ""})
            yield ""

    def reset(self):
        """Reset session: clear messages, release KV pages, reset KV cache."""
        self._messages.clear()
        self._prefix_token_count = 0
        ctx = self._llm.context
        if ctx is not None:
            if ctx.is_paged():
                ctx.seq_release(self._seq_id)
            ctx.reset_kv()

    def fork(self, system_msg=None) -> "Session":
        """Create a forked session sharing the KV prefix with this parent.

        In paged KV mode, uses zero-copy seq_share to transfer prefix pages
        from the parent to the child session, avoiding redundant computation.

        In contiguous mode, falls back to message deep-copy only (the child
        will recompute the prefix on first generation).

        Args:
            system_msg: Optional system message override for the forked session.

        Returns:
            A new Session instance with identical message history.
        """
        ctx = self._llm.context

        # Deep copy messages
        new_msgs = [dict(m) for m in self._messages]
        if system_msg and new_msgs:
            if new_msgs[0].get("role") == "system":
                new_msgs[0]["content"] = system_msg
            else:
                new_msgs.insert(0, {"role": "system", "content": system_msg})

        # In paged mode, share prefix KV pages from parent to child seq_id
        if ctx is not None and ctx.is_paged() and self._prefix_token_count > 0:
            child_seq = Session._next_seq_id
            Session._next_seq_id += 1
            ok = ctx.seq_share(self._seq_id, child_seq, 0, self._prefix_token_count)
            if not ok:
                # Fallback: seq_share failed — child will start fresh
                child_seq = Session._next_seq_id
                Session._next_seq_id += 1

            new_session = Session.__new__(Session)
            new_session._llm = self._llm
            new_session._messages = new_msgs
            new_session._gen_defaults = dict(self._gen_defaults)
            new_session._prefix_token_count = self._prefix_token_count
            new_session._seq_id = child_seq

            # Keep parent's pages alive (increments refcount)
            ctx.seq_keep(self._seq_id)
            return new_session

        # Contiguous / no-prefix fallback
        new_session = Session.__new__(Session)
        new_session._llm = self._llm
        new_session._messages = new_msgs
        new_session._gen_defaults = dict(self._gen_defaults)
        new_session._prefix_token_count = 0
        new_session._seq_id = Session._next_seq_id
        Session._next_seq_id += 1
        return new_session


# ===========================================================================
#  VisionLLM — Multimodal (text + image) support
# ===========================================================================


class VisionLLM:
    """High-level multimodal LLM supporting text + image input.

    Wraps forge.MultimodalModel with the same ergonomics as LLM:
      - from_gguf() for loading with vision projector
      - generate_stream() / chat_stream() with image embedding injection
      - create_session() for stateful multi-turn

    Example:
        vllm = VisionLLM.from_gguf("model.gguf", mmproj="mmproj.gguf")
        # Text-only
        for text in vllm.generate_stream("Hello!"):
            print(text, end="")
        # With image
        import numpy as np
        img = np.zeros((224, 224, 3), dtype=np.uint8)  # placeholder
        for text in vllm.generate_stream("Describe:", image=img):
            print(text, end="")
    """

    def __init__(
        self, model_path, mmproj_path=None, device="cuda", kv_cache_dtype="fp32", gpu_layers=-1
    ):
        self._mm = forge.MultimodalModel()
        if mmproj_path:
            self._mm.load_with_mmproj(model_path, mmproj_path, device)
        else:
            self._mm.load(model_path, device)

        self.tokenizer = forge.Tokenizer()
        self.tokenizer.load_from_gguf(model_path)
        self._kv_dtype = kv_cache_dtype
        self._gpu_layers = gpu_layers
        self._ctx = None
        self._template = _ChatTemplate.detect(self.tokenizer)

    @classmethod
    def from_gguf(
        cls, model_path, mmproj_path=None, device="cuda", kv_cache_dtype="fp32", gpu_layers=-1
    ) -> "VisionLLM":
        """Load a multimodal model from GGUF file(s).

        Args:
            model_path: Path to LLM GGUF file.
            mmproj_path: Path to vision projector (mmproj) GGUF file.
            device: "cuda" or "cpu".
            kv_cache_dtype: KV cache data type.
            gpu_layers: Number of layers on GPU (-1 = all).

        Returns:
            VisionLLM instance.
        """
        return cls(model_path, mmproj_path, device, kv_cache_dtype, gpu_layers)

    @property
    def context(self):
        """Lazily-created InferenceContext (shared across calls)."""
        if self._ctx is None:
            self._ctx = self._mm.create_context(self._kv_dtype, self._gpu_layers)
        return self._ctx

    @property
    def mm_model(self):
        return self._mm

    @property
    def template(self):
        return self._template

    def encode_image(self, image):
        """Encode image to embedding tensor.

        Args:
            image: numpy array (H, W, 3) uint8 or PIL Image.

        Returns:
            Image embeddings as numpy array.
        """
        import numpy as np

        if hasattr(image, "convert"):  # PIL Image
            img_array = np.array(image.convert("RGB"))
        else:
            img_array = image

        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        img_array = img_array.astype(np.uint8)
        return self._mm.encode_image(img_array)

    def _inject_image_embeddings(
        self, ctx, prompt_ids, image_embeddings, img_insert_pos=None, num_img_tokens=None
    ):
        """Inject image embeddings into prompt embeddings at the right position."""
        import numpy as np

        if img_insert_pos is None:
            # Default: insert after the first <image> placeholder or at position 1
            img_insert_pos = 1

        if num_img_tokens is None:
            num_img_tokens = image_embeddings.shape[0]

        prompt_array = np.array(prompt_ids, dtype=np.int32)
        text_embeddings = np.array(ctx.get_embeddings(prompt_array))

        if text_embeddings.shape[0] >= img_insert_pos + num_img_tokens:
            text_embeddings[img_insert_pos : img_insert_pos + num_img_tokens] = image_embeddings
        return text_embeddings

    def generate_stream(
        self,
        prompt,
        image=None,
        max_new_tokens=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.1,
        **kwargs,
    ):
        """Streaming text generation with optional image input.

        Args:
            prompt: Text prompt.
            image: Optional numpy array (H, W, 3) uint8 or PIL Image.
            **kwargs: Additional generation parameters.

        Yields:
            Decoded text fragments.
        """
        has_image = image is not None
        image_embeddings = None
        if has_image:
            image_embeddings = self.encode_image(image)

        # Build prompt tokens
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = self._template.apply(self.tokenizer, messages)

        ctx = self.context
        if has_image and image_embeddings is not None:
            combined = self._inject_image_embeddings(ctx, prompt_ids, image_embeddings)
            logits = ctx.forward_with_embeddings(combined, 0)
        else:
            prompt_array = __import__("numpy").array(prompt_ids, dtype=__import__("numpy").int32)
            logits = ctx.forward(prompt_array, 0)

        # Manual decode loop (multimodal can't use Generator directly due to embedding injection)
        import numpy as np

        start_pos = len(prompt_ids)
        generated_ids = []
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        cfg.temperature = temperature
        cfg.top_k = top_k
        cfg.top_p = top_p
        cfg.repeat_penalty = repeat_penalty

        token_buffer = []
        for _ in range(max_new_tokens):
            # Sample from logits
            last_logits = np.array(logits[-1, :]) if logits.shape[0] > 1 else np.array(logits[0, :])
            tid = _greedy_sample(last_logits, cfg)
            if tid == cfg.eos_token_id:
                break

            generated_ids.append(tid)
            token_buffer.append(tid)
            if len(token_buffer) >= 4:
                try:
                    yield self.tokenizer.decode(token_buffer, skip_special=True)
                    token_buffer.clear()
                except UnicodeDecodeError:
                    pass

            next_arr = np.array([tid], dtype=np.int32)
            logits = ctx.forward(next_arr, start_pos)
            start_pos += 1

        if token_buffer:
            try:
                yield self.tokenizer.decode(token_buffer, skip_special=True)
            except UnicodeDecodeError:
                pass

    def generate(self, prompt, image=None, max_new_tokens=256, **kwargs) -> str:
        """Non-streaming generation with optional image."""
        result = ""
        for text in self.generate_stream(prompt, image, max_new_tokens, **kwargs):
            result += text
        return result

    def chat_stream(self, messages, image=None, **kwargs):
        """Multi-turn chat with optional image.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            image: Optional image for the CURRENT turn.

        Yields:
            Decoded text fragments.
        """
        prompt_ids = self._template.apply(self.tokenizer, messages)
        # Encode full prompt tokens as prompt string (template handles formatting)
        prompt = self.tokenizer.decode(prompt_ids, skip_special=True)
        yield from self.generate_stream(prompt, image=image, **kwargs)

    def create_session(self, system_msg=None, **kwargs):
        """Create a stateful VisionSession for multi-turn chat."""
        return VisionSession(self, system_msg=system_msg, **kwargs)

    def reset(self):
        """Reset KV cache."""
        if self._ctx:
            self._ctx.reset_kv()


class VisionSession:
    """Stateful multi-turn multimodal chat session."""

    def __init__(self, vllm: VisionLLM, system_msg=None, **kwargs):
        self._vllm = vllm
        self._messages = []
        self._gen_defaults = kwargs
        if system_msg:
            self._messages.append({"role": "system", "content": system_msg})

    @property
    def messages(self) -> list:
        return list(self._messages)

    def chat_stream(self, message, image=None, role="user", **kwargs):
        """Send a message with optional image and stream response."""
        self._messages.append({"role": role, "content": message})
        full = ""
        for text in self._vllm.chat_stream(self._messages, image=image, **kwargs):
            full += text
            yield text
        self._messages.append({"role": "assistant", "content": full})

    def send(self, message, image=None, **kwargs) -> str:
        """Send a message with optional image, return full response."""
        result = ""
        for text in self.chat_stream(message, image=image, **kwargs):
            result += text
        return result

    def chat(self, message, image=None, **kwargs) -> str:
        """Alias for send()."""
        return self.send(message, image=image, **kwargs)

    def reset(self):
        """Reset session: clear messages and KV cache."""
        self._messages.clear()
        self._vllm.reset()


def _greedy_sample(logits, cfg):
    """Greedy sample (or temperature-based if temperature > 0)."""
    import numpy as np

    if cfg.temperature > 0:
        logits = logits / cfg.temperature
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        return int(np.random.choice(len(probs), p=probs))
    else:
        return int(np.argmax(logits))
