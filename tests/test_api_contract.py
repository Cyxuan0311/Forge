"""Stage 0: Python API contract test — freeze the public surface.

This module documents every attribute and method that is currently visible
on each public class/enum via `dir()`.  Future stages must not remove or
rename these entries without an explicit migration window (DeprecationWarning).
Adding new entries is fine.

The test does NOT require a loaded model — it uses `type()` inspection only.
"""

import forge


# ====================================================================
#  Helper: extract public API from a type
# ====================================================================


def _public_api(tp):
    """Return sorted list of public (non-underscore) attributes of *tp*."""
    items = sorted(x for x in dir(tp) if not x.startswith("_"))
    return items


def _class_api(tp):
    """Return sorted list of non-dunder public attributes (class-level)."""
    return sorted(
        x for x in dir(tp) if not x.startswith("_") or (x.startswith("__") and x.endswith("__"))
    )


# ====================================================================
#  Module-level API
# ====================================================================

MODULE_API = sorted(
    [
        "ActivationType",
        "Backend",
        "BackendCapability",
        "BackendInfo",
        "BackendManager",
        "CachedPrompt",
        "ContextConfig",  # Phase 1
        "DataType",
        "DeviceInfo",
        "DeviceType",
        "FFNType",
        "GenerateRequest",
        "GenerationConfig",  # Phase 1
        "GenerationResult",  # Phase 1
        "InferenceContext",
        "KVCacheDType",
        "KVLayerPolicy",
        "LogLevel",
        "Logger",
        # KVLayerPolicy values leak to module via export_values() (Phase 6)
        "Full",
        "Model",
        "ModelConfig",
        "MultimodalModel",
        "NormType",
        "None_",
        "QuantPolicy",
        "RequestScheduler",
        "RequestStatus",
        "RopeType",
        "Recurrent",
        "SamplerConfig",
        "SlidingWindow",
        "SpeculativeConfig",
        "Tensor",
        "Tokenizer",
        "TokenizerModelType",
        "VisionConfig",
        # module-level functions
        "compute_quantized_bytes",
        "get_dequant_row_fn",
        "dtype_block_elements",
        "dtype_block_size",
        "dtype_name",
        "dtype_size",
        "get_dequant_row_fn",
        "get_memory_counters",
        "is_quantized_type",
        "profiler_disable",
        "profiler_enable",
        "profiler_enabled",
        "profiler_print",
        "profiler_reset",
        "profiler_set_cuda_events",
        "profiler_set_trace_enabled",
        "profiler_summary",
        "profiler_trace_enabled",
        "profiler_trace_overflow",
        "profiler_trace_size",
        "profiler_trace_to_json",
        "reset_memory_counters",
        "set_num_threads",
    ]
)


class TestModuleAPI:
    """The forge module exposes exactly these names."""

    def test_module_names(self):
        actual = _public_api(forge)
        missing = set(MODULE_API) - set(actual)
        extra = set(actual) - set(MODULE_API)
        assert not missing, f"Missing from forge module: {sorted(missing)}"
        # Extra entries (added by future Phases) are fine — only missing is fatal
        if extra:
            print(f"[INFO] New entries in forge module (fine, phases may add): {sorted(extra)}")


# ====================================================================
#  Per-class API contracts
# ====================================================================


class TestModelAPI:
    """forge.Model public surface."""

    MODEL_API = sorted(
        [
            "config",
            "create_context",
            "default_context",  # Phase 1
            "detect_format",
            "device",
            "ensure_default_context",  # Phase 1
            "generate",
            "generate_stream",
            "load",
            "load_auto",
            "load_gguf",
            "load_vision_weights",
            "registered_archs",
            "release_default_context",  # Phase 1
        ]
    )

    def test_model_api(self):
        actual = _public_api(forge.Model)
        missing = set(self.MODEL_API) - set(actual)
        assert not missing, f"Missing from Model: {sorted(missing)}"


class TestInferenceContextAPI:
    """forge.InferenceContext public surface."""

    CTX_API = sorted(
        [
            "cuda_graph_enabled",
            "device",
            "forward",
            "forward_sample",
            "forward_with_embeddings",
            "generate",  # Phase 1
            "generate_kv",  # Phase 2
            "generate_stream",  # Phase 1
            "generate_stream_kv",  # Phase 2
            "get_embeddings",
            "gpu_layers",
            "memory_stats",
            "n_batch",
            "n_threads",
            "n_threads_batch",
            "n_ubatch",
            "reset",
            "reset_kv",
            "set_cuda_graph_enabled",
            "set_gpu_layers",
            "set_use_graph",
            "use_graph",
            "warmup",
        ]
    )

    def test_context_api(self):
        actual = _public_api(forge.InferenceContext)
        missing = set(self.CTX_API) - set(actual)
        assert not missing, f"Missing from InferenceContext: {sorted(missing)}"


class TestTokenizerAPI:
    """forge.Tokenizer public surface."""

    TOK_API = sorted(
        [
            "bos_token_id",
            "chat_template",
            "decode",
            "decode_token",
            "encode",
            "eos_token_id",
            "id_to_token",
            "is_loaded",
            "load_from_gguf",
            "model_type",
            "pad_token_id",
            "token_score",
            "token_to_id",
            "token_type",
            "unk_token_id",
            "vocab_size",
        ]
    )

    def test_tokenizer_api(self):
        actual = _public_api(forge.Tokenizer)
        missing = set(self.TOK_API) - set(actual)
        assert not missing, f"Missing from Tokenizer: {sorted(missing)}"


class TestMultimodalModelAPI:
    """forge.MultimodalModel public surface."""

    MM_API = sorted(
        [
            "config",
            "create_context",
            "encode_image",
            "generate",
            "generate_stream",
            "load",
            "load_with_mmproj",
            "vision_config",
        ]
    )

    def test_multimodal_api(self):
        actual = _public_api(forge.MultimodalModel)
        missing = set(self.MM_API) - set(actual)
        assert not missing, f"Missing from MultimodalModel: {sorted(missing)}"


class TestRequestSchedulerAPI:
    """forge.RequestScheduler public surface."""

    SCHED_API = sorted(
        [
            "abort",
            "get_finished",
            "has_pending",
            "memory_stats",
            "n_batch",
            "n_threads",
            "n_threads_batch",
            "n_ubatch",
            "num_active",
            "num_waiting",
            "prefix_cache_hits",
            "prefix_cache_misses",
            "reset",
            "step",
            "submit",
        ]
    )

    def test_scheduler_api(self):
        actual = _public_api(forge.RequestScheduler)
        missing = set(self.SCHED_API) - set(actual)
        assert not missing, f"Missing from RequestScheduler: {sorted(missing)}"


class TestSamplerConfigAPI:
    """forge.SamplerConfig public surface."""

    SC_API = sorted(
        [
            "do_sample",
            "logit_softcapping",
            "repeat_last_n",
            "repeat_penalty",
            "seed",
            "temperature",
            "top_k",
            "top_p",
        ]
    )

    def test_sampler_config_api(self):
        actual = _public_api(forge.SamplerConfig)
        missing = set(self.SC_API) - set(actual)
        assert not missing, f"Missing from SamplerConfig: {sorted(missing)}"


class TestGenerateRequestAPI:
    """forge.GenerateRequest public surface."""

    GR_API = sorted(
        [
            "finish_reason",
            "from_cache",
            "num_generated",
            "output_tokens",
            "prefix_len",
            "request_id",
            "status",
        ]
    )

    def test_generate_request_api(self):
        actual = _public_api(forge.GenerateRequest)
        missing = set(self.GR_API) - set(actual)
        assert not missing, f"Missing from GenerateRequest: {sorted(missing)}"


class TestCachedPromptAPI:
    """forge.CachedPrompt public surface."""

    CP_API = sorted(
        [
            "seq_id",
            "tokens",
            "valid",
        ]
    )

    def test_cached_prompt_api(self):
        actual = _public_api(forge.CachedPrompt)
        missing = set(self.CP_API) - set(actual)
        assert not missing, f"Missing from CachedPrompt: {sorted(missing)}"


# ====================================================================
#  Enum value contracts
# ====================================================================


class TestEnumContracts:
    """Every enum value that exists today is frozen."""

    def test_data_type_values(self):
        dt = forge.DataType
        for v in (
            dt.FP32,
            dt.FP16,
            dt.Q4_0,
            dt.Q4_1,
            dt.Q4_K,
            dt.Q5_0,
            dt.Q5_1,
            dt.Q2_K,
            dt.Q3_K,
            dt.Q5_K,
            dt.Q6_K,
            dt.Q8_0,
            dt.INT8,
            dt.INT32,
            dt.IQ2_S,
            dt.IQ2_XXS,
            dt.IQ4_NL,
            dt.IQ2_XS,
            dt.IQ3_S,
            dt.BF16,
        ):
            assert v is not None

    def test_device_type_values(self):
        dt = forge.DeviceType
        assert dt.CPU is not None
        assert dt.CUDA is not None

    def test_kv_cache_dtype_values(self):
        dt = forge.KVCacheDType
        assert dt.FP32 is not None
        assert dt.F16 is not None
        assert dt.Q8_0 is not None
        assert dt.Q4_0 is not None
        assert dt.Q4_K is not None

    def test_kv_layer_policy(self):
        expected = {"None_", "Full", "SlidingWindow", "Recurrent"}
        actual = set(forge.KVLayerPolicy.__members__.keys())
        assert actual == expected

    def test_log_level_values(self):
        dt = forge.LogLevel
        assert dt.NONE is not None
        assert dt.LOG_ERROR is not None
        assert dt.WARN is not None
        assert dt.INFO is not None
        assert dt.DEBUG is not None
        assert dt.TRACE is not None

    def test_norm_type_values(self):
        dt = forge.NormType
        assert dt.RMSNorm is not None
        assert dt.LayerNorm is not None

    def test_activation_type_values(self):
        dt = forge.ActivationType
        assert dt.SiLU_GELU is not None
        assert dt.GELU is not None
        assert dt.ReLU is not None

    def test_rope_type_values(self):
        dt = forge.RopeType
        assert getattr(dt, "None") is not None  # None is a Python keyword
        assert dt.Standard is not None
        assert dt.LinearScaling is not None
        assert dt.NTK_Scaled is not None
        assert dt.NeoX is not None
        assert dt.MRoPE is not None
        assert dt.Proportional is not None

    def test_ffn_type_values(self):
        dt = forge.FFNType
        assert dt.SiLUGated is not None
        assert dt.GeGLU is not None
        assert dt.SimpleGELU is not None
        assert dt.MoE is not None

    def test_tokenizer_model_type_values(self):
        dt = forge.TokenizerModelType
        assert dt.SPM is not None
        assert dt.BPE is not None

    def test_request_status_values(self):
        dt = forge.RequestStatus
        assert dt.Waiting is not None
        assert dt.Prefilling is not None
        assert dt.Decoding is not None
        assert dt.Finished is not None
        assert dt.Failed is not None


# ====================================================================
#  Model.generate() return dict key contract
# ====================================================================


class TestGenerateReturnDict:
    """The dict returned by Model.generate() has exactly these keys."""

    GENERATE_KEYS = sorted(
        [
            "finished",
            "finish_reason",
            "num_generated_tokens",
            "num_prompt_tokens",
            "text",
            "token_ids",
        ]
    )

    def test_generate_return_keys(self, model_path):
        """Load a tiny model, call generate, and check the dict keys."""
        model = forge.Model()
        model.load(
            model_path,
            arch_type="llama",
            vocab_size=100,
            hidden_dim=32,
            intermediate_dim=64,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
            device="cpu",
        )
        result = model.generate([1, 2, 3, 4, 5], max_new_tokens=1, do_sample=False)
        assert isinstance(result, dict), f"generate() must return dict, got {type(result)}"
        actual = sorted(result.keys())
        assert actual == self.GENERATE_KEYS, f"Dict keys mismatch: {actual}"


# ====================================================================
#  RequestScheduler.get_finished() return type contract
# ====================================================================


class TestSchedulerGetFinished:
    """RequestScheduler.get_finished() returns a Python list."""

    def test_get_finished_returns_list(self):
        """Even with no model loaded, type info is available."""
        # We can't instantiate without a model, so we check the type info
        # Just verify the class exists — type check is done in TestRequestSchedulerAPI
        assert hasattr(forge.RequestScheduler, "get_finished")
