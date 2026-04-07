import os
import hashlib
import torch

_MODEL_CACHE: dict = {}

_REPO_MAP = {
    "MossTTSDelay (8B)":   "OpenMOSS-Team/MOSS-TTS",
    "MossTTSLocal (1.7B)": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
}

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}

_AUDIO_TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
_io_replaced = False


def _sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def _download(repo_id: str, local_dir: str):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )


def _fix_audio_tokenizer_config(tokenizer_dir: str):
    """
    MossAudioTokenizerConfig has bare class-level annotations with no defaults:
        sampling_rate: int
        downsample_rate: int
        ...
    When transformers (>=5) applies @dataclass to PreTrainedConfig subclasses,
    Python 3.12 raises TypeError because non-default fields follow inherited
    default fields. The __init__ already handles all defaults, so these
    annotations serve no runtime purpose and can be removed.
    We fix the file in our local models dir so transformers caches the fixed version.
    """
    config_path = os.path.join(tokenizer_dir, "configuration_moss_audio_tokenizer.py")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # These are the bare annotations causing the dataclass failure
    bare_annotations = [
        "    sampling_rate: int\n",
        "    downsample_rate: int\n",
        "    causal_transformer_context_duration: float\n",
        "    encoder_kwargs: list[dict[str, Any]]\n",
        "    decoder_kwargs: list[dict[str, Any]]\n",
        "    quantizer_type: str\n",
        "    quantizer_kwargs: dict[str, Any]\n",
    ]

    fixed = content
    for line in bare_annotations:
        fixed = fixed.replace(line, "")

    if fixed != content:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(fixed)
        print("[MOSS-TTS] Fixed configuration_moss_audio_tokenizer.py")


def _replace_torchaudio_io():
    """
    MOSS-TTS calls torchaudio.load() and torchaudio.functional.resample().
    Replace with soundfile + scipy since TorchCodec is unavailable.
    """
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample_poly
    from math import gcd

    def _load(filepath, *args, **kwargs):
        # soundfile returns [T, C]; torchaudio returns [C, T]
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    def _resample(waveform, orig_freq, new_freq, *args, **kwargs):
        orig_freq, new_freq = int(orig_freq), int(new_freq)
        if orig_freq == new_freq:
            return waveform
        g = gcd(orig_freq, new_freq)
        out = resample_poly(waveform.numpy(), new_freq // g, orig_freq // g, axis=-1)
        return torch.from_numpy(out.astype(np.float32))

    import torchaudio
    import torchaudio.functional as F
    torchaudio.load = _load
    F.resample = _resample


class MossTTSModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (
                    list(_REPO_MAP.keys()),
                    {"default": "MossTTSDelay (8B)"},
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto"},
                ),
                "dtype": (
                    ["bfloat16", "float16", "float32"],
                    {"default": "bfloat16"},
                ),
                "attn_implementation": (
                    ["sdpa", "flash_attention_2", "eager"],
                    {"default": "sdpa"},
                ),
            }
        }

    RETURN_TYPES = ("MOSS_TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/MOSS-TTS"

    @classmethod
    def IS_CHANGED(cls, model_type, device, dtype, attn_implementation):
        key = f"{model_type}|{device}|{dtype}|{attn_implementation}"
        return hashlib.md5(key.encode()).hexdigest()

    def load_model(self, model_type, device, dtype, attn_implementation):
        global _io_replaced
        import folder_paths
        from transformers import AutoModel, AutoProcessor

        if not _io_replaced:
            _replace_torchaudio_io()
            _io_replaced = True

        resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        resolved_dtype = _DTYPE_MAP[dtype]

        cache_key = (model_type, resolved_device, dtype, attn_implementation)
        if cache_key in _MODEL_CACHE:
            print(f"[MOSS-TTS] Using cached model: {model_type} on {resolved_device}")
            return (_MODEL_CACHE[cache_key],)

        # Evict any other cached models to free VRAM before loading a new one.
        # Two models simultaneously is unlikely to fit (especially the 8B).
        if _MODEL_CACHE:
            print("[MOSS-TTS] Evicting cached model(s) to free VRAM ...")
            for key in list(_MODEL_CACHE.keys()):
                old = _MODEL_CACHE.pop(key)
                # Move to CPU first so CUDA memory is released immediately
                try:
                    old["model"].cpu()
                except Exception:
                    pass
                try:
                    if hasattr(old["processor"], "audio_tokenizer"):
                        old["processor"].audio_tokenizer.cpu()
                except Exception:
                    pass
                del old
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("[MOSS-TTS] VRAM freed.")

        models_root = folder_paths.get_folder_paths("moss_tts")[0]
        local_dir   = os.path.join(models_root, _sanitize(model_type))
        tokenizer_dir = os.path.join(models_root, "MossAudioTokenizer")
        repo_id     = _REPO_MAP[model_type]

        # Validate flash_attention_2 requirements before attempting to load
        if attn_implementation == "flash_attention_2":
            if resolved_device != "cuda":
                print("[MOSS-TTS] WARNING: flash_attention_2 requires CUDA — falling back to sdpa")
                attn_implementation = "sdpa"
            elif dtype == "float32":
                print("[MOSS-TTS] WARNING: flash_attention_2 requires float16 or bfloat16 — falling back to sdpa")
                attn_implementation = "sdpa"
            else:
                try:
                    import flash_attn  # noqa: F401
                except ImportError:
                    print(
                        "[MOSS-TTS] WARNING: flash-attn package not found — falling back to sdpa.\n"
                        "  To enable Flash Attention 2, install it with:\n"
                        "    pip install flash-attn --no-build-isolation"
                    )
                    attn_implementation = "sdpa"

        try:
            # Download main model
            print(f"[MOSS-TTS] Downloading {repo_id} -> {local_dir}")
            _download(repo_id, local_dir)

            # Download audio tokenizer locally so we never hit HuggingFace at inference time
            print(f"[MOSS-TTS] Downloading {_AUDIO_TOKENIZER_REPO} -> {tokenizer_dir}")
            _download(_AUDIO_TOKENIZER_REPO, tokenizer_dir)

            # Fix the dataclass annotation bug in the local audio tokenizer config
            _fix_audio_tokenizer_config(tokenizer_dir)

            # cuDNN SDP has issues with MOSS-TTS's custom attention — disable for sdpa/eager.
            # Flash Attention 2 bypasses SDP entirely so the flag is irrelevant there.
            if resolved_device == "cuda" and attn_implementation != "flash_attention_2":
                torch.backends.cuda.enable_cudnn_sdp(False)

            print("[MOSS-TTS] Loading processor ...")
            processor = AutoProcessor.from_pretrained(
                local_dir,
                trust_remote_code=True,
                codec_path=tokenizer_dir,   # use local copy, not HuggingFace
            )
            if hasattr(processor, "audio_tokenizer"):
                # audio_tokenizer is separate from the main model so device_map
                # doesn't cover it — place it explicitly.
                processor.audio_tokenizer = processor.audio_tokenizer.to(resolved_device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # free any staging memory from tokenizer load

            print(f"[MOSS-TTS] Loading model ({dtype}) on {resolved_device} ...")
            # Use device_map so accelerate places weights directly on the target
            # device instead of loading to CPU first then copying — avoids needing
            # 2x the model size in memory during load.
            # device_map="auto" additionally offloads layers to CPU/disk if the
            # model doesn't fit entirely in VRAM.
            load_device_map = "auto" if resolved_device == "cuda" else resolved_device
            model = AutoModel.from_pretrained(
                local_dir,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=resolved_dtype,
                device_map=load_device_map,
            )
            model.eval()

            # transformers 5.x DynamicCache needs config.num_hidden_layers directly.
            # MossTTSDelayConfig stores it inside language_config (Qwen3Config).
            if not hasattr(model.config, "num_hidden_layers"):
                model.config.num_hidden_layers = model.config.language_config.num_hidden_layers

            import types

            # transformers 5.x removed _get_initial_cache_position from GenerationMixin
            # but MOSS-TTS _sample still calls it.
            if not hasattr(model, "_get_initial_cache_position"):
                def _get_initial_cache_position(self, cur_len, device, model_kwargs):
                    if "cache_position" not in model_kwargs:
                        past_length = 0
                        pkv = model_kwargs.get("past_key_values")
                        if pkv is not None:
                            if hasattr(pkv, "get_seq_length"):
                                past_length = pkv.get_seq_length()
                            elif hasattr(pkv, "get_usable_length"):
                                past_length = pkv.get_usable_length(cur_len)
                        model_kwargs["cache_position"] = torch.arange(past_length, cur_len, device=device)
                    return model_kwargs
                model._get_initial_cache_position = types.MethodType(_get_initial_cache_position, model)

            # MOSS-TTS uses 3D input_ids [B, T, 1+NQ] and manages its own generation
            # loop via CustomMixin._sample which appends tokens directly. KV cache
            # trimming of 3D tensors corrupts the token stream and produces garbled audio.
            # Disable KV cache so the model always receives the full sequence.
            def _prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                                attention_mask=None, cache_position=None,
                                                **kwargs):
                return {
                    "input_ids": input_ids,
                    "past_key_values": None,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "use_cache": False,
                }
            model.prepare_inputs_for_generation = types.MethodType(_prepare_inputs_for_generation, model)

            sample_rate = 24000
            if hasattr(processor, "model_config") and hasattr(processor.model_config, "sampling_rate"):
                sample_rate = processor.model_config.sampling_rate

            handle = {
                "processor":   processor,
                "model":       model,
                "device":      resolved_device,
                "dtype":       resolved_dtype,
                "model_type":  model_type,
                "sample_rate": sample_rate,
            }
            _MODEL_CACHE[cache_key] = handle
            print(f"[MOSS-TTS] Ready: {model_type} | {resolved_device} | {dtype} | attn={attn_implementation} | sr={sample_rate}")
            return (handle,)

        except Exception as e:
            _MODEL_CACHE.pop(cache_key, None)
            raise RuntimeError(
                f"[MOSS-TTS] Failed to load '{model_type}': {e}\n"
                f"  model dir:     {local_dir}\n"
                f"  tokenizer dir: {tokenizer_dir}"
            ) from e
