from math import gcd

import numpy as np
import torch

_LANGUAGES = [
    "auto", "en", "zh", "de", "es", "fr", "ja", "ko", "pt", "it",
    "nl", "ru", "pl", "tr", "ar", "hi", "id", "sv", "fi", "da",
]

_MODES = ["default", "clone", "continuation", "continuation_clone"]


def _to_mono_resampled(waveform, orig_sr, target_sr):
    """
    ComfyUI AUDIO waveform [C, T] -> mono float32 [1, T] at target_sr.
    Resampling uses scipy so we never depend on torchaudio's I/O backends
    (torchaudio >= 2.9 needs TorchCodec for load/save).
    """
    wav = waveform.detach().float().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    orig_sr, target_sr = int(orig_sr), int(target_sr)
    if orig_sr != target_sr:
        from scipy.signal import resample_poly
        g = gcd(orig_sr, target_sr)
        out = resample_poly(wav.numpy(), target_sr // g, orig_sr // g, axis=-1)
        wav = torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))
    return wav


def _encode_reference(processor, reference_audio, sample_rate):
    """Encode a ComfyUI AUDIO dict into MOSS audio codes [T, NQ]."""
    waveform = reference_audio["waveform"][0]  # [B, C, T] -> [C, T]
    wav = _to_mono_resampled(waveform, reference_audio["sample_rate"], sample_rate)
    if wav.shape[-1] == 0:
        raise ValueError("[MOSS-TTS] reference_audio is empty.")
    with torch.inference_mode():
        return processor.encode_audios_from_wav([wav], sample_rate)[0]


class MossTTSGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MOSS_TTS_MODEL", {}),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of MOSS TTS.",
                    "dynamicPrompts": False,
                }),
                "mode": (
                    _MODES,
                    {"default": "default"},
                ),
                "language": (
                    _LANGUAGES,
                    {"default": "auto"},
                ),
                "temperature": ("FLOAT", {
                    "default": 1.7, "min": 0.1, "max": 3.0, "step": 0.05,
                    "display": "slider",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.01,
                    "display": "slider",
                }),
                "top_k": ("INT", {
                    "default": 25, "min": 1, "max": 200, "step": 1,
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0, "min": 0.8, "max": 2.0, "step": 0.05,
                }),
                "max_new_tokens": ("INT", {
                    "default": 4096, "min": 256, "max": 8192, "step": 128,
                }),
            },
            "optional": {
                "reference_audio": ("AUDIO", {}),
                "reference_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                }),
                # Kept last (and optional) so widget order in existing saved
                # workflows is unchanged.
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "Random seed for sampling. Change it (or set to randomize) to get a different take.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/MOSS-TTS"

    def generate(
        self,
        model,
        text,
        mode,
        language,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_new_tokens,
        reference_audio=None,
        reference_text="",
        seed=0,
    ):
        processor   = model["processor"]
        hf_model    = model["model"]
        device      = model["device"]
        sample_rate = model["sample_rate"]

        needs_reference = mode in ("clone", "continuation", "continuation_clone")
        if needs_reference and reference_audio is None:
            raise ValueError(
                f"[MOSS-TTS] Mode '{mode}' requires a reference_audio input. "
                "Connect a LoadAudio node to the reference_audio socket."
            )

        needs_ref_text = mode in ("continuation", "continuation_clone")
        if needs_ref_text and not (reference_text or "").strip():
            raise ValueError(
                f"[MOSS-TTS] Mode '{mode}' requires reference_text (the transcript "
                "of the reference audio). Fill in the reference_text field."
            )

        ref_codes = None

        try:
            # Only encode the reference when the selected mode actually uses it.
            if needs_reference:
                print("[MOSS-TTS] Encoding reference audio ...")
                ref_codes = _encode_reference(processor, reference_audio, sample_rate)

            lang_kwargs = {} if language == "auto" else {"language": language}

            if mode == "default":
                conversations = [[
                    processor.build_user_message(text=text, **lang_kwargs)
                ]]
                proc_mode = "generation"

            elif mode == "clone":
                conversations = [[
                    processor.build_user_message(
                        text=text,
                        reference=[ref_codes],
                        **lang_kwargs,
                    )
                ]]
                proc_mode = "generation"

            elif mode == "continuation":
                combined = reference_text.strip() + " " + text.strip()
                conversations = [[
                    processor.build_user_message(text=combined, **lang_kwargs),
                    processor.build_assistant_message(audio_codes_list=[ref_codes]),
                ]]
                proc_mode = "continuation"

            else:  # continuation_clone
                combined = reference_text.strip() + " " + text.strip()
                conversations = [[
                    processor.build_user_message(
                        text=combined,
                        reference=[ref_codes],
                        **lang_kwargs,
                    ),
                    processor.build_assistant_message(audio_codes_list=[ref_codes]),
                ]]
                proc_mode = "continuation"

            print(f"[MOSS-TTS] Processing input (mode={mode}, lang={language}) ...")
            batch = processor(conversations, mode=proc_mode)

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            print(f"[MOSS-TTS] Generating audio (max_new_tokens={max_new_tokens}, seed={seed}) ...")
            torch.manual_seed(seed & 0xFFFFFFFFFFFFFFFF)
            with torch.inference_mode():
                outputs = hf_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    audio_temperature=temperature,
                    audio_top_p=top_p,
                    audio_top_k=top_k,
                    audio_repetition_penalty=repetition_penalty,
                )

            print("[MOSS-TTS] Decoding audio ...")
            decoded = processor.decode(outputs)

            if not decoded or decoded[0] is None or not decoded[0].audio_codes_list:
                raise RuntimeError(
                    "[MOSS-TTS] Generation produced no audio output. "
                    "Try increasing max_new_tokens or changing the seed/sampling settings."
                )

            audio_tensor = decoded[0].audio_codes_list[0]

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # [1, T]

            # ComfyUI AUDIO format: [B=1, C, T] float32 on CPU, values in [-1, 1]
            audio_out = torch.clamp(audio_tensor.float().cpu(), -1.0, 1.0).unsqueeze(0)

            print(f"[MOSS-TTS] Done. shape={audio_out.shape}, sr={sample_rate}")
            return ({"waveform": audio_out, "sample_rate": sample_rate},)

        except Exception as e:
            raise RuntimeError(f"[MOSS-TTS] Generation failed: {e}") from e
