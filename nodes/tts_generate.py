import os
import tempfile
import torch
import numpy as np
import soundfile as sf

_LANGUAGES = [
    "auto", "en", "zh", "de", "es", "fr", "ja", "ko", "pt", "it",
    "nl", "ru", "pl", "tr", "ar", "hi", "id", "sv", "fi", "da",
]

_MODES = ["default", "clone", "continuation", "continuation_clone"]


def _save_wav(path, waveform_2d, sample_rate):
    """Save a [C, T] float32 tensor as a WAV file using soundfile."""
    data = waveform_2d.detach().float().numpy()
    if data.ndim == 2:
        data = data.T  # [C, T] -> [T, C]
    sf.write(path, data, sample_rate)


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

        _tmp = None
        ref_audio_path = None

        try:
            if reference_audio is not None:
                _tmp = tempfile.TemporaryDirectory(prefix="moss_tts_ref_")
                ref_audio_path = os.path.join(_tmp.name, "reference.wav")
                # ComfyUI waveform: [B, C, T] float32
                waveform = reference_audio["waveform"][0]  # [C, T]
                ref_sr   = reference_audio["sample_rate"]
                _save_wav(ref_audio_path, waveform, ref_sr)

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
                        reference=[ref_audio_path],
                        **lang_kwargs,
                    )
                ]]
                proc_mode = "generation"

            elif mode == "continuation":
                combined = reference_text.strip() + " " + text.strip()
                conversations = [[
                    processor.build_user_message(text=combined, **lang_kwargs),
                    processor.build_assistant_message(audio_codes_list=[ref_audio_path]),
                ]]
                proc_mode = "continuation"

            else:  # continuation_clone
                combined = reference_text.strip() + " " + text.strip()
                conversations = [[
                    processor.build_user_message(
                        text=combined,
                        reference=[ref_audio_path],
                        **lang_kwargs,
                    ),
                    processor.build_assistant_message(audio_codes_list=[ref_audio_path]),
                ]]
                proc_mode = "continuation"

            print(f"[MOSS-TTS] Processing input (mode={mode}, lang={language}) ...")
            batch = processor(conversations, mode=proc_mode)

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            print(f"[MOSS-TTS] Generating audio (max_new_tokens={max_new_tokens}) ...")
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

            if not decoded or not decoded[0].audio_codes_list:
                raise RuntimeError("[MOSS-TTS] Generation produced no audio output.")

            audio_tensor = decoded[0].audio_codes_list[0]

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # [1, T]

            # ComfyUI AUDIO format: [B=1, C, T] float32 on CPU, values in [-1, 1]
            audio_out = torch.clamp(audio_tensor.float().cpu(), -1.0, 1.0).unsqueeze(0)

            print(f"[MOSS-TTS] Done. shape={audio_out.shape}, sr={sample_rate}")
            return ({"waveform": audio_out, "sample_rate": sample_rate},)

        except Exception as e:
            raise RuntimeError(f"[MOSS-TTS] Generation failed: {e}") from e

        finally:
            if _tmp is not None:
                try:
                    _tmp.cleanup()
                except Exception:
                    pass
