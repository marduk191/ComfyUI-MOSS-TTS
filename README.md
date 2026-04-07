# ComfyUI-MOSS-TTS

ComfyUI custom nodes for [MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) — a high-quality multilingual TTS system from OpenMOSS. Supports voice cloning, continuation, and all sampling parameters. Models are downloaded automatically from HuggingFace on first use.

---

## Models

| Node option | HuggingFace repo | VRAM (bf16) |
|---|---|---|
| MossTTSDelay (8B) | `OpenMOSS-Team/MOSS-TTS` | ~16 GB |
| MossTTSLocal (1.7B) | `OpenMOSS-Team/MOSS-TTS-Local-Transformer` | ~3.4 GB |

The audio tokenizer (`OpenMOSS-Team/MOSS-Audio-Tokenizer`) is downloaded automatically alongside the main model.

Models are saved to `ComfyUI/models/moss_tts/`.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/marduk191/ComfyUI-MOSS-TTS
cd ComfyUI-MOSS-TTS
pip install -r requirements.txt
```

Restart ComfyUI. Models will download automatically when the loader node first runs.

### Flash Attention 2 (optional)

Not Currently implemented.
```

---

## Nodes

### MOSS-TTS Model Loader

Loads and caches the model. Switching models evicts the previous one from VRAM automatically.

| Parameter | Options | Description |
|---|---|---|
| model_type | MossTTSDelay (8B), MossTTSLocal (1.7B) | Which model to load |
| device | auto, cuda, cpu | Target device (`auto` picks CUDA if available) |
| dtype | bfloat16, float16, float32 | Weight precision. `bfloat16` recommended |
| attn_implementation | sdpa, flash_attention_2, eager | Attention backend. `sdpa` recommended unless `flash-attn` is installed |

**Output:** `MOSS_TTS_MODEL` handle — connect to the Generate node.

---

### MOSS-TTS Generate

Generates speech from text. Supports four generation modes.

| Parameter | Default | Description |
|---|---|---|
| text | — | Text to synthesize |
| mode | default | See modes below |
| language | auto | Language hint (`auto` detects automatically) |
| temperature | 1.7 | Audio sampling temperature |
| top_p | 0.8 | Nucleus sampling probability |
| top_k | 25 | Top-k sampling |
| repetition_penalty | 1.0 | Penalise repeated tokens |
| max_new_tokens | 4096 | Maximum audio tokens to generate |
| reference_audio | — | (optional) Reference audio for cloning/continuation |
| reference_text | — | (optional) Transcript of reference audio |

**Output:** `AUDIO` — compatible with all standard ComfyUI audio nodes (Preview Audio, Save Audio, etc).

#### Generation modes

| Mode | Reference audio | Reference text | Description |
|---|---|---|---|
| `default` | No | No | Generate speech in the model's default voice |
| `clone` | Yes | No | Clone the voice from the reference audio |
| `continuation` | Yes | Yes | Continue speech from the reference audio |
| `continuation_clone` | Yes | Yes | Continue speech while also cloning the voice |

---

## Supported languages

`auto`, English, Chinese, German, Spanish, French, Japanese, Korean, Portuguese, Italian, Dutch, Russian, Polish, Turkish, Arabic, Hindi, Indonesian, Swedish, Finnish, Danish

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers >= 5.0.0
- accelerate >= 0.30.0
- CUDA GPU recommended (CPU supported but very slow)
- 25 GB VRAM to run the 8B model entirely on GPU

---

## Notes

- Only one model is kept in VRAM at a time. Switching models in the loader node automatically frees the previous one.
- The 8B model uses `device_map="auto"` which will offload layers to system RAM if VRAM is insufficient.
- `torchaudio` is not required — audio I/O uses `soundfile` and `scipy`.
