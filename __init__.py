import os
import folder_paths

# Register ComfyUI model directory for MOSS-TTS weights
_MOSS_TTS_MODEL_DIR = os.path.join(folder_paths.models_dir, "moss_tts")
os.makedirs(_MOSS_TTS_MODEL_DIR, exist_ok=True)
folder_paths.add_model_folder_path("moss_tts", _MOSS_TTS_MODEL_DIR)

try:
    from .nodes.model_loader import MossTTSModelLoader
    from .nodes.tts_generate import MossTTSGenerate

    NODE_CLASS_MAPPINGS = {
        "MossTTSModelLoader": MossTTSModelLoader,
        "MossTTSGenerate":    MossTTSGenerate,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "MossTTSModelLoader": "MOSS-TTS Model Loader",
        "MossTTSGenerate":    "MOSS-TTS Generate",
    }

    print("[MOSS-TTS] Nodes loaded successfully.")

except Exception as e:
    import traceback
    print(f"[MOSS-TTS] Failed to load nodes: {e}")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
