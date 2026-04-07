"""
Flash Attention 2 patch for MossTTSAttentionWithoutPositionalEmbedding.

At model-load time we inspect the attention class's forward method via
inspect.getsource() to extract projection names, num_heads, and head_dim,
then replace forward with an FA2 implementation using flash_attn_func.

All code lives in the project folder — the downloaded model source is read-only.
"""

import types
import inspect
import re
import torch
import torch.nn as nn


_TARGET_CLASS = "MossTTSAttentionWithoutPositionalEmbedding"


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def _find_linear_children(module):
    """Return {attr_name: nn.Linear} for direct nn.Linear children."""
    return {
        name: child
        for name, child in module.named_children()
        if isinstance(child, nn.Linear)
    }


def _detect_qkvo(linears: dict):
    """
    Match q/k/v/o projection names from a dict of {name: Linear}.
    Tries common naming patterns used in HuggingFace models.
    Returns (q_name, k_name, v_name, o_name) or raises RuntimeError.
    """
    def pick(candidates):
        for pattern in candidates:
            match = next((n for n in linears if pattern in n.lower()), None)
            if match:
                return match
        return None

    q = pick(["q_proj", "q_linear", "query", "wq"])
    k = pick(["k_proj", "k_linear", "key",   "wk"])
    v = pick(["v_proj", "v_linear", "value",  "wv"])
    o = pick(["o_proj", "out_proj", "wo", "c_proj", "output"])

    missing = [name for name, val in [("q", q), ("k", k), ("v", v), ("o", o)] if val is None]
    if missing:
        raise RuntimeError(
            f"[MOSS-TTS FA2] Could not detect projections {missing}. "
            f"Found linears: {list(linears.keys())}"
        )
    return q, k, v, o


def _detect_head_info(module, q_name: str):
    """
    Detect num_heads, num_key_value_heads, head_dim.
    Falls back to inferring from projection output sizes if attributes are absent.
    """
    num_heads     = getattr(module, "num_heads",             None)
    num_kv_heads  = getattr(module, "num_key_value_heads",   None)
    head_dim      = getattr(module, "head_dim",              None)

    q_proj: nn.Linear = getattr(module, q_name)
    hidden_size = q_proj.in_features
    q_out       = q_proj.out_features

    if num_heads is None:
        # Prefer head_dim=64 as a common default; otherwise divide evenly
        if head_dim is not None:
            num_heads = q_out // head_dim
        else:
            # Try to infer from source if possible
            try:
                src = inspect.getsource(type(module).__init__)
                m = re.search(r"num_heads\s*=\s*(\d+)", src)
                if m:
                    num_heads = int(m.group(1))
            except Exception:
                pass
        if num_heads is None:
            num_heads = q_out // 64  # safe guess

    if head_dim is None:
        head_dim = q_out // num_heads

    if num_kv_heads is None:
        # Check whether k_proj has a different output size (GQA)
        k_name_candidate = next(
            (n for n in module._modules if "k_proj" in n or "k_linear" in n or n == "k"), None
        )
        if k_name_candidate:
            k_out = getattr(module, k_name_candidate).out_features
            num_kv_heads = k_out // head_dim
        else:
            num_kv_heads = num_heads

    return num_heads, num_kv_heads, head_dim


def _detect_return_shape(module):
    """
    Try to determine what the original forward returns by inspecting its source.
    Returns a string tag: 'tuple3' | 'tuple2' | 'tensor'
    """
    try:
        src = inspect.getsource(module.forward)
        returns = re.findall(r"return\s+(.+)", src)
        for r in returns:
            commas = r.count(",")
            if commas >= 2:
                return "tuple3"
            if commas == 1:
                return "tuple2"
        return "tensor"
    except Exception:
        return "tuple3"  # safest default for HF-style attention


# ---------------------------------------------------------------------------
# FA2 forward factory
# ---------------------------------------------------------------------------

def _make_fa2_forward(q_name, k_name, v_name, o_name,
                      num_heads, num_kv_heads, head_dim,
                      return_shape, softmax_scale=None):
    """
    Build a forward method that uses flash_attn_func.

    flash_attn_func expects inputs of shape [B, T, H, D] in float16/bfloat16.
    We cast internally and cast back so the rest of the model is unaffected.
    """
    from flash_attn import flash_attn_func

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                cache_position=None, use_cache=False, **kwargs):
        B, T, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        # FA2 only supports fp16/bf16
        if orig_dtype not in (torch.float16, torch.bfloat16):
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = orig_dtype
        hs = hidden_states.to(compute_dtype)

        q = getattr(self, q_name)(hs)   # [B, T, num_heads * head_dim]
        k = getattr(self, k_name)(hs)   # [B, T, num_kv_heads * head_dim]
        v = getattr(self, v_name)(hs)   # [B, T, num_kv_heads * head_dim]

        # Reshape to [B, T, H, D] as expected by flash_attn_func
        q = q.view(B, T, num_heads,    head_dim)
        k = k.view(B, T, num_kv_heads, head_dim)
        v = v.view(B, T, num_kv_heads, head_dim)

        # causal=True — MOSS-TTS is an autoregressive model
        attn_out = flash_attn_func(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=True,
        )  # [B, T, num_heads, head_dim]

        attn_out = attn_out.reshape(B, T, num_heads * head_dim).to(orig_dtype)
        out = getattr(self, o_name)(attn_out)

        if return_shape == "tuple3":
            return out, None, past_key_values
        if return_shape == "tuple2":
            return out, None
        return out

    return forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch_model_fa2(model) -> bool:
    """
    Find all MossTTSAttentionWithoutPositionalEmbedding modules in *model*,
    inspect their structure at runtime, and replace forward with an FA2 version.

    Returns True if at least one module was patched.
    Raises RuntimeError if flash_attn is not installed.
    """
    try:
        import flash_attn  # noqa: F401 — verify availability
    except ImportError:
        raise RuntimeError(
            "[MOSS-TTS FA2] flash-attn package not found.\n"
            "  Install with: pip install flash-attn --no-build-isolation"
        )

    patched = 0
    errors  = []

    for mod_name, module in model.named_modules():
        if type(module).__name__ != _TARGET_CLASS:
            continue

        try:
            linears          = _find_linear_children(module)
            q_n, k_n, v_n, o_n = _detect_qkvo(linears)
            nh, nkvh, hd    = _detect_head_info(module, q_n)
            ret_shape        = _detect_return_shape(module)

            fa2_fwd = _make_fa2_forward(
                q_n, k_n, v_n, o_n, nh, nkvh, hd, ret_shape
            )
            module.forward = types.MethodType(fa2_fwd, module)
            patched += 1
            print(
                f"[MOSS-TTS FA2] Patched {mod_name}: "
                f"heads={nh} kv_heads={nkvh} head_dim={hd} "
                f"q={q_n} k={k_n} v={v_n} o={o_n} ret={ret_shape}"
            )

        except Exception as e:
            errors.append(f"  {mod_name}: {e}")

    if errors:
        print("[MOSS-TTS FA2] WARNING — some modules could not be patched:")
        for err in errors:
            print(err)

    if patched == 0:
        print(
            f"[MOSS-TTS FA2] No modules of class '{_TARGET_CLASS}' found. "
            "Check that the model loaded correctly."
        )
        return False

    print(f"[MOSS-TTS FA2] Done — {patched} attention module(s) patched.")
    return True
