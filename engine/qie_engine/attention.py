"""Route Qwen-Image-2.1's unmasked attention calls through comfy-kitchen's INT8 (Sage-style) kernel.

diffusers' processor calls ``dispatch_attention_fn`` with ``[batch, seq, heads, head_dim]`` tensors.
Masked calls (the causal text segments of the KV-cache prefill) are small and stay on the default path.
Only the blocks the checkpoint marks for INT8 attention use it (``transformer.int8_attention_blocks``:
all but the first block); the rest keep full-precision attention.
"""

import comfy_kitchen as ck
import torch
from diffusers.models.transformers import transformer_qwenimage21 as qi21

_original = qi21.dispatch_attention_fn
_block_allows_int8 = True  # set by hooks on each block's attention module


def _int8_dispatch(query, key, value, attn_mask=None, dropout_p=0.0, **kwargs):
    if not _block_allows_int8 or attn_mask is not None or dropout_p or not query.is_cuda:
        return _original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, **kwargs)
    out = ck.int8_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    return out.transpose(1, 2)


def use_int8_attention(enabled: bool = True) -> bool:
    """Swap the kernel in or out. Returns whether INT8 attention is active."""
    enabled = enabled and torch.cuda.is_available() and ck.int8_attention_is_available()
    qi21.dispatch_attention_fn = _int8_dispatch if enabled else _original
    return enabled


def limit_int8_attention(transformer):
    """Give INT8 attention only to the blocks in ``transformer.int8_attention_blocks``."""
    allowed = getattr(transformer, "int8_attention_blocks", None)
    if allowed is None:
        return

    def enter(flag):
        def hook(module, args):
            global _block_allows_int8
            _block_allows_int8 = flag

        return hook

    for i, block in enumerate(transformer.transformer_blocks):
        block.attn.register_forward_pre_hook(enter(i in allowed))
