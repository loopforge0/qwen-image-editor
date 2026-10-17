"""Build the diffusers Qwen-Image-2.1 pipeline from ComfyUI-format INT8-ConvRot checkpoints.

Configs, processor and VAE come from the official ``Qwen/Qwen-Image-2.1-Turbo`` repo. The transformer and
text encoder come from single-file INT8-ConvRot checkpoints (``Comfy-Org/Qwen-Image-2.1``), which are
roughly half the size of the official bf16 weights.
"""

from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from .quant import FusedSwiGLU, Int8Embedding, Int8Linear, parse_quant_config

QUANT_SUFFIX = ".comfy_quant"
ATTENTION_SUFFIX = ".comfy_attention.config"  # per-layer attention kernel choice, e.g. {"attention": "comfy_kitchen_int8"}


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _read_checkpoint(path: Path, rename=lambda k: k, report=lambda nbytes: None):
    """Split a checkpoint into int8 layers (name -> (weight, scale, config)), plain tensors, and per-layer attention
    settings (attention module name -> config)."""
    quantized, plain, attention = {}, {}, {}
    with safe_open(str(path), "pt") as f:
        keys = set(f.keys())
        for k in [k for k in keys if k.endswith(ATTENTION_SUFFIX)]:
            attention[rename(k[: -len(ATTENTION_SUFFIX)])] = parse_quant_config(f.get_tensor(k))
            keys.discard(k)
        qnames = {k[: -len(QUANT_SUFFIX)] for k in keys if k.endswith(QUANT_SUFFIX)}
        for name in qnames:
            weight, scale = f.get_tensor(f"{name}.weight"), f.get_tensor(f"{name}.weight_scale")
            quantized[rename(name)] = (weight, scale, parse_quant_config(f.get_tensor(name + QUANT_SUFFIX)))
            report(_nbytes(weight) + _nbytes(scale))
        for k in keys:
            owner = k.rsplit(".", 1)[0]
            if owner not in qnames:
                plain[rename(k)] = f.get_tensor(k)
                report(_nbytes(plain[rename(k)]))
    return quantized, plain, attention


def _int8_linear(entry, input_act=None) -> Int8Linear:
    weight, scale, cfg = entry
    if cfg.get("format") != "int8_tensorwise":
        raise ValueError(f"unsupported quant format {cfg}")
    return Int8Linear(weight, scale, cfg.get("convrot", False), cfg.get("convrot_groupsize", 256), input_act)


def _finish(model: nn.Module, plain: dict, dtype: torch.dtype, ignore=()) -> None:
    plain = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in plain.items()}
    missing, unexpected = model.load_state_dict(plain, strict=False, assign=True)
    unexpected = [k for k in unexpected if not k.startswith(tuple(ignore))]
    still_meta = [n for n, p in model.named_parameters() if p.is_meta]
    if unexpected or still_meta:
        raise RuntimeError(f"checkpoint mismatch: unexpected={unexpected[:8]} uninitialized={still_meta[:8]}")


def load_transformer(ckpt: Path, config_dir: Path, dtype=torch.bfloat16, report=lambda nbytes: None) -> QwenImage21Transformer2DModel:
    config = QwenImage21Transformer2DModel.load_config(config_dir / "transformer")
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(config)

    quantized, plain, attention = _read_checkpoint(ckpt, report=report)
    # Which blocks may use INT8 attention. Checkpoints from 2026-10-09 on say so per block (all but the first);
    # earlier uploads of the same weights don't, so they get the same policy.
    n = len(model.transformer_blocks)
    if attention:
        model.int8_attention_blocks = {
            i for i in range(n) if attention.get(f"transformer_blocks.{i}.attn", {}).get("attention") == "comfy_kitchen_int8"
        }
    else:
        model.int8_attention_blocks = set(range(1, n))
    for name, entry in quantized.items():
        if name.endswith(".img_mlp.gate_up"):
            mlp = name[: -len(".gate_up")]
            fused = FusedSwiGLU(_int8_linear(entry), _int8_linear(quantized[f"{mlp}.out"]))
            model.set_submodule(mlp, fused)
        elif name.endswith(".img_mlp.out"):
            continue  # handled with its gate_up
        else:
            model.set_submodule(name, _int8_linear(entry))
    _finish(model, plain, dtype)
    return model.eval().requires_grad_(False)


def _te_rename(k: str) -> str:
    # ComfyUI keeps the flat Qwen3 layout; transformers 5 nests the text model under `language_model`.
    if k.startswith("model.") and not k.startswith("model.visual."):
        return "model.language_model." + k[len("model.") :]
    return k


def load_text_encoder(ckpt: Path, config_dir: Path, dtype=torch.bfloat16, report=lambda nbytes: None) -> Qwen3VLForConditionalGeneration:
    config = AutoConfig.from_pretrained(config_dir / "text_encoder")
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config, dtype=dtype)

    quantized, plain, _ = _read_checkpoint(ckpt, _te_rename, report)
    for name, entry in quantized.items():
        if name == "lm_head":
            # The pipeline reads hidden states, never logits; skip a 151936-wide GEMM per token.
            model.lm_head = nn.Identity()
        elif name.endswith("embed_tokens"):
            weight, scale, cfg = entry
            model.set_submodule(name, Int8Embedding(weight, scale, cfg, dtype))
        else:
            model.set_submodule(name, _int8_linear(entry))
    _finish(model, plain, dtype)
    return model.eval().requires_grad_(False)


def load_pipeline(
    transformer_ckpt: Path,
    text_encoder_ckpt: Path,
    config_dir: Path,
    dtype=torch.bfloat16,
    report=lambda phase, nbytes: None,
) -> QwenImage21Pipeline:
    """``report(phase, nbytes)`` is called as weights are read; phase is "vae", "text_encoder" or "transformer"."""
    config_dir = Path(config_dir)
    vae = AutoencoderKLQwenImage21.from_pretrained(config_dir / "vae", dtype=dtype)
    report("vae", sum(_nbytes(p) for p in vae.parameters()))
    text_encoder = load_text_encoder(Path(text_encoder_ckpt), config_dir, dtype, lambda n: report("text_encoder", n))
    transformer = load_transformer(Path(transformer_ckpt), config_dir, dtype, lambda n: report("transformer", n))
    return QwenImage21Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(config_dir / "scheduler"),
        vae=vae,
        text_encoder=text_encoder,
        processor=Qwen3VLProcessor.from_pretrained(config_dir / "processor"),
        transformer=transformer,
        sample_sigmas=QwenImage21Pipeline.load_config(config_dir)["sample_sigmas"],
    )
