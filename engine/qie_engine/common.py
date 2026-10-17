"""Types shared by the engines. No GPU libraries here, so any platform can import it."""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


def backend() -> str:
    """"mlx" on Apple Silicon Macs, "cuda" everywhere else."""
    return "mlx" if sys.platform == "darwin" else "cuda"


@dataclass
class ModelPaths:
    # CUDA engine: ComfyUI-format INT8 files plus a snapshot of the official repo (configs, processor, VAE)
    transformer: Path | None = None
    text_encoder: Path | None = None
    config_dir: Path | None = None
    # MLX engine: one folder of MLX weights converted on this Mac
    mlx_dir: Path | None = None


@dataclass
class Result:
    image: Image.Image
    seed: int
    width: int
    height: int
    total_s: float
    stages_s: dict = field(default_factory=dict)
    stage_peaks_gib: dict = field(default_factory=dict)  # stage -> [allocated, reserved]
    peak_reserved_gib: float = 0.0
    plan: dict = field(default_factory=dict)  # memory plan for the steps (CUDA stream mode)


_IMAGE_REF = re.compile(r"<image\s*(\d+)>|\bimage\s*#?(\d+)\b", re.IGNORECASE)


def tag_images(prompt: str, n_images: int) -> str:
    """Image references the way Qwen-Image-2.1 was trained on them (its prompt rewriter's rules): with several input
    images, <image1>, <image2>, ...; with one, no tag at all ("the image"). So "the person in image 1" works too."""
    def sub(m):
        k = int(m.group(1) or m.group(2))
        if not 1 <= k <= n_images:
            return m.group(0)
        if n_images == 1:
            return "the image" if m.group(0)[0] in "<i" else "The image"
        return f"<image{k}>"

    return _IMAGE_REF.sub(sub, prompt) if n_images else prompt


def create_engine(paths: ModelPaths, **kwargs):
    """The engine for this platform. Both have the same interface: ``generate(...) -> Result``, plus ``gpu``,
    ``offload``, ``attn``, ``load_s``. kwargs: on_load, and for CUDA vram_gb, offload, attn."""
    if backend() == "mlx":
        from .mlx_engine import MlxEngine

        return MlxEngine(paths, on_load=kwargs.get("on_load"))
    from .engine import Engine

    return Engine(paths, **kwargs)
