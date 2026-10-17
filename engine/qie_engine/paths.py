"""Where the weights are: the config written by ``qie setup``, with environment variables as overrides.

NVIDIA (Windows, Linux):
- ``QIE_TRANSFORMER``: Qwen-Image-2.1-Turbo INT8-ConvRot transformer (.safetensors, ComfyUI format)
- ``QIE_TEXT_ENCODER``: Qwen3-VL 8B INT8-ConvRot text encoder (.safetensors, ComfyUI format)
- ``QIE_CONFIG_DIR``: a snapshot of Qwen/Qwen-Image-2.1-Turbo (configs, processor, VAE)

Apple Silicon:
- ``QIE_MLX_DIR``: the MLX weights converted by ``qie setup``
"""

import os
from pathlib import Path

from . import config
from .common import ModelPaths, backend


class NotSetUp(FileNotFoundError):
    pass


def default_paths() -> ModelPaths:
    files = config.load().get("files", {})

    def pick(env, key):
        value = os.environ.get(env) or files.get(key)
        return Path(value) if value else None

    if backend() == "mlx":
        paths = ModelPaths(mlx_dir=pick("QIE_MLX_DIR", "mlx_model"))
        needed = {"mlx_model": paths.mlx_dir}
    else:
        paths = ModelPaths(
            transformer=pick("QIE_TRANSFORMER", "transformer"),
            text_encoder=pick("QIE_TEXT_ENCODER", "text_encoder"),
            config_dir=pick("QIE_CONFIG_DIR", "config_dir"),
        )
        needed = {"transformer": paths.transformer, "text_encoder": paths.text_encoder, "config_dir": paths.config_dir}
    missing = [name for name, p in needed.items() if p is None or not p.exists()]
    if missing:
        raise NotSetUp(f"The models aren't set up yet ({', '.join(missing)} missing). Run:  uv run qie setup")
    return paths
