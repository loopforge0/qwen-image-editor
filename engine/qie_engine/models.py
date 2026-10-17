"""The model files the editor needs: where they come from, how big they are, and how to find copies already on disk.

Nothing here is redistributed. Each file is downloaded from its Hugging Face repo, at a pinned revision, on the
user's machine after they accept the license.
"""

from dataclasses import dataclass, field

LICENSE_NAME = "Qwen Research License Agreement"
LICENSE_URL = "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE"


@dataclass(frozen=True)
class WeightsFile:
    key: str
    label: str
    repo: str
    revision: str
    path: str  # inside the repo
    size: int  # bytes at the pinned revision; also how existing copies are recognized
    comfy_subdir: str  # where ComfyUI keeps it, under models/
    also_accept: tuple = ()  # sizes of earlier uploads with the same weights

    def matches(self, size: int) -> bool:
        return size == self.size or size in self.also_accept


@dataclass(frozen=True)
class ConfigSnapshot:
    """Configs, tokenizer and VAE from the official repo (not its 14 GB bf16 transformer)."""

    key: str
    label: str
    repo: str
    revision: str
    patterns: list = field(default_factory=list)
    size: int = 0  # approximate, for the disk-space check
    required: tuple = ()  # files that must exist for the snapshot to be usable


COMFY_REPO, COMFY_REV = "Comfy-Org/Qwen-Image-2.1", "df94239739eef7205973a145ffa6f441f7b64e84"

TRANSFORMER = WeightsFile(
    key="transformer",
    label="Image model (Qwen-Image-2.1-Turbo, INT8)",
    repo=COMFY_REPO,
    revision=COMFY_REV,
    path="diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
    size=7_256_787_829,
    comfy_subdir="diffusion_models",
    # The first upload (before 2026-10-09 22:58 UTC): byte-identical tensors, without the per-block attention
    # settings. The engine applies the same settings to it.
    also_accept=(7_256_783_064,),
)
TEXT_ENCODER = WeightsFile(
    key="text_encoder",
    label="Text and vision encoder (Qwen3-VL 8B, INT8)",
    repo=COMFY_REPO,
    revision=COMFY_REV,
    path="text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    size=9_350_798_360,
    comfy_subdir="text_encoders",
)
CONFIG = ConfigSnapshot(
    key="config_dir",
    label="Configs, tokenizer and VAE (official repo)",
    repo="Qwen/Qwen-Image-2.1-Turbo",
    revision="d65dbc9a7e8f6b5479e33dee6030eaab2a906509",
    patterns=[
        "LICENSE",
        "model_index.json",
        "processor/*",
        "scheduler/*",
        "text_encoder/*.json",
        "transformer/config.json",
        "vae/*",
    ],
    size=690_000_000,
    required=(
        "model_index.json",
        "processor/tokenizer.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "transformer/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ),
)

WEIGHTS = (TRANSFORMER, TEXT_ENCODER)
TOTAL_DOWNLOAD = sum(f.size for f in WEIGHTS) + CONFIG.size

# ---------------------------------------------------------------- Apple Silicon (MLX)
# The official bf16 checkpoint, downloaded and then converted to MLX on the user's Mac (converted weights are never
# published). The download is deleted after converting.
MLX_SOURCE = ConfigSnapshot(
    key="mlx_source",
    label="Qwen-Image-2.1-Turbo (official weights, converted on this Mac)",
    repo="Qwen/Qwen-Image-2.1-Turbo",
    revision="d65dbc9a7e8f6b5479e33dee6030eaab2a906509",
    patterns=["LICENSE", "model_index.json", "scheduler/*", "processor/*", "vae/*", "transformer/*", "text_encoder/*"],
    size=32_500_000_000,
    required=(
        "model_index.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ),
)


@dataclass(frozen=True)
class MlxLevel:
    bits: int
    min_ram_gb: float  # this Mac's RAM, as macOS reports it
    size: int  # on disk after converting
    peak_gb: float  # measured, M4 Pro, 1024x1024 edit with one reference


# Measured 2026-10-12 on an M4 Pro with 48 GB. macOS lets the GPU use about 2/3 to 3/4 of
# RAM. 16 and 24 GB Macs don't fit yet, even at 4-bit.
MLX_LEVELS = (
    MlxLevel(bits=8, min_ram_gb=44, size=19_000_000_000, peak_gb=28.8),
    MlxLevel(bits=4, min_ram_gb=30, size=11_000_000_000, peak_gb=19.5),
)


def mlx_level_for(ram_gb: float) -> MlxLevel | None:
    return next((lvl for lvl in MLX_LEVELS if ram_gb >= lvl.min_ram_gb), None)
