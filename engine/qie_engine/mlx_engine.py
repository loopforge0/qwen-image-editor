"""The Apple Silicon engine: Qwen-Image-2.1-Turbo in MLX, through mflux.

One model does both text-to-image and edits: mflux's edit class, which also generates without reference images,
and which quantizes the text encoder too (its text-to-image class keeps a 17.5 GB bf16 encoder). The weights are the
official Turbo checkpoint, converted on this Mac by ``qie setup``.

Measured on an M4 Pro (48 GB), 1024x1024: ~95 s text-to-image, ~150 s edit, at any quantization level; peak
memory 29 GB (8-bit) / 19.5 GB (4-bit). Quantization saves memory, not time: MLX computes in 16-bit either way.
"""

import platform
import subprocess
import threading
import time

from PIL import Image

from .common import ModelPaths, Result, tag_images

STEPS = 8
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]  # then 0
_SCHEDULER = "viggle_turbo"  # the slot our scheduler takes over; see install_turbo_scheduler


def install_turbo_scheduler():
    """Make mflux sample on Turbo's 8 trained sigmas.

    mflux ships the base model's 40-step schedule and a 6-step one for a community LoRA, and its edit path only
    accepts those two names. So the "viggle_turbo" name is pointed at a scheduler with the official Turbo sigmas.
    No LoRA is involved. Must run after the model is built (building it registers mflux's own schedulers)."""
    import mlx.core as mx
    import mflux.models.qwen21.model.qwen21_scheduler  # noqa: F401
    from mflux.models.common import schedulers
    from mflux.models.common.schedulers.base_scheduler import BaseScheduler

    class TurboSigmas(BaseScheduler):
        def __init__(self, config):
            self.config = config
            self._sigmas = mx.array(TURBO_SIGMAS + [0.0], dtype=mx.float32)

        @property
        def sigmas(self):
            return self._sigmas

        def step(self, noise, timestep, latents, **kwargs):
            return latents + noise * (self._sigmas[timestep + 1] - self._sigmas[timestep])

    try:
        schedulers.register_contrib(TurboSigmas, _SCHEDULER)
    except Exception:
        schedulers.SCHEDULER_REGISTRY[_SCHEDULER] = TurboSigmas


def chip_name() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or "Apple Silicon"


def _peak_gib() -> float:
    import mlx.core as mx

    return (getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory)() / 2**30


def _reset_peak():
    import mlx.core as mx

    (getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory)()


class MlxEngine:
    def __init__(self, paths: ModelPaths, on_load=None):
        import mlx.core as mx
        from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

        on_load = on_load or (lambda label, fraction: None)
        self.gpu = chip_name()
        self.offload, self.attn = "unified memory", "MLX"
        self.lock = threading.Lock()
        self._progress = None
        self._steps_done = 0

        t0 = time.perf_counter()
        on_load("Opening the model", 0.0)
        self.model = QwenImage21Edit(model_path=str(paths.mlx_dir))
        install_turbo_scheduler()
        # MLX loads weights lazily, on first use. Load them now, so "ready" means ready and the first image isn't
        # slower than the rest.
        parts = [(name, getattr(self.model, name, None)) for name in ("vae", "text_encoder", "transformer")]
        parts = [(name, module) for name, module in parts if module is not None and hasattr(module, "parameters")]
        labels = {"vae": "Reading the VAE", "text_encoder": "Reading the text encoder", "transformer": "Reading the image model"}
        for i, (name, module) in enumerate(parts):
            on_load(labels[name], i / max(len(parts), 1))
            mx.eval(module.parameters())
        self._count_steps()
        self.load_s = time.perf_counter() - t0
        on_load("Ready", 1.0)

    def close(self):
        """Free the model's memory (unload when idle)."""
        import gc

        import mlx.core as mx

        with self.lock:
            self.model = None
            gc.collect()
            (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()

    def _count_steps(self):
        """Report progress per denoising step by counting transformer calls (one per step at guidance 1)."""
        transformer = getattr(self.model, "transformer", None)
        if transformer is None:
            return
        engine, base = self, type(transformer)

        def __call__(module, *args, **kwargs):
            engine._report("steps", engine._steps_done)
            out = base.__call__(module, *args, **kwargs)
            engine._steps_done += 1
            if engine._steps_done >= STEPS:
                engine._report("decode")  # the transformer's done; the VAE decode comes next
            else:
                engine._report("steps", engine._steps_done)
            return out

        transformer.__class__ = type(base.__name__, (base,), {"__call__": __call__})

    def _report(self, stage, step=0):
        if self._progress:
            self._progress(stage, min(step, STEPS), STEPS if stage == "steps" else 0)

    def generate(
        self,
        prompt: str,
        images: list[Image.Image] | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        use_kv_cache: bool = True,
        progress=None,
    ) -> Result:
        """Same contract as the CUDA engine: references in order (<image1>, ...); without width/height the output
        takes the last reference's shape at about 1 MP, or 1024x1024 without references."""
        import random

        if seed is None:
            seed = random.randrange(2**31 - 1)
        images = [im.convert("RGBA") if im.mode not in ("RGB", "RGBA") else im for im in images or []]
        if (width is None or height is None) and not images:
            width, height = 1024, 1024
        if width and height:  # mflux wants multiples of 32
            width, height = max(256, width // 32 * 32), max(256, height // 32 * 32)
        with self.lock:
            self._progress, self._steps_done = progress, 0
            try:
                self._report("text")
                _reset_peak()
                t0 = time.perf_counter()
                out = self.model.generate_image(
                    seed=seed,
                    prompt=tag_images(prompt, len(images)),
                    num_inference_steps=STEPS,
                    width=width,
                    height=height,
                    image_paths=images or None,
                    use_kv_cache=use_kv_cache,
                    scheduler=_SCHEDULER,
                )
                total = time.perf_counter() - t0
            finally:
                self._progress = None
        image = out.image
        return Result(
            image=image,
            seed=seed,
            width=image.width,
            height=image.height,
            total_s=total,
            stages_s={},
            peak_reserved_gib=_peak_gib(),
        )
