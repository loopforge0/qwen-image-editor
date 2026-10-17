"""One loaded Qwen-Image-2.1-Turbo pipeline, set up for the card it runs on, with a single ``generate`` call.

Memory plan (``offload``):
- ``none``: everything on the GPU (24 GB+ cards).
- ``stream``: the transformer stays resident (it runs every step); the text encoder streams through the GPU one
  layer at a time from pinned RAM (it runs once per prompt); the VAE gets room for its encode/decode by having
  transformer blocks temporarily step aside. Fits a 12 GB card at 1 MP.
- ``model``: diffusers' whole-model offload. Slow; kept as a reference point.
"""

import math
import threading
import time
from pathlib import Path

import torch
from PIL import Image

from .common import ModelPaths, Result, tag_images  # noqa: F401  (re-exported)
from .attention import limit_int8_attention, use_int8_attention
from .loader import load_pipeline
from .offload import PartiallyEvictable, close_units, stream_text_encoder

# Reserved VRAM per megapixel, measured on the RTX 3060. The VAE's own weights are included.
_VAE_DECODE_GIB_PER_MP = 8.2
_VAE_ENCODE_GIB_PER_MP = 2.6
_VAE_SLACK_GIB = 0.4
_MAX_EVICT_BYTES = int(5.5 * 2**30)
# Transformer step memory. K and V of every prefix (text + reference) token, for 32 layers of 4096 channels in bf16.
_KV_BYTES_PER_TOKEN = 32 * 2 * 4096 * 2
_ACT_BYTES_PER_TOKEN = 160 * 2**10  # one block's activations; measured ~145 KB/token at 4k tokens, ~130 at 8k
_STEP_SLACK_GIB = 1.5  # the allocator reserves ~0.5 GiB over what is allocated; spilling past VRAM costs 2-6x


class _Stopwatch:
    """Wall-clock and VRAM peaks per stage, from forward hooks and CUDA syncs. Repeated calls of a stage add up."""

    def __init__(self, modules: dict):
        self.reset()
        for name, module in modules.items():
            module.register_forward_pre_hook(lambda m, a, n=name: self._start(n))
            module.register_forward_hook(lambda m, a, o, n=name: self._stop(n))

    def reset(self):
        self.times, self.peaks = {}, {}  # peaks: name -> [allocated GiB, reserved GiB]
        self.overall = 0
        torch.cuda.reset_peak_memory_stats()

    def _start(self, name):
        torch.cuda.synchronize()
        self.overall = max(self.overall, torch.cuda.max_memory_reserved())  # whatever ran between stages
        torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()

    def _stop(self, name):
        torch.cuda.synchronize()
        self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - self._t0
        alloc, reserved = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
        old = self.peaks.get(name, [0, 0])
        self.peaks[name] = [round(max(old[0], alloc), 2), round(max(old[1], reserved), 2)]
        self.overall = max(self.overall, torch.cuda.max_memory_reserved())


class _LoadProgress:
    """Loading progress as one fraction. Loading is mostly moving bytes: reading the weights from disk, then copying
    them to the GPU or into pinned RAM. Bytes read count fully; bytes copied in memory count less, being faster."""

    COPY_WEIGHT = 0.35

    def __init__(self, callback, phases):  # phases: [(key, label, nbytes, weight)]
        self.callback = callback
        self.phases = {key: [label, nbytes, weight, 0] for key, label, nbytes, weight in phases}
        self.total = sum(n * w for _, _, n, w in phases) or 1
        self.last = -1.0

    def __call__(self, key, nbytes):
        phase = self.phases[key]
        phase[3] += nbytes
        frac = min(1.0, sum(min(done, n) * w for _, n, w, done in self.phases.values()) / self.total)
        if frac - self.last >= 0.005:  # about 200 updates over a load
            self.last = frac
            self.callback(phase[0], frac)


class Engine:
    def __init__(self, paths: ModelPaths, offload: str = "auto", attn: str = "int8", vram_gb: float = 0, on_load=None):
        """``on_load(label, fraction)`` reports loading progress, e.g. ("Reading the image model", 0.42)."""
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        if vram_gb:  # simulate a smaller card
            torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb / total_gib))
            total_gib = vram_gb
        self.offload = offload if offload != "auto" else ("none" if total_gib >= 24 else "stream")
        self.attn = "int8" if attn == "int8" and use_int8_attention() else "sdpa"
        self.gpu = torch.cuda.get_device_name(0)
        self.lock = threading.Lock()  # one generation at a time
        self.last_plan = {}
        self._progress = None
        self._pinned_units = []

        t0 = time.perf_counter()
        sizes = {
            "vae": (Path(paths.config_dir) / "vae/diffusion_pytorch_model.safetensors").stat().st_size,
            "text_encoder": Path(paths.text_encoder).stat().st_size,
            "transformer": Path(paths.transformer).stat().st_size,
        }
        copy = _LoadProgress.COPY_WEIGHT
        phases = [
            ("vae", "Reading the VAE", sizes["vae"], 1),
            ("text_encoder", "Reading the text encoder", sizes["text_encoder"], 1),
            ("transformer", "Reading the image model", sizes["transformer"], 1),
            ("gpu", "Moving the image model to the GPU", sizes["transformer"], copy),
        ]
        if self.offload == "stream":
            phases.append(("pin_text_encoder", "Preparing the text encoder", sizes["text_encoder"], copy))
        progress = _LoadProgress(on_load or (lambda label, frac: None), phases)
        progress.callback("Starting", 0.0)
        pipe = load_pipeline(paths.transformer, paths.text_encoder, paths.config_dir, report=progress)
        if self.attn == "int8":
            limit_int8_attention(pipe.transformer)
        if self.offload == "model":
            pipe.enable_model_cpu_offload()
        elif self.offload == "stream":
            self._setup_stream(pipe, progress)
        else:
            pipe.to("cuda")
        progress("gpu", sizes["transformer"])
        self.pipe = pipe
        self.load_s = time.perf_counter() - t0
        self._watch = _Stopwatch(
            {
                "text_encoder": pipe.text_encoder,
                "vae_encode": pipe.vae.encoder,
                "transformer": pipe.transformer,
                "vae_decode": pipe.vae.decoder,
            }
        )
        # Stage reports for progress UIs: (stage, step, steps), stage in text | refs | steps | decode.
        for stage, module in (("text", pipe.text_encoder), ("refs", pipe.vae.encoder), ("decode", pipe.vae.decoder)):
            module.register_forward_pre_hook(lambda m, a, s=stage: self._report(s))
        pipe.transformer.register_forward_pre_hook(lambda m, a: self._report("steps", self._steps_done))

    def _report(self, stage, step=0):
        if self._progress:
            self._progress(stage, step, getattr(self.pipe, "_num_timesteps", 0) if stage == "steps" else 0)

    def _setup_stream(self, pipe, progress):
        transformer = PartiallyEvictable(pipe.transformer, list(pipe.transformer.transformer_blocks), _MAX_EVICT_BYTES)
        self._pinned_units = list(transformer.units)
        progress("gpu", progress.phases["gpu"][1])
        kv_caches = []

        def plan_steps(module, args, kwargs):
            """On the first step, stream as many blocks as the sequence needs (edits with references are long)."""
            if kwargs.get("kv_cache") is not None and kwargs["kv_cache"] not in kv_caches:
                kv_caches.append(kwargs["kv_cache"])
            if kwargs.get("kv_cache_mode") == "cached":
                return
            img_mask = kwargs["img_mask"]
            # The joint sequence: every vision slot (references and the target alike) expands to 2x2 latent tokens.
            tokens = img_mask.shape[1] + 3 * int(img_mask[0].sum())
            target = math.prod(kwargs["img_shapes"][0][-1])
            prefix = tokens - target
            need = _ACT_BYTES_PER_TOKEN * tokens + (_KV_BYTES_PER_TOKEN * prefix if kwargs.get("kv_cache") is not None else 0)
            need += int(_STEP_SLACK_GIB * 2**30)
            torch.cuda.empty_cache()
            missing = sum(u.nbytes for u in transformer.units if u.gpu is None)
            avail = torch.cuda.mem_get_info()[0] - missing
            count = 0
            if need > avail and transformer.unit_bytes:
                count = math.ceil((need - avail) / transformer.unit_bytes) + 2  # two blocks are in flight
            count = min(count, len(transformer.units))
            transformer.set_streamed(count)
            self.last_plan = {
                "tokens": tokens,
                "prefix_tokens": prefix,
                "streamed_blocks": count,
                "need_gib": round(need / 2**30, 2),
                "avail_gib": round(avail / 2**30, 2),
            }

        def drop_kv_caches():
            for cache in kv_caches:
                for layer in cache.layer_caches:
                    layer.k = layer.v = None
            kv_caches.clear()

        pipe.transformer.register_forward_pre_hook(plan_steps, with_kwargs=True, prepend=True)

        def make_room(gib_per_mp, pixels_per_cell):
            def hook(module, args):
                drop_kv_caches()  # the steps are over once the VAE runs; the reference K/V are dead weight
                h, w = args[0].shape[-2:]
                need = int((gib_per_mp * h * w * pixels_per_cell / 1e6 + _VAE_SLACK_GIB) * 2**30)
                torch.cuda.empty_cache()  # cached fragments can't serve the VAE's large buffers; give them back
                free = torch.cuda.mem_get_info()[0]
                if need > free:
                    transformer.evict(need - free)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            return hook

        release = lambda m, a, o: torch.cuda.empty_cache()  # noqa: E731  VAE buffers are huge; don't hoard them
        pipe.vae.encoder.register_forward_pre_hook(make_room(_VAE_ENCODE_GIB_PER_MP, 1))  # input is pixels
        pipe.vae.encoder.register_forward_hook(release)
        pipe.vae.decoder.register_forward_pre_hook(make_room(_VAE_DECODE_GIB_PER_MP, 256))  # input is 16x16 latents
        pipe.vae.decoder.register_forward_hook(release)
        pipe.vae.to("cuda")
        streamed = stream_text_encoder(pipe.text_encoder, report=lambda n: progress("pin_text_encoder", n))
        self._pinned_units += streamed.units
        pipe.text_encoder.register_forward_hook(release)  # streamed layer buffers

    def close(self):
        """Free the GPU memory and the pinned RAM, so the model can be loaded again later (unload when idle)."""
        import gc

        with self.lock:
            close_units(self._pinned_units)
            self._pinned_units = []
            self.pipe = None
            self._watch = None
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

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
        """Text-to-image, or an edit/composition conditioned on ``images`` (in order: <image1>, <image2>, ...; "image 1" works too).

        Without ``width``/``height``, the output takes the last reference image's aspect ratio at about 1 MP
        (1024x1024 without references). ``progress(stage, step, steps)`` is called as stages start and after each
        step; ``stage`` is one of text, refs, steps, decode.
        """
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
        images = [im.convert("RGBA") if im.mode not in ("RGB", "RGBA") else im for im in images or []]
        def callback(pipe, step, timestep, kwargs):
            self._steps_done = step + 1
            self._report("steps", step + 1)
            return {}

        with self.lock:
            self._watch.reset()
            self.last_plan = {}
            self._progress, self._steps_done = progress, 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            image = self.pipe(
                prompt=tag_images(prompt, len(images)),
                image=images or None,
                width=width,
                height=height,
                generator=torch.Generator("cpu").manual_seed(seed),
                use_kv_cache=use_kv_cache,
                callback_on_step_end=callback,
            ).images[0]
            self._progress = None
            torch.cuda.synchronize()
            self._watch.overall = max(self._watch.overall, torch.cuda.max_memory_reserved())
            return Result(
                image=image,
                seed=seed,
                width=image.width,
                height=image.height,
                total_s=time.perf_counter() - t0,
                stages_s=dict(self._watch.times),
                stage_peaks_gib=dict(self._watch.peaks),
                peak_reserved_gib=self._watch.overall / 2**30,
                plan=dict(self.last_plan),
            )
