"""Find or download the model files, record license acceptance, and write the config the engine reads.

Existing copies are used in place, matched by exact file size: the editor's own models folder, a ComfyUI install,
or the Hugging Face cache. Anything missing is downloaded into the models folder (resumable).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import config, hardware
from .common import backend
from .models import (
    CONFIG, LICENSE_NAME, LICENSE_URL, MLX_LEVELS, MLX_SOURCE, WEIGHTS, ConfigSnapshot, WeightsFile, mlx_level_for,
)

CONFIG_FOLDER = "Qwen-Image-2.1-Turbo"


@dataclass
class Found:
    key: str
    label: str
    path: Path | None  # an existing usable copy, or None if it has to be downloaded
    source: str  # "models folder" | "ComfyUI" | "Hugging Face cache" | "download"
    size: int
    convert_from: Path | None = None  # Mac: official weights already on disk, to convert instead of downloading


# ---------------------------------------------------------------- locating existing copies

def comfy_model_dirs(extra: list[Path] = ()) -> list[Path]:
    """ComfyUI ``models`` folders: the ones given, then the usual install locations."""
    roots = [Path(p).expanduser() for p in extra]
    if os.environ.get("COMFYUI_PATH"):
        roots.append(Path(os.environ["COMFYUI_PATH"]))
    home = Path.home()
    roots += [home / "ComfyUI", home / "Documents/ComfyUI", home / "ComfyUI_windows_portable"]
    if sys.platform == "win32":
        roots += [Path(f"{d}:/") / name for d in "CDE" for name in ("ComfyUI", "ComfyUI_windows_portable")]
    dirs = []
    for root in roots:
        for d in (root, root / "models", root / "ComfyUI/models"):
            if (d / "diffusion_models").is_dir() or (d / "text_encoders").is_dir():
                if d.resolve() not in [x.resolve() for x in dirs]:
                    dirs.append(d)
    return dirs


def _ok(path: Path, f: WeightsFile) -> bool:
    try:
        return path.is_file() and f.matches(path.stat().st_size)
    except OSError:
        return False


def _walk(folder: Path, depth: int):
    """Files in a folder the user chose and its subfolders, a few levels down (skipping hidden ones and Python
    environments, which in a ComfyUI install are large and hold no models)."""
    try:
        entries = list(folder.iterdir())
    except OSError:
        return
    for e in entries:
        try:
            if e.is_file():
                yield e
            elif depth > 0 and e.is_dir() and not e.name.startswith(".") and e.name not in ("venv", "node_modules", "__pycache__"):
                yield from _walk(e, depth - 1)
        except OSError:
            continue


def find_in_folder(f: WeightsFile, folder: Path, depth: int = 3) -> Path | None:
    """A model file in a folder the user chose, laid out like ComfyUI or not: by its name, or (if it was renamed) by its
    exact size."""
    name = Path(f.path).name
    sized = None
    for path in _walk(Path(folder).expanduser(), depth):
        if path.suffix == ".safetensors" and _ok(path, f):
            if path.name == name:
                return path
            sized = sized or path
    return sized


def find_weights(f: WeightsFile, models_dir: Path, comfy_dirs: list[Path], folders: list[Path] = ()) -> Found:
    previous = config.load().get("files", {}).get(f.key)
    if previous and _ok(Path(previous), f):
        return Found(f.key, f.label, Path(previous), "previous setup", f.size)
    own = models_dir / f.path
    if _ok(own, f):
        return Found(f.key, f.label, own, "models folder", f.size)
    name = Path(f.path).name
    for d in comfy_dirs:
        if _ok(d / f.comfy_subdir / name, f):
            return Found(f.key, f.label, d / f.comfy_subdir / name, "ComfyUI", f.size)
    for folder in folders:
        hit = find_in_folder(f, folder)
        if hit:
            return Found(f.key, f.label, hit, "folder you chose", f.size)
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(f.repo, f.path, revision=f.revision)
    if isinstance(cached, str) and _ok(Path(cached), f):
        return Found(f.key, f.label, Path(cached), "Hugging Face cache", f.size)
    return Found(f.key, f.label, None, "download", f.size)


def _snapshot_ok(path: Path, snap: ConfigSnapshot) -> bool:
    return all((path / r).is_file() for r in snap.required)


def _subdirs(folder: Path) -> list[Path]:
    try:
        return [d for d in folder.iterdir() if d.is_dir() and not d.name.startswith(".")]
    except OSError:
        return []


def find_config(snap: ConfigSnapshot, models_dir: Path, folders: list[Path] = ()) -> Found:
    previous = config.load().get("files", {}).get(snap.key)
    if previous and _snapshot_ok(Path(previous), snap):
        return Found(snap.key, snap.label, Path(previous), "previous setup", snap.size)
    own = models_dir / CONFIG_FOLDER
    if _snapshot_ok(own, snap):
        return Found(snap.key, snap.label, own, "models folder", snap.size)
    # A download of the official repo in a folder the user chose (the folder itself, or one inside it).
    for folder in [Path(f).expanduser() for f in folders]:
        for cand in [folder] + sorted((d for d in _subdirs(folder)), key=lambda d: d.name):
            if _snapshot_ok(cand, snap):
                return Found(snap.key, snap.label, cand, "folder you chose", snap.size)
    from huggingface_hub import constants, snapshot_download

    # The Hugging Face cache, at the pinned revision: looked at directly first, which doesn't depend on how the
    # library resolves things offline (once, on a fresh install, its lookup missed a complete snapshot).
    direct = Path(constants.HF_HUB_CACHE) / f"models--{snap.repo.replace('/', '--')}" / "snapshots" / snap.revision
    if _snapshot_ok(direct, snap):
        return Found(snap.key, snap.label, direct, "Hugging Face cache", snap.size)

    try:
        cached = Path(snapshot_download(snap.repo, revision=snap.revision, allow_patterns=snap.patterns, local_files_only=True))
        if _snapshot_ok(cached, snap):
            return Found(snap.key, snap.label, cached, "Hugging Face cache", snap.size)
    except Exception as exc:  # usually just "not in the cache"
        print(f"{snap.label}: not usable from the Hugging Face cache ({type(exc).__name__}: {exc})", flush=True)
    return Found(snap.key, snap.label, None, "download", snap.size)


def find_mlx(models_dir: Path, bits: int, folders: list[Path] = ()) -> Found:
    """Apple Silicon: MLX weights converted at ``bits``. Also accepts, from ``folders`` (chosen by the user): an
    existing MLX conversion (used as is, at its own level), or the official Turbo download (converted from)."""
    level = next(lvl for lvl in MLX_LEVELS if lvl.bits == bits)
    previous = config.load().get("files", {}).get("mlx_model")
    for path, source in ((previous, "previous setup"), (models_dir / mlx_folder(bits), "models folder")):
        if path and _mlx_ok(Path(path), bits):
            return Found("mlx_model", mlx_label(bits), Path(path), source, level.size)
    for folder in [Path(f).expanduser() for f in folders]:
        own_bits = mlx_bits_in(folder)
        if own_bits:
            return Found("mlx_model", mlx_label(own_bits), folder, "folder you chose", 0)
        if _snapshot_ok(folder, MLX_SOURCE):
            return Found("mlx_model", mlx_label(bits), None, "convert", 0, convert_from=folder)
    cached = _cached_official()
    if cached:
        return Found("mlx_model", mlx_label(bits), None, "convert", 0, convert_from=cached)
    # Otherwise: download the official weights (deleted again after converting), then convert.
    return Found("mlx_model", mlx_label(bits), None, "download", MLX_SOURCE.size)


def mlx_bits_in(folder: Path) -> int | None:
    """The quantization level of an MLX conversion made by mflux, or None if ``folder`` isn't one. (The official
    download also has transformer/config.json, but a diffusers index rather than mflux's model.safetensors.index.json.)"""
    index = folder / "transformer" / "model.safetensors.index.json"
    if not (index.is_file() and (folder / "text_encoder" / "config.json").is_file() and (folder / "vae" / "config.json").is_file()):
        return None
    try:
        level = json.loads(index.read_text()).get("metadata", {}).get("quantization_level")
        return int(level) if level not in (None, "None") else None
    except (OSError, ValueError):
        return None


def _cached_official() -> Path | None:
    """The official Turbo weights in the Hugging Face cache (e.g. from `hf download`), if they're all there."""
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(MLX_SOURCE.repo, revision=MLX_SOURCE.revision, allow_patterns=MLX_SOURCE.patterns,
                                      local_files_only=True))
        return path if _snapshot_ok(path, MLX_SOURCE) else None
    except Exception:
        return None


def mlx_label(bits: int) -> str:
    return f"Qwen-Image-2.1-Turbo for this Mac (MLX, {bits}-bit)"


def mlx_folder(bits: int) -> str:
    return f"Qwen-Image-2.1-Turbo-MLX-{bits}bit"


def _bits_of(found: Found) -> int:
    return int(found.label.rsplit(", ", 1)[-1].split("-")[0])


def _mlx_ok(path: Path, bits: int) -> bool:
    return mlx_bits_in(path) == bits


def mlx_bits(requested: int | None = None) -> int | None:
    if requested:
        return requested
    ram = hardware.system_ram_bytes()
    level = mlx_level_for(ram / 2**30) if ram else None
    return level.bits if level else None


def locate(models_dir: Path, comfyui: list[Path] = (), bits: int | None = None) -> list[Found]:
    if backend() == "mlx":
        return [find_mlx(models_dir, mlx_bits(bits) or MLX_LEVELS[-1].bits, comfyui)]
    dirs = comfy_model_dirs(comfyui)
    folders = [Path(p).expanduser() for p in comfyui]
    return [find_weights(f, models_dir, dirs, folders) for f in WEIGHTS] + [find_config(CONFIG, models_dir, folders)]


# ---------------------------------------------------------------- downloading

SLOW_BYTES_PER_S = 10e6
SLOW_TIP = (
    "Slow download? A free Hugging Face access token sometimes helps: create one at "
    "https://huggingface.co/settings/tokens, set it as HF_TOKEN, and run setup again."
)


def _reporting_tqdm(on_progress, label: str):
    """A progress-bar class for huggingface_hub that also reports bytes (or files) done to ``on_progress``."""
    from tqdm.auto import tqdm as base

    class Reporting(base):
        # Byte bars report even without a total: hf-xet's "Downloading bytes" bar has none, and its
        # "Reconstructing" bar only learns its total at the end. The expected size comes from the manifest instead.
        def update(self, n=1):
            out = super().update(n)
            if self.unit == "B":  # snapshot_download's files-done bar would jump: a dozen tiny files, then one big one
                on_progress(id(self), label, self.n)
            return out

    return Reporting


class _Progress:
    """Per-file progress -> one stream of (label, fraction of everything, bytes/s), plus the slow-download tip."""

    def __init__(self, total_bytes: int, callback):
        self.total, self.callback = max(total_bytes, 1), callback
        self.done_before, self.current_size, self.bars = 0, 0, {}
        self.t0, self.tip_shown = time.monotonic(), False
        from huggingface_hub import get_token

        self.has_token = bool(get_token())

    def start_file(self, size):
        self.current_size, self.t0, self.bars = size, time.monotonic(), {}

    def finish_file(self, size):
        self.done_before += size

    def __call__(self, bar, label, n):
        self.bars[bar] = n
        in_file = min(max(self.bars.values()), self.current_size)
        elapsed = time.monotonic() - self.t0
        rate = in_file / elapsed if elapsed > 0 else 0.0
        tip = None
        if not self.has_token and elapsed > 45 and rate < SLOW_BYTES_PER_S and not self.tip_shown:
            self.tip_shown, tip = True, SLOW_TIP
        self.callback(label=label, fraction=min((self.done_before + in_file) / self.total, 0.999), rate=rate, tip=tip)


def download_all(missing: list[Found], models_dir: Path, on_progress=None) -> None:
    """Download (and on a Mac, convert) everything in ``missing``; fills in each one's path.

    ``on_progress(label=..., fraction=..., rate=..., tip=...)`` is called as data arrives. Without it, the
    slow-download tip is printed."""

    def relay(**kw):
        if on_progress:
            on_progress(**kw)
        elif kw.get("tip"):
            print(f"\n  {kw['tip']}\n", flush=True)

    progress = _Progress(sum(f.size for f in missing), relay)
    models_dir.mkdir(parents=True, exist_ok=True)
    for f in missing:
        progress.start_file(f.size)
        f.path = _download_one(f, models_dir, progress)
        progress.finish_file(f.size)
        f.source = "downloaded"


def _download_one(found: Found, models_dir: Path, progress: _Progress) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    bar = _reporting_tqdm(progress, f"Downloading: {found.label}")
    if found.key == CONFIG.key:
        path = Path(snapshot_download(CONFIG.repo, revision=CONFIG.revision, allow_patterns=CONFIG.patterns,
                                      local_dir=models_dir / CONFIG_FOLDER, tqdm_class=bar))
        if not _snapshot_ok(path, CONFIG):
            raise RuntimeError(f"The download of {CONFIG.repo} is incomplete; run setup again to resume.")
        return path
    if found.key == "mlx_model":
        return _download_and_convert_mlx(found, models_dir, progress, bar)
    f = next(w for w in WEIGHTS if w.key == found.key)
    path = Path(hf_hub_download(f.repo, f.path, revision=f.revision, local_dir=models_dir, tqdm_class=bar))
    if not _ok(path, f):
        raise RuntimeError(f"{path.name} has the wrong size after downloading; delete it and run setup again.")
    return path


def _download_and_convert_mlx(found: Found, models_dir: Path, progress: _Progress, bar) -> Path:
    from huggingface_hub import snapshot_download

    bits = _bits_of(found)
    target = models_dir / mlx_folder(bits)
    if found.convert_from:  # the user's own copy (or the Hugging Face cache): convert from it, never delete it
        source, ours = found.convert_from, False
    else:
        source, ours = models_dir / "Qwen-Image-2.1-Turbo-bf16", True
        snapshot_download(MLX_SOURCE.repo, revision=MLX_SOURCE.revision, allow_patterns=MLX_SOURCE.patterns,
                          local_dir=source, tqdm_class=bar)
        if not _snapshot_ok(source, MLX_SOURCE):
            raise RuntimeError(f"The download of {MLX_SOURCE.repo} is incomplete; run setup again to resume.")
    progress.callback(label=f"Converting the model for this Mac ({bits}-bit; a few minutes)", fraction=0.999, rate=0.0, tip=None)
    # In its own process, so the converter's memory goes back to macOS as soon as it's done.
    code = (
        "import sys; from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit; "
        "QwenImage21Edit(quantize=int(sys.argv[1]), model_path=sys.argv[2]).save_model(sys.argv[3])"
    )
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    subprocess.run([sys.executable, "-c", code, str(bits), str(source), str(partial)], check=True)
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    if not _mlx_ok(target, bits):
        raise RuntimeError(f"The conversion to {target} looks incomplete; run setup again.")
    if ours:
        shutil.rmtree(source, ignore_errors=True)  # our own 32 GB download, only needed for converting
    return target


# ---------------------------------------------------------------- status (used by `qie doctor` and the app)

def status() -> dict:
    """What's configured and whether it's usable, without touching the network."""
    cfg = config.load()
    files = cfg.get("files", {})
    items = []
    if backend() == "mlx":
        p, bits = files.get("mlx_model"), cfg.get("mlx_bits")
        items.append({"key": "mlx_model", "label": mlx_label(bits) if bits else "Qwen-Image-2.1-Turbo for this Mac (MLX)",
                      "path": p, "ok": bool(p) and bool(bits) and _mlx_ok(Path(p), bits)})
    else:
        for f in WEIGHTS:
            p = files.get(f.key)
            items.append({"key": f.key, "label": f.label, "path": p, "ok": bool(p) and _ok(Path(p), f)})
        p = files.get(CONFIG.key)
        items.append({"key": CONFIG.key, "label": CONFIG.label, "path": p, "ok": bool(p) and _snapshot_ok(Path(p), CONFIG)})
    accepted = bool(cfg.get("license", {}).get("accepted_at"))
    return {
        "config_file": str(config.config_path()),
        "backend": backend(),
        "license_accepted": accepted,
        "models": items,
        "ready": accepted and all(i["ok"] for i in items),
    }


def accept_license():
    cfg = config.load()
    if not cfg.get("license", {}).get("accepted_at"):
        cfg["license"] = {"name": LICENSE_NAME, "url": LICENSE_URL, "accepted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        config.save(cfg)


def record(found: list[Found], models_dir: Path, license_accepted: bool):
    if license_accepted:
        accept_license()
    cfg = config.load()
    cfg["version"] = 1
    cfg["models_dir"] = str(models_dir)
    cfg["files"] = {f.key: str(f.path) for f in found if f.path}
    for f in found:
        if f.key == "mlx_model":
            cfg["mlx_bits"] = _bits_of(f)
    config.save(cfg)


def disk_needed(found: list[Found]) -> int:
    """Free disk needed: downloads, plus on a Mac the converted copy that briefly sits next to the download."""
    need = 0
    for f in found:
        if f.path is None:
            need += f.size
            if f.key == "mlx_model":
                need += next(lvl.size for lvl in MLX_LEVELS if lvl.bits == _bits_of(f))
    return need


def hardware_check(models_dir: Path, found: list[Found]) -> hardware.Report:
    return hardware.check(models_dir, disk_needed(found))
