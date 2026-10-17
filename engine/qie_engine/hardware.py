"""Can this machine run the editor? A plain-language verdict plus the numbers behind it."""

import ctypes
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

GIB = 2**30
MIN_DRIVER = 580  # PyTorch's CUDA 13.0 build needs an R580 or newer NVIDIA driver


@dataclass
class Report:
    verdict: str = "unsupported"  # supported | experimental | unsupported
    summary: str = ""
    problems: list = field(default_factory=list)  # each one stops the editor from working
    warnings: list = field(default_factory=list)  # it may work, but slowly or not at all
    os: str = ""
    gpu: str | None = None
    vram_gib: float | None = None
    compute_capability: str | None = None
    driver: str | None = None
    ram_gib: float | None = None
    free_disk_gib: float | None = None

    def to_dict(self):
        return asdict(self)


def system_ram_bytes() -> int | None:
    try:
        if sys.platform == "win32":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong)
                    for n in ("total", "avail", "totalPage", "availPage", "totalVirtual", "availVirtual", "availExt")
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def nvidia_driver() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output([exe, "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=10)
        return out.strip().splitlines()[0]
    except Exception:
        return None


def check(disk_path: Path | None = None, need_disk_bytes: int = 0) -> Report:
    r = Report(os=f"{platform.system()} {platform.release()} ({platform.machine()})")
    ram = system_ram_bytes()
    r.ram_gib = round(ram / GIB, 1) if ram else None
    if disk_path is not None:
        probe = Path(disk_path)
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        r.free_disk_gib = round(shutil.disk_usage(probe).free / GIB, 1)

    if sys.platform == "darwin":
        _check_mac(r)
        return _verdict(r)
    r.driver = nvidia_driver()

    import torch

    if not torch.cuda.is_available():
        if r.driver:
            r.problems.append(
                f"An NVIDIA driver ({r.driver}) is installed but CUDA can't be used. Update the driver to "
                f"version {MIN_DRIVER} or newer."
            )
        elif sys.platform != "darwin":
            r.problems.append("No NVIDIA GPU was found. The editor needs an NVIDIA GPU with 12 GB of memory or more.")
    else:
        props = torch.cuda.get_device_properties(0)
        r.gpu = props.name
        r.vram_gib = round(props.total_memory / GIB, 1)
        r.compute_capability = f"{props.major}.{props.minor}"
        if r.vram_gib < 7.5:
            r.problems.append(f"The {r.gpu} has {r.vram_gib} GB of memory; the editor needs at least 8 GB (12 GB or more recommended).")
        elif r.vram_gib < 11.5:
            r.warnings.append(f"The {r.gpu} has {r.vram_gib} GB of memory. 8 GB cards are untested and may be slow; 12 GB or more is recommended.")
        if (props.major, props.minor) < (8, 0):
            r.warnings.append(f"The {r.gpu} is older than the RTX 30 series and is untested.")
        if r.driver:
            try:
                if int(r.driver.split(".")[0]) < MIN_DRIVER:
                    r.problems.append(f"NVIDIA driver {r.driver} is too old; update to version {MIN_DRIVER} or newer.")
            except ValueError:
                pass

    if r.ram_gib is not None:
        if r.ram_gib < 15:
            r.problems.append(f"This machine has {r.ram_gib} GB of RAM; the editor needs at least 16 GB (32 GB recommended).")
        elif r.ram_gib < 30:
            r.warnings.append(f"This machine has {r.ram_gib} GB of RAM. 16 GB is untested; 32 GB is recommended.")
    if r.free_disk_gib is not None and need_disk_bytes and r.free_disk_gib * GIB < need_disk_bytes:
        r.problems.append(f"Not enough free disk space: {need_disk_bytes / GIB:.1f} GB needed, {r.free_disk_gib} GB free.")

    return _verdict(r)


def _verdict(r: Report) -> Report:
    if r.problems:
        r.verdict, r.summary = "unsupported", "This machine can't run the editor yet."
    elif r.warnings:
        r.verdict, r.summary = "experimental", "This machine should work, with the notes above."
    else:
        r.verdict, r.summary = "supported", "This machine is supported."
    return r


def _check_mac(r: Report):
    """Apple Silicon: MLX needs macOS 14+; memory decides the quantization level (models.MLX_LEVELS)."""
    from .models import MLX_LEVELS, mlx_level_for

    if platform.machine() != "arm64":
        r.problems.append("Intel Macs aren't supported: the editor needs an Apple Silicon Mac (M1 or newer).")
        return
    try:
        r.gpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        r.gpu = "Apple Silicon"
    major = int((platform.mac_ver()[0] or "0").split(".")[0])
    if major < 14:
        r.problems.append(f"macOS {platform.mac_ver()[0]} is too old; the editor needs macOS 14 (Sonoma) or newer.")
    if r.ram_gib is not None:
        level = mlx_level_for(r.ram_gib)
        if level is None:
            lowest = MLX_LEVELS[-1]
            r.problems.append(
                f"This Mac has {r.ram_gib} GB of memory; the editor needs a Mac with 32 GB or more "
                f"(it uses about {lowest.peak_gb:.0f} GB at its smallest setting)."
            )
    # M5 GPUs have matrix units that MLX uses; earlier chips do this work on general-purpose GPU cores.
    if r.gpu and any(f"M{n}" in r.gpu for n in (1, 2, 3, 4)):
        r.warnings.append(
            f"On the {r.gpu}, expect roughly 1.5-2.5 minutes per 1024x1024 image (an M4 Pro measured ~95 s, ~150 s "
            f"for an edit). M5 Macs are several times faster."
        )
