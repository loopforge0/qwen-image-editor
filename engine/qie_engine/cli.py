"""The ``qie`` command.

    qie setup      check this machine, accept the license, find or download the models
    qie doctor     show what's set up and whether it works
    qie app        start the editor in the browser
    qie mcp        the MCP server for chat apps (Claude Desktop, Goose, ...); they start it themselves
    qie connect    add the editor to Claude Desktop and/or Goose (qie disconnect removes it)
    qie generate   generate or edit one image from the command line
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the repository


def _out(line=""):
    print(line, flush=True)


def _gb(n):
    return f"{n / 1e9:.2f} GB" if n < 10e9 else f"{n / 1e9:.1f} GB"


def _ask(question, default=False) -> bool | None:
    """Yes/no from the keyboard; None when nobody can answer (no terminal, or input closed)."""
    if not sys.stdin.isatty():
        return None
    hint = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {hint} ").strip().lower()
    except EOFError:  # e.g. stdin is NUL on Windows, which still reports itself as a terminal
        _out()
        return None
    return default if not answer else answer in ("y", "yes")


def _print_hardware(r):
    _out("This machine")
    if r.gpu:
        _out(f"  GPU    {r.gpu}" + (f" · {r.vram_gib} GB" if r.vram_gib else "") + (f" · driver {r.driver}" if r.driver else ""))
    if r.ram_gib:
        _out(f"  RAM    {r.ram_gib} GB")
    if r.free_disk_gib is not None:
        _out(f"  Disk   {r.free_disk_gib} GB free")
    for p in r.problems:
        _out(f"  ✗ {p}")
    for w in r.warnings:
        _out(f"  ! {w}")
    _out(f"  {'✓' if r.verdict == 'supported' else '!' if r.verdict == 'experimental' else '✗'} {r.summary}")


# ---------------------------------------------------------------- setup

def cmd_setup(args):
    from . import config, install
    from .models import LICENSE_NAME, LICENSE_URL

    models_dir = Path(args.models_dir or config.load().get("models_dir") or config.default_models_dir()).expanduser()
    _out("Qwen Image Editor setup")
    _out()
    found = install.locate(models_dir, args.comfyui, args.mlx_bits)
    report = install.hardware_check(models_dir, found)
    _print_hardware(report)
    _out()

    _out("Models")
    width = max(len(f.label) for f in found)
    for f in found:
        if f.path:
            _out(f"  ✓ {f.label:<{width}}  found ({f.source}): {f.path}")
        else:
            _out(f"  ↓ {f.label:<{width}}  {_gb(f.size)} to download")
    missing = [f for f in found if f.path is None]
    _out()

    if args.check:
        return 0 if report.verdict != "unsupported" else 1
    if report.verdict == "unsupported" and not args.force:
        _out("Setup stopped: fix the problems above, or run again with --force to set up anyway.")
        return 1

    accepted = bool(config.load().get("license", {}).get("accepted_at"))
    if not accepted:
        _out("License")
        _out(f"  Qwen-Image-2.1 is released under the {LICENSE_NAME}:")
        _out("    · non-commercial use only: research or evaluation")
        _out("    · commercial use needs a separate license from Qwen")
        _out(f"  Full text: {LICENSE_URL}")
        answer = True if args.accept_license else _ask("  Do you accept the license?")
        if args.accept_license:
            _out("  Accepted with --accept-license.")
        elif answer is None:
            _out("  Setup stopped: run it in a terminal to accept the license, or pass --accept-license.")
            return 1
        elif not answer:
            _out("  Setup stopped: the models can't be used without accepting the license.")
            return 1
        _out()

    if missing:
        total = sum(f.size for f in missing)
        if not args.yes and _ask(f"Download {_gb(total)} to {models_dir}?", default=True) is False:
            _out("Setup stopped before downloading. Point it at existing files with --comfyui PATH, or change the folder with --models-dir.")
            return 1
        _out(f"Downloading to {models_dir} (safe to interrupt; running setup again resumes)")
        models_dir.mkdir(parents=True, exist_ok=True)
        for f in missing:
            _out(f"  {f.label} · {_gb(f.size)}" + ("  (then converted for this Mac)" if f.key == "mlx_model" else ""))
        install.download_all(missing, models_dir)
        _out()

    install.record(found, models_dir, license_accepted=True)
    _out(f"Done. Settings saved to {config.config_path()}")
    _out("Start the editor with:  uv run qie app")
    return 0


# ---------------------------------------------------------------- doctor

def cmd_doctor(args):
    from . import config, hardware, install

    st = install.status()
    report = hardware.check(Path(config.load().get("models_dir") or config.default_models_dir()))
    if args.json:
        _out(json.dumps({"hardware": report.to_dict(), **st}, indent=2))
        return 0 if st["ready"] and report.verdict != "unsupported" else 1
    _print_hardware(report)
    _out()
    _out("Setup")
    _out(f"  Settings  {st['config_file']}")
    _out(f"  {'✓' if st['license_accepted'] else '✗'} License {'accepted' if st['license_accepted'] else 'not accepted yet'}")
    for m in st["models"]:
        _out(f"  {'✓' if m['ok'] else '✗'} {m['label']}: {m['path'] or 'not set up'}")
    _out()
    _out("Ready to go: uv run qie app" if st["ready"] else "Not ready yet. Run:  uv run qie setup")
    return 0 if st["ready"] else 1


# ---------------------------------------------------------------- app and generate

def cmd_app(args):
    import runpy

    sys.argv = [str(ROOT / "app/server.py")] + args.rest
    runpy.run_path(str(ROOT / "app/server.py"), run_name="__main__")
    return 0


def cmd_mcp(args):
    import runpy

    # stdout carries the MCP protocol: nothing else may print to it.
    runpy.run_path(str(ROOT / "app/mcp_server/server.py"), run_name="__main__")
    return 0


def cmd_connect(args):
    from . import integrations

    apps = {a["key"]: a for a in integrations.status()}
    keys = args.apps or [k for k, a in apps.items() if a["installed"]]
    if args.list or not keys:
        for a in apps.values():
            state = "connected" if a["connected"] else "installed, not connected" if a["installed"] else "not found"
            _out(f"{a['key']:8} {a['name']}: {state}")
        if not keys and not args.list:
            _out("Neither Claude Desktop nor Goose was found.")
        return 0
    code = 0
    for key in keys:
        try:
            app = (integrations.disconnect(key, args.this_install) if args.cmd == "disconnect"
                   else integrations.connect(key))
            if args.cmd == "disconnect" and integrations.entry_of(app):
                _out(f"Left {app.name} as it is: it uses another copy of the editor.")
                continue
            done = "Removed from" if args.cmd == "disconnect" else "Connected to"
            _out(f"{done} {app.name} ({app.config}). {app.restart}")
        except Exception as exc:
            _out(f"{key}: {exc}")
            code = 1
    return code


def cmd_generate(args):
    from PIL import Image

    from .common import create_engine
    from .paths import default_paths

    width = height = None
    if args.size:
        width, height = map(int, args.size.lower().split("x"))
    images = [Image.open(p) for p in args.image]
    engine = create_engine(default_paths(), vram_gb=args.vram_gb)
    _out(f"loaded in {engine.load_s:.1f}s  ({engine.gpu}, offload={engine.offload}, attn={engine.attn})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    for _ in range(args.count):
        r = engine.generate(args.prompt, images, width, height, seed, use_kv_cache=not args.no_kv_cache)
        path = out / f"{len(list(out.glob('*.png'))):04d}_s{r.seed}.png"
        r.image.save(path)
        stages = "  ".join(f"{k} {v:.2f}s" for k, v in r.stages_s.items())
        _out(f"{path}  {r.width}x{r.height}  {r.total_s:.2f}s  ({stages})  peak {r.peak_reserved_gib:.1f} GiB")
        seed = r.seed + 1
    return 0


def main(argv=None):
    if not sys.stdout.isatty() and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # piped output (an installer, a log) gets UTF-8, not the legacy code page
    ap = argparse.ArgumentParser(prog="qie", description="Qwen Image Editor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="check this machine, accept the license, find or download the models")
    p.add_argument("--models-dir", help="where to download the models (default: the editor's data folder)")
    p.add_argument("--comfyui", action="append", default=[], metavar="PATH", help="a folder to reuse model files from: ComfyUI, or any folder with them; repeatable")
    p.add_argument("--accept-license", action="store_true", help="accept the Qwen Research License without asking")
    p.add_argument("-y", "--yes", action="store_true", help="download without asking")
    p.add_argument("--check", action="store_true", help="only report; change nothing")
    p.add_argument("--force", action="store_true", help="set up even if this machine looks unsupported")
    p.add_argument("--mlx-bits", type=int, choices=[4, 8], help="Mac only: quantization level (default: chosen from this Mac's memory)")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("doctor", help="show what's set up and whether it works")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("app", help="start the editor in the browser (options: --port, --no-browser)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_app)

    p = sub.add_parser("mcp", help="the MCP server for chat apps (stdio)")
    p.set_defaults(fn=cmd_mcp)

    for name, what in (("connect", "add the editor to chat apps"), ("disconnect", "remove the editor from chat apps")):
        p = sub.add_parser(name, help=f"{what} (claude, goose; default: the ones installed)")
        p.add_argument("apps", nargs="*", metavar="APP")
        p.add_argument("--list", action="store_true", help="only show which apps are installed and connected")
        if name == "disconnect":
            p.add_argument("--this-install", action="store_true", help="only where the entry runs this copy of the editor")
        p.set_defaults(fn=cmd_connect)

    p = sub.add_parser("generate", help="generate or edit one image")
    p.add_argument("prompt")
    p.add_argument("-i", "--image", action="append", default=[], help="reference image; repeat for several (<image1>, <image2>, ... in the prompt)")
    p.add_argument("--size", help="WIDTHxHEIGHT; default: the last reference's shape at ~1 MP, or 1024x1024")
    p.add_argument("--seed", type=int)
    p.add_argument("-n", "--count", type=int, default=1, help="number of images (seed, seed+1, ...)")
    p.add_argument("-o", "--out", default="outputs/generate")
    p.add_argument("--no-kv-cache", action="store_true", help="recompute reference tokens every step (less VRAM, slower)")
    p.add_argument("--vram-gb", type=float, default=0, help="cap usable VRAM to simulate a smaller card")
    p.set_defaults(fn=cmd_generate)

    args, unknown = ap.parse_known_args(argv)
    if args.cmd == "app":
        args.rest = unknown + args.rest  # --port, --no-browser, ... belong to the server
    elif unknown:
        ap.error(f"unrecognized arguments: {' '.join(unknown)}")
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
