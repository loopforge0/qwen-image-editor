"""The editor's web server: prompt and reference images in, results and history out.

    uv run qie app            then open http://127.0.0.1:7861

Standard library only (besides the engine). Uploads and results are kept in the editor's data folder
(``qie_engine.config.home()/history``), with one JSON line per generation in history.jsonl, so the history
survives restarts.
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from qie_engine import config  # noqa: E402

DATA = config.home() / "history"
INPUTS = DATA / "inputs"
RESULTS = DATA / "results"
HISTORY = DATA / "history.jsonl"
OLD_DATA = ROOT / "outputs" / "app"  # where history lived before 2026-10-12
STATIC = Path(__file__).resolve().parent / "static"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}


class State:
    def __init__(self):
        self.engine = None
        self.status = {"state": "loading", "busy": False, "stage": None, "step": 0, "steps": 0}
        self.lock = threading.Lock()  # one generation at a time
        self.history_lock = threading.Lock()  # history.jsonl and the files it refers to
        self.last_used = time.time()
        self.idle_minutes = 30
        self.vram_gb = 0

    # ------------------------------------------------------------ unload when idle

    def watch_idle(self):
        """Unload the model after ``idle_minutes`` without a generation; the next one loads it again."""
        while True:
            time.sleep(20)
            idle_s = time.time() - self.last_used
            if self.idle_minutes and self.status["state"] == "ready" and not self.status["busy"] and idle_s > self.idle_minutes * 60:
                if not self.lock.acquire(blocking=False):
                    continue
                try:
                    print(f"Unloading the model after {self.idle_minutes} minutes without use.", flush=True)
                    engine, self.engine = self.engine, None
                    self.status.update(state="idle")
                    engine.close()
                finally:
                    self.lock.release()

    def ensure_loaded(self):
        """Called with self.lock held, before generating."""
        if self.status["state"] == "idle":
            self.load(self.vram_gb)
            if self.status["state"] != "ready":
                raise RuntimeError(self.status.get("error", "The model didn't load."))

    def load(self, vram_gb):
        try:
            from qie_engine.paths import default_paths
            from qie_engine.worker import EngineProcess

            def on_load(label, fraction):
                self.status.update(load_stage=label, load_pct=round(fraction * 100))

            self.status.update(state="loading", load_pct=0, load_stage="Starting")
            self.engine = EngineProcess(default_paths(), vram_gb=vram_gb, on_load=on_load)
            e = self.engine
            self.last_used = time.time()
            self.status.update(state="ready", gpu=e.gpu, offload=e.offload, attn=e.attn, load_s=round(e.load_s, 1))
            print(f"Model ready in {e.load_s:.1f} s ({e.gpu}).", flush=True)
        except Exception as exc:  # shown in the UI
            traceback.print_exc()
            self.status.update(state="error", error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ first-run setup (the same steps as `qie setup`)

    def setup_plan(self, comfyui=()):
        """What setup would do here: the hardware verdict, which model files exist where, what's left to download."""
        from qie_engine import config, install
        from qie_engine.common import backend
        from qie_engine.models import LICENSE_NAME, LICENSE_URL

        models_dir = Path(config.load().get("models_dir") or config.default_models_dir())
        found = install.locate(models_dir, [Path(p) for p in comfyui if p])
        report = install.hardware_check(models_dir, found)
        self._plan = (models_dir, found)
        return {
            "backend": backend(),
            "hardware": report.to_dict(),
            "models_dir": str(models_dir),
            "models": [{"label": f.label, "path": str(f.path) if f.path else None, "source": f.source, "size": f.size,
                        "convert": f.key == "mlx_model",
                        "convert_from": str(f.convert_from) if f.convert_from else None} for f in found],
            "download_bytes": sum(f.size for f in found if f.path is None),
            "disk_needed": install.disk_needed(found),
            "license": {"name": LICENSE_NAME, "url": LICENSE_URL, "accepted": install.status()["license_accepted"]},
            "chat_apps": chat_apps(),
        }

    def start_setup(self, comfyui, vram_gb):
        if self.status["state"] not in ("needs_setup", "error"):
            raise RuntimeError("Setup is already running or done.")
        plan = self.setup_plan(comfyui)
        self.status.update(state="setting_up", setup={"label": "Starting", "fraction": 0.0, "rate": 0, "tip": None})
        threading.Thread(target=self._run_setup, args=(vram_gb,), daemon=True).start()
        return plan

    def _run_setup(self, vram_gb):
        from qie_engine import install

        models_dir, found = self._plan
        try:
            install.accept_license()

            def on_progress(label, fraction, rate, tip):
                prev = self.status.get("setup") or {}
                self.status["setup"] = {"label": label, "fraction": round(fraction, 4), "rate": rate,
                                        "tip": tip or prev.get("tip")}

            missing = [f for f in found if f.path is None]
            if missing:
                install.download_all(missing, models_dir, on_progress)
            install.record(found, models_dir, license_accepted=True)
            print("Setup done.", flush=True)
        except Exception as exc:
            traceback.print_exc()
            self.status.update(state="needs_setup", setup_error=f"{type(exc).__name__}: {exc}")
            return
        self.status.update(state="loading", setup=None, setup_error=None)
        self.load(vram_gb)

    def _records(self):
        if not HISTORY.exists():
            return []
        return [json.loads(line) for line in HISTORY.read_text(encoding="utf-8").splitlines() if line.strip()]

    def history(self):
        with self.history_lock:
            return self._records()[::-1]

    def delete(self, ids, everything, keep):
        """Remove records and their result images, then any uploaded image nothing refers to any more.

        ``keep`` lists uploads still in use outside the history (the references in the composer)."""
        with self.history_lock:
            records = self._records()
            gone = [r for r in records if everything or r["id"] in ids]
            left = [r for r in records if r not in gone]
            tmp = HISTORY.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in left), encoding="utf-8")
            tmp.replace(HISTORY)
            for r in gone:
                (RESULTS / f"{r['id']}.png").unlink(missing_ok=True)
            used = set(keep)
            for r in left:
                used.update(r.get("refs", []))
                used.update(s["id"] for s in r.get("sources") or [])
            for f in INPUTS.glob("*.png"):
                if f.stem not in used:
                    f.unlink(missing_ok=True)
        return {"deleted": [r["id"] for r in gone]}

    def generate(self, req):
        refs = [Image.open(INPUTS / f"{ref}.png") for ref in req.get("refs", [])]
        width, height = req.get("width"), req.get("height")
        seed = req.get("seed")

        def progress(stage, step, steps):
            self.status.update(stage=stage, step=step, steps=steps)

        self.ensure_loaded()
        self.status.update(busy=True, stage="queued", step=0, steps=0, started=time.time())
        try:
            r = self.engine.generate(
                req.get("prompt", ""),
                refs,
                int(width) if width else None,
                int(height) if height else None,
                int(seed) if seed not in (None, "") else None,
                use_kv_cache=req.get("kv", True),
                progress=progress,
            )
        finally:
            self.status.update(busy=False, stage=None)
            self.last_used = time.time()
        rid = time.strftime("%Y%m%d-%H%M%S") + f"-{r.seed}"
        record = {
            "id": rid,
            "prompt": req.get("prompt", ""),
            "refs": req.get("refs", []),  # what the model saw (marks drawn in)
            "sources": req.get("sources"),  # the clean references and their marks, to reopen for editing
            "size_mode": req.get("size_mode", "auto"),
            "width": r.width,
            "height": r.height,
            "seed": r.seed,
            "kv": req.get("kv", True),
            "total_s": round(r.total_s, 2),
            "stages_s": {k: round(v, 2) for k, v in r.stages_s.items()},
            "peak_reserved_gib": round(r.peak_reserved_gib, 2),
            "plan": r.plan,
        }
        with self.history_lock:
            r.image.save(RESULTS / f"{rid}.png")
            with open(HISTORY, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return record


STATE = State()
ARGS = None  # command-line options, set by main()


def chat_apps() -> list[dict]:
    """Claude Desktop and Goose: whether they're installed, and connected to the editor (its MCP server)."""
    from qie_engine import integrations

    try:
        return integrations.status()
    except Exception as exc:
        print(f"Couldn't check the chat apps: {exc}", flush=True)
        return []


def save_upload(data: bytes) -> dict:
    image = Image.open(io.BytesIO(data))
    image.load()
    image = image.convert("RGBA" if "A" in image.getbands() or image.mode == "P" else "RGB")
    buf = io.BytesIO()
    image.save(buf, "PNG")
    ref = hashlib.sha256(buf.getvalue()).hexdigest()[:16]
    path = INPUTS / f"{ref}.png"
    if not path.exists():
        path.write_bytes(buf.getvalue())
    return {"id": ref, "width": image.width, "height": image.height}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if "/api/status" not in (args[0] if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def _send(self, code, body: bytes, ctype: str, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=31536000, immutable" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/static/index.html"
        if path.startswith("/static/"):
            file = STATIC / Path(path[len("/static/"):]).name  # no directory traversal
            if file.suffix in CONTENT_TYPES and file.exists():
                return self._send(200, file.read_bytes(), CONTENT_TYPES[file.suffix])
        if path == "/api/status":
            return self._json(STATE.status)
        if path == "/api/setup":
            return self._json(STATE.setup_plan())
        if path == "/api/history":
            return self._json(STATE.history())
        if path == "/api/chat-apps":
            return self._json(chat_apps())
        for prefix, folder in (("/files/inputs/", INPUTS), ("/files/results/", RESULTS)):
            if path.startswith(prefix):
                name = Path(path[len(prefix):]).name  # no directory traversal
                file = folder / name
                if file.suffix == ".png" and file.exists():
                    return self._send(200, file.read_bytes(), "image/png", cache=True)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            if self.path == "/api/quit":
                # Answer first, then stop: the page shows a "stopped" message.
                threading.Timer(0.3, lambda: os._exit(0)).start()
                print("Quit from the editor.", flush=True)
                return self._json({"ok": True})
            if self.path == "/api/setup/plan":
                return self._json(STATE.setup_plan(json.loads(body).get("comfyui", [])))
            if self.path == "/api/setup/start":
                req = json.loads(body)
                if not req.get("accept_license"):
                    return self._json({"error": "The license has to be accepted first."}, 400)
                return self._json(STATE.start_setup(req.get("comfyui", []), ARGS.vram_gb))
            if self.path == "/api/chat-apps":
                from qie_engine import integrations

                # It changes other apps' settings: only from the editor's own page. A JSON request from another
                # site needs a CORS preflight, which this server doesn't answer, and a page's Origin must match.
                origin = self.headers.get("Origin")
                if (self.headers.get("Content-Type") or "").split(";")[0] != "application/json" or (
                        origin and origin.split("://", 1)[-1] != self.headers.get("Host")):
                    return self._json({"error": "Only from the editor's page."}, 403)
                req = json.loads(body)
                (integrations.connect if req.get("connect") else integrations.disconnect)(req["key"])
                return self._json(chat_apps())
            if self.path == "/api/upload":
                return self._json(save_upload(body))
            if self.path == "/api/history/delete":
                req = json.loads(body)
                return self._json(STATE.delete(set(req.get("ids", [])), bool(req.get("all")), req.get("keep", [])))
            if self.path == "/api/generate":
                if STATE.status["state"] not in ("ready", "idle"):
                    return self._json({"error": "The model is still loading."}, 503)
                if not STATE.lock.acquire(blocking=False):
                    return self._json({"error": "Already generating."}, 409)
                try:
                    return self._json(STATE.generate(json.loads(body)))
                finally:
                    STATE.lock.release()
        except Exception as exc:
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        self._send(404, b"not found", "text/plain")


def in_container() -> bool:
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists() or os.environ.get("container"):
        return True
    try:
        return any(s in Path("/proc/1/cgroup").read_text() for s in ("docker", "containerd", "kubepods"))
    except OSError:
        return False


def move_old_history():
    """History used to live in the repository; it belongs with the user's data."""
    if (OLD_DATA / "history.jsonl").exists() and not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(OLD_DATA), str(DATA))
        print(f"Moved the editor history to {DATA}", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="qie app")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", help="address to listen on (default 127.0.0.1, or 0.0.0.0 inside a container); "
                                   "0.0.0.0 makes the editor reachable from other machines, and it has no login")
    ap.add_argument("--vram-gb", type=float, default=0, help="cap usable VRAM to simulate a smaller card")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser")
    ap.add_argument("--idle-minutes", type=float, help="unload the model after this many minutes without use "
                                                       "(default 30, or idle_minutes in the settings; 0 = never)")
    return ap.parse_args(argv)


def start(args) -> ThreadingHTTPServer:
    """Set everything up and open the port; the caller runs ``serve_forever`` (or a thread does, for the tray)."""
    container = in_container()
    # Inside a container, the container's own 127.0.0.1 can't be reached from outside; only ports published with
    # `docker run -p` can, so listening on every interface exposes nothing more than the user chose to publish.
    host = args.host or ("0.0.0.0" if container else "127.0.0.1")

    move_old_history()
    for d in (INPUTS, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    global ARGS
    ARGS = args
    STATE.vram_gb = args.vram_gb
    STATE.idle_minutes = args.idle_minutes if args.idle_minutes is not None else config.load().get("idle_minutes", 30)
    threading.Thread(target=STATE.watch_idle, daemon=True).start()
    from qie_engine import install, integrations

    if not container:
        threading.Thread(target=integrations.repair, daemon=True).start()  # chat apps pointing at a moved install

    if install.status()["ready"]:
        threading.Thread(target=STATE.load, args=(args.vram_gb,), daemon=True).start()
    else:
        STATE.status["state"] = "needs_setup"  # the page shows the setup screen
        print("First run: finish setup in the browser.", flush=True)
    server = ThreadingHTTPServer((host, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Qwen Image Editor: {url}   (loading the model; Ctrl+C to stop)", flush=True)
    if container:
        print(f"Running in a container: open {url} on the host, with the port published (docker run -p 127.0.0.1:{args.port}:{args.port}).", flush=True)
    elif host not in ("127.0.0.1", "localhost"):
        print(f"Listening on {host}: anyone who can reach this machine on port {args.port} can use the editor.", flush=True)
    if not args.no_browser and not container:
        webbrowser.open(url)
    return server


def main(argv=None):
    server = start(parse_args(argv))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        # Exit right away: a normal interpreter shutdown can stall on a thread that's inside the GPU library, and
        # the operating system frees the GPU and pinned memory anyway.
        print("\nStopping.", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
