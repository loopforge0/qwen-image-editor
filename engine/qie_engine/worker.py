"""The engine in a child process.

Unloading the model (when idle) then means ending that process: the operating system takes back every byte of
GPU memory, pinned RAM and caches, which freeing objects inside one long-lived process doesn't reliably do. A crash
in the GPU code also stays in the child. ``EngineProcess`` has the same interface as the engines themselves.
"""

import multiprocessing
import threading
import traceback


def _serve(conn, paths, kwargs):
    """Child process: load the engine, then answer generate requests until told to quit."""
    import sys

    if sys.stdout is None or sys.stderr is None:  # started from the tray (no console): log like the tray does
        from . import config

        log = open(config.home() / "editor.log", "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = sys.stderr = log
    try:
        from .common import create_engine

        engine = create_engine(paths, on_load=lambda label, frac: conn.send(("load", label, frac)), **kwargs)
        conn.send(("ready", engine.gpu, engine.offload, engine.attn, engine.load_s))
    except BaseException as exc:
        traceback.print_exc()
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:  # the editor went away
            return
        if msg[0] == "quit":
            return
        try:
            result = engine.generate(**msg[1], progress=lambda stage, step, steps: conn.send(("progress", stage, step, steps)))
            conn.send(("result", result))
        except BaseException as exc:
            traceback.print_exc()
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class EngineProcess:
    def __init__(self, paths, on_load=None, **kwargs):
        ctx = multiprocessing.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, paths, kwargs), daemon=True, name="qie-engine")
        self._proc.start()
        child.close()
        self.lock = threading.Lock()
        while True:
            msg = self._recv()
            if msg[0] == "load":
                if on_load:
                    on_load(msg[1], msg[2])
            elif msg[0] == "ready":
                self.gpu, self.offload, self.attn, self.load_s = msg[1:]
                return
            else:
                self.close()
                raise RuntimeError(msg[1])

    def _recv(self):
        try:
            return self._conn.recv()
        except EOFError:
            self._proc.join(5)
            raise RuntimeError(f"The engine stopped unexpectedly (exit code {self._proc.exitcode}); see the log.") from None

    def generate(self, prompt, images=None, width=None, height=None, seed=None, use_kv_cache=True, progress=None):
        with self.lock:
            self._conn.send(("generate", dict(prompt=prompt, images=images, width=width, height=height, seed=seed,
                                              use_kv_cache=use_kv_cache)))
            while True:
                msg = self._recv()
                if msg[0] == "progress":
                    if progress:
                        progress(*msg[1:])
                elif msg[0] == "result":
                    return msg[1]
                else:
                    raise RuntimeError(msg[1])

    def close(self):
        """End the engine process; its memory goes back to the system."""
        try:
            self._conn.send(("quit",))
        except (OSError, EOFError):
            pass
        self._proc.join(15)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(5)
