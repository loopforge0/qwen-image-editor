"""The editor as a tray (Windows) or menu-bar (macOS) app: the server runs in the background, the icon shows its
status and offers Open, Show log and Quit. Started by the installers' shortcuts as ``qie-tray`` (no console).

Output goes to the log file in the editor's data folder, since there's no console to show it.
"""

import importlib.util
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path

from . import config

ROOT = Path(__file__).resolve().parents[2]  # the editor's files (repository or install folder)
URL = "http://127.0.0.1:7861"


def log_to_file():
    """No console: send output (and the engine process's, which inherits it) to editor.log."""
    path = config.home() / "editor.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    log = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} started\n")
    sys.stdout = sys.stderr = log
    return path


def already_running() -> bool:
    try:
        urllib.request.urlopen(URL + "/api/status", timeout=1)
        return True
    except OSError:
        return False


def load_server():
    spec = importlib.util.spec_from_file_location("qie_server", ROOT / "app" / "server.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["qie_server"] = module
    spec.loader.exec_module(module)
    return module


def status_text(status: dict) -> str:
    state = status.get("state")
    if state == "needs_setup":
        return "Setup needed: open the editor to continue"
    if state == "setting_up":
        frac = (status.get("setup") or {}).get("fraction") or 0
        return f"Setting up · {frac * 100:.0f}%"
    if state == "loading":
        return f"Loading the model · {status.get('load_pct') or 0}%"
    if state == "idle":
        return "Idle: model unloaded to free memory"
    if state == "error":
        return "Error: see the log"
    if status.get("busy"):
        return "Generating"
    return f"Ready · {status.get('gpu', '')}".rstrip(" ·")


def open_path(path: Path):
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    else:
        subprocess.run(["open", str(path)], check=False)


def main():
    try:
        run()
    except Exception:
        # The Mac launcher shows its own dialog when this exits with an error; on Windows nothing else would.
        if sys.platform != "win32":
            raise
        traceback.print_exc()  # into editor.log, once log_to_file has run
        import ctypes

        message = f"Qwen Image Editor stopped with an error.\n\nDetails are in:\n{config.home() / 'editor.log'}"
        ctypes.windll.user32.MessageBoxW(None, message, "Qwen Image Editor", 0x10 | 0x10000)  # error icon, in front
        sys.exit(1)


def run():
    # --background: started by the MCP server for a chat app, so no browser tab (unless setup still has to be done).
    background = "--background" in sys.argv[1:]
    if already_running():
        if not background:
            webbrowser.open(URL)
        return
    log_path = log_to_file()
    server_module = load_server()
    httpd = server_module.start(server_module.parse_args(["--no-browser"] if background else []))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    import pystray
    from PIL import Image

    state = server_module.STATE

    def quit_editor(icon, _item):
        print("Quit from the tray.", flush=True)
        icon.stop()
        os._exit(0)  # the engine process sees its pipe close and exits too

    icon = pystray.Icon(
        "qwen-image-editor",
        Image.open(ROOT / "app" / "static" / "icon.png"),
        "Qwen Image Editor",
        pystray.Menu(
            pystray.MenuItem(lambda _item: status_text(state.status), None, enabled=False),
            pystray.MenuItem("Open Qwen Image Editor", lambda *_: webbrowser.open(URL), default=True),
            pystray.MenuItem("Show log", lambda *_: open_path(log_path)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", quit_editor),
        ),
    )

    def keep_status_current():
        last = None
        while True:
            text = status_text(state.status)
            if text != last:
                last = text
                icon.title = f"Qwen Image Editor: {text}"
                try:
                    icon.update_menu()
                except Exception:
                    pass
            time.sleep(1.5)

    threading.Thread(target=keep_status_current, daemon=True).start()
    if sys.platform == "darwin":  # a menu-bar item, not a Dock icon
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

            NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception:
            pass
    icon.run()


if __name__ == "__main__":
    main()
