"""Where the editor keeps its settings and models, and the settings file itself.

Everything lives in one folder per user, so removing it removes the editor's data:
- Windows: %LOCALAPPDATA%\\qwen-image-editor
- macOS:   ~/Library/Application Support/qwen-image-editor
- Linux:   $XDG_DATA_HOME/qwen-image-editor (default ~/.local/share/qwen-image-editor)

``QIE_HOME`` overrides the folder.
"""

import json
import os
import sys
from pathlib import Path

APP = "qwen-image-editor"


def home() -> Path:
    if os.environ.get("QIE_HOME"):
        return Path(os.environ["QIE_HOME"]).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / APP
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support" / APP
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / APP


def config_path() -> Path:
    return home() / "config.json"


def default_models_dir() -> Path:
    return home() / "models"


def load() -> dict:
    try:
        return json.loads(config_path().read_text(encoding="utf-8-sig"))  # utf-8-sig: Notepad and PowerShell add a BOM
    except FileNotFoundError:
        return {}


def save(cfg: dict):
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tmp.replace(path)
