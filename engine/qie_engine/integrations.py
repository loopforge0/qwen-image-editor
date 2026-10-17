"""Connect the editor to chat apps (Claude Desktop, Goose, ChatGPT) as an MCP server.

Each app's config file gets an entry that runs ``qie mcp`` from this install, through the same uv and project as the
editor itself. The rest of the file is left as it is, and a copy of it is kept before the first change.
"""

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

NAME = "qwen-image-editor"
ROOT = Path(__file__).resolve().parents[2]  # the project folder: the repo, or the installed app


@dataclass
class ChatApp:
    key: str
    name: str
    config: Path
    installed: bool
    restart: str  # how to make the app load the change


def _home(*parts) -> Path:
    return Path.home().joinpath(*parts)


def chat_apps() -> list[ChatApp]:
    # The ChatGPT desktop app keeps its MCP servers in the Codex settings it shares with Codex.
    codex = Path(os.environ["CODEX_HOME"]) if os.environ.get("CODEX_HOME") else _home(".codex")
    chatgpt_cfg = codex / "config.toml"
    chatgpt = codex.is_dir()
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", _home("AppData/Roaming")))
        local = Path(os.environ.get("LOCALAPPDATA", _home("AppData/Local")))
        claude_cfg = appdata / "Claude/claude_desktop_config.json"
        claude = (claude_cfg.parent.is_dir() or (local / "AnthropicClaude").is_dir()
                  or any((local / "Packages").glob("Claude_*")))
        goose_cfg = appdata / "Block/goose/config/config.yaml"
        goose = goose_cfg.parent.is_dir() or (local / "Programs/Goose").is_dir()
        chatgpt = chatgpt or any((local / "Packages").glob("OpenAI.Codex_*"))  # the Store package's name
    elif sys.platform == "darwin":
        apps = [Path("/Applications"), _home("Applications")]
        claude_cfg = _home("Library/Application Support/Claude/claude_desktop_config.json")
        claude = claude_cfg.parent.is_dir() or any((a / "Claude.app").is_dir() for a in apps)
        goose_cfg = _home(".config/goose/config.yaml")
        goose = goose_cfg.parent.is_dir() or any((a / "Goose.app").is_dir() for a in apps)
        chatgpt = chatgpt or any((a / "ChatGPT.app").is_dir() for a in apps)
    else:
        claude_cfg, claude = _home(".config/Claude/claude_desktop_config.json"), False
        goose_cfg = _home(".config/goose/config.yaml")
        goose = goose_cfg.parent.is_dir()
    where = "the menu bar" if sys.platform == "darwin" else "its tray icon"
    return [
        ChatApp("claude", "Claude Desktop", claude_cfg, claude, f"Quit Claude Desktop (from {where}) and open it again."),
        ChatApp("goose", "Goose", goose_cfg, goose, "Quit Goose and open it again."),
        ChatApp("chatgpt", "ChatGPT", chatgpt_cfg, chatgpt, "Quit ChatGPT and open it again."),
    ]


def get_app(key: str) -> ChatApp:
    for a in chat_apps():
        if a.key == key:
            return a
    raise ValueError(f"Unknown chat app {key!r}: use " + ", ".join(a.key for a in chat_apps()))


def mcp_command() -> tuple[str, list[str], dict]:
    """How a chat app starts the MCP server: (program, arguments, environment)."""
    if "/AppTranslocation/" in str(ROOT) or str(ROOT).startswith("/Volumes/"):
        # Opened from the disk image, or from a temporary copy macOS makes of apps opened straight from a download:
        # gone after a restart.
        raise RuntimeError("Move Qwen Image Editor to the Applications folder first, then open it from there.")
    uv = os.environ.get("UV") or shutil.which("uv")  # uv sets UV for the programs it runs
    if not uv:
        raise RuntimeError("Couldn't find uv, which runs the editor.")
    uv = Path(uv)
    if uv.name.lower() == "uvw.exe" and (uv.parent / "uv.exe").exists():
        uv = uv.parent / "uv.exe"  # uvw has no console, so no stdin/stdout for the chat app to talk over
    env = {k: os.environ[k] for k in ("UV_PROJECT_ENVIRONMENT",) if os.environ.get(k)}  # the Mac app's venv
    # --no-sync: use the environment the installer or the editor set up. Syncing on every start would fail while
    # another chat app is running the server (its files are in use) and slows the start.
    return str(uv), ["run", "--no-sync", "--project", str(ROOT), "qie", "mcp"], env


# ---------------------------------------------------------------- the config files

def _read(app: ChatApp) -> dict:
    if not app.config.exists():
        return {}
    text = app.config.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    if app.key == "goose":
        import yaml

        data = yaml.safe_load(text)
    elif app.key == "chatgpt":
        import tomllib

        data = tomllib.loads(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{app.config} isn't a settings file this can change.")
    return data


def _write(app: ChatApp, data: dict):
    app.config.parent.mkdir(parents=True, exist_ok=True)
    backup = app.config.with_name(app.config.name + f".before-{NAME}")
    if app.config.exists() and not backup.exists():
        shutil.copy2(app.config, backup)
    if app.key == "goose":
        import yaml

        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp = app.config.with_name(app.config.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))  # bytes: no CRLF on Windows
    os.replace(tmp, app.config)


def _section(app: ChatApp) -> str:
    return {"goose": "extensions", "chatgpt": "mcp_servers"}.get(app.key, "mcpServers")


# The Codex settings file is TOML with comments and many sections the app manages; Python can read TOML but not
# write it, so the editor's own tables are cut out and appended as text, and everything else stays byte for byte.
TOML_TABLE = f"[mcp_servers.{NAME}"


def _toml_without_entry(text: str) -> str:
    out, ours = [], False
    for line in text.splitlines(keepends=True):
        head = line.strip()
        if head.startswith("["):
            ours = head.startswith(TOML_TABLE + "]") or head.startswith(TOML_TABLE + ".")
        if not ours:
            out.append(line)
    return "".join(out).rstrip("\n") + "\n"


def _toml_entry() -> str:
    cmd, args, env = mcp_command()
    q = json.dumps  # a JSON string is a valid TOML basic string
    lines = [f"\n{TOML_TABLE}]", f"command = {q(cmd)}", f"args = [{', '.join(q(a) for a in args)}]",
             "startup_timeout_sec = 120", "tool_timeout_sec = 600"]  # an image can take minutes on a Mac
    if env:
        lines += [f"\n{TOML_TABLE}.env]"] + [f"{k} = {q(v)}" for k, v in env.items()]
    return "\n".join(lines) + "\n"


def _write_toml(app: ChatApp, connect: bool):
    _read(app)  # refuse to touch a file that isn't valid TOML
    text = app.config.read_text(encoding="utf-8") if app.config.exists() else ""
    text = _toml_without_entry(text) if text.strip() else ""
    if connect:
        text += _toml_entry()
    app.config.parent.mkdir(parents=True, exist_ok=True)
    backup = app.config.with_name(app.config.name + f".before-{NAME}")
    if app.config.exists() and not backup.exists():
        shutil.copy2(app.config, backup)
    tmp = app.config.with_name(app.config.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, app.config)


def _entry(app: ChatApp) -> dict:
    cmd, args, env = mcp_command()
    if app.key == "goose":
        return {"enabled": True, "type": "stdio", "name": NAME, "display_name": "Qwen Image Editor",
                "description": "Generate and edit images locally with Qwen-Image-2.1",
                "cmd": cmd, "args": args, "envs": env, "env_keys": [], "timeout": 600}
    return {"command": cmd, "args": args, **({"env": env} if env else {})}


def entry_of(app: ChatApp) -> dict | None:
    try:
        return (_read(app).get(_section(app)) or {}).get(NAME)
    except (OSError, ValueError):
        return None


def connect(key: str) -> ChatApp:
    app = get_app(key)
    if app.key == "chatgpt":
        _write_toml(app, True)
        return app
    data = _read(app)
    section = data.get(_section(app)) or {}
    section[NAME] = _entry(app)
    data[_section(app)] = section
    _write(app, data)
    return app


def _project(entry: dict) -> str | None:
    args = entry.get("args") or []
    return args[args.index("--project") + 1] if "--project" in args[:-1] else None


def disconnect(key: str, this_install_only: bool = False) -> ChatApp:
    """Remove the entry; with this_install_only (the uninstaller), only if it runs this install, not another copy."""
    app = get_app(key)
    data = _read(app)
    section = data.get(_section(app)) or {}
    if this_install_only and NAME in section and Path(_project(section[NAME]) or "").resolve() != ROOT:
        return app
    if NAME in section and app.key == "chatgpt":
        _write_toml(app, False)
    elif NAME in section:
        del section[NAME]
        _write(app, data)
    return app


def status() -> list[dict]:
    out = []
    for app in chat_apps():
        entry = entry_of(app)
        out.append({"key": app.key, "name": app.name, "installed": app.installed or entry is not None,
                    "connected": entry is not None, "restart": app.restart})
    return out


def repair():
    """Point existing entries at this install when the one they name is gone (the app was moved or reinstalled
    somewhere else). Entries that still work are left alone, and apps that aren't connected stay that way."""
    for app in chat_apps():
        entry = entry_of(app)
        if not entry:
            continue
        project = _project(entry)
        cmd = entry.get("cmd") or entry.get("command") or ""
        if project and Path(project).is_dir() and Path(cmd).exists():
            continue
        try:
            connect(app.key)
            print(f"Updated the {app.name} connection to this install.", flush=True)
        except Exception as exc:  # never stop the editor over this
            print(f"Couldn't update the {app.name} connection: {exc}", flush=True)
