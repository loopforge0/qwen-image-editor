"""MCP server: the editor inside chat apps (Claude Desktop, Goose, VS Code, ...).

    uv run qie mcp        (stdio; chat apps start it themselves, see the README)

A thin client of the editor (app/server.py on 127.0.0.1:7861), which it starts in the background when it isn't
running, so the model stays loaded in one place for the browser editor and every chat app, and results land in the
same history.

The model gets four tools: generate_image, edit_image, open_editor and list_images, and three for loop mode
(loop.py): start_loop, inspect_image, mark_image and review_image. In apps that support MCP Apps,
results open in the "studio" widget (studio.html): compare views, and drawing marks on images to edit them. The widget
works through tools only it can call (visibility "app"), and gets every image as data, not from localhost, which
several apps block.
"""

import asyncio
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

from mcp.server.apps import Apps
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, ImageContent, TextContent
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from qie_engine import config  # noqa: E402

sys.path.insert(0, str(HERE))
import loop as loops  # noqa: E402

URL = os.environ.get("QIE_URL", "http://127.0.0.1:7861")
STUDIO = "ui://qwen-image-editor/studio.html"
MAX_IMAGES = 10  # Qwen-Image-2.1 edits take up to about 10 input images
CHUNK = 100_000  # characters of base64 per widget call; Claude cuts tool results at ~150k characters

# The same colours and wording as the browser editor (app/static/app.js).
MARK_COLORS = {"red": "#ff3b47", "green": "#22d861", "blue": "#3b82f6", "magenta": "#f23ccd", "yellow": "#ffd028", "cyan": "#22d3ee"}
NO_MARKS = "Do not render the circles in the output."
RESULT_ID = re.compile(r"^\d{8}-\d{6}-\d+$")
INPUT_ID = re.compile(r"^[0-9a-f]{16}$")
LOG = config.home() / "mcp.log"


def log(line: str):
    """To mcp.log in the editor's data folder; stdout is the MCP connection and chat apps hide stderr."""
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] {line}\n")
    except OSError:
        pass


# ---------------------------------------------------------------- the editor

class EditorError(Exception):
    pass


def _request(path: str, data: bytes | None = None, ctype: str = "application/json", timeout: float = 10) -> bytes:
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": ctype} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error")
        except Exception:
            msg = None
        raise EditorError(msg or f"The editor answered HTTP {e.code} for {path}.") from None


def _json(path: str, obj: Any = None, timeout: float = 10) -> Any:
    return json.loads(_request(path, None if obj is None else json.dumps(obj).encode(), timeout=timeout))


def editor_status() -> dict | None:
    try:
        return _json("/api/status", timeout=2)
    except (OSError, EditorError, ValueError):
        return None


def start_editor():
    """Start the tray app in the background (no browser tab), detached so it outlives this process: closing the chat
    app shouldn't stop the editor in the middle of something."""
    scripts = Path(sys.executable).parent
    tray = scripts / ("qie-tray.exe" if sys.platform == "win32" else "qie-tray")
    cmd = [str(tray), "--background"] if tray.exists() else [sys.executable, "-m", "qie_engine.cli", "app", "--no-browser"]
    kw: dict[str, Any] = dict(cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        try:  # out of the chat app's job object, if it allows that, so the editor isn't closed along with the app
            subprocess.Popen(cmd, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw)
            return
        except OSError:
            pass
        subprocess.Popen(cmd, creationflags=flags, **kw)
    else:
        subprocess.Popen(cmd, start_new_session=True, **kw)


async def ensure_editor(ctx: Context | None) -> dict:
    """The editor's status once it can generate: started if needed, loaded, set up."""
    st = editor_status()
    if st is None:
        start_editor()
        for _ in range(120):
            await asyncio.sleep(0.5)
            if (st := editor_status()) is not None:
                break
        else:
            raise ToolError("Qwen Image Editor didn't start. Open it from the Start menu (or Applications on a Mac) and try again.")
    if st["state"] in ("needs_setup", "setting_up"):
        if st["state"] == "needs_setup":
            webbrowser.open(URL)
        raise ToolError(f"Qwen Image Editor isn't set up yet: finish the one-time setup in the browser ({URL}), then try again.")
    t0 = time.time()
    while st["state"] == "loading":
        if ctx:
            await ctx.report_progress(st.get("load_pct") or 0, 100, f"Loading the model: {st.get('load_stage') or ''}".rstrip(": "))
        if time.time() - t0 > 180:
            raise ToolError("The model is still loading. Try again in a minute.")
        await asyncio.sleep(1)
        st = editor_status() or st
    if st["state"] == "error":
        raise ToolError(f"Qwen Image Editor couldn't load the model: {st.get('error') or 'see the log (tray icon > Show log)'}.")
    return st


STAGE_SPAN = {"queued": (0, 2), "text": (2, 12), "refs": (12, 22), "steps": (22, 92), "decode": (92, 100)}


def progress_of(st: dict) -> tuple[float, str]:
    stage = st.get("stage") or "queued"
    lo, hi = STAGE_SPAN.get(stage, (0, 2))
    if stage == "steps" and st.get("steps"):
        return lo + (hi - lo) * st["step"] / st["steps"], f"Generating: step {st['step']} of {st['steps']}"
    return lo, {"queued": "Starting", "text": "Reading the prompt", "refs": "Reading the images", "decode": "Finishing"}.get(stage, "Working")


def _generate_blocking(body: dict) -> dict:
    """POST /api/generate, waiting while another generation (the browser editor, another chat) is running."""
    for _ in range(600):
        try:
            return _json("/api/generate", body, timeout=900)
        except EditorError as e:
            if str(e) not in ("Already generating.", "The model is still loading."):
                raise ToolError(str(e)) from None
        time.sleep(1)
    raise ToolError("The editor stayed busy for 10 minutes; try again later.")


async def run_generation(ctx: Context | None, body: dict) -> dict:
    await ensure_editor(ctx)
    job = asyncio.ensure_future(asyncio.to_thread(_generate_blocking, body))
    while not job.done():
        await asyncio.wait({job}, timeout=1)
        if ctx and not job.done() and (st := editor_status()) and st.get("busy"):
            pct, msg = progress_of(st)
            await ctx.report_progress(pct, 100, msg)
    return job.result()


# ---------------------------------------------------------------- images

def image_bytes(image_id: str) -> bytes:
    folder = "results" if RESULT_ID.match(image_id) else "inputs"
    try:
        return _request(f"/files/{folder}/{image_id}.png", timeout=30)
    except EditorError:
        raise ToolError(f"There's no image {image_id!r} in the editor's history.") from None


def upload(data: bytes) -> dict:
    """Into the editor's uploads: {id, width, height}."""
    try:
        return json.loads(_request("/api/upload", data, "application/octet-stream", timeout=60))
    except EditorError as e:
        raise ToolError(f"The editor couldn't read that image: {e}") from None


def resolve_image(ref: str) -> dict:
    """An image the model named: a result or upload id from the history, a file on this computer, or a data: URL.
    Returns the upload {id, width, height} the editor works from."""
    ref = ref.strip().strip('"')
    if RESULT_ID.match(ref):
        return upload(image_bytes(ref))
    if INPUT_ID.match(ref):
        with Image.open(io.BytesIO(image_bytes(ref))) as im:
            return {"id": ref, "width": im.width, "height": im.height}
    if ref.startswith("data:image/"):
        return upload(base64.b64decode(ref.split(",", 1)[1]))
    path = Path(os.path.expandvars(ref)).expanduser()
    if path.is_file():
        return upload(path.read_bytes())
    raise ToolError(f"Couldn't find the image {ref!r}. Use an id from list_images or a previous result, or the full path "
                    "of an image file on this computer.")


def transparent(im: Image.Image) -> bool:
    """Whether the image has see-through pixels (Qwen-Image-2.1 makes RGBA images when asked)."""
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    return im.mode in ("RGBA", "LA") and im.getchannel("A").getextrema()[0] < 250


def on_checkerboard(im: Image.Image) -> Image.Image:
    """Transparent pixels shown over a light checkerboard, as image editors do. Dropping the alpha instead shows the
    colour stored under them, which is arbitrary (often magenta) and reads as a real background."""
    im = im.convert("RGBA")
    cell = max(8, round(max(im.size) / 40))
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(bg)
    for y in range(0, im.size[1], cell):
        for x in range((y // cell) % 2 * cell, im.size[0], 2 * cell):
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(210, 210, 210, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def encode(data: bytes, max_px: int, quality: int = 88, keep_alpha: bool = False) -> tuple[str, int, int, str]:
    """PNG bytes as base64, at most max_px on the long side: JPEG (WebP would be smaller, but not every app shows it),
    or with keep_alpha a transparent image as PNG. Otherwise transparency goes over a checkerboard.
    Returns (data, width, height, mime)."""
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        alpha = transparent(im)
        im = im.convert("RGBA" if alpha else "RGB")
        if max_px and max(w, h) > max_px:
            im.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        if alpha and keep_alpha:
            im.save(buf, "PNG", optimize=True)
            mime = "image/png"
        else:
            (on_checkerboard(im) if alpha else im).save(buf, "JPEG", quality=quality, optimize=True)
            mime = "image/jpeg"
    return base64.b64encode(buf.getvalue()).decode(), w, h, mime


def stroke_width(w: int, h: int) -> int:
    return max(3, round(max(w, h) / 190))  # as in the browser editor: visible to the model at any size


def flatten(image_id: str, marks: list[dict]) -> str:
    """The image with its marks drawn into the pixels, the way the model sees them; returns the new upload's id."""
    with Image.open(io.BytesIO(image_bytes(image_id))) as im:
        im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
        draw = ImageDraw.Draw(im)
        sw = stroke_width(*im.size)
        for m in marks:
            pts = [tuple(p) for p in m.get("points") or []]
            if len(pts) < 2:
                continue
            if m.get("closed"):
                pts.append(pts[0])
            color = MARK_COLORS.get(m.get("color"), MARK_COLORS["red"])
            draw.line(pts, fill=color, width=sw, joint="curve")
            for x, y in (pts[0], pts[-1]):  # round ends
                draw.ellipse((x - sw / 2, y - sw / 2, x + sw / 2, y + sw / 2), fill=color)
        buf = io.BytesIO()
        im.save(buf, "PNG")
    return upload(buf.getvalue())["id"]


# ---------------------------------------------------------------- results

def clean_prompt(prompt: str) -> str:
    return prompt.replace(" " + NO_MARKS, "")


def summary(rec: dict) -> dict:
    """What the widget needs about a generation."""
    return {
        "id": rec["id"], "prompt": clean_prompt(rec.get("prompt", "")), "width": rec["width"], "height": rec["height"],
        "seed": rec.get("seed"), "total_s": rec.get("total_s"), "path": str(result_file(rec["id"]) or ""),
        "sources": [{"id": s["id"], "w": s.get("w"), "h": s.get("h"), "marks": s.get("marks") or []} for s in rec.get("sources") or []]
                   or [{"id": i} for i in rec.get("refs") or []],
    }


def result_file(rec_id: str) -> Path | None:
    path = config.home() / "history" / "results" / f"{rec_id}.png"
    return path if path.exists() else None


# When the user has the widget open as a panel (full screen or on the side), it follows every result itself; the
# widget each tool call adds to the chat then shows a small card instead of a second copy. The panel checks in here.
_panel_seen = 0.0


def panel_open() -> bool:
    return time.time() - _panel_seen < 6


def result_for_model(rec: dict, preview_px: int) -> CallToolResult:
    raw = image_bytes(rec["id"])
    data, _, _, _ = encode(raw, preview_px, quality=80)
    with Image.open(io.BytesIO(raw)) as im:
        see_through = transparent(im)
    path = result_file(rec["id"])
    n = len(rec.get("refs") or [])
    text = (f"{'Edited' if n else 'Generated'} image {rec['id']} ({rec['width']}×{rec['height']}, {rec.get('total_s', 0):.1f} s, "
            f"seed {rec.get('seed')}{f', from {n} image' + ('s' if n > 1 else '') if n else ''})."
            + (" It has a transparent background (the preview shows it as a grey and white checkerboard)."
               if see_through else "")
            + (f" Saved as {path}." if path else "")
            + f" To change it further, call edit_image with images [\"{rec['id']}\"].")
    return CallToolResult(
        content=[TextContent(type="text", text=text), ImageContent(type="image", data=data, mime_type="image/jpeg")],
        structured_content={"view": "result", "result": summary(rec), "panel": panel_open()},
    )


def size_args(width: int | None, height: int | None) -> dict:
    if (width is None) != (height is None):
        raise ToolError("Give both width and height, or neither.")
    if width is not None and not (256 <= width <= 2048 and 256 <= height <= 2048):
        raise ToolError("Width and height must be between 256 and 2048.")
    return {"width": width, "height": height, "size_mode": "custom" if width else "auto"}


# ---------------------------------------------------------------- the server

def studio_html() -> str:
    """studio.html with the MCP Apps helper library inlined: apps render widgets that load scripts from a CDN blank."""
    lib = (HERE / "vendor" / "ext-apps-2.0.3.js").read_text(encoding="utf-8")
    head, _, tail = lib.rpartition("export{")
    names = dict(reversed(p.split(" as ")) if " as " in p else (p, p) for p in tail.split("}")[0].split(","))
    # In its own scope: as a plain script its minified top-level names (S, $, ...) would become globals and clash
    # with the widget's own, which stops the widget's script from running at all.
    lib = "(()=>{" + head + "globalThis.ExtApps={" + ",".join(f"{k}:{v}" for k, v in names.items()) + "};})();"
    return (HERE / "studio.html").read_text(encoding="utf-8").replace("<!--EXT-APPS-->", f"<script>{lib}</script>")


INSTRUCTIONS_HEAD = """Qwen Image Editor generates and edits images with Qwen-Image-2.1, locally on the user's computer.
If this app has its own image generator too, use these tools whenever the user mentions Qwen, the image editor,
local or private generation, or images made or marked up here.

- generate_image makes a new image from a description. edit_image changes existing images (up to 10).
- The widget shows results, lets the user pick earlier ones from its history, and lets them mark up images with coloured circles and lines ("Edit this", or
  images they drop into open_editor). They then say in the chat what to change. What they marked is kept as a draft:
  when they refer to marks, circles, "this", "here" or "the one I picked", or an image they just added, call view_draft
  to see it, then edit_image with that draft. (edit_image also stops once to tell you if the user has marks you
  haven't seen, or has picked a different image from the one you named.)
- mark_image lets you circle things yourself (up to 3, one colour each) and edit the marked copy, naming each
  circle by colour: the way to point at one specific thing among several.
- Images the user attaches to the chat can't be passed to these tools. Use a result id, an image path on this
  computer, or call open_editor so the user can drop the image in and mark what to change.
- Each image takes about 10-20 s on an NVIDIA GPU and 1.5-3 minutes on a Mac.
- Loop mode: when the user wants you to keep going until the image is right ("loop", "iterate", "until it's good",
  "check it and fix it"), or gives a goal with things that can be checked (exact text, counts, positions), call
  start_loop first. It sets up a checklist and a review after every image.

"""

PROMPT_RULES = """Writing prompts: you write them from the user's words, the way Qwen-Image-2.1 was trained (these follow Qwen's own
prompt rewriter).
- New images: one detailed English paragraph describing the finished picture as someone looking at it would: first
  the medium, style, subject and setting; then what is where in the frame, with counts, colours and positions; then
  the lighting; then the overall composition and mood. A few sentences for a simple request, up to about 400 words
  for a rich scene. Text that should appear goes in double quotes, with its size, weight and colour. No instructions
  or quality words ("masterpiece", "4K").
- Edits: lead with the operation ("Replace...", "Remove...", "Make...") and make the change strong and unmistakable.
  Name the user's marks by colour and shape ("the vase in the red circle", "along the blue line"). Cover the rest
  with one blanket clause, "keep everything else unchanged", without describing it: describing what should stay
  makes it drift. Faces, products' designs and the image's medium (photo, illustration, ...) stay unless the user
  asks. Add nothing the user didn't ask for.
- Several images: refer to them as <image1>, <image2>, ... in the order given, and say each one's role ("Put the
  person from <image2> on the sofa in <image1>"). To keep someone's likeness, point at their image rather than
  describing their face. With one image, just say "the image".
- Text in an edited image: quote the exact words, in the language and lettering the image already uses unless asked.
- Size (new images): pick the shape from what the picture is, and always pass width and height. Landscape scenes
  (places, rooms, groups of people side by side, wide views, anything framed horizontally): 16:9, 1344x768.
  Portraits and tall subjects (one person standing, a phone wallpaper, a poster, a tall building): 9:16, 768x1344.
  Square only when it suits the subject (an icon, a sticker, a single object, a profile picture) or is asked for.
  Other shapes when the user names one or the use calls for it: 4:3 1152x864, 3:4 864x1152, 3:2 1248x832,
  2:3 832x1248, 21:9 1536x640. Stay near 1 megapixel. Edits keep the source image's shape unless asked to change it.
- Transparent background: begin with "This is an RGBA image with transparency." and end with "The image has alpha
  channel and the background is transparent." word for word."""
INSTRUCTIONS = INSTRUCTIONS_HEAD + PROMPT_RULES

DRAFTS = config.home() / "drafts.json"


def _load_drafts() -> list[dict]:
    try:
        return json.loads(DRAFTS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def prepare(images: list[dict]) -> tuple[list[dict], list[str]]:
    """Widget images ([{id, w, h, marks}]) as the editor needs them: results become uploads ("Edit this"), and images
    with marks get a copy with the marks drawn in. Returns (sources, refs): the clean images and what the model sees."""
    sources = []
    for i in images:
        img = {"id": i["id"], "w": i.get("w"), "h": i.get("h"), "marks": i.get("marks") or []}
        if RESULT_ID.match(img["id"]):
            up = resolve_image(img["id"])
            img.update(id=up["id"], w=up["width"], h=up["height"])
        sources.append(img)
    return sources, [flatten(s["id"], s["marks"]) if s["marks"] else s["id"] for s in sources]


DRAFT_FRESH_S = 2 * 3600  # how long the user's latest markup counts as what they're talking about


def _save_drafts(drafts: list[dict]):
    DRAFTS.parent.mkdir(parents=True, exist_ok=True)
    DRAFTS.write_text(json.dumps(drafts[:20]), encoding="utf-8")  # the last 20


def save_draft(images: list[dict], draft_id: str = "") -> dict:
    """The widget saves the user's markup as they draw; the same draft id is updated until it's used for an edit."""
    sources, refs = prepare(images)
    drafts = [d for d in _load_drafts() if d["id"] != draft_id]
    draft = {"id": draft_id or os.urandom(3).hex(), "created": time.time(), "images": sources, "refs": refs,
             "origs": [i["id"] for i in images], "used": False, "seen": False}
    _save_drafts([draft] + drafts)
    return draft


def drop_draft(draft_id: str):
    _save_drafts([d for d in _load_drafts() if d["id"] != draft_id])


def mark_draft(draft_id: str, **flags):
    drafts = _load_drafts()
    for d in drafts:
        if d["id"] == draft_id:
            d.update(flags)
    _save_drafts(drafts)


def pending_draft(images: list[str] | None) -> tuple[dict | None, bool]:
    """What the user last did in the editor (marked up images, or picked one from the history), if it's recent and
    not used for an edit yet; and whether it's about the images the model named (or the model named none)."""
    drafts = _load_drafts()
    d = drafts[0] if drafts else None
    if not d or d.get("used", True) or time.time() - d["created"] > DRAFT_FRESH_S:
        return None, False
    if images:
        names = {i.strip().strip('"') for i in images}
        return d, bool(names & (set(d.get("origs", [])) | {s["id"] for s in d["images"]}))
    return d, True


def picked(d: dict) -> str:
    """The images in a draft, as the model knows them: result ids with their prompts."""
    recs = {r["id"]: r for r in _json("/api/history")}
    return ", ".join(f"{o} ({clean_prompt(recs[o].get('prompt', ''))[:80]!r})" if o in recs else o for o in d.get("origs", []))


def get_draft(draft_id: str | None) -> dict:
    drafts = _load_drafts()
    if not drafts:
        raise ToolError("There are no drafts: the user hasn't marked up any images in the editor yet (open_editor).")
    if not draft_id:
        return drafts[0]  # the latest
    for d in drafts:
        if d["id"] == draft_id.strip().removeprefix("draft "):
            return d
    raise ToolError(f"There's no draft {draft_id!r}. The latest is {drafts[0]['id']}.")


def where(points: list, w: int, h: int) -> str:
    """Where a mark is, in words and percentages of the image."""
    xs, ys = [p[0] / w for p in points], [p[1] / h for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    col = "left" if cx < 0.36 else "right" if cx > 0.64 else "centre"
    row = "top" if cy < 0.36 else "bottom" if cy > 0.64 else "middle"
    place = "the centre" if (row, col) == ("middle", "centre") else f"the {row} {col}" if row != "middle" else f"the middle {col}"
    size = (x1 - x0) * (y1 - y0)
    return f"{place} (x {x0:.0%}-{x1:.0%}, y {y0:.0%}-{y1:.0%}; {'small' if size < 0.04 else 'large' if size > 0.25 else 'medium'})"


def describe(draft: dict) -> str:
    lines = []
    for n, s in enumerate(draft["images"], 1):
        marks = "; ".join(f"{m['color']} {'circle around' if m.get('closed') else 'line across'} {where(m['points'], s['w'], s['h'])}"
                          for m in s["marks"] if m.get("points")) or "no marks"
        label = f"<image{n}>" if len(draft["images"]) > 1 else "The image"  # how the prompt refers to it
        lines.append(f"{label} ({s['w']}×{s['h']}): {marks}")
    return "\n".join(lines)


apps = Apps()  # the tools bound to the widget; registered before the server is made, which collects them
_cache: dict[tuple[str, int], tuple[str, int, int, str]] = {}
_uploads: dict[str, list[str]] = {}


def preview_px(ctx: Context) -> int:
    # With the widget, the model only needs enough to judge the result; without it the user sees this preview.
    from mcp.server.apps import client_supports_apps

    try:
        apps_ok = client_supports_apps(ctx)
    except Exception:
        apps_ok = None
    log(f"client supports MCP Apps: {apps_ok}")
    return {True: 512, False: 768}.get(apps_ok, 640)


@apps.tool(resource_uri=STUDIO, title="Generate image")
async def generate_image(prompt: str, ctx: Context, width: int | None = None, height: int | None = None,
                         seed: int | None = None, loop: str | None = None) -> CallToolResult:
    """Generate a new image from a text description with Qwen-Image-2.1, on the user's own computer (private, no
    usage limits; ~10-20 s on NVIDIA, 1.5-3 minutes on a Mac).

    prompt: one detailed English paragraph describing the finished picture: medium, style, subject and setting first,
      then what is where (counts, colours, positions), the lighting, and the overall composition and mood. Text to
      show goes in double quotes, with its size, weight and colour. No quality words ("masterpiece", "4K").

    width/height: 256-2048, near 1 megapixel. Pass them every time, from the picture's shape: 1344x768 (16:9) for
      landscape scenes, 768x1344 (9:16) for portraits and tall subjects, 1024x1024 only for square subjects (icons,
      single objects) or when asked; other ratios when the user names one. Without them it's 1024x1024.
    loop: the loop id from start_loop, when working in loop mode.
    Returns the image and its id, which edit_image accepts."""
    lp = loop_of(loop)
    log(f"generate_image {width}x{height}{f' loop {loop}' if loop else ''}: {prompt[:80]!r}")
    rec = await run_generation(ctx, {"prompt": prompt, "refs": [], "seed": seed, **size_args(width, height)})
    log(f"generate_image -> {rec['id']} in {rec.get('total_s')} s")
    return in_loop(lp, rec, "generate", [], result_for_model(rec, preview_px(ctx)))


@apps.tool(resource_uri=STUDIO, title="Edit image")
async def edit_image(instruction: str, ctx: Context, images: list[str] | None = None, draft: str | None = None,
                     width: int | None = None, height: int | None = None, seed: int | None = None,
                     loop: str | None = None) -> CallToolResult:
    """Edit images with Qwen-Image-2.1: change, add or remove things, restyle, or combine several images (up to 10).

    Give either images or draft:
    images: each one is an id from a previous result or list_images, or the full path of an image file on this
      computer. With several, refer to them in the instruction as <image1>, <image2>, ... in the order given and
      say each one's role, e.g. "Put the person from <image2> on the sofa in <image1>". With one, say "the image".
    draft: the id of images the user marked up in the editor (see view_draft); the model sees the marks, so refer to
      them by colour: "replace the lamp in the red circle with a potted plant". With neither, edits what the
      user last put in the editor.
    instruction: lead with the change and make it strong, then one blanket clause for the rest, e.g. "Make it night
      with warm light in the windows; keep everything else unchanged." Don't describe what should stay (it drifts).
    width/height: optional output size (default: the shape of the last image, about 1 megapixel).
    loop: the loop id from start_loop, when working in loop mode."""
    lp = loop_of(loop)
    if not draft:
        d, matches = await asyncio.to_thread(pending_draft, images)
        if d and not matches and not d.get("seen") and lp is None:
            # The user picked or marked up other images in the editor than the ones the model named: ask once.
            await asyncio.to_thread(mark_draft, d["id"], seen=True)
            log(f"edit_image: stopped, the user has other images open (draft {d['id']})")
            what = "marked up" if any(s["marks"] for s in d["images"]) else "selected"
            raise ToolError(f"Check first: in the editor the user has {what} {await asyncio.to_thread(picked, d)}, not the "
                            f"image you named (draft {d['id']}):\n{describe(d)}\nIf they mean that, call view_draft and then "
                            f"edit_image with draft \"{d['id']}\". If not, call edit_image again as before.")
        if d and not matches:
            d = None
        if d and any(s["marks"] for s in d["images"]) and not d.get("seen"):
            await asyncio.to_thread(mark_draft, d["id"], seen=True)
            log(f"edit_image: stopped for unseen marks in draft {d['id']}")
            raise ToolError(f"The user has marked up {'this image' if len(d['images']) == 1 else 'these images'} in the editor "
                            f"(draft {d['id']}):\n{describe(d)}\nCall view_draft to see the marks, then edit_image with "
                            f"draft \"{d['id']}\" and an instruction that names the marks by colour.")
        if d:
            draft = d["id"]  # what the user put in the editor, marks or not
    if draft:
        d = await asyncio.to_thread(get_draft, draft)
        sources, refs = d["images"], d["refs"]
        draft = d["id"]
    elif images:
        if len(images) > MAX_IMAGES:
            raise ToolError(f"At most {MAX_IMAGES} images per edit.")
        ups = [await asyncio.to_thread(resolve_image, ref) for ref in images]
        sources = [{"id": u["id"], "w": u["width"], "h": u["height"], "marks": []} for u in ups]
        refs = [u["id"] for u in ups]
    else:
        raise ToolError("Give the images to edit, or a draft (or use generate_image for a new image).")
    prompt = instruction.strip()
    if (any(s["marks"] for s in sources) or any(r in _marked for r in refs)) and NO_MARKS not in prompt:
        prompt = prompt.rstrip(". \n") + ". " + NO_MARKS
    body = {"prompt": prompt, "refs": refs, "seed": seed, "sources": sources, **size_args(width, height)}
    log(f"edit_image {len(refs)} images{f' (draft {draft})' if draft else ''}: {instruction[:80]!r}")
    rec = await run_generation(ctx, body)
    if draft:
        await asyncio.to_thread(mark_draft, draft, used=True)
    log(f"edit_image -> {rec['id']} in {rec.get('total_s')} s")
    return in_loop(lp, rec, "edit", list(images or []), result_for_model(rec, preview_px(ctx)))


@apps.tool(resource_uri=STUDIO, title="Open the image editor")
async def open_editor(ctx: Context, images: list[str] | None = None) -> CallToolResult:
    """Open the editor widget in the chat, where the user drops or pastes images and draws marks around what should
    change; then they tell you in the chat what to do, and you call view_draft and edit_image. Use it when the user
    wants to edit an image you can't pass to edit_image (for example one attached to the chat), or to point at areas.

    images: optional, up to 10 to start with: result ids or full paths of image files on this computer."""
    log(f"open_editor {images}")
    await ensure_editor(ctx)
    ups = [await asyncio.to_thread(resolve_image, ref) for ref in (images or [])[:MAX_IMAGES]]
    text = ("Opened the image editor. The user adds images and marks areas there, then tells you what to change; "
            "use view_draft to see what they put in, then edit_image.")
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content={"view": "canvas", "images": [{"id": u["id"], "w": u["width"], "h": u["height"]} for u in ups],
                            "panel": panel_open()},
    )


async def list_images(limit: int = 10) -> str:
    """The most recent generations and edits (newest first): id, size, prompt. Ids work with edit_image."""
    await ensure_editor(None)
    recs = await asyncio.to_thread(_json, "/api/history")
    if not recs:
        return "No images yet."
    lines = [f"{r['id']}  {r['width']}×{r['height']}  {clean_prompt(r.get('prompt', ''))[:120]}" for r in recs[:max(1, min(limit, 50))]]
    d, _ = await asyncio.to_thread(pending_draft, None)
    if d:
        lines.append(f"\nOpen in the editor now (draft {d['id']}): {', '.join(d.get('origs', []))}\n{describe(d)}")
    return "\n".join(lines)


# ---------------------------------------------------------------- loop mode (loop.py)

def loop_of(loop_id: str | None) -> dict | None:
    if not loop_id:
        return None
    lp = loops.get(loop_id)
    if lp is None:
        raise ToolError(f"There's no loop {loop_id!r} (loops last as long as the chat app runs). Call start_loop again.")
    return lp


def in_loop(lp: dict | None, rec: dict, kind: str, sources: list[str], result: CallToolResult) -> CallToolResult:
    """A result made in a loop: recorded, and its text says what comes next."""
    if lp is None:
        return result
    loops.record_made(lp, rec["id"], kind, sources)
    result.content[0].text += loops.made_note(lp, rec["id"])
    return result


async def start_loop(goal: str, checks: list[str], max_rounds: int = 5, start_images: list[str] | None = None) -> str:
    """Start loop mode: make an image, review it strictly against a checklist, fix it, and repeat until every check
    passes or the rounds run out. Use it when the user wants you to iterate until the image is right.

    goal: what the user wants, in their words.
    checks: 3-8 yes/no checks you write from the goal, each about one thing you can see in the picture, specific
      enough that two people would agree: 'The sign reads exactly "OPEN LATE" in red neon', 'Exactly three cats, all
      on the sofa', 'It is raining, with wet reflections on the street'. Cover every concrete part of the goal; add
      one for overall quality if the goal implies it ('Faces and hands look natural'). When the goal adds people or
      objects to a scene, add one for blending: 'Everything added looks photographed there: same light, a shadow
      where it touches the ground, same sharpness, right scale'.
    max_rounds: how many images to make at most (default 5).
    start_images: optional images to start from (a sketch, a photo to change, references): result ids or file paths.
    Returns the loop id and the steps to follow."""
    checks = [c.strip() for c in checks if c and c.strip()]
    if not 1 <= len(checks) <= 12:
        raise ToolError("Give 3-8 checks, each one yes/no thing you can see in the picture.")
    st = await ensure_editor(None)
    candidates = 1  # one image, reviewed, then decide: a second only when the review asks for it
    start = [str(i).strip() for i in (start_images or []) if str(i).strip()][:MAX_IMAGES]
    lp = loops.new_loop(goal.strip(), checks, max(1, min(int(max_rounds), 10)), candidates, start)
    log(f"start_loop {lp['id']}: {len(checks)} checks, {lp['max_rounds']} rounds, {candidates} candidates: {goal[:80]!r}")
    return f"Loop {lp['id']} started.\n\n" + loops.guide(lp, PROMPT_RULES)


GRID = {"top-left": (0, 0), "top": (1, 0), "top-right": (2, 0), "left": (0, 1), "center": (1, 1), "right": (2, 1),
        "bottom-left": (0, 2), "bottom": (1, 2), "bottom-right": (2, 2)}


def crop_box(area: str, w: int, h: int) -> tuple[int, int, int, int] | None:
    """None for the whole image; a 3x3 grid cell (with a margin) by name, or "x,y,w,h" as fractions of the image."""
    area = (area or "full").strip().lower()
    if area in ("", "full", "whole", "all"):
        return None
    if area in GRID:
        cx, cy = GRID[area]
        x0, y0, x1, y1 = (cx - 0.15) / 3, (cy - 0.15) / 3, (cx + 1.15) / 3, (cy + 1.15) / 3
    else:
        try:
            fx, fy, fw, fh = (float(v) for v in area.split(","))
        except ValueError:
            raise ToolError(f"area: 'full', one of {', '.join(GRID)}, or 'x,y,w,h' as fractions (e.g. '0.3,0.1,0.4,0.3').") from None
        x0, y0, x1, y1 = fx, fy, fx + fw, fy + fh
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
    if x1 - x0 < 0.03 or y1 - y0 < 0.03:
        raise ToolError("That area is too small to look at.")
    return round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)


def inspect_bytes(image_id: str, area: str) -> tuple[str, str]:
    raw = image_bytes(image_id)
    with Image.open(io.BytesIO(raw)) as im:
        w, h = im.size
        box = crop_box(area, w, h)
        if box:
            im = im.crop(box)
        buf = io.BytesIO()
        im.save(buf, "PNG")
    data, cw, ch, _ = encode(buf.getvalue(), 896, quality=82)
    where = "the whole image" if not box else f"pixels {box[0]}-{box[2]} across, {box[1]}-{box[3]} down"
    return data, f"{image_id}, {where} of {w}x{h}."


async def inspect_image(image: str, area: str = "full", loop: str = "") -> CallToolResult:
    """Look at an image closely: the whole image larger than a result's preview, or zoomed into one area. Use it to
    check details before judging: text, hands, faces, small or counted objects, edges.

    image: a result id (or an id from list_images).
    area: "full" (default); a ninth of the image with some margin: top-left, top, top-right, left, center, right,
      bottom-left, bottom, bottom-right; or "x,y,w,h" as fractions of the width and height ("0.25,0.6,0.5,0.4").
    loop: the loop id, in loop mode."""
    image = image.strip()
    lp = loops.get(loop) if loop else None
    data, text = await asyncio.to_thread(inspect_bytes, image, area)
    if lp is not None:
        lp["inspected"].add(image)
    log(f"inspect_image {image} {area}")
    return CallToolResult(content=[TextContent(type="text", text=text + " Judge what is there, not what was asked for."),
                                   ImageContent(type="image", data=data, mime_type="image/jpeg")])


_marked: set[str] = set()  # uploads with circles mark_image drew: edits of them ask for the circles to go
MARK_ORDER = ["red", "green", "blue"]


def draw_marks(image_id: str, marks: list[dict]) -> tuple[str, str, list[str]]:
    """Circles around areas of an image, as the user's marks look; returns (upload id, preview, descriptions)."""
    with Image.open(io.BytesIO(image_bytes(image_id))) as im:
        im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)
        sw = stroke_width(w, h)
        said, used = [], set()
        for k, m in enumerate(marks):
            color = (m.get("color") or "").lower()
            if color not in MARK_COLORS or color in used:
                color = next(c for c in MARK_ORDER + list(MARK_COLORS) if c not in used)
            used.add(color)
            box = crop_box(m.get("area") or "", w, h) or (0, 0, w, h)
            pad = sw * 2  # around the thing, not on it
            x0, y0, x1, y1 = max(box[0] - pad, sw), max(box[1] - pad, sw), min(box[2] + pad, w - sw), min(box[3] + pad, h - sw)
            draw.ellipse((x0, y0, x1, y1), outline=MARK_COLORS[color], width=sw)
            said.append(f"{color} circle: {(m.get('what') or 'area').strip()} ({where([[x0, y0], [x1, y1]], w, h)})")
        buf = io.BytesIO()
        im.save(buf, "PNG")
    up = upload(buf.getvalue())
    _marked.add(up["id"])
    data, _, _, _ = encode(buf.getvalue(), 768, quality=80)
    return up["id"], data, said


async def mark_image(image: str, marks: list[dict], loop: str = "") -> CallToolResult:
    """Circle specific things on an image, so an edit can name them by colour: the way to change one thing among
    several (one person in a group, one word on a sign, a small object) without the edit touching the wrong one.
    Returns a marked copy: edit that copy with edit_image and name each circle by its colour ("Replace the lamp in
    the red circle with a potted plant; make the mug in the green circle blue"). The circles don't stay in the result.

    image: a result id (or a file path).
    marks: 1-3, each {"area": where it is, "what": what it is, "color": optional red/green/blue/magenta/yellow/cyan}.
      area: a ninth of the image (top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right)
      or "x,y,w,h" as fractions of the width and height, fitted tightly around the thing ("0.62,0.18,0.15,0.2").
      Look at the image (inspect_image) first so the circles land on the right things; check the returned copy.
    Keep it to 3 marks and one change per mark: more marked changes in one edit get mixed up."""
    if not marks:
        raise ToolError("Give 1-3 marks.")
    if len(marks) > 3:
        raise ToolError("At most 3 marks per edit: more marked changes in one request get mixed up. Mark the 3 most "
                        "important now and the rest in the next round.")
    image = image.strip()
    src = image if RESULT_ID.match(image) or INPUT_ID.match(image) else (await asyncio.to_thread(resolve_image, image))["id"]
    mid, data, said = await asyncio.to_thread(draw_marks, src, marks)
    log(f"mark_image {image}: {len(marks)} marks -> {mid}")
    text = (f"Marked copy {mid} of {image}:\n" + "\n".join(f"- {d}" for d in said)
            + f"\nCheck the circles sit around the right things (redo with adjusted areas if not), then edit_image with "
              f"images [\"{mid}\"]" + (f" and loop=\"{loop}\"" if loop else "") + ", naming each circle by colour, one "
              "change per circle, and nothing else changed.")
    return CallToolResult(content=[TextContent(type="text", text=text), ImageContent(type="image", data=data, mime_type="image/jpeg")])


async def review_image(loop: str, image: str, checks: list[dict], worst: str = "", major: bool = False) -> str:
    """Record your review of an image in loop mode and get the next step.

    loop: the loop id. image: the result id you reviewed.
    checks: one entry per check, in the loop's order: {"passed": true/false, "seen": "what is actually in the image
      for this check", "fix": "local" or "global"}. Say what you saw, not what was asked: the words as written, each
      counted thing and where. fix, for a failing check: "local" if fixing it changes one thing or area (an edit,
      marked if needed), "global" if it changes the whole picture (layout, count, pose, camera, style, lighting).
    worst: the single most visible defect in the image, on the checklist or not ("the second cat has five legs").
    major: true when a viewer would notice that defect at a glance."""
    lp = loop_of(loop)
    out = loops.review(lp, image.strip(), checks or [], worst or "", bool(major))
    log(f"review_image {loop} {image}: {out.splitlines()[0]}")
    return out

# Tools for the widget only. Each answers with its data twice: as structured content, and as JSON text for apps
# that pass only the text on to the widget.

def app_result(payload: dict) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))], structured_content=payload)


@apps.tool(resource_uri=STUDIO, visibility=["app"])
async def studio_image(id: str, max_px: int = 1600, chunk: int = 0) -> CallToolResult:
    """An image as base64 JPEG, in chunks (hosts cut long tool results)."""
    key = (id, max_px)
    if key not in _cache:
        if len(_cache) > 24:
            _cache.pop(next(iter(_cache)))
        _cache[key] = await asyncio.to_thread(lambda: encode(image_bytes(id), max_px, keep_alpha=True))
    data, w, h, mime = _cache[key]
    chunks = max(1, -(-len(data) // CHUNK))
    part = data[chunk * CHUNK:(chunk + 1) * CHUNK]
    log(f"studio_image {id} max_px={max_px} part {chunk + 1}/{chunks}")
    return app_result({"mime": mime, "data": part, "chunk": chunk, "chunks": chunks, "width": w, "height": h})


@apps.tool(resource_uri=STUDIO, visibility=["app"])
async def studio_upload(data: str, token: str = "", last: bool = True) -> CallToolResult:
    """An image from the widget, base64, in chunks: pass the returned token with the next part."""
    token = token or os.urandom(6).hex()
    _uploads.setdefault(token, []).append(data)
    if not last:
        return app_result({"token": token})
    raw = base64.b64decode("".join(_uploads.pop(token)))
    up = await asyncio.to_thread(upload, raw)
    log(f"studio_upload {len(raw)} bytes -> {up['id']} {up['width']}x{up['height']}")
    return app_result({"id": up["id"], "w": up["width"], "h": up["height"]})


@apps.tool(resource_uri=STUDIO, visibility=["app"])
async def studio_save_draft(images: list[dict], draft: str = "") -> CallToolResult:
    """Keep the user's marked-up images as a draft for the model to edit; called as they draw. Pass the returned id
    back to update the same draft; no images drops it."""
    if not images:
        if draft:
            await asyncio.to_thread(drop_draft, draft)
        return app_result({"draft": ""})
    if len(images) > MAX_IMAGES:
        raise ToolError(f"At most {MAX_IMAGES} images per edit.")
    d = await asyncio.to_thread(save_draft, images, draft)
    log(f"studio_save_draft {d['id']}: {len(images)} images, {sum(len(s['marks']) for s in d['images'])} marks")
    return app_result({"draft": d["id"], "description": describe(d)})


async def view_draft(draft: str = "") -> CallToolResult:
    """See the images the user marked up in the editor widget: each image with the user's coloured marks drawn on
    it, and where each mark is. Call this before edit_image(draft=...) so the prompt names the right things.
    draft: the id from the user's message; empty means the latest draft."""
    d = await asyncio.to_thread(get_draft, draft or None)
    await asyncio.to_thread(mark_draft, d["id"], seen=True)
    n = len(d["refs"])
    px = 640 if n == 1 else 448 if n <= 3 else 320  # keeps the result well under what chat apps accept
    content: list = [TextContent(type="text", text=f"Draft {d['id']}"
                                 + f"\n{describe(d)}\nThe marks are drawn on the images below, in order.")]
    for ref in d["refs"]:
        data, _, _, _ = await asyncio.to_thread(encode, image_bytes(ref), px, 75)
        content.append(ImageContent(type="image", data=data, mime_type="image/jpeg"))
    log(f"view_draft {d['id']}")
    return CallToolResult(content=content)


def save_to_downloads(rec_id: str) -> Path:
    src = result_file(rec_id)
    if not src:
        raise ToolError("That image isn't in the editor's history any more.")
    downloads = Path.home() / "Downloads"
    downloads.mkdir(exist_ok=True)
    dest = downloads / f"qwen-{rec_id}.png"
    shutil.copyfile(src, dest)
    try:  # show it, selected
        if sys.platform == "win32":
            subprocess.Popen(["explorer", f"/select,{dest}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(dest)])
    except OSError:
        pass
    return dest


@apps.tool(resource_uri=STUDIO, visibility=["app"])
async def studio_download(id: str) -> CallToolResult:
    """Save a result, full size with any transparency, to the Downloads folder and show it there."""
    if not RESULT_ID.match(id):
        raise ToolError("Only results can be downloaded.")
    dest = await asyncio.to_thread(save_to_downloads, id)
    log(f"studio_download {id} -> {dest}")
    return app_result({"path": str(dest)})


# Each widget registers when it opens. Only the newest one shows the progress of a generation it didn't start (in
# apps that add a tool call's widget only when the result is in, nothing else would show it); older ones go quiet.
_widget_seq = 0


@apps.tool(resource_uri=STUDIO, visibility=["app"])
async def studio_hello() -> CallToolResult:
    """A widget opened: its number. The highest is the newest."""
    global _widget_seq
    _widget_seq += 1
    return app_result({"seq": _widget_seq})


@apps.tool(resource_uri=STUDIO, visibility=["app"])
async def studio_status(panel: bool = False, widget: int = 0) -> CallToolResult:
    """The editor's state and the progress of the running generation. panel: asked by a widget open as a panel.
    widget: the asking widget's number (studio_hello); the answer says whether it's the newest."""
    global _panel_seen
    if panel:
        _panel_seen = time.time()
    st = editor_status() or {"state": "offline"}
    pct, msg = progress_of(st) if st.get("busy") else (0, "")
    return app_result({"state": st["state"], "busy": bool(st.get("busy")), "pct": pct, "message": msg, "gpu": st.get("gpu"),
                       "latest": widget >= _widget_seq, "panel": panel_open()})


@apps.tool(resource_uri=STUDIO, visibility=["app"])
async def studio_history(limit: int = 12) -> CallToolResult:
    """Recent generations for the widget's history strip."""
    recs = await asyncio.to_thread(_json, "/api/history")
    return app_result({"history": [summary(r) for r in recs[:max(1, min(limit, 50))]]})


@apps.tool(resource_uri=STUDIO, visibility=["app"])
async def studio_log(message: str) -> CallToolResult:
    """What the widget saw (errors, the app it runs in), for the log: chat apps don't show a widget's console."""
    log("widget: " + message[:2000])
    return app_result({"ok": True})


apps.add_html_resource(STUDIO, studio_html(), title="Qwen Image Editor", prefers_border=False)
mcp = MCPServer("qwen-image-editor", title="Qwen Image Editor", instructions=INSTRUCTIONS, extensions=[apps])
mcp.tool(title="List recent images")(list_images)
mcp.tool(title="View the user's marked-up images")(view_draft)
mcp.tool(title="Start loop mode")(start_loop)
mcp.tool(title="Inspect an image")(inspect_image)
mcp.tool(title="Mark things on an image")(mark_image)
mcp.tool(title="Review an image")(review_image)


def main():
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
