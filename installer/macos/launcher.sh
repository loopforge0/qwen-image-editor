#!/bin/bash
# Runs when "Qwen Image Editor" is opened: starts the menu-bar app, which runs the editor and opens it in the
# browser. The menu-bar icon has Open, Show log and Quit. Log: see $LOG below.

RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
DATA="$HOME/Library/Application Support/qwen-image-editor"
LOG="$DATA/editor.log"
URL="http://127.0.0.1:7861"
TITLE="Qwen Image Editor"
mkdir -p "$DATA"

say() {  # a dialog, brought to the front (the app has no Dock icon, so it would otherwise open behind windows)
  osascript -e 'activate' -e "display dialog \"$1\" buttons {\"OK\"} default button 1 ${2:-} with title \"$TITLE\"" >/dev/null 2>&1
}

# Opened from the disk image: the packages would point into it and break once it's ejected.
case "$RES" in
  /Volumes/*)
    say "Drag Qwen Image Editor to the Applications folder first, then open it from there." "with icon caution"
    exit 0 ;;
esac

# Already running? Then just open it.
if curl -sf --max-time 1 "$URL/api/status" >/dev/null 2>&1; then
  open "$URL"
  exit 0
fi

# One launcher at a time: a second double-click while the first is still installing or starting just says so.
LOCK="$DATA/launcher.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null; then
    say "Qwen Image Editor is still starting. It opens in your browser when it is ready."
    exit 0
  fi
  rm -rf "$LOCK" && mkdir "$LOCK" || exit 1  # left over from a launcher that was killed
fi
echo $$ >"$LOCK/pid"
trap 'rm -rf "$LOCK"; [ -n "${DIALOG:-}" ] && kill "$DIALOG" 2>/dev/null' EXIT

# The app bundle stays untouched: Python and packages go in the data folder.
export UV_PROJECT_ENVIRONMENT="$DATA/venv"
export PYTHONDONTWRITEBYTECODE=1
cd "$RES/app" || exit 1
echo "=== $(date) starting" >>"$LOG"

# First start, or a new version with different packages: install them with a progress window up (progress.js,
# which follows uv's output), so it doesn't look like nothing happened. It closes when the install is done.
STAMP="$UV_PROJECT_ENVIRONMENT/.installed"
WANT="$(shasum uv.lock | cut -d' ' -f1)"
if [ "$(cat "$STAMP" 2>/dev/null)" != "$WANT" ]; then
  INSTALL_LOG="$DATA/install.log"
  : >"$INSTALL_LOG"
  osascript -l JavaScript "$RES/progress.js" "$INSTALL_LOG" >/dev/null 2>&1 &
  DIALOG=$!  # osascript itself, so killing it closes the window
  "$RES/bin/uv" sync --frozen >"$INSTALL_LOG" 2>&1
  synced=$?
  cat "$INSTALL_LOG" >>"$LOG"
  if [ $synced -ne 0 ]; then
    kill "$DIALOG" 2>/dev/null
    say "Installing components failed. Check the internet connection and open Qwen Image Editor again.\n\nDetails are in:\n$INSTALL_LOG" "with icon caution"
    exit 1
  fi
  echo "$WANT" >"$STAMP"
  kill "$DIALOG" 2>/dev/null
  DIALOG=
fi

"$RES/bin/uv" run --frozen qie-tray >>"$LOG" 2>&1
code=$?
# 0: quit from the editor; 130/143: stopped by a signal (logout, Activity Monitor). Anything else is an error.
if [ $code -ne 0 ] && [ $code -ne 130 ] && [ $code -ne 143 ]; then
  say "Qwen Image Editor stopped with an error.\n\nDetails are in:\n$LOG" "with icon caution"
fi
exit $code
