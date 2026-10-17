#!/usr/bin/env bash
# Runs inside the container: follow the README on a machine that has nothing but git and curl, and report each step.
# The repository is mounted read-only at /src; results (log, images) go to /results.
set -uo pipefail

RESULTS=()
started=$(date +%s)

summary() {
  echo
  echo "=== Summary ($(( $(date +%s) - started )) s total)"
  printf '  %s\n' "${RESULTS[@]}"
}

# step NAME COMMAND...   run a step; stop the test at the first failure
step() {
  local name="$1"; shift
  local t0; t0=$(date +%s)
  echo
  echo "=== $name"
  if "$@"; then
    RESULTS+=("PASS  $(printf '%5s' $(( $(date +%s) - t0 )))s  $name")
  else
    RESULTS+=("FAIL  $(printf '%5s' $(( $(date +%s) - t0 )))s  $name")
    summary
    exit 1
  fi
}

gpu_visible() { nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; }

install_uv() {
  curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && uv --version
}

# Stand-in for `git clone`: copy what a clone would contain (what .gitignore leaves out stays out).
get_code() {
  mkdir -p "$HOME/qwen-image-editor"
  tar -C /src --exclude=./.venv --exclude=./outputs --exclude='*/__pycache__' --exclude=./.playwright-mcp -cf - . \
    | tar -C "$HOME/qwen-image-editor" -xf -
  cd "$HOME/qwen-image-editor" && ls
}

not_ready_yet() { ! uv run qie doctor; }  # must report "not set up" before setup has run

setup() {
  if [ "${ACCEPT_LICENSE:-0}" != "1" ]; then
    echo "The test downloads Qwen-Image-2.1, which is under the Qwen Research License (non-commercial use only)."
    echo "Rerun with -AcceptLicense (PowerShell) or --accept-license to accept it for this test."
    return 1
  fi
  uv run qie setup --accept-license --yes
}

text_to_image() {
  uv run qie generate "A ceramic teapot on a wooden table by a window, soft morning light" --seed 1 -o /results \
    | tee /results/t2i.txt
}

edit_with_reference() {
  uv run qie generate "Make it night time, lit by a single candle" -i /results/0000_s1.png --seed 1 -o /results \
    | tee /results/edit.txt
}

editor_serves() {
  uv run qie app --no-browser > /results/app.log 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do
    if curl -sf http://127.0.0.1:7861/api/status | grep -q '"ready"'; then
      curl -sf -o /dev/null http://127.0.0.1:7861/ && curl -sf -o /dev/null http://127.0.0.1:7861/static/app.js
      local ok=$?
      kill "$pid"
      return $ok
    fi
    if curl -sf http://127.0.0.1:7861/api/status | grep -q '"error"'; then
      curl -s http://127.0.0.1:7861/api/status; kill "$pid"; return 1
    fi
    sleep 2
  done
  kill "$pid"
  return 1
}

step "GPU visible in the container" gpu_visible
step "Install uv" install_uv
export PATH="$HOME/.local/bin:$PATH"
step "Get the code" get_code
cd "$HOME/qwen-image-editor"
step "uv sync" uv sync
step "qie doctor says not set up" not_ready_yet
step "qie setup (license, downloads)" setup
step "qie doctor says ready" uv run qie doctor
step "Text-to-image, 1024x1024" text_to_image
step "Edit with a reference image" edit_with_reference
step "Editor starts and serves the page" editor_serves
summary
